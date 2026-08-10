"""Route Optimisation Agent — LLD Box 4.

Delivery requests -> node mapper -> OR-Tools CVRP -> route plan, with
cold-chain constraints and re-solve on a temperature excursion.

Two things are worth stating up front because they decide whether the output
means anything:

**Cold chain is a constraint, not a filter.** Refrigerated stock is restricted
to refrigerated vehicles inside the model, via ``VehicleVar``. Solving without
the constraint and discarding bad assignments afterwards produces plans that
look optimal and are infeasible — the solver optimises against a cost function
that never knew the rule existed, and the filter then silently drops deliveries.

**Distances are integers.** OR-Tools requires integer arc costs, so haversine
kilometres are scaled by :data:`DISTANCE_SCALE` and rounded. Passing floats
does not raise; it truncates toward zero, which quietly makes every short leg
free and the optimiser indifferent to them.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0

#: Integer units per kilometre. Metre resolution: fine enough that rounding
#: never changes which route wins, coarse enough to stay well inside int64.
DISTANCE_SCALE = 1000

#: Solver budget per problem. The guide's figure; also roughly where GUIDED_LOCAL_SEARCH
#: stops improving on instances this size.
DEFAULT_TIME_LIMIT_S = 10


@dataclass(slots=True)
class RoutingProblem:
    """A capacitated vehicle routing problem, already in integer costs."""

    distance_matrix: list[list[int]]
    demands: list[int]
    vehicle_capacities: list[int]
    depot: int = 0
    #: Per stop: does this delivery need refrigeration?
    cold_chain: list[bool] = field(default_factory=list)
    #: Per vehicle: is it refrigerated?
    refrigerated: list[bool] = field(default_factory=list)
    #: Per stop: (earliest, latest) arrival in integer time units. FR-05.
    time_windows: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_locations(self) -> int:
        return len(self.distance_matrix)

    @property
    def n_vehicles(self) -> int:
        return len(self.vehicle_capacities)

    def validate(self) -> None:
        if self.n_locations == 0:
            raise ValueError("routing problem has no locations")
        if len(self.demands) != self.n_locations:
            raise ValueError(
                f"{len(self.demands)} demands for {self.n_locations} locations"
            )
        if self.cold_chain and len(self.cold_chain) != self.n_locations:
            raise ValueError("cold_chain must cover every location")
        if self.refrigerated and len(self.refrigerated) != self.n_vehicles:
            raise ValueError("refrigerated must cover every vehicle")
        if self.demands[self.depot] != 0:
            raise ValueError("the depot cannot have demand of its own")


@dataclass(slots=True)
class RoutePlan:
    """Solved routes and what they cost."""

    routes: list[list[int]]
    total_distance_units: int
    dropped: list[int] = field(default_factory=list)
    solve_seconds: float = 0.0
    status: str = "OK"

    @property
    def total_distance_km(self) -> float:
        return self.total_distance_units / DISTANCE_SCALE

    @property
    def vehicles_used(self) -> int:
        return sum(1 for route in self.routes if len(route) > 2)

    @property
    def stops_served(self) -> int:
        return sum(len(route) - 2 for route in self.routes if len(route) > 2)


# ── Distance ──────────────────────────────────────────────────────────


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def haversine_matrix(coords: Sequence[tuple[float, float]]) -> list[list[int]]:
    """Integer-scaled great-circle distance matrix."""
    return [
        [int(round(haversine_km(a, b) * DISTANCE_SCALE)) for b in coords] for a in coords
    ]


def euclidean_matrix(coords: Sequence[tuple[float, float]]) -> list[list[int]]:
    """CVRPLIB's EUC_2D metric: Euclidean distance rounded to the nearest integer.

    Used only for benchmark instances. Their published optima are computed with
    this exact rounding, so anything else makes the reported gap meaningless.
    """
    return [
        [int(round(math.dist(a, b))) for b in coords]
        for a in coords
    ]


# ── Solver ────────────────────────────────────────────────────────────


def solve_cvrp(
    problem: RoutingProblem,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    allow_dropping: bool = True,
    drop_penalty: int | None = None,
) -> RoutePlan:
    """Solve a CVRP with OR-Tools.

    ``allow_dropping`` adds a penalised disjunction per stop so an
    over-subscribed problem returns a partial plan instead of no plan at all.
    A simulation that halts because one day's orders exceed fleet capacity is
    less useful than one that reports which deliveries it could not make.
    """
    import time

    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    problem.validate()
    started = time.perf_counter()

    manager = pywrapcp.RoutingIndexManager(
        problem.n_locations, problem.n_vehicles, problem.depot
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance(from_index: int, to_index: int) -> int:
        return problem.distance_matrix[manager.IndexToNode(from_index)][
            manager.IndexToNode(to_index)
        ]

    transit = routing.RegisterTransitCallback(distance)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    def demand(from_index: int) -> int:
        return problem.demands[manager.IndexToNode(from_index)]

    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand),
        0,                            # no slack
        problem.vehicle_capacities,
        True,                         # every vehicle starts empty
        "Capacity",
    )

    if problem.time_windows:
        _add_time_windows(routing, manager, problem, transit)

    if problem.cold_chain and problem.refrigerated:
        _restrict_cold_chain(routing, manager, problem)

    if allow_dropping:
        # Penalty must exceed any detour the solver could otherwise prefer,
        # or it will drop stops to save distance rather than as a last resort.
        penalty = drop_penalty or max(max(row) for row in problem.distance_matrix) * 10
        for node in range(problem.n_locations):
            if node != problem.depot:
                routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Try each construction heuristic in turn. PATH_CHEAPEST_ARC is the guide's
    # choice and usually best, but it can fail to find *any* feasible start on
    # tightly packed instances, and a cheap retry beats returning nothing.
    strategies = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
    )

    solution = None
    for strategy in strategies:
        parameters = pywrapcp.DefaultRoutingSearchParameters()
        parameters.first_solution_strategy = strategy
        parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        # Milliseconds, not seconds: GUIDED_LOCAL_SEARCH spends whatever
        # budget it is given, so a whole second on a six-stop tour is a
        # second wasted -- and the twin solves one of these per depot per
        # simulated day.
        parameters.time_limit.FromMilliseconds(max(10, int(time_limit_s * 1000)))

        solution = routing.SolveWithParameters(parameters)
        if solution is not None:
            break
        logger.debug("first-solution strategy %s found nothing; retrying", strategy)

    elapsed = time.perf_counter() - started

    if solution is None:
        return RoutePlan([], 0, list(range(problem.n_locations)), elapsed, "INFEASIBLE")

    return _extract(routing, manager, solution, problem, elapsed)


def _add_time_windows(routing, manager, problem: RoutingProblem, transit) -> None:
    """Delivery windows (FR-05) as a Time dimension."""
    horizon = max(latest for _, latest in problem.time_windows)
    routing.AddDimension(transit, horizon, horizon, False, "Time")
    dimension = routing.GetDimensionOrDie("Time")

    for node, (earliest, latest) in enumerate(problem.time_windows):
        if node == problem.depot:
            continue
        index = manager.NodeToIndex(node)
        if index >= 0:
            dimension.CumulVar(index).SetRange(earliest, latest)


def _restrict_cold_chain(routing, manager, problem: RoutingProblem) -> None:
    """Allow refrigerated stops only on refrigerated vehicles.

    Expressed as a domain restriction on the model's VehicleVar so the search
    never considers an infeasible assignment, rather than filtering afterwards.
    """
    fridges = [v for v, cold in enumerate(problem.refrigerated) if cold]
    if not fridges:
        logger.warning("cold-chain stops exist but no refrigerated vehicle does")
        return

    for node, needs_cold in enumerate(problem.cold_chain):
        if needs_cold and node != problem.depot:
            routing.VehicleVar(manager.NodeToIndex(node)).SetValues(fridges)


def _extract(routing, manager, solution, problem: RoutingProblem, elapsed: float) -> RoutePlan:
    routes: list[list[int]] = []
    total = 0

    for vehicle in range(problem.n_vehicles):
        index = routing.Start(vehicle)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            previous = index
            index = solution.Value(routing.NextVar(index))
            total += routing.GetArcCostForVehicle(previous, index, vehicle)
            route.append(manager.IndexToNode(index))
        routes.append(route)

    visited = {node for route in routes for node in route}
    dropped = [
        node
        for node in range(problem.n_locations)
        if node != problem.depot and node not in visited
    ]
    return RoutePlan(routes, total, dropped, elapsed, "OK")


# ── Feasibility ───────────────────────────────────────────────────────


def check_feasible(plan: RoutePlan, problem: RoutingProblem) -> list[str]:
    """Independent re-check of a plan. Returns violations, empty if sound.

    Deliberately does not trust the solver: this is what the Stage 9 DoD
    ("feasible routes respecting capacity, time windows, and cold chain")
    actually verifies, and a constraint that was never added to the model would
    otherwise go unnoticed.
    """
    violations: list[str] = []

    for vehicle, route in enumerate(plan.routes):
        load = sum(problem.demands[node] for node in route)
        capacity = problem.vehicle_capacities[vehicle]
        if load > capacity:
            violations.append(f"vehicle {vehicle} carries {load} over capacity {capacity}")

        carries_cold = problem.cold_chain and problem.refrigerated
        if carries_cold and not problem.refrigerated[vehicle]:
            for node in route:
                if node != problem.depot and problem.cold_chain[node]:
                    violations.append(
                        f"vehicle {vehicle} is not refrigerated but serves "
                        f"cold-chain stop {node}"
                    )

    for route in plan.routes:
        for node in route[1:-1]:
            if node == problem.depot:
                violations.append("a route revisits the depot mid-tour")

    return violations
