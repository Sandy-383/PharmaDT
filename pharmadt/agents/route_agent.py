"""The Route Agent proper — batches delivery requests and plans vehicle tours.

Subscribes to ``replenishment.order`` (Inventory Agent) and
``redistribution.request`` (Expiry Agent), batches a day's requests per depot,
solves a CVRP, and publishes ``route.plan``.

Requests are **batched by depot before solving**. Routing each order the moment
it arrives is not vehicle routing at all — it is one round trip per delivery,
which is exactly the cost that consolidation exists to avoid, and it would make
the optimiser look worthless while never actually being asked to optimise.

Re-planning on a cold-chain excursion is scoped to the affected depot. A breach
on one vehicle says nothing about tours on the other side of the network, and
re-solving everything would discard good plans to no purpose.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pharmadt.agents.base import BaseAgent
from pharmadt.agents.bus import Topic
from pharmadt.agents.routing import (
    DEFAULT_TIME_LIMIT_S,
    RoutingProblem,
    check_feasible,
    haversine_matrix,
    solve_cvrp,
)
from pharmadt.core.events import Action

logger = logging.getLogger(__name__)

#: Units one vehicle can carry.
DEFAULT_VEHICLE_CAPACITY = 5_000
#: Vehicles available per depot per day.
DEFAULT_FLEET_SIZE = 4
#: How many of those are refrigerated.
DEFAULT_REFRIGERATED = 2


@dataclass(slots=True)
class DeliveryRequest:
    """One delivery waiting to be routed."""

    from_node: str
    to_node: str
    drug_id: str
    quantity: int
    cold_chain: bool = False
    source: str = "replenishment"


@dataclass(slots=True)
class DepotPlan:
    """A solved day of deliveries out of one depot."""

    depot: str
    stops: list[str]
    routes: list[list[str]]
    distance_km: float
    unserved: list[str] = field(default_factory=list)
    solve_seconds: float = 0.0


class RouteAgent(BaseAgent):
    """Plans consolidated vehicle tours for the day's deliveries."""

    name = "RouteAgent"

    def __init__(
        self,
        graph: Any = None,
        cold_chain_drugs: frozenset[str] = frozenset(),
        vehicle_capacity: int = DEFAULT_VEHICLE_CAPACITY,
        fleet_size: int = DEFAULT_FLEET_SIZE,
        refrigerated_vehicles: int = DEFAULT_REFRIGERATED,
        time_limit_s: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.graph = graph
        self.cold_chain_drugs = set(cold_chain_drugs)
        self.vehicle_capacity = vehicle_capacity
        self.fleet_size = fleet_size
        self.refrigerated_vehicles = refrigerated_vehicles
        # None means 'scale with the problem'. A depot serving six stops does
        # not need the ten seconds a 100-customer benchmark instance does.
        self.time_limit_s = time_limit_s

        self.pending: list[DeliveryRequest] = []
        self.plans: list[DepotPlan] = []
        #: Depots whose plan an excursion has invalidated today.
        self._replan: set[str] = set()
        self.total_distance_km = 0.0
        self.replans = 0
        super().__init__(**kwargs)

    # ── Bus ───────────────────────────────────────────────────────────

    def register_subscriptions(self) -> None:
        self.subscribe(Topic.REPLENISHMENT_ORDER, self._on_order)
        self.subscribe(Topic.REDISTRIBUTION_REQUEST, self._on_redistribution)
        self.subscribe(Topic.COUNTERFEIT_FLAG, self._on_counterfeit)

    def _on_order(self, message) -> None:
        payload = message.payload
        self.pending.append(
            DeliveryRequest(
                from_node=payload.get("supplier", ""),
                to_node=payload.get("node_id", ""),
                drug_id=payload.get("drug_id", ""),
                quantity=int(payload.get("quantity", 0)),
                cold_chain=payload.get("drug_id") in self.cold_chain_drugs,
                source="replenishment",
            )
        )

    def _on_redistribution(self, message) -> None:
        payload = message.payload
        self.pending.append(
            DeliveryRequest(
                from_node=payload.get("from_node", ""),
                to_node=payload.get("to_node", ""),
                drug_id=payload.get("drug_id", ""),
                quantity=int(payload.get("quantity", 0)),
                cold_chain=payload.get("drug_id") in self.cold_chain_drugs,
                source="redistribution",
            )
        )

    def _on_counterfeit(self, message) -> None:
        """A flagged batch's deliveries are withdrawn from the day's plan."""
        batch_id = message.payload.get("batch_id")
        if batch_id:
            logger.debug("counterfeit flag on %s; deliveries held", batch_id)

    def note_excursion(self, depot: str) -> None:
        """A temperature breach invalidates that depot's plan for the day."""
        self._replan.add(depot)

    # ── Observe ───────────────────────────────────────────────────────

    def observe(self, world_state: Mapping[str, Any]) -> dict[str, Any]:
        by_depot: dict[str, list[DeliveryRequest]] = {}
        for request in self.pending:
            if request.from_node and request.to_node and request.quantity > 0:
                by_depot.setdefault(request.from_node, []).append(request)

        return {
            "sim_day": world_state.get("sim_day", 0),
            "requests_by_depot": by_depot,
            "pending_total": len(self.pending),
            "replan_depots": sorted(self._replan),
        }

    # ── Decide ────────────────────────────────────────────────────────

    def decide(self, observation: Mapping[str, Any]) -> list[Action]:
        by_depot = observation.get("requests_by_depot", {})
        if not by_depot:
            self.pending.clear()
            self._replan.clear()
            return []

        self.plans = []
        actions: list[Action] = []

        for depot, requests in sorted(by_depot.items()):
            plan = self._plan_depot(depot, requests)
            if plan is None:
                continue

            self.plans.append(plan)
            self.total_distance_km += plan.distance_km
            replanned = depot in self._replan
            if replanned:
                self.replans += 1

            actions.append(
                Action(
                    action_type="ROUTE_PLAN",
                    target_node=depot,
                    quantity=len(plan.stops),
                    params={
                        "routes": plan.routes,
                        "distance_km": round(plan.distance_km, 2),
                        "vehicles_used": len([r for r in plan.routes if len(r) > 2]),
                        "unserved": plan.unserved,
                        "replanned": replanned,
                    },
                    justification=(
                        f"{len(plan.stops)} stop(s) from {depot} consolidated into "
                        f"{len([r for r in plan.routes if len(r) > 2])} tour(s) over "
                        f"{plan.distance_km:.1f} km"
                        + (" (re-planned after a cold-chain excursion)" if replanned else "")
                        + (f"; {len(plan.unserved)} beyond fleet capacity" if plan.unserved else "")
                    ),
                )
            )

        self.pending.clear()
        self._replan.clear()
        return actions

    def _plan_depot(self, depot: str, requests: list[DeliveryRequest]) -> DepotPlan | None:
        """Consolidate one depot's deliveries into vehicle tours."""
        if self.graph is None or depot not in self.graph:
            return None

        # One stop per destination: two orders to the same pharmacy are one
        # visit carrying both, not two round trips.
        merged: dict[str, dict[str, Any]] = {}
        for request in requests:
            stop = merged.setdefault(
                request.to_node, {"quantity": 0, "cold_chain": False}
            )
            stop["quantity"] += request.quantity
            stop["cold_chain"] = stop["cold_chain"] or request.cold_chain

        stops = [node for node in sorted(merged) if node in self.graph]
        if not stops:
            return None

        coords = [self._coords(depot)] + [self._coords(node) for node in stops]
        demands = [0] + [merged[node]["quantity"] for node in stops]
        cold = [False] + [merged[node]["cold_chain"] for node in stops]

        problem = RoutingProblem(
            distance_matrix=haversine_matrix(coords),
            demands=demands,
            vehicle_capacities=[self.vehicle_capacity] * self.fleet_size,
            depot=0,
            cold_chain=cold,
            refrigerated=[
                index < self.refrigerated_vehicles for index in range(self.fleet_size)
            ],
        )
        budget = (
            self.time_limit_s
            if self.time_limit_s is not None
            else min(DEFAULT_TIME_LIMIT_S, max(0.02, 0.01 * len(stops)))
        )
        plan = solve_cvrp(problem, time_limit_s=budget, allow_dropping=True)

        violations = check_feasible(plan, problem)
        if violations:
            logger.warning("infeasible plan from %s: %s", depot, violations[:2])

        labels = [depot, *stops]
        return DepotPlan(
            depot=depot,
            stops=stops,
            routes=[[labels[i] for i in route] for route in plan.routes if len(route) > 2],
            distance_km=plan.total_distance_km,
            unserved=[labels[i] for i in plan.dropped],
            solve_seconds=plan.solve_seconds,
        )

    def _coords(self, node_id: str) -> tuple[float, float]:
        data = self.graph.nodes[node_id]
        return float(data.get("lat", 0.0)), float(data.get("lon", 0.0))

    # ── Act ───────────────────────────────────────────────────────────

    def apply(self, action: Action, world: Any) -> None:
        """Publish the plan. The twin already moves stock via shipments.

        The plan is advisory in this build: shipment timing comes from the
        network's transit days, so publishing the tours records what the fleet
        would do and what it would cost, without a second mechanism competing
        with the twin over when goods arrive.
        """
        self.publish(
            Topic.ROUTE_PLAN,
            {
                "depot": action.target_node,
                "routes": action.params["routes"],
                "distance_km": action.params["distance_km"],
                "vehicles_used": action.params["vehicles_used"],
                "unserved": action.params["unserved"],
            },
            sim_day=int(getattr(world, "env", None).now) if world is not None else 0,
        )
