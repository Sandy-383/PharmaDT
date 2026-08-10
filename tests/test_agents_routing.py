"""CVRP solver and Route Agent.

``check_feasible`` is deliberately independent of the solver: it re-derives
capacity and cold-chain compliance from the plan. A constraint that was never
added to the model would otherwise pass unnoticed, because the solver only
reports on rules it was told about.
"""

from __future__ import annotations

import networkx as nx
import pytest

from pharmadt.agents.bus import MessageBus, Topic
from pharmadt.agents.route_agent import RouteAgent
from pharmadt.agents.routing import (
    DISTANCE_SCALE,
    RoutingProblem,
    check_feasible,
    euclidean_matrix,
    haversine_km,
    haversine_matrix,
    solve_cvrp,
)

# A depot at the origin with four customers on a unit square.
SQUARE = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, -1.0)]


def problem(demands=None, capacities=None, **kw) -> RoutingProblem:
    return RoutingProblem(
        distance_matrix=euclidean_matrix(SQUARE),
        demands=demands or [0, 1, 1, 1, 1],
        vehicle_capacities=capacities or [4],
        depot=0,
        **kw,
    )


# ── Distance ──────────────────────────────────────────────────────────


def test_haversine_matches_a_known_separation() -> None:
    """Bengaluru to Mysuru is about 125 km."""
    assert haversine_km((12.9716, 77.5946), (12.2958, 76.6394)) == pytest.approx(125, abs=8)


def test_a_point_is_zero_distance_from_itself() -> None:
    assert haversine_km((12.0, 77.0), (12.0, 77.0)) == pytest.approx(0.0)


def test_the_matrix_is_symmetric_with_a_zero_diagonal() -> None:
    matrix = haversine_matrix([(12.0, 77.0), (13.0, 78.0), (14.0, 76.0)])
    for i, row in enumerate(matrix):
        assert row[i] == 0
        for j, value in enumerate(row):
            assert value == matrix[j][i]


def test_distances_are_integers_scaled_to_metres() -> None:
    """OR-Tools truncates floats, which would make short legs free."""
    matrix = haversine_matrix([(12.0, 77.0), (12.01, 77.0)])
    assert all(isinstance(v, int) for row in matrix for v in row)
    assert matrix[0][1] == pytest.approx(1.11 * DISTANCE_SCALE, rel=0.05)


# ── Solving ───────────────────────────────────────────────────────────


def test_a_simple_problem_visits_every_customer() -> None:
    plan = solve_cvrp(problem(), time_limit_s=1, allow_dropping=False)
    assert plan.status == "OK"
    assert plan.stops_served == 4
    assert not check_feasible(plan, problem())


def test_capacity_forces_more_vehicles() -> None:
    one_big = solve_cvrp(problem(capacities=[4]), time_limit_s=1, allow_dropping=False)
    two_small = solve_cvrp(
        problem(capacities=[2, 2]), time_limit_s=1, allow_dropping=False
    )
    assert one_big.vehicles_used == 1
    assert two_small.vehicles_used == 2


def test_capacity_is_never_exceeded() -> None:
    spec = problem(demands=[0, 3, 3, 3, 3], capacities=[6, 6])
    plan = solve_cvrp(spec, time_limit_s=1, allow_dropping=False)
    assert check_feasible(plan, spec) == []


def test_an_oversubscribed_problem_drops_stops_rather_than_failing() -> None:
    """A run that halts on one bad day is less useful than a partial plan."""
    spec = problem(demands=[0, 10, 10, 10, 10], capacities=[10])
    plan = solve_cvrp(spec, time_limit_s=1, allow_dropping=True)
    assert plan.dropped
    assert plan.stops_served < 4


# ── Cold chain ────────────────────────────────────────────────────────


def test_cold_stock_only_travels_on_refrigerated_vehicles() -> None:
    spec = problem(
        capacities=[4, 4],
        cold_chain=[False, True, True, False, False],
        refrigerated=[True, False],
    )
    plan = solve_cvrp(spec, time_limit_s=2, allow_dropping=False)
    assert check_feasible(plan, spec) == []


def test_the_feasibility_check_catches_a_cold_chain_breach() -> None:
    """Proves the checker is independent of the solver."""
    spec = problem(
        capacities=[4, 4],
        cold_chain=[False, True, False, False, False],
        refrigerated=[False, True],
    )
    forged = type(solve_cvrp(spec, time_limit_s=1))(
        routes=[[0, 1, 0], [0, 2, 3, 4, 0]], total_distance_units=0
    )
    violations = check_feasible(forged, spec)
    assert any("not refrigerated" in v for v in violations)


def test_the_checker_catches_an_overloaded_vehicle() -> None:
    spec = problem(demands=[0, 5, 5, 5, 5], capacities=[6, 6])
    forged = type(solve_cvrp(spec, time_limit_s=1))(
        routes=[[0, 1, 2, 3, 4, 0], [0, 0]], total_distance_units=0
    )
    assert any("over capacity" in v for v in check_feasible(forged, spec))


# ── Validation ────────────────────────────────────────────────────────


def test_a_depot_with_demand_is_rejected() -> None:
    with pytest.raises(ValueError, match="depot"):
        problem(demands=[5, 1, 1, 1, 1]).validate()


def test_mismatched_demand_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="demands"):
        problem(demands=[0, 1]).validate()


def test_an_empty_problem_is_rejected() -> None:
    with pytest.raises(ValueError, match="no locations"):
        RoutingProblem([], [], [1]).validate()


# ── The agent ─────────────────────────────────────────────────────────


@pytest.fixture
def graph() -> nx.DiGraph:
    g = nx.DiGraph()
    coords = {
        "DC": (12.90, 77.60), "PH1": (12.95, 77.62),
        "PH2": (12.85, 77.55), "PH3": (12.92, 77.70),
    }
    for node, (lat, lon) in coords.items():
        g.add_node(node, node_type="DISTRIBUTOR" if node == "DC" else "PHARMACY",
                   lat=lat, lon=lon)
    for ph in ("PH1", "PH2", "PH3"):
        g.add_edge("DC", ph, transit_days=1, distance_km=10.0)
    return g


@pytest.fixture
def agent(graph: nx.DiGraph) -> RouteAgent:
    return RouteAgent(graph=graph, cold_chain_drugs=frozenset({"D2"}),
                      bus=MessageBus(), time_limit_s=0.5)


def test_orders_are_collected_from_the_bus(agent: RouteAgent) -> None:
    agent.bus.publish(
        Topic.REPLENISHMENT_ORDER,
        {"supplier": "DC", "node_id": "PH1", "drug_id": "D1", "quantity": 100},
        sender="InventoryAgent",
    )
    assert len(agent.pending) == 1
    assert agent.pending[0].to_node == "PH1"


def test_redistribution_requests_are_routed_too(agent: RouteAgent) -> None:
    agent.bus.publish(
        Topic.REDISTRIBUTION_REQUEST,
        {"from_node": "DC", "to_node": "PH2", "drug_id": "D1", "quantity": 50},
        sender="ExpiryAgent",
    )
    assert agent.pending[0].source == "redistribution"


def test_cold_chain_drugs_are_marked_on_arrival(agent: RouteAgent) -> None:
    agent.bus.publish(
        Topic.REPLENISHMENT_ORDER,
        {"supplier": "DC", "node_id": "PH1", "drug_id": "D2", "quantity": 10},
        sender="InventoryAgent",
    )
    assert agent.pending[0].cold_chain is True


def test_a_days_orders_are_consolidated_into_one_tour(agent: RouteAgent) -> None:
    """Routing each order as it arrives is round trips, not vehicle routing."""
    for stop in ("PH1", "PH2", "PH3"):
        agent.bus.publish(
            Topic.REPLENISHMENT_ORDER,
            {"supplier": "DC", "node_id": stop, "drug_id": "D1", "quantity": 100},
            sender="InventoryAgent",
        )
    actions = agent.decide(agent.observe({"sim_day": 1}))

    assert len(actions) == 1
    assert actions[0].action_type == "ROUTE_PLAN"
    assert actions[0].params["vehicles_used"] == 1
    assert actions[0].params["distance_km"] > 0


def test_two_orders_to_one_stop_become_a_single_visit(agent: RouteAgent) -> None:
    for drug in ("D1", "D3"):
        agent.bus.publish(
            Topic.REPLENISHMENT_ORDER,
            {"supplier": "DC", "node_id": "PH1", "drug_id": drug, "quantity": 50},
            sender="InventoryAgent",
        )
    agent.decide(agent.observe({"sim_day": 1}))
    assert agent.plans[0].stops == ["PH1"]


def test_pending_requests_are_cleared_each_day(agent: RouteAgent) -> None:
    """Otherwise yesterday's deliveries are planned again today."""
    agent.bus.publish(
        Topic.REPLENISHMENT_ORDER,
        {"supplier": "DC", "node_id": "PH1", "drug_id": "D1", "quantity": 10},
        sender="InventoryAgent",
    )
    agent.decide(agent.observe({"sim_day": 1}))
    assert agent.pending == []


def test_no_requests_means_no_plan(agent: RouteAgent) -> None:
    assert agent.decide(agent.observe({"sim_day": 1})) == []


def test_an_excursion_marks_the_depot_for_replanning(agent: RouteAgent) -> None:
    agent.note_excursion("DC")
    agent.bus.publish(
        Topic.REPLENISHMENT_ORDER,
        {"supplier": "DC", "node_id": "PH1", "drug_id": "D1", "quantity": 10},
        sender="InventoryAgent",
    )
    actions = agent.decide(agent.observe({"sim_day": 1}))
    assert actions[0].params["replanned"] is True
    assert "re-planned" in actions[0].justification


def test_an_unknown_depot_is_skipped_rather_than_crashing(agent: RouteAgent) -> None:
    agent.bus.publish(
        Topic.REPLENISHMENT_ORDER,
        {"supplier": "NOWHERE", "node_id": "PH1", "drug_id": "D1", "quantity": 10},
        sender="InventoryAgent",
    )
    assert agent.decide(agent.observe({"sim_day": 1})) == []
