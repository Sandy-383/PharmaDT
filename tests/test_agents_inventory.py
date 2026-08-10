"""Inventory Agent: reorder point, echelon lead time, and basket scaling.

The echelon tests exist because the first version of this agent used the last
hop's transit time and was measurably *worse* than the naive baseline it was
meant to beat — 363% more stockouts. That regression must not come back.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from pharmadt.agents.bus import MessageBus, Topic
from pharmadt.agents.inventory import InventoryAgent, _mean_and_std
from pharmadt.config import settings


@pytest.fixture
def chain() -> nx.DiGraph:
    """MFG -> WH -> DC -> PH, one transit day per hop."""
    graph = nx.DiGraph()
    for node_id, node_type in (
        ("MFG", "MANUFACTURER"), ("WH", "WAREHOUSE"),
        ("DC", "DISTRIBUTOR"), ("PH", "PHARMACY"),
    ):
        graph.add_node(node_id, node_type=node_type)
    for a, b in (("MFG", "WH"), ("WH", "DC"), ("DC", "PH")):
        graph.add_edge(a, b, transit_days=1, distance_km=100.0)
    return graph


@pytest.fixture
def agent(chain: nx.DiGraph) -> InventoryAgent:
    return InventoryAgent(graph=chain, bus=MessageBus())


def state(on_hand: int, history: list[int], pending: int = 0, capacity: int = 10_000) -> dict:
    return {
        "sim_day": 10,
        "nodes": {
            "PH": {
                "node_id": "PH",
                "node_type": "PHARMACY",
                "stock_by_drug": {"D1": on_hand},
                "pending_inbound": {"D1": pending},
                "storage_capacity": capacity,
                "demand_history": {"D1": history},
            }
        },
    }


# ── Echelon lead time (the regression that must not return) ───────────


def test_lead_time_spans_the_whole_chain_not_the_last_hop(agent: InventoryAgent) -> None:
    """A pharmacy's stock is exposed for the full pipeline latency.

    Upstream tiers only reorder when their own position dips, so sizing on the
    final leg understates the risk period by the depth of the chain.
    """
    assert agent._lead_time("DC", "PH") == 3  # MFG->WH->DC->PH
    assert agent._lead_time("WH", "DC") == 2
    assert agent._lead_time("MFG", "WH") == 1


def test_lead_time_is_cached_per_node(agent: InventoryAgent) -> None:
    assert agent._lead_time("DC", "PH") == agent._lead_time("DC", "PH")
    assert agent._echelon_lead_time["PH"] == 3


def test_a_cycle_in_the_graph_cannot_hang_the_walk(chain: nx.DiGraph) -> None:
    chain.add_edge("PH", "MFG", transit_days=1)
    assert InventoryAgent(graph=chain)._lead_time("DC", "PH") > 0


def test_without_a_graph_it_falls_back_rather_than_crashing() -> None:
    assert InventoryAgent(graph=None)._lead_time("A", "B") == settings.default_lead_time_days


# ── Reorder point ─────────────────────────────────────────────────────


def test_no_order_while_the_position_is_above_the_reorder_point(
    agent: InventoryAgent,
) -> None:
    assert agent.decide(agent.observe(state(on_hand=5000, history=[40] * 28))) == []


def test_an_order_is_raised_once_the_position_drops(agent: InventoryAgent) -> None:
    actions = agent.decide(agent.observe(state(on_hand=10, history=[40] * 28)))
    assert len(actions) == 1
    assert actions[0].action_type == "REORDER"
    assert actions[0].target_node == "PH"
    assert actions[0].drug_id == "D1"
    assert actions[0].quantity > 0


def test_the_reorder_point_matches_the_formula(agent: InventoryAgent) -> None:
    """ROP = mu * risk + z * sigma * sqrt(risk), risk = echelon L + review."""
    history = [30, 50] * 14
    mean, sigma = _mean_and_std(history)
    risk = 3 + agent.review_period_days
    expected = int(mean * risk + agent.z * sigma * math.sqrt(risk))

    actions = agent.decide(agent.observe(state(on_hand=1, history=history)))
    assert actions[0].params["reorder_point"] == expected


def test_pending_inbound_counts_toward_the_position(agent: InventoryAgent) -> None:
    """Ignoring it would re-order every day until the first delivery landed."""
    history = [40] * 28
    without = agent.decide(agent.observe(state(on_hand=10, history=history)))
    with_pending = agent.decide(
        agent.observe(state(on_hand=10, history=history, pending=5000))
    )
    assert without and not with_pending


def test_a_volatile_series_gets_more_safety_stock(agent: InventoryAgent) -> None:
    """The whole point of the agent over a fixed threshold."""
    steady = agent.decide(agent.observe(state(on_hand=1, history=[40] * 28)))
    volatile = agent.decide(agent.observe(state(on_hand=1, history=[5, 75] * 14)))
    assert volatile[0].params["safety_stock"] > steady[0].params["safety_stock"]


def test_a_drug_nobody_asks_for_is_never_ordered(agent: InventoryAgent) -> None:
    """Ordering against sampling noise would fill a shelf with dead stock."""
    assert agent.decide(agent.observe(state(on_hand=0, history=[0] * 28))) == []


def test_the_justification_records_the_arithmetic(agent: InventoryAgent) -> None:
    """NFR-08: an examiner should not need the source to see why it ordered."""
    actions = agent.decide(agent.observe(state(on_hand=1, history=[40] * 28)))
    text = actions[0].justification
    for fragment in ("reorder point", "mu=", "sigma=", "echelon L=", "risk period="):
        assert fragment in text


# ── Capacity ──────────────────────────────────────────────────────────


def test_orders_are_scaled_to_fit_rather_than_served_alphabetically() -> None:
    """First-come allocation would let D1 fill the shelf and starve D3."""
    graph = nx.DiGraph()
    graph.add_node("SUP", node_type="DISTRIBUTOR")
    graph.add_node("PH", node_type="PHARMACY")
    graph.add_edge("SUP", "PH", transit_days=1)
    agent = InventoryAgent(graph=graph, bus=MessageBus())

    world_state = {
        "sim_day": 5,
        "nodes": {
            "PH": {
                "node_id": "PH", "node_type": "PHARMACY",
                "stock_by_drug": {"D1": 0, "D2": 0, "D3": 0},
                "pending_inbound": {},
                "storage_capacity": 300,
                "demand_history": {d: [40] * 28 for d in ("D1", "D2", "D3")},
            }
        },
    }
    actions = agent.decide(agent.observe(world_state))

    assert {a.drug_id for a in actions} == {"D1", "D2", "D3"}
    assert sum(a.quantity for a in actions) <= 300


# ── Bus integration ───────────────────────────────────────────────────


def test_a_published_forecast_overrides_observed_history(agent: InventoryAgent) -> None:
    """Stage 7's forecast is more current than a 28-day trailing mean."""
    quiet = agent.decide(agent.observe(state(on_hand=400, history=[10] * 28)))
    agent.bus.publish(
        Topic.FORECAST_DATA,
        {"node_id": "PH", "drug_id": "D1", "mean_daily": 200.0},
        sender="DemandAgent",
    )
    alarmed = agent.decide(agent.observe(state(on_hand=400, history=[10] * 28)))

    assert not quiet, "400 units covers 10/day"
    assert alarmed, "a forecast of 200/day should trigger a reorder"


def test_the_agent_subscribes_to_the_topics_the_diagram_gives_it(
    agent: InventoryAgent,
) -> None:
    assert agent.bus.subscribers(Topic.FORECAST_DATA)
    assert agent.bus.subscribers(Topic.SHORTAGE_ALERT)


# ── Statistics helper ─────────────────────────────────────────────────


def test_mean_and_std_of_a_flat_series() -> None:
    assert _mean_and_std([10] * 5) == (10.0, 0.0)


def test_mean_and_std_uses_the_sample_denominator() -> None:
    mean, sigma = _mean_and_std([2, 4, 4, 4, 5, 5, 7, 9])
    assert mean == 5.0
    assert sigma == pytest.approx(2.138, abs=0.001)  # n-1, not n


@pytest.mark.parametrize("series", [[], [7]])
def test_a_series_too_short_to_have_variance(series: list[int]) -> None:
    assert _mean_and_std(series)[1] == 0.0
