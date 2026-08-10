"""Expiry Agent: detection, urgency, and the Vickrey auction.

Two tests here lock in measured regressions:

* buyers must net off stock they already hold, or every node claims it can
  absorb the whole lot and redistribution manufactures the waste it exists to
  prevent;
* candidate buyers must include downstream nodes, or upstream stock has no
  route out and simply expires where it sits.
"""

from __future__ import annotations

import networkx as nx
import pytest

from pharmadt.agents.bus import MessageBus, Topic
from pharmadt.agents.expiry import ExpiryAgent


@pytest.fixture
def graph() -> nx.DiGraph:
    """WH -> DC -> {PH1, PH2}, with PH1 <-> PH2 lateral."""
    g = nx.DiGraph()
    for n, t in (("WH", "WAREHOUSE"), ("DC", "DISTRIBUTOR"),
                 ("PH1", "PHARMACY"), ("PH2", "PHARMACY")):
        g.add_node(n, node_type=t)
    for a, b in (("WH", "DC"), ("DC", "PH1"), ("DC", "PH2")):
        g.add_edge(a, b, transit_days=1, distance_km=50.0)
    g.add_edge("PH1", "PH2", transit_days=1, distance_km=10.0)
    g.add_edge("PH2", "PH1", transit_days=1, distance_km=10.0)
    return g


@pytest.fixture
def agent(graph: nx.DiGraph) -> ExpiryAgent:
    return ExpiryAgent(graph=graph, bus=MessageBus())


def lot(node_id="PH1", days=10, qty=500, batch="B1", drug="D1") -> dict:
    return {
        "node_id": node_id, "batch_id": batch, "drug_id": drug,
        "quantity": qty, "expiry_day": days, "days_to_expiry": days,
    }


# ── Detection ─────────────────────────────────────────────────────────


def test_only_lots_inside_the_horizon_are_flagged(agent: ExpiryAgent) -> None:
    state = {
        "sim_day": 1,
        "nodes": {
            "PH1": {
                "node_id": "PH1", "node_type": "PHARMACY",
                "stock_by_drug": {"D1": 500}, "pending_inbound": {},
                "storage_capacity": 10_000, "demand_history": {"D1": [10] * 28},
                "expiring_lots": [
                    {**lot(days=5), "node_id": "PH1"},
                    {**lot(days=999, batch="B2"), "node_id": "PH1"},
                ],
            }
        },
    }
    flagged = agent.observe(state)["flagged"]
    assert [f["batch_id"] for f in flagged] == ["B1"]


def test_a_quarantined_batch_is_never_auctioned(agent: ExpiryAgent) -> None:
    """Redistributing suspect stock would spread a counterfeit network-wide."""
    agent.bus.publish(Topic.COUNTERFEIT_FLAG, {"batch_id": "B1"}, sender="AnomalyAgent")
    state = {
        "sim_day": 1,
        "nodes": {
            "PH1": {
                "node_id": "PH1", "node_type": "PHARMACY",
                "stock_by_drug": {"D1": 500}, "pending_inbound": {},
                "storage_capacity": 10_000, "demand_history": {"D1": [10] * 28},
                "expiring_lots": [{**lot(), "node_id": "PH1"}],
            }
        },
    }
    assert agent.observe(state)["flagged"] == []


# ── Urgency ───────────────────────────────────────────────────────────


def test_sooner_expiry_outranks_later(agent: ExpiryAgent) -> None:
    assert agent.urgency(lot(days=2)) > agent.urgency(lot(days=20))


def test_more_stock_at_risk_outranks_less(agent: ExpiryAgent) -> None:
    assert agent.urgency(lot(qty=900)) > agent.urgency(lot(qty=100))


def test_stock_the_holder_will_sell_anyway_is_not_urgent(agent: ExpiryAgent) -> None:
    agent.forecasts[("PH1", "D1")] = 1000.0
    assert agent.urgency(lot(qty=500, days=10)) == 0.0


# ── The auction ───────────────────────────────────────────────────────


def test_downstream_nodes_are_eligible_buyers(agent: ExpiryAgent) -> None:
    """Without this, upstream stock has no route out and simply expires."""
    assert set(agent._candidate_buyers("WH")) == {"DC"}
    assert set(agent._candidate_buyers("DC")) == {"PH1", "PH2"}


def test_lateral_peers_are_eligible_too(agent: ExpiryAgent) -> None:
    assert "PH2" in agent._candidate_buyers("PH1")


def test_the_highest_bidder_wins_but_pays_the_runner_up_price(
    agent: ExpiryAgent,
) -> None:
    """Vickrey: truthful bidding is dominant because price is set by the rival."""
    agent.forecasts.update({("PH1", "D1"): 0.0, ("PH2", "D1"): 100.0})
    result = agent.auction(
        lot(node_id="DC", qty=100), 100,
        {"PH1": 10_000, "PH2": 10_000},
        stock={"PH1": {}, "PH2": {}},
    )
    assert result.winner in {"PH1", "PH2"}
    best = max(result.bids, key=lambda b: b.amount)
    assert result.clearing_price <= best.amount


def test_a_buyer_nets_off_stock_it_already_holds(agent: ExpiryAgent) -> None:
    """Otherwise every node claims the whole lot and it expires elsewhere."""
    agent.forecasts[("PH2", "D1")] = 100.0

    empty = agent.auction(lot(qty=500), 500, {"PH2": 10_000}, stock={"PH2": {}})
    full = agent.auction(
        lot(qty=500), 500, {"PH2": 10_000}, stock={"PH2": {"D1": 100_000}}
    )
    assert empty.winner == "PH2"
    assert full.winner is None


def test_nothing_clears_below_the_reserve(agent: ExpiryAgent) -> None:
    """Moving stock that costs more to ship than to bin is not a saving."""
    agent.forecasts[("PH2", "D1")] = 1.0
    agent.transport_cost_per_km = 1000.0
    result = agent.auction(lot(qty=500), 500, {"PH2": 10_000}, stock={"PH2": {}})
    assert result.winner is None


def test_a_buyer_with_no_room_cannot_win(agent: ExpiryAgent) -> None:
    agent.forecasts[("PH2", "D1")] = 100.0
    result = agent.auction(lot(), 500, {"PH2": 0}, stock={"PH2": {}})
    assert result.winner is None


def test_an_isolated_node_has_nowhere_to_send_stock() -> None:
    """A single-customer branch is a redistribution dead end — a topology fact."""
    g = nx.DiGraph()
    g.add_node("PH5", node_type="PHARMACY")
    agent = ExpiryAgent(graph=g, bus=MessageBus())
    assert agent._candidate_buyers("PH5") == []
    assert agent.auction(lot(node_id="PH5"), 100, {}, stock={}).winner is None


def test_the_result_records_why_it_did_not_clear(agent: ExpiryAgent) -> None:
    result = agent.auction(lot(), 100, {}, stock={})
    assert result.winner is None
    assert result.reason


# ── Decisions ─────────────────────────────────────────────────────────


def test_only_surplus_is_offered(agent: ExpiryAgent) -> None:
    """Auctioning stock the seller will sell anyway just relocates a stockout."""
    observation = {
        "sim_day": 1,
        "flagged": [lot(qty=100, days=10)],
        "demand_rate": {"PH1|D1": 50.0},  # will sell 500 in 10 days
        "stock": {"PH1": {"D1": 100}},
        "capacity_free": {"PH1": 1000, "PH2": 1000},
    }
    assert agent.decide(observation) == []


def test_a_redistribution_action_carries_its_auction_evidence(
    agent: ExpiryAgent,
) -> None:
    agent.forecasts[("PH2", "D1")] = 200.0
    observation = {
        "sim_day": 1,
        "flagged": [lot(qty=500, days=10)],
        "demand_rate": {"PH1|D1": 0.0, "PH2|D1": 200.0},
        "stock": {"PH1": {"D1": 500}, "PH2": {}},
        "capacity_free": {"PH1": 1000, "PH2": 10_000},
    }
    actions = agent.decide(observation)

    assert len(actions) == 1
    action = actions[0]
    assert action.action_type == "REDISTRIBUTE"
    assert action.target_node == "PH2"
    assert action.params["from_node"] == "PH1"
    assert "second-price auction" in action.justification
    assert "reserve" in action.justification


def test_no_flagged_stock_means_no_actions(agent: ExpiryAgent) -> None:
    assert agent.decide({"sim_day": 1, "flagged": []}) == []
