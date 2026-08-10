"""Demand Agent: forecasting scope, alerts, hotspots, and model fallback.

The scoping tests exist because applying the forecaster to upstream nodes
produced bullwhip amplification — a measured 68% inventory increase — and that
regression must not return.
"""

from __future__ import annotations

import numpy as np
import pytest

from pharmadt.agents.bus import MessageBus, Topic
from pharmadt.agents.demand import CONSUMER_FACING, DemandAgent
from pharmadt.core.interfaces import DemandModel


@pytest.fixture
def agent() -> DemandAgent:
    return DemandAgent(bus=MessageBus())


def world(nodes: dict) -> dict:
    return {"sim_day": 30, "nodes": nodes}


def node(node_type: str, history: list[int], stock: int = 1000) -> dict:
    return {
        "node_id": "N",
        "node_type": node_type,
        "stock_by_drug": {"D1": stock},
        "pending_inbound": {},
        "storage_capacity": 10_000,
        "demand_history": {"D1": history},
    }


# ── Scope: the bullwhip regression ────────────────────────────────────


def test_only_consumer_facing_nodes_are_forecast(agent: DemandAgent) -> None:
    """Upstream "demand" is the Inventory Agent's own order flow.

    Forecasting it and feeding the result back into the policy that produced it
    closes a positive feedback loop: a moving average lands on a recent order
    burst, roughly doubles the estimate, which enlarges the next order.
    """
    state = world(
        {
            "PH": {**node("PHARMACY", [40] * 28), "node_id": "PH"},
            "WH": {**node("WAREHOUSE", [900, 0, 0, 0, 0, 0, 0] * 4), "node_id": "WH"},
            "DC": {**node("DISTRIBUTOR", [500, 0, 0, 0, 0, 0, 0] * 4), "node_id": "DC"},
        }
    )
    observed = {s["node_id"] for s in agent.observe(state)["series"]}
    assert observed == {"PH"}


def test_the_consumer_facing_set_is_what_the_domain_says_it_is() -> None:
    assert frozenset({"PHARMACY", "HOSPITAL"}) == CONSUMER_FACING


def test_hospitals_are_forecast_too(agent: DemandAgent) -> None:
    state = world({"H": {**node("HOSPITAL", [30] * 28), "node_id": "H"}})
    assert len(agent.observe(state)["series"]) == 1


def test_a_series_with_no_demand_yet_is_skipped(agent: DemandAgent) -> None:
    state = world({"PH": {**node("PHARMACY", [0] * 28), "node_id": "PH"}})
    assert agent.observe(state)["series"] == []


# ── Forecasting ───────────────────────────────────────────────────────


def test_a_steady_series_forecasts_near_its_own_level(agent: DemandAgent) -> None:
    forecast = agent._forecast([[40] * 28])
    assert forecast.shape == (1, agent.horizon)
    assert np.allclose(forecast, 40.0, atol=1.0)


def test_short_history_is_padded_with_its_mean_not_with_zeros(
    agent: DemandAgent,
) -> None:
    """Zero-padding would tell the model the shop was shut."""
    padded = agent._pad([10, 20, 30])
    assert len(padded) == agent.lookback
    assert padded[0] == pytest.approx(20.0)
    assert padded[-3:] == [10.0, 20.0, 30.0]


def test_a_broken_model_degrades_to_the_cheap_forecaster(agent: DemandAgent) -> None:
    """A model failure must not take the whole simulation down."""

    class Broken(DemandModel):
        def fit(self, X, y): ...
        def predict(self, X, horizon=14):
            raise RuntimeError("model exploded")
        def get_weights(self): return []
        def set_weights(self, w): ...

    agent.model = Broken()
    forecast = agent._forecast([[40] * 28])
    assert np.allclose(forecast, 40.0, atol=1.0)


def test_a_working_model_is_actually_used() -> None:
    class Constant(DemandModel):
        def fit(self, X, y): ...
        def predict(self, X, horizon=14):
            return np.full((len(X), horizon), 123.0)
        def get_weights(self): return []
        def set_weights(self, w): ...

    agent = DemandAgent(model=Constant(), bus=MessageBus())
    assert np.allclose(agent._forecast([[40] * 28]), 123.0)


# ── Decisions ─────────────────────────────────────────────────────────


def test_a_forecast_action_is_always_produced(agent: DemandAgent) -> None:
    state = world({"PH": {**node("PHARMACY", [40] * 28), "node_id": "PH"}})
    actions = agent.decide(agent.observe(state))
    assert actions[0].action_type == "FORECAST"
    assert len(actions[0].params["forecasts"]) == 1


def test_forecasts_are_batched_into_one_action(agent: DemandAgent) -> None:
    """One action per pair would add ~11,000 identical audit rows a year."""
    nodes = {
        f"PH{i}": {**node("PHARMACY", [40] * 28), "node_id": f"PH{i}"} for i in range(5)
    }
    actions = agent.decide(agent.observe(world(nodes)))
    forecasts = [a for a in actions if a.action_type == "FORECAST"]
    assert len(forecasts) == 1
    assert len(forecasts[0].params["forecasts"]) == 5


def test_low_stock_raises_a_shortage_alert(agent: DemandAgent) -> None:
    state = world({"PH": {**node("PHARMACY", [40] * 28, stock=10), "node_id": "PH"}})
    actions = agent.decide(agent.observe(state))
    alerts = [a for a in actions if a.action_type == "SHORTAGE_ALERT"]

    assert len(alerts) == 1
    assert alerts[0].quantity > 0
    assert "covers less than" in alerts[0].justification


def test_ample_stock_raises_no_alert(agent: DemandAgent) -> None:
    state = world({"PH": {**node("PHARMACY", [40] * 28, stock=100_000), "node_id": "PH"}})
    actions = agent.decide(agent.observe(state))
    assert not [a for a in actions if a.action_type == "SHORTAGE_ALERT"]


def test_rising_demand_raises_a_hotspot(agent: DemandAgent) -> None:
    """The Expiry Agent steers near-expiry stock toward hotspots."""
    rising = [10] * 21 + [90] * 7
    state = world({"PH": {**node("PHARMACY", rising, stock=100_000), "node_id": "PH"}})
    actions = agent.decide(agent.observe(state))
    assert [a for a in actions if a.action_type == "DEMAND_HOTSPOT"]


def test_flat_demand_raises_no_hotspot(agent: DemandAgent) -> None:
    state = world({"PH": {**node("PHARMACY", [40] * 28, stock=100_000), "node_id": "PH"}})
    actions = agent.decide(agent.observe(state))
    assert not [a for a in actions if a.action_type == "DEMAND_HOTSPOT"]


def test_no_series_means_no_actions(agent: DemandAgent) -> None:
    assert agent.decide({"sim_day": 1, "series": []}) == []


# ── Publication ───────────────────────────────────────────────────────


def test_forecasts_reach_the_bus(agent: DemandAgent) -> None:
    received = []
    agent.bus.subscribe(Topic.FORECAST_DATA, received.append)

    state = world({"PH": {**node("PHARMACY", [40] * 28), "node_id": "PH"}})
    agent.step(state, world=None, sim_day=30)

    assert len(received) == 1
    assert received[0].payload["node_id"] == "PH"
    assert received[0].payload["mean_daily"] > 0
    assert received[0].sender == "DemandAgent"


def test_shortage_alerts_reach_the_bus(agent: DemandAgent) -> None:
    received = []
    agent.bus.subscribe(Topic.SHORTAGE_ALERT, received.append)

    state = world({"PH": {**node("PHARMACY", [40] * 28, stock=5), "node_id": "PH"}})
    agent.step(state, world=None, sim_day=30)

    assert received and received[0].payload["shortfall"] > 0


def test_every_decision_is_logged_for_the_audit_trail(agent: DemandAgent) -> None:
    state = world({"PH": {**node("PHARMACY", [40] * 28, stock=5), "node_id": "PH"}})
    agent.step(state, world=None, sim_day=30)

    kinds = {d.action["type"] for d in agent.decisions}
    assert "FORECAST" in kinds
    assert all(d.agent_name == "DemandAgent" for d in agent.decisions)


def test_the_latest_forecast_is_retrievable(agent: DemandAgent) -> None:
    """The Stage 14 dashboard reads this."""
    state = world({"PH": {**node("PHARMACY", [40] * 28), "node_id": "PH"}})
    agent.step(state, world=None, sim_day=30)

    forecast = agent.forecast_for("PH", "D1")
    assert forecast is not None and len(forecast) == agent.horizon
    assert agent.forecast_for("PH", "NOPE") is None
