"""The three abstract contracts are genuinely abstract.

These tests are cheap insurance on the most expensive thing to get wrong. If a
required method silently stopped being abstract, a Stage 11 federated client
could be constructed without ``get_weights`` and fail only during training —
far from the cause.
"""

from __future__ import annotations

import numpy as np
import pytest

from pharmadt.core.events import Action, EventType
from pharmadt.core.interfaces import Agent, DemandModel, ProvenanceLedger


@pytest.mark.parametrize("abstract", [ProvenanceLedger, Agent, DemandModel])
def test_interfaces_cannot_be_instantiated(abstract: type) -> None:
    with pytest.raises(TypeError):
        abstract()


def test_partial_ledger_implementation_is_rejected() -> None:
    class HalfLedger(ProvenanceLedger):
        def record_event(
            self, batch_id, event_type, from_node, to_node, payload, signer_node, *, sim_day=0
        ):
            return "deadbeef"

        def get_provenance(self, batch_id):
            return []

        # verify_chain and verify_batch_fingerprint deliberately missing.

    with pytest.raises(TypeError):
        HalfLedger()


def test_partial_demand_model_is_rejected() -> None:
    class NoWeights(DemandModel):
        def fit(self, X, y) -> None: ...

        def predict(self, X, horizon: int = 14):
            return np.zeros(horizon)

        # get_weights/set_weights missing — exactly the Stage 11 failure the
        # guide warns about.

    with pytest.raises(TypeError):
        NoWeights()


class _ConstantDemandModel(DemandModel):
    """Minimal complete implementation used to prove the contract is satisfiable."""

    def __init__(self) -> None:
        self._w = [np.zeros(3, dtype=np.float32)]

    def fit(self, X, y) -> None:
        self._w = [np.asarray(y, dtype=np.float32).mean(keepdims=True)]

    def predict(self, X, horizon: int = 14) -> np.ndarray:
        return np.full(horizon, float(self._w[0].mean()), dtype=np.float32)

    def get_weights(self) -> list[np.ndarray]:
        return [w.copy() for w in self._w]

    def set_weights(self, w: list[np.ndarray]) -> None:
        self._w = [np.asarray(x, dtype=np.float32) for x in w]


def test_complete_demand_model_instantiates_and_forecasts() -> None:
    model = _ConstantDemandModel()
    model.fit(None, [10.0, 20.0, 30.0])
    forecast = model.predict(None, horizon=7)

    assert forecast.shape == (7,)
    assert np.allclose(forecast, 20.0)


def test_demand_model_weights_round_trip() -> None:
    """FedAvg exchanges nothing but these arrays, so the round-trip must be exact."""
    source = _ConstantDemandModel()
    source.fit(None, [1.0, 2.0, 3.0])

    target = _ConstantDemandModel()
    target.set_weights(source.get_weights())

    for a, b in zip(source.get_weights(), target.get_weights(), strict=True):
        assert np.array_equal(a, b)


def test_get_weights_returns_a_copy_not_a_view() -> None:
    """A view would let the Flower server mutate a client's live parameters."""
    model = _ConstantDemandModel()
    weights = model.get_weights()
    weights[0][:] = 999.0

    assert not np.allclose(model.get_weights()[0], 999.0)


def test_agent_contract_is_satisfiable() -> None:
    class StubAgent(Agent):
        name = "StubAgent"

        def observe(self, world_state):
            return {"stock": world_state.get("stock", 0)}

        def decide(self, observation):
            if observation["stock"] < 10:
                return [
                    Action(
                        action_type="REORDER",
                        quantity=100,
                        justification="stock below threshold",
                    )
                ]
            return []

        def act(self, actions, world) -> None:
            world["orders"] = list(actions)

    agent = StubAgent()
    world: dict = {"stock": 4}

    actions = agent.decide(agent.observe(world))
    agent.act(actions, world)

    assert agent.name == "StubAgent"
    assert len(world["orders"]) == 1
    assert world["orders"][0].action_type == "REORDER"


def test_record_event_signature_matches_event_fields() -> None:
    """``Event`` must be a field-for-field pass-through into the ledger.

    If these drift apart, anchoring an event becomes a manual translation step
    and the chain can silently disagree with the event log.
    """
    import inspect

    params = set(inspect.signature(ProvenanceLedger.record_event).parameters)
    params.discard("self")

    # Every field an Event carries must be expressible in a ledger append,
    # including sim_day — the provenance table declares it NOT NULL.
    event_fields = {"batch_id", "event_type", "from_node", "to_node", "payload", "sim_day"}
    assert event_fields <= params
    assert "signer_node" in params, "the ledger additionally needs a signer"

    # And the event carries a real EventType, not a bare string.
    assert isinstance(EventType.BATCH_CREATED, EventType)
