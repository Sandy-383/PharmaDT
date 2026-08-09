"""The agent skeleton: the daily cycle, audit logging, and execution order.

The Stage 5 Definition of Done is the last test in this file — a dummy agent
subscribes, receives a published message, and writes an AgentDecision row.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from pharmadt.agents.base import (
    AGENT_ORDER,
    AgentOrchestrator,
    BaseAgent,
    DecisionRecord,
    jsonable,
    persist_decisions,
)
from pharmadt.agents.bus import Message, MessageBus, Topic
from pharmadt.core.events import Action
from pharmadt.core.models import AgentDecision


class RecordingAgent(BaseAgent):
    """Orders whenever observed stock falls below a threshold."""

    name = "InventoryAgent"

    def __init__(self, threshold: int = 10, **kwargs: Any) -> None:
        self.applied: list[Action] = []
        self.threshold = threshold
        super().__init__(**kwargs)

    def observe(self, world_state: Mapping[str, Any]) -> dict[str, Any]:
        return {"stock": world_state.get("stock", 0)}

    def decide(self, observation: Mapping[str, Any]) -> list[Action]:
        if observation["stock"] >= self.threshold:
            return []
        return [
            Action(
                action_type="REORDER",
                target_node="NODE-PH-01",
                drug_id="DRUG-001",
                quantity=100,
                justification=f"stock {observation['stock']} below {self.threshold}",
            )
        ]

    def apply(self, action: Action, world: Any) -> None:
        self.applied.append(action)


class QuietAgent(BaseAgent):
    name = "AnomalyAgent"

    def observe(self, world_state: Mapping[str, Any]) -> dict[str, Any]:
        return {"seen": True}

    def decide(self, observation: Mapping[str, Any]) -> list[Action]:
        return []

    def apply(self, action: Action, world: Any) -> None:  # pragma: no cover
        raise AssertionError("QuietAgent never produces actions")


@pytest.fixture
def bus() -> MessageBus:
    return MessageBus()


# ── The canonical cycle ───────────────────────────────────────────────


def test_step_runs_observe_decide_act_in_order(bus: MessageBus) -> None:
    agent = RecordingAgent(bus=bus)
    actions = agent.step({"stock": 4}, world=None, sim_day=3)

    assert len(actions) == 1
    assert actions[0].action_type == "REORDER"
    assert agent.applied == actions


def test_an_agent_above_threshold_does_nothing(bus: MessageBus) -> None:
    agent = RecordingAgent(bus=bus)
    assert agent.step({"stock": 50}, world=None, sim_day=1) == []
    assert agent.applied == []


# ── Audit logging (NFR-08) ────────────────────────────────────────────


def test_every_action_produces_an_audit_record(bus: MessageBus) -> None:
    agent = RecordingAgent(bus=bus)
    agent.step({"stock": 4}, world=None, sim_day=9)

    assert len(agent.decisions) == 1
    record = agent.decisions[0]
    assert record.agent_name == "InventoryAgent"
    assert record.sim_day == 9
    assert record.inputs == {"stock": 4}
    assert record.action["type"] == "REORDER"
    assert record.action["quantity"] == 100
    assert "below" in record.justification


def test_choosing_to_do_nothing_is_also_recorded(bus: MessageBus) -> None:
    """"Why did the agent not reorder here?" is an audit question too."""
    agent = QuietAgent(bus=bus)
    agent.step({}, world=None, sim_day=2)

    assert len(agent.decisions) == 1
    assert agent.decisions[0].action == {"type": "NONE"}
    assert agent.decisions[0].justification


def test_records_accumulate_across_days(bus: MessageBus) -> None:
    agent = RecordingAgent(bus=bus)
    for day in range(5):
        agent.step({"stock": 1}, world=None, sim_day=day)

    assert [r.sim_day for r in agent.decisions] == [0, 1, 2, 3, 4]


def test_draining_hands_over_and_resets(bus: MessageBus) -> None:
    agent = RecordingAgent(bus=bus)
    agent.step({"stock": 1}, world=None, sim_day=0)

    assert len(agent.drain_decisions()) == 1
    assert agent.decisions == []


def test_a_subclass_cannot_override_act() -> None:
    """The audit trail is only trustworthy if it cannot be skipped.

    Overriding act() would silently bypass logging, and NFR-08 would hold for
    four agents and quietly not for the fifth.
    """
    with pytest.raises(TypeError, match="may not override act"):

        class Sneaky(BaseAgent):
            name = "RouteAgent"

            def observe(self, world_state):
                return {}

            def decide(self, observation):
                return []

            def apply(self, action, world):
                pass

            def act(self, actions, world):  # the forbidden override
                pass


def test_apply_is_still_abstract() -> None:
    class Incomplete(BaseAgent):
        name = "RouteAgent"

        def observe(self, world_state):
            return {}

        def decide(self, observation):
            return []

    with pytest.raises(TypeError):
        Incomplete()


# ── JSONB coercion ────────────────────────────────────────────────────


def test_numpy_scalars_survive_coercion() -> None:
    """np.int64 is not a Python int on Windows and JSONB rejects it."""
    coerced = jsonable({"a": np.int64(5), "b": np.float64(1.5)})
    assert coerced == {"a": 5, "b": 1.5}
    assert isinstance(coerced["a"], int)


def test_numpy_arrays_become_lists() -> None:
    assert jsonable({"v": np.array([1, 2, 3])}) == {"v": [1, 2, 3]}


def test_nested_structures_are_coerced_throughout() -> None:
    value = {"outer": [{"inner": np.int64(7)}]}
    assert jsonable(value) == {"outer": [{"inner": 7}]}


def test_unserialisable_values_fall_back_to_text() -> None:
    assert isinstance(jsonable({"o": object()})["o"], str)


def test_a_record_renders_a_database_row() -> None:
    record = DecisionRecord(
        agent_name="DemandAgent", sim_day=4,
        inputs={"x": np.int64(2)}, action={"type": "FORECAST"}, justification="because",
    )
    row = record.as_row()
    assert row["agent_name"] == "DemandAgent"
    assert row["inputs"] == {"x": 2}


# ── Orchestration ─────────────────────────────────────────────────────


def _agent_named(name: str) -> BaseAgent:
    return type(f"{name}Stub", (QuietAgent,), {"name": name})()


def test_agents_run_in_dependency_order_not_registration_order() -> None:
    """Each agent consumes what the previous publishes."""
    orchestrator = AgentOrchestrator()
    orchestrator.register(*[_agent_named(n) for n in reversed(AGENT_ORDER)])

    assert [a.name for a in orchestrator.agents] == list(AGENT_ORDER)


def test_an_unknown_agent_runs_last() -> None:
    orchestrator = AgentOrchestrator()
    orchestrator.register(_agent_named("CustomAgent"), _agent_named("DemandAgent"))

    assert [a.name for a in orchestrator.agents] == ["DemandAgent", "CustomAgent"]


def test_registration_puts_every_agent_on_the_shared_bus() -> None:
    bus = MessageBus()
    orchestrator = AgentOrchestrator(bus=bus)
    first, second = RecordingAgent(), QuietAgent()
    orchestrator.register(first, second)

    assert first.bus is bus
    assert second.bus is bus


def test_run_agents_returns_each_agents_actions() -> None:
    orchestrator = AgentOrchestrator()
    orchestrator.register(RecordingAgent(), QuietAgent())

    results = orchestrator.run_agents({"stock": 2}, sim_day=1)

    assert set(results) == {"InventoryAgent", "AnomalyAgent"}
    assert len(results["InventoryAgent"]) == 1
    assert results["AnomalyAgent"] == []


def test_collect_decisions_drains_every_agent() -> None:
    orchestrator = AgentOrchestrator()
    orchestrator.register(RecordingAgent(), QuietAgent())
    orchestrator.run_agents({"stock": 2}, sim_day=1)

    records = orchestrator.collect_decisions()
    assert {r.agent_name for r in records} == {"InventoryAgent", "AnomalyAgent"}
    assert orchestrator.collect_decisions() == []


def test_an_orchestrator_with_no_agents_is_harmless() -> None:
    assert AgentOrchestrator().run_agents({}, sim_day=0) == {}


# ── Definition of Done ────────────────────────────────────────────────


def test_dummy_agent_subscribes_receives_and_writes_a_decision_row(
    db_session: Session,
) -> None:
    """Stage 5 DoD, end to end.

    An agent subscribes to a topic, receives a published message, acts on it,
    and the resulting AgentDecision reaches the database.
    """

    class SubscribingAgent(BaseAgent):
        name = "ExpiryAgent"

        def __init__(self, **kwargs: Any) -> None:
            self.inbox: list[Message] = []
            super().__init__(**kwargs)

        def register_subscriptions(self) -> None:
            self.subscribe(Topic.COUNTERFEIT_FLAG, self.inbox.append)

        def observe(self, world_state: Mapping[str, Any]) -> dict[str, Any]:
            return {"flagged_batches": [m.payload["batch_id"] for m in self.inbox]}

        def decide(self, observation: Mapping[str, Any]) -> list[Action]:
            return [
                Action(
                    action_type="QUARANTINE",
                    batch_id=batch_id,
                    justification=f"{batch_id} flagged as counterfeit by the Anomaly Agent",
                )
                for batch_id in observation["flagged_batches"]
            ]

        def apply(self, action: Action, world: Any) -> None:
            world.setdefault("quarantined", []).append(action.batch_id)

    bus = MessageBus()
    world: dict[str, Any] = {}
    orchestrator = AgentOrchestrator(world=world, bus=bus)
    agent = SubscribingAgent()
    orchestrator.register(agent)

    # 1. The Anomaly Agent publishes; the subscriber receives it.
    delivered = bus.publish(
        Topic.COUNTERFEIT_FLAG, {"batch_id": "BATCH-0007"},
        sender="AnomalyAgent", sim_day=12,
    )
    assert delivered == 1
    assert agent.inbox[0].payload["batch_id"] == "BATCH-0007"

    # 2. The daily cycle turns that message into an action.
    orchestrator.run_agents({}, sim_day=12)
    assert world["quarantined"] == ["BATCH-0007"]

    # 3. The decision is written to the database.
    @contextmanager
    def factory() -> Iterator[Session]:
        yield db_session

    written = persist_decisions(orchestrator.collect_decisions(), session_factory=factory)
    assert written == 1

    row = db_session.scalars(
        select(AgentDecision).where(AgentDecision.agent_name == "ExpiryAgent")
    ).one()
    assert row.sim_day == 12
    assert row.action["type"] == "QUARANTINE"
    assert row.action["batch_id"] == "BATCH-0007"
    assert "counterfeit" in row.justification
    assert row.inputs["flagged_batches"] == ["BATCH-0007"]
    assert row.created_at is not None


def test_persisting_nothing_writes_nothing() -> None:
    assert persist_decisions([]) == 0
