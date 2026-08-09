"""The skeleton every agent plugs into, and the orchestrator that runs them.

Two things live here that deliberately do not live in the individual agents:

* **Decision logging.** ``BaseAgent.act`` writes an audit record for every
  action, and subclasses are structurally prevented from overriding it. NFR-08
  is then a property of the framework rather than a rule five separate agents
  each have to remember.
* **Execution order.** Agents run Demand -> Inventory -> Expiry -> Route ->
  Anomaly, because each consumes what the previous publishes. Running them in
  registration order would make the results depend on import order.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from pharmadt.agents.bus import Message, MessageBus, Topic
from pharmadt.core.events import Action
from pharmadt.core.interfaces import Agent

logger = logging.getLogger(__name__)

#: Dependency order for the daily cycle. Demand forecasts feed the Inventory
#: Agent, whose orders feed the Route Agent, and so on down the chain.
AGENT_ORDER: tuple[str, ...] = (
    "DemandAgent",
    "InventoryAgent",
    "ExpiryAgent",
    "RouteAgent",
    "AnomalyAgent",
)


def jsonable(value: Any) -> Any:
    """Coerce a value into something JSONB will accept.

    Agents compute with NumPy, and ``np.int64`` is not a Python ``int`` on
    Windows — inserting one raises deep inside the driver, at the end of a long
    simulation, far from the agent that produced it.
    """
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [jsonable(v) for v in value]
    if isinstance(value, str | bool | int | float) or value is None:
        return value
    if hasattr(value, "tolist"):  # numpy array
        return value.tolist()
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return str(value)


@dataclass(slots=True)
class DecisionRecord:
    """One audit entry, buffered in memory until the run ends.

    Not written immediately: NFR-01 asks for 1000 simulated steps per second,
    and a round trip per decision would put the ceiling far below that. These
    are bulk-inserted after ``env.run`` returns, exactly as the event log and
    the provenance ledger are.
    """

    agent_name: str
    sim_day: int
    inputs: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    justification: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "sim_day": self.sim_day,
            "inputs": jsonable(self.inputs),
            "action": jsonable(self.action),
            "justification": self.justification,
        }


class BaseAgent(Agent):
    """Canonical observe -> decide -> act cycle, run once per simulated day."""

    #: Overridden by each concrete agent; must match a name in AGENT_ORDER.
    name: str = "BaseAgent"

    def __init__(self, bus: MessageBus | None = None, name: str | None = None) -> None:
        self.bus = bus if bus is not None else MessageBus()
        if name is not None:
            self.name = name
        self.decisions: list[DecisionRecord] = []
        self.register_subscriptions()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse subclasses that override ``act``.

        The audit trail is only trustworthy if it cannot be skipped. Overriding
        ``act`` would bypass decision logging silently, and NFR-08 would then
        hold for four agents and quietly not for the fifth. Subclasses
        customise :meth:`apply` instead.
        """
        super().__init_subclass__(**kwargs)
        if "act" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} may not override act(); it carries the NFR-08 "
                "audit logging. Override apply() to change what an action does."
            )

    # ── The cycle ─────────────────────────────────────────────────────

    def step(self, world_state: Mapping[str, Any], world: Any, sim_day: int) -> list[Action]:
        """Run one full day for this agent."""
        self._sim_day = sim_day
        observation = self.observe(world_state)
        self._last_observation = observation
        actions = self.decide(observation)
        self.act(actions, world)
        return list(actions)

    def act(self, actions: Sequence[Action], world: Any) -> None:
        """Apply each action and record it. Not overridable — see __init_subclass__."""
        observation = getattr(self, "_last_observation", {})
        sim_day = getattr(self, "_sim_day", 0)

        if not actions:
            # An agent that considered the day and chose to do nothing is an
            # audit answer in its own right: "why did it not reorder here?"
            self.decisions.append(
                DecisionRecord(
                    agent_name=self.name,
                    sim_day=sim_day,
                    inputs=dict(observation),
                    action={"type": "NONE"},
                    justification="No action met this agent's criteria today.",
                )
            )
            return

        for action in actions:
            self.apply(action, world)
            self.decisions.append(
                DecisionRecord(
                    agent_name=self.name,
                    sim_day=sim_day,
                    inputs=dict(observation),
                    action=self._action_as_dict(action),
                    justification=action.justification,
                )
            )

    @abstractmethod
    def apply(self, action: Action, world: Any) -> None:
        """Carry out one action against the world."""

    # ── Bus helpers ───────────────────────────────────────────────────

    def register_subscriptions(self) -> None:
        """Override to subscribe to topics. Called once at construction."""

    def publish(
        self, topic: Topic, payload: Mapping[str, Any], sim_day: int | None = None
    ) -> int:
        return self.bus.publish(
            topic,
            payload,
            sender=self.name,
            sim_day=self._resolve_day(sim_day),
        )

    def subscribe(self, topic: Topic, handler) -> None:
        self.bus.subscribe(topic, handler)

    # ── Audit ─────────────────────────────────────────────────────────

    def drain_decisions(self) -> list[DecisionRecord]:
        """Hand over buffered decisions and reset."""
        drained, self.decisions = self.decisions, []
        return drained

    def _resolve_day(self, sim_day: int | None) -> int:
        return getattr(self, "_sim_day", 0) if sim_day is None else sim_day

    @staticmethod
    def _action_as_dict(action: Action) -> dict[str, Any]:
        return {
            "type": action.action_type,
            "target_node": action.target_node,
            "drug_id": action.drug_id,
            "batch_id": action.batch_id,
            "quantity": action.quantity,
            "params": dict(action.params),
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class AgentOrchestrator:
    """Runs every registered agent once per simulated day, in dependency order."""

    def __init__(self, world: Any = None, bus: MessageBus | None = None) -> None:
        self.world = world
        self.bus = bus if bus is not None else MessageBus()
        self._agents: list[BaseAgent] = []

    def register(self, *agents: BaseAgent) -> AgentOrchestrator:
        """Add agents. Their bus is replaced with the orchestrator's."""
        for agent in agents:
            agent.bus = self.bus
            agent.register_subscriptions()
            self._agents.append(agent)
        self._agents.sort(key=self._order_key)
        return self

    @property
    def agents(self) -> tuple[BaseAgent, ...]:
        return tuple(self._agents)

    @staticmethod
    def _order_key(agent: BaseAgent) -> tuple[int, str]:
        """Known agents in dependency order; anything else last, by name."""
        try:
            return (AGENT_ORDER.index(agent.name), agent.name)
        except ValueError:
            return (len(AGENT_ORDER), agent.name)

    def run_agents(
        self, world_state: Mapping[str, Any], sim_day: int
    ) -> dict[str, list[Action]]:
        """One full agent cycle for ``sim_day``. Called by the SimPy loop daily."""
        results: dict[str, list[Action]] = {}
        for agent in self._agents:
            results[agent.name] = agent.step(world_state, self.world, sim_day)
        return results

    # ── Audit ─────────────────────────────────────────────────────────

    def collect_decisions(self) -> list[DecisionRecord]:
        """Drain every agent's buffer."""
        records: list[DecisionRecord] = []
        for agent in self._agents:
            records.extend(agent.drain_decisions())
        return records

    def __len__(self) -> int:
        return len(self._agents)


def persist_decisions(
    records: Sequence[DecisionRecord],
    session_factory: Callable[[], AbstractContextManager[Session]] | None = None,
) -> int:
    """Bulk-insert buffered decisions. Call after the run, never during it.

    ``session_factory`` is injectable so tests write into a transaction that is
    rolled back, matching the ledger's arrangement.
    """
    if not records:
        return 0

    from pharmadt.core.db import session_scope
    from pharmadt.core.models import AgentDecision

    factory = session_factory if session_factory is not None else session_scope
    with factory() as session:
        session.bulk_insert_mappings(AgentDecision, [r.as_row() for r in records])
    return len(records)


__all__ = [
    "AGENT_ORDER",
    "AgentOrchestrator",
    "BaseAgent",
    "DecisionRecord",
    "Message",
    "MessageBus",
    "Topic",
    "jsonable",
    "persist_decisions",
]
