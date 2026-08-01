"""The event and action vocabulary shared by the twin, the agents, and the ledger.

This module deliberately has no dependency on SQLAlchemy or SimPy. The twin
(Stage 3) emits :class:`Event`, the message bus (Stage 5) routes it, the ledger
(Stage 4) anchors the subset in :data:`LEDGER_EVENT_TYPES`, and the agents
(Stages 6-10) return :class:`Action`. Keeping it dependency-free is what lets
all four import it without a cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class EventType(StrEnum):
    """Everything that can happen in the simulated supply chain.

    The first seven are the physical custody events the provenance ledger
    records; see :data:`LEDGER_EVENT_TYPES`. The remainder are simulation
    telemetry — they drive KPIs and agent observations but are not handoffs,
    so signing and chaining them would only dilute the audit trail.
    """

    # ── Ledger-anchored custody events ────────────────────────────────
    BATCH_CREATED = "BATCH_CREATED"
    SHIPMENT_DISPATCHED = "SHIPMENT_DISPATCHED"
    SHIPMENT_RECEIVED = "SHIPMENT_RECEIVED"
    COLD_CHAIN_EXCURSION = "COLD_CHAIN_EXCURSION"
    REDISTRIBUTION = "REDISTRIBUTION"
    DISPENSED = "DISPENSED"
    RECALLED = "RECALLED"

    # ── Simulation telemetry (not ledger-anchored) ────────────────────
    DEMAND_FULFILLED = "DEMAND_FULFILLED"
    DEMAND_UNFULFILLED = "DEMAND_UNFULFILLED"
    STOCK_EXPIRED = "STOCK_EXPIRED"
    REPLENISHMENT_ORDERED = "REPLENISHMENT_ORDERED"


#: The subset of :class:`EventType` the ledger will accept. Stage 4 validates
#: against this so telemetry can never be written into the custody chain.
LEDGER_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.BATCH_CREATED,
        EventType.SHIPMENT_DISPATCHED,
        EventType.SHIPMENT_RECEIVED,
        EventType.COLD_CHAIN_EXCURSION,
        EventType.REDISTRIBUTION,
        EventType.DISPENSED,
        EventType.RECALLED,
    }
)

_EMPTY: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Event:
    """One thing that happened on one simulated day.

    The field order mirrors ``ProvenanceLedger.record_event`` so that anchoring
    an event is a field-for-field pass-through rather than a translation step
    that could silently disagree with the chain.

    Frozen because the twin's event log is replayed by the KPI collector, the
    ledger, and the tests; a mutable event would let one consumer perturb what
    the others see.
    """

    event_type: EventType
    sim_day: int
    batch_id: str | None = None
    from_node: str | None = None
    to_node: str | None = None
    payload: Mapping[str, Any] = field(default=_EMPTY)

    @property
    def is_ledger_anchored(self) -> bool:
        """Whether Stage 4 will write this event to the provenance chain."""
        return self.event_type in LEDGER_EVENT_TYPES


@dataclass(frozen=True, slots=True)
class Action:
    """A single decision an agent wants applied to the world.

    Agents never mutate the twin directly — ``Agent.decide`` returns
    ``list[Action]`` and ``Agent.act`` applies them. That indirection is what
    lets Stage 12 swap a heuristic agent for a MADDPG policy without touching
    the simulation, and what makes every decision loggable to ``AgentDecision``
    for NFR-08.
    """

    action_type: str
    target_node: str | None = None
    drug_id: str | None = None
    batch_id: str | None = None
    quantity: int | None = None
    params: Mapping[str, Any] = field(default=_EMPTY)
    justification: str = ""
