"""The event vocabulary shared by the twin, the bus, and the ledger."""

from __future__ import annotations

import dataclasses

import pytest

from pharmadt.core.events import LEDGER_EVENT_TYPES, Action, Event, EventType

# The seven custody events the guide specifies for the provenance ledger.
EXPECTED_LEDGER_EVENTS = {
    "BATCH_CREATED",
    "SHIPMENT_DISPATCHED",
    "SHIPMENT_RECEIVED",
    "COLD_CHAIN_EXCURSION",
    "REDISTRIBUTION",
    "DISPENSED",
    "RECALLED",
}


def test_ledger_event_types_are_exactly_the_seven_specified() -> None:
    assert {e.value for e in LEDGER_EVENT_TYPES} == EXPECTED_LEDGER_EVENTS


def test_ledger_event_types_are_a_subset_of_all_event_types() -> None:
    assert set(EventType) >= LEDGER_EVENT_TYPES


def test_telemetry_events_are_not_ledger_anchored() -> None:
    """Signing KPI telemetry would dilute the custody trail it sits next to."""
    telemetry = set(EventType) - LEDGER_EVENT_TYPES
    assert telemetry, "expected simulation telemetry alongside custody events"
    for event_type in telemetry:
        assert Event(event_type=event_type, sim_day=0).is_ledger_anchored is False


@pytest.mark.parametrize("event_type", sorted(LEDGER_EVENT_TYPES))
def test_custody_events_are_ledger_anchored(event_type: EventType) -> None:
    assert Event(event_type=event_type, sim_day=1, batch_id="B1").is_ledger_anchored


def test_event_type_is_a_string_enum() -> None:
    """JSONB payloads and the message bus both serialise these directly."""
    assert EventType.DISPENSED == "DISPENSED"
    assert f"{EventType.DISPENSED}" == "DISPENSED"


def test_event_is_immutable() -> None:
    """The event log is replayed by KPIs, the ledger, and tests alike."""
    event = Event(event_type=EventType.DISPENSED, sim_day=3, batch_id="B1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.sim_day = 4  # type: ignore[misc]


def test_event_payload_defaults_are_not_shared() -> None:
    """A mutable shared default would leak payload between unrelated events."""
    a = Event(event_type=EventType.RECALLED, sim_day=1)
    b = Event(event_type=EventType.RECALLED, sim_day=2)
    assert a.payload == {}
    with pytest.raises(TypeError):
        a.payload["injected"] = True  # type: ignore[index]
    assert b.payload == {}


def test_event_carries_the_full_handoff() -> None:
    event = Event(
        event_type=EventType.SHIPMENT_DISPATCHED,
        sim_day=12,
        batch_id="BATCH-0001",
        from_node="NODE-WH-01",
        to_node="NODE-PH-03",
        payload={"quantity": 250},
    )
    assert event.from_node == "NODE-WH-01"
    assert event.to_node == "NODE-PH-03"
    assert event.payload["quantity"] == 250


def test_action_defaults_are_empty_and_immutable() -> None:
    action = Action(action_type="REORDER")
    assert action.target_node is None
    assert action.quantity is None
    assert action.justification == ""
    with pytest.raises(dataclasses.FrozenInstanceError):
        action.quantity = 5  # type: ignore[misc]


def test_action_records_a_justification_for_the_audit_trail() -> None:
    """NFR-08 requires every agent decision to be explainable."""
    action = Action(
        action_type="REDISTRIBUTE",
        target_node="NODE-PH-02",
        drug_id="DRUG-002",
        quantity=40,
        justification="NODE-PH-02 projected to stock out in 3 days; surplus at NODE-PH-01.",
    )
    assert action.justification
