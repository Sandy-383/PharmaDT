"""The reporting harnesses: gate checks, benchmark parsing, shipment reconstruction.

These produce the numbers that go in the report, so a defect here misstates a
result while every simulation underneath it is correct. That is the worst kind
of bug in a project like this, because nothing looks broken.
"""

from __future__ import annotations

import pytest

from pharmadt.gate import Check
from pharmadt.ml.benchmark_routing import _vehicle_count
from pharmadt.ml.train_anomaly import shipment_records


# ── Gate reporting ────────────────────────────────────────────────────


def test_a_passing_check_renders_as_pass() -> None:
    text = Check("1", "the thing", True, "evidence here").render()
    assert "[PASS]" in text
    assert "evidence here" in text


def test_a_failing_check_renders_as_fail() -> None:
    assert "[FAIL]" in Check("2", "the thing", False, "what went wrong").render()


def test_a_check_always_carries_its_evidence() -> None:
    """A gate that printed only PASS would be an assertion, not a measurement."""
    check = Check("3", "name", True, "13,123 of 13,123 anchored", {"anchored": 13_123})
    assert "13,123" in check.render()
    assert check.metrics["anchored"] == 13_123


# ── CVRPLIB instance naming ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "expected"),
    [("X-n101-k25", 25), ("X-n106-k14", 14), ("X-n1001-k43", 43), ("A-n32-k5", 5)],
)
def test_the_fleet_size_is_read_from_the_instance_name(name: str, expected: int) -> None:
    """CVRPLIB encodes it as -k<N>; guessing would change the problem."""
    assert _vehicle_count(name, fallback=99) == expected


def test_an_unrecognised_name_falls_back() -> None:
    assert _vehicle_count("mystery-instance", fallback=7) == 7


# ── Shipment reconstruction ───────────────────────────────────────────


class FakeEvent:
    def __init__(self, kind, day, payload, batch=None, frm=None, to=None):
        self.event_type = kind
        self.sim_day = day
        self.payload = payload
        self.batch_id = batch
        self.from_node = frm
        self.to_node = to


class FakeWorld:
    def __init__(self, events):
        self.events = events
        import networkx as nx

        self.graph = nx.DiGraph()
        self.graph.add_edge("DC", "PH", distance_km=42.0)


def test_a_completed_shipment_is_reconstructed_with_its_transit_time() -> None:
    world = FakeWorld([
        FakeEvent("SHIPMENT_DISPATCHED", 5, {"shipment_id": "S1", "quantity": 100},
                  batch="B1", frm="DC", to="PH"),
        FakeEvent("SHIPMENT_RECEIVED", 8, {"shipment_id": "S1", "quantity": 100},
                  batch="B1", frm="DC", to="PH"),
    ])
    records = shipment_records(world)

    assert len(records) == 1
    assert records[0]["transit_days"] == 3
    assert records[0]["distance_km"] == 42.0
    assert records[0]["is_known_route"] is True


def test_a_shipment_still_in_flight_is_excluded() -> None:
    """Its transit time is unknown, and imputing one would invent a feature."""
    world = FakeWorld([
        FakeEvent("SHIPMENT_DISPATCHED", 5, {"shipment_id": "S1", "quantity": 100},
                  batch="B1", frm="DC", to="PH"),
    ])
    assert shipment_records(world) == []


def test_excursions_are_attributed_to_the_right_shipment() -> None:
    world = FakeWorld([
        FakeEvent("SHIPMENT_DISPATCHED", 1, {"shipment_id": "S1", "quantity": 10},
                  batch="B1", frm="DC", to="PH"),
        FakeEvent("SHIPMENT_DISPATCHED", 1, {"shipment_id": "S2", "quantity": 10},
                  batch="B2", frm="DC", to="PH"),
        FakeEvent("COLD_CHAIN_EXCURSION", 2, {"shipment_id": "S1", "temp_c": 14.0}),
        FakeEvent("COLD_CHAIN_EXCURSION", 2, {"shipment_id": "S1", "temp_c": 16.0}),
        FakeEvent("SHIPMENT_RECEIVED", 3, {"shipment_id": "S1"}, frm="DC", to="PH"),
        FakeEvent("SHIPMENT_RECEIVED", 3, {"shipment_id": "S2"}, frm="DC", to="PH"),
    ])
    by_id = {r["shipment_id"]: r for r in shipment_records(world)}

    assert by_id["S1"]["excursion_count"] == 2
    assert by_id["S1"]["excursion_severity"] == pytest.approx(11.0)  # |16 - 5|
    assert by_id["S2"]["excursion_count"] == 0
    assert by_id["S2"]["cold_chain"] is False


def test_an_unknown_route_is_marked_as_such() -> None:
    """An unusual node pair is itself an anomaly feature."""
    world = FakeWorld([
        FakeEvent("SHIPMENT_DISPATCHED", 1, {"shipment_id": "S1", "quantity": 5},
                  batch="B1", frm="NOWHERE", to="ELSEWHERE"),
        FakeEvent("SHIPMENT_RECEIVED", 2, {"shipment_id": "S1"},
                  frm="NOWHERE", to="ELSEWHERE"),
    ])
    assert shipment_records(world)[0]["is_known_route"] is False


def test_an_empty_event_log_yields_no_shipments() -> None:
    assert shipment_records(FakeWorld([])) == []
