"""End-to-end simulation: the Stage 3 Definition of Done.

The reproducibility test is the important one. Stage 4 hashes this event stream
into a chain and Stage 15 compares ablation arms against each other; neither
means anything if the same seed produces a different run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from pharmadt.config import settings
from pharmadt.core.events import EventType
from pharmadt.twin.simulation import (
    build_world,
    compute_kpis,
    event_rows,
    run_simulation,
    write_event_log,
)

SHORT_RUN = 60


@pytest.fixture(scope="module")
def finished_world():
    """One 60-day run shared by the assertions that only read from it."""
    return run_simulation(build_world(), days=SHORT_RUN)


# ── Reproducibility ───────────────────────────────────────────────────


def test_same_seed_produces_an_identical_event_log() -> None:
    first = event_rows(run_simulation(build_world(seed=42), days=SHORT_RUN))
    second = event_rows(run_simulation(build_world(seed=42), days=SHORT_RUN))

    assert first, "simulation produced no events"
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_same_seed_produces_identical_kpis() -> None:
    a = compute_kpis(run_simulation(build_world(seed=7), days=SHORT_RUN))
    b = compute_kpis(run_simulation(build_world(seed=7), days=SHORT_RUN))
    assert a == b


def test_a_different_seed_produces_a_different_run() -> None:
    """Otherwise the seed is not actually wired to anything."""
    a = event_rows(run_simulation(build_world(seed=1), days=SHORT_RUN))
    b = event_rows(run_simulation(build_world(seed=2), days=SHORT_RUN))
    assert json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True)


# ── Structure and behaviour ───────────────────────────────────────────


def test_network_matches_the_prescribed_topology(finished_world) -> None:
    assert len(finished_world.nodes) == settings.num_nodes == 12


def test_the_clock_advances_the_requested_number_of_days(finished_world) -> None:
    assert int(finished_world.env.now) == SHORT_RUN


def test_goods_actually_move_through_the_chain(finished_world) -> None:
    kinds = {e.event_type for e in finished_world.events}
    assert EventType.DISPENSED in kinds
    assert EventType.REPLENISHMENT_ORDERED in kinds
    assert EventType.SHIPMENT_DISPATCHED in kinds
    assert EventType.SHIPMENT_RECEIVED in kinds
    assert finished_world.shipments_delivered > 0


def test_demand_is_recorded_for_every_retail_node_and_day(finished_world) -> None:
    retail = [n for n in finished_world.nodes.values() if n.sells_to_consumers]
    expected = len(retail) * len(finished_world.drug_ids()) * SHORT_RUN
    assert len(finished_world.demand_outcomes) == expected


def test_no_node_exceeds_its_own_storage_capacity(finished_world) -> None:
    for node in finished_world.nodes.values():
        assert node.total_stock() <= node.storage_capacity, node.node_id


def test_fulfilment_never_exceeds_demand(finished_world) -> None:
    for outcome in finished_world.demand_outcomes:
        assert 0 <= outcome.fulfilled <= outcome.demanded


def test_events_are_within_the_simulated_window(finished_world) -> None:
    for event in finished_world.events:
        assert 0 <= event.sim_day <= SHORT_RUN


def test_snapshot_exposes_every_node_to_agents(finished_world) -> None:
    snapshot = finished_world.snapshot()
    assert set(snapshot["nodes"]) == set(finished_world.nodes)
    assert snapshot["drugs"] == finished_world.drug_ids()


# ── KPIs ──────────────────────────────────────────────────────────────


def test_kpis_are_internally_consistent(finished_world) -> None:
    kpis = compute_kpis(finished_world)

    assert kpis["units_fulfilled"] + kpis["units_short"] == kpis["units_demanded"]
    assert kpis["service_level"] == pytest.approx(1 - kpis["stockout_rate"])
    assert 0.0 <= kpis["stockout_rate"] <= 1.0
    assert kpis["average_inventory"] > 0
    assert kpis["nodes"] == 12


def test_baseline_leaves_headroom_for_the_inventory_agent(finished_world) -> None:
    """Stage 6 must be able to beat this measurably.

    A baseline at a perfect service level would leave that agent nothing to
    win; one in the double digits would mean the twin is broken rather than
    naive. Both ends are asserted deliberately.
    """
    kpis = compute_kpis(finished_world)
    assert 0.0 < kpis["stockout_rate"] < 0.15


# ── Output ────────────────────────────────────────────────────────────


def test_event_log_is_written_in_both_formats(finished_world, tmp_path: Path) -> None:
    written = write_event_log(finished_world, tmp_path, fmt="both")

    assert {p.name for p in written} == {"event_log.json", "event_log.csv"}
    rows = json.loads((tmp_path / "event_log.json").read_text(encoding="utf-8"))
    assert len(rows) == len(finished_world.events)
    assert rows[0]["seq"] == 0

    csv_text = (tmp_path / "event_log.csv").read_text(encoding="utf-8")
    assert csv_text.startswith("seq,sim_day,event_type")
    # newline="" on the writer; otherwise Windows interleaves blank lines.
    assert "\n\n" not in csv_text


def test_event_rows_are_emitted_in_chronological_order(finished_world) -> None:
    days = [row["sim_day"] for row in event_rows(finished_world)]
    assert days == sorted(days)


# ── Definition of Done ────────────────────────────────────────────────


@pytest.mark.slow
def test_full_year_over_twelve_nodes_runs_well_inside_thirty_seconds() -> None:
    started = time.perf_counter()
    world = run_simulation(build_world(), days=365)
    elapsed = time.perf_counter() - started

    assert int(world.env.now) == 365
    assert elapsed < 30.0, f"took {elapsed:.1f}s"

    # NFR-01 asks for 1000 steps/second, but a line tracer costs roughly an
    # order of magnitude. Under coverage this would measure the tracer rather
    # than the simulation, so the throughput claim is only asserted when the
    # interpreter is running untraced.
    if sys.gettrace() is None:
        assert 365 / elapsed > 1000, f"{365 / elapsed:,.0f} steps/s"
