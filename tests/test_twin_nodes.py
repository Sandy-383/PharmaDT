"""Node runtime: FEFO inventory, capacity accounting, and the demand window.

Several tests here are regressions for bugs that produced plausible-looking but
wrong KPIs rather than crashes — the failure mode this project is most exposed
to, since nobody eyeballs 15,000 events.
"""

from __future__ import annotations

import numpy as np
import pytest

from pharmadt.core.models import NodeType
from pharmadt.twin.nodes import (
    DEMAND_HISTORY_DAYS,
    DemandProfile,
    Lot,
    TwinNode,
    date_of,
    sim_day_of,
)


@pytest.fixture
def pharmacy() -> TwinNode:
    return TwinNode("PH", NodeType.PHARMACY, storage_capacity=1_000, has_cold_storage=False)


# ── Calendar ──────────────────────────────────────────────────────────


def test_day_and_date_round_trip() -> None:
    assert sim_day_of(date_of(0)) == 0
    assert sim_day_of(date_of(97)) == 97


# ── FEFO inventory ────────────────────────────────────────────────────


def test_lots_are_consumed_first_expired_first_out(pharmacy: TwinNode) -> None:
    pharmacy.add_lot(Lot("LATE", "D1", expiry_day=90, quantity=100))
    pharmacy.add_lot(Lot("EARLY", "D1", expiry_day=10, quantity=100))

    fulfilled, drawn = pharmacy.consume("D1", 150)

    assert fulfilled == 150
    # The short-dated lot must be emptied before the long-dated one is touched;
    # FIFO by arrival would strand it and inflate the wastage KPI.
    assert drawn == [("EARLY", 100), ("LATE", 50)]


def test_consume_reports_a_partial_draw(pharmacy: TwinNode) -> None:
    pharmacy.add_lot(Lot("B1", "D1", expiry_day=90, quantity=30))
    fulfilled, drawn = pharmacy.consume("D1", 100)
    assert fulfilled == 30
    assert drawn == [("B1", 30)]
    assert pharmacy.stock_of("D1") == 0


def test_consume_on_empty_stock_is_a_noop(pharmacy: TwinNode) -> None:
    assert pharmacy.consume("D1", 10) == (0, [])


def test_same_batch_merges_rather_than_duplicating(pharmacy: TwinNode) -> None:
    pharmacy.add_lot(Lot("B1", "D1", expiry_day=90, quantity=10))
    pharmacy.add_lot(Lot("B1", "D1", expiry_day=90, quantity=15))
    assert len(pharmacy.lots["D1"]) == 1
    assert pharmacy.stock_of("D1") == 25


def test_expired_lots_are_removed_and_reported(pharmacy: TwinNode) -> None:
    pharmacy.add_lot(Lot("OLD", "D1", expiry_day=5, quantity=40))
    pharmacy.add_lot(Lot("GOOD", "D1", expiry_day=500, quantity=60))

    wasted = pharmacy.remove_expired(sim_day=5)

    assert wasted == [("OLD", "D1", 40)]
    assert pharmacy.stock_of("D1") == 60


def test_unexpired_stock_survives_the_scan(pharmacy: TwinNode) -> None:
    pharmacy.add_lot(Lot("GOOD", "D1", expiry_day=100, quantity=60))
    assert pharmacy.remove_expired(sim_day=99) == []
    assert pharmacy.stock_of("D1") == 60


# ── Capacity ──────────────────────────────────────────────────────────


def test_available_space_excludes_stock_already_in_transit(pharmacy: TwinNode) -> None:
    """Ordering against free space alone double-commits the same shelf."""
    pharmacy.add_lot(Lot("B1", "D1", expiry_day=90, quantity=400))
    pharmacy.pending_inbound["D1"] = 300

    assert pharmacy.free_capacity() == 600
    assert pharmacy.available_space() == 300


def test_utilisation_is_bounded(pharmacy: TwinNode) -> None:
    pharmacy.add_lot(Lot("B1", "D1", expiry_day=90, quantity=5_000))
    assert pharmacy.utilisation() == 1.0


# ── Demand window ─────────────────────────────────────────────────────


def test_same_day_demand_accumulates_into_one_bucket(pharmacy: TwinNode) -> None:
    """Three orders in a day are one day of demand, not three."""
    for _ in range(3):
        pharmacy.record_demand("D1", 100, sim_day=0)
    assert list(pharmacy.demand_history["D1"]) == [300]


def test_quiet_days_are_zero_filled(pharmacy: TwinNode) -> None:
    pharmacy.record_demand("D1", 10, sim_day=0)
    pharmacy.record_demand("D1", 20, sim_day=4)
    assert list(pharmacy.demand_history["D1"]) == [10, 0, 0, 0, 20]


def test_mean_recent_demand_is_a_daily_rate_not_a_per_order_average() -> None:
    """Regression: upstream nodes read order lumps as single days of demand.

    A warehouse is asked for stock roughly weekly. Averaging over the number of
    orders rather than the number of days inflated its estimate sevenfold, and
    the error compounded at every tier — warehouses ended up holding nine
    months of network demand while pharmacies stocked out.
    """
    warehouse = TwinNode("WH", NodeType.WAREHOUSE, 100_000, True)
    warehouse.record_demand("D1", 700, sim_day=0)
    warehouse.record_demand("D1", 700, sim_day=7)
    warehouse.settle_day(13)

    # 1400 units over 14 observed days.
    assert warehouse.mean_recent_demand("D1") == pytest.approx(100.0)


def test_first_observation_is_spread_over_elapsed_days() -> None:
    """Regression: the warm-up spike.

    On a node's first order there was no history, so one lump read as the whole
    daily rate and the (s, S) review ordered its full horizon times that. One
    warehouse ended day 1 holding sixteen times its own target.
    """
    warehouse = TwinNode("WH", NodeType.WAREHOUSE, 100_000, True)
    warehouse.record_demand("D1", 1_000, sim_day=9)

    # Ten elapsed days (0..9), not one.
    assert len(warehouse.demand_history["D1"]) == 10
    assert warehouse.mean_recent_demand("D1") == pytest.approx(100.0)


def test_demand_window_is_capped_at_28_days(pharmacy: TwinNode) -> None:
    for day in range(100):
        pharmacy.record_demand("D1", 1, sim_day=day)
    assert len(pharmacy.demand_history["D1"]) == DEMAND_HISTORY_DAYS


def test_settle_day_is_idempotent_within_a_day(pharmacy: TwinNode) -> None:
    pharmacy.record_demand("D1", 5, sim_day=3)
    pharmacy.settle_day(3)
    pharmacy.settle_day(3)
    assert list(pharmacy.demand_history["D1"]) == [0, 0, 0, 5]


def test_mean_demand_falls_back_to_the_profile_before_any_history(
    pharmacy: TwinNode,
) -> None:
    """Day 0 with an empty history must not compute a reorder point of zero."""
    pharmacy.demand_profiles["D1"] = DemandProfile(mean=42.0)
    assert pharmacy.mean_recent_demand("D1") == 42.0


def test_drugs_handled_remembers_a_fully_depleted_drug() -> None:
    """Regression: a sold-out drug used to drop out of the review set forever."""
    distributor = TwinNode("DC", NodeType.DISTRIBUTOR, 50_000, True)
    distributor.record_demand("D1", 100, sim_day=0)
    assert distributor.stock_of("D1") == 0
    assert "D1" in distributor.drugs_handled()


# ── Demand profile ────────────────────────────────────────────────────


def test_weekends_reduce_expected_demand() -> None:
    profile = DemandProfile(mean=100.0, seasonal_amplitude=0.0, weekend_factor=0.5)
    weekdays = [d for d in range(14) if date_of(d).weekday() < 5]
    weekends = [d for d in range(14) if date_of(d).weekday() >= 5]
    assert profile.expected(weekends[0]) < profile.expected(weekdays[0])


def test_sampling_is_reproducible_for_a_fixed_generator() -> None:
    profile = DemandProfile(mean=50.0)
    a = [profile.sample(d, np.random.default_rng(7)) for d in range(20)]
    b = [profile.sample(d, np.random.default_rng(7)) for d in range(20)]
    assert a == b


def test_sampled_demand_is_overdispersed() -> None:
    """A plain Poisson would understate stockout risk."""
    profile = DemandProfile(mean=50.0, dispersion=0.5, seasonal_amplitude=0.0)
    rng = np.random.default_rng(0)
    draws = [profile.sample(10, rng) for _ in range(4_000)]
    assert np.var(draws) > np.mean(draws) * 2


def test_zero_mean_yields_no_demand() -> None:
    assert DemandProfile(mean=0.0).sample(3, np.random.default_rng(0)) == 0
