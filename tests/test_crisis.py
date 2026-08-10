"""Crisis scenarios: loading, reversible injection, and resilience metrics.

The reversibility tests are the important ones. If an effect did not undo
cleanly, "time to recover" could never be measured — the disruption would never
lift — and two scenarios could not share a simulation without corrupting each
other's state.
"""

from __future__ import annotations

import pytest

from pharmadt.crisis.injector import apply_scenario, revert_scenario
from pharmadt.crisis.scenarios import (
    SCENARIO_DIR,
    Effect,
    Scenario,
    load_all,
    load_scenario,
    measure_resilience,
)


@pytest.fixture
def world():
    from pharmadt.twin.simulation import build_world

    return build_world(seed=42)


def scenario(*effects: Effect, trigger: int = 10, duration: int = 20) -> Scenario:
    return Scenario("test", "a test scenario", trigger, duration, list(effects))


# ── Definitions ───────────────────────────────────────────────────────


def test_all_four_scenarios_load() -> None:
    """FR-09 asks for parameterised what-ifs; these are the four."""
    scenarios = load_all()
    assert {s.name for s in scenarios} == {
        "pandemic_surge", "factory_shutdown", "coldchain_failure", "route_disruption",
    }


@pytest.mark.parametrize("name", [p.stem for p in sorted(SCENARIO_DIR.glob("*.yaml"))])
def test_every_scenario_is_well_formed(name: str) -> None:
    loaded = load_scenario(name)
    assert loaded.trigger_day >= 0
    assert loaded.duration_days > 0
    assert loaded.effects
    assert loaded.end_day == loaded.trigger_day + loaded.duration_days
    assert loaded.description


def test_a_missing_scenario_is_an_error() -> None:
    with pytest.raises(FileNotFoundError):
        load_scenario("no_such_scenario")


# ── Reversibility ─────────────────────────────────────────────────────


def test_a_demand_surge_applies_and_fully_reverts(world) -> None:
    node_id = "NODE-PH-01"
    drug_id = next(iter(world.nodes[node_id].demand_profiles))
    before = world.nodes[node_id].demand_profiles[drug_id].mean

    spec = scenario(Effect("demand_multiplier", magnitude=10.0, node_types=["PHARMACY"]))
    undo = apply_scenario(world, spec)
    assert world.nodes[node_id].demand_profiles[drug_id].mean == pytest.approx(before * 10)

    revert_scenario(world, spec, undo)
    assert world.nodes[node_id].demand_profiles[drug_id].mean == pytest.approx(before)


def test_disabling_a_node_applies_and_reverts(world) -> None:
    spec = scenario(Effect("disable_node", node_types=["MANUFACTURER"]))
    undo = apply_scenario(world, spec)
    assert "NODE-MFG-01" in world.disabled_nodes

    revert_scenario(world, spec, undo)
    assert world.disabled_nodes == set()


def test_cold_chain_risk_applies_and_reverts(world) -> None:
    spec = scenario(Effect("coldchain_risk", magnitude=15.0))
    undo = apply_scenario(world, spec)
    assert world.coldchain_risk_multiplier == pytest.approx(15.0)

    revert_scenario(world, spec, undo)
    assert world.coldchain_risk_multiplier == pytest.approx(1.0)


def test_removed_edges_come_back_with_their_attributes(world) -> None:
    """Edges are removed, not made expensive: a washed-out road is impassable."""
    source = "NODE-WH-02"
    before = {t: dict(world.graph[source][t]) for t in world.graph.successors(source)}
    assert before

    spec = scenario(Effect("remove_edges", from_nodes=[source]))
    undo = apply_scenario(world, spec)
    assert list(world.graph.successors(source)) == []

    revert_scenario(world, spec, undo)
    after = {t: dict(world.graph[source][t]) for t in world.graph.successors(source)}
    assert after == before


def test_two_scenarios_compose_without_corrupting_each_other(world) -> None:
    """Each holds its own undo record, so neither restores the other's values."""
    node_id = "NODE-PH-01"
    drug_id = next(iter(world.nodes[node_id].demand_profiles))
    original = world.nodes[node_id].demand_profiles[drug_id].mean

    first = scenario(Effect("demand_multiplier", magnitude=2.0, node_types=["PHARMACY"]))
    second = scenario(Effect("demand_multiplier", magnitude=3.0, node_types=["PHARMACY"]))

    undo_first = apply_scenario(world, first)
    undo_second = apply_scenario(world, second)
    assert world.nodes[node_id].demand_profiles[drug_id].mean == pytest.approx(original * 6)

    # Revert out of order, as overlapping windows would.
    revert_scenario(world, second, undo_second)
    assert world.nodes[node_id].demand_profiles[drug_id].mean == pytest.approx(original * 2)
    revert_scenario(world, first, undo_first)
    assert world.nodes[node_id].demand_profiles[drug_id].mean == pytest.approx(original)


def test_active_scenarios_are_tracked(world) -> None:
    spec = scenario(Effect("coldchain_risk", magnitude=2.0))
    undo = apply_scenario(world, spec)
    assert "test" in world.active_scenarios
    revert_scenario(world, spec, undo)
    assert "test" not in world.active_scenarios


def test_an_effect_naming_an_unknown_node_is_ignored(world) -> None:
    spec = scenario(Effect("remove_edges", from_nodes=["NOWHERE"]))
    revert_scenario(world, spec, apply_scenario(world, spec))  # must not raise


# ── Metrics ───────────────────────────────────────────────────────────


class FakeWorld:
    """Just enough world for the metric functions."""

    def __init__(self, outcomes):
        from pharmadt.twin.simulation import DemandOutcome

        self.demand_outcomes = [DemandOutcome("N", "D", d, dem, ful)
                                for d, dem, ful in outcomes]


def test_recovery_requires_the_rate_to_stay_down() -> None:
    """A single quiet day mid-shortage is not a recovery."""
    # One deceptively good day in the middle of a thirty-day shortage.
    outcomes = [
        (day, 100, (100 if day == 20 else 0) if 10 <= day < 40 else 100)
        for day in range(60)
    ]

    metrics = measure_resilience(
        FakeWorld(outcomes), scenario(trigger=10, duration=30), 59, baseline_rate=0.0
    )
    assert metrics.time_to_recover_days is not None
    assert metrics.time_to_recover_days > 10, "the single good day must not count"


def test_a_disruption_that_never_lifts_reports_no_recovery() -> None:
    outcomes = [(d, 100, 100 if d < 10 else 0) for d in range(60)]
    metrics = measure_resilience(
        FakeWorld(outcomes), scenario(trigger=10, duration=40), 59, baseline_rate=0.0
    )
    assert metrics.recovered is False
    assert metrics.time_to_recover_days is None


def test_peak_and_unmet_are_measured_over_the_window() -> None:
    outcomes = [(d, 100, 100 if d < 10 or d >= 20 else 50) for d in range(40)]
    metrics = measure_resilience(
        FakeWorld(outcomes), scenario(trigger=10, duration=10), 39, baseline_rate=0.0
    )
    assert metrics.peak_stockout == pytest.approx(0.5)
    assert metrics.total_unmet_units == 500


def test_detection_needs_a_rise_above_baseline() -> None:
    outcomes = [(d, 100, 100) for d in range(40)]
    metrics = measure_resilience(
        FakeWorld(outcomes), scenario(trigger=10, duration=10), 39, baseline_rate=0.0
    )
    assert metrics.time_to_detect_days is None
    assert metrics.peak_stockout == 0.0
