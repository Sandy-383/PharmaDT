"""End-to-end smoke tests for the reporting harnesses.

Each deliverable — the gate, the ablation matrix, the crisis runner, the CVRPLIB
benchmark, the federated experiment — is exercised at the smallest size that
still runs the real code path. These are the artefacts an examiner will ask to
see regenerated, and "it worked when I ran it last week" is not a guarantee that
survives a refactor.

Kept deliberately small: the point is that the plumbing holds, not to reproduce
the published numbers, which the full runs do.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


# ── Ablation harness ──────────────────────────────────────────────────


def test_a_variant_runs_and_reports_every_metric() -> None:
    from pharmadt.ablation import REPORTED, run_variant

    row = run_variant("baseline", days=20, seed=42)
    assert row["variant"] == "baseline"
    assert all(row[metric] is not None for metric in REPORTED)


def test_variants_are_comparable_on_the_same_seed() -> None:
    from pharmadt.ablation import compare

    rows, summary = compare(["baseline", "inventory"], days=20, seeds=[42])
    assert len(rows) == 2
    assert {s["variant"] for s in summary} == {"baseline", "inventory"}
    assert all(s["runs"] == 1 for s in summary)


def test_the_agent_variant_records_its_decisions() -> None:
    from pharmadt.ablation import run_variant

    assert run_variant("inventory", days=20, seed=42)["agent_decisions"] > 0


# ── Stage 15 matrix ───────────────────────────────────────────────────


def test_the_experiment_matrix_builds_and_renders() -> None:
    from pharmadt.evaluation import render, run_matrix, validate_abstract

    matrix = run_matrix(seeds=[42], days=20, include_route=False)
    assert len(matrix) >= 3

    table = render(matrix)
    assert table.isascii()
    assert "baseline" in table
    assert validate_abstract(matrix)


# ── Integration gate ──────────────────────────────────────────────────


def test_the_gate_runs_every_check_and_reports_evidence() -> None:
    """The Stage 10.5 harness itself, at 20 days rather than 365."""
    from pharmadt.gate import run_gate

    checks, context = run_gate(days=20, seed=42)

    assert len(checks) == 6
    assert all(c.detail for c in checks), "a check with no evidence proves nothing"
    assert {c.number for c in checks} == {"1", "2", "3", "4", "5", "6"}

    by_number = {c.number: c for c in checks}
    assert by_number["1"].passed, by_number["1"].detail
    assert by_number["2"].passed, by_number["2"].detail
    assert by_number["4"].passed, by_number["4"].detail  # tamper detection


def test_the_gate_leaves_the_chain_verifying_afterwards() -> None:
    """The tamper step must restore what it changed, or the demo is one-shot."""
    from pharmadt.gate import run_gate

    _, context = run_gate(days=20, seed=42)
    assert context["ledger"].verify_chain() is True


# ── Crisis runner ─────────────────────────────────────────────────────


def test_a_crisis_scenario_runs_and_measures_recovery() -> None:
    from pharmadt.crisis.experiment import run_one
    from pharmadt.crisis.scenarios import Effect, Scenario

    scenario = Scenario(
        "smoke", "a short demand surge", trigger_day=5, duration_days=5,
        effects=[Effect("demand_multiplier", magnitude=8.0, node_types=["PHARMACY"])],
    )
    metrics = run_one(scenario, days=25, seed=42, with_agents=False)

    assert metrics.scenario == "smoke"
    assert metrics.peak_stockout >= 0.0
    assert len(metrics.daily_stockout) == 26


def test_the_same_scenario_runs_under_both_policies() -> None:
    from pharmadt.crisis.experiment import run_one
    from pharmadt.crisis.scenarios import Effect, Scenario

    scenario = Scenario(
        "smoke", "manufacturer halts", trigger_day=5, duration_days=5,
        effects=[Effect("disable_node", node_types=["MANUFACTURER"])],
    )
    base = run_one(scenario, days=25, seed=42, with_agents=False)
    agents = run_one(scenario, days=25, seed=42, with_agents=True)

    assert base.total_unmet_units >= 0
    assert agents.total_unmet_units >= 0


# ── CVRPLIB benchmark ─────────────────────────────────────────────────


def test_the_routing_benchmark_reports_a_gap_against_a_published_optimum() -> None:
    from pathlib import Path

    from pharmadt.ml.benchmark_routing import benchmark
    from pharmadt.ml.preprocessing import RAW

    if not any((RAW / "cvrplib").glob("*.vrp")):
        pytest.skip("CVRPLIB instances not downloaded; run `make data`")

    rows = benchmark(time_limit_s=2)
    assert rows
    for row in rows:
        assert row["optimum"] > 0
        assert row["expected_stops"] == row["n"] - 1
        # An unsolved instance must report no gap rather than a flattering one.
        if row["gap_pct"] is None:
            assert row["served"] < row["expected_stops"]
        else:
            assert row["served"] == row["expected_stops"]
    assert Path("experiments").exists()


# ── Federated experiment ──────────────────────────────────────────────


def test_the_federated_experiment_produces_comparable_variants() -> None:
    from pathlib import Path

    from pharmadt.federated.experiment import run_experiment

    if not Path("data/processed/rossmann_sales.parquet").exists():
        pytest.skip("Rossmann data not built; run `make data`")

    results = run_experiment(
        stores=6, clients=3, rounds=2, epsilons=(5.0,), verbose=False
    )
    for label in ("centralised", "federated_iid", "federated_noniid"):
        assert results[label]["sMAPE"] > 0

    # Every variant must be scored on the same held-out set.
    assert results["_config"]["test_windows"] > 0
