"""Normalising the experiment artifacts for the dashboard.

These functions decide what number an examiner reads off the screen, so a bug
here misreports a result while every experiment underneath it was correct.
That is the same failure mode the evaluation tests guard against, one layer up.
"""

from __future__ import annotations

import json

import pytest

from pharmadt.api.results import (
    BUILDERS,
    PROTOCOLS,
    Artifact,
    _evaluation,
    _forecasting,
    _gate,
    _mean_std,
    _num,
    load,
    load_all,
)

# ── Formatting ────────────────────────────────────────────────────────


def test_a_missing_metric_is_a_dash_not_a_zero() -> None:
    """Zero would read as a free delivery rather than a column that does not apply."""
    assert _num(None) == "--"
    assert _num(float("nan")) == "--"
    assert _num(0) == "0.00"


def test_numbers_are_grouped_for_reading() -> None:
    assert _num(62942, 0) == "62,942"


def test_mean_is_never_shown_without_its_spread() -> None:
    assert _mean_std({"mean": 2.0, "std": 1.414}, 2) == "2.00 +/- 1.41"


def test_an_absent_summary_renders_as_a_dash() -> None:
    assert _mean_std(None) == "--"
    assert _mean_std({"mean": float("nan")}) == "--"


# ── Registry ──────────────────────────────────────────────────────────


def test_every_registered_artifact_names_the_command_that_makes_it() -> None:
    """A result that says 'not run' without a command just relocates the confusion."""
    for artifact in load_all():
        assert artifact.command.startswith("make ")
        assert artifact.question.endswith("?")


def test_an_unknown_key_is_not_invented() -> None:
    assert load("no-such-experiment") is None


def test_every_builder_key_is_unique() -> None:
    keys = [b[0] for b in BUILDERS]
    assert len(keys) == len(set(keys))


# ── Missing and malformed inputs ──────────────────────────────────────


def test_a_missing_artifact_reports_absent_rather_than_empty(monkeypatch, tmp_path) -> None:
    """An experiment that was never run must not render as a table of zeros."""
    monkeypatch.setattr("pharmadt.api.results.EXPERIMENTS", tmp_path)
    artifact = load("gate")

    assert artifact.present is False
    assert artifact.rows == []
    assert artifact.command == "make gate"


def test_a_malformed_artifact_is_reported_not_raised(monkeypatch, tmp_path) -> None:
    """A truncated JSON file must not take the whole dashboard down with a 500."""
    (tmp_path / "integration_gate.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr("pharmadt.api.results.EXPERIMENTS", tmp_path)
    artifact = load("gate")

    assert artifact.present is False
    assert "Could not read" in artifact.footnote


# ── The comparisons that carry the claims ─────────────────────────────


def _matrix(baseline: float, control: float, full: float) -> dict:
    def arm(wastage: float) -> dict:
        return {
            "summary": {
                "stockout_pct": {"mean": 1.0, "std": 0.1, "n": 10},
                "wastage_units": {"mean": wastage, "std": 0.0, "n": 10},
                "forecast_mape": {"mean": 5.0, "std": 0.0, "n": 10},
                "delivery_km": {"mean": float("nan"), "std": 0.0, "n": 0},
                "average_inventory": {"mean": 50_000.0, "std": 0.0, "n": 10},
            },
            "seeds": list(range(10)),
        }

    return {
        "baseline (no agents)": arm(baseline),
        "+ inventory & demand": arm(control),
        "+ expiry (redistribution)": arm(full),
    }


def test_wastage_is_measured_against_the_no_redistribution_control() -> None:
    """Not against the no-agent baseline, which wastes less by stocking out."""
    art = Artifact("k", "t", "q?", "make x", "f")
    _evaluation(_matrix(baseline=50.0, control=1000.0, full=650.0), art)

    reduction = next(h for h in art.headline if h["label"] == "wastage reduction")
    assert reduction["value"] == "35.0%"
    assert "no-redistribution control" in reduction["note"]


def test_the_confounding_is_stated_in_the_footnote() -> None:
    art = Artifact("k", "t", "q?", "make x", "f")
    _evaluation(_matrix(50.0, 1000.0, 650.0), art)
    assert "runs out instead" in art.footnote


def test_a_missing_delivery_column_stays_a_dash_through_the_pipeline() -> None:
    art = Artifact("k", "t", "q?", "make x", "f")
    _evaluation(_matrix(50.0, 1000.0, 650.0), art)
    assert all(row[4] == "--" for row in art.rows)


def test_the_forecast_baseline_is_seasonal_naive_not_naive() -> None:
    """Beating 'yesterday' is trivial for weekly-seasonal demand; the seasonal
    baseline is the one the claim has to clear."""
    art = Artifact("k", "t", "q?", "make x", "f")
    _forecasting(
        {
            "shared_protocol": {
                "naive": {"sMAPE": 32.49},
                "seasonal_naive": {"sMAPE": 25.78},
                "lstm": {"sMAPE": 18.07},
            }
        },
        art,
    )
    headline = next(h for h in art.headline if "seasonal-naive" in h["label"])
    assert headline["value"] == "30%"  # from 25.78, not from 32.49


def test_both_forecasting_protocols_are_shown() -> None:
    """They give different answers, so showing only the flattering one would
    overstate the deep model."""
    art = Artifact("k", "t", "q?", "make x", "f")
    _forecasting(
        {
            "shared_protocol": {"seasonal_naive": {"sMAPE": 25.78}, "lstm": {"sMAPE": 18.07}},
            "broad_sweep": {"seasonal_naive": {"sMAPE": 30.06}, "lstm": {"sMAPE": 13.19}},
        },
        art,
    )
    labels = {h["label"] for h in art.headline}
    assert labels == {"LSTM vs seasonal-naive", "on the broad sweep"}
    assert set(PROTOCOLS) == {"shared_protocol", "broad_sweep"}


def test_the_footnote_names_the_methods_that_beat_the_lstm() -> None:
    """An examiner who reads the table will see it; better to have said it first."""
    art = Artifact("k", "t", "q?", "make x", "f")
    _forecasting({"shared_protocol": {"seasonal_naive": {"sMAPE": 25.78}}}, art)
    assert "moving average" in art.footnote


def test_a_failed_gate_check_is_rendered_as_failed() -> None:
    art = Artifact("k", "t", "q?", "make x", "f")
    _gate([{"number": "1", "name": "a", "passed": True, "detail": "ok"},
           {"number": "2", "name": "b", "passed": False, "detail": "nope"}], art)

    assert art.headline[0]["value"] == "1/2"
    assert [r[2] for r in art.rows] == ["PASS", "FAIL"]


# ── Against the real artifacts on this machine ────────────────────────


@pytest.mark.slow
def test_every_present_artifact_renders_a_rectangular_table() -> None:
    """A row longer than the header would silently drop a column in the browser."""
    for artifact in load_all():
        if not artifact.present:
            continue
        for row in artifact.rows:
            assert len(row) == len(artifact.columns), artifact.key


@pytest.mark.slow
def test_the_normalised_shape_is_json_serialisable() -> None:
    """It is served straight over HTTP, so a stray numpy scalar would 500."""
    json.dumps([a.as_dict() for a in load_all()])
