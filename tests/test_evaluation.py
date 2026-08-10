"""Stage 15 evaluation: aggregation, rendering, and claim validation.

The aggregation tests matter more than they look. The report table is built
from these functions, so a bug here would put a wrong number in front of an
examiner while every simulation underneath it was correct.
"""

from __future__ import annotations

import math

import pytest

from pharmadt.evaluation import METRICS, aggregate, render, validate_abstract


def row(stockout=1.0, wastage=100.0, mape=5.0, delivery=1000.0, inventory=50_000.0):
    return {
        "stockout_pct": stockout,
        "wastage_units": wastage,
        "forecast_mape": mape,
        "delivery_km": delivery,
        "average_inventory": inventory,
    }


def matrix(*configs) -> dict:
    return {name: {"summary": aggregate(rows), "seeds": [1], "seconds": 1.0}
            for name, rows in configs}


# ── Aggregation ───────────────────────────────────────────────────────


def test_the_mean_is_reported_with_its_spread() -> None:
    """A mean alone hid wastage swinging between 0 and 1,755 units."""
    summary = aggregate([row(stockout=1.0), row(stockout=3.0)])
    assert summary["stockout_pct"]["mean"] == pytest.approx(2.0)
    assert summary["stockout_pct"]["std"] == pytest.approx(1.414, abs=0.01)


def test_a_single_seed_has_no_spread() -> None:
    summary = aggregate([row()])
    assert summary["stockout_pct"]["std"] == 0.0
    assert summary["stockout_pct"]["n"] == 1


def test_missing_metrics_are_dropped_not_counted_as_zero() -> None:
    """A config with no Route Agent has no delivery cost; zero would be a lie."""
    summary = aggregate([row(delivery=float("nan")), row(delivery=float("nan"))])
    assert math.isnan(summary["delivery_km"]["mean"])
    assert summary["delivery_km"]["n"] == 0
    assert summary["stockout_pct"]["n"] == 2


def test_partially_missing_metrics_average_over_what_exists() -> None:
    summary = aggregate([row(mape=4.0), row(mape=float("nan")), row(mape=6.0)])
    assert summary["forecast_mape"]["mean"] == pytest.approx(5.0)
    assert summary["forecast_mape"]["n"] == 2


def test_every_declared_metric_is_aggregated() -> None:
    summary = aggregate([row()])
    assert set(summary) == set(METRICS)


# ── Rendering ─────────────────────────────────────────────────────────


def test_the_table_is_ascii_only() -> None:
    """The demo runs on a Windows console that cannot encode a plus-minus sign."""
    text = render(matrix(("baseline", [row(), row(stockout=2.0)])))
    assert text.isascii(), [c for c in text if not c.isascii()]


def test_a_missing_metric_renders_as_a_dash_not_a_number() -> None:
    text = render(matrix(("no route", [row(delivery=float("nan"))])))
    assert "--" in text


def test_every_configuration_gets_a_row() -> None:
    text = render(matrix(("baseline", [row()]), ("+ agents", [row()])))
    assert "baseline" in text
    assert "+ agents" in text
    assert len(text.splitlines()) == 4  # header, rule, two rows


# ── Claim validation ──────────────────────────────────────────────────


def test_a_measured_wastage_reduction_is_checked_against_the_claim() -> None:
    findings = validate_abstract(
        matrix(("baseline", [row(wastage=1000.0)]), ("full", [row(wastage=650.0)]))
    )
    text = " ".join(findings)
    assert "35.0%" in text
    assert "MEETS" in text


def test_exceeding_the_claim_is_labelled_as_exceeding() -> None:
    findings = validate_abstract(
        matrix(("baseline", [row(wastage=1000.0)]), ("full", [row(wastage=100.0)]))
    )
    assert "EXCEEDS" in " ".join(findings)


def test_falling_short_is_stated_plainly() -> None:
    """The guide's instruction: update the abstract to the measured value."""
    findings = validate_abstract(
        matrix(("baseline", [row(wastage=1000.0)]), ("full", [row(wastage=950.0)]))
    )
    assert "FALLS SHORT OF" in " ".join(findings)


def test_a_baseline_with_no_wastage_says_so_instead_of_dividing_by_zero() -> None:
    findings = validate_abstract(
        matrix(("baseline", [row(wastage=0.0)]), ("full", [row(wastage=0.0)]))
    )
    text = " ".join(findings)
    assert "no headroom" in text
    assert "Stage 8" in text


def test_the_stockout_reduction_is_always_reported() -> None:
    findings = validate_abstract(
        matrix(("baseline", [row(stockout=1.0)]), ("full", [row(stockout=0.1)]))
    )
    assert "90.0% reduction" in " ".join(findings)
