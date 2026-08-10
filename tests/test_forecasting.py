"""Forecast metrics, windowing, and baselines.

These guard the scoreboard. A bug here would not make a model fail — it would
make a bad model look good, which is worse.
"""

from __future__ import annotations

import numpy as np
import pytest

from pharmadt.ml.forecasting import (
    HORIZON,
    LOOKBACK,
    beats_seasonal_naive,
    evaluate,
    make_windows,
    mape,
    mase,
    moving_average_forecast,
    naive_forecast,
    seasonal_naive_forecast,
    seasonal_naive_scale,
    smape,
)

# ── Metrics ───────────────────────────────────────────────────────────


def test_a_perfect_forecast_scores_zero_error() -> None:
    actual = np.array([10.0, 20.0, 30.0])
    scores = evaluate(actual, actual, scale=1.0)
    assert scores["MAPE"] == pytest.approx(0.0)
    assert scores["sMAPE"] == pytest.approx(0.0)
    assert scores["RMSE"] == pytest.approx(0.0)


def test_mape_ignores_zero_actuals_rather_than_dividing_by_them() -> None:
    """Pharmacies have zero-demand days; MAPE is undefined there."""
    actual = np.array([0.0, 100.0])
    predicted = np.array([50.0, 110.0])
    assert mape(actual, predicted) == pytest.approx(10.0)  # only the 100 counts


def test_mape_is_nan_when_every_actual_is_zero() -> None:
    assert np.isnan(mape(np.zeros(3), np.ones(3)))


def test_smape_stays_defined_where_mape_cannot() -> None:
    actual = np.array([0.0, 0.0])
    assert np.isfinite(smape(actual, np.array([5.0, 5.0])))


def test_evaluate_reports_how_many_actuals_were_zero() -> None:
    """Without this, an excluded-zeros MAPE looks like a complete one."""
    scores = evaluate(np.array([0.0, 0.0, 10.0, 10.0]), np.ones(4), scale=1.0)
    assert scores["zero_actuals_pct"] == pytest.approx(50.0)


def test_mase_below_one_means_it_beat_the_reference() -> None:
    actual = np.array([10.0, 10.0])
    assert mase(actual, np.array([10.0, 10.0]), scale=5.0) == pytest.approx(0.0)
    assert mase(actual, np.array([15.0, 15.0]), scale=5.0) == pytest.approx(1.0)
    assert mase(actual, np.array([20.0, 20.0]), scale=5.0) == pytest.approx(2.0)


def test_mase_is_nan_for_a_degenerate_scale() -> None:
    assert np.isnan(mase(np.ones(2), np.ones(2), scale=0.0))


def test_seasonal_naive_scores_exactly_one_against_its_own_scale() -> None:
    """Makes the whole table readable: 1.000 is the bar, below it is better."""
    rng = np.random.default_rng(0)
    history = rng.uniform(50, 150, size=(6, LOOKBACK))
    actual = rng.uniform(50, 150, size=(6, HORIZON))

    scale = seasonal_naive_scale(actual, history)
    scores = evaluate(actual, seasonal_naive_forecast(history, HORIZON), scale)
    assert scores["MASE"] == pytest.approx(1.0)


# ── Windowing ─────────────────────────────────────────────────────────


def test_windows_have_the_declared_shapes() -> None:
    X, y = make_windows(list(range(100)))
    assert X.shape[1:] == (LOOKBACK, 1)
    assert y.shape[1] == HORIZON
    assert len(X) == len(y)


def test_a_window_target_is_the_days_that_follow_its_input() -> None:
    X, y = make_windows(list(range(100)), stride=1)
    assert X[0, -1, 0] == pytest.approx(LOOKBACK - 1)
    assert y[0, 0] == pytest.approx(LOOKBACK)


def test_extra_channels_are_stacked_alongside_demand() -> None:
    values = list(range(80))
    X, _ = make_windows(values, extras=[np.ones(80), np.zeros(80)])
    assert X.shape[-1] == 3
    assert X[0, :, 1].tolist() == [1.0] * LOOKBACK


def test_a_series_too_short_yields_no_windows() -> None:
    """Never straddle a boundary; a short store contributes nothing instead."""
    X, y = make_windows(list(range(LOOKBACK + HORIZON - 1)))
    assert len(X) == 0 and len(y) == 0


def test_stride_controls_window_overlap() -> None:
    dense, _ = make_windows(list(range(200)), stride=1)
    sparse, _ = make_windows(list(range(200)), stride=5)
    assert len(sparse) < len(dense)


# ── Baselines ─────────────────────────────────────────────────────────


def test_naive_repeats_the_last_observation() -> None:
    assert naive_forecast(np.array([1.0, 2.0, 9.0]), 3).tolist() == [9.0, 9.0, 9.0]


def test_seasonal_naive_repeats_the_last_week() -> None:
    history = np.arange(14, dtype=float)
    assert seasonal_naive_forecast(history, 7).tolist() == list(range(7, 14))


def test_seasonal_naive_tiles_beyond_one_season() -> None:
    history = np.arange(14, dtype=float)
    forecast = seasonal_naive_forecast(history, 14)
    assert forecast.tolist() == list(range(7, 14)) * 2


def test_seasonal_naive_falls_back_when_history_is_shorter_than_a_week() -> None:
    assert seasonal_naive_forecast(np.array([3.0, 4.0]), 3).tolist() == [4.0, 4.0, 4.0]


def test_moving_average_is_flat_at_the_recent_mean() -> None:
    history = np.array([0.0] * 7 + [10.0] * 7)
    assert moving_average_forecast(history, 3).tolist() == [10.0, 10.0, 10.0]


def test_baselines_handle_batches() -> None:
    history = np.arange(28, dtype=float).reshape(2, 14)
    assert seasonal_naive_forecast(history, HORIZON).shape == (2, HORIZON)
    assert naive_forecast(history, HORIZON).shape == (2, HORIZON)


# ── The bar ───────────────────────────────────────────────────────────


def test_beating_the_bar_is_decided_by_a_function_not_by_prose() -> None:
    results = {
        "seasonal_naive": {"sMAPE": 30.0},
        "good": {"sMAPE": 15.0},
        "bad": {"sMAPE": 45.0},
    }
    assert beats_seasonal_naive(results, "good")
    assert not beats_seasonal_naive(results, "bad")


def test_a_missing_model_does_not_count_as_beating_it() -> None:
    assert not beats_seasonal_naive({"seasonal_naive": {"sMAPE": 30.0}}, "absent")
