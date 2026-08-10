"""Train and score the demand forecasters. Stage 7's Definition of Done.

Produces the comparison table: naive / seasonal-naive / moving-average /
Prophet / LSTM / ensemble. **Beating seasonal-naive is the bar that matters** —
a neural model that merely matches it has bought nothing for its complexity.

Evaluation protocol, stated because it decides whether the numbers mean
anything:

* Every model forecasts the same 14 days from the same origin, given only
  history up to that origin. Anything else compares models on different
  questions.
* Origins are in the **test** split only. The LSTM early-stops on validation;
  test is touched exactly once, at scoring time.
* Prophet refits per origin, which is why the shared protocol uses one origin
  per store. A second, broader sweep scores the cheap models over every origin
  so the headline is not resting on a single date.

Usage::

    python -m pharmadt.ml.train_demand                 # default sizes
    python -m pharmadt.ml.train_demand --stores 400 --eval-stores 60
    python -m pharmadt.ml.train_demand --skip-prophet  # fast iteration
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pharmadt.config import settings
from pharmadt.ml.forecasting import (
    BASELINES,
    HORIZON,
    LOOKBACK,
    beats_seasonal_naive,
    comparison_table,
    evaluate,
    make_windows,
    seasonal_naive_scale,
)

logger = logging.getLogger(__name__)

SALES = Path("data/processed/rossmann_sales.parquet")
ARTIFACTS = Path("experiments")


def _load() -> pd.DataFrame:
    if not SALES.exists():
        raise SystemExit(f"{SALES} missing. Run `make data` first.")
    return pd.read_parquet(SALES)


#: Channels the model sees. The first two are in demand units and get scaled by
#: the window mean; the rest are already bounded and must not be.
DEMAND_CHANNELS = 2


def _series_for(frame: pd.DataFrame, store_id: int) -> pd.DataFrame:
    return frame[frame["store_id"] == store_id].sort_values("date")


def _feature_channels(series: pd.DataFrame) -> list[np.ndarray]:
    """Per-timestep covariates alongside demand.

    Day-of-week goes in as sin/cos rather than as 0-6. An integer would tell
    the network that Sunday (6) is six times Monday (0) and that the two are
    maximally distant, when they are in fact adjacent.
    """
    dow = series["day_of_week"].to_numpy(float)
    sales = series["sales"].to_numpy(float)
    rolling = pd.Series(sales).rolling(7, min_periods=1).mean().to_numpy()

    return [
        rolling,
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
        series["promo"].to_numpy(float),
        series["school_holiday"].to_numpy(float),
    ]


def build_training_windows(
    frame: pd.DataFrame, store_ids: list[int], stride: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Windows from the train split, validation windows from the val split.

    Built per store so no window straddles a boundary between two shops.
    """
    X_train, y_train, X_val, y_val = [], [], [], []

    for store_id in store_ids:
        series = _series_for(frame, store_id)
        for split, X_out, y_out in (
            ("train", X_train, y_train),
            ("val", X_val, y_val),
        ):
            part = series[series["split"] == split]
            if part.empty:
                continue
            X, y = make_windows(
                part["sales"].to_numpy(float),
                extras=_feature_channels(part),
                lookback=LOOKBACK,
                horizon=HORIZON,
                stride=stride,
            )
            if len(X):
                X_out.append(X)
                y_out.append(y)

    def stack(chunks: list[np.ndarray], width: int) -> np.ndarray:
        return (
            np.concatenate(chunks)
            if chunks
            else np.empty((0, LOOKBACK, 1) if width else (0, HORIZON), np.float32)
        )

    return (
        stack(X_train, 1), stack(y_train, 0),
        stack(X_val, 1), stack(y_val, 0),
    )


def evaluation_origins(
    frame: pd.DataFrame, store_ids: list[int]
) -> list[dict[str, Any]]:
    """One evaluation case per store: history up to the origin, next 14 actuals."""
    cases = []
    for store_id in store_ids:
        series = _series_for(frame, store_id)
        past = series[series["split"] != "test"]
        history = past["sales"].to_numpy(float)
        actual = series.loc[series["split"] == "test", "sales"].to_numpy(float)

        if len(history) < LOOKBACK or len(actual) < HORIZON:
            continue

        # The model's input window is the final LOOKBACK days before the
        # origin, with the same channels it was trained on.
        tail = past.iloc[-LOOKBACK:]
        window = np.stack(
            [tail["sales"].to_numpy(float), *_feature_channels(tail)], axis=-1
        )
        cases.append(
            {
                "store_id": store_id,
                "history": history,
                "dates": past["date"],
                "window": window,
                "actual": actual[:HORIZON],
            }
        )
    return cases


def score_shared_protocol(
    cases: list[dict[str, Any]], lstm: Any, prophet_cls: Any | None
) -> dict[str, dict[str, float]]:
    """Every model, same origins, same horizons."""
    actuals = np.stack([c["actual"] for c in cases])
    windows = np.stack([c["window"] for c in cases])
    scale = seasonal_naive_scale(actuals, windows[..., 0])

    predictions: dict[str, np.ndarray] = {
        name: forecast(windows[..., 0], HORIZON) for name, forecast in BASELINES.items()
    }
    predictions["lstm"] = lstm.predict(windows, HORIZON)

    if prophet_cls is not None:
        rows = []
        for index, case in enumerate(cases):
            model = prophet_cls()
            model.fit(pd.DataFrame({"ds": case["dates"], "y": case["history"]}))
            rows.append(model.predict(horizon=HORIZON))
            if (index + 1) % 10 == 0:
                logger.info("  prophet: fitted %d/%d series", index + 1, len(cases))
        predictions["prophet"] = np.stack(rows)

        # Ensemble the two real models. Equal weights: with one origin per
        # store there is no honest held-out set left to tune weights on, and
        # tuning them on these same points would be leakage.
        predictions["ensemble"] = 0.5 * (predictions["lstm"] + predictions["prophet"])

    return {
        name: evaluate(actuals, predicted, scale)
        for name, predicted in predictions.items()
    }


def score_broad_sweep(
    frame: pd.DataFrame, store_ids: list[int], lstm: Any, stride: int = 7
) -> dict[str, dict[str, float]]:
    """Cheap models over every test origin, as a check on the headline table."""
    X, y = [], []
    for store_id in store_ids:
        series = _series_for(frame, store_id)
        part = series[series["split"] == "test"]
        if part.empty:
            continue
        Xi, yi = make_windows(
            part["sales"].to_numpy(float),
            extras=_feature_channels(part),
            lookback=LOOKBACK, horizon=HORIZON, stride=stride,
        )
        if len(Xi):
            X.append(Xi)
            y.append(yi)
    if not X:
        return {}

    X_all, y_all = np.concatenate(X), np.concatenate(y)
    history = X_all[..., 0]

    scale = seasonal_naive_scale(y_all, history)
    results = {
        name: evaluate(y_all, forecast(history, HORIZON), scale)
        for name, forecast in BASELINES.items()
    }
    results["lstm"] = evaluate(y_all, lstm.predict(X_all, HORIZON), scale)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and score demand forecasters.")
    parser.add_argument("--stores", type=int, default=300, help="stores used for training")
    parser.add_argument("--eval-stores", type=int, default=40,
                        help="stores in the shared-protocol comparison")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--skip-prophet", action="store_true")
    parser.add_argument("--out", default="experiments/demand_comparison.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    frame = _load()

    rng = np.random.default_rng(settings.sim_seed)
    available = np.sort(frame["store_id"].unique())
    chosen = rng.choice(available, size=min(args.stores, len(available)), replace=False)
    train_stores = sorted(int(s) for s in chosen)
    eval_stores = train_stores[: args.eval_stores]

    print(f"Training on {len(train_stores)} stores, "
          f"evaluating {len(eval_stores)} under the shared protocol\n")

    started = time.perf_counter()
    X_train, y_train, X_val, y_val = build_training_windows(
        frame, train_stores, stride=args.stride
    )
    print(f"windows: {len(X_train):,} train, {len(X_val):,} val "
          f"({time.perf_counter() - started:.1f}s)")

    from pharmadt.ml.lstm import LSTMForecaster

    lstm = LSTMForecaster(
        n_features=X_train.shape[-1],
        demand_channels=DEMAND_CHANNELS,
        epochs=args.epochs,
    )
    print(f"{lstm!r}")
    started = time.perf_counter()
    lstm.fit(X_train, y_train, X_val, y_val, verbose=False)
    print(f"trained in {time.perf_counter() - started:.1f}s "
          f"over {len(lstm.history_)} epochs "
          f"(best val {min(h['val'] for h in lstm.history_):.5f})\n")

    prophet_cls = None
    if not args.skip_prophet:
        from pharmadt.ml.prophet_model import ProphetForecaster

        prophet_cls = ProphetForecaster
        print(f"Fitting Prophet on {len(eval_stores)} series...")

    cases = evaluation_origins(frame, eval_stores)
    started = time.perf_counter()
    shared = score_shared_protocol(cases, lstm, prophet_cls)
    print(f"scored {len(cases)} origins x {HORIZON} days "
          f"({time.perf_counter() - started:.1f}s)\n")

    print(f"Shared protocol - {len(cases)} origins, {HORIZON}-day horizon")
    print(comparison_table(shared))

    broad = score_broad_sweep(frame, eval_stores, lstm)
    if broad:
        print(f"\nBroad sweep - every test origin ({broad['naive']['n']:,} points)")
        print(comparison_table(broad))

    print("\n" + "=" * 60)
    for name in ("lstm", "prophet", "ensemble"):
        if name in shared:
            verdict = "BEATS" if beats_seasonal_naive(shared, name) else "does NOT beat"
            print(f"  {name:<10} {verdict} seasonal-naive on sMAPE")
    zero_pct = shared["naive"]["zero_actuals_pct"]
    print(f"\n  {zero_pct:.1f}% of actuals are zero; MAPE excludes them, "
          "sMAPE and MASE do not.")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"shared_protocol": shared, "broad_sweep": broad}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
