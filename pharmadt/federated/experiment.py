"""Stage 11 Definition of Done: centralised vs federated, IID vs non-IID vs DP.

Every variant is scored on the **same held-out test set**, which no client
sees during training. Scoring each variant on its own clients' data would make
the skewed split look easy — a client that only saw two stores is very good at
predicting those two stores.

Usage::

    python -m pharmadt.federated.experiment
    python -m pharmadt.federated.experiment --clients 5 --rounds 30
    python -m pharmadt.federated.experiment --quick     # smaller, for iteration
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from pharmadt.federated.client import PharmaClient
from pharmadt.federated.partition import dirichlet_partition, iid_partition, skew_report
from pharmadt.federated.privacy import noise_for_epsilon
from pharmadt.federated.server import run_federated
from pharmadt.ml.forecasting import evaluate, seasonal_naive_scale

RESULTS = Path("experiments/federated.json")


def load_windows(stores: int, seed: int) -> dict[str, Any]:
    """Training windows per store, plus a global held-out test set."""
    import pandas as pd

    from pharmadt.ml.forecasting import HORIZON, LOOKBACK, make_windows
    from pharmadt.ml.train_demand import SALES, _feature_channels, _series_for

    if not SALES.exists():
        raise SystemExit(f"{SALES} missing. Run `make data` first.")

    frame = pd.read_parquet(SALES)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.sort(frame["store_id"].unique()), size=stores, replace=False)

    X_parts, y_parts, series = [], [], []
    Xt_parts, yt_parts = [], []

    for index, store_id in enumerate(sorted(int(s) for s in chosen)):
        rows = _series_for(frame, store_id)
        train = rows[rows["split"] == "train"]
        test = rows[rows["split"] == "test"]

        if not train.empty:
            X, y = make_windows(
                train["sales"].to_numpy(float), extras=_feature_channels(train),
                lookback=LOOKBACK, horizon=HORIZON, stride=4,
            )
            if len(X):
                X_parts.append(X)
                y_parts.append(y)
                series.append(np.full(len(X), index))
        if not test.empty:
            Xt, yt = make_windows(
                test["sales"].to_numpy(float), extras=_feature_channels(test),
                lookback=LOOKBACK, horizon=HORIZON, stride=7,
            )
            if len(Xt):
                Xt_parts.append(Xt)
                yt_parts.append(yt)

    return {
        "X": np.concatenate(X_parts),
        "y": np.concatenate(y_parts),
        "series": np.concatenate(series),
        "X_test": np.concatenate(Xt_parts),
        "y_test": np.concatenate(yt_parts),
    }


def score(model: Any, data: dict[str, Any]) -> dict[str, float]:
    """Score on the shared held-out set."""
    X, y = data["X_test"], data["y_test"]
    scale = seasonal_naive_scale(y, X[..., 0])
    return evaluate(y, model.predict(X), scale)


def make_factory(n_features: int, epochs: int, seed: int):
    from pharmadt.ml.lstm import LSTMForecaster
    from pharmadt.ml.train_demand import DEMAND_CHANNELS

    def factory():
        return LSTMForecaster(
            n_features=n_features,
            demand_channels=DEMAND_CHANNELS,
            epochs=epochs,
            patience=epochs,          # no early stopping inside a federated round
            seed=seed,
        )

    return factory


def run_experiment(
    stores: int = 40,
    clients: int = 5,
    rounds: int = 30,
    local_epochs: int = 1,
    alpha: float = 0.5,
    seed: int = 42,
    epsilons: tuple[float, ...] = (1.0, 5.0, 10.0),
    verbose: bool = True,
) -> dict[str, Any]:
    data = load_windows(stores, seed)
    factory = make_factory(data["X"].shape[-1], local_epochs, seed)
    results: dict[str, Any] = {}

    def report(label: str, model_or_weights: Any) -> dict[str, float]:
        model = factory()
        if isinstance(model_or_weights, list):
            model.set_weights(model_or_weights)
        else:
            model = model_or_weights
        scores = score(model, data)
        results[label] = {k: round(v, 4) for k, v in scores.items() if v is not None}
        if verbose:
            print(f"  {label:<26} sMAPE {scores['sMAPE']:>7.3f}   MASE {scores['MASE']:>6.3f}")
        return scores

    # ── Centralised reference ─────────────────────────────────────────
    if verbose:
        print("\nTraining centralised reference (all data pooled)...")
    central = factory()
    central.epochs = max(rounds * local_epochs, 10)
    central.fit(data["X"], data["y"])
    report("centralised", central)

    # ── Federated, IID ────────────────────────────────────────────────
    iid_shards = iid_partition(data["X"], data["y"], clients, seed)
    if verbose:
        print(f"\nFederated IID: {clients} clients, {rounds} rounds...")
    iid_result = run_federated(
        [PharmaClient(s, factory, local_epochs) for s in iid_shards],
        factory, rounds=rounds, seed=seed, label="federated_iid", verbose=verbose,
    )
    report("federated_iid", iid_result.final_weights)
    results["federated_iid"]["skew"] = skew_report(iid_shards)

    # ── Federated, non-IID ────────────────────────────────────────────
    skewed = dirichlet_partition(
        data["X"], data["y"], data["series"], clients, alpha=alpha, seed=seed
    )
    if verbose:
        print(f"\nFederated non-IID (Dirichlet alpha={alpha})...")
    noniid_result = run_federated(
        [PharmaClient(s, factory, local_epochs) for s in skewed],
        factory, rounds=rounds, seed=seed, label="federated_noniid", verbose=verbose,
    )
    report("federated_noniid", noniid_result.final_weights)
    results["federated_noniid"]["skew"] = skew_report(skewed)

    # ── FedProx on the same skewed split ──────────────────────────────
    if verbose:
        print("\nFedProx (mu=0.01) on the same skewed split...")
    prox_result = run_federated(
        [PharmaClient(s, factory, local_epochs, proximal_mu=0.01) for s in skewed],
        factory, rounds=rounds, seed=seed, label="fedprox_noniid", verbose=False,
    )
    report("fedprox_noniid", prox_result.final_weights)

    # ── Federated + differential privacy ──────────────────────────────
    for epsilon in epsilons:
        sigma = noise_for_epsilon(epsilon, rounds, sample_rate=1.0)
        if verbose:
            print(f"\nFederated + DP at epsilon={epsilon} (sigma={sigma:.3f})...")
        dp_result = run_federated(
            [PharmaClient(s, factory, local_epochs) for s in skewed],
            factory, rounds=rounds, clip_norm=None, noise_multiplier=sigma,
            seed=seed, label=f"dp_eps{epsilon}", verbose=False,
        )
        label = f"federated_dp_eps{epsilon:g}"
        report(label, dp_result.final_weights)
        results[label]["privacy"] = dp_result.privacy.as_dict() if dp_result.privacy else None

    results["_config"] = {
        "stores": stores, "clients": clients, "rounds": rounds,
        "local_epochs": local_epochs, "alpha": alpha,
        "train_windows": int(len(data["X"])), "test_windows": int(len(data["X_test"])),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated learning experiments.")
    parser.add_argument("--stores", type=int, default=40)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--quick", action="store_true", help="small run for iteration")
    args = parser.parse_args()

    if args.quick:
        args.stores, args.clients, args.rounds = 12, 3, 5

    started = time.perf_counter()
    results = run_experiment(
        stores=args.stores, clients=args.clients, rounds=args.rounds,
        local_epochs=args.local_epochs, alpha=args.alpha,
        epsilons=(1.0, 5.0, 10.0) if not args.quick else (5.0,),
    )
    elapsed = time.perf_counter() - started

    print(f"\n{'=' * 68}")
    header = f"{'variant':<26}{'sMAPE':>9}{'MAPE':>9}{'MASE':>8}{'epsilon':>10}"
    print(header)
    print("-" * len(header))
    for label, scores in results.items():
        if label.startswith("_"):
            continue
        eps = (scores.get("privacy") or {}).get("epsilon")
        eps_text = f"{eps:>10.2f}" if eps else f"{'inf':>10}"
        print(f"{label:<26}{scores['sMAPE']:>9.3f}{scores['MAPE']:>9.3f}"
              f"{scores['MASE']:>8.3f}{eps_text}")

    central = results["centralised"]["sMAPE"]
    noniid = results["federated_noniid"]["sMAPE"]
    iid = results["federated_iid"]["sMAPE"]
    print(f"\n  federated cost (non-IID vs centralised): "
          f"{(noniid - central) / central:+.1%} sMAPE")
    print(f"  heterogeneity cost (non-IID vs IID):     {(noniid - iid) / iid:+.1%} sMAPE")
    print("\n  Raw demand data never left a client: only weight arrays and sample")
    print( "  counts cross the boundary (asserted in tests/test_federated.py).")
    print(f"\n  completed in {elapsed:.0f}s")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"  Wrote {RESULTS}")


if __name__ == "__main__":
    main()
