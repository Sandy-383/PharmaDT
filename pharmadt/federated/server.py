"""FedAvg aggregation and the federated training loop.

The aggregation is a sample-weighted mean of client weights — FedAvg as
originally specified. It is written out rather than delegated because it is
eight lines, it is the thing the report claims, and a reader should be able to
check it.

Weighting by sample count is not cosmetic: an unweighted mean would give a
client holding twenty windows the same influence as one holding two thousand,
which under the Dirichlet split is most of them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pharmadt.federated.client import ClientUpdate, PharmaClient
from pharmadt.federated.privacy import (
    PrivacyBudget,
    accountant_epsilon,
    add_gaussian_noise,
    clip_update,
)


@dataclass(slots=True)
class RoundResult:
    """What one federated round produced."""

    round_number: int
    n_clients: int
    mean_smape: float
    clipped_fraction: float = 0.0


@dataclass(slots=True)
class FederatedResult:
    """Outcome of a full federated run."""

    label: str
    rounds: list[RoundResult] = field(default_factory=list)
    final_weights: list[np.ndarray] = field(default_factory=list)
    privacy: PrivacyBudget | None = None
    client_summaries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def best_smape(self) -> float:
        return min((r.mean_smape for r in self.rounds), default=float("nan"))

    @property
    def final_smape(self) -> float:
        return self.rounds[-1].mean_smape if self.rounds else float("nan")


def federated_average(updates: Sequence[ClientUpdate]) -> list[np.ndarray]:
    """Sample-weighted mean of client weights. This is FedAvg."""
    if not updates:
        raise ValueError("cannot aggregate zero client updates")

    total = sum(u.n_samples for u in updates)
    if total == 0:
        raise ValueError("every client reported zero samples")

    n_tensors = len(updates[0].weights)
    if any(len(u.weights) != n_tensors for u in updates):
        raise ValueError("clients disagree on the model architecture")

    return [
        sum(
            (u.weights[i].astype(np.float64) * (u.n_samples / total) for u in updates),
            start=np.zeros_like(updates[0].weights[i], dtype=np.float64),
        ).astype(updates[0].weights[i].dtype)
        for i in range(n_tensors)
    ]


def run_federated(
    clients: list[PharmaClient],
    model_factory: Callable[[], Any],
    rounds: int = 30,
    clip_norm: float | None = None,
    noise_multiplier: float = 0.0,
    delta: float = 1e-5,
    seed: int = 42,
    label: str = "federated",
    verbose: bool = False,
) -> FederatedResult:
    """Train ``rounds`` of FedAvg across ``clients``.

    With ``noise_multiplier > 0`` each client's *update* is clipped and the
    aggregate is perturbed — central differential privacy, with the budget
    tracked by Opacus.
    """
    if not clients:
        raise ValueError("federated training needs at least one client")

    rng = np.random.default_rng(seed)
    global_weights = model_factory().get_weights()
    # Calibrated below from the first round's updates when not supplied. A clip
    # norm chosen blind either never binds or binds so hard every client
    # contributes the same thing.
    active_clip = clip_norm
    result = FederatedResult(label=label)
    result.client_summaries = [
        {"client_id": c.client_id, "n_samples": c.n_samples} for c in clients
    ]

    for round_number in range(1, rounds + 1):
        updates: list[ClientUpdate] = []
        clipped = 0

        raw = [client.fit(global_weights) for client in clients]

        if noise_multiplier > 0 and active_clip is None:
            from pharmadt.federated.privacy import median_update_norm

            active_clip = median_update_norm(
                [
                    [w - g for w, g in zip(u.weights, global_weights, strict=True)]
                    for u in raw
                ]
            )
            if verbose:
                print(f"    adaptive clip norm = {active_clip:.3f}")

        for update in raw:
            if noise_multiplier > 0:
                # Clip the delta, not the weights: the sensitivity that matters
                # is how far one client can move the model, not where to.
                delta_w = [
                    w - g for w, g in zip(update.weights, global_weights, strict=True)
                ]
                bounded, original_norm = clip_update(delta_w, active_clip)
                if original_norm > active_clip:
                    clipped += 1
                update = ClientUpdate(
                    update.client_id,
                    [g + d for g, d in zip(global_weights, bounded, strict=True)],
                    update.n_samples,
                    update.metrics,
                )
            updates.append(update)

        global_weights = federated_average(updates)

        if noise_multiplier > 0:
            global_weights = add_gaussian_noise(
                global_weights, noise_multiplier, active_clip, rng, len(clients)
            )

        scores = [c.evaluate(global_weights)["sMAPE"] for c in clients]
        weights = np.array([c.n_samples for c in clients], dtype=float)
        mean_smape = float(np.average(scores, weights=weights))

        result.rounds.append(
            RoundResult(round_number, len(clients), mean_smape, clipped / len(clients))
        )
        if verbose and (round_number % 5 == 0 or round_number == 1):
            print(f"    round {round_number:>3}  sMAPE {mean_smape:.3f}")

    result.final_weights = global_weights
    if noise_multiplier > 0:
        result.privacy = PrivacyBudget(
            epsilon=accountant_epsilon(noise_multiplier, rounds, 1.0, delta),
            delta=delta,
            rounds=rounds,
            noise_multiplier=noise_multiplier,
            clip_norm=active_clip or 0.0,
        )
    return result
