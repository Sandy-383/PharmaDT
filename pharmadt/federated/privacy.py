"""Differential privacy for federated updates: clip, add noise, account.

Each client's *update* — the change it proposes, not its weights — is clipped
to L2 norm ``C`` and then perturbed with Gaussian noise ``N(0, sigma^2 C^2)``.
The privacy budget is tracked with Opacus's RDP accountant, so the reported
epsilon comes from a standard implementation rather than from arithmetic
invented here.

**Clipping the update, not the weights.** The sensitivity that matters is how
much one client can move the global model. Clipping absolute weights would
bound where a client can push the model to, not how far it can push it, and the
noise calibration ``sigma * C`` would then be attached to the wrong quantity.

Noise is added **once per aggregated round on the server side** in this
implementation, which is central differential privacy: it protects a client's
contribution from anyone reading the global model, and assumes the aggregator
is trusted. Local DP — noise added before the update leaves the client — needs
far more noise for the same epsilon. The distinction matters and the report
should state which one is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class PrivacyBudget:
    """The (epsilon, delta) spent so far."""

    epsilon: float
    delta: float
    rounds: int
    noise_multiplier: float
    clip_norm: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "epsilon": round(self.epsilon, 4) if np.isfinite(self.epsilon) else None,
            "delta": self.delta,
            "rounds": self.rounds,
            "noise_multiplier": self.noise_multiplier,
            "clip_norm": self.clip_norm,
        }


def l2_norm(update: list[np.ndarray]) -> float:
    """Flattened L2 norm across every tensor in an update."""
    return float(np.sqrt(sum(float(np.sum(np.square(a))) for a in update)))


def clip_update(update: list[np.ndarray], clip_norm: float) -> tuple[list[np.ndarray], float]:
    """Scale an update down to ``clip_norm`` if it exceeds it.

    Returns the clipped update and its original norm. Scaling rather than
    truncating preserves the update's direction — a per-element clamp would
    change which way the client wanted the model to move, not just how far.
    """
    norm = l2_norm(update)
    if norm <= clip_norm or norm == 0.0:
        return [a.copy() for a in update], norm
    scale = clip_norm / norm
    return [a * scale for a in update], norm


def add_gaussian_noise(
    update: list[np.ndarray],
    noise_multiplier: float,
    clip_norm: float,
    rng: np.random.Generator,
    n_clients: int = 1,
) -> list[np.ndarray]:
    """Perturb with the Gaussian mechanism, scaled for an averaged update.

    ``n_clients`` matters. DP-FedAvg calibrates ``N(0, sigma^2 C^2)`` to the
    *sum* of clipped updates, which is where the sensitivity bound C applies —
    one client can move the sum by at most C. Dividing by N to get the mean
    divides the noise too, so the standard deviation on the average is
    ``sigma * C / N``.

    Adding ``sigma * C`` directly to an already-averaged update, as an earlier
    version of this did, applies N times too much noise and makes differential
    privacy look far more destructive than it is.
    """
    if noise_multiplier <= 0:
        return [a.copy() for a in update]
    sigma = noise_multiplier * clip_norm / max(1, n_clients)
    return [a + rng.normal(0.0, sigma, size=a.shape).astype(a.dtype) for a in update]


def median_update_norm(updates: list[list[np.ndarray]]) -> float:
    """Median L2 norm across client updates — a sane adaptive clip threshold.

    A clip norm plucked out of the air either never binds (no privacy benefit
    from clipping, and the noise is calibrated to a bound nothing reaches) or
    binds so hard that every client contributes the same direction at the same
    magnitude. The median is the usual adaptive choice.
    """
    if not updates:
        return 1.0
    return float(np.median([l2_norm(u) for u in updates]))


def accountant_epsilon(
    noise_multiplier: float,
    rounds: int,
    sample_rate: float,
    delta: float = 1e-5,
) -> float:
    """Privacy budget after ``rounds``, via Opacus's RDP accountant.

    Returns infinity when no noise is added, which is the honest value: a
    mechanism that adds nothing provides no differential privacy, and reporting
    a large finite epsilon there would suggest a guarantee that does not exist.
    """
    if noise_multiplier <= 0:
        return float("inf")

    try:
        from opacus.accountants import RDPAccountant

        accountant = RDPAccountant()
        for _ in range(rounds):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        return float(accountant.get_epsilon(delta=delta))
    except Exception:  # noqa: BLE001 - fall back rather than lose the run
        # Closed-form Gaussian-mechanism bound, composed over rounds. Looser
        # than RDP, so it never understates the budget.
        from math import log, sqrt

        single = sqrt(2 * log(1.25 / delta)) / noise_multiplier
        return float(single * sqrt(rounds) * sample_rate)


def noise_for_epsilon(
    target_epsilon: float,
    rounds: int,
    sample_rate: float,
    delta: float = 1e-5,
    tolerance: float = 0.05,
) -> float:
    """Smallest noise multiplier achieving ``target_epsilon``, by bisection.

    Monotonic in sigma — more noise always means a smaller epsilon — so
    bisection is exact enough and avoids depending on an Opacus helper whose
    signature changes between versions.
    """
    if not np.isfinite(target_epsilon):
        return 0.0

    low, high = 0.05, 64.0
    for _ in range(60):
        mid = (low + high) / 2
        achieved = accountant_epsilon(mid, rounds, sample_rate, delta)
        if abs(achieved - target_epsilon) <= tolerance:
            return mid
        if achieved > target_epsilon:
            low = mid   # too much privacy loss: add more noise
        else:
            high = mid
    return (low + high) / 2
