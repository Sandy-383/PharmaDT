"""Inject known anomalies so detection can be measured against ground truth.

Without injection there are no labels, and without labels a detector can only
be admired rather than evaluated. Each injected shipment is tagged, so
precision and recall are computed against something known rather than against
the detector's own opinion.

Four anomaly kinds, chosen because each is a different *shape* of outlier:

* ``QUANTITY``   — a shipment far larger than that route ever carries.
* ``TRANSIT``    — a delivery that takes much longer than the lane allows.
* ``COLD_CHAIN`` — repeated temperature excursions on one shipment.
* ``COUNTERFEIT``— a batch whose fingerprint does not match its own fields.

The last one matters most. It is invisible in the shipment features by
construction: a counterfeit batch moves normally, in normal quantities, over a
normal route. **Only the ledger cross-check catches it**, which is precisely
the point of research gap 3.2.3 — the ledger as a live input to a detection
system rather than an isolated record store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class AnomalyKind(StrEnum):
    QUANTITY = "QUANTITY"
    TRANSIT = "TRANSIT"
    COLD_CHAIN = "COLD_CHAIN"
    COUNTERFEIT = "COUNTERFEIT"


#: Kinds the shipment features can express. COUNTERFEIT is deliberately absent.
FEATURE_VISIBLE = (AnomalyKind.QUANTITY, AnomalyKind.TRANSIT, AnomalyKind.COLD_CHAIN)


@dataclass(slots=True)
class InjectionReport:
    """What was injected, so the evaluation knows the truth."""

    labels: np.ndarray
    kinds: list[str] = field(default_factory=list)
    counterfeit_batches: list[str] = field(default_factory=list)

    @property
    def n_injected(self) -> int:
        return int(self.labels.sum())

    def kind_mask(self, kind: AnomalyKind) -> np.ndarray:
        return np.array([k == kind for k in self.kinds], dtype=bool)


def inject_anomalies(
    shipments: list[dict[str, Any]],
    rate: float = 0.05,
    seed: int = 42,
    kinds: tuple[AnomalyKind, ...] = tuple(AnomalyKind),
) -> InjectionReport:
    """Corrupt a labelled fraction of ``shipments`` in place.

    Perturbations are multiplicative and drawn from a range, not set to one
    fixed extreme value. A constant marker would be trivially separable and the
    reported recall would measure nothing but the marker.
    """
    if not shipments:
        return InjectionReport(np.zeros(0, dtype=bool))

    rng = np.random.default_rng(seed)
    n = len(shipments)
    labels = np.zeros(n, dtype=bool)
    assigned = ["" for _ in range(n)]
    counterfeits: list[str] = []

    n_inject = max(1, int(n * rate))
    chosen = rng.choice(n, size=min(n_inject, n), replace=False)

    for index in chosen:
        kind = AnomalyKind(rng.choice([str(k) for k in kinds]))
        shipment = shipments[int(index)]
        labels[index] = True
        assigned[index] = str(kind)
        shipment["anomaly_kind"] = str(kind)

        if kind is AnomalyKind.QUANTITY:
            shipment["quantity"] = int(shipment["quantity"] * rng.uniform(4.0, 12.0))
        elif kind is AnomalyKind.TRANSIT:
            shipment["transit_days"] = float(
                shipment["transit_days"] * rng.uniform(3.0, 8.0)
            )
        elif kind is AnomalyKind.COLD_CHAIN:
            shipment["excursion_count"] = int(rng.integers(3, 9))
            shipment["excursion_severity"] = float(rng.uniform(4.0, 12.0))
        elif kind is AnomalyKind.COUNTERFEIT:
            # Nothing about the shipment changes. That is the whole point: the
            # ML features cannot see this, and only the ledger can.
            batch_id = shipment.get("batch_id")
            if batch_id:
                counterfeits.append(batch_id)
            shipment["forged_fingerprint"] = True

    return InjectionReport(labels, assigned, counterfeits)
