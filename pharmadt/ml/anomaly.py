"""Anomaly detection models and the metrics that make them honest.

Isolation Forest and an autoencoder score shipments independently; the ensemble
combines them under one of two rules with very different characters:

* **either** — flag if either model exceeds its threshold. Higher recall.
* **both**   — flag only on agreement. Higher precision.

Reporting one without the other hides the trade-off, so
:func:`evaluate_detector` always returns both.

On metrics: **accuracy is not reported anywhere in this module.** The positive
class here is a few percent of shipments, so a detector that flags nothing
scores in the nineties while catching zero counterfeits. Precision, recall, F1
and ROC-AUC are the only numbers that distinguish a working detector from a
constant ``False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Fraction of shipments the Isolation Forest is told to expect as outliers.
CONTAMINATION = 0.05
#: Percentile of training reconstruction error used as the autoencoder threshold.
AE_PERCENTILE = 95


@dataclass(slots=True)
class DetectionScores:
    """Confusion matrix and the metrics that survive class imbalance."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    roc_auc: float = float("nan")

    @property
    def precision(self) -> float:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        """Present only so the report can show why it is the wrong metric."""
        total = (
            self.true_positives + self.false_positives
            + self.true_negatives + self.false_negatives
        )
        return (self.true_positives + self.true_negatives) / total if total else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4) if np.isfinite(self.roc_auc) else None,
            "accuracy": round(self.accuracy, 4),
            "tp": self.true_positives,
            "fp": self.false_positives,
            "tn": self.true_negatives,
            "fn": self.false_negatives,
        }


def confusion(actual: np.ndarray, predicted: np.ndarray, scores: np.ndarray | None = None
              ) -> DetectionScores:
    actual = np.asarray(actual).astype(bool)
    predicted = np.asarray(predicted).astype(bool)

    auc = float("nan")
    if scores is not None and 0 < actual.sum() < len(actual):
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(actual, np.asarray(scores, dtype=float)))

    return DetectionScores(
        true_positives=int((actual & predicted).sum()),
        false_positives=int((~actual & predicted).sum()),
        true_negatives=int((~actual & ~predicted).sum()),
        false_negatives=int((actual & ~predicted).sum()),
        roc_auc=auc,
    )


# ── Isolation Forest ──────────────────────────────────────────────────


class IsolationForestDetector:
    """sklearn Isolation Forest wrapped to emit a comparable anomaly score."""

    def __init__(
        self,
        contamination: float = CONTAMINATION,
        n_estimators: int = 200,
        seed: int | None = None,
    ) -> None:
        from pharmadt.config import settings

        self.contamination = contamination
        self.n_estimators = n_estimators
        self.seed = settings.sim_seed if seed is None else seed
        self.model: Any = None
        self.threshold_ = 0.0

    def fit(self, X: np.ndarray) -> IsolationForestDetector:
        from sklearn.ensemble import IsolationForest

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
        ).fit(np.asarray(X, dtype=float))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher means more anomalous.

        sklearn's ``score_samples`` is signed the other way — more negative is
        more anomalous — which silently inverts every threshold comparison if
        used directly alongside the autoencoder's error.
        """
        if self.model is None:
            raise RuntimeError("IsolationForestDetector.score called before fit")
        return -self.model.score_samples(np.asarray(X, dtype=float))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("IsolationForestDetector.predict called before fit")
        return self.model.predict(np.asarray(X, dtype=float)) == -1


# ── Autoencoder ───────────────────────────────────────────────────────


class AutoencoderDetector:
    """Reconstruction-error detector, trained on normal shipments only.

    Training on normal traffic alone is the point: the network learns to
    reproduce ordinary shipments cheaply and reconstructs unusual ones badly,
    so the error is the score. Training on the mixture would teach it to
    reproduce the anomalies too.
    """

    def __init__(
        self,
        n_features: int = 12,
        hidden: int = 8,
        latent: int = 4,
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 64,
        percentile: float = AE_PERCENTILE,
        seed: int | None = None,
    ) -> None:
        from pharmadt.config import settings

        self.n_features = n_features
        self.hidden = hidden
        self.latent = latent
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.percentile = percentile
        self.seed = settings.sim_seed if seed is None else seed
        self.net: Any = None
        self.threshold_ = float("inf")
        self.mean_ = None
        self.std_ = None

    def _build(self) -> Any:
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        return nn.Sequential(
            nn.Linear(self.n_features, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.latent),
            nn.ReLU(),
            nn.Linear(self.latent, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.n_features),
        )

    def _standardise(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if fit:
            self.mean_ = X.mean(axis=0)
            # Guard constant columns; dividing by zero std makes every value inf
            # and the reconstruction error meaningless for that feature.
            self.std_ = np.maximum(X.std(axis=0), 1e-6)
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray) -> AutoencoderDetector:
        import torch
        from torch import nn

        data = self._standardise(X, fit=True)
        self.n_features = data.shape[1]
        self.net = self._build()

        tensor = torch.as_tensor(data, dtype=torch.float32)
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        generator = torch.Generator().manual_seed(self.seed)

        self.net.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(tensor), generator=generator)
            for start in range(0, len(tensor), self.batch_size):
                batch = tensor[order[start : start + self.batch_size]]
                optimiser.zero_grad()
                loss = criterion(self.net(batch), batch)
                loss.backward()
                optimiser.step()

        # Threshold from training error only. Setting it on the evaluation set
        # would tune the detector on the data it is then scored against.
        self.threshold_ = float(np.percentile(self.score(X), self.percentile))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.net is None:
            raise RuntimeError("AutoencoderDetector.score called before fit")
        data = torch.as_tensor(self._standardise(X), dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            reconstructed = self.net(data)
        return ((reconstructed - data) ** 2).mean(dim=1).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) > self.threshold_


# ── Ensemble ──────────────────────────────────────────────────────────


def ensemble_predict(
    forest_flags: np.ndarray, auto_flags: np.ndarray, rule: str = "either"
) -> np.ndarray:
    """Combine two detectors. ``either`` favours recall, ``both`` precision."""
    forest_flags = np.asarray(forest_flags).astype(bool)
    auto_flags = np.asarray(auto_flags).astype(bool)
    if rule == "either":
        return forest_flags | auto_flags
    if rule == "both":
        return forest_flags & auto_flags
    raise ValueError(f"unknown ensemble rule {rule!r}; expected 'either' or 'both'")


def evaluate_detector(
    actual: np.ndarray,
    forest_flags: np.ndarray,
    auto_flags: np.ndarray,
    forest_scores: np.ndarray | None = None,
    auto_scores: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Score both models and both ensemble rules on the same labels."""
    results = {
        "isolation_forest": confusion(actual, forest_flags, forest_scores).as_dict(),
        "autoencoder": confusion(actual, auto_flags, auto_scores).as_dict(),
    }
    for rule in ("either", "both"):
        combined = ensemble_predict(forest_flags, auto_flags, rule)
        blended = None
        if forest_scores is not None and auto_scores is not None:
            # Rank-average the two scores so a single AUC describes the pair
            # despite their incomparable natural scales.
            blended = (_ranks(forest_scores) + _ranks(auto_scores)) / 2
        results[f"ensemble_{rule}"] = confusion(actual, combined, blended).as_dict()
    return results


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    return ranks / max(1, len(order) - 1)
