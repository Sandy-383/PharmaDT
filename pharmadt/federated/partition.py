"""Split training data across simulated clients, IID and non-IID.

The IID-versus-non-IID gap is one of the strongest experimental results
available here, so both splits are produced from the same pool and differ only
in how they are drawn.

**Non-IID via Dirichlet(alpha=0.5).** Each client's share of each source series
is drawn from a Dirichlet, which produces the volume and pattern skew real
pharmacies exhibit: one client ends up dominated by two or three stores, another
sees a broad mix. Splitting uniformly instead would make federated learning look
easy for a reason that does not hold in deployment — with identical
distributions everywhere, FedAvg converges almost as if the data were pooled,
and the whole difficulty the method exists to address disappears.

Lower alpha means more skew. 0.5 is the guide's figure and the usual choice in
the federated literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class ClientShard:
    """One client's private data. This never leaves the client (NFR-03)."""

    client_id: str
    X: np.ndarray
    y: np.ndarray
    source_series: list[int]

    def __len__(self) -> int:
        return len(self.X)

    def summary(self) -> dict[str, Any]:
        """Non-sensitive description, safe to log or transmit."""
        return {
            "client_id": self.client_id,
            "n_samples": len(self.X),
            "n_series": len(self.source_series),
        }


def iid_partition(
    X: np.ndarray, y: np.ndarray, n_clients: int, seed: int = 42
) -> list[ClientShard]:
    """Shuffle and deal evenly. The optimistic baseline."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    chunks = np.array_split(order, n_clients)

    return [
        ClientShard(f"client-{i}", X[idx], y[idx], sorted(int(v) for v in idx[:5]))
        for i, idx in enumerate(chunks)
    ]


def dirichlet_partition(
    X: np.ndarray,
    y: np.ndarray,
    series_ids: np.ndarray,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples: int = 20,
) -> list[ClientShard]:
    """Skewed split: each series is divided across clients by a Dirichlet draw.

    Partitioning by *series* rather than by row is what makes the skew
    realistic. Real pharmacies differ in which products move, not in which
    random subset of days they observed, and a row-wise Dirichlet would produce
    clients that are statistically identical however uneven their sizes.
    """
    rng = np.random.default_rng(seed)
    series_ids = np.asarray(series_ids)
    buckets: list[list[int]] = [[] for _ in range(n_clients)]

    for series in np.unique(series_ids):
        rows = np.flatnonzero(series_ids == series)
        rng.shuffle(rows)
        proportions = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(proportions) * len(rows)).astype(int)[:-1]
        for client, chunk in enumerate(np.split(rows, cuts)):
            buckets[client].extend(int(r) for r in chunk)

    # A client with almost nothing contributes noise to the average and makes
    # the comparison about sampling luck rather than about heterogeneity.
    donor = max(range(n_clients), key=lambda c: len(buckets[c]))
    for client in range(n_clients):
        while len(buckets[client]) < min_samples and len(buckets[donor]) > min_samples * 2:
            buckets[client].append(buckets[donor].pop())

    shards = []
    for client, rows in enumerate(buckets):
        index = np.array(sorted(rows), dtype=int)
        shards.append(
            ClientShard(
                f"client-{client}",
                X[index],
                y[index],
                sorted({int(s) for s in series_ids[index]}),
            )
        )
    return shards


def skew_report(shards: list[ClientShard]) -> dict[str, Any]:
    """Quantify how uneven a split actually is, so the label can be checked."""
    sizes = np.array([len(s) for s in shards], dtype=float)
    share = sizes / max(sizes.sum(), 1)
    # Gini: 0 is a perfectly even split, 1 is one client holding everything.
    sorted_share = np.sort(share)
    n = len(sorted_share)
    gini = float(
        (2 * np.sum((np.arange(1, n + 1)) * sorted_share) - (n + 1) * sorted_share.sum())
        / (n * sorted_share.sum())
    ) if sorted_share.sum() else 0.0

    return {
        "n_clients": len(shards),
        "sizes": [int(v) for v in sizes],
        "smallest": int(sizes.min()) if len(sizes) else 0,
        "largest": int(sizes.max()) if len(sizes) else 0,
        "gini": round(gini, 4),
        "series_per_client": [len(s.source_series) for s in shards],
    }
