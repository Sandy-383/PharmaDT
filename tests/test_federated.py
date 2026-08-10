"""Federated learning: partitioning, FedAvg, privacy, and NFR-03.

The load-bearing test is ``test_raw_data_never_leaves_the_client``. NFR-03 is
the claim that makes federated learning worth doing at all, and it is checked
by inspecting every array that crosses the client boundary rather than by
trusting the design.
"""

from __future__ import annotations

import numpy as np
import pytest

from pharmadt.federated.client import ClientUpdate, PharmaClient
from pharmadt.federated.partition import (
    ClientShard,
    dirichlet_partition,
    iid_partition,
    skew_report,
)
from pharmadt.federated.privacy import (
    accountant_epsilon,
    add_gaussian_noise,
    clip_update,
    l2_norm,
    median_update_norm,
    noise_for_epsilon,
)
from pharmadt.federated.server import federated_average, run_federated


class TinyModel:
    """Two-parameter stand-in for the LSTM: fast, and exercises the same contract."""

    def __init__(self) -> None:
        self.w = [np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32)]
        self.epochs = 1

    def get_weights(self) -> list[np.ndarray]:
        return [a.copy() for a in self.w]

    def set_weights(self, w: list[np.ndarray]) -> None:
        self.w = [np.asarray(a, dtype=np.float32).copy() for a in w]

    def fit(self, X, y=None):
        # Move toward the shard's mean so different data gives different weights.
        target = float(np.mean(y)) if y is not None and len(y) else 0.0
        self.w = [a + 0.1 * (target - a) for a in self.w]
        return self

    def predict(self, X, horizon: int = 14):
        return np.full((len(X), horizon), float(self.w[0].mean()), dtype=np.float32)


def shard(name="c0", n=40, value=1.0) -> ClientShard:
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    X = rng.normal(value, 0.5, size=(n, 28, 1)).astype(np.float32)
    y = rng.normal(value, 0.5, size=(n, 14)).astype(np.float32)
    return ClientShard(name, X, y, [0])


# ── NFR-03: data locality ─────────────────────────────────────────────


def test_raw_data_never_leaves_the_client() -> None:
    """NFR-03. The reason federated learning is worth the accuracy cost.

    Every array in the update is compared against the client's actual training
    data. A shard that leaked would show up here as a shape and content match.
    """
    private = shard("secret", n=50, value=7.0)
    client = PharmaClient(private, TinyModel)
    update = client.fit(TinyModel().get_weights())

    import dataclasses

    # The response is a closed structure. Adding raw data to it would require
    # editing this class, which a reviewer sees, rather than quietly attaching
    # another key to a free-form dict.
    assert isinstance(update, ClientUpdate)
    assert {f.name for f in dataclasses.fields(update)} == {
        "client_id", "weights", "n_samples", "metrics"
    }

    for array in update.weights:
        assert array.shape != private.X.shape
        assert array.shape != private.y.shape
        assert array.size < private.X.size
        # And nothing that merely reshapes to the data either.
        assert not np.array_equal(array.ravel()[: private.y.size], private.y.ravel())

    # Metrics carry counts, never values.
    for value in update.metrics.values():
        assert isinstance(value, float)
    assert update.n_samples == 50


def test_the_shard_is_private_to_the_client() -> None:
    """No public attribute exposes the training data."""
    client = PharmaClient(shard(), TinyModel)
    public = [a for a in dir(client) if not a.startswith("_")]
    for name in public:
        value = getattr(client, name, None)
        assert not isinstance(value, ClientShard)


def test_a_shard_summary_carries_no_values() -> None:
    summary = shard("c1", n=12).summary()
    assert set(summary) == {"client_id", "n_samples", "n_series"}


# ── Partitioning ──────────────────────────────────────────────────────


@pytest.fixture
def pool():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 28, 1)).astype(np.float32)
    y = rng.normal(size=(600, 14)).astype(np.float32)
    series = np.repeat(np.arange(20), 30)
    return X, y, series


def test_an_iid_split_is_even_and_complete(pool) -> None:
    X, y, _ = pool
    shards = iid_partition(X, y, 5, seed=1)
    assert len(shards) == 5
    assert sum(len(s) for s in shards) == len(X)
    assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 1


def test_a_dirichlet_split_is_measurably_more_skewed(pool) -> None:
    """The point of alpha=0.5: real clients differ in volume and mix."""
    X, y, series = pool
    even = skew_report(iid_partition(X, y, 5, seed=1))
    skewed = skew_report(dirichlet_partition(X, y, series, 5, alpha=0.5, seed=1))
    assert skewed["gini"] > even["gini"]


def test_a_lower_alpha_skews_harder(pool) -> None:
    X, y, series = pool
    mild = skew_report(dirichlet_partition(X, y, series, 5, alpha=5.0, seed=1))
    harsh = skew_report(dirichlet_partition(X, y, series, 5, alpha=0.1, seed=1))
    assert harsh["gini"] > mild["gini"]


def test_no_client_is_left_with_almost_nothing(pool) -> None:
    """A near-empty client contributes noise, not heterogeneity."""
    X, y, series = pool
    shards = dirichlet_partition(X, y, series, 5, alpha=0.1, seed=3, min_samples=20)
    assert min(len(s) for s in shards) >= 20


# ── FedAvg ────────────────────────────────────────────────────────────


def test_the_average_is_weighted_by_sample_count() -> None:
    """Unweighted, a 20-window client would outvote a 2,000-window one."""
    a = ClientUpdate("a", [np.array([0.0])], n_samples=10)
    b = ClientUpdate("b", [np.array([10.0])], n_samples=90)
    assert federated_average([a, b])[0][0] == pytest.approx(9.0)


def test_equal_clients_average_evenly() -> None:
    a = ClientUpdate("a", [np.array([0.0, 4.0])], n_samples=50)
    b = ClientUpdate("b", [np.array([2.0, 0.0])], n_samples=50)
    assert federated_average([a, b])[0].tolist() == [1.0, 2.0]


def test_aggregating_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="zero client updates"):
        federated_average([])


def test_clients_disagreeing_on_architecture_is_an_error() -> None:
    """Positional averaging would otherwise silently corrupt the global model."""
    a = ClientUpdate("a", [np.zeros(2)], n_samples=1)
    b = ClientUpdate("b", [np.zeros(2), np.zeros(2)], n_samples=1)
    with pytest.raises(ValueError, match="architecture"):
        federated_average([a, b])


def test_a_federated_run_produces_a_model_from_every_client() -> None:
    clients = [PharmaClient(shard(f"c{i}", value=float(i)), TinyModel) for i in range(5)]
    result = run_federated(clients, TinyModel, rounds=5, seed=1)

    assert len(result.rounds) == 5
    assert len(result.client_summaries) == 5
    assert result.final_weights


def test_federated_training_needs_at_least_one_client() -> None:
    with pytest.raises(ValueError, match="at least one client"):
        run_federated([], TinyModel, rounds=1)


# ── Privacy ───────────────────────────────────────────────────────────


def test_clipping_preserves_direction_and_bounds_magnitude() -> None:
    """A per-element clamp would change which way the client wanted to move."""
    update = [np.array([3.0, 4.0])]          # norm 5
    clipped, original = clip_update(update, 1.0)

    assert original == pytest.approx(5.0)
    assert l2_norm(clipped) == pytest.approx(1.0)
    assert clipped[0][1] / clipped[0][0] == pytest.approx(4 / 3)


def test_an_update_within_the_bound_is_untouched() -> None:
    update = [np.array([0.3, 0.4])]
    clipped, original = clip_update(update, 1.0)
    assert original == pytest.approx(0.5)
    assert clipped[0].tolist() == pytest.approx([0.3, 0.4])


def test_noise_shrinks_with_more_clients() -> None:
    """DP-FedAvg calibrates noise to the sum; the mean divides it by N."""
    rng = np.random.default_rng(0)
    base = [np.zeros(20_000, dtype=np.float64)]

    few = add_gaussian_noise(base, 1.0, 1.0, np.random.default_rng(0), n_clients=1)
    many = add_gaussian_noise(base, 1.0, 1.0, rng, n_clients=10)
    assert np.std(many[0]) < np.std(few[0]) / 5


def test_no_noise_means_no_privacy_guarantee() -> None:
    """Reporting a finite epsilon here would imply a guarantee that is absent."""
    assert accountant_epsilon(0.0, rounds=10, sample_rate=1.0) == float("inf")
    assert np.isfinite(accountant_epsilon(1.0, rounds=10, sample_rate=1.0))


def test_more_noise_buys_a_smaller_epsilon() -> None:
    quiet = accountant_epsilon(0.5, rounds=20, sample_rate=1.0)
    loud = accountant_epsilon(4.0, rounds=20, sample_rate=1.0)
    assert loud < quiet


def test_more_rounds_spend_more_budget() -> None:
    assert accountant_epsilon(1.0, 50, 1.0) > accountant_epsilon(1.0, 10, 1.0)


@pytest.mark.parametrize("target", [1.0, 5.0, 10.0])
def test_the_solver_hits_the_requested_epsilon(target: float) -> None:
    sigma = noise_for_epsilon(target, rounds=30, sample_rate=1.0)
    achieved = accountant_epsilon(sigma, rounds=30, sample_rate=1.0)
    assert achieved == pytest.approx(target, rel=0.1)


def test_an_infinite_target_needs_no_noise() -> None:
    assert noise_for_epsilon(float("inf"), rounds=30, sample_rate=1.0) == 0.0


def test_the_adaptive_clip_norm_tracks_the_updates() -> None:
    """A blind clip norm either never binds or flattens every client."""
    updates = [[np.array([0.0, 0.6])], [np.array([0.8, 0.0])], [np.array([0.0, 1.0])]]
    assert median_update_norm(updates) == pytest.approx(0.8)


def test_a_differentially_private_run_reports_its_budget() -> None:
    clients = [PharmaClient(shard(f"c{i}"), TinyModel) for i in range(3)]
    result = run_federated(
        clients, TinyModel, rounds=3, noise_multiplier=1.0, clip_norm=None, seed=1
    )
    assert result.privacy is not None
    assert np.isfinite(result.privacy.epsilon)
    assert result.privacy.clip_norm > 0
