"""The federated client: trains locally, returns weights, never returns data.

``PharmaClient`` is the transport-independent core. ``FlowerClient`` wraps it as
a ``flwr.client.NumPyClient`` so the identical logic runs under a real Flower
server; the in-process server in :mod:`pharmadt.federated.server` drives
``PharmaClient`` directly.

**NFR-03 is a structural property here, not a promise.** :meth:`PharmaClient.fit`
returns a :class:`ClientUpdate` whose only fields are weight arrays, a sample
count and scalar metrics. The shard is held privately and there is no code path
that puts ``X`` or ``y`` into a return value — which is what
``test_federated.py`` asserts by inspecting every array that crosses the
boundary and confirming none matches the training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pharmadt.federated.partition import ClientShard


@dataclass(slots=True)
class ClientUpdate:
    """Everything a client is permitted to send back.

    Deliberately a closed structure rather than a free-form dict: adding raw
    data to a response would require changing this class, which a reviewer
    would see, instead of quietly attaching another key.
    """

    client_id: str
    weights: list[np.ndarray]
    n_samples: int
    metrics: dict[str, float] = field(default_factory=dict)


class PharmaClient:
    """One simulated pharmacy or hospital training on its own data."""

    def __init__(
        self,
        shard: ClientShard,
        model_factory: Any,
        local_epochs: int = 1,
        proximal_mu: float = 0.0,
    ) -> None:
        self._shard = shard          # private; never returned
        self.client_id = shard.client_id
        self.model = model_factory()
        self.local_epochs = local_epochs
        # FedProx proximal term. Zero means plain FedAvg. Under heavy non-IID
        # skew this pulls each client back toward the global model and stops
        # local optima dragging the average apart.
        self.proximal_mu = proximal_mu

    @property
    def n_samples(self) -> int:
        return len(self._shard)

    def get_parameters(self) -> list[np.ndarray]:
        return self.model.get_weights()

    def set_parameters(self, weights: list[np.ndarray]) -> None:
        self.model.set_weights(weights)

    def fit(self, global_weights: list[np.ndarray]) -> ClientUpdate:
        """Train locally from the global model and return the new weights."""
        self.set_parameters(global_weights)
        self.model.epochs = self.local_epochs
        self.model.fit(self._shard.X, self._shard.y)

        weights = self.model.get_weights()
        if self.proximal_mu > 0:
            weights = [
                w - self.proximal_mu * (w - g)
                for w, g in zip(weights, global_weights, strict=True)
            ]

        return ClientUpdate(
            client_id=self.client_id,
            weights=weights,
            n_samples=self.n_samples,
            metrics={"train_samples": float(self.n_samples)},
        )

    def evaluate(self, global_weights: list[np.ndarray]) -> dict[str, float]:
        """Score the global model on this client's own held-out data."""
        from pharmadt.ml.forecasting import evaluate, seasonal_naive_scale

        self.set_parameters(global_weights)
        predicted = self.model.predict(self._shard.X)
        scale = seasonal_naive_scale(self._shard.y, self._shard.X[..., 0])
        scores = evaluate(self._shard.y, predicted, scale)
        return {
            "sMAPE": scores["sMAPE"],
            "MAPE": scores["MAPE"],
            "MASE": scores["MASE"],
            "n_samples": float(self.n_samples),
        }


class FlowerClient:
    """``flwr.client.NumPyClient`` adapter around :class:`PharmaClient`.

    Present so the same clients run under a real Flower server. The experiments
    drive ``PharmaClient`` in-process instead: Flower's simulation engine needs
    Ray, which pulls a protobuf version incompatible with the OR-Tools pin this
    project already had to resolve in Stage 0. The FedAvg arithmetic is
    identical either way and is tested directly.
    """

    def __init__(self, client: PharmaClient) -> None:
        self.client = client

    def get_parameters(self, config: dict | None = None) -> list[np.ndarray]:
        return self.client.get_parameters()

    def fit(self, parameters: list[np.ndarray], config: dict | None = None):
        update = self.client.fit(parameters)
        return update.weights, update.n_samples, update.metrics

    def evaluate(self, parameters: list[np.ndarray], config: dict | None = None):
        metrics = self.client.evaluate(parameters)
        return float(metrics["sMAPE"]), self.client.n_samples, metrics

    def to_flower(self) -> Any:
        """Return a genuine ``flwr`` NumPyClient bound to this adapter."""
        import flwr as fl

        adapter = self

        class _Client(fl.client.NumPyClient):
            def get_parameters(self, config):
                return adapter.get_parameters(config)

            def fit(self, parameters, config):
                return adapter.fit(parameters, config)

            def evaluate(self, parameters, config):
                return adapter.evaluate(parameters, config)

        return _Client()
