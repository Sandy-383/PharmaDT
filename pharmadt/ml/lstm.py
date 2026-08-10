"""LSTM demand forecaster (PyTorch), implementing the DemandModel contract.

Architecture per the guide: 2 LSTM layers, hidden 64, dropout 0.2, linear head
emitting all 14 horizons at once.

**Direct multi-horizon, not recursive.** Feeding a prediction back in to get the
next day compounds its own error, so by day 14 a recursive model is mostly
forecasting its earlier mistakes. One head with 14 outputs costs nothing extra
and has no such feedback path.

Trained on ``log1p`` demand. Retail demand has a long right tail, and MSE on raw
counts would let a handful of busy days dominate the gradient while the ordinary
days the twin actually runs on get ignored.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pharmadt.config import settings
from pharmadt.core.interfaces import DemandModel
from pharmadt.ml.forecasting import HORIZON, LOOKBACK


class _Net(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float,
                 horizon: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            # PyTorch ignores dropout on a single layer and warns; guard it.
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])  # last timestep only


class LSTMForecaster(DemandModel):
    """Global forecaster shared across series.

    One model over all series rather than one per series: it learns patterns
    common to every pharmacy, works for a series with little history, and —
    decisively for Stage 11 — gives federated learning a single parameter
    vector to average. Per-series models cannot be FedAvg'd at all.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        horizon: int = HORIZON,
        lookback: int = LOOKBACK,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 256,
        patience: int = 4,
        seed: int | None = None,
        device: str = "cpu",
        scale_by_window: bool = True,
        demand_channels: int = 1,
    ) -> None:
        # Leading channels measured in demand units, which the window scaling
        # applies to. Calendar and flag channels (day-of-week sin/cos, promo,
        # holiday) are already bounded and must NOT be divided by a demand
        # mean -- doing so would make "was there a promotion" mean something
        # different at a busy shop than at a quiet one.
        self.demand_channels = demand_channels
        # Per-window scaling. One global model spans stores whose daily volumes
        # differ severalfold, so on raw values most of its capacity goes on
        # inferring each series' level from its window instead of learning
        # shape. Dividing by the window mean makes every sample the same
        # question -- "what is next relative to recent?" -- and puts the loss in
        # the same relative units the sMAPE scoreboard uses.
        self.scale_by_window = scale_by_window
        self.n_features = n_features
        self.horizon = horizon
        self.lookback = lookback
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device(device)
        self.seed = settings.sim_seed if seed is None else seed

        torch.manual_seed(self.seed)
        self.net = _Net(n_features, hidden, layers, dropout, horizon).to(self.device)
        self.history_: list[dict[str, float]] = []

    # ── Training ──────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = False,
    ) -> LSTMForecaster:
        """Train with early stopping on validation loss.

        Early stopping restores the best weights rather than keeping the last
        epoch's. Without that, stopping "early" still ships whichever weights
        happened to be current when patience ran out — usually the overfitted ones.
        """
        torch.manual_seed(self.seed)
        X_scaled, y_scaled, _ = self._rescale(X, y)
        inputs = self._features(X_scaled)
        targets = self._targets(y_scaled)

        has_validation = X_val is not None and y_val is not None and len(X_val) > 0
        if has_validation:
            X_val_scaled, y_val_scaled, _ = self._rescale(X_val, y_val)
            val_inputs = self._features(X_val_scaled)
            val_targets = self._targets(y_val_scaled)

        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_loss, best_state, waited = float("inf"), None, 0
        generator = torch.Generator().manual_seed(self.seed)

        for epoch in range(self.epochs):
            self.net.train()
            permutation = torch.randperm(len(inputs), generator=generator)
            running = 0.0

            for start in range(0, len(inputs), self.batch_size):
                batch = permutation[start : start + self.batch_size]
                optimiser.zero_grad()
                loss = criterion(self.net(inputs[batch]), targets[batch])
                loss.backward()
                # Recurrent nets are prone to exploding gradients; without this
                # a single bad batch can wreck an otherwise converged model.
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimiser.step()
                running += loss.item() * len(batch)

            train_loss = running / len(inputs)
            if has_validation:
                self.net.eval()
                with torch.no_grad():
                    monitored = criterion(self.net(val_inputs), val_targets).item()
            else:
                monitored = train_loss

            self.history_.append({"epoch": epoch, "train": train_loss, "val": monitored})
            if verbose:
                print(f"  epoch {epoch:>3}  train {train_loss:.5f}  val {monitored:.5f}")

            if monitored < best_loss - 1e-5:
                best_loss, waited = monitored, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    # ── Inference ─────────────────────────────────────────────────────

    def predict(self, X: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
        """Forecast in demand units, undoing whatever scaling training used."""
        self.net.eval()
        X_scaled, _, scale = self._rescale(X, None)
        inputs = self._features(X_scaled)
        with torch.no_grad():
            predicted = self.net(inputs).cpu().numpy()

        forecast = predicted * scale if self.scale_by_window else np.expm1(predicted)
        # Negative demand is not a thing.
        forecast = np.clip(forecast, 0.0, None)
        return forecast[:, :horizon] if horizon <= forecast.shape[1] else forecast

    def _rescale(
        self, X: np.ndarray, y: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Normalise each window by its own mean; return the scale to undo it.

        The scale comes from the *input* window only. Deriving it from the
        target would leak the answer's magnitude into the question.
        """
        X = np.asarray(X, dtype=np.float32)
        y_array = None if y is None else np.asarray(y, dtype=np.float32)

        if not self.scale_by_window:
            return (
                np.log1p(X),
                None if y_array is None else np.log1p(y_array),
                np.ones((len(X), 1), dtype=np.float32),
            )

        demand = X[..., 0] if X.ndim == 3 else X
        scale = demand.mean(axis=-1, keepdims=True).astype(np.float32)
        # A window of pure zeros would divide by zero and poison the batch.
        scale = np.maximum(scale, 1e-6)

        if X.ndim == 3:
            X_scaled = X.copy()
            channels = min(self.demand_channels, X.shape[-1])
            X_scaled[..., :channels] = X[..., :channels] / scale[..., None]
        else:
            X_scaled = X / scale

        y_scaled = None if y_array is None else y_array / scale
        return X_scaled, y_scaled, scale

    # ── Federated exchange (Stage 11) ─────────────────────────────────

    def get_weights(self) -> list[np.ndarray]:
        """Parameters as Flower NDArrays, in stable state_dict order.

        FedAvg averages positionally, so a reordered list would not error — it
        would silently produce a corrupt global model. ``state_dict`` ordering
        is deterministic for a fixed architecture, which is what makes this safe.
        """
        return [v.detach().cpu().numpy() for v in self.net.state_dict().values()]

    def set_weights(self, w: list[np.ndarray]) -> None:
        state = self.net.state_dict()
        if len(w) != len(state):
            raise ValueError(
                f"expected {len(state)} arrays for this architecture, got {len(w)}"
            )
        self.net.load_state_dict(
            {
                key: torch.as_tensor(array, dtype=value.dtype)
                for (key, value), array in zip(state.items(), w, strict=True)
            }
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _features(self, array: np.ndarray) -> torch.Tensor:
        """Inputs as (batch, lookback, features); a bare series gains a channel."""
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(-1) if tensor.ndim == 2 else tensor

    def _targets(self, array: np.ndarray) -> torch.Tensor:
        """Targets stay (batch, horizon).

        Emphatically not run through :meth:`_features` — adding a channel here
        makes the target (batch, horizon, 1), which broadcasts against a
        (batch, horizon) prediction instead of failing, so the loss would be
        computed over a silently wrong pairing.
        """
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def __repr__(self) -> str:
        return f"<LSTMForecaster {self.n_parameters:,} params, horizon={self.horizon}>"
