"""Prophet demand forecaster, implementing the DemandModel contract.

Prophet and the LSTM are complementary rather than competing: Prophet models
trend, weekly and yearly seasonality, and holidays explicitly, while the LSTM
picks up short-term nonlinearity the additive decomposition cannot express. The
Stage 2 EDA found the seasonality here is a **December spike rather than a
smooth wave**, which is precisely the shape a sinusoid fits badly and Prophet's
holiday terms fit well.

Prophet is per-series by construction, which is also why it cannot be federated:
there is no shared parameter vector to average. ``get_weights`` therefore
returns the fitted seasonality coefficients for inspection, and Stage 11
federates the LSTM.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pharmadt.core.interfaces import DemandModel
from pharmadt.ml.forecasting import HORIZON

logger = logging.getLogger(__name__)

# Prophet/cmdstanpy narrate every fit at INFO. Fitting 30 series would bury
# anything worth reading, so they are quietened here rather than globally.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def _locate_bundled_cmdstan() -> str | None:
    """Point cmdstanpy at the CmdStan that ships inside the Prophet wheel.

    Prophet 1.1.5 bundles CmdStan under ``prophet/stan_model/`` but does not
    export ``CMDSTAN``, so on Windows ``cmdstanpy.cmdstan_path()`` raises "No
    CmdStan installation found" and Prophet then fails while *logging* that
    failure -- surfacing as a bare ``AttributeError: no attribute
    'stan_backend'`` that says nothing about the real cause.

    Setting the variable ourselves keeps the fix inside the repository rather
    than in a machine's environment, so a teammate's clone works unmodified.
    """
    import os

    if os.environ.get("CMDSTAN"):
        return os.environ["CMDSTAN"]

    import prophet as _prophet

    stan_root = Path(_prophet.__file__).parent / "stan_model"
    candidates = sorted(stan_root.glob("cmdstan-*")) if stan_root.is_dir() else []
    for candidate in reversed(candidates):  # newest first
        if not (candidate / "bin").is_dir():
            continue

        # cmdstanpy refuses a CmdStan tree without a makefile, but the wheel
        # ships a *trimmed* one (bin/ and stan/ only) because prophet_model.bin
        # is already compiled -- nothing here ever needs to build. The stub
        # satisfies the check without pretending a toolchain exists. The
        # alternative, cmdstanpy.install_cmdstan(), downloads ~500MB and needs
        # a C++ compiler to produce a binary we already have.
        makefile = candidate / "makefile"
        if not makefile.exists():
            try:
                makefile.write_text(
                    "# Stub for cmdstanpy's path validation.\n"
                    "# Prophet ships a precompiled model binary; nothing is built here.\n",
                    encoding="utf-8",
                )
                logger.debug("wrote CmdStan makefile stub at %s", makefile)
            except OSError as exc:  # read-only install
                logger.warning("could not write %s: %s", makefile, exc)

        os.environ["CMDSTAN"] = str(candidate)
        logger.debug("using bundled CmdStan at %s", candidate)
        return str(candidate)

    logger.warning("no bundled CmdStan found under %s; Prophet may fail", stan_root)
    return None


class ProphetForecaster(DemandModel):
    """Additive trend + seasonality + holidays, fitted per series."""

    def __init__(
        self,
        country_holidays: str | None = "IN",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = "additive",
        horizon: int = HORIZON,
    ) -> None:
        self.country_holidays = country_holidays
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.horizon = horizon
        self.model_: Any = None
        self.last_date_: pd.Timestamp | None = None

    # ── Training ──────────────────────────────────────────────────────

    def fit(self, X: Any, y: Any = None) -> ProphetForecaster:
        """Fit one series.

        ``X`` is either a DataFrame with ``ds``/``y`` columns, or a sequence of
        demand values paired with ``y`` as dates.
        """
        _locate_bundled_cmdstan()
        from prophet import Prophet

        frame = self._as_frame(X, y)
        self.last_date_ = frame["ds"].max()

        model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
        )
        if self.country_holidays:
            # Indian holidays: the network is modelled across Karnataka, so
            # national holidays are the ones that move dispensing volume.
            model.add_country_holidays(country_name=self.country_holidays)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(frame)

        self.model_ = model
        return self

    # ── Inference ─────────────────────────────────────────────────────

    def predict(self, X: Any = None, horizon: int = HORIZON) -> np.ndarray:
        """Forecast ``horizon`` days beyond the fitted history."""
        if self.model_ is None:
            raise RuntimeError("ProphetForecaster.predict called before fit")

        future = self.model_.make_future_dataframe(periods=horizon, freq="D")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model_.predict(future)

        # Prophet is unbounded below and will happily predict negative demand
        # on a quiet series; clip rather than pass nonsense to the twin.
        return np.clip(forecast["yhat"].to_numpy()[-horizon:], 0.0, None)

    # ── DemandModel contract ──────────────────────────────────────────

    def get_weights(self) -> list[np.ndarray]:
        """Fitted coefficients, for inspection rather than for FedAvg.

        Prophet fits one model per series with its own changepoints, so there
        is no architecture shared across clients to average. Stage 11 federates
        the LSTM; this exists so the interface holds and the asymmetry is
        visible rather than hidden behind a NotImplementedError.
        """
        if self.model_ is None:
            return [np.zeros(1, dtype=np.float32)]
        params = self.model_.params
        return [np.asarray(params[k], dtype=np.float32).ravel() for k in sorted(params)]

    def set_weights(self, w: list[np.ndarray]) -> None:
        raise NotImplementedError(
            "Prophet fits per-series changepoints and cannot accept averaged "
            "weights. Stage 11 federates the LSTM instead."
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _as_frame(X: Any, y: Any = None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            if {"ds", "y"} <= set(X.columns):
                return X[["ds", "y"]].dropna()
            raise ValueError("DataFrame must carry 'ds' and 'y' columns")

        values = np.asarray(X, dtype=float).ravel()
        dates = (
            pd.to_datetime(y)
            if y is not None
            else pd.date_range("2013-01-01", periods=len(values), freq="D")
        )
        return pd.DataFrame({"ds": dates, "y": values})

    def __repr__(self) -> str:
        return f"<ProphetForecaster fitted={self.model_ is not None}>"


class EnsembleForecaster(DemandModel):
    """Inverse-error weighted blend of several forecasters.

    Averaging beats either model alone when their errors are not perfectly
    correlated, which is the usual case for a structural model and a neural one.
    Weights come from validation error, never test error — weighting by test
    performance is leakage dressed up as model selection.
    """

    def __init__(self, models: dict[str, DemandModel], weights: dict[str, float] | None = None):
        self.models = models
        self.weights = weights or dict.fromkeys(models, 1.0 / len(models))

    @classmethod
    def from_validation_errors(
        cls, models: dict[str, DemandModel], errors: dict[str, float]
    ) -> EnsembleForecaster:
        inverse = {name: 1.0 / max(err, 1e-6) for name, err in errors.items()}
        total = sum(inverse.values())
        return cls(models, {name: value / total for name, value in inverse.items()})

    def fit(self, X: Any, y: Any = None) -> EnsembleForecaster:
        for model in self.models.values():
            model.fit(X, y)
        return self

    def predict(self, X: Any, horizon: int = HORIZON) -> np.ndarray:
        blended = None
        for name, model in self.models.items():
            prediction = np.asarray(model.predict(X, horizon), dtype=float)
            weighted = prediction * self.weights[name]
            blended = weighted if blended is None else blended + weighted
        return blended

    def get_weights(self) -> list[np.ndarray]:
        return [np.array([self.weights[n] for n in sorted(self.models)], dtype=np.float32)]

    def set_weights(self, w: list[np.ndarray]) -> None:
        for name, value in zip(sorted(self.models), np.asarray(w[0]).ravel(), strict=True):
            self.weights[name] = float(value)
