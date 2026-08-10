"""Central configuration for PharmaDT.

Every tunable constant in the system lives here. No other module may hardcode a
connection string, simulation parameter, or threshold — import ``settings``
instead. Later stages extend :class:`Settings` with their own fields rather
than introducing constants at their point of use.
"""

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Persistence ───────────────────────────────────────────────────
    database_url: str = (
        "postgresql+psycopg2://pharmadt:pharmadt@localhost:5432/pharmadt"
    )

    # ── Simulation (Stage 3) ──────────────────────────────────────────
    # Seeds every stochastic process. Two runs at the same seed must produce
    # byte-identical event logs; Stage 3 asserts this in a test.
    sim_seed: int = 42
    sim_days: int = 365
    num_nodes: int = 12

    # Calendar date that simulated day 0 maps to. Batch manufacture and expiry
    # dates are real dates, so the twin needs a fixed epoch to convert against;
    # leaving it floating would make seeded data non-reproducible across runs.
    sim_start_date: date = date(2026, 1, 1)

    # ── Network topology (Stage 3) ────────────────────────────────────
    transit_speed_km_per_day: float = 400.0
    transit_cost_base_per_unit: float = 0.5
    transit_cost_per_km_per_unit: float = 0.002

    # ── Consumer demand (Stage 3) ─────────────────────────────────────
    # Analytic defaults. Stage 2 refits these per (node, drug) from Rossmann
    # and injects them as a DemandProfile; the shape of the model does not
    # change, only where the numbers come from.
    base_daily_demand: float = 40.0
    demand_dispersion: float = 0.35
    demand_weekend_factor: float = 0.6
    demand_seasonal_amplitude: float = 0.25

    # ── Baseline replenishment policy (Stage 3) ───────────────────────
    # A fixed-threshold (s, S) policy. Stage 6's Inventory Agent must beat
    # this; it is the control arm of that comparison, not a placeholder.
    #
    # order_up_to_days must stay at or below the 28-day demand window in
    # twin.nodes.DEMAND_HISTORY_DAYS. A node orders its whole horizon in one
    # lump, and its supplier estimates demand by averaging observed orders over
    # that window — so a horizon longer than the window biases the supplier's
    # rate estimate upward by their ratio, and the bias compounds at every tier.
    # At 42 against a 28-day window the warehouses ended up holding nine months
    # of network demand.
    # The reorder point covers lead time plus the review period and nothing
    # else. That naivety is the point: it ignores demand variability entirely,
    # which is exactly the gap Stage 6's safety-stock term (z·σ·√L, z = 1.65)
    # is meant to close. Padding it here would leave that agent nothing to win.
    reorder_point_days: int = 3
    order_up_to_days: int = 21

    # ── Inventory Agent (Stage 6) ─────────────────────────────────────
    # Normal quantile for the safety-stock term. The (s, S) baseline has no
    # such term at all, which is the gap the agent exists to close.
    #
    # The implementation guide prescribes 1.65 (the textbook 95% service
    # level). Measured over five seeds, 0.84 dominates it here on every metric
    # at once -- same stockout reduction, an eighth of the wastage, and half
    # the extra inventory:
    #
    #   policy         stockout    short   waste    avg_inv
    #   baseline        0.00256     1059       6    116,698
    #   z = 0.84        0.00002       10     101    139,156
    #   z = 1.65        0.00004       18     878    157,414
    #
    # The reason is that once the risk period is specified correctly (echelon
    # lead time, not one hop) the order-up-to level already carries most of the
    # buffer, so additional safety stock mostly sits until it expires. Raise it
    # back toward 1.65 if wastage stops mattering and stockouts dominate.
    service_level_z: float = 0.84
    # Fallback lead time when the network graph has no edge to price.
    default_lead_time_days: int = 2

    # ── Manufacturing (Stage 3) ───────────────────────────────────────
    # Stage 13's factory-shutdown scenario works by driving capacity to zero,
    # so production has to be a real constraint rather than infinite supply.
    production_batch_size: int = 20_000
    production_capacity_per_day: int = 120_000

    # ── Cold chain (Stage 3) ──────────────────────────────────────────
    coldchain_excursion_prob_per_day: float = 0.02
    ambient_temp_c: float = 25.0
    cold_setpoint_c: float = 5.0
    cold_excursion_temp_c: float = 12.0

    # ── Expiry (Stage 3 wastage, Stage 8 alerting) ────────────────────
    expiry_alert_days: int = 30  # FR-04

    # ── API (Stage 14) ────────────────────────────────────────────────
    # Signs bearer tokens. SecretStr so an accidental repr of settings cannot
    # print it. The default is a development value and the API says so on
    # startup; a deployment sets API_SECRET_KEY in the environment.
    api_secret_key: SecretStr = SecretStr("dev-only-not-a-production-secret")

    # ── Dataset acquisition (Stage 2) ─────────────────────────────────
    # Kaggle personal access token, used as a Bearer credential. SecretStr so
    # it cannot be printed by an accidental repr of the settings object.
    kaggle_api_token: SecretStr | None = None

    # ── Expiry Agent (Stage 8) ────────────────────────────────────────
    # How far ahead the redistribution auction looks. Deliberately *not*
    # expiry_alert_days: FR-04's 30 days is a *detection* threshold, and
    # detection and action are different questions. At 30 days the receiving
    # node usually cannot sell the stock either, so the transfer relocates the
    # waste instead of preventing it. Measured over seeds 45/47/49:
    #
    #   horizon    30d    60d    90d   120d
    #   wastage  3,728  3,728  4,456    625
    #
    # Stock has to move while somebody still has time to sell it.
    redistribution_horizon_days: int = 120
    # Nominal economics for the redistribution auction. Absolute values do not
    # matter — only their ratios decide whether a transfer is worth making, so
    # these are stated as a unit price with everything else relative to it.
    unit_value: float = 10.0
    # Disposal costs money on top of losing the stock, so avoiding it is worth
    # more than the goods alone. This sets the auction's reserve price.
    disposal_cost_per_unit: float = 2.0
    # Per unit, per km. Above roughly a tenth of unit value, transport swamps
    # the saving and redistribution stops being worth doing at any distance.
    transport_cost_per_unit_km: float = 0.002

    # ── Provenance ledger (Stage 4) ───────────────────────────────────
    # Records per Merkle block. The root is written onto the last record of
    # each block, so inclusion proofs cost O(log 64) = 6 sibling hashes.
    merkle_interval: int = 64
    # Private signing keys. Gitignored — in production these belong in an HSM
    # or KMS, which the report states explicitly rather than implying.
    keys_dir: Path = Path("data/keys")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()
