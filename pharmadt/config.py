"""Central configuration for PharmaDT.

Every tunable constant in the system lives here. No other module may hardcode a
connection string, simulation parameter, or threshold — import ``settings``
instead. Later stages extend :class:`Settings` with their own fields rather
than introducing constants at their point of use.
"""

from datetime import date
from functools import lru_cache
from pathlib import Path

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
