"""Central configuration for PharmaDT.

Every tunable constant in the system lives here. No other module may hardcode a
connection string, simulation parameter, or threshold — import ``settings``
instead. Later stages extend :class:`Settings` with their own fields rather
than introducing constants at their point of use.
"""

from datetime import date
from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()
