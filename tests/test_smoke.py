"""Stage 0 smoke test: the package imports, config loads, the database answers.

This is the Stage 0 Definition of Done. It deliberately touches a real
Postgres rather than SQLite — the ledger in Stage 4 depends on Postgres
features (JSONB, BIGSERIAL, plpgsql triggers) that SQLite cannot emulate, so
a green test against SQLite would be a false signal.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from pharmadt.config import settings
from pharmadt.core.db import engine


def test_settings_load():
    assert settings.sim_days > 0
    assert settings.num_nodes > 0
    assert settings.sim_seed is not None
    assert settings.database_url.startswith("postgresql")


def test_database_connects():
    try:
        with engine.connect() as conn:
            assert conn.execute(text("SELECT 1")).scalar_one() == 1
    except OperationalError as exc:
        pytest.fail(
            "Could not reach Postgres. Run `make db-up` (or "
            f"`docker compose up -d db`) and retry.\n\n{exc}"
        )


def test_postgres_is_version_15():
    """Stage 4's append-only triggers are written against Postgres 15."""
    with engine.connect() as conn:
        version = conn.execute(text("SHOW server_version")).scalar_one()
    assert version.startswith("15"), f"expected Postgres 15, got {version}"
