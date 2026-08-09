"""The append-only guarantee is a property of the database.

If these fail, the project's central claim is false regardless of what the
application code does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

SCHEMA_SQL = Path("pharmadt/ledger/schema.sql")


def _blocked(session: Session, statement: str) -> str:
    """Run a statement expecting the append-only trigger to reject it."""
    with pytest.raises(DBAPIError) as caught, session.begin_nested():
        session.execute(text(statement))
    return str(caught.value)


# ── Triggers ──────────────────────────────────────────────────────────


def test_update_is_blocked(db_session: Session) -> None:
    message = _blocked(db_session, "UPDATE provenance_records SET sim_day = 0")
    assert "append-only" in message


def test_delete_is_blocked(db_session: Session) -> None:
    message = _blocked(db_session, "DELETE FROM provenance_records")
    assert "append-only" in message


def test_truncate_is_blocked(db_session: Session) -> None:
    """Row-level DELETE triggers never fire for TRUNCATE.

    Without a statement-level guard the entire ledger could be erased in one
    statement while the other two triggers sat and watched.
    """
    message = _blocked(db_session, "TRUNCATE provenance_records")
    assert "append-only" in message


def test_all_three_guards_are_installed(db_session: Session) -> None:
    installed = set(
        db_session.scalars(
            text(
                "SELECT tgname FROM pg_trigger "
                "WHERE tgrelid = 'provenance_records'::regclass AND NOT tgisinternal"
            )
        )
    )
    assert installed == {"no_update", "no_delete", "no_truncate"}


def test_insert_is_still_allowed(db_session: Session) -> None:
    """Append-only means append-only, not read-only."""
    db_session.execute(
        text(
            "INSERT INTO provenance_records "
            "(batch_id, event_type, payload, sim_day, prev_hash, record_hash,"
            " signer_node, signature) "
            "VALUES ('B-TEST', 'BATCH_CREATED', '{}'::jsonb, 1, :z, :h, 'N', 'sig')"
        ),
        {"z": "0" * 64, "h": "f" * 64},
    )


# ── Reference DDL still matches the database ──────────────────────────


def _declared_columns() -> set[str]:
    body = SCHEMA_SQL.read_text(encoding="utf-8")
    create = re.search(
        r"CREATE TABLE provenance_records\s*\((.*?)\n\);", body, re.DOTALL
    )
    assert create, "schema.sql no longer declares provenance_records"
    return {
        match.group(1)
        for line in create.group(1).splitlines()
        if (match := re.match(r"\s*(\w+)\s+\w", line))
    }


def test_schema_sql_still_describes_the_live_table(db_session: Session) -> None:
    """schema.sql is documentation, so it can silently drift from the migration."""
    live = set(
        db_session.scalars(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'provenance_records'"
            )
        )
    )
    assert _declared_columns() == live


def test_schema_sql_declares_every_guard() -> None:
    body = SCHEMA_SQL.read_text(encoding="utf-8")
    for trigger in ("no_update", "no_delete", "no_truncate"):
        assert trigger in body


def test_record_hash_is_unique(db_session: Session) -> None:
    """Two records sharing a hash would make the chain ambiguous."""
    constraints = set(
        db_session.scalars(
            text(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = 'provenance_records'::regclass AND contype = 'u'"
            )
        )
    )
    assert constraints, "record_hash has no uniqueness constraint"


def test_batch_lookup_is_indexed(db_session: Session) -> None:
    """get_provenance filters on batch_id over the whole ledger."""
    indexes = set(
        db_session.scalars(
            text("SELECT indexname FROM pg_indexes WHERE tablename = 'provenance_records'")
        )
    )
    assert "idx_prov_batch" in indexes
