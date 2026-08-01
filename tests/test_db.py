"""The session helper commits on success and rolls back on failure.

Unlike the rest of the suite these tests commit for real, because that is the
behaviour under test. Each cleans up after itself in a ``finally``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import delete

from pharmadt.core.db import session_scope
from pharmadt.core.models import Drug

COMMIT_ID = "TMP-DRUG-COMMIT"
ROLLBACK_ID = "TMP-DRUG-ROLLBACK"


@pytest.fixture(autouse=True)
def _cleanup() -> Iterator[None]:
    yield
    with session_scope() as session:
        session.execute(delete(Drug).where(Drug.drug_id.in_([COMMIT_ID, ROLLBACK_ID])))


def test_session_scope_commits_on_success() -> None:
    with session_scope() as session:
        session.add(Drug(drug_id=COMMIT_ID, name="Temp", shelf_life_days=30))

    with session_scope() as session:
        assert session.get(Drug, COMMIT_ID) is not None


def test_session_scope_rolls_back_on_exception() -> None:
    """A half-written simulation day must not survive the exception that killed it."""
    with pytest.raises(RuntimeError, match="boom"), session_scope() as session:
        session.add(Drug(drug_id=ROLLBACK_ID, name="Temp", shelf_life_days=30))
        session.flush()
        raise RuntimeError("boom")

    with session_scope() as session:
        assert session.get(Drug, ROLLBACK_ID) is None


def test_session_scope_reraises_the_original_error() -> None:
    """The rollback must not mask the exception that caused it."""

    class Sentinel(Exception):
        pass

    with pytest.raises(Sentinel), session_scope() as session:
        session.add(Drug(drug_id=ROLLBACK_ID, name="Temp", shelf_life_days=30))
        raise Sentinel
