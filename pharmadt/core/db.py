"""Database engine, declarative base, and session factory.

Stage 1 defines the ORM entities on top of :class:`Base`. Stage 4's ledger
writes through the same engine so that provenance records and domain rows
share a transaction boundary.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from pharmadt.config import settings

# pool_pre_ping guards against connections dropped by a restarted container.
engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """Declarative base shared by every ORM entity (Stage 1)."""


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
