"""Shared pytest fixtures.

Every test runs inside a transaction that is rolled back afterwards, so the
suite can be run against the same database the simulation uses without ever
leaving a row behind. Constraint tests use ``begin_nested`` savepoints, because
a failed INSERT would otherwise poison the surrounding transaction and cascade
into unrelated failures.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import pytest
from sqlalchemy.orm import Session

from pharmadt.core.db import engine
from pharmadt.core.models import Batch, Drug, Node, NodeType, compute_batch_fingerprint


@pytest.fixture
def db_session() -> Iterator[Session]:
    """A session whose work is always rolled back."""
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    try:
        yield session
    finally:
        session.close()
        # A failed flush may already have unwound the transaction; rolling back
        # a dead transaction only produces a confusing warning on top of the
        # real failure.
        if transaction.is_active:
            transaction.rollback()
        connection.close()


@pytest.fixture
def drug(db_session: Session) -> Drug:
    d = Drug(
        drug_id="TEST-DRUG-01",
        name="Test Analgesic 100mg",
        atc_code="N02BE01",
        shelf_life_days=365,
    )
    db_session.add(d)
    db_session.flush()
    return d


@pytest.fixture
def cold_drug(db_session: Session) -> Drug:
    d = Drug(
        drug_id="TEST-DRUG-02",
        name="Test Vaccine",
        atc_code="J07BB02",
        shelf_life_days=180,
        requires_cold_chain=True,
        temp_min_c=2.0,
        temp_max_c=8.0,
    )
    db_session.add(d)
    db_session.flush()
    return d


@pytest.fixture
def manufacturer(db_session: Session) -> Node:
    n = Node(
        node_id="TEST-MFG-01",
        name="Test Plant",
        node_type=NodeType.MANUFACTURER,
        lat=12.9716,
        lon=77.5946,
        storage_capacity=100_000,
        has_cold_storage=True,
    )
    db_session.add(n)
    db_session.flush()
    return n


@pytest.fixture
def pharmacy(db_session: Session) -> Node:
    n = Node(
        node_id="TEST-PH-01",
        name="Test Pharmacy",
        node_type=NodeType.PHARMACY,
        lat=12.9250,
        lon=77.5938,
        storage_capacity=5_000,
        has_cold_storage=False,
    )
    db_session.add(n)
    db_session.flush()
    return n


@pytest.fixture
def batch(db_session: Session, drug: Drug, manufacturer: Node) -> Batch:
    mfg = date(2026, 1, 1)
    expiry = date(2027, 1, 1)
    quantity = 5_000
    b = Batch(
        batch_id="TEST-BATCH-01",
        drug_id=drug.drug_id,
        manufacturer_id=manufacturer.node_id,
        mfg_date=mfg,
        expiry_date=expiry,
        quantity=quantity,
        batch_fingerprint=compute_batch_fingerprint(
            "TEST-BATCH-01", drug.drug_id, manufacturer.node_id, mfg, expiry, quantity
        ),
    )
    db_session.add(b)
    db_session.flush()
    return b
