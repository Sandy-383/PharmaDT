"""Stage 1 Definition of Done: the domain model maps, constrains, and seeds.

The constraint tests matter as much as the mapping tests. Each one asserts that
a specific class of corrupt data is rejected by Postgres rather than by
application code, because the twin in Stage 3 bulk-inserts events and will not
be running the ORM validators on every row.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from pharmadt.core.models import (
    AgentDecision,
    Batch,
    DemandRecord,
    Drug,
    InventoryRecord,
    Node,
    NodeType,
    Shipment,
    ShipmentStatus,
    compute_batch_fingerprint,
)
from pharmadt.core.seed import (
    N_BATCHES,
    _build_batches,
    _build_drugs,
    _build_nodes,
)


def _violates(session: Session, obj) -> None:
    """Assert that flushing ``obj`` trips a database constraint.

    Wrapped in a savepoint so the enclosing test transaction survives.
    """
    with pytest.raises(IntegrityError), session.begin_nested():
        session.add(obj)
        session.flush()


# ── Mapping and round-trips ───────────────────────────────────────────


def test_drug_round_trips(db_session: Session, drug: Drug) -> None:
    fetched = db_session.get(Drug, "TEST-DRUG-01")
    assert fetched is not None
    assert fetched.name == "Test Analgesic 100mg"
    assert fetched.requires_cold_chain is False
    assert fetched.is_controlled is False


def test_node_enum_round_trips(db_session: Session, manufacturer: Node) -> None:
    fetched = db_session.get(Node, "TEST-MFG-01")
    assert fetched is not None
    # Must come back as the enum member, not a bare string.
    assert fetched.node_type is NodeType.MANUFACTURER
    assert fetched.public_key is None, "keypairs are only issued in Stage 4"


def test_batch_relationships_resolve(db_session: Session, batch: Batch) -> None:
    assert batch.drug.drug_id == "TEST-DRUG-01"
    assert batch.manufacturer.node_type is NodeType.MANUFACTURER
    assert batch in batch.drug.batches


def test_shipment_endpoints_resolve_to_distinct_nodes(
    db_session: Session, batch: Batch, manufacturer: Node, pharmacy: Node
) -> None:
    shipment = Shipment(
        shipment_id="TEST-SHIP-01",
        from_node=manufacturer.node_id,
        to_node=pharmacy.node_id,
        batch_id=batch.batch_id,
        quantity=100,
        dispatch_day=1,
        eta_day=3,
        temp_log=[{"sim_day": 1, "temp_c": 5.2}],
    )
    db_session.add(shipment)
    db_session.flush()

    assert shipment.status is ShipmentStatus.PENDING, "default status"
    assert shipment.origin.node_id == manufacturer.node_id
    assert shipment.destination.node_id == pharmacy.node_id
    assert shipment.temp_log[0]["temp_c"] == 5.2


def test_agent_decision_stores_jsonb(db_session: Session) -> None:
    decision = AgentDecision(
        agent_name="InventoryAgent",
        sim_day=17,
        inputs={"stock": 40, "reorder_point": 120},
        action={"type": "REORDER", "quantity": 300},
        justification="Stock 40 below reorder point 120; ordering to cover 14-day lead time.",
    )
    db_session.add(decision)
    db_session.flush()
    db_session.refresh(decision)

    assert decision.decision_id is not None
    assert decision.inputs["reorder_point"] == 120
    assert decision.created_at is not None, "server_default timestamp must populate"


def test_demand_record_reports_stockout(
    db_session: Session, drug: Drug, pharmacy: Node
) -> None:
    record = DemandRecord(
        node_id=pharmacy.node_id,
        drug_id=drug.drug_id,
        sim_day=5,
        quantity_demanded=100,
        quantity_fulfilled=60,
    )
    db_session.add(record)
    db_session.flush()
    assert record.is_stockout is True


# ── Constraints ───────────────────────────────────────────────────────


def test_batch_expiry_must_follow_manufacture(
    db_session: Session, drug: Drug, manufacturer: Node
) -> None:
    _violates(
        db_session,
        Batch(
            batch_id="BAD-BATCH-01",
            drug_id=drug.drug_id,
            manufacturer_id=manufacturer.node_id,
            mfg_date=date(2026, 6, 1),
            expiry_date=date(2026, 1, 1),
            quantity=10,
            batch_fingerprint="0" * 64,
        ),
    )


def test_inventory_cannot_go_negative(
    db_session: Session, batch: Batch, pharmacy: Node
) -> None:
    _violates(
        db_session,
        InventoryRecord(
            node_id=pharmacy.node_id,
            batch_id=batch.batch_id,
            quantity_on_hand=-1,
            sim_day=1,
        ),
    )


def test_inventory_is_unique_per_node_batch_day(
    db_session: Session, batch: Batch, pharmacy: Node
) -> None:
    first = InventoryRecord(
        node_id=pharmacy.node_id, batch_id=batch.batch_id, quantity_on_hand=10, sim_day=1
    )
    db_session.add(first)
    db_session.flush()

    _violates(
        db_session,
        InventoryRecord(
            node_id=pharmacy.node_id,
            batch_id=batch.batch_id,
            quantity_on_hand=20,
            sim_day=1,
        ),
    )


def test_cannot_fulfil_more_than_demanded(
    db_session: Session, drug: Drug, pharmacy: Node
) -> None:
    _violates(
        db_session,
        DemandRecord(
            node_id=pharmacy.node_id,
            drug_id=drug.drug_id,
            sim_day=9,
            quantity_demanded=50,
            quantity_fulfilled=80,
        ),
    )


def test_shipment_endpoints_must_differ(
    db_session: Session, batch: Batch, manufacturer: Node
) -> None:
    _violates(
        db_session,
        Shipment(
            shipment_id="BAD-SHIP-01",
            from_node=manufacturer.node_id,
            to_node=manufacturer.node_id,
            batch_id=batch.batch_id,
            quantity=10,
            dispatch_day=1,
            eta_day=2,
        ),
    )


def test_shipment_cannot_arrive_before_dispatch(
    db_session: Session, batch: Batch, manufacturer: Node, pharmacy: Node
) -> None:
    _violates(
        db_session,
        Shipment(
            shipment_id="BAD-SHIP-02",
            from_node=manufacturer.node_id,
            to_node=pharmacy.node_id,
            batch_id=batch.batch_id,
            quantity=10,
            dispatch_day=5,
            eta_day=2,
        ),
    )


def test_cold_chain_drug_requires_a_temperature_band(db_session: Session) -> None:
    _violates(
        db_session,
        Drug(
            drug_id="BAD-DRUG-01",
            name="Unenforceable Cold Chain Product",
            shelf_life_days=365,
            requires_cold_chain=True,
            temp_min_c=None,
            temp_max_c=None,
        ),
    )


def test_node_coordinates_must_be_on_earth(db_session: Session) -> None:
    _violates(
        db_session,
        Node(
            node_id="BAD-NODE-01",
            name="Nowhere",
            node_type=NodeType.PHARMACY,
            lat=91.0,
            lon=0.0,
            storage_capacity=10,
        ),
    )


# ── Batch fingerprint (the anti-counterfeit primitive) ────────────────


def test_fingerprint_is_deterministic() -> None:
    args = ("B1", "D1", "M1", date(2026, 1, 1), date(2027, 1, 1), 500)
    assert compute_batch_fingerprint(*args) == compute_batch_fingerprint(*args)


def test_fingerprint_is_sha256_hex() -> None:
    fp = compute_batch_fingerprint(
        "B1", "D1", "M1", date(2026, 1, 1), date(2027, 1, 1), 500
    )
    assert len(fp) == 64
    assert set(fp) <= set("0123456789abcdef")


def test_fingerprint_changes_when_quantity_changes() -> None:
    base = ("B1", "D1", "M1", date(2026, 1, 1), date(2027, 1, 1), 500)
    tampered = ("B1", "D1", "M1", date(2026, 1, 1), date(2027, 1, 1), 501)
    assert compute_batch_fingerprint(*base) != compute_batch_fingerprint(*tampered)


def test_fingerprint_field_boundaries_are_unambiguous() -> None:
    """Bare concatenation would collide here; the ``|`` separator prevents it.

    A forger who can shift a character across a field boundary and keep the
    same digest can relabel a batch without detection. This is the specific
    attack the separator exists to defeat.
    """
    a = compute_batch_fingerprint(
        "AB", "C", "M1", date(2026, 1, 1), date(2027, 1, 1), 500
    )
    b = compute_batch_fingerprint(
        "A", "BC", "M1", date(2026, 1, 1), date(2027, 1, 1), 500
    )
    assert a != b


def test_batch_recomputes_its_own_stored_fingerprint(batch: Batch) -> None:
    assert batch.compute_fingerprint() == batch.batch_fingerprint


# ── Seed fixture ──────────────────────────────────────────────────────


def test_seed_builds_the_prescribed_topology() -> None:
    nodes = _build_nodes()
    assert len(nodes) == 12

    by_type: dict[NodeType, int] = {}
    for node in nodes:
        by_type[node.node_type] = by_type.get(node.node_type, 0) + 1

    assert by_type[NodeType.MANUFACTURER] == 1
    assert by_type[NodeType.WAREHOUSE] == 2
    assert by_type[NodeType.DISTRIBUTOR] == 3
    assert by_type[NodeType.PHARMACY] == 6


def test_seed_builds_five_drugs_covering_every_variant() -> None:
    drugs = _build_drugs()
    assert len(drugs) == 5
    assert any(d.requires_cold_chain for d in drugs)
    assert any(d.is_controlled for d in drugs)
    assert all(d.shelf_life_days > 0 for d in drugs)


def test_seed_is_reproducible_for_a_fixed_seed() -> None:
    """Same seed, byte-identical batches — the ledger depends on this."""
    import random

    first = _build_batches(random.Random(42))
    second = _build_batches(random.Random(42))

    assert len(first) == N_BATCHES
    assert [b.batch_id for b in first] == [b.batch_id for b in second]
    assert [b.quantity for b in first] == [b.quantity for b in second]
    assert [b.batch_fingerprint for b in first] == [b.batch_fingerprint for b in second]


def test_seed_batches_carry_correct_fingerprints() -> None:
    import random

    for b in _build_batches(random.Random(42)):
        assert b.compute_fingerprint() == b.batch_fingerprint


def test_seed_batches_expire_after_manufacture() -> None:
    import random

    for b in _build_batches(random.Random(42)):
        assert b.expiry_date > b.mfg_date


def test_seed_includes_batches_near_expiry() -> None:
    """Stage 8's Expiry Agent needs something to detect in the first year."""
    import random

    from pharmadt.config import settings

    horizon = settings.sim_start_date + timedelta(days=settings.sim_days)
    expiring = [b for b in _build_batches(random.Random(42)) if b.expiry_date <= horizon]
    assert expiring, "no batch expires within the simulated year"


def test_full_seed_inserts_cleanly(db_session: Session) -> None:
    """All three entity sets satisfy every FK and constraint together.

    Clears any committed seed data first so the counts are exact. That happens
    inside the test transaction, so the fixture's rollback puts it all back —
    running this suite never costs you your seeded database.
    """
    import random

    for model in (
        AgentDecision,
        DemandRecord,
        InventoryRecord,
        Shipment,
        Batch,
        Drug,
        Node,
    ):
        db_session.execute(delete(model))
    db_session.flush()

    db_session.add_all(_build_nodes())
    db_session.add_all(_build_drugs())
    db_session.flush()
    db_session.add_all(_build_batches(random.Random(42)))
    db_session.flush()

    assert db_session.scalar(select(func.count()).select_from(Node)) == 12
    assert db_session.scalar(select(func.count()).select_from(Drug)) == 5
    assert db_session.scalar(select(func.count()).select_from(Batch)) == N_BATCHES
