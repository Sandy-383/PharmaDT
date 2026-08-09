"""The provenance ledger end to end: appending, linking, verifying, detecting.

Every test runs inside a transaction that is rolled back, so the real ledger is
never touched — an append-only table offers no way to undo a test that wrote to
it for real.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select, text

from pharmadt.config import settings
from pharmadt.core.events import Event, EventType
from pharmadt.core.models import ProvenanceRecord
from pharmadt.ledger.chain import HashChainLedger, LedgerError, default_signer
from pharmadt.ledger.crypto import GENESIS_PREV_HASH
from pharmadt.ledger.keyring import UnauthorisedSigner

MANUFACTURER = "NODE-MFG-01"
WAREHOUSE = "NODE-WH-01"
PHARMACY = "NODE-PH-01"


def append(ledger: HashChainLedger, batch_id: str = "BATCH-0001", day: int = 1, **kw) -> str:
    params = {
        "batch_id": batch_id,
        "event_type": EventType.BATCH_CREATED,
        "from_node": None,
        "to_node": MANUFACTURER,
        "payload": {"quantity": 100},
        "signer_node": MANUFACTURER,
        "sim_day": day,
    }
    params.update(kw)
    return ledger.record_event(**params)


def records(session) -> list[ProvenanceRecord]:
    return list(session.scalars(select(ProvenanceRecord).order_by(ProvenanceRecord.seq)))


# ── Appending and linking ─────────────────────────────────────────────


def test_the_first_record_links_to_the_genesis_anchor(empty_ledger, db_session) -> None:
    append(empty_ledger)
    assert records(db_session)[0].prev_hash == GENESIS_PREV_HASH


def test_each_record_links_to_its_predecessor(empty_ledger, db_session) -> None:
    for day in range(4):
        append(empty_ledger, day=day)

    chain = records(db_session)
    assert len(chain) == 4
    for earlier, later in zip(chain[:-1], chain[1:], strict=True):
        assert later.prev_hash == earlier.record_hash


def test_record_event_returns_the_stored_hash(empty_ledger, db_session) -> None:
    returned = append(empty_ledger)
    assert records(db_session)[0].record_hash == returned


def test_appending_populates_every_column(empty_ledger, db_session) -> None:
    append(empty_ledger, payload={"quantity": 42, "drug_id": "DRUG-001"}, day=7)
    record = records(db_session)[0]

    assert record.batch_id == "BATCH-0001"
    assert record.event_type == "BATCH_CREATED"
    assert record.to_node == MANUFACTURER
    assert record.payload == {"quantity": 42, "drug_id": "DRUG-001"}
    assert record.sim_day == 7
    assert record.signer_node == MANUFACTURER
    assert len(record.signature) > 0
    assert record.recorded_at is not None


# ── Permissioning (NFR-04) ────────────────────────────────────────────


def test_a_node_outside_the_registry_cannot_write(empty_ledger) -> None:
    """The public-key allow-list is this project's channel policy."""
    with pytest.raises(UnauthorisedSigner):
        append(empty_ledger, signer_node="NODE-IMPOSTOR")


def test_telemetry_cannot_be_written_into_the_custody_chain(empty_ledger) -> None:
    with pytest.raises(LedgerError, match="telemetry"):
        append(empty_ledger, event_type=EventType.DEMAND_UNFULFILLED)


# ── Verification ──────────────────────────────────────────────────────


def test_an_untouched_chain_verifies(empty_ledger) -> None:
    for day in range(6):
        append(empty_ledger, day=day)

    assert empty_ledger.verify_chain() is True
    assert empty_ledger.last_result.records_checked == 6
    assert empty_ledger.last_result.broken_at_seq is None


def test_an_empty_chain_verifies_vacuously(empty_ledger) -> None:
    assert empty_ledger.verify_chain() is True


def test_editing_a_payload_is_caught_and_the_record_named(
    empty_ledger, db_session
) -> None:
    """The Stage 4 demo in one test."""
    for day in range(6):
        append(empty_ledger, day=day)
    target = records(db_session)[2].seq

    db_session.execute(text("ALTER TABLE provenance_records DISABLE TRIGGER no_update"))
    db_session.execute(
        text("UPDATE provenance_records SET payload = '{\"quantity\": 999999}' WHERE seq = :s"),
        {"s": target},
    )
    db_session.execute(text("ALTER TABLE provenance_records ENABLE TRIGGER no_update"))

    result = empty_ledger.verify_chain_detailed()
    assert result.valid is False
    assert result.broken_at_seq == target
    assert "record_hash" in result.reason


def test_editing_any_hashed_column_is_caught(empty_ledger, db_session) -> None:
    for day in range(4):
        append(empty_ledger, day=day)
    target = records(db_session)[1].seq

    db_session.execute(text("ALTER TABLE provenance_records DISABLE TRIGGER no_update"))
    db_session.execute(
        text("UPDATE provenance_records SET sim_day = 9999 WHERE seq = :s"), {"s": target}
    )
    db_session.execute(text("ALTER TABLE provenance_records ENABLE TRIGGER no_update"))

    assert empty_ledger.verify_chain_detailed().broken_at_seq == target


def test_removing_a_record_breaks_the_link(empty_ledger, db_session) -> None:
    """Deletion is caught by the prev_hash link, not by the content hash."""
    for day in range(6):
        append(empty_ledger, day=day)
    target = records(db_session)[2].seq

    db_session.execute(text("ALTER TABLE provenance_records DISABLE TRIGGER no_delete"))
    db_session.execute(text("DELETE FROM provenance_records WHERE seq = :s"), {"s": target})
    db_session.execute(text("ALTER TABLE provenance_records ENABLE TRIGGER no_delete"))

    result = empty_ledger.verify_chain_detailed()
    assert result.valid is False
    assert "prev_hash" in result.reason


def test_a_forged_signature_is_caught(empty_ledger, db_session) -> None:
    for day in range(3):
        append(empty_ledger, day=day)
    target = records(db_session)[1].seq

    db_session.execute(text("ALTER TABLE provenance_records DISABLE TRIGGER no_update"))
    db_session.execute(
        text("UPDATE provenance_records SET signature = '00ff' WHERE seq = :s"), {"s": target}
    )
    db_session.execute(text("ALTER TABLE provenance_records ENABLE TRIGGER no_update"))

    result = empty_ledger.verify_chain_detailed()
    assert result.broken_at_seq == target
    assert "signature" in result.reason


def test_a_record_reattributed_to_another_node_fails_verification(
    empty_ledger, db_session
) -> None:
    """Non-repudiation: a signature only verifies for the node that made it."""
    for day in range(3):
        append(empty_ledger, day=day)
    target = records(db_session)[1].seq

    db_session.execute(text("ALTER TABLE provenance_records DISABLE TRIGGER no_update"))
    db_session.execute(
        text("UPDATE provenance_records SET signer_node = :n WHERE seq = :s"),
        {"n": PHARMACY, "s": target},
    )
    db_session.execute(text("ALTER TABLE provenance_records ENABLE TRIGGER no_update"))

    assert empty_ledger.verify_chain_detailed().broken_at_seq == target


def test_verification_can_be_scoped_to_a_range(empty_ledger, db_session) -> None:
    for day in range(8):
        append(empty_ledger, day=day)
    chain = records(db_session)

    result = empty_ledger.verify_chain_detailed(chain[2].seq, chain[5].seq)
    assert result.valid is True
    assert result.records_checked == 4


# ── Reading ───────────────────────────────────────────────────────────


def test_provenance_returns_only_that_batch_in_order(empty_ledger) -> None:
    append(empty_ledger, batch_id="BATCH-0001", day=1)
    append(empty_ledger, batch_id="BATCH-0002", day=2)
    append(empty_ledger, batch_id="BATCH-0001", day=3)

    trace = empty_ledger.get_provenance("BATCH-0001")
    assert [r["sim_day"] for r in trace] == [1, 3]
    assert [r["seq"] for r in trace] == sorted(r["seq"] for r in trace)


def test_provenance_of_an_unknown_batch_is_empty(empty_ledger) -> None:
    assert empty_ledger.get_provenance("BATCH-NEVER") == []


def test_provenance_rows_are_plain_serialisable_data(empty_ledger) -> None:
    """The Stage 14 API serves these directly, so no ORM objects may leak."""
    import json

    append(empty_ledger)
    json.dumps(empty_ledger.get_provenance("BATCH-0001"))


def test_height_and_tip_track_the_chain(empty_ledger) -> None:
    assert empty_ledger.height() == 0
    assert empty_ledger.tip() == GENESIS_PREV_HASH

    last = append(empty_ledger)
    assert empty_ledger.height() == 1
    assert empty_ledger.tip() == last


# ── Anti-counterfeit (the Stage 10 signal) ────────────────────────────


def test_a_genuine_fingerprint_is_accepted(ledger, batch) -> None:
    assert ledger.verify_batch_fingerprint(batch.batch_id, batch.batch_fingerprint)


def test_a_forged_fingerprint_is_rejected(ledger, batch) -> None:
    assert not ledger.verify_batch_fingerprint(batch.batch_id, "0" * 64)


def test_an_unknown_batch_id_is_rejected(ledger, batch) -> None:
    """A product presenting an unrecognised identifier is equally suspect."""
    assert not ledger.verify_batch_fingerprint("BATCH-COUNTERFEIT", batch.batch_fingerprint)


# ── Bulk anchoring from the twin ──────────────────────────────────────


def _custody_event(day: int) -> Event:
    return Event(
        event_type=EventType.SHIPMENT_DISPATCHED,
        sim_day=day,
        batch_id="BATCH-0001",
        from_node=WAREHOUSE,
        to_node=PHARMACY,
        payload={"quantity": 10},
    )


def test_anchoring_writes_only_custody_events(empty_ledger) -> None:
    events = [
        _custody_event(1),
        Event(event_type=EventType.DEMAND_UNFULFILLED, sim_day=1, from_node=PHARMACY),
        _custody_event(2),
    ]
    assert empty_ledger.anchor_events(events) == 2
    assert empty_ledger.height() == 2


def test_anchoring_skips_events_with_no_batch(empty_ledger) -> None:
    """A custody record about no particular batch has nothing to trace."""
    orphan = Event(
        event_type=EventType.SHIPMENT_DISPATCHED, sim_day=1, from_node=WAREHOUSE
    )
    assert empty_ledger.anchor_events([orphan]) == 0


def test_a_bulk_anchored_chain_verifies(empty_ledger) -> None:
    assert empty_ledger.anchor_events([_custody_event(d) for d in range(30)]) == 30
    assert empty_ledger.verify_chain() is True


def test_anchoring_continues_an_existing_chain(empty_ledger, db_session) -> None:
    first = append(empty_ledger)
    empty_ledger.anchor_events([_custody_event(2)])

    chain = records(db_session)
    assert chain[1].prev_hash == first
    assert empty_ledger.verify_chain() is True


def test_anchoring_nothing_is_a_noop(empty_ledger) -> None:
    assert empty_ledger.anchor_events([]) == 0


@pytest.mark.parametrize(
    ("event_type", "expected"),
    [
        (EventType.BATCH_CREATED, PHARMACY),
        (EventType.SHIPMENT_DISPATCHED, WAREHOUSE),
        (EventType.SHIPMENT_RECEIVED, PHARMACY),
        (EventType.DISPENSED, WAREHOUSE),
    ],
)
def test_the_signer_is_the_node_holding_custody(event_type, expected) -> None:
    event = Event(
        event_type=event_type,
        sim_day=1,
        batch_id="B1",
        from_node=WAREHOUSE,
        to_node=PHARMACY,
    )
    assert default_signer(event) == expected


def test_telemetry_has_no_signer() -> None:
    assert default_signer(Event(event_type=EventType.STOCK_EXPIRED, sim_day=1)) is None


# ── Merkle anchoring ──────────────────────────────────────────────────


def test_a_closed_block_gets_a_root_and_proofs_verify(empty_ledger, db_session) -> None:
    interval = settings.merkle_interval
    empty_ledger.anchor_events([_custody_event(d) for d in range(interval * 2)])

    anchored = db_session.scalars(
        select(ProvenanceRecord.seq)
        .where(ProvenanceRecord.merkle_root.isnot(None))
        .order_by(ProvenanceRecord.seq)
    ).all()
    assert anchored, "no Merkle root was written after two full blocks"
    assert all(seq % interval == 0 for seq in anchored)

    proof_seq = anchored[0]
    proof, root = empty_ledger.inclusion_proof(proof_seq)
    assert len(proof) > 0
    assert len(root) == 64
    assert empty_ledger.verify_record_inclusion(proof_seq) is True


def test_a_block_that_has_not_closed_yet_has_no_root(empty_ledger, db_session) -> None:
    empty_ledger.anchor_events([_custody_event(1)])
    seq = records(db_session)[0].seq

    with pytest.raises(LedgerError, match="has not closed"):
        empty_ledger.inclusion_proof(seq)
    assert empty_ledger.verify_record_inclusion(seq) is False


def test_a_proof_for_a_nonexistent_record_is_an_error(empty_ledger) -> None:
    with pytest.raises(LedgerError, match="no record"):
        empty_ledger.inclusion_proof(999_999_999)
