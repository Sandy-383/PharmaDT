"""The Stage 4 demonstration: prove the ledger is tamper-evident.

Walks the whole Definition of Done end to end — the trigger refuses mutation,
the chain verifies, a batch's trace reads manufacturer to patient, an inclusion
proof checks against its Merkle root, an edited record is caught and named, and
a forged fingerprint is rejected.

The tamper step restores what it changed, so this is safe and repeatable.

Usage::

    make tamper-demo
    python -m pharmadt.ledger.demo --seq 200
"""

from __future__ import annotations

import argparse
import json

from sqlalchemy import func, select, text

from pharmadt.core.db import session_scope
from pharmadt.core.models import Batch, ProvenanceRecord
from pharmadt.ledger.chain import HashChainLedger

RULE = "-" * 72


def _heading(number: int, title: str) -> None:
    print(f"\n{RULE}\n{number}. {title}\n{RULE}")


def _trigger_blocks_mutation() -> None:
    _heading(1, "The database itself refuses to mutate the ledger")
    try:
        with session_scope() as session:
            session.execute(
                text("UPDATE provenance_records SET sim_day = sim_day + 1 WHERE seq = 1")
            )
    except Exception as exc:  # noqa: BLE001 - the exact driver error is the evidence
        message = str(exc).splitlines()[0]
        print(f"  UPDATE rejected: {message}")
    else:
        print("  UPDATE SUCCEEDED -- the append-only trigger is missing!")
        return

    try:
        with session_scope() as session:
            session.execute(text("DELETE FROM provenance_records WHERE seq = 1"))
    except Exception as exc:  # noqa: BLE001
        print(f"  DELETE rejected: {str(exc).splitlines()[0]}")
    else:
        print("  DELETE SUCCEEDED -- the append-only trigger is missing!")

    print("\n  Immutability is a property of the database, not a promise made by")
    print("  the application. It holds even against a bug or a psql prompt.")


def _verify(ledger: HashChainLedger, label: str) -> None:
    result = ledger.verify_chain_detailed()
    if result.valid:
        print(f"  {label}: VALID over {result.records_checked:,} records")
    else:
        print(f"  {label}: BROKEN at seq {result.broken_at_seq}")
        print(f"           reason: {result.reason}")


def _show_trace(ledger: HashChainLedger) -> str | None:
    _heading(3, "Manufacturer-to-patient trace for one batch")
    # Pick the batch with the widest variety of custody events. Choosing the
    # first batch by id tends to land on opening stock that was only ever
    # dispensed, which shows none of the handoffs this trace exists to prove.
    with session_scope() as session:
        batch_id = session.scalar(
            select(ProvenanceRecord.batch_id)
            .group_by(ProvenanceRecord.batch_id)
            .order_by(
                func.count(func.distinct(ProvenanceRecord.event_type)).desc(),
                ProvenanceRecord.batch_id,
            )
            .limit(1)
        )
    if batch_id is None:
        print("  no records")
        return None

    trace = ledger.get_provenance(batch_id)
    print(f"  {batch_id} -- {len(trace)} custody events\n")
    print(f"  {'seq':>6}  {'day':>4}  {'event':<22} {'from':<12} {'to':<12} signer")
    for record in trace[:12]:
        print(
            f"  {record['seq']:>6}  {record['sim_day']:>4}  {record['event_type']:<22} "
            f"{(record['from_node'] or '-'):<12} {(record['to_node'] or '-'):<12} "
            f"{record['signer_node']}"
        )
    if len(trace) > 12:
        print(f"  ... {len(trace) - 12} more")
    return batch_id


def _show_inclusion_proof(ledger: HashChainLedger, seq: int) -> None:
    _heading(4, "Merkle inclusion proof")
    try:
        proof, root = ledger.inclusion_proof(seq)
    except Exception as exc:  # noqa: BLE001
        print(f"  seq {seq}: {exc}")
        return

    print(f"  Proving seq {seq} belongs under root {root[:24]}...")
    print(f"  Audit path is {len(proof)} sibling hashes, not the whole block:")
    for sibling, side in proof:
        print(f"    {side}  {sibling[:32]}...")
    print(f"\n  verify_inclusion -> {ledger.verify_record_inclusion(seq)}")


def _tamper(ledger: HashChainLedger, seq: int) -> None:
    _heading(5, f"Tampering with record {seq}")

    with session_scope() as session:
        original = session.scalar(
            select(ProvenanceRecord.payload).where(ProvenanceRecord.seq == seq)
        )
    if original is None:
        print(f"  no record at seq {seq}")
        return

    forged = {**original, "quantity": 999_999}
    print(f"  before: {json.dumps(original, sort_keys=True)[:70]}")
    print(f"  after:  {json.dumps(forged, sort_keys=True)[:70]}")
    print("\n  Getting past the trigger is itself part of the story -- an attacker")
    print("  must first disable a database-level control:\n")
    print("    ALTER TABLE provenance_records DISABLE TRIGGER no_update;")

    with session_scope() as session:
        session.execute(text("ALTER TABLE provenance_records DISABLE TRIGGER no_update"))
        session.execute(
            text("UPDATE provenance_records SET payload = CAST(:p AS jsonb) WHERE seq = :s"),
            {"p": json.dumps(forged), "s": seq},
        )
        session.execute(text("ALTER TABLE provenance_records ENABLE TRIGGER no_update"))

    print()
    _verify(ledger, "verify_chain")
    print("\n  The hash chain caught an edit the trigger could not prevent, and")
    print("  named the exact record. Restoring it now.\n")

    with session_scope() as session:
        session.execute(text("ALTER TABLE provenance_records DISABLE TRIGGER no_update"))
        session.execute(
            text("UPDATE provenance_records SET payload = CAST(:p AS jsonb) WHERE seq = :s"),
            {"p": json.dumps(original), "s": seq},
        )
        session.execute(text("ALTER TABLE provenance_records ENABLE TRIGGER no_update"))

    _verify(ledger, "verify_chain after restore")


def _counterfeit_check(ledger: HashChainLedger) -> None:
    _heading(6, "Anti-counterfeit fingerprint check")
    with session_scope() as session:
        batch = session.scalars(select(Batch).order_by(Batch.batch_id).limit(1)).first()
    if batch is None:
        print("  no batches")
        return

    genuine = batch.batch_fingerprint
    forged = "0" * 64

    print(f"  batch {batch.batch_id}")
    print(f"    genuine fingerprint -> {ledger.verify_batch_fingerprint(batch.batch_id, genuine)}")
    print(f"    forged  fingerprint -> {ledger.verify_batch_fingerprint(batch.batch_id, forged)}"
          "   (counterfeit detected)")
    print(f"    unknown batch id    -> {ledger.verify_batch_fingerprint('BATCH-FAKE', genuine)}"
          "   (unknown product is equally suspect)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate ledger tamper-evidence.")
    parser.add_argument("--seq", type=int, default=None, help="record to tamper with")
    args = parser.parse_args()

    ledger = HashChainLedger()
    height = ledger.height()
    if height == 0:
        raise SystemExit(
            "The ledger is empty. Run `make sim-anchor` first to populate it."
        )

    print(f"\nProvenance ledger: {height:,} records, tip {ledger.tip()[:24]}...")

    _trigger_blocks_mutation()

    _heading(2, "Verifying the untouched chain")
    _verify(ledger, "verify_chain")

    _show_trace(ledger)
    _show_inclusion_proof(ledger, seq=min(5, height))
    _tamper(ledger, seq=args.seq if args.seq is not None else max(1, height // 2))
    _counterfeit_check(ledger)
    print()


if __name__ == "__main__":
    main()
