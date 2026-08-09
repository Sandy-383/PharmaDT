"""The hash-chained provenance ledger — this project's replacement for Fabric.

Each record binds its own content and its predecessor's hash into a SHA-256
digest, which is then signed with the acting node's P-256 private key. Altering
any record changes its digest and orphans everything after it, so tampering is
detectable without trusting the application that wrote the rows.

Two ways in:

* :meth:`HashChainLedger.record_event` appends one record in its own
  transaction, serialised against concurrent writers by an advisory lock.
* :meth:`HashChainLedger.anchor_events` builds a whole run's worth of chain in
  memory and inserts it once. The simulation uses this — NFR-01 asks for 1000
  steps per second, and a round trip per event would put the ceiling three
  orders of magnitude below that.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from pharmadt.config import settings
from pharmadt.core.db import session_scope
from pharmadt.core.events import LEDGER_EVENT_TYPES, Event, EventType
from pharmadt.core.interfaces import ProvenanceLedger
from pharmadt.core.models import Batch, ProvenanceRecord, compute_batch_fingerprint
from pharmadt.core.schemas import ChainVerificationResult
from pharmadt.ledger import merkle
from pharmadt.ledger.crypto import GENESIS_PREV_HASH, compute_record_hash
from pharmadt.ledger.keyring import NodeKeyring, UnauthorisedSigner

logger = logging.getLogger(__name__)

#: Arbitrary but fixed key for the Postgres advisory lock that serialises
#: appends. Two concurrent writers reading the same tip would fork the chain.
APPEND_LOCK_KEY = 0x5048524D  # "PHRM"

#: Which node signs which event. The signer is always the node with physical
#: custody at the moment the event occurs — that is what makes the signature
#: mean something rather than being a rubber stamp from a central authority.
SIGNER_SIDE: dict[EventType, str] = {
    EventType.BATCH_CREATED: "to_node",
    EventType.SHIPMENT_DISPATCHED: "from_node",
    EventType.SHIPMENT_RECEIVED: "to_node",
    EventType.COLD_CHAIN_EXCURSION: "from_node",
    EventType.REDISTRIBUTION: "from_node",
    EventType.DISPENSED: "from_node",
    EventType.RECALLED: "from_node",
}


class LedgerError(Exception):
    """Raised when a record cannot be appended."""


def default_signer(event: Event) -> str | None:
    """The node whose key should sign ``event``, or None if it cannot be determined."""
    side = SIGNER_SIDE.get(event.event_type)
    if side is None:
        return None
    return getattr(event, side) or event.from_node or event.to_node


def _record_fields(
    batch_id: str,
    event_type: str,
    from_node: str | None,
    to_node: str | None,
    payload: Mapping[str, Any],
    sim_day: int,
    signer_node: str,
) -> dict[str, Any]:
    """The exact field set covered by a record's hash.

    ``dict(payload)`` is not cosmetic: the twin hands out ``MappingProxyType``
    payloads, which ``json.dumps`` cannot serialise and would silently coerce to
    a repr string via ``default=str`` — producing a digest that no verifier
    reading plain JSONB back from Postgres could ever reproduce.

    ``recorded_at`` is deliberately outside the hash. It is assigned by the
    database, and a TIMESTAMPTZ is not guaranteed to render back to the same
    string it went in as, which would fail verification on records nobody had
    touched. The trigger already blocks any UPDATE to it.
    """
    return {
        "batch_id": batch_id,
        "event_type": str(event_type),
        "from_node": from_node,
        "to_node": to_node,
        "payload": dict(payload),
        "sim_day": sim_day,
        "signer_node": signer_node,
    }


class HashChainLedger(ProvenanceLedger):
    """Append-only, signed, hash-chained provenance over PostgreSQL."""

    def __init__(
        self,
        keyring: NodeKeyring | None = None,
        session_factory: Callable[[], AbstractContextManager[Session]] | None = None,
    ) -> None:
        self.keyring = keyring if keyring is not None else NodeKeyring()
        self.last_result: ChainVerificationResult | None = None
        # Injectable so tests can run against a transaction that is rolled back
        # afterwards. An append-only table cannot be cleaned up after the fact,
        # so without this every test run would permanently grow the real ledger.
        self._session = session_factory if session_factory is not None else session_scope

    # ── Appending ─────────────────────────────────────────────────────

    def record_event(
        self,
        batch_id: str,
        event_type: EventType,
        from_node: str | None,
        to_node: str | None,
        payload: Mapping[str, Any],
        signer_node: str,
        *,
        sim_day: int = 0,
    ) -> str:
        """Append one signed record and return its ``record_hash``.

        ``sim_day`` is an addition to the interface as specified. The specified
        table declares it NOT NULL and the specified signature has no way to
        supply it; smuggling it through the payload would leave a first-class,
        queried column buried in an unindexable JSONB key.
        """
        if event_type not in LEDGER_EVENT_TYPES:
            raise LedgerError(
                f"{event_type} is simulation telemetry, not a custody event. "
                "Anchoring it would dilute the trail it sits next to."
            )
        if not self.keyring.is_authorised(signer_node):
            raise UnauthorisedSigner(
                f"{signer_node} is not in the public-key registry and may not write"
            )

        with self._session() as session:
            # Serialise appends: two writers reading the same tip would each
            # link to it and fork the chain into two valid-looking branches.
            session.execute(
                text("SELECT pg_advisory_xact_lock(:key)"), {"key": APPEND_LOCK_KEY}
            )

            prev_hash = self._tip(session)
            fields = _record_fields(
                batch_id, event_type, from_node, to_node, payload, sim_day, signer_node
            )
            record_hash = compute_record_hash(fields, prev_hash)
            signature = self.keyring.sign(signer_node, record_hash)

            session.add(
                ProvenanceRecord(
                    batch_id=batch_id,
                    event_type=str(event_type),
                    from_node=from_node,
                    to_node=to_node,
                    payload=dict(payload),
                    sim_day=sim_day,
                    prev_hash=prev_hash,
                    record_hash=record_hash,
                    signer_node=signer_node,
                    signature=signature,
                )
            )
            session.flush()
            self._anchor_completed_blocks(session)

        return record_hash

    def anchor_events(self, events: Sequence[Event], *, skip_unsigned: bool = True) -> int:
        """Append every ledger-anchored event in ``events``. Returns the count.

        The chain is built in memory from the current tip and written in one
        pass, so a full simulation costs a single transaction rather than one
        per event.
        """
        anchorable = [e for e in events if e.is_ledger_anchored and e.batch_id]
        if not anchorable:
            return 0

        rows: list[dict[str, Any]] = []
        with self._session() as session:
            session.execute(
                text("SELECT pg_advisory_xact_lock(:key)"), {"key": APPEND_LOCK_KEY}
            )
            prev_hash = self._tip(session)

            for event in anchorable:
                signer = default_signer(event)
                if signer is None or not self.keyring.is_authorised(signer):
                    if skip_unsigned:
                        logger.warning(
                            "skipping %s for %s: no authorised signer (%s)",
                            event.event_type,
                            event.batch_id,
                            signer,
                        )
                        continue
                    raise UnauthorisedSigner(f"{signer} may not write")

                fields = _record_fields(
                    event.batch_id,
                    event.event_type,
                    event.from_node,
                    event.to_node,
                    event.payload,
                    event.sim_day,
                    signer,
                )
                record_hash = compute_record_hash(fields, prev_hash)
                rows.append(
                    {
                        **fields,
                        "prev_hash": prev_hash,
                        "record_hash": record_hash,
                        "signature": self.keyring.sign(signer, record_hash),
                    }
                )
                prev_hash = record_hash

            if rows:
                session.bulk_insert_mappings(ProvenanceRecord, rows)
                session.flush()
                self._anchor_completed_blocks(session)

        return len(rows)

    # ── Reading ───────────────────────────────────────────────────────

    def get_provenance(self, batch_id: str) -> list[dict[str, Any]]:
        """Every record for ``batch_id`` in ``seq`` order — the full trace."""
        with self._session() as session:
            records = session.scalars(
                select(ProvenanceRecord)
                .where(ProvenanceRecord.batch_id == batch_id)
                .order_by(ProvenanceRecord.seq)
            ).all()
            return [self._to_dict(r) for r in records]

    def height(self) -> int:
        with self._session() as session:
            return int(session.scalar(select(func.count()).select_from(ProvenanceRecord)) or 0)

    def tip(self) -> str:
        with self._session() as session:
            return self._tip(session)

    # ── Verification ──────────────────────────────────────────────────

    def verify_chain(self, start: int | None = None, end: int | None = None) -> bool:
        """Recompute every hash and signature over ``[start, end]``.

        Returns False at the first break; the offending ``seq`` is logged and
        left on :attr:`last_result` for the caller to render.
        """
        self.last_result = self.verify_chain_detailed(start, end)
        if not self.last_result.valid:
            logger.error(
                "chain verification failed at seq %s: %s",
                self.last_result.broken_at_seq,
                self.last_result.reason,
            )
        return self.last_result.valid

    def verify_chain_detailed(
        self, start: int | None = None, end: int | None = None
    ) -> ChainVerificationResult:
        """The same walk as :meth:`verify_chain`, reporting where and why."""
        with self._session() as session:
            query = select(ProvenanceRecord).order_by(ProvenanceRecord.seq)
            if start is not None:
                query = query.where(ProvenanceRecord.seq >= start)
            if end is not None:
                query = query.where(ProvenanceRecord.seq <= end)
            records = session.scalars(query).all()

            checked = 0
            # A partial range links to whatever its predecessor recorded, so the
            # expected prev_hash is only known a priori for a walk from genesis.
            expected_prev = GENESIS_PREV_HASH if not records or records[0].seq == 1 else None

            for record in records:
                checked += 1

                if expected_prev is not None and record.prev_hash != expected_prev:
                    return ChainVerificationResult(
                        valid=False,
                        records_checked=checked,
                        broken_at_seq=record.seq,
                        reason=(
                            f"prev_hash {record.prev_hash[:12]}... does not link to "
                            f"{expected_prev[:12]}...; a record was inserted or removed"
                        ),
                    )

                fields = _record_fields(
                    record.batch_id,
                    record.event_type,
                    record.from_node,
                    record.to_node,
                    record.payload,
                    record.sim_day,
                    record.signer_node,
                )
                recomputed = compute_record_hash(fields, record.prev_hash)
                if recomputed != record.record_hash:
                    return ChainVerificationResult(
                        valid=False,
                        records_checked=checked,
                        broken_at_seq=record.seq,
                        reason="record_hash does not match its contents; the record was edited",
                    )

                if not self.keyring.verify(
                    record.signer_node, record.record_hash, record.signature
                ):
                    return ChainVerificationResult(
                        valid=False,
                        records_checked=checked,
                        broken_at_seq=record.seq,
                        reason=(
                            f"signature does not verify against {record.signer_node}'s "
                            "registered public key"
                        ),
                    )

                expected_prev = record.record_hash

            return ChainVerificationResult(valid=True, records_checked=checked)

    def verify_batch_fingerprint(self, batch_id: str, presented: str) -> bool:
        """Recompute a batch's identity fingerprint and compare.

        A mismatch is the anti-counterfeit signal the Anomaly Agent consumes in
        Stage 10. An unknown batch id is also a failure — a product presenting
        an identifier the ledger has never seen is exactly as suspect as one
        presenting a wrong hash.
        """
        with self._session() as session:
            batch = session.get(Batch, batch_id)
            if batch is None:
                return False
            expected = compute_batch_fingerprint(
                batch.batch_id,
                batch.drug_id,
                batch.manufacturer_id,
                batch.mfg_date,
                batch.expiry_date,
                batch.quantity,
            )
        return expected == presented

    # ── Merkle anchoring ──────────────────────────────────────────────

    def inclusion_proof(self, seq: int) -> tuple[list[merkle.ProofStep], str]:
        """Audit path proving record ``seq`` sits under its block's root."""
        hashes, root, index = self._block_of(seq)
        if root is None:
            raise LedgerError(
                f"seq {seq} is in a block that has not closed yet; its root is "
                f"written on the {settings.merkle_interval}th record"
            )
        return merkle.inclusion_proof(hashes, index), root

    def verify_record_inclusion(self, seq: int) -> bool:
        """Check one record against its stored Merkle root."""
        hashes, root, index = self._block_of(seq)
        if root is None:
            return False
        proof = merkle.inclusion_proof(hashes, index)
        return merkle.verify_inclusion(hashes[index], proof, root)

    def _block_of(self, seq: int) -> tuple[list[str], str | None, int]:
        """Record hashes of ``seq``'s Merkle block, its stored root, and the index."""
        interval = settings.merkle_interval
        block_start = ((seq - 1) // interval) * interval + 1
        block_end = block_start + interval - 1

        with self._session() as session:
            rows = session.execute(
                select(ProvenanceRecord.seq, ProvenanceRecord.record_hash)
                .where(ProvenanceRecord.seq.between(block_start, block_end))
                .order_by(ProvenanceRecord.seq)
            ).all()
            root = session.scalar(
                select(ProvenanceRecord.merkle_root).where(ProvenanceRecord.seq == block_end)
            )

        if not rows:
            raise LedgerError(f"no record at seq {seq}")
        hashes = [h for _, h in rows]
        try:
            index = [s for s, _ in rows].index(seq)
        except ValueError as exc:
            raise LedgerError(f"no record at seq {seq}") from exc
        return hashes, root, index

    def _anchor_completed_blocks(self, session) -> None:
        """Write a Merkle root onto the last record of every newly closed block.

        Uses raw SQL because the append-only trigger blocks ORM updates too —
        the root has to be set with the trigger briefly out of the way, inside
        the same transaction that inserted the block.
        """
        interval = settings.merkle_interval
        highest = session.scalar(select(func.max(ProvenanceRecord.seq))) or 0
        complete_blocks = highest // interval
        if complete_blocks == 0:
            return

        pending = session.scalars(
            select(ProvenanceRecord.seq)
            .where(
                ProvenanceRecord.seq <= complete_blocks * interval,
                ProvenanceRecord.seq % interval == 0,
                ProvenanceRecord.merkle_root.is_(None),
            )
            .order_by(ProvenanceRecord.seq)
        ).all()
        if not pending:
            return

        for block_end in pending:
            hashes = session.scalars(
                select(ProvenanceRecord.record_hash)
                .where(ProvenanceRecord.seq.between(block_end - interval + 1, block_end))
                .order_by(ProvenanceRecord.seq)
            ).all()
            root = merkle.merkle_root(list(hashes))
            session.execute(
                text(
                    "ALTER TABLE provenance_records DISABLE TRIGGER no_update;"
                    " UPDATE provenance_records SET merkle_root = :root WHERE seq = :seq;"
                    " ALTER TABLE provenance_records ENABLE TRIGGER no_update;"
                ),
                {"root": root, "seq": block_end},
            )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _tip(session) -> str:
        """The newest record's hash, or the genesis anchor if the chain is empty."""
        tip = session.scalar(
            select(ProvenanceRecord.record_hash)
            .order_by(ProvenanceRecord.seq.desc())
            .limit(1)
        )
        return tip or GENESIS_PREV_HASH

    @staticmethod
    def _to_dict(record: ProvenanceRecord) -> dict[str, Any]:
        return {
            "seq": record.seq,
            "batch_id": record.batch_id,
            "event_type": record.event_type,
            "from_node": record.from_node,
            "to_node": record.to_node,
            "payload": record.payload,
            "sim_day": record.sim_day,
            "recorded_at": record.recorded_at.isoformat() if record.recorded_at else None,
            "prev_hash": record.prev_hash,
            "record_hash": record.record_hash,
            "signer_node": record.signer_node,
            "signature": record.signature,
            "merkle_root": record.merkle_root,
        }
