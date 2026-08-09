"""Stage 4: provenance ledger

Creates the append-only, hash-chained provenance table and the triggers that
make "append-only" a property of the database rather than a promise made by the
application. The DDL is inlined rather than read from ledger/schema.sql, because
a migration is immutable history and must not depend on a file that can change
underneath it; schema.sql is the readable reference and a test asserts the two
still agree.

Revision ID: f7e366a62d5b
Revises: b48094db5775
"""

from collections.abc import Sequence

from alembic import op

revision: str = "f7e366a62d5b"
down_revision: str | Sequence[str] | None = "b48094db5775"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE provenance_records (
            seq           BIGSERIAL PRIMARY KEY,
            batch_id      TEXT NOT NULL,
            event_type    TEXT NOT NULL,
            from_node     TEXT,
            to_node       TEXT,
            payload       JSONB NOT NULL,
            sim_day       INTEGER NOT NULL,
            recorded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            prev_hash     CHAR(64) NOT NULL,
            record_hash   CHAR(64) NOT NULL UNIQUE,
            signer_node   TEXT NOT NULL,
            signature     TEXT NOT NULL,
            merkle_root   CHAR(64)
        );
        """
    )
    op.execute("CREATE INDEX idx_prov_batch ON provenance_records(batch_id);")

    op.execute(
        """
        CREATE OR REPLACE FUNCTION block_mutation() RETURNS trigger AS $$
        BEGIN RAISE EXCEPTION 'provenance_records is append-only'; END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER no_update BEFORE UPDATE ON provenance_records
            FOR EACH ROW EXECUTE FUNCTION block_mutation();
        """
    )
    op.execute(
        """
        CREATE TRIGGER no_delete BEFORE DELETE ON provenance_records
            FOR EACH ROW EXECUTE FUNCTION block_mutation();
        """
    )
    # Row-level DELETE triggers do not fire for TRUNCATE, so without this the
    # append-only guarantee has a hole wide enough to erase the whole ledger in
    # one statement. TRUNCATE triggers are statement-level by definition.
    op.execute(
        """
        CREATE TRIGGER no_truncate BEFORE TRUNCATE ON provenance_records
            FOR EACH STATEMENT EXECUTE FUNCTION block_mutation();
        """
    )


def downgrade() -> None:
    # Dropping the table takes its triggers with it; the function is shared by
    # nothing else, so it goes too.
    op.execute("DROP TABLE IF EXISTS provenance_records;")
    op.execute("DROP FUNCTION IF EXISTS block_mutation();")
