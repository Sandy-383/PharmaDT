-- Provenance ledger schema — reference copy.
--
-- The authoritative, executable version of this DDL lives in the Alembic
-- migration that introduced it (alembic/versions/*_stage_4_provenance_ledger.py).
-- Migrations are immutable history and must not read from a file that can be
-- edited afterwards, so this file is kept as the readable reference and a test
-- (tests/test_ledger_schema.py) asserts the live database still matches it.
--
-- Apply with `make migrate`, not by running this file directly.

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

-- The manufacturer-to-patient trace is a lookup by batch, so it gets an index.
CREATE INDEX idx_prov_batch ON provenance_records(batch_id);

-- Immutability is enforced at the database layer, not in application code.
-- This is the load-bearing claim: the audit trail holds even against a bug,
-- a careless migration, or a developer with a psql prompt.
CREATE OR REPLACE FUNCTION block_mutation() RETURNS trigger AS $$
BEGIN RAISE EXCEPTION 'provenance_records is append-only'; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER no_update BEFORE UPDATE ON provenance_records
    FOR EACH ROW EXECUTE FUNCTION block_mutation();

CREATE TRIGGER no_delete BEFORE DELETE ON provenance_records
    FOR EACH ROW EXECUTE FUNCTION block_mutation();

-- Row-level DELETE triggers do not fire for TRUNCATE. Without a statement-level
-- guard the whole ledger could be erased in one statement while both triggers
-- above sat and watched.
CREATE TRIGGER no_truncate BEFORE TRUNCATE ON provenance_records
    FOR EACH STATEMENT EXECUTE FUNCTION block_mutation();

-- Demonstrating tamper-evidence (Stage 4 Definition of Done) requires getting
-- past the trigger first, which is itself the point:
--
--   ALTER TABLE provenance_records DISABLE TRIGGER no_update;
--   UPDATE provenance_records SET payload = '{"quantity": 999999}' WHERE seq = 5;
--   ALTER TABLE provenance_records ENABLE TRIGGER no_update;
--
-- verify_chain() then returns False and names seq 5. Run `make tamper-demo`.
