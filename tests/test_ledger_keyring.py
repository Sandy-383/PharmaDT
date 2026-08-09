"""Key issuance and the public-key registry.

The registry is this project's replacement for Fabric's MSP and channel policy,
so "who is allowed to write" is decided here and nowhere else.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from pharmadt.core.models import Node
from pharmadt.ledger import crypto
from pharmadt.ledger.keyring import (
    NodeKeyring,
    UnauthorisedSigner,
    generate_node_keypair,
    issue_keys,
)

REAL_NODE = "NODE-MFG-01"


@pytest.fixture
def scoped_factory(db_session: Session):
    @contextmanager
    def factory() -> Iterator[Session]:
        yield db_session

    return factory


@pytest.fixture
def isolated_keyring(tmp_path: Path, scoped_factory) -> NodeKeyring:
    return NodeKeyring(keys_dir=tmp_path, session_factory=scoped_factory)


# ── Keypair generation ────────────────────────────────────────────────


def test_a_generated_pair_signs_and_verifies() -> None:
    private_key, public_pem = generate_node_keypair()
    digest = "a" * 64
    signature = crypto.sign(private_key, digest)
    assert crypto.verify(crypto.public_key_from_pem(public_pem), digest, signature)


def test_each_node_gets_a_distinct_key() -> None:
    """Shared keys would make signatures useless for attribution."""
    _, first = generate_node_keypair()
    _, second = generate_node_keypair()
    assert first != second


# ── Issuance ──────────────────────────────────────────────────────────


def test_issuance_writes_a_key_file_and_registers_the_public_half(
    tmp_path: Path, scoped_factory, db_session: Session
) -> None:
    issued = issue_keys(keys_dir=tmp_path, rotate=True, session_factory=scoped_factory)

    assert len(issued) == 12
    for node_id in issued:
        assert (tmp_path / f"{node_id}.pem").exists()

    registered = db_session.scalars(
        select(Node.public_key).where(Node.node_id == REAL_NODE)
    ).one()
    assert registered.startswith("-----BEGIN PUBLIC KEY-----")


def test_issuance_never_writes_a_private_key_into_the_database(
    tmp_path: Path, scoped_factory, db_session: Session
) -> None:
    """The registry is public by design; a leaked private half breaks everything."""
    issue_keys(keys_dir=tmp_path, rotate=True, session_factory=scoped_factory)

    for pem in db_session.scalars(select(Node.public_key)):
        assert "PRIVATE" not in (pem or "")


def test_issuance_skips_nodes_that_already_hold_a_key(
    tmp_path: Path, scoped_factory
) -> None:
    issue_keys(keys_dir=tmp_path, rotate=True, session_factory=scoped_factory)
    assert issue_keys(keys_dir=tmp_path, session_factory=scoped_factory) == []


def test_rotation_replaces_an_existing_key(tmp_path: Path, scoped_factory) -> None:
    issue_keys(keys_dir=tmp_path, rotate=True, session_factory=scoped_factory)
    before = (tmp_path / f"{REAL_NODE}.pem").read_bytes()

    issue_keys(keys_dir=tmp_path, rotate=True, session_factory=scoped_factory)
    assert (tmp_path / f"{REAL_NODE}.pem").read_bytes() != before


# ── Authorisation ─────────────────────────────────────────────────────


def test_a_seeded_node_is_authorised(isolated_keyring: NodeKeyring, scoped_factory) -> None:
    issue_keys(
        keys_dir=isolated_keyring.keys_dir, rotate=True, session_factory=scoped_factory
    )
    isolated_keyring.refresh()
    assert isolated_keyring.is_authorised(REAL_NODE)


def test_an_unknown_node_is_not_authorised(isolated_keyring: NodeKeyring) -> None:
    assert not isolated_keyring.is_authorised("NODE-IMPOSTOR")


def test_signing_for_an_unregistered_node_is_refused(
    isolated_keyring: NodeKeyring,
) -> None:
    """Fails at the point of the mistake, not silently at verification time."""
    with pytest.raises(UnauthorisedSigner, match="registry"):
        isolated_keyring.sign("NODE-IMPOSTOR", "a" * 64)


def test_verifying_an_unknown_signer_returns_false(isolated_keyring: NodeKeyring) -> None:
    assert isolated_keyring.verify("NODE-IMPOSTOR", "a" * 64, "00ff") is False


def test_a_missing_private_key_says_how_to_fix_it(
    isolated_keyring: NodeKeyring, scoped_factory
) -> None:
    issue_keys(
        keys_dir=isolated_keyring.keys_dir, rotate=True, session_factory=scoped_factory
    )
    isolated_keyring.refresh()
    isolated_keyring.key_path(REAL_NODE).unlink()

    with pytest.raises(UnauthorisedSigner, match="keyring"):
        isolated_keyring.sign(REAL_NODE, "a" * 64)


# ── Signing round trip ────────────────────────────────────────────────


def test_a_keyring_verifies_what_it_signed(
    isolated_keyring: NodeKeyring, scoped_factory
) -> None:
    issue_keys(
        keys_dir=isolated_keyring.keys_dir, rotate=True, session_factory=scoped_factory
    )
    isolated_keyring.refresh()

    digest = "b" * 64
    assert isolated_keyring.verify(REAL_NODE, digest, isolated_keyring.sign(REAL_NODE, digest))


def test_one_nodes_signature_does_not_verify_as_another(
    isolated_keyring: NodeKeyring, scoped_factory
) -> None:
    """Non-repudiation: a record names the node that actually signed it."""
    issue_keys(
        keys_dir=isolated_keyring.keys_dir, rotate=True, session_factory=scoped_factory
    )
    isolated_keyring.refresh()

    digest = "c" * 64
    signature = isolated_keyring.sign(REAL_NODE, digest)
    assert not isolated_keyring.verify("NODE-PH-01", digest, signature)


def test_refresh_picks_up_a_newly_issued_key(
    isolated_keyring: NodeKeyring, scoped_factory
) -> None:
    """The registry is cached; verification checks it once per record."""
    assert isolated_keyring.registry() == {} or REAL_NODE in isolated_keyring.registry()

    issue_keys(
        keys_dir=isolated_keyring.keys_dir, rotate=True, session_factory=scoped_factory
    )
    isolated_keyring.refresh()
    assert REAL_NODE in isolated_keyring.registry()
