"""Canonical serialisation, hashing, and ECDSA signatures.

Determinism is the property under test throughout. A serialiser that is merely
*usually* stable produces a chain that fails verification on records nobody
touched, which is worse than no verification — it teaches you to ignore alarms.
"""

from __future__ import annotations

from datetime import date

import pytest

from pharmadt.ledger.crypto import (
    GENESIS_PREV_HASH,
    canonical,
    compute_record_hash,
    generate_private_key,
    private_key_from_pem,
    private_key_to_pem,
    public_key_from_pem,
    public_key_to_pem,
    sha256_hex,
    sign,
    verify,
)

# ── Canonical serialisation ───────────────────────────────────────────


def test_key_order_does_not_change_the_bytes() -> None:
    """Postgres JSONB does not preserve insertion order; sorting makes that moot."""
    assert canonical({"a": 1, "b": 2}) == canonical({"b": 2, "a": 1})


def test_nested_key_order_does_not_change_the_bytes() -> None:
    assert canonical({"o": {"x": 1, "y": 2}}) == canonical({"o": {"y": 2, "x": 1}})


def test_serialisation_carries_no_incidental_whitespace() -> None:
    assert canonical({"a": 1, "b": 2}) == b'{"a":1,"b":2}'


def test_dates_serialise_to_a_stable_string() -> None:
    assert canonical({"d": date(2026, 1, 1)}) == b'{"d":"2026-01-01"}'


def test_distinct_content_serialises_differently() -> None:
    assert canonical({"a": 1}) != canonical({"a": 2})


# ── Hashing ───────────────────────────────────────────────────────────


def test_sha256_matches_the_known_digest_of_empty_input() -> None:
    assert sha256_hex(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_record_hash_is_hex_and_full_length() -> None:
    digest = compute_record_hash({"a": 1}, GENESIS_PREV_HASH)
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_record_hash_is_stable_for_the_same_input() -> None:
    a = compute_record_hash({"x": 1, "y": "z"}, GENESIS_PREV_HASH)
    b = compute_record_hash({"y": "z", "x": 1}, GENESIS_PREV_HASH)
    assert a == b


def test_editing_content_changes_the_hash() -> None:
    """The entire tamper-evidence claim reduces to this."""
    before = compute_record_hash({"quantity": 100}, GENESIS_PREV_HASH)
    after = compute_record_hash({"quantity": 999999}, GENESIS_PREV_HASH)
    assert before != after


def test_the_same_content_under_a_different_parent_hashes_differently() -> None:
    """This is what makes the chain a chain rather than a list."""
    fields = {"quantity": 100}
    assert compute_record_hash(fields, GENESIS_PREV_HASH) != compute_record_hash(
        fields, "a" * 64
    )


def test_genesis_anchor_is_sixty_four_zeros() -> None:
    """A nullable prev_hash would make genesis and a deleted parent identical."""
    assert GENESIS_PREV_HASH == "0" * 64


# ── ECDSA ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def keypair():
    private_key = generate_private_key()
    return private_key, private_key.public_key()


def test_a_signature_verifies_against_its_own_key(keypair) -> None:
    private_key, public_key = keypair
    digest = "a" * 64
    assert verify(public_key, digest, sign(private_key, digest))


def test_a_signature_fails_against_a_different_key(keypair) -> None:
    """The registry is the permissioning layer; a stranger's key must not pass."""
    private_key, _ = keypair
    digest = "a" * 64
    stranger = generate_private_key().public_key()
    assert not verify(stranger, digest, sign(private_key, digest))


def test_a_signature_does_not_cover_a_different_digest(keypair) -> None:
    private_key, public_key = keypair
    assert not verify(public_key, "b" * 64, sign(private_key, "a" * 64))


@pytest.mark.parametrize(
    "bad_signature",
    ["", "zz", "not-hex", "ab", "00" * 70],
    ids=["empty", "odd-length", "non-hex", "too-short", "wrong-der"],
)
def test_malformed_signatures_are_rejected_without_raising(keypair, bad_signature) -> None:
    """A forger supplies garbage at least as often as a well-formed wrong key."""
    _, public_key = keypair
    assert verify(public_key, "a" * 64, bad_signature) is False


def test_signing_twice_gives_different_bytes_that_both_verify(keypair) -> None:
    """ECDSA draws a random nonce. Signatures are verified, never compared."""
    private_key, public_key = keypair
    digest = "a" * 64
    first, second = sign(private_key, digest), sign(private_key, digest)

    assert first != second
    assert verify(public_key, digest, first)
    assert verify(public_key, digest, second)


# ── Key serialisation ─────────────────────────────────────────────────


def test_public_key_survives_a_pem_round_trip(keypair) -> None:
    private_key, public_key = keypair
    restored = public_key_from_pem(public_key_to_pem(public_key))
    assert verify(restored, "a" * 64, sign(private_key, "a" * 64))


def test_private_key_survives_a_pem_round_trip(keypair) -> None:
    private_key, public_key = keypair
    restored = private_key_from_pem(private_key_to_pem(private_key))
    assert verify(public_key, "a" * 64, sign(restored, "a" * 64))


def test_public_pem_is_a_public_key_and_carries_no_secret(keypair) -> None:
    _, public_key = keypair
    pem = public_key_to_pem(public_key)
    assert pem.startswith("-----BEGIN PUBLIC KEY-----")
    assert "PRIVATE" not in pem


def test_a_non_ec_pem_is_rejected() -> None:
    with pytest.raises(ValueError):
        public_key_from_pem("-----BEGIN PUBLIC KEY-----\nnonsense\n-----END PUBLIC KEY-----")
