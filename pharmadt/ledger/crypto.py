"""Canonical serialisation, SHA-256 hashing, and ECDSA sign/verify.

These are the primitives the whole tamper-evidence argument rests on, so each
one is deliberately small enough to read in full.

The central requirement is determinism: the same logical record must serialise
to the same bytes on every machine, in every process, forever. Anything less and
``verify_chain`` fails intermittently on records nobody touched, which is worse
than no verification at all — it trains you to ignore the alarm.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ec import (
    EllipticCurvePrivateKey,
    EllipticCurvePublicKey,
)

#: The chain's anchor. The first record has no predecessor, so it links to a
#: fixed all-zero hash rather than to NULL — a nullable prev_hash would make
#: "genesis" and "someone deleted the parent" indistinguishable.
GENESIS_PREV_HASH = "0" * 64

#: Curve mandated by NFR-04. P-256 is the ECDSA curve X.509/MSP would have used.
CURVE = ec.SECP256R1


def canonical(obj: Mapping[str, Any]) -> bytes:
    """Serialise a mapping to deterministic bytes.

    ``sort_keys=True`` is non-negotiable. Python preserves insertion order and
    Postgres JSONB does not preserve it at all, so without sorting the same
    record hashes differently depending on whether it was just built or just
    read back — the classic intermittent chain failure.

    ``separators`` strips the whitespace ``json.dumps`` would otherwise insert,
    and ``default=str`` gives dates and Decimals a stable textual form.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_record_hash(payload_fields: Mapping[str, Any], prev_hash: str) -> str:
    """Hash a record's content together with its predecessor's hash.

    Binding ``prev_hash`` into the digest is what makes the chain a chain:
    altering any record changes its hash, which orphans every record after it.
    """
    return sha256_hex(canonical(payload_fields) + prev_hash.encode())


# ── ECDSA (NIST P-256) ────────────────────────────────────────────────


def generate_private_key() -> EllipticCurvePrivateKey:
    return ec.generate_private_key(CURVE())


def sign(private_key: EllipticCurvePrivateKey, record_hash: str) -> str:
    """Sign a record hash. Returns a hex-encoded DER signature.

    Note that ECDSA draws a random nonce, so signing the same hash twice yields
    different bytes. That is correct and expected — signatures are verified, not
    compared.
    """
    return private_key.sign(record_hash.encode(), ec.ECDSA(hashes.SHA256())).hex()


def verify(public_key: EllipticCurvePublicKey, record_hash: str, signature_hex: str) -> bool:
    """Check a signature against a record hash. Never raises."""
    try:
        public_key.verify(
            bytes.fromhex(signature_hex),
            record_hash.encode(),
            ec.ECDSA(hashes.SHA256()),
        )
    except (InvalidSignature, ValueError):
        # ValueError covers malformed hex and malformed DER, which a forger
        # supplies at least as often as a well-formed but wrong signature.
        return False
    return True


# ── Key serialisation ─────────────────────────────────────────────────


def public_key_to_pem(public_key: EllipticCurvePublicKey) -> str:
    return public_key.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()


def public_key_from_pem(pem: str) -> EllipticCurvePublicKey:
    key = serialization.load_pem_public_key(pem.encode())
    if not isinstance(key, EllipticCurvePublicKey):
        raise ValueError("not an elliptic-curve public key")
    return key


def private_key_to_pem(private_key: EllipticCurvePrivateKey) -> bytes:
    # Unencrypted: this is a simulation and the report says plainly that
    # production keys belong in an HSM or KMS rather than on disk.
    return private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def private_key_from_pem(pem: bytes) -> EllipticCurvePrivateKey:
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, EllipticCurvePrivateKey):
        raise ValueError("not an elliptic-curve private key")
    return key
