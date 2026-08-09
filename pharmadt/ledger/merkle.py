"""Merkle tree over record hashes, with O(log n) inclusion proofs.

The hash chain already proves that nothing changed, but verifying it costs a
walk over every record. Merkle anchoring lets a single record be proved present
in a block of 64 with six sibling hashes instead — the difference between an
auditor checking one batch's history and re-verifying the whole ledger.

This follows RFC 6962 (Certificate Transparency) rather than Bitcoin's tree.
Bitcoin duplicates the final leaf when a level has an odd number of nodes, which
makes some distinct leaf sets produce the same root (CVE-2012-2459). RFC 6962
splits at the largest power of two instead and domain-separates leaves (0x00)
from internal nodes (0x01), so no leaf digest can ever be mistaken for an
internal one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

#: Sibling side, as seen from the node being proved.
LEFT = "L"
RIGHT = "R"

#: One step of an audit path: the sibling's hex digest and which side it sits on.
ProofStep = tuple[str, str]


def _leaf_hash(record_hash: str) -> bytes:
    return hashlib.sha256(b"\x00" + bytes.fromhex(record_hash)).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _split_point(n: int) -> int:
    """Largest power of two strictly less than ``n`` (RFC 6962 §2)."""
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def _root(leaves: Sequence[bytes]) -> bytes:
    if len(leaves) == 1:
        return leaves[0]
    k = _split_point(len(leaves))
    return _node_hash(_root(leaves[:k]), _root(leaves[k:]))


def merkle_root(record_hashes: Sequence[str]) -> str:
    """Root over ``record_hashes``, as hex."""
    if not record_hashes:
        raise ValueError("cannot build a Merkle tree over zero records")
    return _root([_leaf_hash(h) for h in record_hashes]).hex()


def _audit_path(leaves: Sequence[bytes], index: int) -> list[ProofStep]:
    if len(leaves) == 1:
        return []
    k = _split_point(len(leaves))
    if index < k:
        return [*_audit_path(leaves[:k], index), (_root(leaves[k:]).hex(), RIGHT)]
    return [*_audit_path(leaves[k:], index - k), (_root(leaves[:k]).hex(), LEFT)]


def inclusion_proof(record_hashes: Sequence[str], index: int) -> list[ProofStep]:
    """Sibling path proving ``record_hashes[index]`` is in the tree.

    Ordered leaf-upward, so verification folds the list left to right.
    """
    if not 0 <= index < len(record_hashes):
        raise IndexError(f"index {index} outside a block of {len(record_hashes)}")
    return _audit_path([_leaf_hash(h) for h in record_hashes], index)


def verify_inclusion(record_hash: str, proof: Sequence[ProofStep], root: str) -> bool:
    """Recompute the root from one record and its audit path."""
    try:
        current = _leaf_hash(record_hash)
        for sibling_hex, side in proof:
            sibling = bytes.fromhex(sibling_hex)
            if side == RIGHT:
                current = _node_hash(current, sibling)
            elif side == LEFT:
                current = _node_hash(sibling, current)
            else:
                return False
    except ValueError:
        # Malformed hex anywhere in the supplied proof.
        return False
    return current.hex() == root
