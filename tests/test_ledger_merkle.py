"""Merkle tree and inclusion proofs (RFC 6962).

The second-preimage test is the reason this follows Certificate Transparency
rather than Bitcoin: Bitcoin duplicates the final leaf on odd levels, which lets
distinct leaf sets share a root (CVE-2012-2459).
"""

from __future__ import annotations

import hashlib
import math

import pytest

from pharmadt.ledger.merkle import inclusion_proof, merkle_root, verify_inclusion


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def block(n: int) -> list[str]:
    return [digest(f"record-{i}") for i in range(n)]


# ── Roots ─────────────────────────────────────────────────────────────


def test_root_is_hex_and_full_length() -> None:
    root = merkle_root(block(64))
    assert len(root) == 64
    assert set(root) <= set("0123456789abcdef")


def test_root_is_deterministic() -> None:
    assert merkle_root(block(64)) == merkle_root(block(64))


def test_root_changes_when_any_leaf_changes() -> None:
    leaves = block(64)
    tampered = [*leaves[:30], digest("forged"), *leaves[31:]]
    assert merkle_root(leaves) != merkle_root(tampered)


def test_root_depends_on_leaf_order() -> None:
    leaves = block(8)
    assert merkle_root(leaves) != merkle_root(list(reversed(leaves)))


def test_leaves_are_domain_separated_from_the_raw_hash() -> None:
    """A one-leaf root must not equal the leaf, or a leaf could pose as a tree."""
    leaf = digest("only")
    assert merkle_root([leaf]) != leaf


def test_padding_a_block_by_repeating_its_last_leaf_changes_the_root() -> None:
    """Bitcoin's tree collides here; RFC 6962's does not (CVE-2012-2459)."""
    leaves = block(3)
    assert merkle_root(leaves) != merkle_root([*leaves, leaves[-1]])


def test_an_empty_block_has_no_root() -> None:
    with pytest.raises(ValueError):
        merkle_root([])


# ── Inclusion proofs ──────────────────────────────────────────────────


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 13, 64])
def test_every_leaf_can_prove_its_own_membership(size: int) -> None:
    leaves = block(size)
    root = merkle_root(leaves)
    for index, leaf in enumerate(leaves):
        assert verify_inclusion(leaf, inclusion_proof(leaves, index), root)


def test_a_proof_is_logarithmic_not_linear() -> None:
    """The whole point: audit one record without re-reading the block."""
    leaves = block(64)
    assert len(inclusion_proof(leaves, 5)) == math.log2(64) == 6


def test_a_proof_does_not_verify_against_a_different_root() -> None:
    leaves = block(64)
    proof = inclusion_proof(leaves, 5)
    assert not verify_inclusion(leaves[5], proof, merkle_root(block(32)))


def test_a_proof_does_not_cover_a_leaf_that_is_not_in_the_block() -> None:
    leaves = block(64)
    proof = inclusion_proof(leaves, 5)
    assert not verify_inclusion(digest("never-recorded"), proof, merkle_root(leaves))


def test_a_proof_for_one_leaf_does_not_prove_another() -> None:
    leaves = block(64)
    proof = inclusion_proof(leaves, 5)
    assert not verify_inclusion(leaves[6], proof, merkle_root(leaves))


def test_reordering_a_proof_breaks_it() -> None:
    leaves = block(64)
    root = merkle_root(leaves)
    proof = inclusion_proof(leaves, 5)
    assert not verify_inclusion(leaves[5], list(reversed(proof)), root)


def test_flipping_a_sibling_side_breaks_the_proof() -> None:
    leaves = block(64)
    root = merkle_root(leaves)
    proof = inclusion_proof(leaves, 5)
    flipped = [(h, "L" if side == "R" else "R") for h, side in proof]
    assert not verify_inclusion(leaves[5], flipped, root)


def test_a_malformed_proof_is_rejected_without_raising() -> None:
    leaves = block(8)
    root = merkle_root(leaves)
    assert verify_inclusion(leaves[0], [("not-hex", "R")], root) is False
    assert verify_inclusion(leaves[0], [(digest("x"), "sideways")], root) is False


@pytest.mark.parametrize("index", [-1, 64, 999])
def test_proving_a_leaf_outside_the_block_is_an_error(index: int) -> None:
    with pytest.raises(IndexError):
        inclusion_proof(block(64), index)
