"""Per-node ECDSA keypairs and the public-key registry.

The registry is this project's replacement for Fabric's MSP and channel policy.
A node's public key lives in the ``nodes`` table; its private key lives on disk
under ``data/keys/`` and is never committed. A record signed by a key that is
not in the registry is rejected, which is precisely what "only authorised peers
write" means in NFR-04.

Storing private keys as unencrypted PEM on disk is a simulation convenience and
the report says so plainly: in production they belong in an HSM or KMS.

Usage::

    python -m pharmadt.ledger.keyring          # issue keys to nodes lacking one
    python -m pharmadt.ledger.keyring --rotate # replace every existing key
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ec import (
    EllipticCurvePrivateKey,
    EllipticCurvePublicKey,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from pharmadt.config import settings
from pharmadt.core.db import session_scope
from pharmadt.core.models import Node
from pharmadt.ledger import crypto


def generate_node_keypair() -> tuple[EllipticCurvePrivateKey, str]:
    """A fresh P-256 keypair as (private key, public-key PEM)."""
    private_key = crypto.generate_private_key()
    return private_key, crypto.public_key_to_pem(private_key.public_key())


class UnauthorisedSigner(Exception):
    """Raised when a node without a registered public key tries to sign."""


class NodeKeyring:
    """Loads signing keys and answers "is this node allowed to write?".

    The public-key registry is read once and cached. ``verify_chain`` checks a
    signature per record, and re-querying the registry for each of thousands of
    records would make verification cost dominated by round trips rather than
    by cryptography.
    """

    def __init__(
        self,
        keys_dir: Path | None = None,
        session_factory: Callable[[], AbstractContextManager[Session]] | None = None,
    ) -> None:
        self.keys_dir = Path(keys_dir) if keys_dir is not None else settings.keys_dir
        # Injectable for the same reason as the ledger's: tests issue keys into
        # a transaction that is rolled back rather than into the real registry.
        self._session = session_factory if session_factory is not None else session_scope
        self._registry: dict[str, str] | None = None
        self._public: dict[str, EllipticCurvePublicKey] = {}
        self._private: dict[str, EllipticCurvePrivateKey] = {}

    # ── Registry ──────────────────────────────────────────────────────

    def registry(self) -> dict[str, str]:
        """node_id → public-key PEM, for every node that has one."""
        if self._registry is None:
            with self._session() as session:
                rows = session.execute(
                    select(Node.node_id, Node.public_key).order_by(Node.node_id)
                ).all()
            self._registry = {n: pem for n, pem in rows if pem}
        return self._registry

    def refresh(self) -> None:
        """Drop cached keys — call after issuing or rotating."""
        self._registry = None
        self._public.clear()
        self._private.clear()

    def is_authorised(self, node_id: str) -> bool:
        return node_id in self.registry()

    def public_key(self, node_id: str) -> EllipticCurvePublicKey:
        if node_id not in self._public:
            pem = self.registry().get(node_id)
            if pem is None:
                raise UnauthorisedSigner(f"{node_id} has no registered public key")
            self._public[node_id] = crypto.public_key_from_pem(pem)
        return self._public[node_id]

    # ── Private keys ──────────────────────────────────────────────────

    def key_path(self, node_id: str) -> Path:
        return self.keys_dir / f"{node_id}.pem"

    def private_key(self, node_id: str) -> EllipticCurvePrivateKey:
        if node_id not in self._private:
            path = self.key_path(node_id)
            if not path.exists():
                raise UnauthorisedSigner(
                    f"no private key for {node_id} at {path}. Run "
                    "`python -m pharmadt.ledger.keyring` to issue one."
                )
            self._private[node_id] = crypto.private_key_from_pem(path.read_bytes())
        return self._private[node_id]

    # ── Signing ───────────────────────────────────────────────────────

    def sign(self, node_id: str, record_hash: str) -> str:
        """Sign on behalf of ``node_id``.

        Authorisation is checked before signing rather than at verification
        time, so an unregistered node fails loudly at the point of the mistake
        instead of writing records that silently fail to verify later.
        """
        if not self.is_authorised(node_id):
            raise UnauthorisedSigner(f"{node_id} is not in the public-key registry")
        return crypto.sign(self.private_key(node_id), record_hash)

    def verify(self, node_id: str, record_hash: str, signature: str) -> bool:
        """Verify a signature against the registry. Unknown signer → False."""
        try:
            public_key = self.public_key(node_id)
        except (UnauthorisedSigner, ValueError):
            return False
        return crypto.verify(public_key, record_hash, signature)


def issue_keys(
    keys_dir: Path | None = None,
    rotate: bool = False,
    session_factory: Callable[[], AbstractContextManager[Session]] | None = None,
) -> list[str]:
    """Give every node a keypair. Returns the node ids that received one."""
    keyring = NodeKeyring(keys_dir, session_factory)
    keyring.keys_dir.mkdir(parents=True, exist_ok=True)

    issued: list[str] = []
    with keyring._session() as session:
        for node in session.scalars(select(Node).order_by(Node.node_id)):
            if node.public_key and not rotate:
                continue
            private_key, public_pem = generate_node_keypair()
            keyring.key_path(node.node_id).write_bytes(
                crypto.private_key_to_pem(private_key)
            )
            node.public_key = public_pem
            issued.append(node.node_id)

    keyring.refresh()
    return issued


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue per-node signing keys.")
    parser.add_argument(
        "--rotate", action="store_true", help="replace keys for nodes that already have one"
    )
    args = parser.parse_args()

    issued = issue_keys(rotate=args.rotate)
    if issued:
        print(f"Issued {len(issued)} keypair(s) to {settings.keys_dir}:")
        for node_id in issued:
            print(f"  {node_id}")
    else:
        print("Every node already holds a keypair. Use --rotate to replace them.")


if __name__ == "__main__":
    main()
