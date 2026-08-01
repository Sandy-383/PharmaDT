"""The three abstract contracts every downstream layer is written against.

These are the load-bearing seams of the architecture:

* :class:`ProvenanceLedger` lets the agents record custody events without
  knowing whether the backend is this project's hash-chained Postgres ledger or
  a Hyperledger Fabric channel.
* :class:`Agent` lets Stage 12 swap a heuristic policy for a trained MADDPG
  policy without the twin noticing.
* :class:`DemandModel` lets Stage 11 federate any forecaster, because every
  forecaster is required to serialise its own weights from day one.

Changing a signature here is expensive — every layer below depends on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pharmadt.core.events import Action, EventType


class ProvenanceLedger(ABC):
    """Append-only, tamper-evident record of physical custody.

    Implemented in Stage 4 by the hash-chained Postgres ledger. The interface
    is deliberately narrower than that implementation so a Fabric-backed
    ledger could satisfy it unchanged.
    """

    @abstractmethod
    def record_event(
        self,
        batch_id: str,
        event_type: EventType,
        from_node: str | None,
        to_node: str | None,
        payload: Mapping[str, Any],
        signer_node: str,
    ) -> str:
        """Append one signed, chained record. Returns its ``record_hash``.

        ``signer_node`` must hold a keypair whose public key is in the node
        registry; that allow-list is the permissioning layer. Implementations
        must reject an ``event_type`` outside
        :data:`~pharmadt.core.events.LEDGER_EVENT_TYPES`.
        """

    @abstractmethod
    def get_provenance(self, batch_id: str) -> Sequence[Mapping[str, Any]]:
        """Return every record for ``batch_id`` in ``seq`` order.

        Plain mappings rather than ORM rows, so the trace can be serialised
        straight to the Stage 14 API and so the storage layer stays swappable.
        """

    @abstractmethod
    def verify_chain(self, start: int | None = None, end: int | None = None) -> bool:
        """Recompute every hash and signature over ``[start, end]``.

        Returns ``False`` at the first break. Implementations must log the
        offending ``seq`` — the Stage 4 demo turns on being able to name the
        exact record that was tampered with.
        """

    @abstractmethod
    def verify_batch_fingerprint(self, batch_id: str, presented: str) -> bool:
        """Compare a presented SHA-256 fingerprint against the recomputed one.

        A mismatch is the anti-counterfeit signal consumed by the Anomaly
        Agent in Stage 10.
        """


class Agent(ABC):
    """An autonomous decision-maker in the observe/decide/act loop.

    Agents receive a plain ``world_state`` mapping from the twin's state layer
    and never touch SimPy internals. That decoupling is precisely what makes
    the Stage 12 MARL wrapper possible.
    """

    name: str

    @abstractmethod
    def observe(self, world_state: Mapping[str, Any]) -> dict[str, Any]:
        """Project the global world state down to this agent's observation."""

    @abstractmethod
    def decide(self, observation: Mapping[str, Any]) -> list[Action]:
        """Choose zero or more actions. Must not mutate anything."""

    @abstractmethod
    def act(self, actions: Sequence[Action], world: Any) -> None:
        """Apply ``actions`` to the world and log an ``AgentDecision`` row."""


class DemandModel(ABC):
    """A drug-demand forecaster that can be trained centrally or federated.

    ``get_weights``/``set_weights`` exist from Stage 1 rather than Stage 11
    because Flower drives federation entirely through weight exchange. Adding
    them retroactively would mean rewriting every model class.
    """

    @abstractmethod
    def fit(self, X: Any, y: Any) -> None:
        """Train on local data."""

    @abstractmethod
    def predict(self, X: Any, horizon: int = 14) -> np.ndarray:
        """Forecast ``horizon`` days ahead."""

    @abstractmethod
    def get_weights(self) -> list[np.ndarray]:
        """Return parameters as Flower's ``NDArrays``.

        The list order must be stable across calls and identical across
        clients — FedAvg averages positionally, so a reordered list silently
        produces a corrupt global model rather than an error.
        """

    @abstractmethod
    def set_weights(self, w: list[np.ndarray]) -> None:
        """Load parameters produced by :meth:`get_weights`."""
