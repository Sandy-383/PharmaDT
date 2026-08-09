"""State-vector extraction: the only surface agents are allowed to see.

Agents receive plain data from here and never touch SimPy or ``TwinNode``
directly. That boundary is what makes the Stage 12 MARL wrapper possible — a
PettingZoo environment can hand a policy :meth:`NodeState.to_vector` without the
policy knowing a discrete-event simulator exists.

Per the report's §5.2.1 the per-node state is: stock per drug, pending inbound,
temperature zone, storage utilisation, and the rolling 28-day demand history.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pharmadt.config import settings
from pharmadt.core.models import NodeType
from pharmadt.twin.nodes import TwinNode

#: Short window used as a trend signal against the full 28-day mean.
RECENT_WINDOW_DAYS = 7

#: Per-drug slots emitted by :meth:`NodeState.to_vector`.
FEATURES_PER_DRUG = 5


@dataclass(frozen=True, slots=True)
class NodeState:
    """An immutable snapshot of one node on one day."""

    node_id: str
    node_type: NodeType
    sim_day: int
    stock_by_drug: Mapping[str, int]
    pending_inbound: Mapping[str, int]
    temperature_zone: str
    storage_utilisation: float
    storage_capacity: int
    demand_history: Mapping[str, tuple[int, ...]]

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-serialisable form — what agents and the API consume."""
        return {
            "node_id": self.node_id,
            "node_type": str(self.node_type),
            "sim_day": self.sim_day,
            "stock_by_drug": dict(self.stock_by_drug),
            "pending_inbound": dict(self.pending_inbound),
            "temperature_zone": self.temperature_zone,
            "storage_utilisation": round(self.storage_utilisation, 4),
            "storage_capacity": self.storage_capacity,
            "demand_history": {k: list(v) for k, v in self.demand_history.items()},
        }

    def to_vector(self, drugs: Sequence[str]) -> np.ndarray:
        """Fixed-width observation over ``drugs``, scaled to roughly [-1, 1].

        The 28-day history is summarised into level, variability, and a
        short-window trend rather than emitted raw. Raw history would make the
        observation 28 wide per drug for information that is essentially its
        first two moments plus a trend — a needlessly hard credit-assignment
        problem for MADDPG. The full history stays on :attr:`demand_history`
        for anything that wants it.
        """
        capacity = max(1, self.storage_capacity)
        scale = max(1.0, settings.base_daily_demand)

        features: list[float] = []
        for drug_id in drugs:
            history = self.demand_history.get(drug_id, ())
            recent = history[-RECENT_WINDOW_DAYS:]
            features.extend(
                (
                    self.stock_by_drug.get(drug_id, 0) / capacity,
                    self.pending_inbound.get(drug_id, 0) / capacity,
                    (statistics.fmean(history) if history else 0.0) / scale,
                    (statistics.pstdev(history) if len(history) > 1 else 0.0) / scale,
                    (statistics.fmean(recent) if recent else 0.0) / scale,
                )
            )

        features.append(self.storage_utilisation)
        features.append(1.0 if self.temperature_zone == "COLD" else 0.0)
        return np.asarray(features, dtype=np.float32)

    @staticmethod
    def vector_size(n_drugs: int) -> int:
        """Width of :meth:`to_vector` — Stage 12 declares its observation space with this."""
        return n_drugs * FEATURES_PER_DRUG + 2


def node_state(node: TwinNode, sim_day: int, drugs: Sequence[str] | None = None) -> NodeState:
    """Snapshot one node."""
    keys = list(drugs) if drugs is not None else node.drugs_handled()
    return NodeState(
        node_id=node.node_id,
        node_type=node.node_type,
        sim_day=sim_day,
        stock_by_drug={d: node.stock_of(d) for d in keys},
        pending_inbound={d: node.pending_inbound.get(d, 0) for d in keys},
        temperature_zone="COLD" if node.has_cold_storage else "AMBIENT",
        storage_utilisation=node.utilisation(),
        storage_capacity=node.storage_capacity,
        demand_history={d: tuple(node.demand_history.get(d, ())) for d in keys},
    )


def world_state(
    nodes: Mapping[str, TwinNode], sim_day: int, drugs: Sequence[str]
) -> dict[str, Any]:
    """The whole world as plain data, keyed by node id.

    Node ids are emitted in sorted order so that an agent iterating the mapping
    sees the same sequence on every run — the same reproducibility constraint
    that governs the network builder.
    """
    return {
        "sim_day": sim_day,
        "drugs": list(drugs),
        "nodes": {
            node_id: node_state(nodes[node_id], sim_day, drugs).to_dict()
            for node_id in sorted(nodes)
        },
    }
