"""Inject known anomalies so detection can be measured against ground truth.

Without injection there are no labels, and without labels a detector can only
be admired rather than evaluated. Each injected shipment is tagged, so
precision and recall are computed against something known rather than against
the detector's own opinion.

Four anomaly kinds, chosen because each is a different *shape* of outlier:

* ``QUANTITY``   — a shipment far larger than that route ever carries.
* ``TRANSIT``    — a delivery that takes much longer than the lane allows.
* ``COLD_CHAIN`` — repeated temperature excursions on one shipment.
* ``COUNTERFEIT``— a batch whose fingerprint does not match its own fields.

The last one matters most. It is invisible in the shipment features by
construction: a counterfeit batch moves normally, in normal quantities, over a
normal route. **Only the ledger cross-check catches it**, which is precisely
the point of research gap 3.2.3 — the ledger as a live input to a detection
system rather than an isolated record store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class AnomalyKind(StrEnum):
    QUANTITY = "QUANTITY"
    TRANSIT = "TRANSIT"
    COLD_CHAIN = "COLD_CHAIN"
    COUNTERFEIT = "COUNTERFEIT"


#: Kinds the shipment features can express. COUNTERFEIT is deliberately absent.
FEATURE_VISIBLE = (AnomalyKind.QUANTITY, AnomalyKind.TRANSIT, AnomalyKind.COLD_CHAIN)


@dataclass(slots=True)
class InjectionReport:
    """What was injected, so the evaluation knows the truth."""

    labels: np.ndarray
    kinds: list[str] = field(default_factory=list)
    counterfeit_batches: list[str] = field(default_factory=list)

    @property
    def n_injected(self) -> int:
        return int(self.labels.sum())

    def kind_mask(self, kind: AnomalyKind) -> np.ndarray:
        return np.array([k == kind for k in self.kinds], dtype=bool)


def inject_anomalies(
    shipments: list[dict[str, Any]],
    rate: float = 0.05,
    seed: int = 42,
    kinds: tuple[AnomalyKind, ...] = tuple(AnomalyKind),
) -> InjectionReport:
    """Corrupt a labelled fraction of ``shipments`` in place.

    Perturbations are multiplicative and drawn from a range, not set to one
    fixed extreme value. A constant marker would be trivially separable and the
    reported recall would measure nothing but the marker.
    """
    if not shipments:
        return InjectionReport(np.zeros(0, dtype=bool))

    rng = np.random.default_rng(seed)
    n = len(shipments)
    labels = np.zeros(n, dtype=bool)
    assigned = ["" for _ in range(n)]
    counterfeits: list[str] = []

    n_inject = max(1, int(n * rate))
    chosen = rng.choice(n, size=min(n_inject, n), replace=False)

    for index in chosen:
        kind = AnomalyKind(rng.choice([str(k) for k in kinds]))
        shipment = shipments[int(index)]
        labels[index] = True
        assigned[index] = str(kind)
        shipment["anomaly_kind"] = str(kind)

        if kind is AnomalyKind.QUANTITY:
            shipment["quantity"] = int(shipment["quantity"] * rng.uniform(4.0, 12.0))
        elif kind is AnomalyKind.TRANSIT:
            shipment["transit_days"] = float(
                shipment["transit_days"] * rng.uniform(3.0, 8.0)
            )
        elif kind is AnomalyKind.COLD_CHAIN:
            shipment["excursion_count"] = int(rng.integers(3, 9))
            shipment["excursion_severity"] = float(rng.uniform(4.0, 12.0))
        elif kind is AnomalyKind.COUNTERFEIT:
            # Nothing about the shipment changes. That is the whole point: the
            # ML features cannot see this, and only the ledger can.
            batch_id = shipment.get("batch_id")
            if batch_id:
                counterfeits.append(batch_id)
            shipment["forged_fingerprint"] = True

    return InjectionReport(labels, assigned, counterfeits)


# ── Crisis injection (Stage 13) ───────────────────────────────────────


def apply_scenario(world: Any, scenario: Any) -> dict[str, Any]:
    """Apply every effect and return exactly what is needed to undo it.

    The undo state is captured *before* anything changes and returned rather
    than stored globally, so two scenarios overlapping in time each hold their
    own record and neither can restore the other's values by accident.
    """
    undo: dict[str, Any] = {"demand": {}, "nodes": [], "coldchain": None, "edges": []}

    for effect in scenario.effects:
        if effect.kind == "demand_multiplier":
            for node in _matching(world, effect):
                for drug_id, profile in node.demand_profiles.items():
                    undo["demand"][(node.node_id, drug_id)] = profile.mean
                    profile.mean *= effect.magnitude

        elif effect.kind == "disable_node":
            for node in _matching(world, effect):
                if node.node_id not in world.disabled_nodes:
                    world.disabled_nodes.add(node.node_id)
                    undo["nodes"].append(node.node_id)

        elif effect.kind == "coldchain_risk":
            undo["coldchain"] = world.coldchain_risk_multiplier
            world.coldchain_risk_multiplier *= effect.magnitude

        elif effect.kind == "remove_edges":
            for source in effect.from_nodes:
                if source not in world.graph:
                    continue
                for target in list(world.graph.successors(source)):
                    undo["edges"].append(
                        (source, target, dict(world.graph[source][target]))
                    )
                    world.graph.remove_edge(source, target)

    world.active_scenarios.append(scenario.name)
    return undo


def revert_scenario(world: Any, scenario: Any, undo: dict[str, Any]) -> None:
    """Put back everything :func:`apply_scenario` changed."""
    for (node_id, drug_id), mean in undo["demand"].items():
        node = world.nodes.get(node_id)
        if node and drug_id in node.demand_profiles:
            node.demand_profiles[drug_id].mean = mean

    for node_id in undo["nodes"]:
        world.disabled_nodes.discard(node_id)

    if undo["coldchain"] is not None:
        world.coldchain_risk_multiplier = undo["coldchain"]

    for source, target, data in undo["edges"]:
        world.graph.add_edge(source, target, **data)

    if scenario.name in world.active_scenarios:
        world.active_scenarios.remove(scenario.name)


def crisis_process(world: Any, scenario: Any):
    """SimPy process: apply at the trigger day, revert when the window closes."""
    yield world.env.timeout(scenario.trigger_day)
    undo = apply_scenario(world, scenario)

    yield world.env.timeout(scenario.duration_days)
    revert_scenario(world, scenario, undo)


def _matching(world: Any, effect: Any) -> list[Any]:
    """Nodes an effect applies to, by explicit id or by tier."""
    if effect.node_ids:
        return [world.nodes[n] for n in sorted(effect.node_ids) if n in world.nodes]
    if effect.node_types:
        wanted = {t.upper() for t in effect.node_types}
        return [
            world.nodes[n]
            for n in sorted(world.nodes)
            if str(world.nodes[n].node_type) in wanted
        ]
    return [world.nodes[n] for n in sorted(world.nodes)]
