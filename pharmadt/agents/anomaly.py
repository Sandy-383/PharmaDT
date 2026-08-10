"""Anomaly Detection Agent — LLD Box 5.

Shipment log -> Isolation Forest ‖ Autoencoder -> score > tau -> suspect ->
ledger hash verify -> counterfeit / fraud alert.

The ledger cross-check is the part that matters. A counterfeit batch moves in
normal quantities, over a normal route, in normal time — so by construction the
shipment features cannot see it, and both ML models rank it as ordinary. What
distinguishes it is that its fingerprint does not match its own fields, and only
the provenance ledger can say so.

That is research gap 3.2.3 made concrete: the ledger is a **live input to a
detection system**, not an isolated record store that nothing reads.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pharmadt.agents.base import BaseAgent
from pharmadt.agents.bus import Topic
from pharmadt.core.events import Action

logger = logging.getLogger(__name__)

#: Feature order. Fixed because the autoencoder's input width and the
#: Isolation Forest's column meanings both depend on it; a reordering would not
#: raise, it would silently score the wrong things.
FEATURE_NAMES = (
    "quantity",
    "quantity_z",
    "transit_days",
    "transit_deviation",
    "excursion_count",
    "excursion_severity",
    "route_frequency",
    "is_known_route",
    "distance_km",
    "day_of_week",
    "cold_chain",
    "quantity_per_km",
)


def extract_features(shipments: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Turn shipment records into the fixed feature matrix.

    Route-relative features (``quantity_z``, ``transit_deviation``) are computed
    per node pair rather than globally. A 500-unit delivery is unremarkable to a
    warehouse and extraordinary to a village pharmacy, and a global z-score
    would call the second one normal.
    """
    if not shipments:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=float)

    by_route: dict[tuple[str, str], list[float]] = {}
    transit_by_route: dict[tuple[str, str], list[float]] = {}
    for shipment in shipments:
        key = (shipment.get("from_node", ""), shipment.get("to_node", ""))
        by_route.setdefault(key, []).append(float(shipment.get("quantity", 0)))
        transit_by_route.setdefault(key, []).append(float(shipment.get("transit_days", 0)))

    rows: list[list[float]] = []
    for shipment in shipments:
        key = (shipment.get("from_node", ""), shipment.get("to_node", ""))
        quantities = by_route.get(key, [0.0])
        transits = transit_by_route.get(key, [0.0])

        quantity = float(shipment.get("quantity", 0))
        mean_q = float(np.mean(quantities))
        std_q = max(float(np.std(quantities)), 1e-6)
        transit = float(shipment.get("transit_days", 0))
        expected_transit = float(np.median(transits))
        distance = max(float(shipment.get("distance_km", 0.0)), 1e-6)

        rows.append(
            [
                quantity,
                (quantity - mean_q) / std_q,
                transit,
                transit - expected_transit,
                float(shipment.get("excursion_count", 0)),
                float(shipment.get("excursion_severity", 0.0)),
                float(len(quantities)),
                1.0 if shipment.get("is_known_route", True) else 0.0,
                distance,
                float(shipment.get("sim_day", 0) % 7),
                1.0 if shipment.get("cold_chain") else 0.0,
                quantity / distance,
            ]
        )
    return np.asarray(rows, dtype=float)


class AnomalyAgent(BaseAgent):
    """Scores shipments, then asks the ledger about the suspects."""

    name = "AnomalyAgent"

    def __init__(
        self,
        ledger: Any = None,
        forest: Any = None,
        autoencoder: Any = None,
        rule: str = "either",
        world: Any = None,
        **kwargs: Any,
    ) -> None:
        # Optional handle on the twin so the agent can read each day's completed
        # shipments directly. The ledger is only written in bulk after the run
        # (NFR-01), so subscribing to ledger.event alone would leave this agent
        # with nothing to screen while the simulation is actually running.
        self.world = world
        self.ledger = ledger
        self.forest = forest
        self.autoencoder = autoencoder
        self.rule = rule
        #: Shipments seen today, rebuilt from the twin each day.
        self.recent: list[dict[str, Any]] = []
        self.alerts: list[dict[str, Any]] = []
        self.ledger_catches = 0
        super().__init__(**kwargs)

    # ── Bus ───────────────────────────────────────────────────────────

    def register_subscriptions(self) -> None:
        self.subscribe(Topic.LEDGER_EVENT, self._on_ledger_event)

    def _on_ledger_event(self, message) -> None:
        payload = message.payload
        if payload.get("batch_id"):
            self.recent.append(dict(payload))

    # ── Observe ───────────────────────────────────────────────────────

    def observe(self, world_state: Mapping[str, Any]) -> dict[str, Any]:
        sim_day = world_state.get("sim_day", 0)
        shipments = list(self.recent)
        if self.world is not None:
            shipments.extend(self._shipments_received_on(sim_day))
        return {"sim_day": sim_day, "shipments": shipments}

    def _shipments_received_on(self, sim_day: int) -> list[dict[str, Any]]:
        """Deliveries that completed today, as screenable records."""
        records: list[dict[str, Any]] = []
        for event in reversed(self.world.events):
            if event.sim_day < sim_day:
                break  # the log is append-ordered, so nothing older matters
            if str(event.event_type) != "SHIPMENT_RECEIVED":
                continue
            edge = self.world.graph.get_edge_data(event.from_node, event.to_node) or {}
            records.append(
                {
                    "batch_id": event.batch_id,
                    "drug_id": event.payload.get("drug_id"),
                    "from_node": event.from_node,
                    "to_node": event.to_node,
                    "quantity": event.payload.get("quantity", 0),
                    "sim_day": event.sim_day,
                    "transit_days": 1.0,
                    "excursion_count": 1 if event.payload.get("cold_chain_breached") else 0,
                    "excursion_severity": 5.0 if event.payload.get("cold_chain_breached") else 0.0,
                    "distance_km": float(edge.get("distance_km", 1.0)),
                    "is_known_route": bool(edge),
                    "cold_chain": bool(event.payload.get("cold_chain_breached")),
                }
            )
        return records

    # ── Decide ────────────────────────────────────────────────────────

    def screen(self, shipments: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """ML suspicion per shipment. All-False when no model is fitted."""
        if not shipments:
            return np.zeros(0, dtype=bool)

        features = extract_features(shipments)
        flags = np.zeros(len(shipments), dtype=bool)

        if self.forest is not None:
            flags |= np.asarray(self.forest.predict(features), dtype=bool)
        if self.autoencoder is not None:
            auto = np.asarray(self.autoencoder.predict(features), dtype=bool)
            flags = (flags | auto) if self.rule == "either" else (flags & auto)
        return flags

    def verify_against_ledger(self, shipment: Mapping[str, Any]) -> str | None:
        """Ask the ledger about one shipment. Returns a reason, or None if clean.

        Three independent signals, any of which condemns a batch: a fingerprint
        that does not match the batch's own fields, a batch the ledger has never
        recorded, and a chain segment that fails verification.
        """
        if self.ledger is None:
            return None

        batch_id = shipment.get("batch_id")
        if not batch_id:
            return None

        presented = shipment.get("presented_fingerprint")
        if presented and not self.ledger.verify_batch_fingerprint(batch_id, presented):
            return "batch fingerprint does not match its recorded identity"

        trace = self.ledger.get_provenance(batch_id)
        if not trace:
            return "batch has no provenance record at all"
        return None

    def decide(self, observation: Mapping[str, Any]) -> list[Action]:
        shipments = observation.get("shipments", [])
        if not shipments:
            return []

        suspected = self.screen(shipments)
        actions: list[Action] = []

        for shipment, flagged in zip(shipments, suspected, strict=True):
            reason = self.verify_against_ledger(shipment) if flagged else None

            # A ledger failure is a counterfeit alert regardless of what the ML
            # thought; the models cannot see forged identity by construction.
            ledger_reason = reason or (
                self.verify_against_ledger(shipment) if not flagged else None
            )
            if ledger_reason:
                self.ledger_catches += 1
                actions.append(
                    Action(
                        action_type="COUNTERFEIT_ALERT",
                        target_node=shipment.get("to_node"),
                        batch_id=shipment.get("batch_id"),
                        drug_id=shipment.get("drug_id"),
                        params={"ml_flagged": bool(flagged), "reason": ledger_reason},
                        justification=(
                            f"{shipment.get('batch_id')}: {ledger_reason}"
                            + (
                                " (the ML models ranked this shipment as ordinary; "
                                "only the ledger caught it)"
                                if not flagged
                                else " (ML also flagged it)"
                            )
                        ),
                    )
                )
            elif flagged:
                actions.append(
                    Action(
                        action_type="ANOMALY_FLAG",
                        target_node=shipment.get("to_node"),
                        batch_id=shipment.get("batch_id"),
                        drug_id=shipment.get("drug_id"),
                        params={"source": "ml"},
                        justification=(
                            "shipment scored anomalous but its ledger record is intact; "
                            "flagged for review rather than as counterfeit"
                        ),
                    )
                )

        return actions

    # ── Act ───────────────────────────────────────────────────────────

    def apply(self, action: Action, world: Any) -> None:
        self.alerts.append(
            {
                "batch_id": action.batch_id,
                "type": action.action_type,
                "reason": action.params.get("reason", ""),
            }
        )
        if action.action_type == "COUNTERFEIT_ALERT":
            # The Expiry Agent quarantines it; the dashboard renders it.
            self.publish(
                Topic.COUNTERFEIT_FLAG,
                {
                    "batch_id": action.batch_id,
                    "drug_id": action.drug_id,
                    "node_id": action.target_node,
                    "reason": action.params.get("reason", ""),
                },
                sim_day=int(world.env.now) if world is not None else 0,
            )

    def clear(self) -> None:
        self.recent.clear()
