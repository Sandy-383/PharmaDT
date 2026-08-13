"""Inventory Agent — LLD Box 1.

Stock Monitoring Engine -> Threshold Trigger -> Replenishment Order.

Replaces the twin's fixed-threshold (s, S) baseline with a classical reorder
point that accounts for demand *variability*, which is the one thing the
baseline ignores:

    ROP        = mu * L + z * sigma * sqrt(L)
    order-up-to = mu * (L + R) + z * sigma * sqrt(L)

where mu and sigma are the mean and standard deviation of recent daily demand,
L is the supplier lead time, R the review period, and z the normal quantile for
the target service level (1.65 for 95%).

The baseline reorders whenever the position falls below three days of *mean*
demand. That is the same threshold for a steady node and a volatile one, so
every stockout it suffers comes from variance it never modelled. The Rossmann
fit in Stage 2 measured dispersion between 0.14 and 0.46 across real series, so
that variance is not hypothetical.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pharmadt.agents.base import BaseAgent
from pharmadt.agents.bus import Topic
from pharmadt.config import settings
from pharmadt.core.events import Action

#: Demand below this is treated as no demand; ordering against sampling noise
#: would fill a pharmacy with a drug nobody asks for.
MIN_MEAN_DEMAND = 0.05


class InventoryAgent(BaseAgent):
    """Monitors stock and issues replenishment orders."""

    name = "InventoryAgent"

    def __init__(
        self,
        graph: Any = None,
        z: float | None = None,
        coverage_days: int | None = None,
        review_period_days: int = 1,
        **kwargs: Any,
    ) -> None:
        self.graph = graph
        self.z = settings.service_level_z if z is None else z
        # How far ahead the order-up-to level reaches once an order is placed.
        self.coverage_days = (
            settings.order_up_to_days if coverage_days is None else coverage_days
        )
        # The agent reviews every simulated day, so stock is exposed for the
        # lead time plus one review interval.
        self.review_period_days = review_period_days
        self._echelon_lead_time: dict[str, int] = {}
        # Forecasts published by the Demand Agent, keyed (node_id, drug_id).
        # Stage 7 fills this; until then the agent runs on observed history
        # alone, which is exactly the ablation row the report needs.
        self.forecasts: dict[tuple[str, str], float] = {}
        super().__init__(**kwargs)

    # ── Bus ───────────────────────────────────────────────────────────

    def register_subscriptions(self) -> None:
        self.subscribe(Topic.FORECAST_DATA, self._on_forecast)
        self.subscribe(Topic.SHORTAGE_ALERT, self._on_shortage)

    def _on_forecast(self, message) -> None:
        payload = message.payload
        node_id, drug_id = payload.get("node_id"), payload.get("drug_id")
        if node_id and drug_id and "mean_daily" in payload:
            self.forecasts[(node_id, drug_id)] = float(payload["mean_daily"])

    def _on_shortage(self, message) -> None:
        """A shortage alert raises the effective service level for that pair."""
        payload = message.payload
        node_id, drug_id = payload.get("node_id"), payload.get("drug_id")
        if node_id and drug_id and "mean_daily" in payload:
            # Trust the forecast over history when a shortage is predicted.
            self.forecasts[(node_id, drug_id)] = float(payload["mean_daily"])

    # ── Observe ───────────────────────────────────────────────────────

    def observe(self, world_state: Mapping[str, Any]) -> dict[str, Any]:
        """Per (node, drug) stock position and demand statistics."""
        positions: list[dict[str, Any]] = []

        for node_id, node in world_state.get("nodes", {}).items():
            supplier = self._supplier_of(node_id)
            if supplier is None:
                continue  # the manufacturer has nobody upstream to order from

            lead_time = self._lead_time(supplier, node_id)
            history: Mapping[str, Sequence[int]] = node.get("demand_history", {})
            stock: Mapping[str, int] = node.get("stock_by_drug", {})
            inbound: Mapping[str, int] = node.get("pending_inbound", {})

            # Soonest expiry per drug among the stock this node already holds.
            soonest: dict[str, int] = {}
            for lot in node.get("expiring_lots", []):
                drug = lot["drug_id"]
                days = int(lot["days_to_expiry"])
                if drug not in soonest or days < soonest[drug]:
                    soonest[drug] = days

            for drug_id, series in history.items():
                mean, sigma = _mean_and_std(series)
                forecast = self.forecasts.get((node_id, drug_id))
                if forecast is not None:
                    mean = forecast

                positions.append(
                    {
                        "node_id": node_id,
                        "drug_id": drug_id,
                        "supplier": supplier,
                        "on_hand": int(stock.get(drug_id, 0)),
                        "pending_inbound": int(inbound.get(drug_id, 0)),
                        "mean_daily": round(mean, 4),
                        "std_daily": round(sigma, 4),
                        "lead_time_days": lead_time,
                        "days_to_expiry": soonest.get(drug_id),
                        "free_space": max(
                            0,
                            int(node.get("storage_capacity", 0))
                            - int(sum(stock.values())),
                        ),
                    }
                )

        return {"sim_day": world_state.get("sim_day", 0), "positions": positions}

    # ── Decide ────────────────────────────────────────────────────────

    def decide(self, observation: Mapping[str, Any]) -> list[Action]:
        actions: list[Action] = []
        # Space is finite, so competing orders for one node are scaled together
        # rather than served in name order until the shelf is full.
        by_node: dict[str, list[dict[str, Any]]] = {}

        for position in observation.get("positions", []):
            proposal = self._propose(position)
            if proposal is not None:
                by_node.setdefault(position["node_id"], []).append(proposal)

        for node_id, proposals in sorted(by_node.items()):
            free_space = proposals[0]["free_space"]
            total = sum(p["quantity"] for p in proposals)
            scale = min(1.0, free_space / total) if total > free_space and total else 1.0

            for proposal in proposals:
                quantity = int(proposal["quantity"] * scale)
                if quantity <= 0:
                    continue
                actions.append(
                    Action(
                        action_type="REORDER",
                        target_node=node_id,
                        drug_id=proposal["drug_id"],
                        quantity=quantity,
                        params={
                            "supplier": proposal["supplier"],
                            "reorder_point": proposal["reorder_point"],
                            "position": proposal["position"],
                            "safety_stock": proposal["safety_stock"],
                        },
                        justification=(
                            f"position {proposal['position']} below reorder point "
                            f"{proposal['reorder_point']} "
                            f"(mu={proposal['mean_daily']:.1f}/day, "
                            f"sigma={proposal['std_daily']:.1f}, "
                            f"echelon L={proposal['lead_time_days']}d, "
                            f"risk period={proposal['risk_period_days']}d, "
                            f"safety={proposal['safety_stock']}); "
                            f"ordering {quantity} to reach the order-up-to level"
                        ),
                    )
                )

        return actions

    def _propose(self, position: Mapping[str, Any]) -> dict[str, Any] | None:
        mean = position["mean_daily"]
        if mean <= MIN_MEAN_DEMAND:
            return None

        sigma = position["std_daily"]
        # Risk period, not just transit: stock is exposed until the *next*
        # review's order could arrive, so the review interval belongs here too.
        risk_period = max(1, position["lead_time_days"] + self.review_period_days)

        # The term the baseline is missing: protection against demand varying
        # over the risk period, not just its average.
        safety_stock = self.z * sigma * math.sqrt(risk_period)
        reorder_point = mean * risk_period + safety_stock

        # Shelf-life-aware coverage. Ordering a fixed horizon regardless of how
        # long the stock already held has left is what makes a service-level
        # policy waste perishables: under FEFO the oldest units are issued
        # first, so a position larger than can be consumed before they expire
        # guarantees that the excess is thrown away.
        #
        # The binding horizon is therefore the shorter of the planning coverage
        # and the time the current stock has left. Measured on a 365-day run
        # with the CMS-calibrated drug mix, this is the difference between
        # wastage the Expiry Agent has to clean up and wastage never created.
        coverage = self.coverage_days
        days_left = position.get("days_to_expiry")
        if days_left is not None:
            coverage = min(coverage, max(0, days_left))

        order_up_to = mean * (risk_period + coverage) + safety_stock

        stock_position = position["on_hand"] + position["pending_inbound"]
        if stock_position >= reorder_point:
            return None

        quantity = int(order_up_to - stock_position)
        if quantity <= 0:
            return None

        return {
            "drug_id": position["drug_id"],
            "supplier": position["supplier"],
            "quantity": quantity,
            "position": stock_position,
            "reorder_point": int(reorder_point),
            "safety_stock": int(safety_stock),
            "mean_daily": mean,
            "std_daily": sigma,
            "lead_time_days": position["lead_time_days"],
            "risk_period_days": risk_period,
            "free_space": position["free_space"],
        }

    # ── Act ───────────────────────────────────────────────────────────

    def apply(self, action: Action, world: Any) -> None:
        """Place the order on the twin and announce it to the Route Agent."""
        from pharmadt.twin.processes import _place_order

        node = world.nodes[action.target_node]
        supplier = action.params["supplier"]
        _place_order(
            world, node, supplier, action.drug_id, action.quantity, int(world.env.now)
        )

        self.publish(
            Topic.REPLENISHMENT_ORDER,
            {
                "node_id": action.target_node,
                "supplier": supplier,
                "drug_id": action.drug_id,
                "quantity": action.quantity,
            },
            sim_day=int(world.env.now),
        )

    # ── Network helpers ───────────────────────────────────────────────

    def _supplier_of(self, node_id: str) -> str | None:
        if self.graph is None:
            return None
        from pharmadt.twin.network import upstream_supplier

        return upstream_supplier(self.graph, node_id)

    def _lead_time(self, supplier: str, node_id: str) -> int:
        """Cumulative (echelon) transit from the source down to ``node_id``.

        Emphatically **not** the last hop. In a multi-tier chain an upstream
        tier only reorders when its own position dips, so a pharmacy's stock is
        exposed for the whole pipeline latency rather than for its final leg.
        Sizing the reorder point on one hop understates the risk period by the
        depth of the chain, and the agent then reorders later than the naive
        baseline it was meant to beat — measurably worse, which is exactly what
        the first version of this agent did.
        """
        if self.graph is None:
            return settings.default_lead_time_days
        if node_id in self._echelon_lead_time:
            return self._echelon_lead_time[node_id]

        from pharmadt.twin.network import upstream_supplier

        total, current, seen = 0, node_id, {node_id}
        while True:
            parent = upstream_supplier(self.graph, current)
            if parent is None or parent in seen:
                break
            total += int(
                self.graph[parent][current].get(
                    "transit_days", settings.default_lead_time_days
                )
            )
            seen.add(parent)
            current = parent

        lead_time = max(total, settings.default_lead_time_days if total == 0 else total)
        self._echelon_lead_time[node_id] = lead_time
        return lead_time


def _mean_and_std(series: Sequence[int]) -> tuple[float, float]:
    """Sample mean and standard deviation of a demand window."""
    values = [float(v) for v in series]
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(variance)
