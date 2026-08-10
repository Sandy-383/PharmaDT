"""Expiry Management Agent — LLD Box 3.

Expiry Detector -> Flag Engine (urgency score) -> Vickrey auction -> Redistribution.

The auction is a **sealed-bid second-price (Vickrey) auction**, named because
the choice is defensible rather than arbitrary. Truthful bidding is a dominant
strategy under Vickrey: a node cannot gain by overstating how much it can
consume, because the price it pays is set by the runner-up's bid, not its own.
In a first-price auction every node has an incentive to shade its bid, and the
allocation stops tracking who can actually use the stock — which is the only
thing redistribution is trying to discover.

Economics, all relative to ``settings.unit_value``:

* **Bid** = units the buyer can plausibly consume before expiry x unit value,
  minus transport cost over the lateral edge.
* **Reserve** = disposal cost avoided. Below it, moving the stock costs more
  than binning it, so nothing is redistributed.
* **Surplus** = only the quantity the *seller* cannot consume in time is
  offered. Auctioning stock the seller will sell anyway just moves a stockout.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pharmadt.agents.base import BaseAgent
from pharmadt.agents.bus import Topic
from pharmadt.config import settings
from pharmadt.core.events import Action

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Bid:
    """One buyer's sealed bid."""

    node_id: str
    amount: float
    consumable: int


@dataclass(frozen=True, slots=True)
class AuctionResult:
    """Outcome of one lot's auction, including why it cleared or did not."""

    winner: str | None
    clearing_price: float
    quantity: int
    reserve: float
    bids: tuple[Bid, ...]
    reason: str


class ExpiryAgent(BaseAgent):
    """Detects near-expiry stock and redistributes it by auction."""

    name = "ExpiryAgent"

    def __init__(
        self,
        graph: Any = None,
        alert_days: int | None = None,
        unit_value: float | None = None,
        disposal_cost: float | None = None,
        transport_cost_per_km: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.graph = graph
        self.alert_days = (
            settings.redistribution_horizon_days if alert_days is None else alert_days
        )
        self.unit_value = settings.unit_value if unit_value is None else unit_value
        self.disposal_cost = (
            settings.disposal_cost_per_unit if disposal_cost is None else disposal_cost
        )
        self.transport_cost_per_km = (
            settings.transport_cost_per_unit_km
            if transport_cost_per_km is None
            else transport_cost_per_km
        )
        #: Forecast demand per (node, drug), from the Demand Agent.
        self.forecasts: dict[tuple[str, str], float] = {}
        #: Nodes with surging demand — preferred destinations, all else equal.
        self.hotspots: set[tuple[str, str]] = set()
        #: Batches flagged as counterfeit; never auctioned.
        self._quarantined: set[str] = set()
        self.auctions: list[AuctionResult] = []
        # Set before super().__init__ because that calls register_subscriptions,
        # whose handlers write into these.
        super().__init__(**kwargs)

    # ── Bus ───────────────────────────────────────────────────────────

    def register_subscriptions(self) -> None:
        self.subscribe(Topic.FORECAST_DATA, self._on_forecast)
        self.subscribe(Topic.DEMAND_HOTSPOT, self._on_hotspot)
        self.subscribe(Topic.COUNTERFEIT_FLAG, self._on_counterfeit)

    def _on_forecast(self, message) -> None:
        payload = message.payload
        key = (payload.get("node_id"), payload.get("drug_id"))
        if all(key) and "mean_daily" in payload:
            self.forecasts[key] = float(payload["mean_daily"])

    def _on_hotspot(self, message) -> None:
        payload = message.payload
        key = (payload.get("node_id"), payload.get("drug_id"))
        if all(key):
            self.hotspots.add(key)

    def _on_counterfeit(self, message) -> None:
        """A flagged batch is quarantined, never auctioned.

        Redistributing suspect stock would spread a counterfeit across the
        network under cover of an efficiency measure.
        """
        batch_id = message.payload.get("batch_id")
        if batch_id:
            self._quarantined.add(batch_id)

    # ── Observe ───────────────────────────────────────────────────────

    def observe(self, world_state: Mapping[str, Any]) -> dict[str, Any]:
        sim_day = world_state.get("sim_day", 0)
        flagged: list[dict[str, Any]] = []

        for node_id, node in world_state.get("nodes", {}).items():
            for lot in node.get("expiring_lots", []):
                if lot["days_to_expiry"] > self.alert_days:
                    continue
                if lot["batch_id"] in self._quarantined:
                    continue
                flagged.append({**lot, "node_id": node_id})

        # Observed throughput per (node, drug), as a fallback where the Demand
        # Agent has no forecast. It only forecasts consumer-facing nodes, so
        # without this a warehouse could never bid and near-expiry stock would
        # be stranded upstream — which is exactly where it was piling up.
        demand_rate: dict[str, float] = {}
        for node_id, node in world_state.get("nodes", {}).items():
            for drug_id, window in node.get("demand_history", {}).items():
                values = list(window)
                if values:
                    demand_rate[f"{node_id}|{drug_id}"] = sum(values) / len(values)

        return {
            "sim_day": sim_day,
            "flagged": flagged,
            "demand_rate": demand_rate,
            "stock": {
                node_id: node.get("stock_by_drug", {})
                for node_id, node in world_state.get("nodes", {}).items()
            },
            "capacity_free": {
                node_id: max(
                    0,
                    int(node.get("storage_capacity", 0))
                    - int(sum(node.get("stock_by_drug", {}).values())),
                )
                for node_id, node in world_state.get("nodes", {}).items()
            },
        }

    # ── Scoring ───────────────────────────────────────────────────────

    def urgency(self, lot: Mapping[str, Any]) -> float:
        """Rank flagged lots: sooner, larger, and less locally wanted first.

        Combines the three things that decide how much value is about to be
        lost — days remaining, quantity at risk, and whether the holder can
        consume it — so the auction runs in order of what is most at stake.
        """
        days_left = max(1, lot["days_to_expiry"])
        local_demand = self.forecasts.get((lot["node_id"], lot["drug_id"]), 0.0)
        at_risk = max(0.0, lot["quantity"] - local_demand * days_left)
        return (at_risk * self.unit_value) / days_left

    def _surplus(self, lot: Mapping[str, Any], rates: Mapping[str, float] | None = None) -> int:
        """Units the holder cannot plausibly sell before expiry."""
        days_left = max(0, lot["days_to_expiry"])
        local_demand = self._demand_of(lot["node_id"], lot["drug_id"], rates or {})
        return max(0, int(lot["quantity"] - local_demand * days_left))

    # ── Auction ───────────────────────────────────────────────────────

    def _candidate_buyers(self, node_id: str) -> list[str]:
        """Peers on the same tier, plus the nodes this one supplies.

        Lateral transfer alone leaves upstream stock stranded: a warehouse has
        no same-tier neighbour, so near-expiry stock there had no route out and
        simply expired. Pushing it one tier down — toward consumption — is both
        what a real distributor does and the only way that stock is ever sold.
        """
        if self.graph is None:
            return []
        from pharmadt.twin.network import lateral_peers

        peers = set(lateral_peers(self.graph, node_id))
        peers.update(self.graph.successors(node_id))
        peers.discard(node_id)
        return sorted(peers)

    def _transport_cost(self, seller: str, buyer: str, quantity: int) -> float:
        if self.graph is None or not self.graph.has_edge(seller, buyer):
            return float("inf")
        distance = float(self.graph[seller][buyer].get("distance_km", 0.0))
        return distance * self.transport_cost_per_km * quantity

    def _demand_of(self, node_id: str, drug_id: str, rates: Mapping[str, float]) -> float:
        """Forecast if the Demand Agent has one, else observed throughput."""
        forecast = self.forecasts.get((node_id, drug_id))
        if forecast is not None:
            return forecast
        return float(rates.get(f"{node_id}|{drug_id}", 0.0))

    def auction(
        self,
        lot: Mapping[str, Any],
        quantity: int,
        capacity_free: Mapping[str, int],
        rates: Mapping[str, float] | None = None,
        stock: Mapping[str, Mapping[str, int]] | None = None,
    ) -> AuctionResult:
        """Run one sealed-bid second-price auction for ``quantity`` units."""
        seller, drug_id = lot["node_id"], lot["drug_id"]
        days_left = max(0, lot["days_to_expiry"])
        reserve = quantity * self.disposal_cost

        rates = rates or {}
        bids: list[Bid] = []
        for buyer in self._candidate_buyers(seller):
            demand = self._demand_of(buyer, drug_id, rates)
            # Net off what the buyer already holds. Without this every node
            # claims it can absorb the whole lot, stock lands where it will not
            # sell, and redistribution manufactures the waste it exists to
            # prevent -- measurably so on seeds that previously wasted nothing.
            already = int((stock or {}).get(buyer, {}).get(drug_id, 0))
            headroom = max(0, int(demand * days_left) - already)
            consumable = min(quantity, headroom, capacity_free.get(buyer, 0))
            if consumable <= 0:
                continue

            value = consumable * self.unit_value
            cost = self._transport_cost(seller, buyer, consumable)
            if (buyer, drug_id) in self.hotspots:
                # A hotspot will consume it faster and with more certainty,
                # so it values the same units slightly higher.
                value *= 1.1
            amount = value - cost
            if amount > 0:
                bids.append(Bid(buyer, amount, consumable))

        if not bids:
            return AuctionResult(None, 0.0, 0, reserve, (), "no node can consume it in time")

        ranked = sorted(bids, key=lambda b: (-b.amount, b.node_id))
        winner = ranked[0]
        if winner.amount < reserve:
            return AuctionResult(
                None, 0.0, 0, reserve, tuple(ranked),
                f"best bid {winner.amount:.0f} below reserve {reserve:.0f}",
            )

        # Second price: the runner-up's bid, or the reserve if unopposed.
        clearing = ranked[1].amount if len(ranked) > 1 else reserve
        return AuctionResult(
            winner.node_id, clearing, winner.consumable, reserve, tuple(ranked),
            "cleared",
        )

    # ── Decide ────────────────────────────────────────────────────────

    def decide(self, observation: Mapping[str, Any]) -> list[Action]:
        flagged = observation.get("flagged", [])
        if not flagged:
            return []

        capacity = dict(observation.get("capacity_free", {}))
        rates = observation.get("demand_rate", {})
        stock = observation.get("stock", {})
        actions: list[Action] = []
        self.auctions = []

        # Most value at risk first, so scarce peer capacity goes where it saves
        # the most rather than to whichever lot happened to be scanned first.
        for lot in sorted(flagged, key=self.urgency, reverse=True):
            surplus = self._surplus(lot, rates)
            if surplus <= 0:
                continue

            result = self.auction(lot, surplus, capacity, rates, stock)
            self.auctions.append(result)
            if result.winner is None:
                continue

            capacity[result.winner] = max(0, capacity.get(result.winner, 0) - result.quantity)
            actions.append(
                Action(
                    action_type="REDISTRIBUTE",
                    target_node=result.winner,
                    drug_id=lot["drug_id"],
                    batch_id=lot["batch_id"],
                    quantity=result.quantity,
                    params={
                        "from_node": lot["node_id"],
                        "days_to_expiry": lot["days_to_expiry"],
                        "clearing_price": round(result.clearing_price, 2),
                        "reserve": round(result.reserve, 2),
                        "bidders": len(result.bids),
                    },
                    justification=(
                        f"{lot['batch_id']} expires in {lot['days_to_expiry']}d with "
                        f"{surplus} surplus units at {lot['node_id']}; "
                        f"{result.winner} won a second-price auction against "
                        f"{len(result.bids) - 1} rival bid(s) at "
                        f"{result.clearing_price:.0f} (reserve {result.reserve:.0f})"
                    ),
                )
            )

        return actions

    # ── Act ───────────────────────────────────────────────────────────

    def apply(self, action: Action, world: Any) -> None:
        from pharmadt.twin.processes import transfer_lot

        moved = transfer_lot(
            world,
            from_node=action.params["from_node"],
            to_node=action.target_node,
            batch_id=action.batch_id,
            drug_id=action.drug_id,
            quantity=action.quantity,
            day=int(world.env.now),
        )
        if moved:
            self.publish(
                Topic.REDISTRIBUTION_REQUEST,
                {
                    "from_node": action.params["from_node"],
                    "to_node": action.target_node,
                    "batch_id": action.batch_id,
                    "drug_id": action.drug_id,
                    "quantity": moved,
                },
                sim_day=int(world.env.now),
            )

    @property
    def cleared_auctions(self) -> list[AuctionResult]:
        return [a for a in self.auctions if a.winner is not None]
