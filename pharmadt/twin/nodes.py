"""Runtime state of a single facility: what it holds, what it expects, what it sold.

Inventory is held as batch lots rather than a per-drug integer, because every
downstream feature needs lot identity: expiry is a property of the lot (Stage 8),
provenance is a chain over lots (Stage 4), and a recall targets a lot (Stage 13).
A single quantity column would make all three impossible.

Lots are consumed First-Expired-First-Out. FIFO by arrival would leave
short-dated stock sitting behind long-dated stock and inflate the wastage KPI
for reasons that have nothing to do with the policy being measured.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from pharmadt.config import settings
from pharmadt.core.models import NodeType

#: Rolling demand window exposed to agents, per the report's state vector.
DEMAND_HISTORY_DAYS = 28


def sim_day_of(calendar_date: date) -> int:
    """Convert a calendar date to a simulated day index."""
    return (calendar_date - settings.sim_start_date).days


def date_of(sim_day: int) -> date:
    """Convert a simulated day index back to a calendar date."""
    return settings.sim_start_date + timedelta(days=sim_day)


@dataclass(slots=True)
class Lot:
    """A quantity of one batch physically present at one node."""

    batch_id: str
    drug_id: str
    expiry_day: int
    quantity: int


@dataclass(slots=True)
class DemandProfile:
    """Daily consumer demand for one (node, drug).

    Stage 2 refits ``mean`` and ``dispersion`` per series from Rossmann and
    replaces these instances wholesale; the sampling model does not change,
    only the provenance of its parameters.
    """

    mean: float
    dispersion: float = field(default_factory=lambda: settings.demand_dispersion)
    weekend_factor: float = field(default_factory=lambda: settings.demand_weekend_factor)
    seasonal_amplitude: float = field(
        default_factory=lambda: settings.demand_seasonal_amplitude
    )

    def expected(self, sim_day: int) -> float:
        """Mean demand on ``sim_day``, with weekday and seasonal effects."""
        rate = self.mean
        if date_of(sim_day).weekday() >= 5:
            rate *= self.weekend_factor
        rate *= 1.0 + self.seasonal_amplitude * math.sin(2 * math.pi * sim_day / 365.0)
        return max(0.0, rate)

    def sample(self, sim_day: int, rng: np.random.Generator) -> int:
        """Draw one day's demand.

        Gamma-mixed Poisson, i.e. a negative binomial. Real pharmaceutical
        demand is overdispersed — a plain Poisson would have variance equal to
        its mean and would understate stockout risk, which is the KPI the whole
        project is judged on.
        """
        rate = self.expected(sim_day)
        if rate <= 0:
            return 0
        if self.dispersion <= 0:
            return int(rng.poisson(rate))
        shape = 1.0 / (self.dispersion**2)
        scale = rate / shape
        return int(rng.poisson(rng.gamma(shape, scale)))


class TwinNode:
    """One facility's mutable state across the simulation."""

    def __init__(
        self,
        node_id: str,
        node_type: NodeType,
        storage_capacity: int,
        has_cold_storage: bool,
    ) -> None:
        self.node_id = node_id
        self.node_type = node_type
        self.storage_capacity = storage_capacity
        self.has_cold_storage = has_cold_storage

        # drug_id -> lots, kept sorted by (expiry_day, batch_id) for FEFO.
        self.lots: dict[str, list[Lot]] = {}
        self.pending_inbound: dict[str, int] = {}
        self.demand_history: dict[str, deque[int]] = {}
        self.demand_profiles: dict[str, DemandProfile] = {}
        self._last_demand_day: dict[str, int] = {}

        # Manufacturers only. Stage 13's shutdown scenario sets this to zero.
        self.production_capacity_per_day: int = 0

    # ── Inventory ─────────────────────────────────────────────────────

    def stock_of(self, drug_id: str) -> int:
        return sum(lot.quantity for lot in self.lots.get(drug_id, ()))

    def total_stock(self) -> int:
        return sum(lot.quantity for lots in self.lots.values() for lot in lots)

    def free_capacity(self) -> int:
        return max(0, self.storage_capacity - self.total_stock())

    def available_space(self) -> int:
        """Free capacity less stock already on its way here.

        Ordering against :meth:`free_capacity` alone double-commits the same
        shelf space to successive orders while the first is still in transit,
        and the node ends up holding more than it can physically store.
        """
        inbound = sum(self.pending_inbound.values())
        return max(0, self.storage_capacity - self.total_stock() - inbound)

    def utilisation(self) -> float:
        if self.storage_capacity <= 0:
            return 0.0
        return min(1.0, self.total_stock() / self.storage_capacity)

    def add_lot(self, lot: Lot) -> None:
        """Receive stock, merging into an existing lot of the same batch."""
        lots = self.lots.setdefault(lot.drug_id, [])
        for existing in lots:
            if existing.batch_id == lot.batch_id:
                existing.quantity += lot.quantity
                return
        lots.append(lot)
        lots.sort(key=lambda lot_: (lot_.expiry_day, lot_.batch_id))

    def consume(self, drug_id: str, quantity: int) -> tuple[int, list[tuple[str, int]]]:
        """Draw ``quantity`` FEFO. Returns (fulfilled, [(batch_id, qty), ...])."""
        lots = self.lots.get(drug_id)
        if not lots or quantity <= 0:
            return 0, []

        remaining = quantity
        drawn: list[tuple[str, int]] = []
        for lot in lots:
            if remaining <= 0:
                break
            take = min(lot.quantity, remaining)
            if take:
                lot.quantity -= take
                remaining -= take
                drawn.append((lot.batch_id, take))

        self._drop_empty(drug_id)
        return quantity - remaining, drawn

    def remove_expired(self, sim_day: int) -> list[tuple[str, str, int]]:
        """Discard lots at or past expiry. Returns [(batch_id, drug_id, qty)]."""
        wasted: list[tuple[str, str, int]] = []
        for drug_id in sorted(self.lots):
            for lot in self.lots[drug_id]:
                if lot.expiry_day <= sim_day and lot.quantity > 0:
                    wasted.append((lot.batch_id, drug_id, lot.quantity))
                    lot.quantity = 0
            self._drop_empty(drug_id)
        return wasted

    def _drop_empty(self, drug_id: str) -> None:
        lots = [lot for lot in self.lots.get(drug_id, ()) if lot.quantity > 0]
        if lots:
            self.lots[drug_id] = lots
        else:
            self.lots.pop(drug_id, None)

    # ── Demand bookkeeping ────────────────────────────────────────────

    @property
    def sells_to_consumers(self) -> bool:
        return self.node_type in (NodeType.PHARMACY, NodeType.HOSPITAL)

    def _advance_to(self, drug_id: str, sim_day: int) -> deque[int]:
        """Extend the demand series to ``sim_day``, zero-filling quiet days.

        The window has to be a true day-indexed series, not a list of events.
        ``mean_recent_demand`` divides by its length, so any day that goes
        unrecorded would shrink the denominator and inflate the estimated rate.

        Two cases make that bite. Upstream nodes are asked for stock only every
        week or so, which would read each order lump as a single day's demand;
        and on the very first order there is no history at all, so a node would
        read one lump as its entire daily rate and order twenty-one times it.
        Both compound tier over tier — the bullwhip effect with the gain set
        far too high.
        """
        history = self.demand_history.setdefault(
            drug_id, deque(maxlen=DEMAND_HISTORY_DAYS)
        )
        last = self._last_demand_day.get(drug_id)
        first_unfilled = 0 if last is None else last + 1

        for _ in range(min(max(0, sim_day - first_unfilled), DEMAND_HISTORY_DAYS)):
            history.append(0)

        if last is None or sim_day > last:
            history.append(0)  # today's bucket, empty until something lands in it
            self._last_demand_day[drug_id] = sim_day

        return history

    def record_demand(self, drug_id: str, quantity: int, sim_day: int) -> None:
        """Add ``quantity`` to this node's demand for ``drug_id`` on ``sim_day``."""
        self._advance_to(drug_id, sim_day)[-1] += quantity

    def settle_day(self, sim_day: int) -> None:
        """Bring every tracked drug's series up to ``sim_day`` before a review."""
        for drug_id in sorted(self.demand_history):
            self._advance_to(drug_id, sim_day)

    def mean_recent_demand(self, drug_id: str) -> float:
        """Observed mean over the rolling window, falling back to the profile.

        The fallback matters on day 0: with an empty history a node would
        compute a reorder point of zero and order nothing until its first
        stockout had already happened.
        """
        history = self.demand_history.get(drug_id)
        if history:
            return sum(history) / len(history)
        profile = self.demand_profiles.get(drug_id)
        return profile.mean if profile else 0.0

    def drugs_handled(self) -> list[str]:
        """Every drug this node stocks, expects, sells, or has been asked for.

        ``demand_history`` is part of the union deliberately. A distributor that
        sells its last unit holds no lots and has nothing inbound; without the
        history it would drop the drug from this set, never review it again, and
        stay stocked out permanently.
        """
        handled = (
            set(self.lots)
            | set(self.pending_inbound)
            | set(self.demand_profiles)
            | set(self.demand_history)
        )
        return sorted(handled)

    def __repr__(self) -> str:
        return f"<TwinNode {self.node_id} {self.node_type} stock={self.total_stock()}>"
