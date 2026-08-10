"""The SimPy environment, the world it acts on, and the ``make sim`` entry point.

Determinism is the property this module exists to protect. Two runs at the same
seed must produce byte-identical event logs, because Stage 4 hashes those events
into a chain and Stage 15 compares ablation arms against each other — neither is
meaningful if the baseline wanders between runs.

Three things buy that determinism: every iteration is over a sorted sequence,
every random draw comes from an explicitly seeded ``Generator`` rather than
global state, and setup draws come from a *separate* generator so that changing
how the world is built cannot shift the simulation's random stream.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import simpy

from pharmadt.config import settings
from pharmadt.core.events import Event
from pharmadt.core.models import NodeType
from pharmadt.twin.network import build_network, network_summary
from pharmadt.twin.nodes import DemandProfile, Lot, TwinNode, sim_day_of
from pharmadt.twin.processes import (
    Order,
    demand_process,
    expiry_process,
    order_process,
)
from pharmadt.twin.state import world_state

DEFAULT_OUTPUT_DIR = Path("data/processed")

EVENT_LOG_COLUMNS = [
    "seq",
    "sim_day",
    "event_type",
    "batch_id",
    "from_node",
    "to_node",
    "payload",
]


@dataclass(frozen=True, slots=True)
class DrugSpec:
    drug_id: str
    shelf_life_days: int
    requires_cold_chain: bool
    temp_min_c: float | None
    temp_max_c: float | None


@dataclass(frozen=True, slots=True)
class BatchSpec:
    batch_id: str
    drug_id: str
    manufacturer_id: str
    created_day: int
    expiry_day: int
    quantity: int


@dataclass(slots=True)
class DemandOutcome:
    """One node/drug/day of demand — the raw material of the stockout KPI."""

    node_id: str
    drug_id: str
    sim_day: int
    demanded: int
    fulfilled: int


class World:
    """Everything the processes act on, and the only place counters live."""

    def __init__(
        self,
        env: simpy.Environment,
        graph: nx.DiGraph,
        nodes: dict[str, TwinNode],
        drugs: dict[str, DrugSpec],
        batches: dict[str, BatchSpec],
        rng: np.random.Generator,
    ) -> None:
        self.env = env
        self.graph = graph
        self.nodes = nodes
        self.drugs = drugs
        self.batches = batches
        self.rng = rng

        self.events: list[Event] = []
        self.demand_outcomes: list[DemandOutcome] = []
        self.pending_orders: dict[str, list[Order]] = defaultdict(list)

        self.wastage_units = 0
        self.excursions = 0
        self.shipments_delivered = 0
        self.inventory_samples: list[int] = []
        # Where the demand parameters came from; reported so a run's KPIs can
        # always be traced back to the data behind them.
        self.demand_source = "analytic-defaults"
        # Set by Stage 5's AgentOrchestrator when agents are attached. Left
        # None, the twin runs its Stage 3 baseline policy untouched, which is
        # exactly the control arm the agents are measured against.
        self.orchestrator: Any = None

        # Stage 6 flips this off when the Inventory Agent takes over ordering.
        self.baseline_policy_enabled = True

        self._production_used: dict[int, int] = defaultdict(int)
        self._order_seq = 0
        self._shipment_seq = 0
        self._batch_seq = 0

    # ── Identifier minting ────────────────────────────────────────────

    def next_order_id(self) -> str:
        self._order_seq += 1
        return f"ORD-{self._order_seq:07d}"

    def next_shipment_id(self) -> str:
        self._shipment_seq += 1
        return f"SHIP-{self._shipment_seq:07d}"

    def next_batch_id(self) -> str:
        self._batch_seq += 1
        return f"BATCH-SIM-{self._batch_seq:06d}"

    # ── Recording ─────────────────────────────────────────────────────

    def emit(self, event: Event) -> None:
        self.events.append(event)

    def record_demand(
        self, node_id: str, drug_id: str, sim_day: int, demanded: int, fulfilled: int
    ) -> None:
        self.demand_outcomes.append(
            DemandOutcome(node_id, drug_id, sim_day, demanded, fulfilled)
        )

    def register_batch(
        self,
        batch_id: str,
        drug_id: str,
        manufacturer_id: str,
        created_day: int,
        expiry_day: int,
        quantity: int,
    ) -> None:
        self.batches[batch_id] = BatchSpec(
            batch_id, drug_id, manufacturer_id, created_day, expiry_day, quantity
        )

    # ── Production budget ─────────────────────────────────────────────

    def production_remaining_today(self, day: int) -> int:
        return max(0, settings.production_capacity_per_day - self._production_used[day])

    def consume_production(self, day: int, quantity: int) -> None:
        self._production_used[day] += quantity

    # ── Views ─────────────────────────────────────────────────────────

    def drug_ids(self) -> list[str]:
        return sorted(self.drugs)

    def snapshot(self) -> dict[str, Any]:
        """Plain world state for agents (Stages 6-10) and the API (Stage 14)."""
        return world_state(self.nodes, int(self.env.now), self.drug_ids())

    def total_inventory(self) -> int:
        return sum(node.total_stock() for node in self.nodes.values())


# ── World construction ────────────────────────────────────────────────


def _load_reference_data() -> tuple[list[Any], dict[str, DrugSpec], list[BatchSpec]]:
    """Read nodes, drugs, and batches once, before the clock starts."""
    from sqlalchemy import select

    from pharmadt.core.db import session_scope
    from pharmadt.core.models import Batch, Drug, Node
    from pharmadt.twin.network import NodeSnapshot

    with session_scope() as session:
        nodes = [
            NodeSnapshot(
                n.node_id, n.node_type, n.lat, n.lon, n.storage_capacity, n.has_cold_storage
            )
            for n in session.scalars(select(Node).order_by(Node.node_id))
        ]
        drugs = {
            d.drug_id: DrugSpec(
                d.drug_id, d.shelf_life_days, d.requires_cold_chain, d.temp_min_c, d.temp_max_c
            )
            for d in session.scalars(select(Drug).order_by(Drug.drug_id))
        }
        batches = [
            BatchSpec(
                b.batch_id,
                b.drug_id,
                b.manufacturer_id,
                sim_day_of(b.mfg_date),
                sim_day_of(b.expiry_date),
                b.quantity,
            )
            for b in session.scalars(select(Batch).order_by(Batch.batch_id))
        ]

    if not nodes or not drugs or not batches:
        raise RuntimeError("No seed data found. Run `make migrate && make seed` first.")
    return nodes, drugs, batches


FITTED_PROFILES = DEFAULT_OUTPUT_DIR / "demand_profiles.parquet"


def _load_fitted_profiles() -> dict[tuple[str, str], DemandProfile]:
    """Demand profiles fitted from Rossmann, if Stage 2 has produced them.

    Absent, the twin falls back to the analytic defaults in ``config``, so the
    simulation still runs on a clean checkout with no datasets downloaded.
    """
    if not FITTED_PROFILES.exists():
        return {}

    import pandas as pd

    frame = pd.read_parquet(FITTED_PROFILES)
    return {
        (row.node_id, row.drug_id): DemandProfile(
            mean=float(row.mean),
            dispersion=float(row.dispersion),
            weekend_factor=float(row.weekend_factor),
            seasonal_amplitude=float(row.seasonal_amplitude),
        )
        for row in frame.itertuples()
    }


def _assign_demand_profiles(
    nodes: dict[str, TwinNode], drug_ids: list[str], rng: np.random.Generator
) -> str:
    """Give each retail node a per-drug demand profile. Returns the source used.

    Means vary between nodes so that the network has genuinely uneven pressure;
    a uniform network would make redistribution (Stage 7) pointless because no
    node would ever hold a surplus while a peer ran dry.
    """
    fitted = _load_fitted_profiles()

    for node_id in sorted(nodes):
        node = nodes[node_id]
        if not node.sells_to_consumers:
            continue
        for drug_id in drug_ids:
            profile = fitted.get((node_id, drug_id))
            if profile is None:
                scale = float(rng.uniform(0.6, 1.4))
                profile = DemandProfile(mean=settings.base_daily_demand * scale)
            else:
                # Keep the generator in step with the analytic path so that
                # switching data sources does not silently shift every other
                # random draw in the run.
                rng.uniform(0.6, 1.4)
            node.demand_profiles[drug_id] = profile

    return "rossmann-fitted" if fitted else "analytic-defaults"


def _opening_position(node: TwinNode, drug_id: str, graph: nx.DiGraph) -> int:
    """Target day-0 stock: enough to cover the (s, S) order-up-to level.

    Starting every node at zero would produce a first quarter of pure stockouts
    that says nothing about the policy under test — the KPI would be measuring
    the cold start, not the strategy.
    """
    if node.sells_to_consumers:
        profile = node.demand_profiles.get(drug_id)
        return int(profile.mean * settings.order_up_to_days) if profile else 0

    # Upstream tiers hold cover for everyone they serve.
    downstream = list(graph.successors(node.node_id))
    return len(downstream) * int(settings.base_daily_demand * settings.order_up_to_days)


def _position_opening_stock(
    world: World, graph: nx.DiGraph, batches: list[BatchSpec]
) -> None:
    """Distribute the seeded batches down the network as opening balances."""
    pool: dict[str, list[BatchSpec]] = defaultdict(list)
    for batch in sorted(batches, key=lambda b: (b.drug_id, b.expiry_day, b.batch_id)):
        pool[batch.drug_id].append(batch)
    remaining = {b.batch_id: b.quantity for b in batches}

    # Retail first, then distributors, then warehouses: the tiers that actually
    # face demand get their cover before upstream soaks up what is left.
    tier_order = (
        (NodeType.PHARMACY, NodeType.HOSPITAL),
        (NodeType.DISTRIBUTOR,),
        (NodeType.WAREHOUSE,),
    )
    for tier in tier_order:
        for node_id in sorted(world.nodes):
            node = world.nodes[node_id]
            if node.node_type not in tier:
                continue
            for drug_id in sorted(pool):
                wanted = min(_opening_position(node, drug_id, graph), node.free_capacity())
                for batch in pool[drug_id]:
                    if wanted <= 0:
                        break
                    take = min(remaining[batch.batch_id], wanted)
                    if take <= 0:
                        continue
                    remaining[batch.batch_id] -= take
                    wanted -= take
                    node.add_lot(Lot(batch.batch_id, drug_id, batch.expiry_day, take))

    # Whatever is left stays with the manufacturer that made it.
    for batch in batches:
        left = remaining[batch.batch_id]
        if left <= 0:
            continue
        maker = world.nodes.get(batch.manufacturer_id)
        if maker is not None:
            maker.add_lot(Lot(batch.batch_id, batch.drug_id, batch.expiry_day, left))


def build_world(seed: int | None = None) -> World:
    """Load reference data, build the graph, and place opening stock."""
    seed = settings.sim_seed if seed is None else seed

    # The guide asks for random and numpy.random to be seeded; do that for any
    # third-party code that reaches for global state, but drive the simulation
    # itself from explicit generators.
    random.seed(seed)
    np.random.seed(seed % (2**32))
    setup_seq, sim_seq = np.random.SeedSequence(seed).spawn(2)
    setup_rng = np.random.default_rng(setup_seq)
    sim_rng = np.random.default_rng(sim_seq)

    node_specs, drugs, batch_specs = _load_reference_data()
    graph = build_network(node_specs)

    nodes: dict[str, TwinNode] = {}
    for spec in sorted(node_specs, key=lambda n: n.node_id):
        node = TwinNode(spec.node_id, spec.node_type, spec.storage_capacity, spec.has_cold_storage)
        if spec.node_type is NodeType.MANUFACTURER:
            node.production_capacity_per_day = settings.production_capacity_per_day
        nodes[spec.node_id] = node

    drug_ids = sorted(drugs)
    demand_source = _assign_demand_profiles(nodes, drug_ids, setup_rng)

    env = simpy.Environment()
    world = World(
        env=env,
        graph=graph,
        nodes=nodes,
        drugs=drugs,
        batches={b.batch_id: b for b in batch_specs},
        rng=sim_rng,
    )
    world.demand_source = demand_source
    _position_opening_stock(world, graph, batch_specs)
    return world


def attach_agents(world: World, *agents: Any, disable_baseline: bool = True) -> Any:
    """Attach agents to a world and stand the baseline policy down.

    The twin's fixed-threshold (s, S) review and the Inventory Agent both order
    stock. Leaving both running would double every order and make the
    comparison meaningless — the agent would look like it caused the surplus it
    was actually competing with.
    """
    from pharmadt.agents.base import AgentOrchestrator

    orchestrator = AgentOrchestrator(world=world)
    orchestrator.register(*agents)
    world.orchestrator = orchestrator
    if disable_baseline:
        world.baseline_policy_enabled = False
    return orchestrator


def _monitor_process(world: World):
    """Sample total inventory daily. Instrumentation, not domain behaviour."""
    while True:
        world.inventory_samples.append(world.total_inventory())
        yield world.env.timeout(1)


def _agent_process(world: World):
    """Run one agent cycle per simulated day.

    Scheduled before the node processes so agents observe the state at the
    start of the day and any orders they publish are picked up by that same
    day's fulfilment pass. Running them last would put a full day's lag between
    every decision and its effect, which would show up as an agent that looks
    slower to react than it is.
    """
    while True:
        world.orchestrator.run_agents(world.snapshot(), int(world.env.now))
        yield world.env.timeout(1)


def run_simulation(world: World, days: int | None = None) -> World:
    """Start every process and advance the clock."""
    days = settings.sim_days if days is None else days

    # Registered first so it runs first each day; see _agent_process.
    if world.orchestrator is not None:
        world.env.process(_agent_process(world))

    for node_id in sorted(world.nodes):
        node = world.nodes[node_id]
        if node.demand_profiles:
            world.env.process(demand_process(world, node))
        world.env.process(order_process(world, node))
        world.env.process(expiry_process(world, node))

    world.env.process(_monitor_process(world))
    world.env.run(until=days)
    return world


# ── Results ───────────────────────────────────────────────────────────


def compute_kpis(world: World) -> dict[str, Any]:
    demanded = sum(o.demanded for o in world.demand_outcomes)
    fulfilled = sum(o.fulfilled for o in world.demand_outcomes)
    shortfall = demanded - fulfilled
    stockout_days = sum(1 for o in world.demand_outcomes if o.fulfilled < o.demanded)

    return {
        "sim_days": int(world.env.now),
        "nodes": len(world.nodes),
        "events": len(world.events),
        "units_demanded": demanded,
        "units_fulfilled": fulfilled,
        "units_short": shortfall,
        # Unit-weighted: the share of demand that went unmet.
        "stockout_rate": round(shortfall / demanded, 6) if demanded else 0.0,
        # Occurrence-weighted: the share of node/drug/days with any shortfall.
        "stockout_day_rate": (
            round(stockout_days / len(world.demand_outcomes), 6)
            if world.demand_outcomes
            else 0.0
        ),
        "service_level": round(fulfilled / demanded, 6) if demanded else 1.0,
        "wastage_units": world.wastage_units,
        "cold_chain_excursions": world.excursions,
        "shipments_delivered": world.shipments_delivered,
        "average_inventory": (
            round(sum(world.inventory_samples) / len(world.inventory_samples), 2)
            if world.inventory_samples
            else 0.0
        ),
    }


def event_rows(world: World) -> list[dict[str, Any]]:
    """Flatten the event log into serialisable rows, in emission order."""
    return [
        {
            "seq": i,
            "sim_day": e.sim_day,
            "event_type": str(e.event_type),
            "batch_id": e.batch_id or "",
            "from_node": e.from_node or "",
            "to_node": e.to_node or "",
            "payload": json.dumps(dict(e.payload), sort_keys=True),
        }
        for i, e in enumerate(world.events)
    ]


def write_event_log(world: World, out_dir: Path, fmt: str = "both") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = event_rows(world)
    written: list[Path] = []

    if fmt in ("json", "both"):
        path = out_dir / "event_log.json"
        path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        written.append(path)

    if fmt in ("csv", "both"):
        path = out_dir / "event_log.csv"
        # newline="" per csv docs, else Windows writes blank lines between rows.
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_LOG_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)

    return written


def persist_demand_records(world: World) -> int:
    """Bulk-insert demand outcomes after the run, never during it."""
    from pharmadt.core.db import session_scope
    from pharmadt.core.models import DemandRecord

    rows = [
        {
            "node_id": o.node_id,
            "drug_id": o.drug_id,
            "sim_day": o.sim_day,
            "quantity_demanded": o.demanded,
            "quantity_fulfilled": o.fulfilled,
        }
        for o in world.demand_outcomes
    ]
    if not rows:
        return 0
    with session_scope() as session:
        session.bulk_insert_mappings(DemandRecord, rows)
    return len(rows)


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PharmaDT digital twin.")
    parser.add_argument("--days", type=int, default=settings.sim_days)
    parser.add_argument("--seed", type=int, default=settings.sim_seed)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--format", choices=["json", "csv", "both"], default="both")
    parser.add_argument(
        "--persist", action="store_true", help="bulk-insert demand records afterwards"
    )
    parser.add_argument(
        "--anchor",
        action="store_true",
        help="append custody events to the provenance ledger after the run",
    )
    args = parser.parse_args()

    setup_started = time.perf_counter()
    world = build_world(seed=args.seed)
    setup_elapsed = time.perf_counter() - setup_started

    run_started = time.perf_counter()
    run_simulation(world, days=args.days)
    run_elapsed = time.perf_counter() - run_started

    kpis = compute_kpis(world)
    written = write_event_log(world, args.out, args.format)

    print(f"Network:  {network_summary(world.graph)}")
    print(f"Seed:     {args.seed}")
    print(f"Demand:   {world.demand_source}")
    print(f"Setup:    {setup_elapsed:.2f}s (loading reference data)")
    # NFR-01's 1000 steps/s is a property of the loop, so it is timed alone.
    print(f"Run:      {run_elapsed:.2f}s for {kpis['sim_days']} simulated days "
          f"({kpis['sim_days'] / run_elapsed:,.0f} steps/s)")
    print()
    print("Baseline KPIs (fixed-threshold (s, S) policy, no agents):")
    for key in (
        "events",
        "units_demanded",
        "units_fulfilled",
        "units_short",
        "stockout_rate",
        "stockout_day_rate",
        "service_level",
        "wastage_units",
        "cold_chain_excursions",
        "shipments_delivered",
        "average_inventory",
    ):
        print(f"  {key:<22} {kpis[key]}")

    print()
    for path in written:
        print(f"Event log: {path}")

    if args.persist:
        print(f"Persisted: {persist_demand_records(world)} demand records")
        if world.orchestrator is not None:
            from pharmadt.agents.base import persist_decisions

            written = persist_decisions(world.orchestrator.collect_decisions())
            print(f"Persisted: {written} agent decisions")

    if args.anchor:
        # Imported here so a plain `make sim` never needs the ledger or its keys.
        from pharmadt.ledger.chain import HashChainLedger

        ledger = HashChainLedger()
        anchored = ledger.anchor_events(world.events)
        print(f"Anchored:  {anchored} custody events (chain height {ledger.height()})")


if __name__ == "__main__":
    main()
