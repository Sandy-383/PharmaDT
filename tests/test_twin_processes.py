"""Each SimPy process exercised in isolation.

The guide's warning is specific: a process that forgets to ``yield`` does not
raise, it simply never runs, and the only symptom is a KPI that reads zero. Two
guards here address that directly — a check that every process is a generator
function, and a test per process that it changes the state it is supposed to.
"""

from __future__ import annotations

import inspect

import networkx as nx
import numpy as np
import pytest
import simpy

from pharmadt.config import settings
from pharmadt.core.events import EventType
from pharmadt.core.models import NodeType
from pharmadt.twin.nodes import DemandProfile, Lot, TwinNode
from pharmadt.twin.processes import (
    ShipmentInTransit,
    coldchain_process,
    demand_process,
    expiry_process,
    order_process,
    shipment_process,
)
from pharmadt.twin.simulation import BatchSpec, DrugSpec, World

ALL_PROCESSES = [
    demand_process,
    order_process,
    shipment_process,
    expiry_process,
    coldchain_process,
]


def make_world(*, transit_days: int = 2, cold_chain: bool = False) -> World:
    """A two-node world: one supplier, one pharmacy, one drug."""
    graph = nx.DiGraph()
    graph.add_node(
        "SUP", node_type=NodeType.WAREHOUSE, lat=0.0, lon=0.0,
        storage_capacity=1_000_000, has_cold_storage=True,
    )
    graph.add_node(
        "PH", node_type=NodeType.PHARMACY, lat=0.0, lon=1.0,
        storage_capacity=5_000, has_cold_storage=True,
    )
    graph.add_edge(
        "SUP", "PH", distance_km=100.0, transit_days=transit_days,
        cost_per_unit=1.0, cold_chain_capable=True,
    )

    return World(
        env=simpy.Environment(),
        graph=graph,
        nodes={
            "SUP": TwinNode("SUP", NodeType.WAREHOUSE, 1_000_000, True),
            "PH": TwinNode("PH", NodeType.PHARMACY, 5_000, True),
        },
        drugs={
            "D1": DrugSpec(
                "D1", 365, cold_chain, 2.0 if cold_chain else None, 8.0 if cold_chain else None
            )
        },
        batches={"B1": BatchSpec("B1", "D1", "SUP", 0, 300, 100_000)},
        rng=np.random.default_rng(0),
    )


def flat_profile(mean: float) -> DemandProfile:
    """Deterministic demand — no weekday, seasonal, or dispersion noise."""
    return DemandProfile(
        mean=mean, dispersion=0.0, weekend_factor=1.0, seasonal_amplitude=0.0
    )


@pytest.mark.parametrize("process", ALL_PROCESSES, ids=lambda f: f.__name__)
def test_every_process_is_a_generator_function(process) -> None:
    """A non-generator 'process' is scheduled and silently never runs."""
    assert inspect.isgeneratorfunction(process)


# ── 1. Demand ─────────────────────────────────────────────────────────


def test_demand_process_consumes_stock_and_logs_outcomes() -> None:
    world = make_world()
    pharmacy = world.nodes["PH"]
    pharmacy.demand_profiles["D1"] = flat_profile(10.0)
    pharmacy.add_lot(Lot("B1", "D1", expiry_day=300, quantity=1_000))

    world.env.process(demand_process(world, pharmacy))
    world.env.run(until=5)

    assert len(world.demand_outcomes) == 5
    assert pharmacy.stock_of("D1") < 1_000
    assert any(e.event_type is EventType.DISPENSED for e in world.events)
    assert len(pharmacy.demand_history["D1"]) == 5


def test_demand_process_flags_a_stockout_when_stock_runs_out() -> None:
    world = make_world()
    pharmacy = world.nodes["PH"]
    pharmacy.demand_profiles["D1"] = flat_profile(10.0)

    world.env.process(demand_process(world, pharmacy))
    world.env.run(until=3)

    shortfalls = [e for e in world.events if e.event_type is EventType.DEMAND_UNFULFILLED]
    assert len(shortfalls) == 3
    assert all(o.fulfilled == 0 for o in world.demand_outcomes)
    assert shortfalls[0].payload["shortfall"] > 0


# ── 2. Ordering and fulfilment ────────────────────────────────────────


def test_order_process_replenishes_a_depleted_pharmacy() -> None:
    world = make_world(transit_days=1)
    supplier, pharmacy = world.nodes["SUP"], world.nodes["PH"]
    supplier.add_lot(Lot("B1", "D1", expiry_day=300, quantity=100_000))
    pharmacy.demand_profiles["D1"] = flat_profile(50.0)

    world.env.process(demand_process(world, pharmacy))
    world.env.process(order_process(world, pharmacy))
    world.env.process(order_process(world, supplier))
    world.env.run(until=20)

    kinds = {e.event_type for e in world.events}
    assert EventType.REPLENISHMENT_ORDERED in kinds
    assert EventType.SHIPMENT_DISPATCHED in kinds
    assert EventType.SHIPMENT_RECEIVED in kinds
    assert pharmacy.stock_of("D1") > 0
    assert supplier.stock_of("D1") < 100_000


def test_ordering_never_exceeds_available_space() -> None:
    world = make_world(transit_days=1)
    supplier, pharmacy = world.nodes["SUP"], world.nodes["PH"]
    supplier.add_lot(Lot("B1", "D1", expiry_day=300, quantity=1_000_000))
    pharmacy.demand_profiles["D1"] = flat_profile(400.0)

    world.env.process(demand_process(world, pharmacy))
    world.env.process(order_process(world, pharmacy))
    world.env.process(order_process(world, supplier))
    world.env.run(until=60)

    assert pharmacy.total_stock() <= pharmacy.storage_capacity


def test_basket_is_scaled_rather_than_served_in_name_order() -> None:
    """Regression: capacity used to go to whichever drug sorted first.

    DRUG-001 took the whole shelf and later drugs stocked out permanently for
    no reason other than their position in the alphabet.
    """
    world = make_world(transit_days=1)
    supplier, pharmacy = world.nodes["SUP"], world.nodes["PH"]
    world.drugs["D2"] = DrugSpec("D2", 365, False, None, None)
    world.batches["B2"] = BatchSpec("B2", "D2", "SUP", 0, 300, 100_000)
    supplier.add_lot(Lot("B1", "D1", expiry_day=300, quantity=100_000))
    supplier.add_lot(Lot("B2", "D2", expiry_day=300, quantity=100_000))

    for drug_id in ("D1", "D2"):
        pharmacy.demand_profiles[drug_id] = flat_profile(300.0)

    world.env.process(demand_process(world, pharmacy))
    world.env.process(order_process(world, pharmacy))
    world.env.process(order_process(world, supplier))
    world.env.run(until=40)

    ordered = {
        e.payload["drug_id"]
        for e in world.events
        if e.event_type is EventType.REPLENISHMENT_ORDERED
    }
    assert ordered == {"D1", "D2"}


def test_manufacturer_produces_to_cover_a_shortfall() -> None:
    world = make_world(transit_days=1)
    world.nodes["SUP"] = TwinNode("SUP", NodeType.MANUFACTURER, 1_000_000, True)
    world.graph.nodes["SUP"]["node_type"] = NodeType.MANUFACTURER
    pharmacy = world.nodes["PH"]
    pharmacy.demand_profiles["D1"] = flat_profile(50.0)

    world.env.process(demand_process(world, pharmacy))
    world.env.process(order_process(world, pharmacy))
    world.env.process(order_process(world, world.nodes["SUP"]))
    world.env.run(until=25)

    created = [e for e in world.events if e.event_type is EventType.BATCH_CREATED]
    assert created, "manufacturer never produced despite unfilled orders"
    assert created[0].payload["quantity"] > 0


def test_production_respects_the_daily_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 13's shutdown scenario works by driving this to zero."""
    monkeypatch.setattr(settings, "production_capacity_per_day", 0)

    world = make_world(transit_days=1)
    world.nodes["SUP"] = TwinNode("SUP", NodeType.MANUFACTURER, 1_000_000, True)
    world.graph.nodes["SUP"]["node_type"] = NodeType.MANUFACTURER
    pharmacy = world.nodes["PH"]
    pharmacy.demand_profiles["D1"] = flat_profile(50.0)

    world.env.process(demand_process(world, pharmacy))
    world.env.process(order_process(world, pharmacy))
    world.env.process(order_process(world, world.nodes["SUP"]))
    world.env.run(until=20)

    assert not [e for e in world.events if e.event_type is EventType.BATCH_CREATED]


# ── 3. Transit ────────────────────────────────────────────────────────


def test_shipment_is_held_for_the_full_transit_time() -> None:
    world = make_world(transit_days=3)
    pharmacy = world.nodes["PH"]
    pharmacy.pending_inbound["D1"] = 100

    shipment = ShipmentInTransit(
        shipment_id="S1", from_node="SUP", to_node="PH", drug_id="D1",
        lots=[("B1", 100)], quantity=100, dispatch_day=0, eta_day=3,
        requires_cold_chain=False,
    )
    world.env.process(shipment_process(world, shipment))

    world.env.run(until=2)
    assert pharmacy.stock_of("D1") == 0, "stock arrived before its ETA"

    world.env.run(until=5)
    assert pharmacy.stock_of("D1") == 100
    assert pharmacy.pending_inbound["D1"] == 0
    assert world.shipments_delivered == 1


# ── 4. Expiry ─────────────────────────────────────────────────────────


def test_expiry_process_discards_expired_stock() -> None:
    """Proves the wastage KPI reads zero because nothing expired, not because
    the process silently never ran."""
    world = make_world()
    pharmacy = world.nodes["PH"]
    pharmacy.add_lot(Lot("B1", "D1", expiry_day=3, quantity=500))

    world.env.process(expiry_process(world, pharmacy))
    world.env.run(until=10)

    assert pharmacy.stock_of("D1") == 0
    assert world.wastage_units == 500
    expired = [e for e in world.events if e.event_type is EventType.STOCK_EXPIRED]
    assert len(expired) == 1
    assert expired[0].payload["quantity"] == 500


def test_expiry_process_leaves_in_date_stock_alone() -> None:
    world = make_world()
    pharmacy = world.nodes["PH"]
    pharmacy.add_lot(Lot("B1", "D1", expiry_day=999, quantity=500))

    world.env.process(expiry_process(world, pharmacy))
    world.env.run(until=30)

    assert pharmacy.stock_of("D1") == 500
    assert world.wastage_units == 0


# ── 5. Cold chain ─────────────────────────────────────────────────────


def _cold_shipment() -> ShipmentInTransit:
    return ShipmentInTransit(
        shipment_id="S1", from_node="SUP", to_node="PH", drug_id="D1",
        lots=[("B1", 100)], quantity=100, dispatch_day=0, eta_day=4,
        requires_cold_chain=True,
    )


def test_coldchain_process_flags_a_breach(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "coldchain_excursion_prob_per_day", 1.0)

    world = make_world(transit_days=4, cold_chain=True)
    shipment = _cold_shipment()
    world.env.process(coldchain_process(world, shipment))
    world.env.run(until=6)

    assert shipment.excursion is True
    assert world.excursions > 0
    breaches = [e for e in world.events if e.event_type is EventType.COLD_CHAIN_EXCURSION]
    assert breaches
    assert breaches[0].payload["temp_c"] > breaches[0].payload["temp_max_c"]


def test_coldchain_process_logs_temperature_every_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "coldchain_excursion_prob_per_day", 0.0)

    world = make_world(transit_days=4, cold_chain=True)
    shipment = _cold_shipment()
    world.env.process(coldchain_process(world, shipment))
    world.env.run(until=6)

    assert len(shipment.temp_log) == 4
    assert shipment.excursion is False
    assert world.excursions == 0
