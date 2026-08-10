"""The five SimPy processes that drive the simulation forward.

Every function here is a generator and every one yields. A SimPy process that
forgets to yield does not raise — it simply never runs, and the symptom is a KPI
that is quietly zero. Each process therefore has an isolation test.

None of these touch PostgreSQL. NFR-01 asks for 1000 time steps per second and a
round trip to the database per event would put the ceiling three orders of
magnitude below that, so events accumulate in memory and are bulk-inserted after
``env.run`` returns.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import simpy

from pharmadt.config import settings
from pharmadt.core.events import Event, EventType
from pharmadt.core.models import NodeType
from pharmadt.twin.network import upstream_supplier
from pharmadt.twin.nodes import Lot, TwinNode

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from pharmadt.twin.simulation import World


@dataclass(slots=True)
class Order:
    """A replenishment request sitting in a supplier's queue.

    Partially fulfilled orders stay queued with a reduced quantity rather than
    being dropped. Dropping the remainder would let the requesting node's
    ``pending_inbound`` stay permanently inflated, and the (s, S) review would
    then never reorder — a stockout that never recovers.
    """

    order_id: str
    from_node: str
    to_node: str
    drug_id: str
    quantity: int
    placed_day: int


@dataclass(slots=True)
class ShipmentInTransit:
    """A consignment between two nodes, tracked in memory only."""

    shipment_id: str
    from_node: str
    to_node: str
    drug_id: str
    lots: list[tuple[str, int]]
    quantity: int
    dispatch_day: int
    eta_day: int
    requires_cold_chain: bool
    temp_log: list[dict[str, Any]] = field(default_factory=list)
    excursion: bool = False


# ── 1. Consumer demand ────────────────────────────────────────────────


def demand_process(world: World, node: TwinNode) -> Generator[simpy.Event, None, None]:
    """Daily consumer demand at a retail node, served FEFO."""
    env = world.env
    while True:
        day = int(env.now)
        for drug_id in sorted(node.demand_profiles):
            demanded = node.demand_profiles[drug_id].sample(day, world.rng)
            node.record_demand(drug_id, demanded, day)

            fulfilled, drawn = node.consume(drug_id, demanded)
            world.record_demand(node.node_id, drug_id, day, demanded, fulfilled)

            for batch_id, quantity in drawn:
                world.emit(
                    Event(
                        event_type=EventType.DISPENSED,
                        sim_day=day,
                        batch_id=batch_id,
                        from_node=node.node_id,
                        payload={"drug_id": drug_id, "quantity": quantity},
                    )
                )
            if fulfilled < demanded:
                world.emit(
                    Event(
                        event_type=EventType.DEMAND_UNFULFILLED,
                        sim_day=day,
                        from_node=node.node_id,
                        payload={
                            "drug_id": drug_id,
                            "demanded": demanded,
                            "fulfilled": fulfilled,
                            "shortfall": demanded - fulfilled,
                        },
                    )
                )
        yield env.timeout(1)


# ── 2. Ordering and fulfilment ────────────────────────────────────────


def order_process(world: World, node: TwinNode) -> Generator[simpy.Event, None, None]:
    """Review stock, publish replenishment orders, and fulfil inbound ones.

    The review half is the fixed-threshold (s, S) baseline the guide asks Stage 6
    to beat. When the Inventory Agent arrives it publishes orders onto the same
    queue and this function stops reviewing — the fulfilment half is unchanged.
    """
    env = world.env
    while True:
        day = int(env.now)
        if world.baseline_policy_enabled:
            _review_and_order(world, node, day)
        _fulfil_orders(world, node, day)
        yield env.timeout(1)


def _review_and_order(world: World, node: TwinNode, day: int) -> None:
    """Fixed-threshold (s, S): order up to S when the position falls below s."""
    supplier = upstream_supplier(world.graph, node.node_id)
    if supplier is None:
        return

    node.settle_day(day)

    wanted: dict[str, int] = {}
    for drug_id in node.drugs_handled():
        mean_demand = node.mean_recent_demand(drug_id)
        if mean_demand <= 0:
            continue

        position = node.stock_of(drug_id) + node.pending_inbound.get(drug_id, 0)
        if position >= mean_demand * settings.reorder_point_days:
            continue

        shortfall = int(mean_demand * settings.order_up_to_days - position)
        if shortfall > 0:
            wanted[drug_id] = shortfall

    if not wanted:
        return

    # Scale the whole basket to fit, rather than serving drugs in name order
    # until the shelf is full. First-come allocation would let DRUG-001 take
    # everything and leave DRUG-005 permanently stocked out for no reason
    # other than its position in the alphabet.
    space = node.available_space()
    total = sum(wanted.values())
    if total > space:
        scale = space / total
        wanted = {drug_id: int(q * scale) for drug_id, q in wanted.items()}

    for drug_id in sorted(wanted):
        if wanted[drug_id] > 0:
            _place_order(world, node, supplier, drug_id, wanted[drug_id], day)


def _place_order(
    world: World, node: TwinNode, supplier: str, drug_id: str, quantity: int, day: int
) -> None:
    order = Order(
        order_id=world.next_order_id(),
        from_node=supplier,
        to_node=node.node_id,
        drug_id=drug_id,
        quantity=quantity,
        placed_day=day,
    )
    world.pending_orders[supplier].append(order)
    node.pending_inbound[drug_id] = node.pending_inbound.get(drug_id, 0) + quantity

    # The supplier treats a downstream order as its own demand. Without this the
    # (s, S) review upstream sees zero demand, never reorders, and the chain
    # starves above the retail tier.
    world.nodes[supplier].record_demand(drug_id, quantity, day)

    world.emit(
        Event(
            event_type=EventType.REPLENISHMENT_ORDERED,
            sim_day=day,
            from_node=node.node_id,
            to_node=supplier,
            payload={"drug_id": drug_id, "quantity": quantity, "order_id": order.order_id},
        )
    )


def _fulfil_orders(world: World, node: TwinNode, day: int) -> None:
    """Dispatch against queued orders, oldest first, partial fills allowed."""
    queue = world.pending_orders[node.node_id]
    if not queue:
        return
    # A crisis scenario can take a node offline (Stage 13). Orders queue rather
    # than vanish, so the backlog clears when the disruption lifts -- which is
    # what makes time-to-recover a meaningful measurement.
    if node.node_id in world.disabled_nodes:
        return

    remaining_queue: list[Order] = []
    for order in queue:
        if node.node_type is NodeType.MANUFACTURER:
            _produce_if_needed(world, node, order.drug_id, order.quantity, day)

        available, drawn = node.consume(order.drug_id, order.quantity)
        if available > 0:
            _dispatch(world, node, order, drawn, available, day)
            order.quantity -= available
        if order.quantity > 0:
            remaining_queue.append(order)

    world.pending_orders[node.node_id] = remaining_queue


def _produce_if_needed(
    world: World, node: TwinNode, drug_id: str, wanted: int, day: int
) -> None:
    """Manufacture to cover a shortfall, within the day's capacity.

    Capacity is a real constraint rather than infinite supply so that Stage 13's
    factory-shutdown scenario has something to switch off.
    """
    shortfall = wanted - node.stock_of(drug_id)
    if shortfall <= 0:
        return

    # Capped by the day's capacity and by somewhere to put the output. Without
    # the space cap a single large order can run the line all day and fill the
    # plant past its own storage limit.
    budget = min(world.production_remaining_today(day), node.free_capacity())
    if budget <= 0:
        return

    # Round up to whole batches; a batch is the unit of provenance.
    size = max(1, settings.production_batch_size)
    quantity = min(budget, -(-min(shortfall, budget) // size) * size)
    if quantity <= 0:
        return

    drug = world.drugs[drug_id]
    batch_id = world.next_batch_id()
    expiry_day = day + drug.shelf_life_days

    node.add_lot(Lot(batch_id=batch_id, drug_id=drug_id, expiry_day=expiry_day, quantity=quantity))
    world.register_batch(batch_id, drug_id, node.node_id, day, expiry_day, quantity)
    world.consume_production(day, quantity)

    world.emit(
        Event(
            event_type=EventType.BATCH_CREATED,
            sim_day=day,
            batch_id=batch_id,
            to_node=node.node_id,
            payload={"drug_id": drug_id, "quantity": quantity, "expiry_day": expiry_day},
        )
    )


def _dispatch(
    world: World,
    node: TwinNode,
    order: Order,
    lots: list[tuple[str, int]],
    quantity: int,
    day: int,
) -> None:
    edge = world.graph.edges[node.node_id, order.to_node]
    shipment = ShipmentInTransit(
        shipment_id=world.next_shipment_id(),
        from_node=node.node_id,
        to_node=order.to_node,
        drug_id=order.drug_id,
        lots=lots,
        quantity=quantity,
        dispatch_day=day,
        eta_day=day + edge["transit_days"],
        requires_cold_chain=world.drugs[order.drug_id].requires_cold_chain,
    )

    for batch_id, batch_quantity in lots:
        world.emit(
            Event(
                event_type=EventType.SHIPMENT_DISPATCHED,
                sim_day=day,
                batch_id=batch_id,
                from_node=shipment.from_node,
                to_node=shipment.to_node,
                payload={
                    "shipment_id": shipment.shipment_id,
                    "drug_id": shipment.drug_id,
                    "quantity": batch_quantity,
                    "eta_day": shipment.eta_day,
                },
            )
        )

    world.env.process(shipment_process(world, shipment))
    if shipment.requires_cold_chain:
        world.env.process(coldchain_process(world, shipment))


# ── 3. Transit ────────────────────────────────────────────────────────


def shipment_process(
    world: World, shipment: ShipmentInTransit
) -> Generator[simpy.Event, None, None]:
    """Hold stock in transit, then deliver it."""
    env = world.env
    yield env.timeout(shipment.eta_day - shipment.dispatch_day)

    day = int(env.now)
    destination = world.nodes[shipment.to_node]

    for batch_id, quantity in shipment.lots:
        destination.add_lot(
            Lot(
                batch_id=batch_id,
                drug_id=shipment.drug_id,
                expiry_day=world.batches[batch_id].expiry_day,
                quantity=quantity,
            )
        )
        world.emit(
            Event(
                event_type=EventType.SHIPMENT_RECEIVED,
                sim_day=day,
                batch_id=batch_id,
                from_node=shipment.from_node,
                to_node=shipment.to_node,
                payload={
                    "shipment_id": shipment.shipment_id,
                    "drug_id": shipment.drug_id,
                    "quantity": quantity,
                    "cold_chain_breached": shipment.excursion,
                },
            )
        )

    outstanding = destination.pending_inbound.get(shipment.drug_id, 0)
    destination.pending_inbound[shipment.drug_id] = max(0, outstanding - shipment.quantity)
    world.shipments_delivered += 1


# ── 4. Expiry ─────────────────────────────────────────────────────────


def expiry_process(world: World, node: TwinNode) -> Generator[simpy.Event, None, None]:
    """Daily scan that discards expired stock — the wastage KPI."""
    env = world.env
    while True:
        yield env.timeout(1)
        day = int(env.now)
        for batch_id, drug_id, quantity in node.remove_expired(day):
            world.wastage_units += quantity
            world.emit(
                Event(
                    event_type=EventType.STOCK_EXPIRED,
                    sim_day=day,
                    batch_id=batch_id,
                    from_node=node.node_id,
                    payload={"drug_id": drug_id, "quantity": quantity},
                )
            )


# ── 5. Cold chain ─────────────────────────────────────────────────────


def coldchain_process(
    world: World, shipment: ShipmentInTransit
) -> Generator[simpy.Event, None, None]:
    """Sample temperature daily in transit and flag excursions."""
    env = world.env
    drug = world.drugs[shipment.drug_id]
    transit_days = max(1, shipment.eta_day - shipment.dispatch_day)

    for _ in range(transit_days):
        yield env.timeout(1)
        day = int(env.now)

        risk = settings.coldchain_excursion_prob_per_day * world.coldchain_risk_multiplier
        breached = world.rng.random() < min(1.0, risk)
        temp_c = settings.cold_excursion_temp_c if breached else settings.cold_setpoint_c
        shipment.temp_log.append({"sim_day": day, "temp_c": temp_c})

        if not breached:
            continue
        if drug.temp_max_c is not None and temp_c <= drug.temp_max_c:
            continue

        shipment.excursion = True
        world.excursions += 1
        for batch_id, quantity in shipment.lots:
            world.emit(
                Event(
                    event_type=EventType.COLD_CHAIN_EXCURSION,
                    sim_day=day,
                    batch_id=batch_id,
                    from_node=shipment.from_node,
                    to_node=shipment.to_node,
                    payload={
                        "shipment_id": shipment.shipment_id,
                        "drug_id": shipment.drug_id,
                        "quantity": quantity,
                        "temp_c": temp_c,
                        "temp_max_c": drug.temp_max_c,
                    },
                )
            )


def transfer_lot(
    world: World,
    from_node: str,
    to_node: str,
    batch_id: str,
    drug_id: str,
    quantity: int,
    day: int,
) -> int:
    """Move part of a lot laterally between nodes. Returns units actually moved.

    Physical stock moves immediately rather than through the shipment pipeline.
    Lateral transfers are same-tier hops between peers served by one
    distributor, and a near-expiry lot that spent two days in transit would
    often arrive with nothing left to sell -- which would make redistribution
    look useless for a reason that is an artefact of the model rather than of
    the policy.

    The batch keeps its identity across the move, so the provenance chain
    records a REDISTRIBUTION handoff for the same batch_id rather than
    inventing a new one.
    """
    source = world.nodes.get(from_node)
    destination = world.nodes.get(to_node)
    if source is None or destination is None or quantity <= 0:
        return 0

    lots = source.lots.get(drug_id, [])
    lot = next((lot_ for lot_ in lots if lot_.batch_id == batch_id), None)
    if lot is None:
        return 0

    moved = min(quantity, lot.quantity, destination.available_space())
    if moved <= 0:
        return 0

    lot.quantity -= moved
    source._drop_empty(drug_id)
    destination.add_lot(
        Lot(batch_id=batch_id, drug_id=drug_id, expiry_day=lot.expiry_day, quantity=moved)
    )

    world.emit(
        Event(
            event_type=EventType.REDISTRIBUTION,
            sim_day=day,
            batch_id=batch_id,
            from_node=from_node,
            to_node=to_node,
            payload={"drug_id": drug_id, "quantity": moved},
        )
    )
    return moved
