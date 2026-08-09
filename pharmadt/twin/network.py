"""The supply-chain network as a directed graph.

Topology is manufacturer → warehouses → distributors → pharmacies, plus lateral
edges between pharmacies served by the same distributor. Those lateral edges
exist for Stage 7's redistribution agent: without them a surplus at one pharmacy
can only reach a stocked-out neighbour by travelling back up to the distributor
and down again, which is not what lateral transshipment means.

Everything here iterates sorted sequences rather than sets. Set iteration order
over strings varies with PYTHONHASHSEED, and a graph built in a different edge
order produces a different event log — which would break the reproducibility
guarantee the whole simulation rests on.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import networkx as nx

from pharmadt.config import settings
from pharmadt.core.models import NodeType

EARTH_RADIUS_KM = 6371.0088


class NodeLike(Protocol):
    """Structural type satisfied by both the ORM ``Node`` and the seed's ``NodeSpec``."""

    node_id: str
    node_type: NodeType
    lat: float
    lon: float
    storage_capacity: int
    has_cold_storage: bool


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _transit_days(distance_km: float) -> int:
    """At least one day: a same-day handoff would let stock teleport."""
    return max(1, math.ceil(distance_km / settings.transit_speed_km_per_day))


def _cost_per_unit(distance_km: float) -> float:
    return settings.transit_cost_base_per_unit + settings.transit_cost_per_km_per_unit * distance_km


def _add_edge(graph: nx.DiGraph, src: str, dst: str) -> None:
    a, b = graph.nodes[src], graph.nodes[dst]
    distance = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
    graph.add_edge(
        src,
        dst,
        distance_km=round(distance, 3),
        transit_days=_transit_days(distance),
        cost_per_unit=round(_cost_per_unit(distance), 4),
        # A cold-chain shipment needs refrigeration at both ends of the leg;
        # an origin that cannot hold stock cold has already broken the chain
        # before the vehicle departs.
        cold_chain_capable=bool(a["has_cold_storage"] and b["has_cold_storage"]),
    )


def _ids_of(graph: nx.DiGraph, node_type: NodeType) -> list[str]:
    return sorted(n for n, d in graph.nodes(data=True) if d["node_type"] is node_type)


def _nearest(graph: nx.DiGraph, source: str, candidates: Sequence[str]) -> str:
    """Nearest candidate, ties broken by node_id so the result never varies."""
    src = graph.nodes[source]
    return min(
        candidates,
        key=lambda c: (
            haversine_km(src["lat"], src["lon"], graph.nodes[c]["lat"], graph.nodes[c]["lon"]),
            c,
        ),
    )


def build_network(nodes: Sequence[NodeLike]) -> nx.DiGraph:
    """Build the directed network over ``nodes``."""
    graph = nx.DiGraph()

    for node in sorted(nodes, key=lambda n: n.node_id):
        graph.add_node(
            node.node_id,
            node_type=node.node_type,
            lat=node.lat,
            lon=node.lon,
            storage_capacity=node.storage_capacity,
            has_cold_storage=node.has_cold_storage,
        )

    manufacturers = _ids_of(graph, NodeType.MANUFACTURER)
    warehouses = _ids_of(graph, NodeType.WAREHOUSE)
    distributors = _ids_of(graph, NodeType.DISTRIBUTOR)
    retail = _ids_of(graph, NodeType.PHARMACY) + _ids_of(graph, NodeType.HOSPITAL)

    for mfg in manufacturers:
        for warehouse in warehouses:
            _add_edge(graph, mfg, warehouse)

    # Every warehouse reaches every distributor. The redundancy is deliberate:
    # Stage 9's routing agent needs more than one feasible path to optimise
    # over, and Stage 13's route-disruption scenario needs an alternative to
    # fail over to.
    for warehouse in warehouses:
        for distributor in distributors:
            _add_edge(graph, warehouse, distributor)

    # Each retail node is served by its nearest distributor.
    served: dict[str, list[str]] = {d: [] for d in distributors}
    if distributors:
        for node_id in retail:
            distributor = _nearest(graph, node_id, distributors)
            _add_edge(graph, distributor, node_id)
            served[distributor].append(node_id)

    # Lateral transshipment edges within each distributor's territory.
    for distributor in distributors:
        siblings = sorted(served[distributor])
        for i, a in enumerate(siblings):
            for b in siblings[i + 1 :]:
                _add_edge(graph, a, b)
                _add_edge(graph, b, a)

    return graph


def load_network_from_db() -> nx.DiGraph:
    """Build the network from the seeded ``nodes`` table."""
    from sqlalchemy import select

    from pharmadt.core.db import session_scope
    from pharmadt.core.models import Node

    with session_scope() as session:
        nodes = list(session.scalars(select(Node).order_by(Node.node_id)))
        # Detach plain snapshots so the graph outlives the session.
        specs = [
            NodeSnapshot(
                n.node_id, n.node_type, n.lat, n.lon, n.storage_capacity, n.has_cold_storage
            )
            for n in nodes
        ]
    return build_network(specs)


class NodeSnapshot:
    """Session-independent copy of the node fields the graph needs."""

    __slots__ = ("node_id", "node_type", "lat", "lon", "storage_capacity", "has_cold_storage")

    def __init__(
        self,
        node_id: str,
        node_type: NodeType,
        lat: float,
        lon: float,
        storage_capacity: int,
        has_cold_storage: bool,
    ) -> None:
        self.node_id = node_id
        self.node_type = node_type
        self.lat = lat
        self.lon = lon
        self.storage_capacity = storage_capacity
        self.has_cold_storage = has_cold_storage


def upstream_supplier(graph: nx.DiGraph, node_id: str) -> str | None:
    """The node this one replenishes from: fastest inbound leg, ties by id.

    Lateral pharmacy-to-pharmacy edges are excluded — a peer is a
    redistribution partner, not a supplier, and treating one as a supplier
    would let two stocked-out pharmacies order from each other forever.
    """
    node_type = graph.nodes[node_id]["node_type"]
    candidates = [
        pred
        for pred in graph.predecessors(node_id)
        if graph.nodes[pred]["node_type"] is not node_type
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda p: (graph.edges[p, node_id]["transit_days"], p))


def lateral_peers(graph: nx.DiGraph, node_id: str) -> list[str]:
    """Same-tier neighbours reachable for redistribution (Stage 7)."""
    node_type = graph.nodes[node_id]["node_type"]
    return sorted(
        succ
        for succ in graph.successors(node_id)
        if graph.nodes[succ]["node_type"] is node_type
    )


def network_summary(graph: nx.DiGraph) -> str:
    counts: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        key = str(data["node_type"])
        counts[key] = counts.get(key, 0) + 1
    tiers = ", ".join(f"{k.lower()}={v}" for k, v in sorted(counts.items()))
    return f"{graph.number_of_nodes()} nodes ({tiers}), {graph.number_of_edges()} edges"
