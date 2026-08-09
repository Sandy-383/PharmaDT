"""Network construction: topology, edge attributes, and build determinism."""

from __future__ import annotations

import pytest

from pharmadt.core.models import NodeType
from pharmadt.twin.network import (
    NodeSnapshot,
    build_network,
    haversine_km,
    lateral_peers,
    network_summary,
    upstream_supplier,
)

TOPOLOGY = [
    ("MFG", NodeType.MANUFACTURER, 12.9716, 77.5946, 500_000, True),
    ("WH-1", NodeType.WAREHOUSE, 12.9141, 77.6101, 200_000, True),
    ("WH-2", NodeType.WAREHOUSE, 15.3647, 75.1240, 200_000, False),
    ("DC-1", NodeType.DISTRIBUTOR, 12.2958, 76.6394, 50_000, True),
    ("DC-2", NodeType.DISTRIBUTOR, 17.3297, 76.8343, 50_000, False),
    ("PH-1", NodeType.PHARMACY, 12.3052, 76.6552, 5_000, True),
    ("PH-2", NodeType.PHARMACY, 12.2900, 76.6300, 5_000, False),
    ("PH-3", NodeType.PHARMACY, 17.3400, 76.8400, 5_000, True),
]


@pytest.fixture
def graph():
    return build_network([NodeSnapshot(*spec) for spec in TOPOLOGY])


# ── Geometry ──────────────────────────────────────────────────────────


def test_haversine_matches_a_known_distance() -> None:
    # Bengaluru to Mysuru is roughly 125 km great-circle.
    assert haversine_km(12.9716, 77.5946, 12.2958, 76.6394) == pytest.approx(125, abs=6)


def test_haversine_is_zero_for_the_same_point() -> None:
    assert haversine_km(12.0, 77.0, 12.0, 77.0) == pytest.approx(0.0)


# ── Topology ──────────────────────────────────────────────────────────


def test_manufacturer_reaches_every_warehouse(graph) -> None:
    assert sorted(graph.successors("MFG")) == ["WH-1", "WH-2"]


def test_every_warehouse_reaches_every_distributor(graph) -> None:
    """Redundant by design: Stage 9 needs alternatives to optimise over and
    Stage 13's route disruption needs somewhere to fail over to."""
    for warehouse in ("WH-1", "WH-2"):
        assert {"DC-1", "DC-2"} <= set(graph.successors(warehouse))


def test_pharmacies_attach_to_their_nearest_distributor(graph) -> None:
    assert graph.has_edge("DC-1", "PH-1")
    assert graph.has_edge("DC-1", "PH-2")
    assert graph.has_edge("DC-2", "PH-3")
    assert not graph.has_edge("DC-2", "PH-1")


def test_lateral_edges_join_peers_in_both_directions(graph) -> None:
    assert graph.has_edge("PH-1", "PH-2")
    assert graph.has_edge("PH-2", "PH-1")


def test_isolated_pharmacy_has_no_lateral_peers(graph) -> None:
    assert lateral_peers(graph, "PH-3") == []
    assert lateral_peers(graph, "PH-1") == ["PH-2"]


# ── Edge attributes ───────────────────────────────────────────────────


def test_every_edge_carries_the_full_attribute_set(graph) -> None:
    for _, _, data in graph.edges(data=True):
        assert data.keys() >= {
            "distance_km",
            "transit_days",
            "cost_per_unit",
            "cold_chain_capable",
        }


def test_transit_is_never_instantaneous(graph) -> None:
    """A zero-day leg would let stock teleport between nodes."""
    for _, _, data in graph.edges(data=True):
        assert data["transit_days"] >= 1


def test_cold_chain_requires_refrigeration_at_both_ends(graph) -> None:
    assert graph.edges["MFG", "WH-1"]["cold_chain_capable"] is True
    assert graph.edges["MFG", "WH-2"]["cold_chain_capable"] is False
    assert graph.edges["DC-1", "PH-2"]["cold_chain_capable"] is False


def test_cost_rises_with_distance(graph) -> None:
    near = graph.edges["DC-1", "PH-1"]
    far = graph.edges["MFG", "WH-2"]
    assert far["distance_km"] > near["distance_km"]
    assert far["cost_per_unit"] > near["cost_per_unit"]


# ── Supplier resolution ───────────────────────────────────────────────


def test_supplier_is_the_upstream_tier(graph) -> None:
    assert upstream_supplier(graph, "PH-1") == "DC-1"
    assert upstream_supplier(graph, "DC-1") in {"WH-1", "WH-2"}
    assert upstream_supplier(graph, "WH-1") == "MFG"


def test_manufacturer_has_no_supplier(graph) -> None:
    assert upstream_supplier(graph, "MFG") is None


def test_a_peer_is_never_treated_as_a_supplier(graph) -> None:
    """Otherwise two stocked-out pharmacies would order from each other forever."""
    assert upstream_supplier(graph, "PH-2") == "DC-1"


# ── Determinism ───────────────────────────────────────────────────────


def test_build_is_deterministic_including_edge_order() -> None:
    """Set iteration over strings varies with PYTHONHASHSEED; a different edge
    order would produce a different event log from the same seed."""
    specs = [NodeSnapshot(*spec) for spec in TOPOLOGY]
    first = build_network(specs)
    second = build_network(list(reversed(specs)))

    assert list(first.nodes) == list(second.nodes)
    assert list(first.edges) == list(second.edges)


def test_summary_reports_the_tier_counts(graph) -> None:
    summary = network_summary(graph)
    assert "8 nodes" in summary
    assert "pharmacy=3" in summary
