"""The agent-facing state layer.

This is the contract Stage 12 builds an observation space on, so its width has
to be a function of the drug count alone — an observation whose length depends
on what a node happens to be holding cannot be fed to a fixed-size policy.
"""

from __future__ import annotations

import json

import numpy as np

from pharmadt.core.models import NodeType
from pharmadt.twin.nodes import Lot, TwinNode
from pharmadt.twin.state import FEATURES_PER_DRUG, NodeState, node_state, world_state

DRUGS = ["D1", "D2", "D3"]


def make_node(node_id: str = "PH") -> TwinNode:
    node = TwinNode(node_id, NodeType.PHARMACY, storage_capacity=1_000, has_cold_storage=True)
    node.add_lot(Lot("B1", "D1", expiry_day=300, quantity=250))
    node.pending_inbound["D2"] = 100
    for day in range(10):
        node.record_demand("D1", 20 + day, sim_day=day)
    return node


def test_snapshot_captures_the_reported_state_fields() -> None:
    state = node_state(make_node(), sim_day=10, drugs=DRUGS)

    assert state.stock_by_drug["D1"] == 250
    assert state.pending_inbound["D2"] == 100
    assert state.temperature_zone == "COLD"
    assert state.storage_utilisation == 0.25
    assert len(state.demand_history["D1"]) == 10


def test_ambient_node_reports_ambient() -> None:
    node = TwinNode("DC", NodeType.DISTRIBUTOR, 500, has_cold_storage=False)
    assert node_state(node, 0, DRUGS).temperature_zone == "AMBIENT"


def test_vector_width_depends_only_on_the_drug_count() -> None:
    empty = node_state(TwinNode("X", NodeType.PHARMACY, 100, False), 0, DRUGS)
    stocked = node_state(make_node(), 10, DRUGS)

    expected = NodeState.vector_size(len(DRUGS))
    assert expected == len(DRUGS) * FEATURES_PER_DRUG + 2
    assert empty.to_vector(DRUGS).shape == (expected,)
    assert stocked.to_vector(DRUGS).shape == (expected,)


def test_vector_is_float32_and_roughly_normalised() -> None:
    """Unnormalised inputs make MADDPG's critics diverge."""
    vector = node_state(make_node(), 10, DRUGS).to_vector(DRUGS)
    assert vector.dtype == np.float32
    assert np.all(np.abs(vector) <= 2.0)


def test_empty_node_yields_a_finite_zero_vector() -> None:
    vector = node_state(TwinNode("X", NodeType.PHARMACY, 100, False), 0, DRUGS).to_vector(DRUGS)
    assert np.all(np.isfinite(vector))
    assert not vector[: len(DRUGS) * FEATURES_PER_DRUG].any()


def test_cold_flag_is_the_last_feature() -> None:
    cold = node_state(make_node(), 10, DRUGS).to_vector(DRUGS)
    warm_node = TwinNode("W", NodeType.PHARMACY, 1_000, has_cold_storage=False)
    warm = node_state(warm_node, 10, DRUGS).to_vector(DRUGS)
    assert cold[-1] == 1.0
    assert warm[-1] == 0.0


def test_world_state_is_json_serialisable() -> None:
    """Agents and the Stage 14 API consume plain data, never ORM rows."""
    nodes = {"PH-2": make_node("PH-2"), "PH-1": make_node("PH-1")}
    state = world_state(nodes, sim_day=10, drugs=DRUGS)

    encoded = json.dumps(state)
    assert json.loads(encoded)["sim_day"] == 10


def test_world_state_orders_nodes_deterministically() -> None:
    nodes = {"PH-2": make_node("PH-2"), "PH-1": make_node("PH-1"), "PH-3": make_node("PH-3")}
    state = world_state(nodes, sim_day=10, drugs=DRUGS)
    assert list(state["nodes"]) == ["PH-1", "PH-2", "PH-3"]


def test_snapshot_defaults_to_the_drugs_a_node_touches() -> None:
    state = node_state(make_node(), sim_day=10)
    assert set(state.stock_by_drug) == {"D1", "D2"}
