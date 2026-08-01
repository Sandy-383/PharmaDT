"""Seed the database with a minimal, deterministic supply chain.

Twelve nodes, five drugs, twenty batches — the Stage 1 Definition of Done, and
the fixture the twin starts from in Stage 3.

Everything derives from ``settings.sim_seed``, so two runs produce byte-identical
rows. That matters more than it looks: the Stage 4 ledger hashes batch fields, so
a seed that drifted between runs would produce different record hashes and make
chain verification impossible to reason about across runs.

Usage::

    python -m pharmadt.core.seed            # seed if empty
    python -m pharmadt.core.seed --reset    # wipe seeded tables first
"""

from __future__ import annotations

import argparse
import random
from datetime import timedelta
from typing import NamedTuple

from sqlalchemy import delete, select

from pharmadt.config import settings
from pharmadt.core.db import session_scope
from pharmadt.core.models import (
    AgentDecision,
    Batch,
    DemandRecord,
    Drug,
    InventoryRecord,
    Node,
    NodeType,
    Shipment,
    compute_batch_fingerprint,
)

# Five drugs chosen to exercise every branch of the domain model: one plain
# generic, two cold-chain products with different shelf lives, one long-life
# generic, and one controlled substance.
DRUGS: list[dict] = [
    {
        "drug_id": "DRUG-001",
        "name": "Amoxicillin 500mg Capsule",
        "atc_code": "J01CA04",
        "shelf_life_days": 730,
        "requires_cold_chain": False,
        "temp_min_c": None,
        "temp_max_c": None,
        "is_controlled": False,
    },
    {
        "drug_id": "DRUG-002",
        "name": "Insulin Glargine 100IU/mL",
        "atc_code": "A10AE04",
        "shelf_life_days": 540,
        "requires_cold_chain": True,
        "temp_min_c": 2.0,
        "temp_max_c": 8.0,
        "is_controlled": False,
    },
    {
        "drug_id": "DRUG-003",
        "name": "Metformin 850mg Tablet",
        "atc_code": "A10BA02",
        "shelf_life_days": 1095,
        "requires_cold_chain": False,
        "temp_min_c": None,
        "temp_max_c": None,
        "is_controlled": False,
    },
    {
        "drug_id": "DRUG-004",
        "name": "Influenza Vaccine (Quadrivalent)",
        "atc_code": "J07BB02",
        "shelf_life_days": 365,
        "requires_cold_chain": True,
        "temp_min_c": 2.0,
        "temp_max_c": 8.0,
        "is_controlled": False,
    },
    {
        "drug_id": "DRUG-005",
        "name": "Morphine Sulfate 10mg/mL",
        "atc_code": "N02AA01",
        "shelf_life_days": 1095,
        "requires_cold_chain": False,
        "temp_min_c": None,
        "temp_max_c": None,
        "is_controlled": True,
    },
]

class NodeSpec(NamedTuple):
    """One row of the network fixture, named so the table below reads itself."""

    node_id: str
    name: str
    node_type: NodeType
    lat: float
    lon: float
    storage_capacity: int
    has_cold_storage: bool


# 1 manufacturer, 2 warehouses, 3 distributors, 6 pharmacies — the topology the
# guide prescribes for Stage 3 before scaling to the 50 nodes FR-01 requires.
# Coordinates are real locations across Karnataka so the Stage 9 routing
# distances and the Stage 14 map are meaningful rather than arbitrary.
NODES: list[NodeSpec] = [
    NodeSpec("NODE-MFG-01", "Bengaluru Manufacturing Plant",
             NodeType.MANUFACTURER, 12.9716, 77.5946, 500_000, True),
    NodeSpec("NODE-WH-01", "Bengaluru Central Warehouse",
             NodeType.WAREHOUSE, 12.9141, 77.6101, 200_000, True),
    NodeSpec("NODE-WH-02", "Hubballi Regional Warehouse",
             NodeType.WAREHOUSE, 15.3647, 75.1240, 200_000, True),
    NodeSpec("NODE-DC-01", "Mysuru Distribution Centre",
             NodeType.DISTRIBUTOR, 12.2958, 76.6394, 50_000, True),
    NodeSpec("NODE-DC-02", "Mangaluru Distribution Centre",
             NodeType.DISTRIBUTOR, 12.9141, 74.8560, 50_000, True),
    NodeSpec("NODE-DC-03", "Kalaburagi Distribution Centre",
             NodeType.DISTRIBUTOR, 17.3297, 76.8343, 50_000, False),
    NodeSpec("NODE-PH-01", "Jayanagar Pharmacy",
             NodeType.PHARMACY, 12.9250, 77.5938, 5_000, True),
    NodeSpec("NODE-PH-02", "Whitefield Pharmacy",
             NodeType.PHARMACY, 12.9698, 77.7500, 5_000, False),
    NodeSpec("NODE-PH-03", "Mysuru City Pharmacy",
             NodeType.PHARMACY, 12.3052, 76.6552, 5_000, True),
    NodeSpec("NODE-PH-04", "Mangaluru Port Pharmacy",
             NodeType.PHARMACY, 12.8698, 74.8430, 5_000, False),
    NodeSpec("NODE-PH-05", "Belagavi Pharmacy",
             NodeType.PHARMACY, 15.8497, 74.4977, 5_000, False),
    NodeSpec("NODE-PH-06", "Shivamogga Pharmacy",
             NodeType.PHARMACY, 13.9299, 75.5681, 5_000, True),
]

N_BATCHES = 20


def _build_drugs() -> list[Drug]:
    return [Drug(**spec) for spec in DRUGS]


def _build_nodes() -> list[Node]:
    # NodeSpec field names match the Node columns exactly, so this stays correct
    # if a column is added rather than silently dropping it.
    return [Node(**spec._asdict()) for spec in NODES]


def _build_batches(rng: random.Random) -> list[Batch]:
    """Twenty batches from the manufacturer, staggered across the past year.

    Manufacture dates are spread backwards from the simulation epoch so that a
    few batches are already close to expiry on day 0. Without that spread the
    Expiry Agent in Stage 8 would have nothing to detect for its first two
    simulated years, and the wastage KPI would read a meaningless zero.
    """
    manufacturer_id = "NODE-MFG-01"
    batches: list[Batch] = []

    for i in range(N_BATCHES):
        drug = DRUGS[i % len(DRUGS)]
        batch_id = f"BATCH-{i + 1:04d}"
        mfg_date = settings.sim_start_date - timedelta(days=rng.randint(0, 300))
        expiry_date = mfg_date + timedelta(days=drug["shelf_life_days"])
        quantity = rng.randrange(1_000, 20_001, 500)

        batches.append(
            Batch(
                batch_id=batch_id,
                drug_id=drug["drug_id"],
                manufacturer_id=manufacturer_id,
                mfg_date=mfg_date,
                expiry_date=expiry_date,
                quantity=quantity,
                batch_fingerprint=compute_batch_fingerprint(
                    batch_id,
                    drug["drug_id"],
                    manufacturer_id,
                    mfg_date,
                    expiry_date,
                    quantity,
                ),
            )
        )
    return batches


def reset() -> None:
    """Delete all seeded data, children before parents."""
    with session_scope() as session:
        for model in (
            AgentDecision,
            DemandRecord,
            InventoryRecord,
            Shipment,
            Batch,
            Drug,
            Node,
        ):
            session.execute(delete(model))


def seed(force: bool = False) -> dict[str, int]:
    """Insert the fixture. Returns row counts. Idempotent unless ``force``."""
    if force:
        reset()

    with session_scope() as session:
        if session.scalar(select(Drug).limit(1)) is not None:
            raise RuntimeError(
                "Database already contains seed data. Re-run with --reset to replace it."
            )

        rng = random.Random(settings.sim_seed)

        nodes = _build_nodes()
        drugs = _build_drugs()
        session.add_all(nodes)
        session.add_all(drugs)
        # Batches carry FKs to both, so flush before inserting them.
        session.flush()

        batches = _build_batches(rng)
        session.add_all(batches)

        return {"nodes": len(nodes), "drugs": len(drugs), "batches": len(batches)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the PharmaDT database.")
    parser.add_argument(
        "--reset", action="store_true", help="delete existing seed data first"
    )
    args = parser.parse_args()

    counts = seed(force=args.reset)
    print(
        f"Seeded {counts['nodes']} nodes, {counts['drugs']} drugs, "
        f"{counts['batches']} batches (seed={settings.sim_seed})."
    )


if __name__ == "__main__":
    main()
