"""SQLAlchemy entities for the pharmaceutical supply chain.

Constraints are declared at the database level rather than only in application
code. A stockout bug that writes negative inventory should fail loudly at the
INSERT, not silently corrupt a KPI that ends up in the final report.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pharmadt.core.db import Base


class NodeType(StrEnum):
    """Tier of a node in the distribution network."""

    MANUFACTURER = "MANUFACTURER"
    WAREHOUSE = "WAREHOUSE"
    DISTRIBUTOR = "DISTRIBUTOR"
    PHARMACY = "PHARMACY"
    HOSPITAL = "HOSPITAL"


class ShipmentStatus(StrEnum):
    """Lifecycle of a shipment between two nodes."""

    PENDING = "PENDING"
    IN_TRANSIT = "IN_TRANSIT"
    DELIVERED = "DELIVERED"
    DELAYED = "DELAYED"
    CANCELLED = "CANCELLED"


def compute_batch_fingerprint(
    batch_id: str,
    drug_id: str,
    manufacturer_id: str,
    mfg_date: date,
    expiry_date: date,
    quantity: int,
) -> str:
    """Return the canonical SHA-256 identity fingerprint of a batch.

    This is the single source of truth for the anti-counterfeit check: Stage 4's
    ``verify_batch_fingerprint`` recomputes it with this same function. If
    creation and verification ever computed it differently, every batch would
    read as counterfeit — so neither side is allowed its own copy of the rule.

    Fields are joined with an explicit ``|`` separator rather than concatenated.
    Bare concatenation is ambiguous — ``("AB", "C")`` and ``("A", "BC")`` would
    produce the same digest — which is exactly the collision a forger would
    reach for.
    """
    canonical = "|".join(
        [
            batch_id,
            drug_id,
            manufacturer_id,
            mfg_date.isoformat(),
            expiry_date.isoformat(),
            str(quantity),
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Drug(Base):
    """A pharmaceutical product. Cold-chain and controlled-substance aware."""

    __tablename__ = "drugs"

    drug_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # WHO Anatomical Therapeutic Chemical code, e.g. "J01CA04".
    atc_code: Mapped[str | None] = mapped_column(String(16))
    shelf_life_days: Mapped[int] = mapped_column(Integer, nullable=False)
    requires_cold_chain: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    temp_min_c: Mapped[float | None] = mapped_column(Float)
    temp_max_c: Mapped[float | None] = mapped_column(Float)
    is_controlled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    batches: Mapped[list[Batch]] = relationship(back_populates="drug")

    __table_args__ = (
        CheckConstraint("shelf_life_days > 0", name="ck_drug_shelf_life_positive"),
        CheckConstraint(
            "temp_min_c IS NULL OR temp_max_c IS NULL OR temp_min_c < temp_max_c",
            name="ck_drug_temp_range_ordered",
        ),
        # A cold-chain drug with no temperature band is unenforceable: the
        # cold-chain process in Stage 3 would have no threshold to breach.
        CheckConstraint(
            "NOT requires_cold_chain OR (temp_min_c IS NOT NULL AND temp_max_c IS NOT NULL)",
            name="ck_drug_cold_chain_has_range",
        ),
    )

    def __repr__(self) -> str:
        return f"<Drug {self.drug_id} {self.name!r}>"


class Node(Base):
    """A facility in the network.

    ``public_key`` is the permissioning layer for the Stage 4 ledger: a record
    signed by a key absent from this registry is rejected. It is nullable
    because nodes exist from Stage 1 but keypairs are only issued in Stage 4.
    """

    __tablename__ = "nodes"

    node_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    node_type: Mapped[NodeType] = mapped_column(
        SAEnum(NodeType, name="node_type"), nullable=False, index=True
    )
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    storage_capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    has_cold_storage: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    public_key: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("lat BETWEEN -90 AND 90", name="ck_node_lat_range"),
        CheckConstraint("lon BETWEEN -180 AND 180", name="ck_node_lon_range"),
        CheckConstraint("storage_capacity >= 0", name="ck_node_capacity_nonneg"),
    )

    def __repr__(self) -> str:
        return f"<Node {self.node_id} {self.node_type}>"


class Batch(Base):
    """A manufactured lot of one drug — the unit of provenance tracking."""

    __tablename__ = "batches"

    batch_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    drug_id: Mapped[str] = mapped_column(
        ForeignKey("drugs.drug_id"), nullable=False, index=True
    )
    manufacturer_id: Mapped[str] = mapped_column(
        ForeignKey("nodes.node_id"), nullable=False, index=True
    )
    mfg_date: Mapped[date] = mapped_column(Date, nullable=False)
    expiry_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    batch_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)

    drug: Mapped[Drug] = relationship(back_populates="batches")
    manufacturer: Mapped[Node] = relationship()

    __table_args__ = (
        CheckConstraint("expiry_date > mfg_date", name="ck_batch_expiry_after_mfg"),
        CheckConstraint("quantity >= 0", name="ck_batch_quantity_nonneg"),
        CheckConstraint(
            "char_length(batch_fingerprint) = 64", name="ck_batch_fingerprint_sha256"
        ),
    )

    def compute_fingerprint(self) -> str:
        """Recompute this batch's fingerprint from its own fields."""
        return compute_batch_fingerprint(
            self.batch_id,
            self.drug_id,
            self.manufacturer_id,
            self.mfg_date,
            self.expiry_date,
            self.quantity,
        )

    def __repr__(self) -> str:
        return f"<Batch {self.batch_id} drug={self.drug_id} qty={self.quantity}>"


class InventoryRecord(Base):
    """Stock of one batch held at one node on one simulated day."""

    __tablename__ = "inventory_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"), nullable=False)
    batch_id: Mapped[str] = mapped_column(ForeignKey("batches.batch_id"), nullable=False)
    quantity_on_hand: Mapped[int] = mapped_column(Integer, nullable=False)
    sim_day: Mapped[int] = mapped_column(Integer, nullable=False)

    node: Mapped[Node] = relationship()
    batch: Mapped[Batch] = relationship()

    __table_args__ = (
        # One row per node/batch/day. Without this, a double-write in the twin
        # loop would double-count stock and quietly inflate the inventory KPI.
        UniqueConstraint(
            "node_id", "batch_id", "sim_day", name="uq_inventory_node_batch_day"
        ),
        CheckConstraint("quantity_on_hand >= 0", name="ck_inventory_qty_nonneg"),
        Index("ix_inventory_node_day", "node_id", "sim_day"),
    )

    def __repr__(self) -> str:
        return (
            f"<InventoryRecord {self.node_id}/{self.batch_id} "
            f"day={self.sim_day} qty={self.quantity_on_hand}>"
        )


class Shipment(Base):
    """A quantity of one batch moving between two nodes.

    ``temp_log`` accumulates ``{sim_day, temp_c}`` samples from the Stage 3
    cold-chain process; excursions against the drug's band become
    ``COLD_CHAIN_EXCURSION`` ledger events.
    """

    __tablename__ = "shipments"

    shipment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    from_node: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"), nullable=False)
    to_node: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"), nullable=False)
    batch_id: Mapped[str] = mapped_column(
        ForeignKey("batches.batch_id"), nullable=False, index=True
    )
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    dispatch_day: Mapped[int] = mapped_column(Integer, nullable=False)
    eta_day: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[ShipmentStatus] = mapped_column(
        SAEnum(ShipmentStatus, name="shipment_status"),
        nullable=False,
        default=ShipmentStatus.PENDING,
        index=True,
    )
    temp_log: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )

    origin: Mapped[Node] = relationship(foreign_keys=[from_node])
    destination: Mapped[Node] = relationship(foreign_keys=[to_node])
    batch: Mapped[Batch] = relationship()

    __table_args__ = (
        CheckConstraint("quantity > 0", name="ck_shipment_quantity_positive"),
        CheckConstraint("eta_day >= dispatch_day", name="ck_shipment_eta_after_dispatch"),
        CheckConstraint("from_node <> to_node", name="ck_shipment_distinct_endpoints"),
    )

    def __repr__(self) -> str:
        return f"<Shipment {self.shipment_id} {self.from_node}->{self.to_node} {self.status}>"


class DemandRecord(Base):
    """Demand raised and demand met, per node/drug/day.

    The gap between the two columns is the stockout KPI, which is the headline
    number the whole project is judged on.
    """

    __tablename__ = "demand_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"), nullable=False)
    drug_id: Mapped[str] = mapped_column(ForeignKey("drugs.drug_id"), nullable=False)
    sim_day: Mapped[int] = mapped_column(Integer, nullable=False)
    quantity_demanded: Mapped[int] = mapped_column(Integer, nullable=False)
    quantity_fulfilled: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    node: Mapped[Node] = relationship()
    drug: Mapped[Drug] = relationship()

    __table_args__ = (
        UniqueConstraint("node_id", "drug_id", "sim_day", name="uq_demand_node_drug_day"),
        CheckConstraint("quantity_demanded >= 0", name="ck_demand_demanded_nonneg"),
        CheckConstraint("quantity_fulfilled >= 0", name="ck_demand_fulfilled_nonneg"),
        # Fulfilling more than was demanded is not a business case, it is a bug.
        CheckConstraint(
            "quantity_fulfilled <= quantity_demanded", name="ck_demand_fulfilled_lte_demanded"
        ),
        Index("ix_demand_node_drug_day", "node_id", "drug_id", "sim_day"),
    )

    @property
    def is_stockout(self) -> bool:
        return self.quantity_fulfilled < self.quantity_demanded

    def __repr__(self) -> str:
        return (
            f"<DemandRecord {self.node_id}/{self.drug_id} day={self.sim_day} "
            f"{self.quantity_fulfilled}/{self.quantity_demanded}>"
        )


class AgentDecision(Base):
    """Audit row written on every agent decision.

    This table alone satisfies NFR-08's agent-decision logging requirement.
    ``justification`` is human-readable on purpose — an examiner asking "why did
    the agent reorder here?" should get an answer without reading any code.
    """

    __tablename__ = "agent_decisions"

    decision_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    agent_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    sim_day: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    action: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    justification: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_agent_decision_agent_day", "agent_name", "sim_day"),)

    def __repr__(self) -> str:
        return f"<AgentDecision {self.decision_id} {self.agent_name} day={self.sim_day}>"


# ProvenanceRecord is defined in Stage 4 alongside the append-only trigger that
# enforces its immutability. Declaring the table here without that trigger would
# create a window in which the "immutable" audit log is silently mutable.
