"""Pydantic DTOs — the boundary between the ORM and everything outside it.

The Stage 14 API serves these, never ORM rows. Keeping the two separate means a
column rename does not silently change the shape of a public JSON payload, and
lets the API expose derived fields (``is_stockout``) that have no column.

``NodeType`` and ``ShipmentStatus`` are imported from :mod:`pharmadt.core.models`
rather than redeclared. One definition of the domain vocabulary is worth the
import; two definitions would drift.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pharmadt.core.events import EventType
from pharmadt.core.models import NodeType, ShipmentStatus

_ORM = ConfigDict(from_attributes=True)


# ── Drug ──────────────────────────────────────────────────────────────


class DrugBase(BaseModel):
    name: str
    atc_code: str | None = None
    shelf_life_days: int = Field(gt=0)
    requires_cold_chain: bool = False
    temp_min_c: float | None = None
    temp_max_c: float | None = None
    is_controlled: bool = False

    @model_validator(mode="after")
    def _check_temp_band(self) -> DrugBase:
        if self.requires_cold_chain and (self.temp_min_c is None or self.temp_max_c is None):
            raise ValueError("a cold-chain drug must define temp_min_c and temp_max_c")
        if (
            self.temp_min_c is not None
            and self.temp_max_c is not None
            and self.temp_min_c >= self.temp_max_c
        ):
            raise ValueError("temp_min_c must be below temp_max_c")
        return self


class DrugCreate(DrugBase):
    drug_id: str


class DrugRead(DrugBase):
    model_config = _ORM
    drug_id: str


# ── Node ──────────────────────────────────────────────────────────────


class NodeBase(BaseModel):
    name: str
    node_type: NodeType
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    storage_capacity: int = Field(ge=0)
    has_cold_storage: bool = False


class NodeCreate(NodeBase):
    node_id: str


class NodeRead(NodeBase):
    model_config = _ORM
    node_id: str
    # Public key only — the private half never leaves data/keys/.
    public_key: str | None = None


# ── Batch ─────────────────────────────────────────────────────────────


class BatchBase(BaseModel):
    drug_id: str
    manufacturer_id: str
    mfg_date: date
    expiry_date: date
    quantity: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_dates(self) -> BatchBase:
        if self.expiry_date <= self.mfg_date:
            raise ValueError("expiry_date must be after mfg_date")
        return self


class BatchCreate(BatchBase):
    batch_id: str


class BatchRead(BatchBase):
    model_config = _ORM
    batch_id: str
    batch_fingerprint: str = Field(min_length=64, max_length=64)


# ── Inventory ─────────────────────────────────────────────────────────


class InventoryRecordRead(BaseModel):
    model_config = _ORM
    id: int
    node_id: str
    batch_id: str
    quantity_on_hand: int = Field(ge=0)
    sim_day: int


# ── Shipment ──────────────────────────────────────────────────────────


class ShipmentBase(BaseModel):
    from_node: str
    to_node: str
    batch_id: str
    quantity: int = Field(gt=0)
    dispatch_day: int
    eta_day: int

    @model_validator(mode="after")
    def _check_route(self) -> ShipmentBase:
        if self.from_node == self.to_node:
            raise ValueError("from_node and to_node must differ")
        if self.eta_day < self.dispatch_day:
            raise ValueError("eta_day cannot precede dispatch_day")
        return self


class ShipmentCreate(ShipmentBase):
    shipment_id: str


class ShipmentRead(ShipmentBase):
    model_config = _ORM
    shipment_id: str
    status: ShipmentStatus
    temp_log: list[dict[str, Any]] = Field(default_factory=list)


# ── Demand ────────────────────────────────────────────────────────────


class DemandRecordRead(BaseModel):
    model_config = _ORM
    id: int
    node_id: str
    drug_id: str
    sim_day: int
    quantity_demanded: int = Field(ge=0)
    quantity_fulfilled: int = Field(ge=0)

    @property
    def is_stockout(self) -> bool:
        return self.quantity_fulfilled < self.quantity_demanded


# ── Agent decisions ───────────────────────────────────────────────────


class AgentDecisionRead(BaseModel):
    model_config = _ORM
    decision_id: int
    agent_name: str
    sim_day: int
    inputs: dict[str, Any]
    action: dict[str, Any]
    justification: str
    created_at: datetime


# ── Provenance (served by the Stage 4 ledger) ─────────────────────────


class ProvenanceEntry(BaseModel):
    """One link of a batch's custody chain, as returned by ``get_provenance``."""

    seq: int
    batch_id: str
    event_type: EventType
    from_node: str | None
    to_node: str | None
    signer_node: str
    payload: dict[str, Any]
    prev_hash: str = Field(min_length=64, max_length=64)
    record_hash: str = Field(min_length=64, max_length=64)
    signature: str
    merkle_root: str | None = None


class ChainVerificationResult(BaseModel):
    """Outcome of ``verify_chain`` — the Stage 4 tamper demo renders this."""

    valid: bool
    records_checked: int
    broken_at_seq: int | None = None
    reason: str | None = None
