"""DTO validation — the boundary the Stage 14 API will be written against.

These duplicate some database CHECK constraints on purpose. The database is the
last line of defence; the DTO is the one that returns a 422 to a caller instead
of a 500.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from pharmadt.core.models import NodeType, ShipmentStatus
from pharmadt.core.schemas import (
    BatchCreate,
    BatchRead,
    ChainVerificationResult,
    DemandRecordRead,
    DrugCreate,
    NodeCreate,
    ShipmentCreate,
    ShipmentRead,
)


def test_drug_create_accepts_a_plain_generic() -> None:
    drug = DrugCreate(drug_id="D1", name="Metformin", shelf_life_days=1095)
    assert drug.requires_cold_chain is False


def test_cold_chain_drug_requires_a_temperature_band() -> None:
    with pytest.raises(ValidationError, match="temp_min_c and temp_max_c"):
        DrugCreate(
            drug_id="D2", name="Vaccine", shelf_life_days=365, requires_cold_chain=True
        )


def test_temperature_band_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="below"):
        DrugCreate(
            drug_id="D3",
            name="Vaccine",
            shelf_life_days=365,
            requires_cold_chain=True,
            temp_min_c=8.0,
            temp_max_c=2.0,
        )


def test_shelf_life_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        DrugCreate(drug_id="D4", name="Expired On Arrival", shelf_life_days=0)


def test_node_coordinates_are_bounded() -> None:
    with pytest.raises(ValidationError):
        NodeCreate(
            node_id="N1",
            name="Nowhere",
            node_type=NodeType.PHARMACY,
            lat=95.0,
            lon=0.0,
            storage_capacity=10,
        )


def test_node_type_accepts_its_string_form() -> None:
    node = NodeCreate(
        node_id="N2",
        name="Depot",
        node_type="WAREHOUSE",
        lat=12.9,
        lon=77.6,
        storage_capacity=1000,
    )
    assert node.node_type is NodeType.WAREHOUSE


def test_batch_expiry_must_follow_manufacture() -> None:
    with pytest.raises(ValidationError, match="expiry_date"):
        BatchCreate(
            batch_id="B1",
            drug_id="D1",
            manufacturer_id="N1",
            mfg_date=date(2026, 6, 1),
            expiry_date=date(2026, 1, 1),
            quantity=10,
        )


def test_batch_read_rejects_a_malformed_fingerprint() -> None:
    with pytest.raises(ValidationError):
        BatchRead(
            batch_id="B1",
            drug_id="D1",
            manufacturer_id="N1",
            mfg_date=date(2026, 1, 1),
            expiry_date=date(2027, 1, 1),
            quantity=10,
            batch_fingerprint="tooshort",
        )


def test_shipment_endpoints_must_differ() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        ShipmentCreate(
            shipment_id="S1",
            from_node="N1",
            to_node="N1",
            batch_id="B1",
            quantity=10,
            dispatch_day=1,
            eta_day=2,
        )


def test_shipment_cannot_arrive_before_dispatch() -> None:
    with pytest.raises(ValidationError, match="precede"):
        ShipmentCreate(
            shipment_id="S2",
            from_node="N1",
            to_node="N2",
            batch_id="B1",
            quantity=10,
            dispatch_day=9,
            eta_day=2,
        )


def test_shipment_read_round_trips_status_and_temp_log() -> None:
    shipment = ShipmentRead(
        shipment_id="S3",
        from_node="N1",
        to_node="N2",
        batch_id="B1",
        quantity=10,
        dispatch_day=1,
        eta_day=4,
        status="IN_TRANSIT",
        temp_log=[{"sim_day": 2, "temp_c": 6.1}],
    )
    assert shipment.status is ShipmentStatus.IN_TRANSIT
    assert shipment.temp_log[0]["temp_c"] == 6.1


def test_demand_record_exposes_stockout() -> None:
    record = DemandRecordRead(
        id=1,
        node_id="N1",
        drug_id="D1",
        sim_day=4,
        quantity_demanded=100,
        quantity_fulfilled=100,
    )
    assert record.is_stockout is False


def test_chain_verification_result_reports_the_broken_seq() -> None:
    """The Stage 4 tamper demo turns on naming the exact record."""
    result = ChainVerificationResult(
        valid=False,
        records_checked=1042,
        broken_at_seq=573,
        reason="record_hash mismatch after payload edit",
    )
    assert result.valid is False
    assert result.broken_at_seq == 573
