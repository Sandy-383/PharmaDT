"""Demonstration order books: smooth, mixed (smooth with a few problems), disrupted.

Each scenario writes a complete, internally consistent order book to
``data/orders.json``, replacing whatever was there. Every order is planned
through the real network (supplier, route and lead time come from
:func:`orders.plan_route`), carries a real seeded batch once dispatched, and has
a timestamped history that a customer can follow in the tracking view.

The data is synthetic. Customer names are invented clinics; no real patient or
prescription data is involved.

Usage::

    python -m pharmadt.api.demo_orders smooth
    python -m pharmadt.api.demo_orders mixed
    python -m pharmadt.api.demo_orders disrupted
    python -m pharmadt.api.demo_orders clear

The admin portal's Orders page has the same three buttons.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from pharmadt.api import orders as book

SCENARIOS = ("smooth", "mixed", "disrupted")

#: Customers placing orders. The first two have portal accounts (see
#: ``security.DEMO_USERS``), so their orders show up when you sign in as them.
CUSTOMERS: list[tuple[str, str, str]] = [
    ("customer", "City Care Clinic", "NODE-PH-01"),
    ("hospital", "Mysuru General Hospital", "NODE-PH-03"),
    ("whitefield-clinic", "Whitefield Family Clinic", "NODE-PH-02"),
    ("port-health", "Mangaluru Port Health Centre", "NODE-PH-04"),
    ("belagavi-district", "Belagavi District Hospital", "NODE-PH-05"),
    ("shivamogga-rural", "Shivamogga Rural Clinic", "NODE-PH-06"),
]

#: Typical order sizes per drug, in units: (low, high, step).
ORDER_SIZES = {
    "DRUG-001": (200, 1200, 50),   # amoxicillin: high-volume antibiotic
    "DRUG-002": (40, 300, 10),     # insulin: cold chain, smaller lots
    "DRUG-003": (300, 2000, 100),  # metformin: the volume leader
    "DRUG-004": (50, 400, 25),     # flu vaccine: cold chain, seasonal
    "DRUG-005": (10, 120, 10),     # morphine: controlled, small and checked
}

NOTES = [
    "", "", "", "",
    "Deliver to the pharmacy counter before 11:00.",
    "Monthly restock.",
    "Weekly restock for the outpatient ward.",
    "Please call on arrival.",
]


@dataclass(frozen=True)
class Step:
    """One lifecycle step, offset in hours from when the order was placed."""

    status: str
    after_hours: float
    note: str


# ── Building blocks ───────────────────────────────────────────────────


def _batch_for(drug_id: str, deliver_on: datetime, rng: random.Random) -> str | None:
    """A seeded batch of this drug still in date on delivery, chosen at random.

    Random rather than always the longest-dated one, so the book shows several
    batches and each has its own custody trail on the ledger.
    """
    from pharmadt.config import settings
    from pharmadt.core.seed import _build_batches

    usable = [
        b.batch_id
        for b in _build_batches(random.Random(settings.sim_seed))
        if b.drug_id == drug_id and b.expiry_date > deliver_on.date() + timedelta(days=21)
    ]
    return rng.choice(sorted(usable)) if usable else None


def _deliverable(drug: dict[str, Any], node_id: str) -> bool:
    node = book._node(node_id)
    return node["delivery_point"] and (node["cold_storage"] or not drug["cold_chain"])


def _pick_destination(drug: dict[str, Any], home: str, rng: random.Random) -> str:
    """Usually the customer's own pharmacy; occasionally another with cold storage."""
    if _deliverable(drug, home) and rng.random() < 0.85:
        return home
    options = [n["node_id"] for n in book.catalog()["nodes"] if _deliverable(drug, n["node_id"])]
    return rng.choice(options)


def _quantity(drug_id: str, rng: random.Random) -> int:
    low, high, step = ORDER_SIZES[drug_id]
    return rng.randrange(low, high + 1, step)


def _order(
    seq: int,
    customer: tuple[str, str, str],
    drug_id: str,
    deliver_to: str,
    quantity: int,
    priority: str,
    placed_at: datetime,
    notes: str = "",
) -> dict[str, Any]:
    """A freshly placed order, exactly as ``place_order`` would have written it."""
    username, name, _ = customer
    drug = book._drug(drug_id)
    plan = book.plan_route(deliver_to, priority)
    note = "Order received."
    if drug["controlled"]:
        note += " Controlled substance: held for prescription check before confirmation."
    return {
        "order_id": f"ORD-{seq:05d}",
        "customer": username,
        "customer_name": name,
        "drug_id": drug_id,
        "quantity": quantity,
        "deliver_to": deliver_to,
        "priority": priority,
        "notes": notes,
        "status": "placed",
        "created_at": placed_at.isoformat(),
        "expected_by": (placed_at + timedelta(days=plan["lead_days"])).date().isoformat(),
        "promised_by": (placed_at + timedelta(days=plan["lead_days"])).date().isoformat(),
        "batch_id": None,
        **plan,
        "history": [{"status": "placed", "at": placed_at.isoformat(), "note": note}],
        "issue": None,
        "resolved_issues": [],
    }


def _advance(order: dict[str, Any], steps: list[Step], now: datetime,
             rng: random.Random) -> None:
    """Apply every step whose time has already come. Later steps stay pending,
    so recent orders are genuinely still in progress."""
    placed = datetime.fromisoformat(order["created_at"])
    for step in steps:
        at = (placed + timedelta(hours=step.after_hours)).replace(microsecond=0)
        if at > now:
            break
        if step.status == "dispatched":
            order["batch_id"] = _batch_for(order["drug_id"], at, rng)
            batch = order["batch_id"]
            note = f"{step.note} Batch {batch}." if batch else step.note
            order["expected_by"] = (at + timedelta(days=order["transit_days"])).date().isoformat()
        else:
            note = step.note
        order["status"] = step.status
        order["history"].append({"status": step.status, "at": at.isoformat(), "note": note})


def _smooth_steps(order: dict[str, Any], rng: random.Random) -> list[Step]:
    """A normal order: confirmed within hours, dispatched next morning,
    delivered on the transit time the network promises."""
    urgent = order["priority"] == "urgent"
    controlled = book._drug(order["drug_id"])["controlled"]
    supplier = book._node(order["supplier"])["name"]
    destination = book._node(order["deliver_to"])["name"]

    confirm = rng.uniform(0.3, 1.5) if urgent else rng.uniform(1.5, 6)
    if controlled:
        confirm += rng.uniform(2, 5)   # the prescription check takes a little longer
    dispatch = confirm + (rng.uniform(1, 3) if urgent else rng.uniform(12, 20))
    deliver = dispatch + 24 * order["transit_days"] - rng.uniform(2, 8)

    confirm_note = (
        "Prescription verified. Confirmed and queued for picking."
        if controlled else "Confirmed and queued for picking."
    )
    if urgent:
        confirm_note += " Marked urgent: picked the same day."
    return [
        Step("confirmed", confirm, confirm_note),
        Step("dispatched", dispatch, f"Dispatched from {supplier}."),
        Step("delivered", deliver, f"Delivered to {destination}. Signed for at the counter."),
    ]


# ── Scenarios ─────────────────────────────────────────────────────────


def smooth(now: datetime | None = None, seed: int = 7, days: int = 14,
           count: int = 36) -> list[dict[str, Any]]:
    """A normal fortnight: steady demand, every order on time.

    Older orders are delivered; the last day or two are still moving through
    the pipeline, so the tracking views show every stage at once.
    """
    now = now or datetime.now(UTC)
    rng = random.Random(seed)
    start = now - timedelta(days=days)
    drug_weights = {"DRUG-001": 3, "DRUG-002": 2, "DRUG-003": 4, "DRUG-004": 1, "DRUG-005": 1}
    drug_ids = list(drug_weights)

    # Most orders spread across the fortnight; about a third in the last two
    # days, so the live views show orders at every stage, not just delivered.
    recent = count // 3
    placed_times = [start + timedelta(hours=rng.uniform(0, (days - 2) * 24))
                    for _ in range(count - recent)]
    placed_times += [now - timedelta(hours=rng.uniform(1, 46)) for _ in range(recent)]
    # Working hours, roughly 08:30-18:30 IST, for the older orders.
    placed_times = [
        t if t > now - timedelta(days=2)
        else t.replace(hour=rng.randint(3, 12), minute=rng.randint(0, 59))
        for t in placed_times
    ]
    placed_times = [t.replace(microsecond=0) for t in placed_times
                    if t < now - timedelta(minutes=20)]

    orders = []
    for seq, placed_at in enumerate(sorted(placed_times), start=1):
        customer = rng.choices(CUSTOMERS, weights=[5, 5, 2, 2, 2, 2])[0]
        drug_id = rng.choices(drug_ids, weights=list(drug_weights.values()))[0]
        drug = book._drug(drug_id)
        deliver_to = _pick_destination(drug, customer[2], rng)
        priority = "urgent" if rng.random() < 0.15 else "standard"
        order = _order(seq, customer, drug_id, deliver_to, _quantity(drug_id, rng),
                       priority, placed_at, rng.choice(NOTES))
        _advance(order, _smooth_steps(order, rng), now, rng)
        orders.append(order)
    return orders


def build(scenario: str, now: datetime | None = None) -> list[dict[str, Any]]:
    if scenario == "smooth":
        return smooth(now)
    if scenario == "disrupted":
        from pharmadt.api.disruptions import disrupted

        return disrupted(now)
    if scenario == "mixed":
        from pharmadt.api.disruptions import mixed

        return mixed(now)
    raise ValueError(f"Unknown scenario {scenario!r}; choose from {', '.join(SCENARIOS)}.")


def load(scenario: str) -> dict[str, Any]:
    """Replace the order book with a scenario. Returns a short summary."""
    if scenario == "clear":
        orders: list[dict[str, Any]] = []
    else:
        orders = build(scenario)
    with book._lock:
        book._save(orders)
    counts: dict[str, int] = {}
    for order in orders:
        counts[order["status"]] = counts.get(order["status"], 0) + 1
    return {
        "scenario": scenario,
        "orders": len(orders),
        "by_status": counts,
        "with_issues": sum(1 for o in orders if o.get("issue")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a demonstration order book.")
    parser.add_argument("scenario", choices=(*SCENARIOS, "clear"))
    args = parser.parse_args()
    result = load(args.scenario)
    print(f"Loaded '{result['scenario']}': {result['orders']} orders "
          f"{result['by_status']}, {result['with_issues']} with open issues.")
    print(f"Written to {book.ORDERS_FILE}")


if __name__ == "__main__":
    main()
