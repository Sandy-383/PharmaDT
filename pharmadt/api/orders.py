"""Customer orders: placing, tracking, and fulfilling them.

The customer portal's whole backend. A customer orders a drug for delivery to
one of the network's pharmacies; an admin confirms, dispatches and delivers it.
Each step is timestamped so the customer sees a tracking timeline.

Orders are fulfilled through the same network the twin simulates. The supplying
distributor, the full supply route and the transit time all come from the
network graph, and a dispatched order is assigned a real seeded batch, so a
customer can check its custody trail on the provenance ledger.

**Storage is a JSON file**, not a database table. Orders are demo data layered
on top of the research system, and a table would need its own migration against
the schema the ledger's immutability guarantees are written for. The file
survives the auto-reload ``make api`` runs with, which an in-memory dict would
not. It is written atomically (temp file, then rename) so a crash mid-write
cannot leave half an order book behind.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

ORDERS_FILE = Path("data/orders.json")

#: Lifecycle, in order. ``cancelled`` can branch off any step before delivery.
STATUSES = ("placed", "confirmed", "dispatched", "delivered")
TERMINAL = frozenset({"delivered", "cancelled"})

#: The admin's "advance" action moves an order exactly one step.
NEXT_STATUS = {"placed": "confirmed", "confirmed": "dispatched", "dispatched": "delivered"}

MAX_QUANTITY = 5_000            # a pharmacy's whole storage capacity
MAX_CONTROLLED_QUANTITY = 500   # controlled substances are capped tighter
PRIORITIES = ("standard", "urgent")

_lock = threading.Lock()


class OrderError(ValueError):
    """An order the rules refuse. The message is shown to the customer."""


# ── Catalog: what can be ordered, and where it can go ─────────────────


@lru_cache(maxsize=1)
def _network() -> Any:
    """The seeded network graph. Built from the fixture, so no database needed."""
    from pharmadt.core.seed import NODES
    from pharmadt.twin.network import build_network

    return build_network(NODES)


def catalog() -> dict[str, Any]:
    """Drugs and delivery points, with the facts the order form needs."""
    from pharmadt.core.seed import DRUGS, NODES

    drugs = [
        {
            "drug_id": d["drug_id"],
            "name": d["name"],
            "cold_chain": d["requires_cold_chain"],
            "controlled": d["is_controlled"],
            "max_quantity": MAX_CONTROLLED_QUANTITY if d["is_controlled"] else MAX_QUANTITY,
        }
        for d in DRUGS
    ]
    nodes = [
        {
            "node_id": n.node_id,
            "name": n.name,
            "node_type": str(n.node_type),
            "lat": n.lat,
            "lon": n.lon,
            "cold_storage": n.has_cold_storage,
            "delivery_point": str(n.node_type) in ("PHARMACY", "HOSPITAL"),
        }
        for n in NODES
    ]
    return {"drugs": drugs, "nodes": nodes, "priorities": list(PRIORITIES)}


def _drug(drug_id: str) -> dict[str, Any]:
    for drug in catalog()["drugs"]:
        if drug["drug_id"] == drug_id:
            return drug
    raise OrderError(f"Unknown drug {drug_id!r}.")


def _node(node_id: str) -> dict[str, Any]:
    for node in catalog()["nodes"]:
        if node["node_id"] == node_id:
            return node
    raise OrderError(f"Unknown delivery point {node_id!r}.")


def plan_route(destination: str, priority: str = "standard") -> dict[str, Any]:
    """Who supplies the destination, the route stock takes, and how long it takes.

    The supplier is the destination's upstream distributor. The lead time is
    that last leg's transit, plus a day of picking for standard orders; urgent
    orders are picked the same day.
    """
    import networkx as nx

    from pharmadt.twin.network import upstream_supplier

    if priority not in PRIORITIES:
        raise OrderError(f"Priority must be one of {', '.join(PRIORITIES)}.")
    if not _node(destination)["delivery_point"]:
        raise OrderError(f"{destination} is not a delivery point.")

    graph = _network()
    supplier = upstream_supplier(graph, destination)
    if supplier is None:
        raise OrderError(f"{destination} has no supplying distributor.")

    manufacturer = next(
        n for n, data in graph.nodes(data=True) if str(data["node_type"]) == "MANUFACTURER"
    )
    upstream = nx.shortest_path(graph, manufacturer, supplier, weight="transit_days")
    transit = int(graph.edges[supplier, destination]["transit_days"])
    distance = float(graph.edges[supplier, destination]["distance_km"])
    handling = 0 if priority == "urgent" else 1

    return {
        "supplier": supplier,
        "route": [*upstream, destination],
        "transit_days": transit,
        "lead_days": transit + handling,
        "distance_km": round(distance, 1),
    }


def _assign_batch(drug_id: str) -> str | None:
    """The seeded batch of this drug with the most shelf life left.

    Seeded batches are the ones the ledger anchored at creation, so the
    customer can trace the one they received.
    """
    import random

    from pharmadt.config import settings
    from pharmadt.core.seed import _build_batches

    candidates = [
        b for b in _build_batches(random.Random(settings.sim_seed)) if b.drug_id == drug_id
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda b: b.expiry_date).batch_id


# ── Storage ───────────────────────────────────────────────────────────


def _load() -> list[dict[str, Any]]:
    if not ORDERS_FILE.exists():
        return []
    try:
        return json.loads(ORDERS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt file must not take the portal down; start a fresh book
        # and keep the damaged one for inspection.
        ORDERS_FILE.replace(ORDERS_FILE.with_suffix(".corrupt.json"))
        return []


def _save(orders: list[dict[str, Any]]) -> None:
    ORDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ORDERS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(orders, indent=2), encoding="utf-8")
    os.replace(tmp, ORDERS_FILE)


def _now() -> datetime:
    return datetime.now(UTC)


def _decorate(order: dict[str, Any]) -> dict[str, Any]:
    """Add the names and derived fields the portals display."""
    out = dict(order)
    out["drug_name"] = _drug(order["drug_id"])["name"]
    out["deliver_to_name"] = _node(order["deliver_to"])["name"]
    out["supplier_name"] = _node(order["supplier"])["name"]
    out["next_status"] = NEXT_STATUS.get(order["status"])
    out["can_cancel"] = order["status"] in ("placed", "confirmed")
    out.setdefault("issue", None)
    out.setdefault("resolved_issues", [])
    out.setdefault("promised_by", order["expected_by"])
    # How late the order is, or will be, against the date first promised.
    if order["status"] == "delivered":
        delivered = next(h["at"] for h in reversed(order["history"]) if h["status"] == "delivered")
        finish = datetime.fromisoformat(delivered).date()
    elif order["status"] == "cancelled":
        finish = None
    else:
        finish = datetime.fromisoformat(order["expected_by"]).date()
    promised = datetime.fromisoformat(out["promised_by"]).date()
    out["late_days"] = max(0, (finish - promised).days) if finish else 0
    return out


# ── Disruptions: an issue on an order, and the fix that clears it ─────


def _distance(a: str, b: str) -> float:
    graph = _network()
    if graph.has_edge(a, b):
        return round(float(graph.edges[a, b]["distance_km"]), 1)
    from pharmadt.twin.network import haversine_km

    na, nb = _node(a), _node(b)
    return round(haversine_km(na["lat"], na["lon"], nb["lat"], nb["lon"]), 1)


def apply_fix(order: dict[str, Any], at: datetime) -> None:
    """Carry out the remedy recorded on an order's open issue.

    The remedy is decided when the issue is raised and stored with it, so what
    the admin is offered is exactly what happens. Applying it can reroute the
    order, switch its supplier, replace its batch, push its expected date and
    move it on a step; every change is written to the order's history.
    """
    issue = order.get("issue")
    if not issue:
        raise OrderError(f"{order['order_id']} has no open issue.")
    fix = issue["fix"]

    if fix.get("route"):
        order["route"] = list(fix["route"])
    if fix.get("supplier"):
        order["supplier"] = fix["supplier"]
        order["distance_km"] = _distance(fix["supplier"], order["deliver_to"])
    if fix.get("new_batch"):
        order["batch_id"] = fix["new_batch"]
    if fix.get("delay_days"):
        expected = datetime.fromisoformat(order["expected_by"]).date()
        floor = (at + timedelta(days=1)).date()
        order["expected_by"] = max(expected + timedelta(days=fix["delay_days"]), floor).isoformat()

    order["history"].append(
        {"status": "resolved", "at": at.isoformat(), "note": fix["note"]}
    )
    if fix.get("advance_to") and NEXT_STATUS.get(order["status"]) == fix["advance_to"]:
        order["status"] = fix["advance_to"]
        order["history"].append(
            {"status": fix["advance_to"], "at": at.isoformat(),
             "note": fix.get("advance_note", f"Moved to {fix['advance_to']}.")}
        )

    order.setdefault("resolved_issues", []).append(
        {**issue, "resolved_at": at.isoformat(), "resolution": fix["label"]}
    )
    order["issue"] = None


def resolve_issue(user: dict[str, Any], order_id: str) -> dict[str, Any]:
    """Admin action: apply the recorded fix for an order's open issue."""
    from pharmadt.api.security import ADMIN_ROLES

    if user.get("role") not in ADMIN_ROLES:
        raise OrderError("Only the network administrator can resolve an issue.")
    with _lock:
        orders = _load()
        order = next((o for o in orders if o["order_id"] == order_id), None)
        if order is None:
            raise LookupError(order_id)
        apply_fix(order, _now())
        _save(orders)
    return _decorate(order)


# ── Operations ────────────────────────────────────────────────────────


def place_order(
    customer: dict[str, Any],
    drug_id: str,
    quantity: int,
    deliver_to: str,
    priority: str = "standard",
    notes: str = "",
) -> dict[str, Any]:
    drug = _drug(drug_id)
    node = _node(deliver_to)

    if priority not in PRIORITIES:
        raise OrderError(f"Priority must be one of {', '.join(PRIORITIES)}.")
    if not node["delivery_point"]:
        raise OrderError(f"{node['name']} is not a delivery point.")
    if not 1 <= quantity <= drug["max_quantity"]:
        raise OrderError(
            f"Quantity for {drug['name']} must be between 1 and {drug['max_quantity']:,}."
        )
    # The same rule the twin enforces on shipments: cold-chain stock cannot be
    # left somewhere it will warm up.
    if drug["cold_chain"] and not node["cold_storage"]:
        raise OrderError(
            f"{drug['name']} needs cold storage, and {node['name']} has none. "
            "Choose a delivery point with cold storage."
        )

    from pharmadt.api.security import profile

    plan = plan_route(deliver_to, priority)
    created = _now()
    note = "Order received."
    if drug["controlled"]:
        note += " Controlled substance: held for prescription check before confirmation."

    with _lock:
        orders = _load()
        order = {
            "order_id": f"ORD-{len(orders) + 1:05d}",
            "customer": customer["sub"],
            "customer_name": profile(customer["sub"])["name"],
            "drug_id": drug_id,
            "quantity": int(quantity),
            "deliver_to": deliver_to,
            "priority": priority,
            "notes": notes.strip()[:500],
            "status": "placed",
            "created_at": created.isoformat(),
            "expected_by": (created + timedelta(days=plan["lead_days"])).date().isoformat(),
            "batch_id": None,
            **plan,
            "history": [{"status": "placed", "at": created.isoformat(), "note": note}],
            "promised_by": (created + timedelta(days=plan["lead_days"])).date().isoformat(),
            "issue": None,
            "resolved_issues": [],
        }
        orders.append(order)
        _save(orders)
    return _decorate(order)


def list_orders(user: dict[str, Any], status: str | None = None) -> list[dict[str, Any]]:
    """Newest first. Customers see only their own; admins see everything."""
    from pharmadt.api.security import ADMIN_ROLES

    with _lock:
        orders = _load()
    if user.get("role") not in ADMIN_ROLES:
        orders = [o for o in orders if o["customer"] == user["sub"]]
    if status:
        orders = [o for o in orders if o["status"] == status]
    return [_decorate(o) for o in reversed(orders)]


def get_order(user: dict[str, Any], order_id: str) -> dict[str, Any] | None:
    for order in list_orders(user):
        if order["order_id"] == order_id:
            return order
    return None


def update_status(
    user: dict[str, Any], order_id: str, new_status: str, note: str = ""
) -> dict[str, Any]:
    """Move an order along. Admins advance or cancel; customers may only cancel
    their own, and only before it is dispatched."""
    from pharmadt.api.security import ADMIN_ROLES

    is_admin = user.get("role") in ADMIN_ROLES

    with _lock:
        orders = _load()
        order = next((o for o in orders if o["order_id"] == order_id), None)
        if order is None or (not is_admin and order["customer"] != user["sub"]):
            # Same answer for "not yours" and "does not exist".
            raise LookupError(order_id)

        current = order["status"]
        if current in TERMINAL:
            raise OrderError(f"{order_id} is already {current}.")

        if new_status == "cancelled":
            if not is_admin and current not in ("placed", "confirmed"):
                raise OrderError("An order cannot be cancelled once it has been dispatched.")
        elif order.get("issue"):
            # Moving a disrupted order on without dealing with the disruption
            # would hide it; the fix is what moves it now.
            raise OrderError(f"{order_id} has an open issue: {order['issue']['title']}. "
                             "Apply the fix first.")
        elif not is_admin:
            raise OrderError("Only the network administrator can update an order.")
        elif NEXT_STATUS.get(current) != new_status:
            raise OrderError(f"{order_id} is {current}; it can only move to "
                             f"{NEXT_STATUS.get(current)} next.")

        now = _now()
        default_notes = {
            "confirmed": "Confirmed and queued for picking.",
            "dispatched": f"Dispatched from {_node(order['supplier'])['name']}.",
            "delivered": f"Delivered to {_node(order['deliver_to'])['name']}.",
            "cancelled": "Cancelled by " + ("the administrator." if is_admin else "the customer."),
        }
        if new_status == "dispatched":
            order["batch_id"] = _assign_batch(order["drug_id"])
            if order["batch_id"]:
                default_notes["dispatched"] += f" Batch {order['batch_id']}."
            order["expected_by"] = (
                now + timedelta(days=order["transit_days"])
            ).date().isoformat()

        if new_status == "cancelled" and order.get("issue"):
            order.setdefault("resolved_issues", []).append(
                {**order["issue"], "resolved_at": now.isoformat(), "resolution": "Order cancelled"}
            )
            order["issue"] = None

        order["status"] = new_status
        order["history"].append(
            {"status": new_status, "at": now.isoformat(),
             "note": note.strip()[:300] or default_notes[new_status]}
        )
        _save(orders)
    return _decorate(order)


def summary() -> dict[str, int]:
    """Counts for the admin overview."""
    with _lock:
        orders = _load()
    counts = {s: 0 for s in (*STATUSES, "cancelled")}
    for order in orders:
        counts[order["status"]] = counts.get(order["status"], 0) + 1
    counts["open"] = sum(counts[s] for s in ("placed", "confirmed", "dispatched"))
    counts["urgent_open"] = sum(
        1 for o in orders if o["priority"] == "urgent" and o["status"] not in TERMINAL
    )
    counts["with_issues"] = sum(1 for o in orders if o.get("issue"))
    counts["total"] = len(orders)
    return counts
