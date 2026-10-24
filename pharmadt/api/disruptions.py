"""The disrupted fortnight: the same network, with things going wrong.

Five kinds of trouble, each tied to a part of the project that answers it:

=====================  ===================  =====================================
Issue                  Mirrors              Remedy (and the agent behind it)
=====================  ===================  =====================================
supply_shortage        factory_shutdown     transfer from a nearby pharmacy
                                            (Inventory/Expiry Agent: lateral stock)
cold_chain_excursion   coldchain_failure    quarantine, ship a replacement batch
                                            (Anomaly Agent: excursion detection)
route_blocked          route_disruption     reroute around the closed lanes
                                            (Route Agent: re-planning)
demand_surge           pandemic_surge       ship what is available, balance later
                                            (Demand Agent: surge detection)
counterfeit            anomaly + ledger     quarantine, replace with verified stock
                                            (ledger fingerprint check)
=====================  ===================  =====================================

The book tells a story in time. A factory shutdown about nine days ago and a
heatwave about five days ago were dealt with: those orders show the problem,
the fix and a late delivery in their history. Flooding, a demand surge, a
cold-chain breach and a counterfeit batch are happening *now*: those orders
carry an open issue and wait for the admin to apply the fix, live.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Any

from pharmadt.api import orders as book
from pharmadt.api.demo_orders import (
    CUSTOMERS,
    Step,
    _advance,
    _batch_for,
    _order,
    _smooth_steps,
    smooth,
)

FLOODED_WAREHOUSE = "NODE-WH-02"
DETOUR_WAREHOUSE = "NODE-WH-01"
FAKE_BATCH = "BATCH-7731"   # never anchored on the ledger, so verification fails

_CUSTOMER = {c[0]: c for c in CUSTOMERS}


def _name(node_id: str) -> str:
    return book._node(node_id)["name"]


def _other_batch(drug_id: str, current: str | None, at: datetime,
                 rng: random.Random) -> str | None:
    for _ in range(10):
        batch = _batch_for(drug_id, at, rng)
        if batch != current:
            return batch
    return _batch_for(drug_id, at, rng)


def _peer_with_stock(order: dict[str, Any]) -> str:
    """A pharmacy that can lend stock: a lateral peer if the destination has
    one, otherwise the nearest other delivery point able to hold the drug."""
    from pharmadt.twin.network import lateral_peers

    graph = book._network()
    drug = book._drug(order["drug_id"])
    dest = order["deliver_to"]

    def fits(node_id: str) -> bool:
        node = book._node(node_id)
        return node["cold_storage"] or not drug["cold_chain"]

    peers = [p for p in lateral_peers(graph, dest) if fits(p)]
    if peers:
        return peers[0]
    points = [n for n in book.catalog()["nodes"]
              if n["delivery_point"] and n["node_id"] != dest and fits(n["node_id"])]
    here = book._node(dest)
    return min(points, key=lambda n: (n["lat"] - here["lat"]) ** 2
               + (n["lon"] - here["lon"]) ** 2)["node_id"]


# ── The playbook: what each issue says, and what fixing it does ───────


def make_issue(kind: str, order: dict[str, Any], at: datetime,
               rng: random.Random) -> dict[str, Any]:
    drug = book._drug(order["drug_id"])
    supplier = _name(order["supplier"])
    dest = _name(order["deliver_to"])

    if kind == "route_blocked":
        detour = [DETOUR_WAREHOUSE if n == FLOODED_WAREHOUSE else n for n in order["route"]]
        return {
            "kind": kind, "severity": "high", "agent": "Route Agent",
            "scenario": "route_disruption",
            "title": "Route closed by flooding",
            "detail": (f"Flooding has closed the lanes out of {_name(FLOODED_WAREHOUSE)}. "
                       "This order's route runs through it and cannot move."),
            "customer_text": ("Your delivery is delayed: flooding has closed a road on its "
                              "route. We are rerouting it."),
            "fix": {
                "label": f"Reroute via {_name(DETOUR_WAREHOUSE)}",
                "route": detour, "delay_days": 1,
                "note": (f"Rerouted via {_name(DETOUR_WAREHOUSE)} around the flooded lanes. "
                         "One extra day in transit."),
            },
        }

    if kind == "cold_chain_excursion":
        temp, hours = rng.uniform(10.5, 14.5), rng.randint(2, 5)
        old = order["batch_id"]
        new = _other_batch(order["drug_id"], old, at, rng)
        return {
            "kind": kind, "severity": "high", "agent": "Anomaly Agent",
            "scenario": "coldchain_failure",
            "title": "Cold-chain breach in transit",
            "detail": (f"The temperature logger recorded {temp:.1f} °C for {hours} hours "
                       f"(safe range 2–8 °C). Batch {old} cannot be used."),
            "customer_text": ("This shipment got too warm in transit, so we will not deliver "
                              "it. A replacement is on its way."),
            "fix": {
                "label": "Quarantine and ship a replacement batch",
                "new_batch": new, "delay_days": 1,
                "note": (f"Batch {old} quarantined. Replacement batch {new} dispatched "
                         f"from {supplier} in a refrigerated vehicle."),
            },
        }

    if kind == "supply_shortage":
        peer = _peer_with_stock(order)
        return {
            "kind": kind, "severity": "medium", "agent": "Inventory Agent",
            "scenario": "factory_shutdown",
            "title": "Back-ordered: manufacturer output halted",
            "detail": (f"{_name('NODE-MFG-01')} has stopped production and {supplier} "
                       f"has no {drug['name']} left to pick for this order."),
            "customer_text": ("The manufacturer has paused production, so this item is "
                              "back-ordered. We are sourcing it from a nearby pharmacy."),
            "fix": {
                "label": f"Transfer from {_name(peer)}",
                "supplier": peer, "route": [peer, order["deliver_to"]],
                "new_batch": _batch_for(order["drug_id"], at, rng), "delay_days": 1,
                "advance_to": "dispatched",
                "advance_note": f"Dispatched from {_name(peer)} as a lateral transfer.",
                "note": (f"Surplus located at {_name(peer)}; transfer approved instead "
                         "of waiting for the plant to restart."),
            },
        }

    if kind == "demand_surge":
        share = rng.choice((0.5, 0.6, 0.7))
        now_qty = int(order["quantity"] * share) // 10 * 10
        rest = order["quantity"] - now_qty
        return {
            "kind": kind, "severity": "medium", "agent": "Demand Agent",
            "scenario": "pandemic_surge",
            "title": "Demand surge: stock rationed",
            "detail": (f"Orders for {drug['name']} are running at about "
                       f"{rng.choice((4, 5, 6))}x normal. {supplier} can release "
                       f"{now_qty:,} of {order['quantity']:,} units today."),
            "customer_text": (f"Demand for this medicine is very high. We can send {now_qty:,} "
                              f"units now and the remaining {rest:,} within two days."),
            "fix": {
                "label": f"Ship {now_qty:,} now, {rest:,} within 2 days",
                "new_batch": _batch_for(order["drug_id"], at, rng), "delay_days": 2,
                "advance_to": "dispatched",
                "advance_note": (f"Partial shipment of {now_qty:,} units dispatched "
                                 f"from {supplier}."),
                "note": (f"Allocation agreed: {now_qty:,} units now, balance of {rest:,} "
                         "scheduled for the next replenishment."),
            },
        }

    if kind == "counterfeit":
        new = _batch_for(order["drug_id"], at, rng)
        return {
            "kind": kind, "severity": "high", "agent": "Anomaly Agent + ledger",
            "scenario": None,
            "title": "Batch failed the authenticity check",
            "detail": (f"On arrival at {dest}, batch {order['batch_id']} matched no "
                       "fingerprint on the provenance ledger. The stock is held and must "
                       "not be dispensed."),
            "customer_text": ("The stock delivered for this order failed our authenticity "
                              "check, so it has been withheld. Verified stock is being sent."),
            "fix": {
                "label": "Quarantine and replace with a verified batch",
                "new_batch": new, "delay_days": 2,
                "note": (f"Batch {order['batch_id']} quarantined and reported to the "
                         f"regulator. Verified batch {new} dispatched from {supplier}."),
            },
        }

    raise ValueError(f"Unknown issue kind {kind!r}")


def raise_issue(order: dict[str, Any], kind: str, at: datetime, rng: random.Random) -> None:
    if kind == "counterfeit":
        order["batch_id"] = FAKE_BATCH
    issue = make_issue(kind, order, at, rng)
    issue["raised_at"] = at.isoformat()
    order["issue"] = issue
    order["history"].append(
        {"status": "issue", "at": at.isoformat(), "note": f"{issue['title']}. {issue['detail']}"}
    )


# ── Scripted incidents ────────────────────────────────────────────────

#: (hours ago placed, customer, drug, destination, quantity, priority,
#:  issue kind, stage when it struck, outcome, hours from issue to fix/cancel)
INCIDENTS: list[tuple] = [
    # Factory shutdown, about nine days ago. Resolved by lateral transfer.
    (216, "hospital", "DRUG-003", "NODE-PH-03", 1500, "standard",
     "supply_shortage", "confirmed", "resolved", 10),
    (204, "customer", "DRUG-001", "NODE-PH-01", 800, "standard",
     "supply_shortage", "confirmed", "resolved", 14),
    (192, "port-health", "DRUG-003", "NODE-PH-04", 1200, "standard",
     "supply_shortage", "confirmed", "cancelled", 20),
    # Heatwave, five to six days ago. Replacement batches sent.
    (146, "shivamogga-rural", "DRUG-004", "NODE-PH-06", 300, "standard",
     "cold_chain_excursion", "dispatched", "resolved", 6),
    (122, "hospital", "DRUG-002", "NODE-PH-03", 180, "urgent",
     "cold_chain_excursion", "dispatched", "resolved", 4),
    # A counterfeit caught on arrival, about five days ago.
    (134, "whitefield-clinic", "DRUG-001", "NODE-PH-02", 600, "standard",
     "counterfeit", "dispatched", "resolved", 8),
    # Happening now: flooding cuts the Hubballi lanes.
    (40, "belagavi-district", "DRUG-001", "NODE-PH-05", 500, "standard",
     "route_blocked", "dispatched", "open", None),
    (26, "belagavi-district", "DRUG-003", "NODE-PH-05", 900, "urgent",
     "route_blocked", "confirmed", "open", None),
    (20, "belagavi-district", "DRUG-005", "NODE-PH-05", 40, "standard",
     "route_blocked", "confirmed", "open", None),
    # Happening now: a cold-chain breach on City Care's insulin.
    (30, "customer", "DRUG-002", "NODE-PH-01", 220, "urgent",
     "cold_chain_excursion", "dispatched", "open", None),
    # Happening now: demand surge for antibiotics and vaccine.
    (28, "customer", "DRUG-001", "NODE-PH-01", 1500, "urgent",
     "demand_surge", "confirmed", "open", None),
    (22, "hospital", "DRUG-004", "NODE-PH-03", 350, "urgent",
     "demand_surge", "confirmed", "open", None),
    (18, "port-health", "DRUG-001", "NODE-PH-04", 1000, "urgent",
     "demand_surge", "confirmed", "resolved", 3),
    # Happening now: counterfeit morphine at Shivamogga.
    (52, "shivamogga-rural", "DRUG-005", "NODE-PH-06", 60, "standard",
     "counterfeit", "dispatched", "open", None),
]

#: Ordinary urgent orders from the surge, flowing normally.
SURGE_ORDERS = [
    (14, "hospital", "DRUG-001", "NODE-PH-03", 900),
    (11, "whitefield-clinic", "DRUG-001", "NODE-PH-02", 700),
    (9, "customer", "DRUG-004", "NODE-PH-01", 250),
    (6, "shivamogga-rural", "DRUG-001", "NODE-PH-06", 600),
    (3, "port-health", "DRUG-004", "NODE-PH-06", 200),
]


def _incident(row: tuple, now: datetime, rng: random.Random) -> dict[str, Any]:
    hours_ago, who, drug_id, dest, qty, priority, kind, stage, outcome, gap = row
    placed = (now - timedelta(hours=hours_ago)).replace(microsecond=0)
    order = _order(0, _CUSTOMER[who], drug_id, dest, qty, priority, placed)
    steps = _smooth_steps(order, rng)
    upto = {"confirmed": 1, "dispatched": 2}[stage]

    # Walk the order to the stage where trouble struck.
    stage_at = placed + timedelta(hours=steps[upto - 1].after_hours)
    _advance(order, steps[:upto], stage_at, rng)
    if kind in ("cold_chain_excursion",):
        strike = stage_at + timedelta(hours=rng.uniform(5, 12))       # mid-transit
    elif kind == "counterfeit":
        strike = stage_at + timedelta(hours=24 * order["transit_days"] - 3)  # on arrival
    else:
        strike = stage_at + timedelta(hours=rng.uniform(1, 4))
    strike = min(strike, now - timedelta(minutes=30)).replace(microsecond=0)
    raise_issue(order, kind, strike, rng)

    if outcome == "open":
        return order

    acted = min(strike + timedelta(hours=gap), now - timedelta(minutes=10)).replace(microsecond=0)
    if outcome == "cancelled":
        order["resolved_issues"].append(
            {**order["issue"], "resolved_at": acted.isoformat(), "resolution": "Order cancelled"}
        )
        order["issue"] = None
        order["status"] = "cancelled"
        order["history"].append({"status": "cancelled", "at": acted.isoformat(),
                                 "note": "Cancelled by the customer after the delay; "
                                         "sourced elsewhere."})
        return order

    book.apply_fix(order, acted)

    # After the fix, the order carries on: dispatch if it is not moving yet,
    # then deliver once the (now longer) transit has elapsed.
    remaining: list[Step] = []
    base = (acted - placed).total_seconds() / 3600
    if order["status"] == "confirmed":
        remaining.append(Step("dispatched", base + rng.uniform(1, 3),
                              f"Dispatched from {_name(order['supplier'])}."))
        base = remaining[-1].after_hours
    expected = datetime.fromisoformat(order["expected_by"]).replace(tzinfo=UTC)
    deliver_h = (expected - placed).total_seconds() / 3600 + rng.uniform(4, 9)
    remaining.append(Step("delivered", max(deliver_h, base + 6),
                          f"Delivered to {_name(order['deliver_to'])}. Signed for at the counter."))
    if order["status"] == "dispatched":
        remaining = remaining[-1:]
    _advance_keep_batch(order, remaining, now, rng)
    return order


def _advance_keep_batch(order: dict[str, Any], steps: list[Step], now: datetime,
                        rng: random.Random) -> None:
    """Like ``_advance``, but a dispatch keeps the batch the fix assigned."""
    placed = datetime.fromisoformat(order["created_at"])
    for step in steps:
        at = (placed + timedelta(hours=step.after_hours)).replace(microsecond=0)
        if at > now:
            break
        note = step.note
        if step.status == "dispatched" and order.get("batch_id"):
            note += f" Batch {order['batch_id']}."
        order["status"] = step.status
        order["history"].append({"status": step.status, "at": at.isoformat(), "note": note})


def disrupted(now: datetime | None = None, seed: int = 11) -> list[dict[str, Any]]:
    now = (now or datetime.now(UTC)).replace(microsecond=0)
    rng = random.Random(seed)

    # Background: the ordinary orders of the fortnight, all delivered on time.
    background = [o for o in smooth(now, seed=seed, days=14, count=30)
                  if datetime.fromisoformat(o["created_at"]) < now - timedelta(days=3)]

    incidents = [_incident(row, now, rng) for row in INCIDENTS]

    surge = []
    for hours_ago, who, drug_id, dest, qty in SURGE_ORDERS:
        placed = now - timedelta(hours=hours_ago)
        order = _order(0, _CUSTOMER[who], drug_id, dest, qty, "urgent", placed)
        _advance(order, _smooth_steps(order, rng), now, rng)
        surge.append(order)

    orders = sorted(background + incidents + surge, key=lambda o: o["created_at"])
    for seq, order in enumerate(orders, start=1):
        order["order_id"] = f"ORD-{seq:05d}"
    return orders


#: Incidents for the mixed book: a calm fortnight with a handful of trouble.
#: Two were handled earlier in the fortnight; four are open now.
MIXED_INCIDENTS = [
    row for row in INCIDENTS
    if (row[1], row[6], row[8]) in {
        ("customer", "supply_shortage", "resolved"),
        ("hospital", "cold_chain_excursion", "resolved"),
        ("belagavi-district", "route_blocked", "open"),
        ("customer", "cold_chain_excursion", "open"),
        ("hospital", "demand_surge", "open"),
        ("shivamogga-rural", "counterfeit", "open"),
    }
]


def mixed(now: datetime | None = None, seed: int = 7) -> list[dict[str, Any]]:
    """The smooth fortnight plus a few problems: mostly on time, some to fix live."""
    now = (now or datetime.now(UTC)).replace(microsecond=0)
    rng = random.Random(seed + 100)
    incidents = []
    seen_route = False
    for row in MIXED_INCIDENTS:
        if row[6] == "route_blocked":     # one flooded order is enough to make the point
            if seen_route:
                continue
            seen_route = True
        incidents.append(_incident(row, now, rng))
    orders = sorted(smooth(now, seed=seed) + incidents, key=lambda o: o["created_at"])
    for seq, order in enumerate(orders, start=1):
        order["order_id"] = f"ORD-{seq:05d}"
    return orders
