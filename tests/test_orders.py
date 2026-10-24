"""Customer orders and the two-portal role split."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pharmadt.api import orders as order_book
from pharmadt.api.main import app


@pytest.fixture(autouse=True)
def isolated_book(tmp_path, monkeypatch):
    """Every test gets an empty order book; the real one is never touched."""
    monkeypatch.setattr(order_book, "ORDERS_FILE", tmp_path / "orders.json")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def login(client: TestClient, username: str) -> dict[str, str]:
    r = client.post("/auth/token", data={"username": username, "password": "pharmadt"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def place(client, headers, **kw):
    body = {"drug_id": "DRUG-001", "quantity": 100, "deliver_to": "NODE-PH-01"}
    body.update(kw)
    return client.post("/orders", json=body, headers=headers)


# ── Roles ─────────────────────────────────────────────────────────────


def test_tokens_carry_the_portal_role(client) -> None:
    assert client.post("/auth/token", data={"username": "admin", "password": "pharmadt"}
                       ).json()["role"] == "admin"
    body = client.post("/auth/token", data={"username": "customer", "password": "pharmadt"}).json()
    assert body["role"] == "customer"
    assert body["default_node"] == "NODE-PH-01"


@pytest.mark.parametrize(
    "path", ["/simulation/run", "/crisis/inject?scenario=pandemic_surge", "/ledger/tamper-demo"]
)
def test_a_customer_cannot_drive_the_system(client, path) -> None:
    """Authenticated, but not allowed: 403 rather than 401."""
    assert client.post(path, headers=login(client, "customer")).status_code == 403


def test_an_admin_cannot_place_orders(client) -> None:
    assert place(client, login(client, "admin")).status_code == 403


def test_me_restores_the_profile(client) -> None:
    me = client.get("/auth/me", headers=login(client, "hospital")).json()
    assert me == {"username": "hospital", "role": "customer",
                  "name": "Mysuru General Hospital", "default_node": "NODE-PH-03"}


# ── Placing ───────────────────────────────────────────────────────────


def test_an_order_is_planned_through_the_network(client) -> None:
    r = place(client, login(client, "customer"))
    assert r.status_code == 201
    order = r.json()
    assert order["status"] == "placed"
    assert order["supplier"] == "NODE-DC-01"
    assert order["route"][0] == "NODE-MFG-01" and order["route"][-1] == "NODE-PH-01"
    assert order["deliver_to_name"] == "Jayanagar Pharmacy"


def test_urgent_orders_skip_the_picking_day(client) -> None:
    h = login(client, "customer")
    standard = place(client, h).json()
    urgent = place(client, h, priority="urgent").json()
    assert urgent["lead_days"] == standard["lead_days"] - 1


def test_cold_chain_stock_cannot_go_where_it_will_warm_up(client) -> None:
    """Insulin to Whitefield, which has no cold storage."""
    r = place(client, login(client, "customer"), drug_id="DRUG-002", deliver_to="NODE-PH-02")
    assert r.status_code == 422
    assert "cold storage" in r.json()["detail"]


def test_controlled_quantities_are_capped(client) -> None:
    r = place(client, login(client, "customer"), drug_id="DRUG-005", quantity=501)
    assert r.status_code == 422


def test_a_warehouse_is_not_a_delivery_point(client) -> None:
    r = place(client, login(client, "customer"), deliver_to="NODE-WH-01")
    assert r.status_code == 422


# ── Visibility ────────────────────────────────────────────────────────


def test_customers_only_see_their_own_orders(client) -> None:
    mine = place(client, login(client, "customer")).json()["order_id"]
    theirs = place(client, login(client, "hospital"), deliver_to="NODE-PH-03").json()["order_id"]

    listed = client.get("/orders", headers=login(client, "customer")).json()["orders"]
    seen = {o["order_id"] for o in listed}
    assert mine in seen and theirs not in seen
    # Same 404 as a missing order, so ids cannot be probed.
    assert client.get(f"/orders/{theirs}", headers=login(client, "customer")).status_code == 404


def test_an_admin_sees_every_order(client) -> None:
    place(client, login(client, "customer"))
    place(client, login(client, "hospital"), deliver_to="NODE-PH-03")
    assert client.get("/orders", headers=login(client, "admin")).json()["count"] == 2


def test_orders_need_a_token(client) -> None:
    assert client.get("/orders").status_code == 401


# ── Lifecycle ─────────────────────────────────────────────────────────


def test_an_admin_walks_an_order_to_delivery(client) -> None:
    oid = place(client, login(client, "customer")).json()["order_id"]
    admin = login(client, "admin")
    for step in ("confirmed", "dispatched", "delivered"):
        r = client.post(f"/orders/{oid}/status", json={"status": step}, headers=admin)
        assert r.status_code == 200, r.json()
    order = r.json()
    assert order["status"] == "delivered"
    assert [h["status"] for h in order["history"]] == [
        "placed", "confirmed", "dispatched", "delivered"
    ]
    assert order["batch_id"], "dispatch assigns a traceable batch"


def test_steps_cannot_be_skipped(client) -> None:
    oid = place(client, login(client, "customer")).json()["order_id"]
    r = client.post(
        f"/orders/{oid}/status", json={"status": "delivered"}, headers=login(client, "admin")
    )
    assert r.status_code == 409


def test_a_customer_can_cancel_before_dispatch_only(client) -> None:
    cust, admin = login(client, "customer"), login(client, "admin")
    early = place(client, cust).json()["order_id"]
    assert client.post(f"/orders/{early}/status", json={"status": "cancelled"},
                       headers=cust).json()["status"] == "cancelled"

    late = place(client, cust).json()["order_id"]
    for step in ("confirmed", "dispatched"):
        client.post(f"/orders/{late}/status", json={"status": step}, headers=admin)
    assert client.post(f"/orders/{late}/status", json={"status": "cancelled"},
                       headers=cust).status_code == 409


def test_a_customer_cannot_advance_an_order(client) -> None:
    cust = login(client, "customer")
    oid = place(client, cust).json()["order_id"]
    assert client.post(f"/orders/{oid}/status", json={"status": "confirmed"},
                       headers=cust).status_code == 409


def test_the_book_survives_a_restart(tmp_path) -> None:
    """Persisted to disk, so `make api`'s auto-reload does not wipe it."""
    user = {"sub": "customer", "role": "customer", "name": "City Care Clinic"}
    placed = order_book.place_order(user, "DRUG-003", 10, "NODE-PH-01")
    order_book._network.cache_clear()
    assert order_book.get_order(user, placed["order_id"]) is not None


def test_the_catalog_lists_delivery_points_with_cold_storage(client) -> None:
    cat = client.get("/catalog").json()
    points = {n["node_id"]: n for n in cat["nodes"] if n["delivery_point"]}
    assert len(points) == 6
    assert points["NODE-PH-01"]["cold_storage"] is True
    assert points["NODE-PH-02"]["cold_storage"] is False


# ── Demo scenarios and disruptions ────────────────────────────────────


def test_the_smooth_fortnight_is_entirely_on_time() -> None:
    from pharmadt.api import demo_orders

    book_ = [order_book._decorate(o) for o in demo_orders.smooth()]
    assert len(book_) >= 30
    assert not any(o["issue"] for o in book_)
    assert not any(o["late_days"] for o in book_)
    assert {"delivered", "dispatched", "confirmed"} <= {o["status"] for o in book_}
    # Every dispatched order carries a seeded batch the ledger can trace.
    assert all(o["batch_id"] for o in book_ if o["status"] in ("dispatched", "delivered"))


def test_cold_chain_orders_only_go_to_cold_storage() -> None:
    from pharmadt.api import demo_orders
    from pharmadt.api.disruptions import disrupted

    for order in demo_orders.smooth() + disrupted():
        if order_book._drug(order["drug_id"])["cold_chain"]:
            assert order_book._node(order["deliver_to"])["cold_storage"], order["order_id"]


def test_the_disrupted_fortnight_has_every_kind_of_trouble() -> None:
    from pharmadt.api.disruptions import disrupted

    book_ = disrupted()
    open_kinds = {o["issue"]["kind"] for o in book_ if o.get("issue")}
    handled = {r["kind"] for o in book_ for r in o.get("resolved_issues", [])}
    assert open_kinds >= {"route_blocked", "cold_chain_excursion", "demand_surge", "counterfeit"}
    assert handled >= {"supply_shortage", "cold_chain_excursion", "counterfeit"}


def test_an_open_issue_blocks_the_normal_next_step(client) -> None:
    client.post("/orders/demo?scenario=disrupted", headers=login(client, "admin"))
    admin = login(client, "admin")
    stuck = next(o for o in client.get("/orders", headers=admin).json()["orders"]
                 if o["issue"] and o["next_status"])
    r = client.post(f"/orders/{stuck['order_id']}/status",
                    json={"status": stuck["next_status"]}, headers=admin)
    assert r.status_code == 409
    assert "open issue" in r.json()["detail"]


def test_every_open_issue_can_be_fixed_and_the_order_delivered(client) -> None:
    admin = login(client, "admin")
    assert client.post("/orders/demo?scenario=disrupted", headers=admin).json()["with_issues"] >= 5
    for order in client.get("/orders?", headers=admin).json()["orders"]:
        if not order["issue"]:
            continue
        fixed = client.post(f"/orders/{order['order_id']}/resolve", headers=admin).json()
        assert fixed["issue"] is None
        assert fixed["resolved_issues"][-1]["resolution"] == order["issue"]["fix"]["label"]
        assert fixed["expected_by"] >= order["expected_by"], "a fix never makes it earlier"
        while fixed["next_status"]:
            fixed = client.post(f"/orders/{order['order_id']}/status",
                                json={"status": fixed["next_status"]}, headers=admin).json()
        assert fixed["status"] == "delivered"


def test_a_counterfeit_carries_a_batch_the_ledger_never_saw() -> None:
    import random

    from pharmadt.api.disruptions import FAKE_BATCH, disrupted
    from pharmadt.config import settings
    from pharmadt.core.seed import _build_batches

    genuine = {b.batch_id for b in _build_batches(random.Random(settings.sim_seed))}
    fake = [o for o in disrupted() if o.get("issue") and o["issue"]["kind"] == "counterfeit"]
    assert fake and all(o["batch_id"] == FAKE_BATCH for o in fake)
    assert FAKE_BATCH not in genuine


def test_customers_cannot_apply_fixes(client) -> None:
    client.post("/orders/demo?scenario=disrupted", headers=login(client, "admin"))
    cust = login(client, "customer")
    mine = next(o for o in client.get("/orders", headers=cust).json()["orders"] if o["issue"])
    assert client.post(f"/orders/{mine['order_id']}/resolve", headers=cust).status_code == 403
    # ...but they do see the plain-language explanation.
    assert mine["issue"]["customer_text"]


def test_customers_cannot_load_demo_data(client) -> None:
    assert client.post("/orders/demo?scenario=smooth",
                       headers=login(client, "customer")).status_code == 403


def test_the_mixed_fortnight_is_mostly_calm_with_a_few_open_issues() -> None:
    from pharmadt.api.disruptions import mixed

    book_ = [order_book._decorate(o) for o in mixed()]
    open_issues = [o for o in book_ if o["issue"]]
    assert 3 <= len(open_issues) <= 5
    assert {o["issue"]["kind"] for o in open_issues} == {
        "route_blocked", "cold_chain_excursion", "demand_surge", "counterfeit"
    }
    assert sum(1 for o in book_ if o["resolved_issues"]) == 2
    assert len(book_) - len(open_issues) >= 30
    ids = [o["order_id"] for o in book_]
    assert len(ids) == len(set(ids))
    # Both portal accounts see at least one open problem of their own.
    assert {"customer", "hospital"} <= {o["customer"] for o in open_issues}
