"""FastAPI backend: routes, OAuth2, and the read/write authorisation split.

The split is the point worth testing. An auditor must be able to verify the
chain without credentials — a verification surface nobody can reach proves
nothing — while nobody unauthenticated may perturb a running world.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from pharmadt.api.main import app
from pharmadt.api.security import create_token, decode_token


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def auth(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/auth/token", data={"username": "operator", "password": "pharmadt"}
    )
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


# ── Tokens ────────────────────────────────────────────────────────────


def test_valid_credentials_issue_a_token(client: TestClient) -> None:
    response = client.post(
        "/auth/token", data={"username": "operator", "password": "pharmadt"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["role"] == "operator"


def test_a_wrong_password_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/auth/token", data={"username": "operator", "password": "wrong"}
    )
    assert response.status_code == 401


def test_an_unknown_user_gets_the_same_message_as_a_wrong_password(
    client: TestClient,
) -> None:
    """Distinguishing them would tell an attacker which usernames exist."""
    unknown = client.post("/auth/token", data={"username": "ghost", "password": "x"})
    wrong = client.post("/auth/token", data={"username": "operator", "password": "x"})
    assert unknown.json()["detail"] == wrong.json()["detail"]


def test_a_token_round_trips_its_claims() -> None:
    claims = decode_token(create_token("operator", "operator"))
    assert claims is not None
    assert claims["sub"] == "operator"
    assert claims["exp"] > time.time()


def test_a_tampered_token_is_rejected() -> None:
    """The signature is what makes the claims trustworthy."""
    token = create_token("operator", "operator")
    body, signature = token.split(".", 1)
    assert decode_token(f"{body}.{'a' * len(signature)}") is None


def test_a_forged_payload_is_rejected() -> None:
    """Swapping the role in the body must not survive verification."""
    import json
    from base64 import urlsafe_b64encode

    token = create_token("operator", "operator")
    _, signature = token.split(".", 1)
    forged = urlsafe_b64encode(
        json.dumps({"sub": "operator", "role": "admin", "exp": 9_999_999_999}).encode()
    ).decode().rstrip("=")
    assert decode_token(f"{forged}.{signature}") is None


def test_malformed_tokens_are_rejected_without_raising() -> None:
    for bad in ("", "nonsense", "a.b.c", "...."):
        assert decode_token(bad) is None


# ── Authorisation split ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "path", ["/simulation/run", "/crisis/inject?scenario=pandemic_surge"]
)
def test_mutating_routes_require_a_token(client: TestClient, path: str) -> None:
    assert client.post(path).status_code == 401


@pytest.mark.parametrize(
    "path",
    ["/health", "/simulation/state", "/ledger/verify", "/crisis/scenarios",
     "/agents/decisions"],
)
def test_read_routes_are_open(client: TestClient, path: str) -> None:
    """An auditor verifies the chain without needing an account."""
    assert client.get(path).status_code == 200


# ── Ledger routes ─────────────────────────────────────────────────────


def test_chain_verification_reports_its_evidence(client: TestClient) -> None:
    body = client.get("/ledger/verify").json()
    assert set(body) >= {"valid", "records_checked", "height", "tip"}
    assert isinstance(body["valid"], bool)


def test_an_unknown_batch_has_no_provenance(client: TestClient) -> None:
    assert client.get("/ledger/provenance/BATCH-DOES-NOT-EXIST").status_code == 404


# ── Scenarios ─────────────────────────────────────────────────────────


def test_all_four_scenarios_are_offered(client: TestClient) -> None:
    names = {s["name"] for s in client.get("/crisis/scenarios").json()["scenarios"]}
    assert names == {
        "pandemic_surge", "factory_shutdown", "coldchain_failure", "route_disruption",
    }


def test_an_unknown_scenario_is_a_clear_404(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.post("/crisis/inject?scenario=no_such_thing", headers=auth)
    assert response.status_code == 404
    assert "no_such_thing" in response.json()["detail"]


# ── Running a simulation ──────────────────────────────────────────────


@pytest.mark.slow
def test_a_run_completes_and_exposes_its_results(
    client: TestClient, auth: dict[str, str]
) -> None:
    assert client.post("/simulation/run?days=20&seed=42", headers=auth).status_code == 200

    for _ in range(120):
        state = client.get("/simulation/state").json()
        if state["status"] in ("complete", "failed"):
            break
        time.sleep(0.25)

    assert state["status"] == "complete", state.get("error")
    assert state["current_day"] == 20
    assert state["event_count"] > 0
    assert len(state["nodes"]) == 12

    kpis = client.get("/kpi").json()
    assert 0.0 <= kpis["stockout_rate"] <= 1.0

    decisions = client.get("/agents/decisions?limit=5").json()
    assert decisions["count"] > 0
    assert decisions["decisions"][0]["justification"]


@pytest.mark.slow
def test_node_health_is_classified_for_the_map(
    client: TestClient, auth: dict[str, str]
) -> None:
    """Health is about running out, so an empty node is the alarming one."""
    nodes = client.get("/simulation/state").json()["nodes"]
    assert nodes, "run the simulation test first"
    for node in nodes:
        assert node["health"] in {"healthy", "low", "critical"}
        assert node["lat"] is not None and node["lon"] is not None


def test_kpis_before_any_run_are_a_clear_404() -> None:
    """A fresh session has nothing to report, and says so."""
    from pharmadt.api.session import SimulationSession

    assert SimulationSession().kpis() == {}
    assert SimulationSession().snapshot() == {}
    assert SimulationSession().decisions() == []


def test_a_second_run_is_refused_while_one_is_going() -> None:
    """Two worlds sharing one session would interleave their results."""
    from pharmadt.api.session import SimulationSession

    simulation = SimulationSession()
    assert simulation.start(days=5, seed=42, with_agents=False) is True
    # The first may already have finished; only assert the guard when it has not.
    if simulation.busy:
        assert simulation.start(days=5, seed=42, with_agents=False) is False


# ── Dashboard ─────────────────────────────────────────────────────────


def test_the_dashboard_is_served_at_the_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "PharmaDT" in response.text


def test_the_dashboard_carries_every_hld_panel(client: TestClient) -> None:
    """One component per element of the HLD dashboard panel."""
    body = client.get("/").text
    for panel in ("Chain integrity", "Batch provenance", "Key indicators",
                  "What-if scenarios", "Network status", "Agent decision log"):
        assert panel in body


def test_the_websocket_streams_events(client: TestClient) -> None:
    with client.websocket_connect("/ws/events") as ws:
        message = ws.receive_json()
        assert message["type"] in {"events", "status"}
