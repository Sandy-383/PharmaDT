"""FastAPI backend — Layer 6 of the HLD.

Routes are exactly those the guide specifies, plus the token endpoint OAuth2
needs. The dashboard is served from ``/`` so ``docker compose up`` gives a
working system rather than an API that something else has to be pointed at.

Read endpoints are open; anything that **starts a run or injects a crisis
requires a token**. That split is deliberate: an auditor should be able to
verify the chain without credentials — the whole point of a public verification
surface — while nobody unauthenticated should be able to perturb the world
under someone else's demo.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm

from pharmadt.api.security import authenticate, create_token, require_user
from pharmadt.api.session import session

DASHBOARD = Path(__file__).parent / "static" / "dashboard.html"

app = FastAPI(
    title="PharmaDT",
    version="2.0",
    description=(
        "Digital twin, cryptographic provenance ledger, and five autonomous "
        "agents for a pharmaceutical supply chain."
    ),
)


# ── Auth ──────────────────────────────────────────────────────────────


@app.post("/auth/token", tags=["auth"])
async def issue_token(form: OAuth2PasswordRequestForm = Depends()) -> dict[str, str]:
    """OAuth2 password flow (NFR-04). Demo credentials: operator / pharmadt."""
    user = authenticate(form.username, form.password)
    if user is None:
        # One message for both cases: distinguishing them tells an attacker
        # which usernames exist.
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    return {
        "access_token": create_token(user["username"], user["role"]),
        "token_type": "bearer",
        "role": user["role"],
    }


# ── Simulation ────────────────────────────────────────────────────────


@app.post("/simulation/run", tags=["simulation"])
async def run_simulation_endpoint(
    days: int = Query(365, ge=1, le=3650),
    seed: int = Query(42),
    with_agents: bool = Query(True),
    scenario: str | None = Query(None),
    user: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    """Start a run. Returns immediately; poll /simulation/state for progress."""
    if not session.start(days, seed, with_agents, scenario):
        raise HTTPException(
            status_code=409,
            detail="A simulation is already running. Wait for it to finish.",
        )
    return {"started": True, **session.state.as_dict()}


@app.get("/simulation/state", tags=["simulation"])
async def simulation_state() -> dict[str, Any]:
    """Run status, current day, and per-node stock health for the map."""
    world = session.world
    return {
        **session.state.as_dict(),
        "current_day": int(world.env.now) if world is not None else 0,
        "event_count": len(world.events) if world is not None else 0,
        "nodes": session.node_health(),
    }


@app.get("/kpi", tags=["simulation"])
async def kpis() -> dict[str, Any]:
    """Headline KPIs for the current run."""
    if session.world is None:
        raise HTTPException(status_code=404, detail="No simulation has been run yet.")
    return session.kpis()


# ── Agents ────────────────────────────────────────────────────────────


@app.get("/agents/decisions", tags=["agents"])
async def agent_decisions(
    limit: int = Query(100, ge=1, le=1000),
    agent: str | None = Query(None),
) -> dict[str, Any]:
    """Recent agent decisions with their justifications (NFR-08)."""
    rows = session.decisions(limit, agent)
    return {"count": len(rows), "decisions": rows}


# ── Ledger ────────────────────────────────────────────────────────────


@app.get("/ledger/verify", tags=["ledger"])
async def verify_chain() -> dict[str, Any]:
    """Walk every hash and signature. This drives the dashboard's badge."""
    from pharmadt.ledger.chain import HashChainLedger

    ledger = HashChainLedger()
    result = ledger.verify_chain_detailed()
    return {
        "valid": result.valid,
        "records_checked": result.records_checked,
        "broken_at_seq": result.broken_at_seq,
        "reason": result.reason,
        "height": ledger.height(),
        "tip": ledger.tip(),
    }


@app.get("/ledger/provenance/{batch_id}", tags=["ledger"])
async def provenance(batch_id: str) -> dict[str, Any]:
    """A batch's full custody trail, manufacturer to patient."""
    from pharmadt.ledger.chain import HashChainLedger

    trace = HashChainLedger().get_provenance(batch_id)
    if not trace:
        raise HTTPException(
            status_code=404, detail=f"No provenance recorded for {batch_id}."
        )
    return {"batch_id": batch_id, "records": len(trace), "trace": trace}


@app.post("/ledger/tamper-demo", tags=["ledger"])
async def tamper_demo(
    seq: int | None = Query(None, description="record to edit; default is mid-chain"),
    user: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    """Edit a record, show the chain catching it, and restore it.

    Behind a token because it writes to the ledger, even though it puts back
    exactly what it took. Read-only verification stays open to everyone.
    """
    from pharmadt.ledger.demo import tamper_and_restore

    try:
        return await asyncio.to_thread(tamper_and_restore, seq)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


# ── Crisis ────────────────────────────────────────────────────────────


@app.get("/crisis/scenarios", tags=["crisis"])
async def list_scenarios() -> dict[str, Any]:
    """Available what-if scenarios, for the dashboard's picker."""
    from pharmadt.crisis.scenarios import load_all

    return {
        "scenarios": [
            {
                "name": s.name,
                "description": s.description,
                "trigger_day": s.trigger_day,
                "duration_days": s.duration_days,
            }
            for s in load_all()
        ]
    }


@app.post("/crisis/inject", tags=["crisis"])
async def inject_crisis(
    scenario: str = Query(..., description="scenario name, e.g. pandemic_surge"),
    days: int = Query(365, ge=1, le=3650),
    seed: int = Query(42),
    user: dict[str, Any] = Depends(require_user),
) -> dict[str, Any]:
    """Run the simulation with a disruption injected (FR-09)."""
    from pharmadt.crisis.scenarios import load_scenario

    try:
        loaded = load_scenario(scenario)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"No scenario named {scenario!r}."
        ) from None

    if not session.start(days, seed, True, scenario):
        raise HTTPException(status_code=409, detail="A simulation is already running.")
    return {
        "started": True,
        "scenario": loaded.name,
        "trigger_day": loaded.trigger_day,
        "duration_days": loaded.duration_days,
    }


# ── System readiness ──────────────────────────────────────────────────


@app.get("/system/doctor", tags=["ops"])
async def system_doctor() -> dict[str, Any]:
    """The readiness checks, with the fix beside each failure.

    The same checks ``make doctor`` runs. Surfaced here because the commonest
    demo failure is environmental — a stopped container, an unseeded database —
    and the browser is where someone notices the symptom.
    """
    from pharmadt.doctor import run_checks

    checks = await asyncio.to_thread(run_checks)
    rows = [
        {
            "name": c.name,
            "passed": c.passed,
            "detail": c.detail,
            "fix": c.fix,
            "optional": c.optional,
        }
        for c in checks
    ]
    failed = [r for r in rows if not r["passed"] and not r["optional"]]
    return {
        "ready": not failed,
        "passed": sum(r["passed"] for r in rows),
        "total": len(rows),
        "checks": rows,
    }


# ── Experiment results ────────────────────────────────────────────────


@app.get("/results", tags=["results"])
async def all_results() -> dict[str, Any]:
    """Every experiment's measured results, in one normalised shape."""
    from pharmadt.api.results import load_all

    return {"results": [a.as_dict() for a in await asyncio.to_thread(load_all)]}


@app.get("/results/{key}", tags=["results"])
async def one_result(key: str) -> dict[str, Any]:
    from pharmadt.api.results import load

    artifact = await asyncio.to_thread(load, key)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"No results named {key!r}.")
    return artifact.as_dict()


# ── Live events ───────────────────────────────────────────────────────


@app.websocket("/ws/events")
async def event_stream(websocket: WebSocket) -> None:
    """Stream simulation events as they are emitted.

    Sends only what is new since the last frame rather than the whole log — a
    365-day run emits ~14,000 events, and re-sending the accumulated history
    every tick would grow quadratically and stall the browser.
    """
    await websocket.accept()
    try:
        while True:
            events = session.new_events()
            if events:
                await websocket.send_json(
                    {
                        "type": "events",
                        "day": int(session.world.env.now) if session.world else 0,
                        "events": events,
                    }
                )
            else:
                await websocket.send_json(
                    {"type": "status", **session.state.as_dict()}
                )
            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        return


# ── Dashboard ─────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    if not DASHBOARD.exists():
        return HTMLResponse(
            "<h1>PharmaDT API</h1><p>The dashboard is not built. "
            'API documentation is at <a href="/docs">/docs</a>.</p>'
        )
    return HTMLResponse(DASHBOARD.read_text(encoding="utf-8"))


@app.get("/health", tags=["ops"], include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok"}
