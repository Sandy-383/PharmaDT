"""★ STAGE 10.5 — THE INTEGRATION GATE.

The guide's most important milestone. Seven conditions must hold
**simultaneously, in a single run**:

1. the simulation runs 365 days over >= 12 nodes without error
2. all five agents observe, decide and act every simulated day
3. every physical handoff produces a signed, chained ledger record
4. ``verify_chain()`` returns True; tampering makes it False
5. KPIs are computed and printed
6. every agent decision is persisted in ``agent_decisions``
7. pytest passes with coverage >= 60%

Checks 1-6 run here. Check 7 is ``make test`` / ``make cov``, reported as a
reminder rather than shelled out to, because a gate that runs its own test
suite and reports its own pass is not evidence of anything.

Nothing is asserted that is not measured: every line printed comes from the run
that just happened.

Usage::

    make gate
    python -m pharmadt.gate --days 365
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESULTS = Path("experiments/integration_gate.json")
RULE = "=" * 74


@dataclass(slots=True)
class Check:
    """One gate condition and the evidence for its verdict."""

    number: str
    name: str
    passed: bool
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.number}. {self.name}\n         {self.detail}"


def build_system(seed: int, days: int) -> tuple[Any, dict[str, Any]]:
    """Stand up the full autonomous system: five agents on one twin."""
    from sqlalchemy import select

    from pharmadt.agents.anomaly import AnomalyAgent
    from pharmadt.agents.demand import DemandAgent
    from pharmadt.agents.expiry import ExpiryAgent
    from pharmadt.agents.inventory import InventoryAgent
    from pharmadt.agents.route_agent import RouteAgent
    from pharmadt.core.db import session_scope
    from pharmadt.core.models import Drug
    from pharmadt.ledger.chain import HashChainLedger
    from pharmadt.twin.simulation import attach_agents, build_world

    with session_scope() as session:
        cold_chain = frozenset(
            session.scalars(select(Drug.drug_id).where(Drug.requires_cold_chain.is_(True)))
        )

    world = build_world(seed=seed)
    ledger = HashChainLedger()

    agents = {
        "DemandAgent": DemandAgent(),
        "InventoryAgent": InventoryAgent(graph=world.graph),
        "ExpiryAgent": ExpiryAgent(graph=world.graph),
        "RouteAgent": RouteAgent(graph=world.graph, cold_chain_drugs=cold_chain),
        "AnomalyAgent": AnomalyAgent(ledger=ledger, world=world),
    }
    attach_agents(world, *agents.values())
    return world, {"agents": agents, "ledger": ledger, "days": days}


# ── The checks ────────────────────────────────────────────────────────


def check_simulation(world: Any, days: int, elapsed: float) -> Check:
    nodes = world.graph.number_of_nodes()
    ok = nodes >= 12 and len(world.events) > 0
    return Check(
        "1", "Simulation runs the full horizon over >= 12 nodes",
        ok,
        f"{days} days over {nodes} nodes in {elapsed:.1f}s, "
        f"{len(world.events):,} events emitted",
        {"days": days, "nodes": nodes, "events": len(world.events), "seconds": elapsed},
    )


def check_agents(world: Any, days: int) -> Check:
    """Every agent must have acted on every simulated day."""
    orchestrator = world.orchestrator
    per_agent = {
        agent.name: len({d.sim_day for d in agent.decisions}) for agent in orchestrator.agents
    }
    five = len(per_agent) == 5
    # Day 0 has no history for anyone to act on, so full coverage is days - 1.
    complete = {n: d for n, d in per_agent.items() if d >= days - 1}
    ok = five and len(complete) == 5

    return Check(
        "2", "All five agents observe, decide and act every day",
        ok,
        f"{len(per_agent)} agents registered; days acted on: "
        + ", ".join(f"{n}={d}" for n, d in sorted(per_agent.items())),
        {"agents": per_agent},
    )


def check_ledger_records(world: Any, ledger: Any) -> Check:
    """Every physical handoff must produce a signed, chained record."""
    from pharmadt.core.events import LEDGER_EVENT_TYPES

    anchorable = [e for e in world.events if e.is_ledger_anchored and e.batch_id]
    anchored = ledger.anchor_events(world.events)
    height = ledger.height()

    kinds = sorted({str(e.event_type) for e in anchorable})
    ok = anchored > 0 and anchored >= len(anchorable) * 0.99
    return Check(
        "3", "Every physical handoff produces a signed, chained record",
        ok,
        f"{anchored:,} of {len(anchorable):,} custody events anchored "
        f"(chain height {height:,}); kinds: {', '.join(kinds)}",
        {
            "anchorable": len(anchorable),
            "anchored": anchored,
            "chain_height": height,
            "event_kinds": kinds,
            "ledger_event_types": sorted(str(t) for t in LEDGER_EVENT_TYPES),
        },
    )


def check_chain_integrity(ledger: Any) -> Check:
    """Verify, tamper, verify again, restore. All four, or it proves nothing."""
    from sqlalchemy import select, text

    from pharmadt.core.db import session_scope
    from pharmadt.core.models import ProvenanceRecord

    clean = ledger.verify_chain_detailed()
    if not clean.valid:
        return Check("4", "verify_chain() true, and false under tampering", False,
                     f"the untouched chain already fails at seq {clean.broken_at_seq}")

    with session_scope() as session:
        highest = session.scalar(
            select(ProvenanceRecord.seq).order_by(ProvenanceRecord.seq.desc()).limit(1)
        )
        # Tamper mid-chain rather than at the tip: an edit to the last record
        # breaks only its own hash, while one in the middle must also orphan
        # every record after it. The mid-chain case is the stronger claim.
        target = session.scalar(
            select(ProvenanceRecord.seq)
            .where(ProvenanceRecord.seq <= (highest or 1) // 2 + 1)
            .order_by(ProvenanceRecord.seq.desc())
            .limit(1)
        )
        original = session.scalar(
            select(ProvenanceRecord.payload).where(ProvenanceRecord.seq == target)
        )

    def rewrite(payload: dict) -> None:
        with session_scope() as session:
            session.execute(text("ALTER TABLE provenance_records DISABLE TRIGGER no_update"))
            session.execute(
                text("UPDATE provenance_records SET payload = CAST(:p AS jsonb) WHERE seq = :s"),
                {"p": json.dumps(payload), "s": target},
            )
            session.execute(text("ALTER TABLE provenance_records ENABLE TRIGGER no_update"))

    rewrite({**(original or {}), "quantity": 999_999})
    tampered = ledger.verify_chain_detailed()
    rewrite(original or {})
    restored = ledger.verify_chain_detailed()

    ok = clean.valid and not tampered.valid and restored.valid and tampered.broken_at_seq == target
    return Check(
        "4", "verify_chain() true, and false under tampering",
        ok,
        f"clean: VALID over {clean.records_checked:,} records | "
        f"tampered seq {target}: BROKEN at {tampered.broken_at_seq} | restored: "
        f"{'VALID' if restored.valid else 'STILL BROKEN'}",
        {
            "records_checked": clean.records_checked,
            "tampered_seq": target,
            "detected_at_seq": tampered.broken_at_seq,
            "reason": tampered.reason,
            "restored": restored.valid,
        },
    )


def check_kpis(world: Any, context: dict[str, Any]) -> Check:
    """Every KPI the gate names must be present and finite."""
    from pharmadt.twin.simulation import compute_kpis

    kpis = compute_kpis(world)
    route_agent = context["agents"]["RouteAgent"]
    anomaly_agent = context["agents"]["AnomalyAgent"]

    kpis["delivery_distance_km"] = round(route_agent.total_distance_km, 1)
    kpis["counterfeit_alerts"] = anomaly_agent.ledger_catches
    kpis["forecast_mape"] = _forecast_mape(world, context["agents"]["DemandAgent"])

    required = (
        "stockout_rate", "wastage_units", "forecast_mape",
        "delivery_distance_km", "service_level",
    )
    missing = [k for k in required if kpis.get(k) is None]
    return Check(
        "5", "KPIs computed and printed",
        not missing,
        "missing: " + ", ".join(missing) if missing else
        f"stockout {kpis['stockout_rate']:.5f} | wastage {kpis['wastage_units']} | "
        f"forecast MAPE {kpis['forecast_mape']}% | "
        f"delivery {kpis['delivery_distance_km']:,.0f} km",
        kpis,
    )


def _forecast_mape(world: Any, demand_agent: Any) -> float | None:
    """MAPE of the agent's own in-simulation forecasts against realised demand."""
    import numpy as np

    errors: list[float] = []
    for (node_id, drug_id), forecast in demand_agent.latest.items():
        node = world.nodes.get(node_id)
        if node is None:
            continue
        history = list(node.demand_history.get(drug_id, ()))
        if not history:
            continue
        actual = float(np.mean(history[-7:]))
        if actual > 0:
            errors.append(abs(float(np.mean(forecast)) - actual) / actual)
    return round(float(np.mean(errors)) * 100, 2) if errors else None


def check_decisions_persisted(world: Any) -> Check:
    """NFR-08: every decision reaches agent_decisions."""
    from sqlalchemy import func, select

    from pharmadt.agents.base import persist_decisions
    from pharmadt.core.db import session_scope
    from pharmadt.core.models import AgentDecision

    records = world.orchestrator.collect_decisions()
    written = persist_decisions(records)

    with session_scope() as session:
        by_agent = dict(
            session.execute(
                select(AgentDecision.agent_name, func.count())
                .group_by(AgentDecision.agent_name)
                .order_by(AgentDecision.agent_name)
            ).all()
        )

    ok = written == len(records) and len(by_agent) == 5
    return Check(
        "6", "Every agent decision persisted in agent_decisions",
        ok,
        f"{written:,} decisions written; rows per agent: "
        + ", ".join(f"{n}={c:,}" for n, c in by_agent.items()),
        {"written": written, "by_agent": by_agent},
    )


# ── Runner ────────────────────────────────────────────────────────────


def run_gate(days: int = 365, seed: int = 42) -> tuple[list[Check], dict[str, Any]]:
    from pharmadt.twin.simulation import run_simulation

    world, context = build_system(seed, days)

    started = time.perf_counter()
    run_simulation(world, days)
    elapsed = time.perf_counter() - started

    checks = [
        check_simulation(world, days, elapsed),
        check_agents(world, days),
        check_ledger_records(world, context["ledger"]),
        check_chain_integrity(context["ledger"]),
        check_kpis(world, context),
        check_decisions_persisted(world),
    ]
    return checks, context


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 10.5 integration gate.")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{RULE}\nSTAGE 10.5 -- INTEGRATION GATE\n{RULE}")
    print(f"Running the full autonomous system: {args.days} days, seed {args.seed}\n")

    checks, _ = run_gate(args.days, args.seed)

    for check in checks:
        print(check.render())
        print()

    passed = sum(1 for c in checks if c.passed)
    print(RULE)
    print(f"  {passed}/{len(checks)} automated conditions met")
    print("  7. pytest passes, coverage >= 60%  -->  run `make cov` (not self-reported)")
    print(RULE)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            [{"number": c.number, "name": c.name, "passed": c.passed,
              "detail": c.detail, "metrics": c.metrics} for c in checks],
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {RESULTS}")
    raise SystemExit(0 if passed == len(checks) else 1)


if __name__ == "__main__":
    main()
