"""The API's live simulation session.

One world at a time, held in memory. A simulation is not a request-scoped
object — the dashboard polls its state, streams its events, and injects crises
into the *same* run — so it lives here rather than being rebuilt per request.

Runs happen on a worker thread. SimPy's loop is synchronous and a 365-day run
with agents takes seconds; doing that inside the request handler would block
the event loop and stall the very WebSocket the dashboard uses to watch it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class SessionState:
    """What the dashboard needs to know about the current run."""

    status: str = "idle"          # idle | running | complete | failed
    days: int = 0
    seed: int = 0
    with_agents: bool = True
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    scenarios: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "days": self.days,
            "seed": self.seed,
            "with_agents": self.with_agents,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "scenarios": list(self.scenarios),
        }


class SimulationSession:
    """Holds the current world and serialises access to it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.world: Any = None
        self.orchestrator: Any = None
        self.state = SessionState()
        #: Events already streamed, so the WebSocket can send only what is new.
        self._cursor = 0

    # ── Lifecycle ─────────────────────────────────────────────────────

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, days: int, seed: int, with_agents: bool, scenario: str | None = None) -> bool:
        """Begin a run on a worker thread. False if one is already going."""
        with self._lock:
            if self.busy:
                return False
            self.state = SessionState(
                status="running",
                days=days,
                seed=seed,
                with_agents=with_agents,
                started_at=datetime.now(UTC).isoformat(),
                scenarios=[scenario] if scenario else [],
            )
            self._cursor = 0
            self._thread = threading.Thread(
                target=self._run, args=(days, seed, with_agents, scenario), daemon=True
            )
            self._thread.start()
            return True

    def _run(self, days: int, seed: int, with_agents: bool, scenario: str | None) -> None:
        try:
            from pharmadt.twin.simulation import attach_agents, build_world, run_simulation

            world = build_world(seed=seed)
            if with_agents:
                attach_agents(world, *self._agents(world))
            if scenario:
                from pharmadt.crisis.injector import crisis_process
                from pharmadt.crisis.scenarios import load_scenario

                world.env.process(crisis_process(world, load_scenario(scenario)))

            # Published before the run so the dashboard can watch it fill.
            self.world = world
            self.orchestrator = world.orchestrator
            run_simulation(world, days)

            self.state.status = "complete"
        except Exception as exc:  # noqa: BLE001 - surfaced to the dashboard
            self.state.status = "failed"
            self.state.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.state.finished_at = datetime.now(UTC).isoformat()

    @staticmethod
    def _agents(world: Any) -> list[Any]:
        from sqlalchemy import select

        from pharmadt.agents.demand import DemandAgent
        from pharmadt.agents.expiry import ExpiryAgent
        from pharmadt.agents.inventory import InventoryAgent
        from pharmadt.core.db import session_scope
        from pharmadt.core.models import Drug

        with session_scope() as session:
            cold = frozenset(
                session.scalars(select(Drug.drug_id).where(Drug.requires_cold_chain.is_(True)))
            )

        from pharmadt.agents.route_agent import RouteAgent

        return [
            DemandAgent(),
            InventoryAgent(graph=world.graph),
            ExpiryAgent(graph=world.graph),
            RouteAgent(graph=world.graph, cold_chain_drugs=cold),
        ]

    # ── Reads ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        if self.world is None:
            return {}
        return self.world.snapshot()

    def kpis(self) -> dict[str, Any]:
        if self.world is None:
            return {}
        from pharmadt.twin.simulation import compute_kpis

        return compute_kpis(self.world)

    def decisions(self, limit: int = 100, agent: str | None = None) -> list[dict[str, Any]]:
        """Recent agent decisions, newest first."""
        if self.orchestrator is None:
            return []
        rows = [
            {
                "agent_name": a.name,
                "sim_day": d.sim_day,
                "action": d.action,
                "justification": d.justification,
            }
            for a in self.orchestrator.agents
            for d in a.decisions
            if agent is None or a.name == agent
        ]
        rows.sort(key=lambda r: r["sim_day"], reverse=True)
        return rows[:limit]

    def new_events(self, limit: int = 200) -> list[dict[str, Any]]:
        """Events emitted since the last call — the WebSocket's payload."""
        if self.world is None:
            return []
        events = self.world.events
        batch = events[self._cursor : self._cursor + limit]
        self._cursor += len(batch)
        return [
            {
                "sim_day": e.sim_day,
                "event_type": str(e.event_type),
                "batch_id": e.batch_id,
                "from_node": e.from_node,
                "to_node": e.to_node,
                "payload": dict(e.payload),
            }
            for e in batch
        ]

    def node_health(self) -> list[dict[str, Any]]:
        """Per-node stock health, for the map. Green / amber / red."""
        if self.world is None:
            return []

        rows = []
        for node_id in sorted(self.world.nodes):
            node = self.world.nodes[node_id]
            utilisation = node.utilisation()
            # Health is about running *out*, not about being full, so an empty
            # node is the alarming one.
            if utilisation < 0.05:
                health = "critical"
            elif utilisation < 0.20:
                health = "low"
            else:
                health = "healthy"
            data = self.world.graph.nodes[node_id]
            rows.append(
                {
                    "node_id": node_id,
                    "node_type": str(data.get("node_type", "")),
                    "lat": data.get("lat"),
                    "lon": data.get("lon"),
                    "stock": node.total_stock(),
                    "capacity": node.storage_capacity,
                    "utilisation": round(utilisation, 4),
                    "health": health,
                    "offline": node_id in self.world.disabled_nodes,
                }
            )
        return rows


#: The process-wide session the routes operate on.
session = SimulationSession()
