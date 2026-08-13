"""Stage 15: the experiment matrix, run over many seeds, plus claim validation.

This table is the core of the final report. Every configuration adds one agent
to the previous row, so each line answers "what did this component buy?" rather
than only "how good is the finished system?".

**Ten seeds, mean ± standard deviation.** The guide is explicit that single-run
numbers are not defensible, and it is right: the Stage 8 wastage figures swung
between 0 and 1,755 units across seeds. A mean without a spread beside it would
have hidden that entirely.

Usage::

    python -m pharmadt.evaluation                 # 10 seeds, full matrix
    python -m pharmadt.evaluation --seeds 3       # quick pass
    python -m pharmadt.evaluation --no-route      # skip the slow CVRP arm
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

RESULTS = Path("experiments/ablation_matrix.json")

#: Columns of the report table, in the order the guide lists them.
METRICS = (
    "stockout_pct",
    "wastage_units",
    "forecast_mape",
    "delivery_km",
    "average_inventory",
)


def _cold_chain_drugs() -> frozenset[str]:
    from sqlalchemy import select

    from pharmadt.core.db import session_scope
    from pharmadt.core.models import Drug

    with session_scope() as session:
        return frozenset(
            session.scalars(select(Drug.drug_id).where(Drug.requires_cold_chain.is_(True)))
        )


def build_configurations(include_route: bool = True) -> dict[str, Callable[[Any], list]]:
    """Cumulative configurations: each row adds one component to the last."""
    from pharmadt.agents.anomaly import AnomalyAgent
    from pharmadt.agents.demand import DemandAgent
    from pharmadt.agents.expiry import ExpiryAgent
    from pharmadt.agents.inventory import InventoryAgent
    from pharmadt.agents.route_agent import RouteAgent
    from pharmadt.ledger.chain import HashChainLedger

    cold = _cold_chain_drugs()

    configs: dict[str, Callable[[Any], list]] = {
        "baseline (no agents)": lambda w: [],
        "+ inventory & demand": lambda w: [DemandAgent(), InventoryAgent(graph=w.graph)],
        "+ expiry (redistribution)": lambda w: [
            DemandAgent(), InventoryAgent(graph=w.graph), ExpiryAgent(graph=w.graph),
        ],
    }
    if include_route:
        configs["+ route optimisation"] = lambda w: [
            DemandAgent(), InventoryAgent(graph=w.graph), ExpiryAgent(graph=w.graph),
            RouteAgent(graph=w.graph, cold_chain_drugs=cold),
        ]
        configs["+ anomaly & ledger (full)"] = lambda w: [
            DemandAgent(), InventoryAgent(graph=w.graph), ExpiryAgent(graph=w.graph),
            RouteAgent(graph=w.graph, cold_chain_drugs=cold),
            AnomalyAgent(ledger=HashChainLedger(), world=w),
        ]
    else:
        configs["+ anomaly & ledger"] = lambda w: [
            DemandAgent(), InventoryAgent(graph=w.graph), ExpiryAgent(graph=w.graph),
            AnomalyAgent(ledger=HashChainLedger(), world=w),
        ]
    return configs


def run_once(make_agents: Callable[[Any], list], seed: int, days: int) -> dict[str, float]:
    """One configuration, one seed. Returns the row's metrics."""
    from pharmadt.agents.demand import DemandAgent
    from pharmadt.agents.route_agent import RouteAgent
    from pharmadt.twin.simulation import (
        attach_agents,
        build_world,
        compute_kpis,
        run_simulation,
    )

    world = build_world(seed=seed)
    agents = make_agents(world)
    if agents:
        attach_agents(world, *agents)
    run_simulation(world, days)

    kpis = compute_kpis(world)
    row = {
        "stockout_pct": kpis["stockout_rate"] * 100,
        "wastage_units": float(kpis["wastage_units"]),
        "average_inventory": float(kpis["average_inventory"]),
        "forecast_mape": float("nan"),
        "delivery_km": float("nan"),
    }

    for agent in agents:
        if isinstance(agent, DemandAgent):
            row["forecast_mape"] = _forecast_mape(world, agent)
        elif isinstance(agent, RouteAgent):
            row["delivery_km"] = agent.total_distance_km
    return row


def _forecast_mape(world: Any, agent: Any) -> float:
    """MAPE of the agent's own forecasts against realised demand."""
    import numpy as np

    errors = []
    for (node_id, drug_id), forecast in agent.latest.items():
        node = world.nodes.get(node_id)
        if node is None:
            continue
        history = list(node.demand_history.get(drug_id, ()))
        actual = float(np.mean(history[-7:])) if history else 0.0
        if actual > 0:
            errors.append(abs(float(np.mean(forecast)) - actual) / actual)
    return float(np.mean(errors)) * 100 if errors else float("nan")


def aggregate(rows: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and standard deviation per metric. NaNs are dropped, not counted."""
    import math

    summary: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = [r[metric] for r in rows if not math.isnan(r.get(metric, float("nan")))]
        summary[metric] = {
            "mean": statistics.fmean(values) if values else float("nan"),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return summary


def run_matrix(seeds: Sequence[int], days: int, include_route: bool) -> dict[str, Any]:
    configs = build_configurations(include_route)
    matrix: dict[str, Any] = {}

    for name, make_agents in configs.items():
        started = time.perf_counter()
        rows = [run_once(make_agents, seed, days) for seed in seeds]
        matrix[name] = {
            "summary": aggregate(rows),
            "seeds": list(seeds),
            "seconds": round(time.perf_counter() - started, 1),
        }
        stats = matrix[name]["summary"]
        print(
            f"  {name:<28} stockout {stats['stockout_pct']['mean']:.4f}% "
            f"+/- {stats['stockout_pct']['std']:.4f}   "
            f"wastage {stats['wastage_units']['mean']:.0f} "
            f"+/- {stats['wastage_units']['std']:.0f}   "
            f"({matrix[name]['seconds']}s)"
        )
    return matrix


# ── The report table ──────────────────────────────────────────────────


def render(matrix: dict[str, Any]) -> str:
    import math

    headings = {
        "stockout_pct": "stockout %",
        "wastage_units": "wastage",
        "forecast_mape": "MAPE %",
        "delivery_km": "delivery km",
        "average_inventory": "avg inventory",
    }
    width = max(len(n) for n in matrix) + 2
    header = f"{'configuration':<{width}}" + "".join(f"{headings[m]:>22}" for m in METRICS)
    lines = [header, "-" * len(header)]

    for name, entry in matrix.items():
        row = f"{name:<{width}}"
        for metric in METRICS:
            stats = entry["summary"][metric]
            if math.isnan(stats["mean"]):
                row += f"{'--':>22}"
            elif metric in ("wastage_units", "average_inventory", "delivery_km"):
                row += f"{stats['mean']:>12,.0f} +/-{stats['std']:>7,.0f}"
            else:
                row += f"{stats['mean']:>12.4f} +/-{stats['std']:>7.4f}"
        lines.append(row)
    return "\n".join(lines)


def validate_abstract(matrix: dict[str, Any]) -> list[str]:
    """Check the abstract's promises against what was measured.

    The guide's instruction is unambiguous: if the measured value differs,
    update the abstract to the measured value. A defended real number beats an
    undefended target, and an examiner can run this script.
    """
    import math

    names = list(matrix)
    baseline = matrix[names[0]]["summary"]
    full = matrix[names[-1]]["summary"]
    findings: list[str] = []

    # Claim 1: 30-40% wastage reduction, against the control the guide names —
    # a *no-redistribution* baseline, meaning the agent stack without the
    # Expiry Agent, not the no-agent baseline.
    #
    # The control matters more than it looks. The no-agent baseline wastes
    # little because it holds less stock and stocks out far more often;
    # comparing wastage across two policies running at different service levels
    # measures the service level, not the redistribution. Both comparisons are
    # printed so neither is hidden.
    no_redistribution = next(
        (matrix[n]["summary"] for n in names if "inventory" in n.lower()), None
    )
    full_waste = full["wastage_units"]["mean"]

    if no_redistribution is not None:
        control_waste = no_redistribution["wastage_units"]["mean"]
        if control_waste > 0:
            change = (control_waste - full_waste) / control_waste * 100
            verdict = "MEETS" if 30 <= change <= 40 else (
                "EXCEEDS" if change > 40 else "FALLS SHORT OF"
            )
            findings.append(
                f"Wastage vs the no-redistribution control (the guide's stated "
                f"comparison): {control_waste:,.0f} -> {full_waste:,.0f} units "
                f"= {change:.1f}%  [{verdict} the abstract's 30-40% claim]"
            )

    base_waste = baseline["wastage_units"]["mean"]
    if base_waste > 0:
        raw = (base_waste - full_waste) / base_waste * 100
        findings.append(
            f"Wastage vs the no-agent baseline: {base_waste:,.0f} -> "
            f"{full_waste:,.0f} units = {raw:+.1f}%. This comparison is "
            "confounded and is reported only for completeness: the no-agent "
            "baseline holds "
            f"{baseline['average_inventory']['mean']:,.0f} units against "
            f"{full['average_inventory']['mean']:,.0f} and stocks out "
            f"{baseline['stockout_pct']['mean'] / max(full['stockout_pct']['mean'], 1e-9):.0f}x "
            "more often. It wastes less because it runs out instead."
        )

    # Claim 2: 20-25% forecast improvement.
    mape = full["forecast_mape"]["mean"]
    if not math.isnan(mape):
        findings.append(
            f"Forecast MAPE (in-simulation, full system): {mape:.2f}%. "
            "The 20-25% improvement claim is evidenced by Stage 7's held-out "
            "comparison (LSTM sMAPE 13.19 vs seasonal-naive 30.06 = 56% better), "
            "not by this column."
        )

    # The headline the matrix does establish.
    base_stockout = baseline["stockout_pct"]["mean"]
    full_stockout = full["stockout_pct"]["mean"]
    if base_stockout > 0:
        findings.append(
            f"Stockout reduction: {base_stockout:.4f}% -> {full_stockout:.4f}% "
            f"= {(base_stockout - full_stockout) / base_stockout * 100:.1f}% "
            "reduction (mean over all seeds)"
        )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 15 evaluation matrix.")
    parser.add_argument("--seeds", type=int, default=10,
                        help="number of random seeds; the guide asks for >= 10")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--no-route", action="store_true",
                        help="skip the CVRP arm, which dominates runtime")
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    print(f"\nExperiment matrix: {args.seeds} seeds x {args.days} days\n")

    started = time.perf_counter()
    matrix = run_matrix(seeds, args.days, include_route=not args.no_route)
    elapsed = time.perf_counter() - started

    print(f"\n{'=' * 118}")
    print(render(matrix))
    print(f"{'=' * 118}")

    print("\nAbstract claims, checked against measurement:\n")
    for finding in validate_abstract(matrix):
        print(f"  * {finding}")

    print(f"\n  {args.seeds} seeds, mean +/- standard deviation. "
          "Single-run numbers are not defensible.")
    print(f"  completed in {elapsed / 60:.1f} min")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(matrix, indent=2, default=str), encoding="utf-8")
    print(f"  Wrote {RESULTS}")


if __name__ == "__main__":
    main()
