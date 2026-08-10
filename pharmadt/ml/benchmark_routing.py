"""Validate the CVRP solver against CVRPLIB instances with published optima.

The Stage 9 Definition of Done asks for a gap percentage, and a gap is only
meaningful against a number somebody else computed. Stage 2 downloaded both the
``.vrp`` instances and their ``.sol`` files for exactly this.

Distances use CVRPLIB's EUC_2D convention — Euclidean, rounded to the nearest
integer — because the published optima were computed that way. Any other metric
produces a gap that measures the metric rather than the solver.

Usage::

    python -m pharmadt.ml.benchmark_routing
    python -m pharmadt.ml.benchmark_routing --time-limit 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pharmadt.agents.routing import (
    RoutingProblem,
    check_feasible,
    euclidean_matrix,
    solve_cvrp,
)
from pharmadt.ml.preprocessing import RAW, parse_sol, parse_vrp

RESULTS = Path("experiments/cvrplib_gap.json")


def _vehicle_count(name: str, fallback: int) -> int:
    """CVRPLIB encodes the fleet size in the instance name as ``-k<N>``."""
    for part in name.split("-"):
        if part.startswith("k") and part[1:].isdigit():
            return int(part[1:])
    return fallback


def benchmark(time_limit_s: int = 10) -> list[dict[str, Any]]:
    source = RAW / "cvrplib"
    if not source.exists():
        raise SystemExit(f"{source} missing. Run `make data` first.")

    rows: list[dict[str, Any]] = []
    for vrp_path in sorted(source.glob("*.vrp")):
        instance = parse_vrp(vrp_path.read_text(encoding="utf-8"))
        sol_path = vrp_path.with_suffix(".sol")
        optimum = parse_sol(sol_path.read_text(encoding="utf-8")) if sol_path.exists() else None
        if optimum is None:
            continue

        # Node ids are 1-based in the file; index 0 is the depot.
        coords = [(x, y) for _, x, y in instance["coords"]]
        demands = [instance["demands"].get(node, 0) for node, _, _ in instance["coords"]]
        vehicles = _vehicle_count(instance["name"] or vrp_path.stem, 10)

        problem = RoutingProblem(
            distance_matrix=euclidean_matrix(coords),
            demands=demands,
            vehicle_capacities=[instance["capacity"]] * vehicles,
            depot=0,
        )
        # No dropping: a benchmark that skips stops is not solving the instance.
        plan = solve_cvrp(problem, time_limit_s=time_limit_s, allow_dropping=False)
        violations = check_feasible(plan, problem)

        # An unsolved instance produces zero routes, which trivially violates
        # nothing and costs nothing. Scoring that as a feasible plan with a
        # -100% gap would report failure as a record-breaking result, so a plan
        # only counts if it actually visits every customer.
        expected_stops = instance["dimension"] - 1
        solved = plan.status == "OK" and plan.stops_served == expected_stops
        gap = (
            (plan.total_distance_units - optimum) / optimum * 100
            if solved and optimum
            else None
        )

        rows.append(
            {
                "instance": instance["name"] or vrp_path.stem,
                "n": instance["dimension"],
                "vehicles": vehicles,
                "cost": plan.total_distance_units if solved else None,
                "optimum": optimum,
                "gap_pct": round(gap, 2) if gap is not None else None,
                "seconds": round(plan.solve_seconds, 2),
                "served": plan.stops_served,
                "expected_stops": expected_stops,
                "status": plan.status if solved else "NO SOLUTION",
                "feasible": solved and not violations,
                "violations": violations[:3],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the CVRP solver.")
    parser.add_argument("--time-limit", type=int, default=10)
    args = parser.parse_args()

    rows = benchmark(args.time_limit)
    if not rows:
        raise SystemExit("no CVRPLIB instances with published optima were found")

    header = (
        f"{'instance':<14}{'n':>5}{'veh':>5}{'cost':>10}{'optimum':>10}"
        f"{'gap %':>9}{'sec':>7}{'feasible':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        cost = f"{row['cost']:>10,}" if row["cost"] is not None else f"{'--':>10}"
        gap = f"{row['gap_pct']:>9.2f}" if row["gap_pct"] is not None else f"{'--':>9}"
        print(
            f"{row['instance']:<14}{row['n']:>5}{row['vehicles']:>5}{cost}"
            f"{row['optimum']:>10,.0f}{gap}{row['seconds']:>7.1f}"
            f"{('yes' if row['feasible'] else 'NO'):>10}"
        )

    gaps = [r["gap_pct"] for r in rows if r["gap_pct"] is not None]
    if gaps:
        print(f"\nsolved {len(gaps)}/{len(rows)} instances")
        print(f"mean gap {sum(gaps) / len(gaps):.2f}%  |  best {min(gaps):.2f}%  "
              f"|  worst {max(gaps):.2f}%")
    else:
        print("\nNo instance was solved to completion.")

    for row in rows:
        if row["gap_pct"] is None:
            print(f"  {row['instance']}: {row['status']} "
                  f"({row['served']}/{row['expected_stops']} stops served)")
        elif row["violations"]:
            print(f"  {row['instance']}: {row['violations']}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
