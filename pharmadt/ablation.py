"""Ablation harness: run the twin under different policies and compare KPIs.

Every stage from 6 onward has to answer "did the agent actually help?", and the
only honest answer is the same simulation, the same seed, one thing changed.
Stage 15's ablation matrix is this module with more rows.

Usage::

    python -m pharmadt.ablation                      # every variant
    python -m pharmadt.ablation --variants baseline inventory
    python -m pharmadt.ablation --days 180 --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pharmadt.config import settings

RESULTS_DIR = Path("experiments")

#: KPIs reported for every variant. Stockout rate is the headline; the rest
#: exist so an improvement bought by simply holding more stock is visible as
#: exactly that rather than passing as a free win.
REPORTED = (
    "stockout_rate",
    "service_level",
    "units_short",
    "wastage_units",
    "average_inventory",
    "shipments_delivered",
    "events",
)


def _build(seed: int) -> Any:
    from pharmadt.twin.simulation import build_world

    return build_world(seed=seed)


def variant_baseline(seed: int) -> Any:
    """Stage 3's fixed-threshold (s, S) policy, no agents. The control arm."""
    return _build(seed)


def variant_inventory(seed: int) -> Any:
    """Stage 6: reorder point with a safety-stock term, no baseline policy."""
    from pharmadt.agents.inventory import InventoryAgent
    from pharmadt.twin.simulation import attach_agents

    world = _build(seed)
    attach_agents(world, InventoryAgent(graph=world.graph))
    return world


def variant_demand(seed: int) -> Any:
    """Stage 7: Demand Agent forecasts feed the Inventory Agent's reorder points."""
    from pharmadt.agents.demand import DemandAgent
    from pharmadt.agents.inventory import InventoryAgent
    from pharmadt.twin.simulation import attach_agents

    world = _build(seed)
    attach_agents(world, DemandAgent(), InventoryAgent(graph=world.graph))
    return world


VARIANTS: dict[str, Callable[[int], Any]] = {
    "baseline": variant_baseline,
    "inventory": variant_inventory,
    "demand": variant_demand,
}


def run_variant(name: str, days: int, seed: int) -> dict[str, Any]:
    """Run one variant and return its KPIs."""
    from pharmadt.twin.simulation import compute_kpis, run_simulation

    world = VARIANTS[name](seed)
    run_simulation(world, days)
    kpis = compute_kpis(world)

    result: dict[str, Any] = {"variant": name, "seed": seed, "days": days}
    result.update({k: kpis.get(k) for k in REPORTED})
    if world.orchestrator is not None:
        result["agent_decisions"] = len(world.orchestrator.collect_decisions())
    else:
        result["agent_decisions"] = 0
    return result


def compare(
    names: Sequence[str], days: int, seeds: Sequence[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every (variant, seed) pair. Returns raw rows and per-variant means.

    Several seeds because a single run can flatter a policy by luck; a mean
    over seeds is the weakest claim that is still honest.
    """
    rows = [run_variant(name, days, seed) for name in names for seed in seeds]

    summary: list[dict[str, Any]] = []
    for name in names:
        matching = [r for r in rows if r["variant"] == name]
        entry: dict[str, Any] = {"variant": name, "runs": len(matching)}
        for metric in REPORTED:
            values = [r[metric] for r in matching if r[metric] is not None]
            entry[metric] = round(statistics.fmean(values), 6) if values else None
        entry["agent_decisions"] = int(
            statistics.fmean([r["agent_decisions"] for r in matching])
        )
        summary.append(entry)
    return rows, summary


def _print_table(summary: Sequence[dict[str, Any]]) -> None:
    metrics = ("stockout_rate", "service_level", "units_short", "wastage_units",
               "average_inventory", "agent_decisions")
    width = max(len(s["variant"]) for s in summary) + 2

    header = f"{'variant':<{width}}" + "".join(f"{m:>20}" for m in metrics)
    print(header)
    print("-" * len(header))
    for entry in summary:
        line = f"{entry['variant']:<{width}}"
        for metric in metrics:
            value = entry.get(metric)
            line += f"{value:>20,.5f}" if isinstance(value, float) else f"{value:>20,}"
        print(line)


def _print_delta(summary: Sequence[dict[str, Any]]) -> None:
    """State the Stage 6 DoD claim in the terms the guide asks for."""
    by_name = {s["variant"]: s for s in summary}
    base, agent = by_name.get("baseline"), by_name.get("inventory")
    if not base or not agent:
        return

    before, after = base["stockout_rate"], agent["stockout_rate"]
    if before in (None, 0):
        print("\nBaseline stockout rate is zero; no headroom to improve on.")
        return

    change = (before - after) / before
    direction = "reduction" if change >= 0 else "INCREASE"
    print(
        f"\nStockout rate {before:.5f} -> {after:.5f} "
        f"({abs(change):.1%} {direction} vs the fixed-threshold baseline)"
    )
    stock_change = (agent["average_inventory"] - base["average_inventory"]) / base[
        "average_inventory"
    ]
    print(
        f"Average inventory {base['average_inventory']:,.0f} -> "
        f"{agent['average_inventory']:,.0f} ({stock_change:+.1%}) "
        "-- a stockout win bought purely with more stock is not a win"
    )


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def service_level_frontier(
    days: int, seeds: Sequence[int], z_values: Sequence[float]
) -> list[dict[str, Any]]:
    """Trace the service-level / holding-cost frontier over ``z``.

    The point of the agent is not merely fewer stockouts — it is that the
    trade-off becomes *tunable*. The fixed-threshold baseline offers no dial at
    all, so it sits at one arbitrary point on this curve.
    """
    from pharmadt.agents.inventory import InventoryAgent
    from pharmadt.twin.simulation import attach_agents, compute_kpis, run_simulation

    rows: list[dict[str, Any]] = []
    for z in z_values:
        for seed in seeds:
            world = _build(seed)
            attach_agents(world, InventoryAgent(graph=world.graph, z=z))
            run_simulation(world, days)
            kpis = compute_kpis(world)
            rows.append({"z": z, "seed": seed, **{k: kpis.get(k) for k in REPORTED}})

    summary = []
    for z in z_values:
        matching = [r for r in rows if r["z"] == z]
        entry: dict[str, Any] = {"variant": f"z={z:.2f}", "runs": len(matching)}
        for metric in REPORTED:
            values = [r[metric] for r in matching if r[metric] is not None]
            entry[metric] = round(statistics.fmean(values), 6) if values else None
        entry["agent_decisions"] = 0
        summary.append(entry)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare policies on identical runs.")
    parser.add_argument(
        "--frontier",
        action="store_true",
        help="sweep the safety-stock quantile z instead of comparing variants",
    )
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS),
                        default=sorted(VARIANTS))
    parser.add_argument("--days", type=int, default=settings.sim_days)
    parser.add_argument("--seeds", nargs="+", type=int, default=[settings.sim_seed])
    parser.add_argument("--out", default="experiments/ablation.csv")
    args = parser.parse_args()

    if args.frontier:
        z_values = (0.0, 0.52, 0.84, 1.28, 1.65, 2.33)
        print(f"Service-level frontier over {len(args.seeds)} seed(s), {args.days} days\n")
        summary = service_level_frontier(args.days, args.seeds, z_values)
        _print_table(summary)
        print("\nThe baseline has no such dial; it sits at one arbitrary point.")
        print(f"\nWrote {_write_csv(summary, Path('experiments/frontier.csv'))}")
        return

    print(f"Running {len(args.variants)} variant(s) x {len(args.seeds)} seed(s) "
          f"over {args.days} days\n")
    rows, summary = compare(args.variants, args.days, args.seeds)

    _print_table(summary)
    _print_delta(summary)
    print(f"\nWrote {_write_csv(rows, Path(args.out))}")


if __name__ == "__main__":
    main()
