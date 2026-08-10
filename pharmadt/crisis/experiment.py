"""Stage 13 Definition of Done: four scenarios, a resilience table, recovery curves.

Each scenario runs twice on the same seed — once on the Stage 3 fixed-threshold
baseline, once with the full agent stack. The comparison is what makes
"resilient" in the project title a measurement rather than an adjective.

The guide asks for heuristic-versus-MADDPG here. MADDPG was cut on its own risk
assessment, so the comparison is baseline-versus-agents instead, which answers
the question the report actually needs: does the agentic layer help when
something goes wrong?

Usage::

    python -m pharmadt.crisis.experiment
    python -m pharmadt.crisis.experiment --scenario pandemic_surge --days 365
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RESULTS = Path("experiments/crisis.json")
CURVES = Path("experiments/recovery_curves.png")


def _build(seed: int, with_agents: bool) -> Any:
    from pharmadt.twin.simulation import attach_agents, build_world

    world = build_world(seed=seed)
    if with_agents:
        from pharmadt.agents.demand import DemandAgent
        from pharmadt.agents.expiry import ExpiryAgent
        from pharmadt.agents.inventory import InventoryAgent

        attach_agents(
            world,
            DemandAgent(),
            InventoryAgent(graph=world.graph),
            ExpiryAgent(graph=world.graph),
        )
    return world


def run_one(scenario: Any, days: int, seed: int, with_agents: bool) -> Any:
    """One scenario, one policy. Returns its resilience metrics."""
    from pharmadt.crisis.injector import crisis_process
    from pharmadt.crisis.scenarios import measure_resilience
    from pharmadt.twin.simulation import compute_kpis, run_simulation

    # Undisrupted control on the same seed and policy. Without it, "recovered
    # to baseline" would compare against a number from a different world.
    control = _build(seed, with_agents)
    run_simulation(control, days)
    baseline_rate = compute_kpis(control)["stockout_rate"]

    world = _build(seed, with_agents)
    world.env.process(crisis_process(world, scenario))
    run_simulation(world, days)

    return measure_resilience(world, scenario, days, baseline_rate)


def plot_curves(results: dict[str, dict[str, Any]], path: Path = CURVES) -> Path | None:
    """Recovery curves: stockout rate per day, disruption window shaded."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None

    scenarios = sorted(results)
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(11, 2.6 * len(scenarios)),
                             sharex=True)
    axes = [axes] if len(scenarios) == 1 else list(axes)

    for ax, name in zip(axes, scenarios, strict=True):
        entry = results[name]
        for policy, colour in (("baseline", "#C44E52"), ("agents", "#4C72B0")):
            series = entry[policy].get("daily_stockout") or []
            if series:
                # 7-day rolling mean: daily stockout is spiky, and the shape of
                # the recovery is the point rather than any single day.
                smoothed = [
                    sum(series[max(0, i - 6): i + 1]) / len(series[max(0, i - 6): i + 1])
                    for i in range(len(series))
                ]
                ax.plot(smoothed, label=policy, color=colour, lw=1.6)

        ax.axvspan(entry["trigger_day"], entry["end_day"], color="grey", alpha=0.15)
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_ylabel("stockout rate")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("simulated day")
    fig.suptitle("Recovery curves: disruption window shaded", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    from pharmadt.crisis.scenarios import load_all, load_scenario, summarise

    parser = argparse.ArgumentParser(description="Run the crisis scenarios.")
    parser.add_argument("--scenario", help="one scenario by name; default is all four")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scenarios = [load_scenario(args.scenario)] if args.scenario else load_all()
    if not scenarios:
        raise SystemExit("no scenarios found in data/scenarios/")

    results: dict[str, dict[str, Any]] = {}
    for policy in ("baseline", "agents"):
        print(f"\n{'=' * 74}\n{policy.upper()} POLICY\n{'=' * 74}")
        metrics = []
        for scenario in scenarios:
            print(f"  running {scenario.name} (day {scenario.trigger_day}, "
                  f"{scenario.duration_days}d)...")
            outcome = run_one(scenario, args.days, args.seed, policy == "agents")
            metrics.append(outcome)
            results.setdefault(scenario.name, {
                "description": scenario.description,
                "trigger_day": scenario.trigger_day,
                "end_day": scenario.end_day,
            })[policy] = {**outcome.as_dict(), "daily_stockout": outcome.daily_stockout}
        print()
        print(summarise(metrics))

    print(f"\n{'=' * 74}\nRESILIENCE COMPARISON -- does the agent layer help?\n{'=' * 74}")
    header = (f"{'scenario':<22}{'peak base':>11}{'peak agents':>13}"
              f"{'unmet base':>12}{'unmet agents':>14}{'recovery':>12}")
    print(header)
    print("-" * len(header))
    for name in sorted(results):
        base, agents = results[name]["baseline"], results[name]["agents"]
        rb, ra = base["time_to_recover_days"], agents["time_to_recover_days"]
        recovery = (
            f"{rb}d -> {ra}d" if rb is not None and ra is not None
            else f"{rb or 'never'} -> {ra or 'never'}"
        )
        print(f"{name:<22}{base['peak_stockout']:>11.4f}{agents['peak_stockout']:>13.4f}"
              f"{base['total_unmet_units']:>12,}{agents['total_unmet_units']:>14,}"
              f"{recovery:>12}")

    total_base = sum(r["baseline"]["total_unmet_units"] for r in results.values())
    total_agents = sum(r["agents"]["total_unmet_units"] for r in results.values())
    if total_base:
        change = (total_base - total_agents) / total_base
        direction = "reduction" if change >= 0 else "INCREASE"
        print(f"\n  total unmet demand across all four crises: "
              f"{total_base:,} -> {total_agents:,} ({abs(change):.1%} {direction})")

    curve_path = plot_curves(results)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n  Wrote {RESULTS}")
    if curve_path:
        print(f"  Wrote {curve_path}")


if __name__ == "__main__":
    main()
