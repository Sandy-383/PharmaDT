"""Parameterised what-if scenarios (FR-09) and the resilience metrics.

Scenarios are YAML in ``data/scenarios/``, so a new one needs no code. Each
declares a trigger day, a duration, and one or more effects.

**Every effect is reversible.** The injector records what it changed and puts it
back when the window closes. That is what makes recovery measurable at all: if a
disruption never lifted, "time to recover" would only ever be "never", and two
scenarios could not run in the same simulation without silently corrupting each
other's state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCENARIO_DIR = Path("data/scenarios")


@dataclass(slots=True)
class Effect:
    """One reversible change to the world."""

    kind: str
    magnitude: float = 1.0
    node_types: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    from_nodes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Scenario:
    """A named disruption with a trigger, a duration, and effects."""

    name: str
    description: str
    trigger_day: int
    duration_days: int
    effects: list[Effect] = field(default_factory=list)

    @property
    def end_day(self) -> int:
        return self.trigger_day + self.duration_days

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            trigger_day=int(data["trigger_day"]),
            duration_days=int(data["duration_days"]),
            effects=[Effect(**e) for e in data.get("effects", [])],
        )


def load_scenario(path: str | Path) -> Scenario:
    import yaml

    path = Path(path)
    if not path.exists():
        candidate = SCENARIO_DIR / f"{path.stem}.yaml"
        if not candidate.exists():
            raise FileNotFoundError(f"no scenario at {path} or {candidate}")
        path = candidate
    return Scenario.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_all(directory: Path = SCENARIO_DIR) -> list[Scenario]:
    if not directory.exists():
        return []
    return [load_scenario(p) for p in sorted(directory.glob("*.yaml"))]


# ── Resilience metrics ────────────────────────────────────────────────


@dataclass(slots=True)
class ResilienceMetrics:
    """How badly a disruption hurt, and how long the damage lasted."""

    scenario: str
    baseline_stockout: float
    peak_stockout: float
    peak_day: int
    total_unmet_units: int
    time_to_detect_days: int | None
    time_to_recover_days: int | None
    recovered: bool
    daily_stockout: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "baseline_stockout": round(self.baseline_stockout, 5),
            "peak_stockout": round(self.peak_stockout, 5),
            "peak_day": self.peak_day,
            "total_unmet_units": self.total_unmet_units,
            "time_to_detect_days": self.time_to_detect_days,
            "time_to_recover_days": self.time_to_recover_days,
            "recovered": self.recovered,
        }


def daily_stockout_rate(world: Any, days: int) -> list[float]:
    """Unmet fraction of demand per simulated day."""
    demanded = [0] * (days + 1)
    unmet = [0] * (days + 1)

    for outcome in world.demand_outcomes:
        if 0 <= outcome.sim_day <= days:
            demanded[outcome.sim_day] += outcome.demanded
            unmet[outcome.sim_day] += outcome.demanded - outcome.fulfilled

    return [
        (unmet[d] / demanded[d]) if demanded[d] else 0.0 for d in range(days + 1)
    ]


def measure_resilience(
    world: Any,
    scenario: Scenario,
    days: int,
    baseline_rate: float,
    tolerance: float = 1.5,
    detect_threshold: float = 2.0,
) -> ResilienceMetrics:
    """Recovery metrics for one scenario run.

    ``time_to_recover`` is measured from the trigger day to the first day the
    stockout rate falls back within ``tolerance`` x the undisrupted baseline
    **and stays there**. Requiring persistence matters: a single quiet day in
    the middle of a shortage is not a recovery, and taking the first crossing
    would report one.
    """
    series = daily_stockout_rate(world, days)
    window = series[scenario.trigger_day : min(days, scenario.end_day) + 1] or [0.0]

    peak = max(window)
    peak_day = scenario.trigger_day + window.index(peak)
    unmet = sum(
        outcome.demanded - outcome.fulfilled
        for outcome in world.demand_outcomes
        if scenario.trigger_day <= outcome.sim_day <= days
    )

    detect_at = next(
        (
            day
            for day in range(scenario.trigger_day, min(days, scenario.end_day) + 1)
            if series[day] > max(baseline_rate * detect_threshold, 1e-6)
        ),
        None,
    )

    recover_at = None
    ceiling = max(baseline_rate * tolerance, 1e-6)
    for day in range(scenario.trigger_day, days + 1):
        if series[day] <= ceiling and all(
            value <= ceiling for value in series[day : min(day + 14, days + 1)]
        ):
            recover_at = day
            break

    return ResilienceMetrics(
        scenario=scenario.name,
        baseline_stockout=baseline_rate,
        peak_stockout=peak,
        peak_day=peak_day,
        total_unmet_units=int(unmet),
        time_to_detect_days=(detect_at - scenario.trigger_day) if detect_at else None,
        time_to_recover_days=(recover_at - scenario.trigger_day) if recover_at else None,
        recovered=recover_at is not None,
        daily_stockout=series,
    )


def summarise(metrics: Sequence[ResilienceMetrics]) -> str:
    """Render the resilience comparison table."""
    header = (
        f"{'scenario':<22}{'peak stockout':>15}{'peak day':>10}"
        f"{'unmet units':>13}{'detect':>8}{'recover':>9}"
    )
    lines = [header, "-" * len(header)]
    for m in metrics:
        detect = f"{m.time_to_detect_days}d" if m.time_to_detect_days is not None else "--"
        recover = f"{m.time_to_recover_days}d" if m.recovered else "never"
        lines.append(
            f"{m.scenario:<22}{m.peak_stockout:>15.4f}{m.peak_day:>10}"
            f"{m.total_unmet_units:>13,}{detect:>8}{recover:>9}"
        )
    return "\n".join(lines)
