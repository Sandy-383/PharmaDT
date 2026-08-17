"""Every experiment result, normalised into one shape the dashboard can render.

Each stage writes its own JSON to ``experiments/`` in whatever shape suited the
stage that produced it — a list of gate checks, a dict of federated regimes, a
nested matrix of means and standard deviations. Useful on disk, but it meant
the results were only readable by running seven different make targets and
reading seven different console tables.

This flattens all of them into ``Artifact``: a title, the question it answers,
a few headline numbers, and one table. The dashboard then needs a single
renderer rather than seven, which is the whole reason the results are reachable
from the browser at all.

Nothing here recomputes anything. If an artifact is missing, that is reported
as missing with the command that produces it — never silently as a zero.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPERIMENTS = Path("experiments")


@dataclass(slots=True)
class Artifact:
    """One experiment's results, in the shape the dashboard renders."""

    key: str
    title: str
    question: str
    command: str
    file: str
    present: bool = False
    generated: str | None = None
    headline: list[dict[str, str]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    footnote: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "question": self.question,
            "command": self.command,
            "file": self.file,
            "present": self.present,
            "generated": self.generated,
            "headline": self.headline,
            "columns": self.columns,
            "rows": self.rows,
            "footnote": self.footnote,
        }


def _num(value: Any, places: int = 2) -> str:
    """Format a number for a table cell, or a dash when it is absent.

    A missing metric must not render as ``0`` — the Route Agent arm has no
    delivery distance when the agent is not attached, and zero would read as a
    free delivery rather than as a column that does not apply.
    """
    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "--"
    if places == 0:
        return f"{number:,.0f}"
    return f"{number:,.{places}f}"


def _mean_std(stats: dict[str, Any] | None, places: int = 2) -> str:
    if not stats:
        return "--"
    mean = stats.get("mean")
    if mean is None or mean != mean:
        return "--"
    return f"{_num(mean, places)} +/- {_num(stats.get('std', 0.0), places)}"


# ── One builder per artifact ──────────────────────────────────────────


def _evaluation(data: Any, art: Artifact) -> None:
    art.columns = ["configuration", "stockout %", "wastage units",
                   "MAPE %", "delivery km", "avg inventory"]
    for name, entry in data.items():
        summary = entry.get("summary", {})
        art.rows.append([
            name,
            _mean_std(summary.get("stockout_pct"), 4),
            _mean_std(summary.get("wastage_units"), 0),
            _mean_std(summary.get("forecast_mape"), 2),
            _mean_std(summary.get("delivery_km"), 0),
            _mean_std(summary.get("average_inventory"), 0),
        ])

    names = list(data)
    if not names:
        return
    base = data[names[0]]["summary"]
    full = data[names[-1]]["summary"]

    base_so = base["stockout_pct"]["mean"]
    full_so = full["stockout_pct"]["mean"]
    if base_so > 0:
        art.headline.append({
            "label": "stockout reduction",
            "value": f"{(base_so - full_so) / base_so * 100:.1f}%",
            "note": f"{base_so:.4f}% -> {full_so:.4f}%, mean over "
                    f"{base['stockout_pct']['n']} seeds",
        })

    # The guide names a *no-redistribution* control, not the no-agent baseline.
    # The no-agent arm wastes less only because it stocks out instead, so
    # comparing against it measures the service level rather than the agent.
    control = next((data[n]["summary"] for n in names if "inventory" in n.lower()), None)
    if control:
        control_waste = control["wastage_units"]["mean"]
        full_waste = full["wastage_units"]["mean"]
        if control_waste > 0:
            art.headline.append({
                "label": "wastage reduction",
                "value": f"{(control_waste - full_waste) / control_waste * 100:.1f}%",
                "note": f"{control_waste:,.0f} -> {full_waste:,.0f} units versus the "
                        "no-redistribution control",
            })

    seeds = data[names[0]].get("seeds", [])
    art.headline.append({
        "label": "seeds",
        "value": str(len(seeds)),
        "note": "mean +/- standard deviation; single-run numbers are not defensible",
    })
    art.footnote = (
        "Each row adds one component to the row above it, so every line answers "
        "'what did this agent buy?' rather than only 'how good is the finished "
        "system?'. Wastage is measured against the no-redistribution control "
        "because the no-agent baseline holds less stock and stocks out far more "
        "often -- it wastes less because it runs out instead."
    )


def _gate(data: Any, art: Artifact) -> None:
    art.columns = ["#", "condition", "result", "evidence"]
    passed = 0
    for check in data:
        ok = bool(check.get("passed"))
        passed += ok
        art.rows.append([
            check.get("number", ""),
            check.get("name", ""),
            "PASS" if ok else "FAIL",
            check.get("detail", ""),
        ])
    art.headline.append({
        "label": "conditions met",
        "value": f"{passed}/{len(data)}",
        "note": "the Stage 10.5 gate: the whole system, one run, no arm excluded",
    })
    art.footnote = (
        "The gate tampers with a record in the middle of the chain rather than "
        "at the tip. Editing the newest record proves only that the last hash "
        "was recomputed; editing a middle one proves every subsequent link was "
        "checked too."
    )


def _federated(data: Any, art: Artifact) -> None:
    art.columns = ["regime", "MAPE %", "sMAPE %", "MASE", "RMSE", "privacy"]
    for name, metrics in data.items():
        privacy = metrics.get("privacy") or {}
        epsilon = privacy.get("epsilon")
        art.rows.append([
            name.replace("_", " "),
            _num(metrics.get("MAPE")),
            _num(metrics.get("sMAPE")),
            _num(metrics.get("MASE"), 4),
            _num(metrics.get("RMSE"), 0),
            f"eps = {_num(epsilon, 1)}" if epsilon is not None else "none",
        ])

    central = data.get("centralised", {}).get("sMAPE")
    noniid = data.get("federated_noniid", {}).get("sMAPE")
    if central and noniid:
        art.headline.append({
            "label": "cost of federation",
            "value": f"+{(noniid - central) / central * 100:.1f}% sMAPE",
            "note": f"centralised {central:.2f} -> non-IID federated {noniid:.2f}; "
                    "raw data never leaves a client",
        })
    art.headline.append({
        "label": "clients",
        "value": str(data.get("federated_noniid", {}).get("skew", {}).get("n_clients", "--")),
        "note": "Dirichlet(0.5) non-IID partition, so clients hold genuinely "
                "different distributions",
    })
    art.footnote = (
        "The differential-privacy rows are reported honestly rather than "
        "quietly dropped: at eps = 1 the noise destroys the forecast. That is "
        "the real privacy/utility trade-off at this dataset size, and hiding "
        "the rows that show it would misrepresent the method."
    )


def _anomaly(data: Any, art: Artifact) -> None:
    art.columns = ["detector", "precision", "recall", "F1", "ROC-AUC", "TP", "FP", "FN"]
    best_f1, best_name = 0.0, ""
    for name, m in (data.get("models") or {}).items():
        f1 = float(m.get("f1") or 0)
        if f1 > best_f1:
            best_f1, best_name = f1, name
        art.rows.append([
            name.replace("_", " "),
            _num(m.get("precision"), 4), _num(m.get("recall"), 4),
            _num(m.get("f1"), 4), _num(m.get("roc_auc"), 4),
            m.get("tp", "--"), m.get("fp", "--"), m.get("fn", "--"),
        ])

    art.headline += [
        {"label": "best F1", "value": _num(best_f1, 4),
         "note": f"{best_name.replace('_', ' ')}, over {data.get('shipments', 0):,} shipments"},
        {"label": "anomalies injected", "value": f"{data.get('injected_pct', 0)}%",
         "note": f"{data.get('injected', 0)} of {data.get('shipments', 0):,} shipments"},
    ]
    art.footnote = (
        "Accuracy is deliberately not the headline. With a 5% positive class, "
        "a detector that flags nothing scores 95% accuracy and catches no "
        "counterfeits, so precision, recall and F1 are the honest columns."
    )


def _routing(data: Any, art: Artifact) -> None:
    art.columns = ["instance", "customers", "vehicles", "cost", "optimum", "gap %", "status"]
    gaps = []
    for row in data:
        gap = row.get("gap_pct")
        if gap is not None and row.get("feasible"):
            gaps.append(float(gap))
        art.rows.append([
            row.get("instance", ""), row.get("n", "--"), row.get("vehicles", "--"),
            _num(row.get("cost"), 0), _num(row.get("optimum"), 0),
            _num(gap, 2), row.get("status", ""),
        ])
    if gaps:
        art.headline.append({
            "label": "mean gap to optimum",
            "value": f"{sum(gaps) / len(gaps):.2f}%",
            "note": f"over {len(gaps)} solved CVRPLIB instances with published optima",
        })
    art.footnote = (
        "An instance the solver did not finish is reported as NO SOLUTION, not "
        "as a gap. An early version scored an unsolved instance as a "
        "record-breaking -100% gap -- a failure reported as a result."
    )


#: How each forecasting protocol was run. They answer different questions and
#: give different answers, so both are shown rather than whichever flatters.
PROTOCOLS = {
    "shared_protocol": "all models, identical windows",
    "broad_sweep": "fewer models, more windows",
}


def _forecasting(data: Any, art: Artifact) -> None:
    art.columns = ["protocol", "model", "MAPE %", "sMAPE %", "MASE", "RMSE", "windows"]
    for protocol, label in PROTOCOLS.items():
        models = data.get(protocol)
        if not isinstance(models, dict):
            continue
        first = True
        for name, m in models.items():
            if not isinstance(m, dict):
                continue
            art.rows.append([
                label if first else "",
                name.replace("_", " "),
                _num(m.get("MAPE")), _num(m.get("sMAPE")),
                _num(m.get("MASE"), 4), _num(m.get("RMSE"), 0), m.get("n", "--"),
            ])
            first = False

    # The baseline that counts is seasonal-naive, not naive. Beating "yesterday"
    # is trivial for weekly-seasonal demand; beating "the same day last week" is
    # the claim worth making.
    def improvement(protocol: str) -> tuple[float, float, float] | None:
        models = data.get(protocol) or {}
        base = (models.get("seasonal_naive") or {}).get("sMAPE")
        lstm = (models.get("lstm") or {}).get("sMAPE")
        if not base or not lstm:
            return None
        return base, lstm, (base - lstm) / base * 100

    shared = improvement("shared_protocol")
    broad = improvement("broad_sweep")
    if shared:
        art.headline.append({
            "label": "LSTM vs seasonal-naive",
            "value": f"{shared[2]:.0f}%",
            "note": f"sMAPE {shared[0]:.2f} -> {shared[1]:.2f} on the shared protocol, "
                    "where every model sees identical windows",
        })
    if broad:
        art.headline.append({
            "label": "on the broad sweep",
            "value": f"{broad[2]:.0f}%",
            "note": f"sMAPE {broad[0]:.2f} -> {broad[1]:.2f} over "
                    f"{(data.get('broad_sweep', {}).get('lstm', {}) or {}).get('n', '?')} "
                    "windows, but only four models were run",
        })

    art.footnote = (
        "sMAPE and MASE sit beside MAPE because MAPE is undefined when actual "
        "demand is zero, which happens often at a pharmacy.\n\n"
        "Read the shared protocol first, and note what it shows: on identical "
        "windows a moving average (sMAPE 16.24) and the ensemble (16.22) both "
        "edge out the LSTM (18.07). The LSTM's larger margin on the broad sweep "
        "comes from a different, wider set of windows against a weaker "
        "seasonal-naive. Quoting only the broad sweep would overstate the deep "
        "model; the defensible claim is that the LSTM beats the seasonal "
        "baseline comfortably, not that it beats every classical method."
    )


def _crisis(data: Any, art: Artifact) -> None:
    art.columns = ["scenario", "arm", "peak stockout %", "unmet units",
                   "detect (days)", "recover (days)", "recovered"]
    for name, entry in data.items():
        for arm in ("baseline", "agents"):
            result = entry.get(arm)
            if not isinstance(result, dict):
                continue
            art.rows.append([
                name.replace("_", " ") if arm == "baseline" else "",
                arm,
                _num((result.get("peak_stockout") or 0) * 100, 3),
                _num(result.get("total_unmet_units"), 0),
                result.get("time_to_detect_days", "--"),
                result.get("time_to_recover_days", "--"),
                "yes" if result.get("recovered") else "no",
            ])
    art.headline.append({
        "label": "scenarios",
        "value": str(len(data)),
        "note": "each injected mid-run and then reversed, so recovery is "
                "measured rather than assumed",
    })
    art.footnote = (
        "Every scenario is reversible: the disruption lifts and the run "
        "continues, which is what makes time-to-recover a measurement instead "
        "of an assertion."
    )


#: The artifacts the dashboard offers, in the order they are presented.
BUILDERS: tuple[tuple[str, str, str, str, str, Callable[[Any, Artifact], None]], ...] = (
    ("evaluation", "ablation_matrix.json", "Experiment matrix",
     "What did each agent actually buy?", "make evaluate", _evaluation),
    ("gate", "integration_gate.json", "Integration gate",
     "Does the whole system hold together in one run?", "make gate", _gate),
    ("forecasting", "demand_comparison.json", "Demand forecasting",
     "Does the model beat 'same as last week'?", "make data", _forecasting),
    ("anomaly", "anomaly_detection.json", "Anomaly detection",
     "Can it catch a tampered or counterfeit shipment?", "make anomaly-eval", _anomaly),
    ("routing", "cvrplib_gap.json", "Routing benchmark",
     "How close to optimal are the delivery routes?", "make routing-benchmark", _routing),
    ("federated", "federated.json", "Federated learning",
     "What does privacy cost in forecast accuracy?", "make federated", _federated),
    ("crisis", "crisis.json", "Crisis response",
     "How fast does the network recover from a shock?", "make crisis", _crisis),
)


def load(key: str) -> Artifact | None:
    """Build one artifact, or ``None`` if the key is not one we publish."""
    spec = next((b for b in BUILDERS if b[0] == key), None)
    if spec is None:
        return None

    key, filename, title, question, command, build = spec
    art = Artifact(key=key, title=title, question=question,
                   command=command, file=f"experiments/{filename}")

    path = EXPERIMENTS / filename
    if not path.exists():
        return art

    art.present = True
    art.generated = datetime.fromtimestamp(
        path.stat().st_mtime, tz=UTC
    ).isoformat(timespec="seconds")
    try:
        build(json.loads(path.read_text(encoding="utf-8")), art)
    except Exception as exc:  # noqa: BLE001 - a malformed artifact must not 500
        art.present = False
        art.footnote = f"Could not read {art.file}: {type(exc).__name__}: {exc}"
    return art


def load_all() -> list[Artifact]:
    return [art for key, *_ in BUILDERS if (art := load(key)) is not None]
