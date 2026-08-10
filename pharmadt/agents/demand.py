"""Demand Prediction Agent — LLD Box 2.

Forecasts 14 days ahead per (node, drug) and publishes three things:

* ``forecast.data``       -> the Inventory Agent, which sizes reorder points
* ``shortage.alert``      -> the Inventory Agent, when stock will not last
* ``demand.hotspot``      -> the Expiry Agent, so near-expiry stock is steered
                             toward nodes that will actually consume it

A trained model is optional. Without one the agent forecasts with a moving
average, which the Stage 7 evaluation measured at MASE 0.69 — comfortably
better than seasonal-naive and a legitimate policy in its own right. That
matters for the ablation: "agents with a learned forecaster" and "agents with a
cheap forecaster" become separable rows rather than one confounded result.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pharmadt.agents.base import BaseAgent
from pharmadt.agents.bus import Topic
from pharmadt.core.events import Action
from pharmadt.core.interfaces import DemandModel
from pharmadt.ml.forecasting import HORIZON, LOOKBACK, moving_average_forecast

logger = logging.getLogger(__name__)

#: Cover for this many days when judging whether stock will run out.
COVER_DAYS = 7
#: Demand above this multiple of its own recent mean counts as a hotspot.
HOTSPOT_RATIO = 1.25

#: Node types whose recorded demand is genuine consumer demand.
#:
#: The agent forecasts these and nothing else. At a warehouse or distributor,
#: "demand" is the *order flow the Inventory Agent itself generates* — lumpy
#: bursts rather than smooth consumption. Forecasting that and feeding the
#: result back into the policy that produced it closes a positive feedback
#: loop: a moving average lands on a recent burst, roughly doubles the demand
#: estimate, which enlarges the next order, which enlarges the next estimate.
#:
#: Measured on a 120-day run, the forecast/history ratio was 1.00 at retail
#: nodes and 2.26 (peaking at 4.00) upstream. Leaving upstream in cost 68% more
#: inventory and 700x the wastage. This is bullwhip amplification, and scoping
#: the forecaster to real demand is what removes it.
CONSUMER_FACING = frozenset({"PHARMACY", "HOSPITAL"})


class DemandAgent(BaseAgent):
    """Publishes forecasts, shortage alerts, and demand hotspots."""

    name = "DemandAgent"

    def __init__(
        self,
        model: DemandModel | None = None,
        horizon: int = HORIZON,
        lookback: int = LOOKBACK,
        cover_days: int = COVER_DAYS,
        hotspot_ratio: float = HOTSPOT_RATIO,
        consumer_facing: frozenset[str] = CONSUMER_FACING,
        **kwargs: Any,
    ) -> None:
        self.consumer_facing = consumer_facing
        self.model = model
        self.horizon = horizon
        self.lookback = lookback
        self.cover_days = cover_days
        self.hotspot_ratio = hotspot_ratio
        #: Latest forecast per (node, drug); the dashboard reads this.
        self.latest: dict[tuple[str, str], np.ndarray] = {}
        super().__init__(**kwargs)

    # ── Observe ───────────────────────────────────────────────────────

    def observe(self, world_state: Mapping[str, Any]) -> dict[str, Any]:
        series: list[dict[str, Any]] = []

        for node_id, node in world_state.get("nodes", {}).items():
            if str(node.get("node_type", "")) not in self.consumer_facing:
                continue

            history: Mapping[str, Sequence[int]] = node.get("demand_history", {})
            stock: Mapping[str, int] = node.get("stock_by_drug", {})
            inbound: Mapping[str, int] = node.get("pending_inbound", {})

            for drug_id, window in history.items():
                values = list(window)
                if not values or sum(values) <= 0:
                    continue  # nothing to learn from yet
                series.append(
                    {
                        "node_id": node_id,
                        "drug_id": drug_id,
                        "history": values,
                        "position": int(stock.get(drug_id, 0))
                        + int(inbound.get(drug_id, 0)),
                    }
                )

        return {"sim_day": world_state.get("sim_day", 0), "series": series}

    # ── Forecast ──────────────────────────────────────────────────────

    def _forecast(self, histories: list[list[int]]) -> np.ndarray:
        """One batched forecast for every series, shape (n, horizon)."""
        padded = np.array(
            [self._pad(h) for h in histories], dtype=np.float32
        )

        if self.model is None:
            return np.asarray(moving_average_forecast(padded, self.horizon), dtype=float)

        try:
            return np.asarray(
                self.model.predict(padded[..., None], self.horizon), dtype=float
            )
        except Exception:
            # A model failure must not take the simulation down with it; the
            # cheap forecaster is a working policy, so degrade to it and say so.
            logger.warning("forecast model failed; falling back to moving average",
                           exc_info=True)
            return np.asarray(moving_average_forecast(padded, self.horizon), dtype=float)

    def _pad(self, history: Sequence[int]) -> list[float]:
        """Left-pad a short window with its own mean.

        Zero-padding would tell the model the shop was shut, dragging the
        forecast down for exactly the new series that most need a sane one.
        """
        values = [float(v) for v in history][-self.lookback :]
        if len(values) < self.lookback:
            filler = float(np.mean(values)) if values else 0.0
            values = [filler] * (self.lookback - len(values)) + values
        return values

    # ── Decide ────────────────────────────────────────────────────────

    def decide(self, observation: Mapping[str, Any]) -> list[Action]:
        series = observation.get("series", [])
        if not series:
            return []

        forecasts = self._forecast([s["history"] for s in series])
        sim_day = observation.get("sim_day", 0)

        payloads: list[dict[str, Any]] = []
        actions: list[Action] = []

        for entry, forecast in zip(series, forecasts, strict=True):
            key = (entry["node_id"], entry["drug_id"])
            self.latest[key] = forecast

            mean_daily = float(np.mean(forecast))
            cover_need = float(np.sum(forecast[: self.cover_days]))
            recent_mean = float(np.mean(entry["history"])) or 1e-9

            payloads.append(
                {
                    "node_id": entry["node_id"],
                    "drug_id": entry["drug_id"],
                    "mean_daily": round(mean_daily, 3),
                    "horizon_total": round(float(np.sum(forecast)), 3),
                }
            )

            if entry["position"] < cover_need:
                actions.append(
                    Action(
                        action_type="SHORTAGE_ALERT",
                        target_node=entry["node_id"],
                        drug_id=entry["drug_id"],
                        quantity=int(cover_need - entry["position"]),
                        params={"mean_daily": round(mean_daily, 3),
                                "cover_days": self.cover_days},
                        justification=(
                            f"position {entry['position']} covers less than the "
                            f"{self.cover_days}-day forecast of {cover_need:.0f}"
                        ),
                    )
                )

            if mean_daily > recent_mean * self.hotspot_ratio:
                actions.append(
                    Action(
                        action_type="DEMAND_HOTSPOT",
                        target_node=entry["node_id"],
                        drug_id=entry["drug_id"],
                        params={"mean_daily": round(mean_daily, 3),
                                "recent_mean": round(recent_mean, 3)},
                        justification=(
                            f"forecast {mean_daily:.1f}/day is "
                            f"{mean_daily / recent_mean:.2f}x recent demand; "
                            "near-expiry stock should be steered here"
                        ),
                    )
                )

        # One action carries the whole forecast batch. Emitting one per pair
        # would add ~11,000 audit rows a year that all say the same thing.
        actions.insert(
            0,
            Action(
                action_type="FORECAST",
                params={"forecasts": payloads, "sim_day": sim_day},
                justification=(
                    f"{self.horizon}-day forecast for {len(payloads)} (node, drug) "
                    f"pairs using {'a trained model' if self.model else 'a moving average'}"
                ),
            ),
        )
        return actions

    # ── Act ───────────────────────────────────────────────────────────

    def apply(self, action: Action, world: Any) -> None:
        """Publishing *is* the action; this agent never mutates the twin."""
        sim_day = getattr(self, "_sim_day", 0)

        if action.action_type == "FORECAST":
            for payload in action.params["forecasts"]:
                self.publish(Topic.FORECAST_DATA, payload, sim_day=sim_day)

        elif action.action_type == "SHORTAGE_ALERT":
            self.publish(
                Topic.SHORTAGE_ALERT,
                {
                    "node_id": action.target_node,
                    "drug_id": action.drug_id,
                    "shortfall": action.quantity,
                    **action.params,
                },
                sim_day=sim_day,
            )

        elif action.action_type == "DEMAND_HOTSPOT":
            self.publish(
                Topic.DEMAND_HOTSPOT,
                {
                    "node_id": action.target_node,
                    "drug_id": action.drug_id,
                    **action.params,
                },
                sim_day=sim_day,
            )

    # ── Inspection ────────────────────────────────────────────────────

    def forecast_for(self, node_id: str, drug_id: str) -> np.ndarray | None:
        return self.latest.get((node_id, drug_id))
