"""Anomaly detection: features, models, ensemble rules, and the ledger check.

The load-bearing test here is the last one. A counterfeit batch moves in normal
quantities over a normal route in normal time, so the shipment features cannot
express it — measured at 0/15 recall for the ML models. Only the ledger catches
it, which is the concrete form of research gap 3.2.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from pharmadt.agents.anomaly import FEATURE_NAMES, AnomalyAgent, extract_features
from pharmadt.agents.bus import MessageBus, Topic
from pharmadt.crisis.injector import AnomalyKind, inject_anomalies
from pharmadt.ml.anomaly import (
    AutoencoderDetector,
    IsolationForestDetector,
    confusion,
    ensemble_predict,
    evaluate_detector,
)


def shipment(**kw) -> dict:
    base = {
        "batch_id": "B1", "drug_id": "D1", "from_node": "DC", "to_node": "PH1",
        "quantity": 100, "transit_days": 1.0, "excursion_count": 0,
        "excursion_severity": 0.0, "distance_km": 10.0, "sim_day": 3,
        "is_known_route": True, "cold_chain": False,
    }
    base.update(kw)
    return base


# ── Features ──────────────────────────────────────────────────────────


def test_the_feature_matrix_has_the_declared_width() -> None:
    """The autoencoder's input width depends on this staying fixed."""
    features = extract_features([shipment(), shipment()])
    assert features.shape == (2, len(FEATURE_NAMES))


def test_no_shipments_yields_an_empty_matrix_of_the_right_width() -> None:
    assert extract_features([]).shape == (0, len(FEATURE_NAMES))


def test_quantity_is_scored_relative_to_its_own_route() -> None:
    """500 units is routine to a warehouse and extraordinary to a village shop."""
    records = [shipment(to_node="PH1", quantity=100) for _ in range(9)]
    records.append(shipment(to_node="PH1", quantity=900))
    records += [shipment(to_node="PH2", quantity=900) for _ in range(10)]

    features = extract_features(records)
    outlier_z = features[9, FEATURE_NAMES.index("quantity_z")]
    routine_z = features[10, FEATURE_NAMES.index("quantity_z")]

    assert outlier_z > 2.0
    assert abs(routine_z) < 1.0


def test_transit_deviation_is_relative_to_the_lane(shipment_count: int = 6) -> None:
    records = [shipment(transit_days=1.0) for _ in range(shipment_count)]
    records.append(shipment(transit_days=9.0))
    features = extract_features(records)
    assert features[-1, FEATURE_NAMES.index("transit_deviation")] == pytest.approx(8.0)


def test_a_zero_distance_route_does_not_divide_by_zero() -> None:
    features = extract_features([shipment(distance_km=0.0)])
    assert np.isfinite(features).all()


# ── Models ────────────────────────────────────────────────────────────


@pytest.fixture
def split_data():
    """Tight normal cluster plus obvious outliers."""
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 1.0, size=(300, len(FEATURE_NAMES)))
    weird = rng.normal(14.0, 1.0, size=(20, len(FEATURE_NAMES)))
    X = np.vstack([normal, weird])
    y = np.array([False] * 300 + [True] * 20)
    return normal, X, y


def test_the_isolation_forest_separates_obvious_outliers(split_data) -> None:
    normal, X, y = split_data
    model = IsolationForestDetector().fit(normal)
    assert confusion(y, model.predict(X)).recall > 0.8


def test_higher_isolation_scores_mean_more_anomalous(split_data) -> None:
    """sklearn signs this the other way; using it raw inverts every threshold."""
    normal, X, y = split_data
    scores = IsolationForestDetector().fit(normal).score(X)
    assert scores[y].mean() > scores[~y].mean()


def test_the_autoencoder_separates_obvious_outliers(split_data) -> None:
    normal, X, y = split_data
    model = AutoencoderDetector(n_features=X.shape[1], epochs=30).fit(normal)
    assert confusion(y, model.predict(X)).recall > 0.8


def test_the_autoencoder_reconstructs_normal_data_better(split_data) -> None:
    normal, X, y = split_data
    scores = AutoencoderDetector(n_features=X.shape[1], epochs=30).fit(normal).score(X)
    assert scores[y].mean() > scores[~y].mean()


def test_a_constant_feature_does_not_produce_infinities() -> None:
    """Dividing by a zero standard deviation would poison every score."""
    X = np.ones((60, len(FEATURE_NAMES)))
    model = AutoencoderDetector(n_features=X.shape[1], epochs=5).fit(X)
    assert np.isfinite(model.score(X)).all()


# ── Ensemble and metrics ──────────────────────────────────────────────


def test_either_favours_recall_and_both_favours_precision() -> None:
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert ensemble_predict(a, b, "either").tolist() == [True, True, True, False]
    assert ensemble_predict(a, b, "both").tolist() == [True, False, False, False]


def test_an_unknown_ensemble_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown ensemble rule"):
        ensemble_predict(np.array([True]), np.array([True]), "sometimes")


def test_a_detector_that_flags_nothing_scores_high_accuracy_and_zero_recall() -> None:
    """The reason accuracy is never the headline in this module."""
    actual = np.array([True] * 5 + [False] * 95)
    scores = confusion(actual, np.zeros(100, dtype=bool))
    assert scores.accuracy == pytest.approx(0.95)
    assert scores.recall == 0.0
    assert scores.f1 == 0.0


def test_evaluation_reports_both_models_and_both_rules() -> None:
    actual = np.array([True, False, True, False])
    results = evaluate_detector(actual, np.array([True, False, False, False]),
                                np.array([False, False, True, True]))
    assert set(results) == {
        "isolation_forest", "autoencoder", "ensemble_either", "ensemble_both"
    }


# ── Injection ─────────────────────────────────────────────────────────


def test_injection_labels_exactly_what_it_corrupts() -> None:
    records = [shipment() for _ in range(200)]
    report = inject_anomalies(records, rate=0.1, seed=1)
    assert report.n_injected == 20
    assert sum(1 for r in records if "anomaly_kind" in r) == 20


def test_a_quantity_anomaly_actually_changes_the_quantity() -> None:
    records = [shipment() for _ in range(50)]
    inject_anomalies(records, rate=1.0, seed=2, kinds=(AnomalyKind.QUANTITY,))
    assert all(r["quantity"] > 100 for r in records)


def test_a_counterfeit_leaves_the_shipment_looking_completely_normal() -> None:
    """This is why the ML models cannot see it — measured at 0/15 recall."""
    records = [shipment() for _ in range(20)]
    before = [dict(r) for r in records]
    inject_anomalies(records, rate=1.0, seed=3, kinds=(AnomalyKind.COUNTERFEIT,))

    for original, corrupted in zip(before, records, strict=True):
        for feature in ("quantity", "transit_days", "excursion_count", "distance_km"):
            assert original[feature] == corrupted[feature]
        assert corrupted["forged_fingerprint"] is True


# ── The agent and the ledger cross-check ──────────────────────────────


class FakeLedger:
    """Minimal ProvenanceLedger stand-in: knows which fingerprints are genuine."""

    def __init__(self, genuine: dict[str, str], known: set[str]) -> None:
        self.genuine, self.known = genuine, known

    def verify_batch_fingerprint(self, batch_id: str, presented: str) -> bool:
        return self.genuine.get(batch_id) == presented

    def get_provenance(self, batch_id: str):
        return [{"seq": 1}] if batch_id in self.known else []


def test_the_ledger_catches_a_forgery_the_models_rank_as_ordinary() -> None:
    """Research gap 3.2.3, as a test.

    No ML model is attached, so the shipment is not suspected at all — which is
    exactly the measured situation for counterfeits. The ledger still condemns it.
    """
    ledger = FakeLedger(genuine={"B1": "a" * 64}, known={"B1"})
    agent = AnomalyAgent(ledger=ledger, bus=MessageBus())

    actions = agent.decide(
        {"sim_day": 4, "shipments": [shipment(presented_fingerprint="f" * 64)]}
    )
    assert len(actions) == 1
    assert actions[0].action_type == "COUNTERFEIT_ALERT"
    assert actions[0].params["ml_flagged"] is False
    assert "only the ledger caught it" in actions[0].justification


def test_a_genuine_fingerprint_raises_no_alert() -> None:
    ledger = FakeLedger(genuine={"B1": "a" * 64}, known={"B1"})
    agent = AnomalyAgent(ledger=ledger, bus=MessageBus())
    assert agent.decide(
        {"sim_day": 4, "shipments": [shipment(presented_fingerprint="a" * 64)]}
    ) == []


def test_a_batch_the_ledger_has_never_seen_is_condemned() -> None:
    ledger = FakeLedger(genuine={}, known=set())
    agent = AnomalyAgent(ledger=ledger, bus=MessageBus())
    actions = agent.decide({"sim_day": 4, "shipments": [shipment()]})
    assert actions[0].action_type == "COUNTERFEIT_ALERT"
    assert "no provenance record" in actions[0].params["reason"]


def test_a_counterfeit_alert_reaches_the_bus() -> None:
    """The Expiry Agent quarantines on this; the dashboard renders it."""
    received = []
    ledger = FakeLedger(genuine={}, known=set())
    agent = AnomalyAgent(ledger=ledger, bus=MessageBus())
    agent.bus.subscribe(Topic.COUNTERFEIT_FLAG, received.append)

    agent.step({"sim_day": 4}, world=None, sim_day=4)
    agent.recent.append(shipment())
    agent.step({"sim_day": 4}, world=None, sim_day=4)

    assert received and received[0].payload["batch_id"] == "B1"


def test_without_a_ledger_the_agent_degrades_to_ml_only() -> None:
    agent = AnomalyAgent(ledger=None, bus=MessageBus())
    assert agent.decide({"sim_day": 1, "shipments": [shipment()]}) == []


def test_no_shipments_means_no_alerts() -> None:
    agent = AnomalyAgent(ledger=FakeLedger({}, set()), bus=MessageBus())
    assert agent.decide({"sim_day": 1, "shipments": []}) == []
