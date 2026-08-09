"""Stage 2 pipeline: parsers, distribution fitting, and split hygiene.

The parser and fitting tests run anywhere. The tests that need real data skip
when it is absent, because ``data/raw/`` is gitignored — a fresh clone has no
datasets until someone runs the pipeline, and a suite that fails on checkout
teaches people to ignore red.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmadt.ml.preprocessing import (
    PROCESSED,
    RAW,
    DatasetUnavailable,
    _fit_one_series,
    fit_demand_profiles,
    load_cvrplib,
    load_openfda,
    load_rossmann,
    parse_sol,
    parse_vrp,
    supply_chain_priors,
)

VRP_SAMPLE = """NAME : X-n5-k2
COMMENT : (test instance)
TYPE : CVRP
DIMENSION : 5
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 100
NODE_COORD_SECTION
1 40 50
2 45 68
3 45 70
4 42 66
5 42 68
DEMAND_SECTION
1 0
2 10
3 30
4 16
5 9
DEPOT_SECTION
1
-1
EOF
"""

SOL_SAMPLE = """Route #1: 1 2 3
Route #2: 4
Cost 27591
"""

needs_rossmann = pytest.mark.skipif(
    not (RAW / "rossmann" / "train.csv").exists(),
    reason="run `python -m pharmadt.ml.preprocessing --all` first",
)


# ── CVRPLIB parsing ───────────────────────────────────────────────────


def test_vrp_header_and_sections_are_parsed() -> None:
    instance = parse_vrp(VRP_SAMPLE)
    assert instance["name"] == "X-n5-k2"
    assert instance["capacity"] == 100
    assert instance["dimension"] == 5
    assert len(instance["coords"]) == 5
    assert instance["demands"][3] == 30


def test_the_depot_carries_no_demand() -> None:
    assert parse_vrp(VRP_SAMPLE)["demands"][1] == 0


def test_coordinates_keep_their_node_ids() -> None:
    coords = parse_vrp(VRP_SAMPLE)["coords"]
    assert coords[0] == (1, 40.0, 50.0)
    assert [node for node, _, _ in coords] == [1, 2, 3, 4, 5]


def test_the_published_optimum_is_extracted() -> None:
    """Stage 9 reports its gap against this number."""
    assert parse_sol(SOL_SAMPLE) == 27591.0


def test_a_solution_without_a_cost_line_yields_none() -> None:
    assert parse_sol("Route #1: 1 2 3\n") is None


# ── Demand-shape fitting ──────────────────────────────────────────────


def _series(values: list[float]) -> tuple[pd.Series, pd.Series, pd.Series]:
    n = len(values)
    days = pd.Series(range(n))
    return pd.Series(values), days % 7, (days % 365) + 1


def test_a_flat_series_has_no_variability_and_no_season() -> None:
    sales, dow, doy = _series([100.0] * 364)
    fitted = _fit_one_series(sales, dow, doy)

    assert fitted["dispersion"] == pytest.approx(0.0, abs=1e-6)
    assert fitted["weekend_factor"] == pytest.approx(1.0, abs=1e-6)
    assert fitted["seasonal_amplitude"] == pytest.approx(0.0, abs=1e-3)


def test_a_weekend_dip_is_recovered() -> None:
    values = [50.0 if (d % 7) >= 5 else 100.0 for d in range(364)]
    fitted = _fit_one_series(*_series(values))
    assert fitted["weekend_factor"] == pytest.approx(0.5, abs=0.01)


def test_a_weekend_peak_is_recovered_too() -> None:
    """240 of 1115 Rossmann stores are busier at weekends; the fit must allow it."""
    values = [150.0 if (d % 7) >= 5 else 100.0 for d in range(364)]
    assert _fit_one_series(*_series(values))["weekend_factor"] == pytest.approx(1.5, abs=0.01)


def test_a_known_seasonal_amplitude_is_recovered() -> None:
    amplitude = 0.30
    values = [100.0 * (1 + amplitude * np.sin(2 * np.pi * d / 365)) for d in range(365)]
    fitted = _fit_one_series(*_series(values))
    assert fitted["seasonal_amplitude"] == pytest.approx(amplitude, abs=0.02)


def test_seasonal_amplitude_is_capped() -> None:
    """An unbounded amplitude could drive expected demand negative in the twin."""
    values = [100.0 * (1 + 5.0 * np.sin(2 * np.pi * d / 365)) for d in range(365)]
    assert _fit_one_series(*_series(values))["seasonal_amplitude"] <= 0.9


def test_a_zero_series_does_not_divide_by_zero() -> None:
    fitted = _fit_one_series(*_series([0.0] * 100))
    assert fitted["dispersion"] == 0.0
    assert fitted["seasonal_amplitude"] == 0.0


# ── Rossmann ──────────────────────────────────────────────────────────


@needs_rossmann
def test_closed_days_never_reach_the_pipeline() -> None:
    """A zero on a day the shop was shut is not demand."""
    sales = load_rossmann()
    assert (sales["sales"] > 0).all()


@needs_rossmann
def test_the_split_is_ordered_in_time_with_no_overlap() -> None:
    """A random split lets tomorrow inform yesterday; the score measures the leak."""
    sales = load_rossmann()
    bounds = sales.groupby("split")["date"].agg(["min", "max"])

    assert bounds.loc["train", "max"] < bounds.loc["val", "min"]
    assert bounds.loc["val", "max"] < bounds.loc["test", "min"]
    assert set(bounds.index) == {"train", "val", "test"}


@needs_rossmann
def test_every_split_is_non_empty() -> None:
    assert load_rossmann().groupby("split").size().min() > 0


@needs_rossmann
def test_log_sales_is_the_log1p_of_sales() -> None:
    sales = load_rossmann().head(1000)
    assert np.allclose(sales["log_sales"], np.log1p(sales["sales"]))


@needs_rossmann
def test_the_demand_series_has_no_nulls() -> None:
    sales = load_rossmann()
    assert sales[["store_id", "date", "sales", "day_of_week"]].isna().sum().sum() == 0


# ── Fitted demand profiles ────────────────────────────────────────────


@needs_rossmann
def test_one_profile_per_node_and_drug() -> None:
    nodes = ["NODE-PH-01", "NODE-PH-02"]
    drugs = ["DRUG-001", "DRUG-002", "DRUG-003"]
    profiles = fit_demand_profiles(nodes, drugs)

    assert len(profiles) == len(nodes) * len(drugs)
    assert set(profiles["node_id"]) == set(nodes)
    assert set(profiles["drug_id"]) == set(drugs)


@needs_rossmann
def test_fitting_is_reproducible() -> None:
    """The twin's determinism guarantee reaches back through the fit."""
    a = fit_demand_profiles(["NODE-PH-01"], ["DRUG-001", "DRUG-002"])
    b = fit_demand_profiles(["NODE-PH-01"], ["DRUG-001", "DRUG-002"])
    pd.testing.assert_frame_equal(a, b)


@needs_rossmann
def test_fitted_parameters_are_physically_sensible() -> None:
    profiles = fit_demand_profiles(["NODE-PH-01", "NODE-PH-02"], ["DRUG-001", "DRUG-002"])

    assert (profiles["mean"] > 0).all()
    assert (profiles["dispersion"] >= 0).all()
    assert (profiles["weekend_factor"] > 0).all()
    assert profiles["seasonal_amplitude"].between(0, 0.9).all()
    assert (profiles["observations"] > 0).all()


@needs_rossmann
def test_profiles_are_fitted_on_training_data_only() -> None:
    """Fitting on val or test would leak into the runs the models are scored on."""
    train_rows = (load_rossmann()["split"] == "train").sum()
    profiles = fit_demand_profiles(["NODE-PH-01"], ["DRUG-001"])
    assert profiles["observations"].iloc[0] <= train_rows


# ── Supporting datasets ───────────────────────────────────────────────


@pytest.mark.skipif(
    not any((RAW / "supply_chain").glob("*.csv")) if (RAW / "supply_chain").exists() else True,
    reason="supply-chain dataset not downloaded",
)
def test_supply_chain_priors_summarise_each_parameter() -> None:
    priors = supply_chain_priors()
    assert not priors.empty
    assert {"parameter", "mean", "std", "p05", "p50", "p95"} <= set(priors.columns)
    assert (priors["mean"] > 0).all()
    assert (priors["p05"] <= priors["p50"]).all()
    assert (priors["p50"] <= priors["p95"]).all()


@pytest.mark.skipif(
    not (RAW / "openfda" / "enforcement.json").exists(), reason="openFDA not downloaded"
)
def test_recall_labels_are_binary_and_present() -> None:
    recalls = load_openfda()
    for column in ("is_severe", "is_contamination", "is_cold_chain", "is_mislabelled"):
        assert set(recalls[column].unique()) <= {0, 1}
    assert recalls["is_severe"].sum() > 0, "no Class I recalls to learn from"


@pytest.mark.skipif(
    not (RAW / "openfda" / "enforcement.json").exists(), reason="openFDA not downloaded"
)
def test_the_severe_class_is_a_minority() -> None:
    """Stage 10 must score precision/recall, not accuracy: ~92% is the trivial score."""
    recalls = load_openfda()
    assert 0.0 < recalls["is_severe"].mean() < 0.5


@pytest.mark.skipif(
    not any((RAW / "cvrplib").glob("*.vrp")) if (RAW / "cvrplib").exists() else True,
    reason="CVRPLIB not downloaded",
)
def test_every_benchmark_instance_has_a_published_optimum() -> None:
    benchmarks = load_cvrplib()
    assert not benchmarks.empty
    assert benchmarks["known_optimum"].notna().all()
    assert (benchmarks["known_optimum"] > 0).all()
    assert (benchmarks["total_demand"] > 0).all()


# ── Failure modes ─────────────────────────────────────────────────────


def test_a_missing_dataset_says_how_to_fix_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("pharmadt.ml.preprocessing.RAW", tmp_path)
    with pytest.raises(DatasetUnavailable, match="--all"):
        load_rossmann()


@pytest.mark.skipif(not PROCESSED.exists(), reason="pipeline has not been run")
def test_parquet_outputs_reload() -> None:
    """Parquet over CSV is a 10x reload win during model iteration."""
    for name in ("rossmann_sales", "demand_profiles"):
        path = PROCESSED / f"{name}.parquet"
        if path.exists():
            assert not pd.read_parquet(path).empty
