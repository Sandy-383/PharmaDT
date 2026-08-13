"""Dataset acquisition and preprocessing.

One function per dataset, each returning a tidy DataFrame; a matching writer
saves Parquet into ``data/processed/``. Notebooks are for exploration, this is
the pipeline — nothing here should need a human in the loop.

Usage::

    python -m pharmadt.ml.preprocessing --all
    python -m pharmadt.ml.preprocessing --datasets rossmann openfda
    python -m pharmadt.ml.preprocessing --all --skip-download

Availability, as measured rather than assumed:

============  ==========================================================
Rossmann      Kaggle competition. Needs a token *and* accepted rules.
Supply chain  Kaggle public dataset. Token only.
openFDA       Public, no auth, but intermittently 502s — hence the retry.
CVRPLIB       Served from the PyVRP instance mirror; the canonical
              puc-rio host 404s on the documented instance paths.
CMS Part D    data.cms.gov returns 403 to scripted clients and geo-restricts
              some traffic, so the same CMS release is retrieved from a
              Kaggle mirror. The report cites CMS as the source and the
              mirror as the retrieval route.
USAID         No stable public endpoint was found. Cold-chain excursion
              rates stay at the literature default in config.py, and the
              report records that as a limitation rather than pretending
              otherwise.
============  ==========================================================
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import time
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pharmadt.config import settings

logger = logging.getLogger(__name__)

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

KAGGLE_API = "https://www.kaggle.com/api/v1"
OPENFDA_ENFORCEMENT = "https://api.fda.gov/drug/enforcement.json"
CVRPLIB_MIRROR = "https://raw.githubusercontent.com/PyVRP/Instances/main/CVRP/"
USER_AGENT = "PharmaDT-research/0.1"

#: Rossmann runs 2013-01-01 to 2015-07-31. Split by time, never at random:
#: a random split lets tomorrow's sales inform yesterday's prediction, and the
#: resulting validation score is a measurement of the leak.
TRAIN_END = pd.Timestamp("2015-01-31")
VAL_END = pd.Timestamp("2015-04-30")

#: CVRPLIB instances carrying a published optimum, for the Stage 9 benchmark.
CVRPLIB_INSTANCES = ("X-n101-k25", "X-n106-k14", "X-n110-k13", "X-n115-k10")


class DatasetUnavailable(Exception):
    """Raised when a source cannot be reached and has no local fallback."""


# ── HTTP ──────────────────────────────────────────────────────────────


def _get(url: str, *, retries: int = 4, backoff: float = 2.0, **kwargs) -> requests.Response:
    """GET with exponential backoff.

    openFDA returns a 502 for a few seconds at a time with no pattern, so a
    single attempt makes the pipeline fail for reasons that have nothing to do
    with the pipeline.
    """
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    last: Exception | None = None

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=120, **kwargs)
            if response.status_code < 500:
                return response
            last = DatasetUnavailable(f"{url} returned {response.status_code}")
        except requests.RequestException as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(backoff * (2**attempt))

    raise DatasetUnavailable(f"{url} unreachable after {retries} attempts") from last


def _kaggle_headers() -> dict[str, str]:
    token = settings.kaggle_api_token
    if token is None:
        raise DatasetUnavailable(
            "KAGGLE_API_TOKEN is not set. Add it to .env; see the README."
        )
    return {"Authorization": f"Bearer {token.get_secret_value()}"}


def _download_zip(url: str, destination: Path, *, headers: dict[str, str] | None = None) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    response = _get(url, headers=headers or {})

    if response.status_code == 403:
        raise DatasetUnavailable(
            f"{url} returned 403. For a Kaggle competition this means the rules "
            "have not been accepted for this account yet."
        )
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(destination)
    return destination


def _write(frame: pd.DataFrame, name: str) -> Path:
    """Write a tidy frame to Parquet. 10x faster to reload than CSV."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    logger.info("wrote %s (%d rows)", path, len(frame))
    return path


# ── 1. Rossmann store sales ───────────────────────────────────────────


def download_rossmann() -> Path:
    destination = RAW / "rossmann"
    if (destination / "train.csv").exists():
        return destination
    return _download_zip(
        f"{KAGGLE_API}/competitions/data/download-all/rossmann-store-sales",
        destination,
        headers=_kaggle_headers(),
    )


def load_rossmann() -> pd.DataFrame:
    """Tidy daily store sales with a time-ordered train/val/test split.

    Closed days are dropped: a zero on a day the shop never opened is not
    demand, and leaving them in drags every fitted mean toward zero.
    """
    source = RAW / "rossmann"
    if not (source / "train.csv").exists():
        raise DatasetUnavailable(f"{source}/train.csv missing. Run with --all first.")

    frame = pd.read_csv(source / "train.csv", parse_dates=["Date"], low_memory=False)
    stores = pd.read_csv(source / "store.csv", low_memory=False)

    frame = frame[frame["Open"] == 1]
    # A handful of rows report a store open with zero takings. Those are data
    # errors rather than a day of no demand, so they go too.
    frame = frame[frame["Sales"] > 0]

    frame = frame.merge(stores, on="Store", how="left")
    frame = frame.rename(
        columns={
            "Store": "store_id",
            "Date": "date",
            "Sales": "sales",
            "Customers": "customers",
            "Promo": "promo",
            "StateHoliday": "state_holiday",
            "SchoolHoliday": "school_holiday",
            "StoreType": "store_type",
            "Assortment": "assortment",
        }
    )

    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["is_weekend"] = frame["day_of_week"] >= 5
    frame["day_of_year"] = frame["date"].dt.dayofyear
    frame["month"] = frame["date"].dt.month
    # log1p compresses the long right tail so the LSTM in Stage 6 optimises
    # relative rather than absolute error.
    frame["log_sales"] = np.log1p(frame["sales"])

    frame["split"] = np.select(
        [frame["date"] <= TRAIN_END, frame["date"] <= VAL_END],
        ["train", "val"],
        default="test",
    )

    columns = [
        "store_id", "date", "sales", "log_sales", "customers", "promo",
        "state_holiday", "school_holiday", "store_type", "assortment",
        "day_of_week", "is_weekend", "day_of_year", "month", "split",
    ]
    return frame[columns].sort_values(["store_id", "date"]).reset_index(drop=True)


def preprocess_rossmann() -> Path:
    return _write(load_rossmann(), "rossmann_sales")


# ── 2. Demand profiles fitted from Rossmann ───────────────────────────


def _fit_one_series(sales: pd.Series, day_of_week: pd.Series, day_of_year: pd.Series) -> dict:
    """Extract the *shape* of a demand series: variability, weekday, season."""
    mean = float(sales.mean())
    dispersion = float(sales.std() / mean) if mean > 0 else 0.0

    weekend = sales[day_of_week >= 5]
    weekday = sales[day_of_week < 5]
    weekend_factor = (
        float(weekend.mean() / weekday.mean())
        if len(weekend) and len(weekday) and weekday.mean() > 0
        else 1.0
    )

    # Least-squares fit of a * sin(2*pi*t/365) + b * cos(...) gives the annual
    # amplitude as hypot(a, b), which is the seasonal term the twin applies.
    angle = 2 * np.pi * day_of_year.to_numpy() / 365.0
    design = np.column_stack([np.sin(angle), np.cos(angle), np.ones(len(angle))])
    coefficients, *_ = np.linalg.lstsq(design, sales.to_numpy(dtype=float), rcond=None)
    amplitude = float(np.hypot(coefficients[0], coefficients[1]) / mean) if mean > 0 else 0.0

    return {
        "dispersion": round(dispersion, 4),
        "weekend_factor": round(weekend_factor, 4),
        "seasonal_amplitude": round(min(amplitude, 0.9), 4),
    }


def fit_demand_profiles(node_ids: Iterable[str] | None = None,
                        drug_ids: Iterable[str] | None = None) -> pd.DataFrame:
    """Fit one demand profile per (retail node, drug) from Rossmann stores.

    Only the *shape* is taken from Rossmann — variability, weekday effect,
    seasonality. The level is rescaled to ``base_daily_demand``.

    That is deliberate and it is the mitigation for the validity threat the
    guide flags: Rossmann is retail store takings, not pharmaceutical unit
    demand, so its absolute magnitude means nothing here. Its temporal
    structure is what a demand model needs to learn, and that transfers.
    """
    from pharmadt.core.models import NodeType

    if node_ids is None or drug_ids is None:
        from sqlalchemy import select

        from pharmadt.core.db import session_scope
        from pharmadt.core.models import Drug, Node

        with session_scope() as session:
            if node_ids is None:
                node_ids = list(
                    session.scalars(
                        select(Node.node_id)
                        .where(Node.node_type.in_([NodeType.PHARMACY, NodeType.HOSPITAL]))
                        .order_by(Node.node_id)
                    )
                )
            if drug_ids is None:
                drug_ids = list(session.scalars(select(Drug.drug_id).order_by(Drug.drug_id)))

    node_ids, drug_ids = list(node_ids), list(drug_ids)

    # CMS relative dispensing volumes are attached for *validation*, not
    # applied to the demand level.
    #
    # They belong in the report as evidence that the five synthetic drugs are
    # real products with known volumes, and as a stated limitation: real demand
    # spans roughly tenfold across them while this twin spans about twofold.
    # Multiplying the means by these weights was tried and is a substantive
    # model change, not a validation — it shifts every KPI, and re-tuning the
    # inventory policy around a tenfold spread is work this project has not
    # done. Recording the gap is honest; quietly rescaling into an untuned
    # regime would not be.
    try:
        weights = cms_drug_weights().set_index("drug_id")["weight"].to_dict()
    except DatasetUnavailable:
        logger.warning("CMS weights unavailable; validation column omitted")
        weights = {}

    sales = load_rossmann()
    # Fit on training data only. Touching val or test here would leak straight
    # into the simulation the models are later scored on.
    sales = sales[sales["split"] == "train"]

    # Deterministic store assignment: same seed, same pairing, every run.
    rng = np.random.default_rng(settings.sim_seed)
    available = np.sort(sales["store_id"].unique())
    needed = len(node_ids) * len(drug_ids)
    chosen = rng.choice(available, size=needed, replace=needed > len(available))

    rows = []
    for index, (node_id, drug_id) in enumerate(
        (n, d) for n in node_ids for d in drug_ids
    ):
        series = sales[sales["store_id"] == chosen[index]]
        fitted = _fit_one_series(series["sales"], series["day_of_week"], series["day_of_year"])
        # Level comes from config and is perturbed per series so the network
        # has uneven pressure; without that, redistribution has nothing to do.
        scale = float(rng.uniform(0.6, 1.4))
        cms_weight = float(weights.get(drug_id, 1.0))
        rows.append(
            {
                "node_id": node_id,
                "drug_id": drug_id,
                "source_store_id": int(chosen[index]),
                "mean": round(
                    settings.base_daily_demand
                    * scale
                    * (cms_weight if settings.calibrate_drug_mix else 1.0),
                    4,
                ),
                "cms_weight": round(cms_weight, 4),
                **fitted,
                "observations": len(series),
            }
        )

    return pd.DataFrame(rows)


def preprocess_demand_profiles() -> Path:
    return _write(fit_demand_profiles(), "demand_profiles")


# ── 3. openFDA drug recalls ───────────────────────────────────────────


def download_openfda(limit: int = 5000) -> Path:
    """Page through drug enforcement reports into one JSON file."""
    destination = RAW / "openfda"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "enforcement.json"
    if path.exists():
        return path

    collected: list[dict] = []
    page = 100  # openFDA caps a page at 100 without a paid key.
    for skip in range(0, limit, page):
        response = _get(
            OPENFDA_ENFORCEMENT,
            params={"search": 'status:"Terminated" OR status:"Ongoing"',
                    "limit": page, "skip": skip},
        )
        if response.status_code == 404:
            break  # openFDA answers an exhausted result set with a 404.
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            break
        collected.extend(results)

    path.write_text(json.dumps(collected), encoding="utf-8")
    return path


def load_openfda() -> pd.DataFrame:
    """Recall records with a binary anomaly label for the Stage 10 agent."""
    path = RAW / "openfda" / "enforcement.json"
    if not path.exists():
        raise DatasetUnavailable(f"{path} missing. Run with --all first.")

    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    if frame.empty:
        raise DatasetUnavailable("openFDA returned no recall records")

    keep = [
        c for c in (
            "recall_number", "reason_for_recall", "classification", "status",
            "product_description", "recalling_firm", "distribution_pattern",
            "report_date", "voluntary_mandated", "product_quantity",
        ) if c in frame.columns
    ]
    frame = frame[keep].copy()
    frame["report_date"] = pd.to_datetime(frame["report_date"], format="%Y%m%d", errors="coerce")

    # Class I means "reasonable probability of serious harm or death" — the
    # positive class the anomaly detector is asked to catch.
    frame["is_severe"] = (frame.get("classification") == "Class I").astype(int)

    reason = frame.get("reason_for_recall", pd.Series(dtype=str)).fillna("").str.lower()
    frame["is_contamination"] = reason.str.contains(
        "contamin|steril|microb|particulate", regex=True
    ).astype(int)
    frame["is_cold_chain"] = reason.str.contains(
        "temperature|cold chain|refrigerat|storage condition", regex=True
    ).astype(int)
    frame["is_mislabelled"] = reason.str.contains(
        "label|mislabel|wrong|incorrect", regex=True
    ).astype(int)
    return frame.reset_index(drop=True)


def preprocess_openfda() -> Path:
    return _write(load_openfda(), "openfda_recalls")


# ── 4. CVRPLIB routing benchmark ──────────────────────────────────────


def download_cvrplib(instances: Iterable[str] = CVRPLIB_INSTANCES) -> Path:
    destination = RAW / "cvrplib"
    destination.mkdir(parents=True, exist_ok=True)
    for name in instances:
        for suffix in (".vrp", ".sol"):
            path = destination / f"{name}{suffix}"
            if path.exists():
                continue
            response = _get(f"{CVRPLIB_MIRROR}{name}{suffix}")
            if response.status_code == 200:
                path.write_text(response.text, encoding="utf-8")
    return destination


def parse_vrp(text: str) -> dict:
    """Parse a CVRPLIB ``.vrp`` instance into coordinates, demands, capacity."""
    header: dict[str, str] = {}
    coords: list[tuple[int, float, float]] = []
    demands: list[tuple[int, int]] = []
    section = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "EOF":
            continue
        if line.startswith(("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION")):
            section = line.split("_")[0]
            continue
        if ":" in line and section is None:
            key, _, value = line.partition(":")
            header[key.strip()] = value.strip()
            continue
        parts = line.split()
        if section == "NODE" and len(parts) >= 3:
            coords.append((int(parts[0]), float(parts[1]), float(parts[2])))
        elif section == "DEMAND" and len(parts) >= 2:
            demands.append((int(parts[0]), int(parts[1])))

    return {
        "name": header.get("NAME", ""),
        "capacity": int(header.get("CAPACITY", 0)),
        "dimension": int(header.get("DIMENSION", len(coords))),
        "coords": coords,
        "demands": dict(demands),
    }


def parse_sol(text: str) -> float | None:
    """Published optimum cost from a ``.sol`` file, if it states one."""
    for line in reversed(text.splitlines()):
        if line.lower().startswith("cost"):
            return float(line.split()[-1])
    return None


def load_cvrplib() -> pd.DataFrame:
    """One row per benchmark instance, carrying its known optimum.

    Stage 9 reports the gap between the OR-Tools solution and this column;
    without a published optimum a routing result has nothing to be good against.
    """
    source = RAW / "cvrplib"
    if not source.exists():
        raise DatasetUnavailable(f"{source} missing. Run with --all first.")

    rows = []
    for vrp_path in sorted(source.glob("*.vrp")):
        instance = parse_vrp(vrp_path.read_text(encoding="utf-8"))
        sol_path = vrp_path.with_suffix(".sol")
        optimum = parse_sol(sol_path.read_text(encoding="utf-8")) if sol_path.exists() else None
        rows.append(
            {
                "instance": instance["name"] or vrp_path.stem,
                "dimension": instance["dimension"],
                "capacity": instance["capacity"],
                "total_demand": sum(instance["demands"].values()),
                "known_optimum": optimum,
            }
        )

    if not rows:
        raise DatasetUnavailable("no CVRPLIB instances were downloaded")
    return pd.DataFrame(rows)


def preprocess_cvrplib() -> Path:
    return _write(load_cvrplib(), "cvrplib_benchmarks")


# ── 5. Kaggle supply-chain priors ─────────────────────────────────────


def download_supply_chain() -> Path:
    destination = RAW / "supply_chain"
    if any(destination.glob("*.csv")):
        return destination
    return _download_zip(
        f"{KAGGLE_API}/datasets/download/harshsingh2209/supply-chain-analysis",
        destination,
        headers=_kaggle_headers(),
    )


def load_supply_chain() -> pd.DataFrame:
    """Lead times, shipping times, and defect rates as distribution priors."""
    source = RAW / "supply_chain"
    files = sorted(source.glob("*.csv"))
    if not files:
        raise DatasetUnavailable(f"{source} has no CSV. Run with --all first.")

    frame = pd.read_csv(files[0])
    frame.columns = [c.strip().lower().replace(" ", "_") for c in frame.columns]
    return frame


def supply_chain_priors() -> pd.DataFrame:
    """Collapse the raw table into the handful of parameters the twin needs."""
    frame = load_supply_chain()
    rows = []
    for column, parameter in (
        ("lead_times", "supplier_lead_time_days"),
        ("lead_time", "order_lead_time_days"),
        ("manufacturing_lead_time", "manufacturing_lead_time_days"),
        ("shipping_times", "shipping_time_days"),
        ("defect_rates", "defect_rate_pct"),
    ):
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append(
            {
                "parameter": parameter,
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "p05": round(float(series.quantile(0.05)), 4),
                "p50": round(float(series.median()), 4),
                "p95": round(float(series.quantile(0.95)), 4),
                "observations": int(series.size),
            }
        )
    return pd.DataFrame(rows)


def preprocess_supply_chain() -> Path:
    _write(load_supply_chain(), "supply_chain_raw")
    return _write(supply_chain_priors(), "supply_chain_priors")


# ── 6. CMS Medicare Part D ────────────────────────────────────────────


#: Kaggle mirror of the CMS Part D drug spending/utilisation extract.
#: data.cms.gov answers scripted clients with 403 and geo-restricts some
#: traffic, so the primary portal cannot be automated from here. The mirror
#: carries the same CMS release; the report cites CMS as the source and this
#: as the retrieval route.
CMS_KAGGLE_REF = "fabiovillagran/medicare-part-d-drug-spendingutilization-201519"

#: Columns that identify the utilisation table among the several CSVs shipped
#: in the archive. Selecting by content rather than by filename: the archive
#: also contains a data dictionary and a drug-uses table, and picking the
#: first file alphabetically would load the dictionary.
CMS_REQUIRED = {"generic_name", "year", "total_claims"}


def download_cms() -> Path:
    destination = RAW / "cms"
    if destination.exists() and any(destination.glob("*.csv")):
        return destination
    return _download_zip(
        f"{KAGGLE_API}/datasets/download/{CMS_KAGGLE_REF}",
        destination,
        headers=_kaggle_headers(),
    )


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        c.strip().lower().replace("\n", " ").replace("  ", " ").replace(" ", "_")
        for c in frame.columns
    ]
    return frame


def load_cms() -> pd.DataFrame:
    """Drug-level utilisation by year: claims, dosage units, beneficiaries.

    This is the counterweight to Rossmann's validity threat. Retail takings are
    not drug demand, and CMS is where that claim gets checked against real
    dispensing volumes.
    """
    source = RAW / "cms"
    files = sorted(source.rglob("*.csv")) if source.exists() else []
    if not files:
        raise DatasetUnavailable(
            f"No CMS extract in {source}/. Run `make data`, or download a Part D "
            "utilisation CSV by hand if the Kaggle mirror is unavailable."
        )

    for path in files:
        frame = _normalise(pd.read_csv(path, low_memory=False))
        if not set(frame.columns) >= CMS_REQUIRED:
            continue

        # CMS ships these as thousands-separated strings, so they arrive as
        # object dtype and every arithmetic comparison against them silently
        # becomes a string comparison.
        for column in (
            "total_claims", "total_dosage_units", "total_beneficiaries", "total_spending"
        ):
            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column].astype(str).str.replace(r"[,$]", "", regex=True),
                    errors="coerce",
                )

        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
        frame["generic_name"] = frame["generic_name"].astype(str).str.strip()
        return frame.dropna(subset=["generic_name", "year"])

    raise DatasetUnavailable(
        f"None of the {len(files)} CSV(s) in {source}/ carry drug-level "
        f"utilisation columns {sorted(CMS_REQUIRED)}."
    )




# ── CMS validation of the Rossmann-derived demand mix ─────────────────

#: Our synthetic drugs mapped to the generic names CMS reports them under.
#: Influenza Vaccine is deliberately absent: vaccines are reimbursed under
#: Medicare Part B, not Part D, so a zero here is a fact about the programme
#: rather than a missing match, and silently mapping it to something else
#: would fabricate a validation that did not happen.
CMS_DRUG_MAP: dict[str, str] = {
    "DRUG-001": "Amoxicillin",
    "DRUG-002": "Insulin Glargine",
    "DRUG-003": "Metformin",
    "DRUG-005": "Morphine Sulfate",
}
CMS_UNMAPPED_NOTE = {
    "DRUG-004": "Influenza Vaccine is a Part B benefit; Part D records no claims.",
}


def cms_drug_weights() -> pd.DataFrame:
    """Relative dispensing volume per drug, from real Part D claims.

    Normalised to mean 1.0 so the weights rescale the *mix* between drugs
    without touching the overall demand level, which comes from the twin's own
    ``base_daily_demand``. Drugs CMS does not cover keep a weight of 1.0 rather
    than being dropped — an unmatched drug is not a zero-demand drug.
    """
    frame = load_cms()
    latest = frame[frame["year"] == frame["year"].max()]

    rows = []
    for drug_id, generic in CMS_DRUG_MAP.items():
        matched = latest[
            latest["generic_name"].str.contains(generic.split()[0], case=False, na=False)
        ]
        rows.append(
            {
                "drug_id": drug_id,
                "cms_generic": generic,
                "cms_rows": len(matched),
                "total_claims": float(matched["total_claims"].sum()),
                "total_beneficiaries": float(
                    matched.get("total_beneficiaries", pd.Series(dtype=float)).sum()
                ),
            }
        )
    for drug_id, note in CMS_UNMAPPED_NOTE.items():
        rows.append({"drug_id": drug_id, "cms_generic": None, "cms_rows": 0,
                     "total_claims": float("nan"), "total_beneficiaries": float("nan"),
                     "note": note})

    result = pd.DataFrame(rows)
    matched_claims = result["total_claims"].dropna()
    mean_claims = matched_claims.mean() if len(matched_claims) else 1.0
    result["weight"] = (result["total_claims"] / mean_claims).fillna(1.0).round(4)
    result["year"] = int(frame["year"].max())
    return result.sort_values("drug_id").reset_index(drop=True)


def preprocess_cms() -> Path:
    _write(cms_drug_weights(), "cms_drug_weights")
    return _write(load_cms(), "cms_part_d")


# ── Registry and CLI ──────────────────────────────────────────────────

DATASETS: dict[str, tuple] = {
    "rossmann": (download_rossmann, preprocess_rossmann),
    "demand_profiles": (None, preprocess_demand_profiles),
    "openfda": (download_openfda, preprocess_openfda),
    "cvrplib": (download_cvrplib, preprocess_cvrplib),
    "supply_chain": (download_supply_chain, preprocess_supply_chain),
    "cms": (download_cms, preprocess_cms),
}

#: Sources whose absence is reported rather than fatal.
OPTIONAL = {"cms"}


def run(names: Iterable[str], *, skip_download: bool = False) -> dict[str, str]:
    results: dict[str, str] = {}
    for name in names:
        download, preprocess = DATASETS[name]
        try:
            if download is not None and not skip_download:
                download()
            results[name] = str(preprocess())
        except DatasetUnavailable as exc:
            if name not in OPTIONAL:
                raise
            results[name] = f"SKIPPED: {exc}"
            logger.warning("%s skipped: %s", name, exc)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the PharmaDT datasets.")
    parser.add_argument("--all", action="store_true", help="process every dataset")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=[])
    parser.add_argument("--skip-download", action="store_true", help="use data/raw as-is")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    names = list(DATASETS) if args.all else args.datasets
    if not names:
        parser.error("pass --all or --datasets")

    results = run(names, skip_download=args.skip_download)

    print()
    for name, outcome in results.items():
        marker = "!" if outcome.startswith("SKIPPED") else "+"
        print(f"  {marker} {name:<18} {outcome}")


if __name__ == "__main__":
    main()
