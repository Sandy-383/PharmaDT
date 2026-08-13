"""Check a machine is ready to run PharmaDT, and say what to run if it is not.

The database is state on a machine, not content in the repository. A clean
clone has working code and an empty world, and every symptom of that looks like
a code failure — a stack trace from psycopg2, an empty KPI table, a ledger with
nothing in it. This turns those into a checklist with the fix beside each line.

Usage::

    make doctor
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

#: Datasets `make data` fetches. CMS is optional: it validates the demand mix
#: but nothing depends on it to run.
DATASETS = ("rossmann", "openfda", "cvrplib", "supply_chain")
OPTIONAL_DATASETS = ("cms",)


@dataclass(slots=True)
class Result:
    name: str
    passed: bool
    detail: str
    fix: str = ""
    optional: bool = False

    def render(self, width: int) -> str:
        mark = "ok " if self.passed else ("warn" if self.optional else "FAIL")
        line = f"  [{mark}] {self.name:<{width}} {self.detail}"
        if not self.passed and self.fix:
            line += f"\n         -> {self.fix}"
        return line


def _docker() -> list[Result]:
    try:
        done = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=30
        )
        running = done.returncode == 0
    except (OSError, subprocess.SubprocessError):
        running = False

    results = [
        Result(
            "docker daemon",
            running,
            "reachable" if running else "not reachable",
            "start Docker Desktop and wait for the whale icon to settle",
        )
    ]
    if not running:
        return results

    try:
        done = subprocess.run(
            ["docker", "compose", "ps", "--format", "{{.Name}} {{.Status}}"],
            capture_output=True, text=True, timeout=30,
        )
        line = done.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        line = ""

    healthy = "healthy" in line.lower()
    results.append(
        Result(
            "postgres container",
            healthy,
            line or "not running",
            "make db-up",
        )
    )
    return results


def _database() -> list[Result]:
    from sqlalchemy import inspect, text

    from pharmadt.core.db import engine

    try:
        with engine.connect() as conn:
            version = conn.execute(text("SHOW server_version")).scalar_one()
    except Exception as exc:  # noqa: BLE001 - the message is the diagnosis
        return [
            Result(
                "database connection",
                False,
                f"{type(exc).__name__}",
                "make db-up  (and check DATABASE_URL in .env)",
            )
        ]

    results = [
        Result("postgres version", version.startswith("15"), version.split()[0],
               "docker compose down -v && make db-up  (Stage 4 needs 15)")
    ]

    tables = set(inspect(engine).get_table_names())
    required = {
        "drugs", "nodes", "batches", "inventory_records", "shipments",
        "demand_records", "agent_decisions", "provenance_records",
    }
    missing = sorted(required - tables)
    results.append(
        Result(
            "schema",
            not missing,
            f"{len(required & tables)}/{len(required)} tables"
            + (f", missing {', '.join(missing)}" if missing else ""),
            "make migrate",
        )
    )

    if "provenance_records" in tables:
        with engine.connect() as conn:
            triggers = set(
                conn.scalars(
                    text(
                        "SELECT tgname FROM pg_trigger "
                        "WHERE tgrelid = 'provenance_records'::regclass "
                        "AND NOT tgisinternal"
                    )
                )
            )
        wanted = {"no_update", "no_delete", "no_truncate"}
        results.append(
            Result(
                "append-only triggers",
                wanted <= triggers,
                f"{len(wanted & triggers)}/3 present",
                "make migrate  (the ledger is not tamper-proof without these)",
            )
        )

    if not missing:
        with engine.connect() as conn:
            count = lambda t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar_one()  # noqa: E731
            nodes, drugs, batches = count("nodes"), count("drugs"), count("batches")
            keyed = conn.execute(
                text("SELECT count(*) FROM nodes WHERE public_key IS NOT NULL")
            ).scalar_one()
            ledger = count("provenance_records")

        results += [
            Result("seed fixture", nodes >= 12 and drugs >= 5 and batches >= 20,
                   f"{nodes} nodes, {drugs} drugs, {batches} batches", "make seed"),
            Result("node signing keys", keyed >= 12 and keyed == nodes,
                   f"{keyed}/{nodes} nodes hold a keypair", "make keys"),
            Result("ledger populated", ledger > 0, f"{ledger:,} records",
                   "make sim-anchor  (needed for tamper-demo and the gate)"),
        ]
    return results


def _files() -> list[Result]:
    results = []

    env = Path(".env")
    has_token = env.exists() and "KAGGLE_API_TOKEN" in env.read_text(encoding="utf-8")
    results.append(
        Result(".env", env.exists(), "present" if env.exists() else "missing",
               "cp .env.example .env")
    )
    results.append(
        Result(
            "kaggle token",
            has_token,
            "set" if has_token else "not set",
            "add KAGGLE_API_TOKEN=... to .env, and accept the Rossmann "
            "competition rules on that Kaggle account",
        )
    )

    keys = Path("data/keys")
    n_keys = len(list(keys.glob("*.pem"))) if keys.is_dir() else 0
    results.append(
        Result("private keys on disk", n_keys >= 12, f"{n_keys} .pem file(s)", "make keys")
    )

    for name in DATASETS:
        path = Path("data/raw") / name
        present = path.is_dir() and any(path.iterdir())
        results.append(Result(f"dataset {name}", present,
                              "present" if present else "missing", "make data"))
    for name in OPTIONAL_DATASETS:
        path = Path("data/raw") / name
        present = path.is_dir() and any(path.iterdir())
        results.append(
            Result(f"dataset {name}", present,
                   "present" if present else "missing (validation only)",
                   "make data", optional=True)
        )
    return results


def run_checks() -> list[Result]:
    results = _docker()
    if any(r.name == "postgres container" and r.passed for r in results):
        results += _database()
    return results + _files()


def main() -> None:
    print("\nPharmaDT readiness check\n" + "=" * 60)
    results = run_checks()
    width = max(len(r.name) for r in results)

    for result in results:
        print(result.render(width))

    failed = [r for r in results if not r.passed and not r.optional]
    warned = [r for r in results if not r.passed and r.optional]

    print("=" * 60)
    if not failed:
        print(f"  Ready. {len(results) - len(warned)}/{len(results)} checks passed.")
        if warned:
            print(f"  {len(warned)} optional item(s) absent; nothing depends on them to run.")
    else:
        print(f"  {len(failed)} check(s) need attention. Run the fixes above in order.")
    print()
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
