# PharmaDT build targets.
#
# `make sim` must reproduce a full run from a clean checkout — that is an
# explicit condition of the Stage 10.5 integration gate, so keep it working.

# `data` and `eda` MUST stay in .PHONY: a directory named data/ exists, so
# without it make considers the target already satisfied and silently does
# nothing at all.
.PHONY: help install db-up db-down db-logs migrate migration seed reseed \
        data eda ablation frontier routing-benchmark anomaly-eval gate federated crisis evaluate doctor demo-reset keys sim-anchor verify-chain tamper-demo \
        test cov sim api lint fmt clean

help:
	@echo "PharmaDT targets:"
	@echo "  install   Install pinned Python dependencies"
	@echo "  db-up     Start Postgres 15 and wait until it accepts connections"
	@echo "  db-down   Stop the database"
	@echo "  db-logs   Follow database logs"
	@echo "  migrate   Apply all pending Alembic migrations"
	@echo "  migration Autogenerate a migration:  make migration m=\"add x\""
	@echo "  seed      Insert the 12-node / 5-drug / 20-batch fixture"
	@echo "  reseed    Wipe and re-insert the fixture"
	@echo ""
	@echo "  data         Download and preprocess every dataset (Stage 2)"
	@echo "  eda          Re-execute the EDA notebook"
	@echo ""
	@echo "  keys         Issue per-node ECDSA signing keys"
	@echo "  sim-anchor   Run the twin and append custody events to the ledger"
	@echo "  verify-chain Verify the provenance chain end to end"
	@echo "  tamper-demo  Demonstrate tamper-evidence (Stage 4 DoD)"
	@echo ""
	@echo "  doctor       Check this machine is ready, with the fix for each failure"
	@echo "  demo-reset   Rebuild a clean world so the live demo runs fast"
	@echo "  gate         STAGE 10.5 GATE: the whole system, 6 conditions, one run"
	@echo "  evaluate     Experiment matrix, 10 seeds, mean +/- std (Stage 15)"
	@echo "  crisis       Four disruption scenarios, baseline vs agents"
	@echo "  federated    Centralised vs IID vs non-IID vs differential privacy"
	@echo "  ablation     Compare policies on identical seeds"
	@echo "  frontier     Sweep the safety-stock quantile z"
	@echo "  anomaly-eval Detection metrics: ML alone vs ML + ledger"
	@echo "  routing-benchmark  CVRPLIB gap against published optima"
	@echo ""
	@echo "  test      Run the test suite"
	@echo "  cov       Run tests with a coverage report"
	@echo "  sim       Run the digital twin simulation"
	@echo "  api       Serve the FastAPI backend with reload"
	@echo "  lint      Lint with ruff"
	@echo "  fmt       Format with ruff"
	@echo "  clean     Remove caches and build artifacts"

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

db-up:
	docker compose up -d --wait db

db-down:
	docker compose down

db-logs:
	docker compose logs -f db

migrate:
	alembic upgrade head

# Usage: make migration m="add provenance_records"
migration:
	alembic revision --autogenerate -m "$(m)"

seed:
	python -m pharmadt.core.seed

reseed:
	python -m pharmadt.core.seed --reset

data:
	python -m pharmadt.ml.preprocessing --all

ablation:
	python -m pharmadt.ablation --seeds 42 43 44

frontier:
	python -m pharmadt.ablation --frontier --seeds 42 43 44

evaluate:
	python -m pharmadt.evaluation --seeds 10

crisis:
	python -m pharmadt.crisis.experiment

federated:
	python -m pharmadt.federated.experiment

doctor:
	python -m pharmadt.doctor

# Rebuild a clean world before a live demo. The ledger is append-only, so a
# machine that has run the simulation many times accumulates every record ever
# written -- and verify_chain walks all of them. After ~80k records the gate
# takes a minute, which is a long silence in front of an audience.
#
# Recipes are spelled out rather than delegated with $(MAKE): on Windows that
# expands to "C:/Program Files (x86)/GnuWin32/bin/make", and the spaces and
# parentheses break the shell before anything runs.
demo-reset:
	docker compose down -v
	docker compose up -d --wait db
	alembic upgrade head
	python -m pharmadt.core.seed
	python -m pharmadt.ledger.keyring
	python -m pharmadt.twin.simulation --anchor

gate:
	python -m pharmadt.gate

anomaly-eval:
	python -m pharmadt.ml.train_anomaly

routing-benchmark:
	python -m pharmadt.ml.benchmark_routing --time-limit 30

eda:
	python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_datasets.ipynb

keys:
	python -m pharmadt.ledger.keyring

sim-anchor:
	python -m pharmadt.twin.simulation --anchor

verify-chain:
	python -m pharmadt.ledger.verify

tamper-demo:
	python -m pharmadt.ledger.demo

test:
	python -m pytest

cov:
	python -m pytest --cov=pharmadt --cov-report=term-missing --cov-report=html

sim:
	python -m pharmadt.twin.simulation

api:
	python -m uvicorn pharmadt.api.main:app --reload

lint:
	python -m ruff check pharmadt tests

fmt:
	python -m ruff format pharmadt tests

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True); shutil.rmtree('htmlcov', ignore_errors=True)"
