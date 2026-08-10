# PharmaDT build targets.
#
# `make sim` must reproduce a full run from a clean checkout — that is an
# explicit condition of the Stage 10.5 integration gate, so keep it working.

# `data` and `eda` MUST stay in .PHONY: a directory named data/ exists, so
# without it make considers the target already satisfied and silently does
# nothing at all.
.PHONY: help install db-up db-down db-logs migrate migration seed reseed \
        data eda ablation frontier routing-benchmark anomaly-eval keys sim-anchor verify-chain tamper-demo \
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
