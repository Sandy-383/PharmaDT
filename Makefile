# PharmaDT build targets.
#
# `make sim` must reproduce a full run from a clean checkout — that is an
# explicit condition of the Stage 10.5 integration gate, so keep it working.

.PHONY: help install db-up db-down db-logs migrate migration seed reseed test cov sim api lint fmt clean

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
