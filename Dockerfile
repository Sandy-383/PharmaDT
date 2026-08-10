# Image for the FastAPI backend (Stage 14) and for reproducible runs of the
# simulation (Stage 15 packaging: `docker compose up` from a clean checkout).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# build-essential is a fallback for any dependency without a manylinux wheel;
# in practice every pin in requirements.txt resolves to a prebuilt wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements alone first so the dependency layer caches across code edits.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY alembic.ini ./
COPY alembic/ ./alembic/
COPY data/scenarios/ ./data/scenarios/
COPY pharmadt/ ./pharmadt/

EXPOSE 8000

CMD ["uvicorn", "pharmadt.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
