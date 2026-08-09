# PharmaDT

An agentic AI framework integrating digital twin simulation, cryptographic
provenance, and federated learning for resilient drug supply chains.

**Team:** S Sanjana (1BM23AI159) · Sandeep N (1BM23AI169) · Sai Shreekar G (1BM23AI164)
**Guide:** Dr. Monika Puttaramaiah, Dept. of Machine Learning, BMSCE

---

## Quickstart

```bash
# 1. Environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
make install

# 2. Database
cp .env.example .env
make db-up                      # Postgres 15, waits until healthy
make migrate                    # create the schema
make seed                       # 12 nodes, 5 drugs, 20 batches

# 3. Verify
make test
make sim                        # 365 simulated days, prints baseline KPIs
```

`make help` lists every target.

## Architecture

Six layers, following the project's High-Level Design:

| Layer | Concern | Package |
|---|---|---|
| 2 | Digital twin — SimPy discrete-event simulation over a NetworkX network | `pharmadt/twin/` |
| 3 | Five autonomous agents — inventory, demand, expiry, routing, anomaly | `pharmadt/agents/` |
| 4 | Trust — Cryptographic Provenance Ledger | `pharmadt/ledger/` |
| 5 | Federated learning across simulated clients | `pharmadt/federated/` |
| 6 | FastAPI backend and React dashboard | `pharmadt/api/`, `dashboard/` |

Every component sits behind an abstract base class declared in
`pharmadt/core/interfaces.py` (`ProvenanceLedger`, `Agent`, `DemandModel`), so
a reader can trace a contract without reading any implementation.

### Cryptographic Provenance Ledger

The trust layer is a permissioned, append-only, hash-chained ledger rather
than a distributed blockchain. In a simulated single-consortium supply chain
there are no mutually distrusting, independently operated validator
organisations, so distributed Byzantine consensus is over-provisioned for the
threat model this project addresses. The ledger provides tamper-evidence via
SHA-256 hash chaining, non-repudiation and write-authorisation via per-node
ECDSA (NIST P-256) signatures, and efficient inclusion proofs via Merkle-root
anchoring. Immutability is enforced by a PostgreSQL trigger, so it holds even
against the application layer. Because access goes exclusively through the
`ProvenanceLedger` interface, the system stays forward-compatible with a
Hyperledger Fabric backend without changes to the agent layer.

## Repository layout

```
pharmadt/
├── core/        domain model, DTOs, abstract interfaces, DB session
├── twin/        SimPy simulation, network graph, node state vectors
├── ledger/      hash chain, ECDSA keyring, Merkle proofs, append-only schema
├── agents/      agent base class, message bus, the five agents
├── ml/          LSTM, Prophet, XGBoost, Isolation Forest, autoencoder
├── federated/   Flower client/server, non-IID partitioning, differential privacy
├── marl/        PettingZoo environment, MADDPG
├── crisis/      scenario definitions and event injection
└── api/         FastAPI routes and WebSocket event stream
```

## Environment notes

Verified on Windows 11, Python 3.12.10, Docker Desktop 29.4.3.

The implementation guide specifies Python 3.11 out of concern that Prophet
compiles against the installed NumPy ABI. That does not apply here: Prophet
1.1.5 ships a prebuilt `py3-none-win_amd64` wheel with bundled cmdstan, and
numpy 1.26.4 and torch 2.4.0 both publish cp312 wheels. Python 3.13 is not
supported — neither Prophet nor torch 2.4 publishes wheels for it.

Two pins in `requirements.txt` deviate from the guide because the printed list
cannot be resolved by pip on any Python version:

- `cryptography` 43.0.0 → **42.0.8**, because `flwr==1.10.0` requires
  `cryptography<43.0.0`. This also matches the guide's own software
  requirements table, which asks for cryptography 42.x.
- `ortools` 9.10.4067 → **9.9.3963**, because OR-Tools 9.10 requires
  `protobuf>=5.26.1` while Flower requires `protobuf<5.0.0`. The two are
  mutually exclusive as pinned; 9.9.3963 is the newest OR-Tools on the
  protobuf 4.x line.

`pyarrow`, `requests`, `opacus`, and `ruff` are additions — the guide uses each
one (Parquet output, the openFDA API, the RDP accountant, `make lint`) without
listing it.

## Domain model

Seven tables, defined in `pharmadt/core/models.py` and migrated by Alembic:
`drugs`, `nodes`, `batches`, `inventory_records`, `shipments`, `demand_records`,
`agent_decisions`. `provenance_records` is added in Stage 4, alongside the
append-only trigger that makes it immutable — creating the table earlier would
open a window in which the "immutable" audit log is quietly mutable.

Invariants are enforced as database CHECK constraints, not only in Python:
inventory cannot go negative, a shipment cannot arrive before it departs or be
sent to itself, fulfilled demand cannot exceed demand raised, and a cold-chain
drug must carry a temperature band. The twin bulk-inserts events in Stage 3 and
will not be running ORM validators on every row, so the database has to be the
one holding the line.

`compute_batch_fingerprint` is the single definition of a batch's SHA-256
identity, shared by batch creation and by Stage 4's counterfeit check. Fields
are joined with an explicit separator rather than concatenated, so a forger
cannot shift a character across a field boundary and preserve the digest.

## Digital twin

`make sim` runs a SimPy discrete-event simulation over the 12-node network:
1 manufacturer, 2 warehouses, 3 distributors, 6 pharmacies, with lateral edges
between pharmacies served by the same distributor so Stage 7 has real
transshipment routes rather than a round trip through the distributor.

Five processes drive it — consumer demand, ordering and fulfilment, transit,
expiry, and cold chain. Demand is a gamma-mixed Poisson (i.e. negative
binomial) with weekday and seasonal effects: real pharmaceutical demand is
overdispersed, and a plain Poisson would tie variance to the mean and understate
stockout risk, which is the KPI the project is judged on. Stock is consumed
first-expired-first-out.

Baseline over 365 days at seed 42, ~0.2s (≈2,200 steps/s against NFR-01's
1,000):

| KPI | Value |
|---|---|
| Service level | 99.72% |
| Stockout rate (unit-weighted) | 0.28% |
| Inventory turns | 3.74 / year |
| Cold-chain excursions | 12 |
| Events emitted | 14,566 |

The replenishment policy is a deliberately naive fixed-threshold (s, S): its
reorder point covers lead time and the review period and nothing else. It
ignores demand variability entirely, which is precisely the gap Stage 6's
safety-stock term (z·σ·√L, z = 1.65) is meant to close — padding it here would
leave that agent nothing to win.

Two constraints on the parameters are worth knowing before tuning them.
`order_up_to_days` must stay at or below the 28-day demand window, because a
node orders its whole horizon in one lump and its supplier estimates demand by
averaging observed orders over that window; a longer horizon biases the
supplier's estimate upward by the ratio, and the bias compounds at every tier.
And a node's demand history has to be a day-indexed series with quiet days
zero-filled — counting only the days an order happened to arrive divides by the
wrong denominator and inflates the estimated rate the same way. Both bugs were
present and both produced plausible-looking KPIs rather than crashes; there are
named regression tests for each in `tests/test_twin_nodes.py`.

Two runs at the same seed produce byte-identical event logs. That is load
bearing: Stage 4 hashes this event stream into a chain, and Stage 15 compares
ablation arms against each other.

Demand means are currently analytic defaults from `config.py`. Stage 2 refits
them per (node, drug) from Rossmann and replaces the profiles wholesale; the
sampling model does not change, only where its parameters come from.

## Status

Stages 0, 1, and 3 complete — 150 tests, 93% coverage.

Stage 2 (dataset acquisition) is not started. It runs in parallel with Stage 3
in the guide's plan and needs Kaggle and openFDA access; the twin runs on
analytic demand parameters until it lands.

See the implementation guide for the full 15-stage plan; the Stage 10.5
integration gate is the milestone at which the system is complete and
demonstrable.
