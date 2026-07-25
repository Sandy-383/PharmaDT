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

# 3. Verify
make test
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

## Status

Stage 0 complete. See the implementation guide for the full 15-stage plan; the
Stage 10.5 integration gate is the milestone at which the system is complete
and demonstrable.
