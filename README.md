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

# 4. Datasets (needs KAGGLE_API_TOKEN in .env)
make data                       # download + preprocess into data/processed/
make eda                        # re-execute the EDA notebook

# 5. Provenance ledger
make keys                       # issue per-node ECDSA keypairs
make sim-anchor                 # run the twin, anchor custody events
make verify-chain               # walk every hash and signature
make tamper-demo                # prove tamper-evidence
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

## Datasets

`make data` runs `pharmadt/ml/preprocessing.py` — one loader per dataset, tidy
DataFrame out, Parquet into `data/processed/`. Raw downloads and processed
outputs are both gitignored; nothing here needs a human in the loop except CMS.

| Dataset | Rows | Feeds | Status |
|---|---|---|---|
| Rossmann store sales | 844,338 | Demand models (Stage 6) | Kaggle token + accepted competition rules |
| Demand profiles | 30 | The twin's stochastic demand | Fitted from Rossmann |
| openFDA recalls | 2,624 | Anomaly labels (Stage 10) | Public, but flaky — the loader retries |
| CVRPLIB benchmarks | 4 | Routing gap vs optimum (Stage 9) | PyVRP mirror; the puc-rio paths 404 |
| Supply-chain priors | 5 | Lead-time and defect priors | Kaggle public dataset |
| CMS Part D | — | Validating Rossmann | **Manual** — `data.cms.gov` returns 403 to scripts |

### What fitting the data actually changed

The twin previously ran on three hand-picked constants. Fitting 30 real series
showed one of them was materially wrong:

| Parameter | Assumed | Fitted range | Verdict |
|---|---|---|---|
| `dispersion` | 0.35 | 0.14 – 0.46 | Plausible |
| `weekend_factor` | 0.60 | 0.28 – 1.51 | Wrong *shape* — 240 of 1,115 stores are **busier** at weekends |
| `seasonal_amplitude` | 0.25 | 0.008 – 0.116 | Overstated ~6× on average |

That is the concrete argument for Stage 2 existing. A wrong-but-stable constant
produces a perfectly plausible KPI, so no amount of staring at the simulation
would have surfaced it.

`make sim` now reports which source it used (`rossmann-fitted` or
`analytic-defaults`) and falls back cleanly, so the twin still runs on a clean
checkout with no datasets downloaded.

### Validity threat, stated plainly

Rossmann is **retail drugstore takings in euros**, not units of a drug
dispensed. The pipeline therefore takes only the *shape* of each series —
variability, weekday effect, seasonality — and rescales the level to
`base_daily_demand`. The magnitude does not transfer and is not claimed to.

The prescribed mitigation is validating against CMS Medicare Part D drug-level
utilisation. `data.cms.gov` returns HTTP 403 to scripted clients, so that
extract must be downloaded by hand into `data/raw/cms/`, where the loader picks
it up. **Until then this threat is open, not closed**, and the report should say
so rather than imply otherwise.

`notebooks/01_eda_datasets.ipynb` documents seasonality, weekday effects, and
missingness, and is committed with its figures so it can be read without running
anything.

## Agent framework

`pharmadt/agents/base.py` and `bus.py` are the skeleton the five agents plug
into. Each runs one `observe → decide → act` cycle per simulated day, driven by
`AgentOrchestrator.run_agents(world_state, sim_day)`.

**Bus topics are the labelled edges of the architecture diagram**, declared as an
enum rather than bare strings. A mistyped topic is otherwise the worst kind of
bug: `publish("replenishment.oder", ...)` reaches nobody, raises nothing, and
surfaces only as an agent that mysteriously never acts. A test asserts the enum
still matches the eight edges in the diagram.

| Topic | Publisher | Subscriber |
|---|---|---|
| `forecast.data` | Demand | Inventory |
| `shortage.alert` | Demand | Inventory |
| `demand.hotspot` | Demand | Expiry |
| `counterfeit.flag` | Anomaly | Expiry, Dashboard |
| `replenishment.order` | Inventory | Route |
| `redistribution.request` | Expiry | Route |
| `route.plan` | Route | Digital twin |
| `ledger.event` | Ledger | Anomaly |

Three decisions worth defending:

- **Subclasses are structurally forbidden from overriding `act()`.** `__init_subclass__`
  raises if they try. `act()` carries the NFR-08 audit logging, and an override
  would bypass it silently — the audit trail would hold for four agents and
  quietly not for the fifth. Agents customise `apply()` instead.
- **Decisions are buffered, not written per action.** NFR-01 wants 1000 steps
  per second and a round trip per decision would put the ceiling far below that,
  so they bulk-insert after the run, exactly as the event log and ledger do.
  Choosing *not* to act is recorded too — "why did the agent not reorder here?"
  is an audit question in its own right.
- **Agents run in dependency order** (Demand → Inventory → Expiry → Route →
  Anomaly), not registration order, since each consumes what the previous
  publishes. Registration order would make results depend on import order.

Attaching an agent costs little: a 365-day run with one attached holds ~1,800
steps/s and stays byte-identical across runs. With no orchestrator set, the twin
runs its Stage 3 baseline untouched — which is exactly the control arm Stage 6
is measured against.

## Provenance ledger

A 365-day run anchors **13,614 custody events**, verified end to end in ~1.1s.
Each record binds its content and its predecessor's hash into a SHA-256 digest,
signed with the acting node's P-256 key. `make tamper-demo` walks the whole
argument: the triggers refuse mutation, the chain verifies, a batch's trace
reads manufacturer → warehouse → distributor → pharmacy, an inclusion proof
checks against its Merkle root, an edited record is caught **and named**, and a
forged fingerprint is rejected. It restores what it changes, so it is repeatable.

**Immutability is enforced by the database, not by convention.** Three triggers
guard the table. The third one matters more than it looks: row-level `DELETE`
triggers do not fire for `TRUNCATE`, so without a statement-level guard the
entire ledger could be erased in one statement while the other two sat and
watched.

The two controls are deliberately independent. The triggers *prevent* tampering;
the hash chain *detects* it. An attacker who can `ALTER TABLE ... DISABLE
TRIGGER` defeats the first and still cannot defeat the second — that is exactly
what the demo shows.

Two design choices worth defending:

- **Merkle anchoring follows RFC 6962 (Certificate Transparency), not Bitcoin.**
  Bitcoin duplicates the final leaf when a level has an odd node count, which
  lets distinct leaf sets produce the same root (CVE-2012-2459). RFC 6962 splits
  at the largest power of two and domain-separates leaves from internal nodes.
  A test asserts the collision case is not collidable.
- **`recorded_at` is outside the hash.** It is database-assigned, and a
  `TIMESTAMPTZ` is not guaranteed to render back to the string it went in as —
  which would fail verification on records nobody touched. Intermittent
  tamper alarms are worse than none, because they train you to ignore them. The
  triggers already block any `UPDATE` to it.

The chain is reproducible even though signatures are not: rebuilding from a
dropped volume with freshly issued keys yields the same tip hash, because the
record hash covers content while ECDSA draws a random nonce per signature.

| Requirement | Fabric mechanism | Delivered here |
|---|---|---|
| FR-07 immutable handoff record | Block + world state | Hash-chained append-only row |
| NFR-04 signed transactions | MSP + X.509 CA | ECDSA P-256 per-node keypair |
| NFR-04 only authorised peers write | Channel policy | Public-key allow-list in `nodes` |
| NFR-08 immutable audit trail | Ledger history | `verify_chain()` + DB triggers |
| Anti-counterfeit | Chaincode hash check | SHA-256 batch fingerprint |

## Status

Stages 0, 1, 2, 3, 4, and 5 complete — 322 tests, lint clean.

Stages 6–10 (the five agents) can now be built in parallel against the
framework, which is the split the guide's team plan assumes.

One dataset (CMS Part D) needs a manual download; everything else builds with
`make data`.

Stage 2 (dataset acquisition) is not started. It runs in parallel with Stage 3
in the guide's plan and needs Kaggle and openFDA access; the twin runs on
analytic demand parameters until it lands.

See the implementation guide for the full 15-stage plan; the Stage 10.5
integration gate is the milestone at which the system is complete and
demonstrable.
