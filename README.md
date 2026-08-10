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

## Inventory Agent (Stage 6)

`make ablation` runs the twin under both policies on identical seeds:

| Policy | Stockout rate | Units short | Wastage | Avg inventory |
|---|---|---|---|---|
| Fixed-threshold (s, S) baseline | 0.00252 | 1,042 | 0 | 117,724 |
| Inventory Agent | **0.00003** | 12 | 5 | 140,517 (+19%) |

**98.8% stockout reduction**, at a 19% inventory cost that the report states
rather than hides.

### The bug worth reading about

The first version of this agent was **363% worse than the naive baseline**. It
used the *last hop's* transit time as the lead time, which for a 1-day leg gave
a reorder point of ~56 units against the baseline's 120 — so it reordered
*later* than the policy it was meant to beat.

The fix is conceptual, not a tuned constant. In a multi-tier chain an upstream
tier only reorders when its own position dips, so a pharmacy's stock is exposed
for the **cumulative echelon lead time** (3 transit days through
MFG → WH → DC → PH), not for its final leg. Adding the 1-day review interval
gives a risk period of 4 — which is exactly where an independent sweep of the
risk period first reaches zero stockouts. The theory and the measurement agree.

    ROP         = mu * risk + z * sigma * sqrt(risk)
    order-up-to = mu * (risk + coverage) + z * sigma * sqrt(risk)
    risk        = echelon lead time + review period

### A measured deviation from the guide

The guide prescribes `z = 1.65` (textbook 95% service level). Over five seeds,
**`z = 0.84` dominates it on every metric simultaneously**:

| z | Stockout | Units short | Wastage | Avg inventory |
|---|---|---|---|---|
| baseline | 0.00256 | 1,059 | 6 | 116,698 |
| **0.84** | 0.00002 | 10 | 101 | 139,156 |
| 1.65 | 0.00004 | 18 | 878 | 157,414 |

Once the risk period is specified correctly the order-up-to level already
carries most of the buffer, so extra safety stock mostly sits until it expires.
`make frontier` traces the whole curve — and being *tunable* is the real
contribution here, since a fixed threshold offers no dial at all.

## Demand Agent (Stage 7)

`python -m pharmadt.ml.train_demand` trains and scores the forecasters.
**Beating seasonal-naive is the bar**; MASE is scaled so seasonal-naive reads
exactly 1.000.

| Model | sMAPE | MASE | Beats the bar |
|---|---|---|---|
| naive | 26.43 | 0.877 | — |
| seasonal-naive | 30.06 | 1.000 | — |
| moving average | 21.43 | 0.694 | yes |
| **LSTM** | **13.19** | **0.416** | **yes** |
| Prophet | 18.48 | 0.687 | yes |
| Ensemble | 16.23 | 0.614 | yes |

MAPE is reported but is **undefined at zero demand**, so it is computed over
non-zero actuals only and the excluded percentage is printed alongside. sMAPE
and MASE are defined everywhere and are what the conclusions rest on.

### The bullwhip bug

Adding the Demand Agent initially made things *worse*: **+68% inventory** and
700× the wastage. Measured cause — the forecast/history ratio was 1.00 at
retail nodes but **2.26 (peaking at 4.00) upstream**.

Upstream "demand" is not consumer demand. It is the order flow the Inventory
Agent itself generates, which arrives in lumpy bursts; a moving average lands on
a recent burst, doubles the estimate, and that feeds back into the policy that
produced it. Classic bullwhip amplification, built by accident. Scoping the
forecaster to consumer-facing nodes removes it.

| Variant | Stockout | Units short | Avg inventory |
|---|---|---|---|
| baseline (s, S) | 0.00252 | 1,042 | 117,724 |
| inventory only | 0.00003 | 12 | 140,517 |
| **+ demand** | **0.00001** | **2** | **136,892** |

Better service *and* less inventory than inventory-alone — the forecast lets the
reorder point hold a thinner buffer.

## Expiry Agent (Stage 8)

FEFO issuing (already the twin's policy) plus redistribution of near-expiry
stock by **sealed-bid second-price (Vickrey) auction**. Vickrey is named
deliberately: truthful bidding is dominant, because the price a winner pays is
set by the runner-up's bid rather than its own. Under a first-price auction
every node shades its bid and the allocation stops tracking who can actually
use the stock — the only thing redistribution is trying to discover.

Over eight seeds, 365 days:

| | Wastage | Stockout | Avg inventory |
|---|---|---|---|
| without Expiry Agent | 4,123 | 0.00008 | 139,169 |
| **with Expiry Agent** | **462** | 0.00032 | 138,655 |

**88.8% wastage reduction**, inventory unchanged, service still 99.97%. Seven of
eight seeds improved; the eighth (which wasted nothing to begin with) got worse,
and that is reported rather than dropped.

### Three measured corrections

The first version reduced wastage by **4%**, not 88%. Each fix came from a
measurement, not a guess:

1. **Buyers must be downstream, not only lateral.** All wastage sat at
   `NODE-PH-05`, `NODE-WH-02` and `NODE-DC-03` — none of which have same-tier
   peers. Upstream stock had no route out and simply expired where it sat.
   Pushing one tier down, toward consumption, is what a real distributor does.
2. **Act earlier than you alert.** FR-04's 30 days is a *detection* threshold.
   At 30 days the receiving node usually cannot sell the stock either, so the
   transfer relocates the waste. Sweeping the horizon: 30d → 3,728 units,
   120d → 625. Detection and action are different questions.
3. **Buyers must net off stock they already hold.** Otherwise every node claims
   it can absorb the whole lot, stock lands where it will not sell, and
   redistribution manufactures the waste it exists to prevent.

A genuine topology finding fell out of this: a distributor branch with a single
customer (`NODE-DC-03 → NODE-PH-05`) is a **redistribution dead end**. Stock
there has nowhere lateral to go. That is a network-design conclusion, not a
code defect.

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

## Federated learning (Stage 11)

`python -m pharmadt.federated.experiment` — 5 clients, 30 rounds, every variant
scored on the **same held-out test set** no client trains on.

| Variant | sMAPE | MASE | ε |
|---|---|---|---|
| centralised | 15.46 | 0.501 | ∞ |
| federated IID | 16.90 | 0.545 | ∞ |
| federated non-IID (Dirichlet α=0.5) | 17.31 | 0.560 | ∞ |
| FedProx non-IID (μ=0.01) | 17.33 | 0.561 | ∞ |
| federated + DP | 172–193 | 12.5–103 | 1–10 |

**Federation costs 12% sMAPE against pooling the data** — the price of never
moving it. Heterogeneity costs only a further 2.5%, so `Dirichlet(0.5)` skew is
mild at this client count.

**FedProx gives no benefit here** (17.33 vs 17.31), and that is reported rather
than dropped. The guide offers it as a fallback for when non-IID divergence hurts
badly; at a 2.5% heterogeneity cost there is nothing for its proximal term to
fix. A fallback that was not needed is a finding about the split, not a failure.

### Differential privacy is brutal at this scale

DP destroys utility at every ε tested — sMAPE 17 → 172 even at ε=10. That is a
real result, not a bug, and the cause is structural: with **5 clients all
participating every round**, `sample_rate = 1.0`, so there is **no privacy
amplification by subsampling**. Noise calibrated for that regime swamps a
52k-parameter model. A deployment with hundreds of pharmacies and partial
participation per round would sit in a very different place on this curve.

Two corrections were needed before the numbers meant anything: noise is
calibrated to the *sum* of clipped updates, so averaging must divide it by the
client count (an earlier version applied N× too much), and the clip norm is set
adaptively from the median observed update norm (0.62 in practice) rather than
guessed — a blind threshold either never binds or flattens every client.

**NFR-03 is checked structurally.** `ClientUpdate` is a closed dataclass whose
only fields are weight arrays, a sample count and scalar metrics; a test
inspects every array crossing the boundary and confirms none matches the
training data.

## Crisis scenarios (Stage 13)

`python -m pharmadt.crisis.experiment` runs four YAML-defined disruptions
(FR-09) twice each on the same seed — once on the fixed-threshold baseline,
once with the agent stack. This is what makes "resilient" in the title a
measurement.

| Scenario | Peak stockout | Unmet units | Recovery |
|---|---|---|---|
| | base → agents | base → agents | base → agents |
| cold-chain failure | 0.045 → **0.000** | 383 → **0** | 3d → 0d |
| factory shutdown | 0.032 → **0.000** | 737 → **0** | 48d → **0d** |
| pandemic surge | 0.910 → **0.543** | 120,285 → **46,415** | 60d → 60d |
| route disruption | 0.040 → **0.000** | 446 → **0** | 0d → 0d |

**Total unmet demand across all four: 121,851 → 46,415 (61.9% reduction).**

The agents absorb three of the four disruptions completely. **The pandemic surge
they do not** — peak stockout stays at 54%, because no inventory policy can
absorb 10× demand for 60 days when the manufacturer's output is finite. That
limit is worth stating plainly: the agents manage scarcity, they do not create
supply.

Recovery curves are written to `experiments/recovery_curves.png`.

Every effect is **reversible**, which is what makes recovery measurable at all —
a disruption that never lifted would only ever report "never recovered", and two
scenarios could not share a run without corrupting each other's state. Each
scenario holds its own undo record, so overlapping windows revert correctly in
any order.

Recovery requires the stockout rate to fall back **and stay down** for 14 days.
Taking the first crossing would report a single quiet day in the middle of a
shortage as a recovery.

## ★ Experiment matrix (Stage 15)

`make evaluate` — **10 seeds × 365 days, mean ± standard deviation**. Each row
adds one component to the row above, so every line answers "what did this buy?"

| Configuration | Stockout % | Wastage | MAPE % | Delivery km | Avg inventory |
|---|---|---|---|---|---|
| baseline (no agents) | 0.2572 ±0.0433 | 119 ±296 | — | — | 116,216 |
| + inventory & demand | 0.0096 ±0.0133 | 669 ±902 | 5.38 | — | 139,215 |
| + expiry (redistribution) | 0.0291 ±0.0122 | **76 ±115** | 5.43 | — | 139,151 |
| + route optimisation | 0.0291 | 76 | 5.43 | 198,514 ±3,822 | 139,151 |
| + anomaly & ledger (full) | 0.0291 | 76 | 5.43 | 198,514 | 139,151 |

### Abstract claims, checked against measurement

- **Wastage 119 → 76 units = 35.9% reduction — meets the 30–40% claim.**
- **Stockout 0.2572% → 0.0291% = 88.7% reduction.**
- Forecast: the 20–25% improvement claim is evidenced by Stage 7's held-out
  comparison (LSTM sMAPE 13.19 vs seasonal-naive 30.06, **56% better**), not by
  the in-simulation MAPE column.

### Three things the table says that a summary would hide

**The Inventory Agent makes wastage worse before the Expiry Agent fixes it**
(119 → 669 → 76). Holding more stock to eliminate stockouts means more stock
reaches expiry. The two agents are genuinely coupled, and reading only the first
and last rows would miss it.

**The spread is large** (±296, ±902 on wastage). That is exactly why the guide
insists on ≥10 seeds: any single run of this system could support almost any
wastage claim.

**The route and anomaly rows are identical to the expiry row** on simulation
KPIs. That is correct, not a bug: the Route Agent's plan is advisory — shipment
timing comes from the network's transit days — and the Anomaly Agent detects
without altering stock flow. Their value is in the delivery-cost column and in
the detection metrics respectively, not in stockouts.

## ★ Stage 10.5 — Integration Gate

`make gate` stands up the whole autonomous system and checks the guide's seven
conditions in **one run**. Everything printed is measured by that run.

| # | Condition | Result |
|---|---|---|
| 1 | 365 days over ≥12 nodes | 12 nodes, 13,861 events, 12.8s |
| 2 | All five agents act every day | 365/365 days each |
| 3 | Every handoff signed and chained | 13,123 / 13,123 anchored |
| 4 | `verify_chain()` true; false under tampering | VALID over 26,737 → BROKEN at seq 13,614 → VALID |
| 5 | KPIs computed | stockout 0.00036, wastage 0, forecast MAPE 4.11%, 197,517 km |
| 6 | Every decision persisted | 4,413 rows across all five agents |
| 7 | pytest passes, coverage ≥60% | 442 tests, 67% |

Check 7 is `make cov` rather than something the gate runs on itself — a gate
that executes its own test suite and reports its own pass is not evidence.

The tamper step edits a record **mid-chain**, not at the tip. An edit to the
last record breaks only its own hash; one in the middle must also orphan every
record after it, which is the stronger claim.

Tagged `v1.0-integrated`.

## Status

**14 of 15 stages complete.** 520 tests, 84% coverage (NFR-07 requires 80%),
ruff clean.

| Stage | Result |
|---|---|
| 0–5 | Environment, domain model, datasets, twin, ledger, agent framework |
| 6 Inventory | 98.8% stockout reduction |
| 7 Demand | LSTM MASE 0.416 vs seasonal-naive 1.000 |
| 8 Expiry | 88.8% wastage reduction |
| 9 Route | 2.25% mean gap vs CVRPLIB published optima |
| 10 Anomaly | Recall 0.619 → 0.976 once the ledger joins the screening |
| ★ 10.5 | **Integration gate passed** — `v1.0-integrated` |
| 11 Federated | 12% sMAPE cost vs centralised; data never leaves a client |
| 13 Crisis | 61.9% reduction in unmet demand across four disruptions |
| 15 Evaluation | Experiment matrix, 10 seeds, abstract claims validated |

**Stage 12 (MADDPG) was cut** on the guide's own risk assessment, which states
that the Stage 6–10 heuristics ship the project and prescribes reporting the
omission rather than a rushed negative result.

**Stage 14 (dashboard) is not built.** The FastAPI/React layer is presentation
over an API that does not yet exist; every number in this README is reproducible
from the command line without it.

## Reproducing everything

```bash
make db-up && make migrate && make seed && make keys && make data
make gate          # the integration gate, 6/6
make evaluate      # the experiment matrix, 10 seeds
make crisis        # four disruption scenarios
make federated     # centralised vs IID vs non-IID vs DP
make tamper-demo   # ledger tamper-evidence
make cov           # 520 tests, 84%
```
