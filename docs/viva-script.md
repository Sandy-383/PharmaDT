# PharmaDT — presentation script

Roughly 12 minutes across three speakers, then questions. Written to be
**spoken**, not read — short sentences, one idea each.

**Before you start:** `make db-up`, then run `make gate` once so the ledger is
populated and the database is warm. Have two terminals and the dashboard open.

Notation: **[SAY]** speak this · **[DO]** run this · **[IF ASKED]** a pivot.

---

## Speaker 1 — the problem and the twin (~4 min)

**[SAY]**

Good morning. We built PharmaDT — a system that keeps medicines available in a
supply chain, and proves they're genuine.

Two problems motivated it. The first is availability: when a pharmacy runs out
of a drug, a patient goes without. The second is trust: the WHO estimates around
one in ten medical products in low- and middle-income countries is substandard
or falsified. Those are usually treated as separate problems. We treated them as
one system.

There are three parts. A **digital twin** — a simulated supply chain we can run
experiments on. **Five autonomous agents** that manage it. And a **cryptographic
provenance ledger** that records every physical handoff so nothing can be
altered without detection.

Let me show you the twin first.

**[DO]** `make sim`

**[SAY]**

Twelve nodes — one manufacturer, two warehouses, three distributors, six
pharmacies, positioned on real coordinates across Karnataka. Five drugs. It runs
365 simulated days in about a fifth of a second, roughly 2,600 steps per second,
against a requirement of a thousand.

Five processes drive it: consumer demand, ordering and fulfilment, transit,
expiry, and cold chain.

One modelling decision worth naming. Demand is **gamma-mixed Poisson** —
negative binomial — not plain Poisson. Real pharmaceutical demand is
overdispersed: variance runs higher than the mean. A plain Poisson forces
variance to *equal* the mean, which would systematically understate stockout
risk. Since stockout rate is the headline number this project is judged on, that
choice matters.

The baseline replenishment policy is deliberately naive — a fixed reorder
threshold that ignores demand variability entirely. That's the control arm. If
we'd tuned the baseline, our agents would have nothing to demonstrate.

The whole simulation is **deterministic**: same seed, byte-identical event log.
That matters because the ledger hashes those events, and because every result
you'll see compares one configuration against another.

Handing over to [name], who'll show you the trust layer.

---

## Speaker 2 — the provenance ledger (~4 min)

**[SAY]**

Our proposal said "blockchain". We didn't build one, and I want to explain why
before showing you what we did build.

In a single-consortium supply chain there are no mutually distrusting,
independently operated validators. Distributed Byzantine consensus solves a
problem this system doesn't have. So we kept every property that matters and
dropped only the consensus.

Tamper-evidence through SHA-256 hash chaining. Non-repudiation through per-node
ECDSA P-256 signatures. Write authorisation through a public-key allow-list —
that's the equivalent of Fabric's channel policy. And immutability enforced by
**database triggers**, so it holds even against our own application code.

Everything goes through an abstract `ProvenanceLedger` interface, so a
Hyperledger backend could be swapped in without touching the agents.

Let me demonstrate the property that matters.

**[DO]** `make tamper-demo`

**[SAY]** *(narrate as it scrolls — don't rush it)*

First, the database refuses to mutate the ledger at all. That's an UPDATE and a
DELETE both rejected by a trigger.

Second, the untouched chain verifies — every hash recomputed, every signature
checked, twenty-six thousand records.

Third, here's a batch's full custody trail: manufacturer, warehouse,
distributor, pharmacy. Each handoff signed by whichever node actually had
custody at that moment.

Now the important part. To tamper with a record, we first have to **disable the
trigger** — a database-level control. Then we change one payload field.

*(pause on the BROKEN line)*

The chain verification fails and names the exact record — sequence 13,614.

Note what happened there. Two independent controls. The triggers **prevent**
tampering. The hash chain **detects** it. An attacker who defeats the first
still cannot defeat the second. And we edit **mid-chain**, not at the end —
changing the last record only breaks its own hash, but changing one in the
middle has to orphan everything after it.

It restores the record and re-verifies, so this demo is repeatable.

**[IF ASKED]** *"Is this real cryptography or a toy?"* — NIST P-256, the same
curve X.509 uses. Merkle anchoring follows RFC 6962, the Certificate
Transparency construction, not Bitcoin's — Bitcoin duplicates the final leaf on
odd levels, which lets different data produce the same root. That's CVE-2012-2459.
We have a test asserting our tree isn't collidable that way.

---

## Speaker 3 — the agents, the results, the limits (~4 min)

**[SAY]**

Five agents run every simulated day: demand forecasting, inventory, expiry
redistribution, vehicle routing, anomaly detection. Each does observe, decide,
act — and every decision is logged with a written justification.

**[DO]** `make gate`

**[SAY]**

This is our integration gate. Seven conditions, one run: the simulation
completes, all five agents act every day, every handoff is signed and chained,
the chain verifies and fails under tampering, KPIs are computed, every decision
is persisted, and the test suite passes at over sixty percent coverage. Six
automated, one — the test suite — deliberately not self-reported, because a gate
that grades itself isn't evidence.

Now the results. Ten random seeds, mean and standard deviation, because a single
run of this system could support almost any claim.

**Stockouts fall 88.7 percent.** **Wastage falls 35.9 percent** — that's inside
the 30 to 40 percent our abstract promised. Routing lands within **2.25 percent
of published optima** on standard benchmark instances. Forecasting more than
halves the error of seasonal-naive.

And the result that ties the system together: our machine-learning models catch
transit anomalies and cold-chain breaches almost perfectly — but they catch
**zero** counterfeits. Not a bug. A counterfeit batch moves in normal quantities
over a normal route in normal time; there is nothing in the shipment data to
see. Adding the ledger's fingerprint check lifts detection recall from **0.62 to
0.98**.

That's the argument for building a cryptographic ledger rather than only using
machine learning: ML provably cannot catch that class of fraud, and the ledger
provably can.

Three limitations we want to state ourselves.

**One.** Under a pandemic scenario — ten times demand for sixty days — our
agents cut unmet demand by more than half, but peak stockout still reaches
54 percent. No inventory policy can meet that against finite manufacturing
output. The agents manage scarcity; they don't create supply.

**Two.** We used Rossmann retail data for demand patterns, which isn't
pharmaceutical. We take only the temporal shape and validate the drug mix
against real Medicare Part D dispensing. Four of our five drugs matched real
products. Real demand varies about tenfold across them; our twin varies about
twofold. That's a stated simplification.

**Three.** We cut the reinforcement-learning stage. Our plan flagged it as
highest risk with an explicit trigger to freeze it and report the omission
rather than submit a rushed negative result. We took that decision.

**[SAY — close]**

If there's one thing we'd want you to take from this: every agent we built was
**measured against a control before we believed it**. Twice that caught an agent
that was actively worse than doing nothing clever — once by 363 percent. We
found those because the measurement existed before the confidence did. Thank you.

---

## Question drill

Short answers. Say the short version, stop, let them follow up.

| Question | First sentence out of your mouth |
|---|---|
| Why not real blockchain? | No mutually distrusting validators exist here, so consensus is over-provisioned — we kept every other property. |
| Your accuracy is only 91%? | Accuracy is the wrong metric; a detector that flags nothing scores 95%. |
| Rossmann isn't pharma data. | Correct — we take only temporal shape, and validate the drug mix against CMS. |
| Why z = 0.84 not 1.65? | We tested 1.65 and it was dominated — same stockouts, eight times the wastage. |
| Is the dashboard React? | No, single-page — no build step, no CDN, so it runs anywhere. Deviation is documented. |
| What went wrong? | Three times an agent was worse than the baseline. *(then tell the echelon lead-time story)* |
| What would you do next? | Close the demand-mix gap properly, then attempt MADDPG with the heuristics as the floor. |

## If a demo fails

Don't debug live. Say: *"That needs the database up — the saved output is here,"*
and open `experiments/`. Every result is committed as JSON, and
`recovery_curves.png` is a figure. Move on and offer to re-run afterwards.
