# AgentAtlas

**ML-powered discovery, clustering and top-k selection for an enterprise estate of
10,000 AI agents. A2A is optional.**

An enterprise has thousands of agents spread across teams, frameworks and clouds. A team
has to pick the 10 that matter, either the best 10 for a task, 10 that represent the whole
estate for an audit, or 10 to security-test first. AgentAtlas does this as two separate
planes:

```
                ┌──────────────────────── DISCOVERY PLANE ────────────────────────┐
 k8s exports ──►│ Kubernetes ─┐                                                    │
 cloud inv.  ──►│ Cloud      ─┤                                                    │
 CMDB csv    ──►│ CMDB       ─┤   Candidates ─► Identity ─► Verification ─► Registry│
 LLM gateway ──►│ Gateway    ─┼─► (raw obs.)    resolution   + policy       (SQLite)│
 OTel spans  ──►│ OTel       ─┤                 union-find   graded trust         │
 SDK / API   ──►│ Registration┤                                                    │
 Agent Cards ──►│ A2A (opt.) ─┘                                                    │
                └──────────────────────────────────────┬──────────────────────────┘
                                                       ▼
                ┌──────────────────────── ML PLANE ───────────────────────────────┐
                │ features: capability / behavior / risk views, empirical-Bayes   │
                │           quality, capability inference for undeclared agents   │
                │ clustering: K-Means · K-Medoids(CLARA) · HDBSCAN · hierarchical │
                │ selection: task-aware │ representative │ security-priority      │
                └─────────────────────────────────────────────────────────────────┘
```

## How does discovery work without A2A?

Clustering doesn't discover agents. It organises agents that have already been
discovered. Discovery comes from evidence the enterprise already has:

| Adapter | Evidence family | What it contributes | Needs A2A? |
|---|---|---|---|
| `registration` (SDK / `POST /agents/register`) | declared | identity, owner, capabilities, endpoint | no |
| `kubernetes` (`kubectl get deploy -A -o json`) | runtime | workloads, replicas, multi-agent hosts, MCP servers | no |
| `cloud` (tagged serverless/containers) | runtime | non-k8s workloads | no |
| `cmdb` (catalog export) | ownership | owning team, data classification | no |
| `gateway` (LiteLLM / API gateway usage) | behavioral | who calls LLMs, and whether with **tool calls** | no |
| `otel` (`gen_ai.agent.*` spans) | behavioral | runs, outcomes, tools used, latency, cost | no |
| `a2a` (Agent Cards at `/.well-known/agent-card.json`) | declared | skills + invocation endpoint | **yes** |

How the evidence becomes a registry entry:

* **Adapters only observe.** They emit *candidates* with identity hints (`id:`, `slug:`,
  `url:`). They never decide whether something is an agent.
* **Identity resolution** joins candidates with union-find over shared keys, after
  normalising name variants (`Fraud Scout 00042` = `FRAUD_SCOUT_00042` = `fraud-scout-00042-svc`).
  Replicas and multi-cluster copies become *instances* of one logical agent. A LangGraph
  host serving 12 agents becomes 12 agents. Two different explicit IDs are never merged;
  a conflict is recorded instead.
* **Classification**: an MCP server exposes tools and isn't an agent. A batch job that
  calls an LLM without tools is an `llm_client`. Tool-calling LLM traffic counts as
  behavioural evidence of an agent loop, which is how unregistered **shadow agents** get
  found.
* **Graded trust**: `CANDIDATE → DISCOVERED → VERIFIED → REACHABLE → AUTHORIZED`.
  VERIFIED needs two independent evidence *families*, one of them runtime or behavioural.
  REACHABLE needs an invocation contract. AUTHORIZED needs a policy pass: an owner, a known
  clearance, no blocked tools, and a guardrail-violation rate under threshold. Being
  discovered never implies being trusted.

**What the benchmark says** (adapter ablation, 10k agents, `reports/BENCHMARK.md` §1):

| variant | recall | precision | reachable | authorized |
|---|---|---|---|---|
| all adapters | 0.998 | 1.0 | 0.593 | 0.525 |
| **without A2A** | **0.998** | 1.0 | 0.541 | 0.500 |
| A2A only | 0.116 | 1.0 | 0.0 | 0.0 |
| without registration | 0.996 | 1.0 | 0.116 | 0.055 |

So A2A barely matters for *finding* agents. It matters for *invoking* them. On their own,
Agent Cards would find 12% of the estate. Dropping the self-registration SDK, on the other
hand, collapses the executable share from 52% to 5.5%: without A2A, invocation needs a
registered contract or another explicitly authorised adapter. 2,086 shadow agents (never
registered, no card, not in CMDB) are found only through runtime and telemetry evidence.

## The three selection modes

| Mode | Pipeline | Use it for |
|---|---|---|
| `task` | policy pre-filter → hybrid retrieval (vector top-100 ∪ capability inverted index; optional IVF cluster probing) → hard capability check → score `0.40Q + 0.20C + 0.15R + 0.15L + 0.10K` (all rank-normalised; Q = Beta-posterior lower bound for the task category) → MMR diversity | "best 10 agents to run this task" |
| `representative` | K-Medoids (CLARA + full-data refinement) on the fused capability+behavior view, plus reserved slots for the most atypical HDBSCAN outliers | audit sampling: 10 agents that stand for the estate |
| `security` | exposure score (blocked/write/PII/external tools, violations, unverified, no owner, …) + IsolationForest anomaly, diversified with MMR across capability space | which 10 agents to red-team first, *including unauthorised ones* |

Authorisation and mandatory capabilities are hard constraints. They never enter as weights.

## Results (synthetic estate, `reports/BENCHMARK.md`)

**Task-aware top-10**, mean over 25 tasks. Gain counts only if the agent truly can do the
task, is cleared for its data, and is authorised to run:

| method | NDCG@10 | P@10 | true quality | unauthorized picks | "lucky" picks | redundancy |
|---|---|---|---|---|---|---|
| random | 0.02 | 0.03 | 0.05 | 48% | 0 | 0.08 |
| vector similarity only | 0.38 | 0.51 | 0.53 | 39% | 0 | 0.69 |
| metadata filter only | 0.64 | 0.88 | 0.59 | 0% | 0.04 | 0.55 |
| raw success rate only | 0.48 | 0.52 | 0.76 | 41% | **4.0 of 10** | 0.69 |
| **AgentAtlas hybrid** | **0.82** | **0.95** | 0.75 | **0%** | 0 | 0.58 |
| AgentAtlas, no MMR | 0.86 | 0.99 | 0.79 | 0% | 0 | 0.79 |
| AgentAtlas + IVF (3 clusters probed) | 0.82 | 0.95 | 0.76 | 0% | 0 | 0.64 |

* **Needle test:** 3 sanctions-screening agents are hidden among 10,000. Every authorised
  one is selected, at rank 1.
* **Representative:** K-Medoids covers 10 of 12 latent groups, against 7.0 for random
  sampling, and has the lowest mean distance to the nearest representative. The
  K-Means-nearest-centroid recipe is close behind but includes no outliers.
* **Clustering** (capability view, k=12): K-Means ARI 0.98, hierarchical 0.98, CLARA 0.88.
  HDBSCAN finds 63 finer clusters and puts **96% of the planted outlier agents** in noise,
  against 5% of normal ones.
* **Security:** 200 of the agents are vulnerable (2% base rate). P@10 = 1.0 and P@50 = 0.78
  while spanning all 10 groups. Exposure-only ranking gives P@50 = 1.0 but spans only 6.
* **Capability inference** for the 4,031 agents with no declared capabilities: precision
  0.90, recall 0.95.
* The whole pipeline (discovery ×9 variants, features, 10 clusterings, 200+ selections)
  runs in about 19s on a laptop. One task selection takes about 12 ms.

### Design decisions the benchmark forced

Each of these came from a measured failure:

1. **Filter before retrieval, not after.** Retrieving top-100 first and filtering second
   returned *zero* agents for 2 of 25 tasks, because ineligible agents used up the
   budget. Policy constraints are now a pre-filter (a `WHERE` clause in pgvector terms).
2. **Don't put hard requirements through vector recall.** Vector top-100 contained only
   27% of the truly best 10 agents. Unioning it with a capability inverted index raised
   that to 100% (NDCG 0.81 → 0.82 under MMR, where diversity, not recall, is the
   binding constraint).
3. **Rank-normalise score components.** Raw quality posteriors sit in a band of about
   [0.55, 0.85], so latency and cost percentiles swamped them. Before the fix, ranking was
   no better than random among eligible agents (NDCG 0.670 vs 0.671).
4. **Shrink success rates.** Ranking on raw success rate picks agents with tiny perfect
   records (3/3) 4 times per top-10. The empirical-Bayes lower bound picks none.
5. **MMR costs relevance.** λ=0.75 lowers NDCG from 0.86 to 0.82 and redundancy from 0.79
   to 0.58. This is a policy knob (`mmr_lambda`), not a free lunch.
6. **A centroid-nearest agent is *typical*, not *best*.** That's fine for audit sampling
   and wrong for task selection, which is why the modes are separate.
7. **Anomaly detection alone is a weak security signal here.** IsolationForest P@50 is
   0.58 against 1.0 for the exposure score. It's kept at 30% weight to catch the unusual
   rather than the merely over-privileged.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"
agentatlas generate --out data --n 10000      # synthetic estate + messy infra artefacts
agentatlas discover --data data --no-a2a      # discovery report without A2A
agentatlas clusters --data data               # 12-cluster capability map
agentatlas select --data data task "Screen a supplier against sanctions lists" \
    --require sanctions_screening --classification restricted --category sanctions_screening
agentatlas select --data data representative
agentatlas select --data data security
agentatlas benchmark --data data --out reports
agentatlas serve --data data                  # FastAPI on :8080, docs at /docs
pytest
```

API: `POST /agents/register`, `POST /discovery/run`, `GET /agents`, `GET /agents/{id}`,
`POST /ml/build`, `GET /clusters`, `POST /select/{task|representative|security}`.
An agent registers itself with `agentatlas.sdk.register(...)`.

## Layout

```
src/agentatlas/
  discovery/   base.py (adapter contract, slugify) · adapters.py (7 adapters) · pipeline.py
  registry/    models.py (Candidate, AgentRecord, Status) · identity.py · verification.py · repository.py
  ml/          features.py (views, Beta shrinkage, capability inference, pluggable embedder)
               clustering/algorithms.py (4 algorithms) · clustering/compare.py
  selection/   engine.py (task / representative / security, MMR)
  evaluation/  benchmark.py (all metrics + baselines)
  synthetic/   archetypes.py · generate.py (estate + k8s/cloud/CMDB/gateway/OTel/A2A artefacts)
  api/main.py · service.py · cli.py · sdk.py · tool_catalog.py
tests/         22 tests: identity, policy, discovery e2e, clustering, selection constraints, API
reports/       BENCHMARK.md
```

## Limitations and next steps

* **The data is synthetic and fairly separable.** Latent groups are distinct, so
  clustering ARI is optimistic. The value is the harness: point the adapters at real
  exports and the same metrics run, minus anything that needs ground truth.
* **The default embedder is TF-IDF+SVD**, deterministic and dependency-free.
  `SentenceTransformerEmbedder` is a drop-in, but it isn't benchmarked yet.
* **The weights (`0.40Q…`, risk weights, MMR λ) are hand-set hypotheses.** Tune them
  against real evaluation runs and feedback, not this synthetic set.
* **Not built yet:** live Kubernetes/cloud API clients (adapters read exports), an OTel
  collector receiver, Postgres+pgvector storage, drift detection between registry
  snapshots, a dashboard, and contextual-bandit feedback from real task outcomes.
