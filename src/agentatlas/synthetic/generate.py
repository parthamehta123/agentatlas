"""Generate a reproducible synthetic enterprise with N AI agents.

Two kinds of output:

* ``truth/``  - hidden ground truth (latent group, true per-category quality, planted
  vulnerabilities, which artefacts each agent appears in). Only evaluation reads it.
* ``infra/``  - the messy artefacts a real enterprise actually has: Kubernetes
  deployments, cloud function inventory, a CMDB export, SDK self-registrations, a few
  A2A Agent Cards, LLM-gateway usage logs and OpenTelemetry GenAI span aggregates.
  Discovery runs only on these.

The infra deliberately contains replicas, multi-agent hosts, MCP servers, LLM-calling
batch jobs, plain microservices, name variants and "shadow" agents that were never
registered anywhere.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from agentatlas.synthetic.archetypes import (
    ARCHETYPES,
    CLASSIFICATION_ORDER,
    FRAMEWORKS,
    GENERIC_PHRASES,
    MODELS,
    NEEDLE_CAPABILITY,
    NEEDLE_CATEGORY,
    NEEDLE_PHRASES,
    NEEDLE_TASK,
    Archetype,
)
from agentatlas.tool_catalog import BLOCKED_TOOLS, PII_TOOLS, WRITE_TOOLS

NOUNS = ("sentinel", "copilot", "assistant", "navigator", "analyst", "scout", "pilot",
         "helper", "operator", "advisor", "auditor", "concierge")
RISKY_TOOLS = tuple(sorted(BLOCKED_TOOLS))
# Content filters fire more often for some domains; violations alone are not proof.
BASE_VIOLATION_RATE = {"marketing_content": 0.008, "customer_support": 0.005,
                       "research_rag": 0.004}


@dataclass
class GenConfig:
    n_agents: int = 10_000
    outlier_rate: float = 0.10
    needle_count: int = 3
    vulnerable_rate: float = 0.02
    decoy_risky_rate: float = 0.015
    lucky_count: int = 40
    seed: int = 42
    # artefact coverage probabilities
    p_registration: float = 0.55
    p_a2a: float = 0.12
    p_k8s: float = 0.60
    p_cloud: float = 0.30  # of those not on k8s
    p_cmdb: float = 0.50
    p_otel: float = 0.85
    p_gateway: float = 0.90
    p_k8s_agent_annotation: float = 0.60
    multi_agent_host_rate: float = 0.08
    n_plain_services: int = 800
    n_mcp_servers: int = 150
    n_batch_jobs: int = 120
    telemetry_days: int = 30


@dataclass
class TrueAgent:
    agent_id: str
    slug: str
    name: str
    group: str
    is_outlier: bool
    is_needle: bool
    is_lucky: bool
    vulnerable: bool
    department: str
    owner_team: str
    framework: str
    model: str
    capabilities: list[str]
    tools: list[str]
    categories: list[str]
    true_quality: dict[str, float]
    data_clearance: str
    p95_latency_ms: float
    cost_per_task_usd: float
    error_rate: float
    availability: float
    description: str
    sources: list[str] = field(default_factory=list)
    invocation: str = "none"  # a2a | http | none
    endpoint: str | None = None


def _beta(rng, mean, conc):
    return float(rng.beta(mean * conc, (1 - mean) * conc))


def _description(rng, phrases: list[str]) -> str:
    picks = list(rng.choice(phrases, size=min(3, len(phrases)), replace=False))
    picks.append(str(rng.choice(GENERIC_PHRASES)))
    rng.shuffle(picks)
    return "Agent that " + ", ".join(picks) + "."


def _pick_groups(rng, n):
    weights = rng.dirichlet(np.full(len(ARCHETYPES), 4.0))
    return rng.choice(len(ARCHETYPES), size=n, p=weights)


def _make_agent(rng, idx: int, arche: Archetype, outlier: bool) -> TrueAgent:
    agent_id = f"agt-{idx:05d}"
    noun = str(rng.choice(NOUNS))
    slug = f"{arche.key.split('_')[0]}-{noun}-{idx:05d}"
    name = slug.replace("-", " ").title()
    caps = list(arche.capabilities)
    if len(caps) > 2 and rng.random() < 0.3:
        caps.remove(str(rng.choice(caps[1:])))  # first capability is always core
    tools = [t for t in arche.tools if rng.random() < 0.85] or [arche.tools[0]]
    phrases = list(arche.phrases)
    categories = list(arche.categories)
    latency = arche.latency_ms * float(rng.lognormal(0, 0.35))
    cost = arche.cost_usd * float(rng.lognormal(0, 0.4))
    error = float(rng.beta(2, 60))
    if outlier:
        # unusual agents: capabilities borrowed from other domains + odd operations
        others = rng.choice(len(ARCHETYPES), size=2, replace=False)
        for o in others:
            oa = ARCHETYPES[o]
            caps.append(oa.capabilities[0])
            tools.append(oa.tools[0])
            phrases.extend(oa.phrases[:2])
            categories.append(oa.categories[0])
        latency *= float(rng.choice([0.2, 5.0]))
        cost *= float(rng.choice([0.2, 6.0]))
        error = float(rng.beta(2, 15))
    star = rng.random() < 0.08
    base_mean = 0.85 if star else 0.66
    quality = {c: _beta(rng, base_mean if c in arche.categories else 0.55, 25)
               for c in dict.fromkeys(categories)}
    return TrueAgent(
        agent_id=agent_id, slug=slug, name=name, group=arche.key, is_outlier=outlier,
        is_needle=False, is_lucky=False, vulnerable=False, department=arche.department,
        owner_team=f"{arche.department.lower()}-team-{int(rng.integers(1, 9))}",
        framework=str(rng.choice(FRAMEWORKS)), model=str(rng.choice(MODELS)),
        capabilities=list(dict.fromkeys(caps)), tools=list(dict.fromkeys(tools)),
        categories=list(dict.fromkeys(categories)), true_quality=quality,
        data_clearance=(arche.data_classification if rng.random() < 0.85 else
                        str(rng.choice(CLASSIFICATION_ORDER))),
        p95_latency_ms=round(latency, 1), cost_per_task_usd=round(cost, 5),
        error_rate=round(error, 4), availability=round(float(rng.uniform(0.95, 0.9999)), 4),
        description=_description(rng, phrases),
    )


def generate_agents(cfg: GenConfig) -> list[TrueAgent]:
    rng = np.random.default_rng(cfg.seed)
    groups = _pick_groups(rng, cfg.n_agents)
    outliers = rng.random(cfg.n_agents) < cfg.outlier_rate
    agents = [_make_agent(rng, i, ARCHETYPES[g], bool(o))
              for i, (g, o) in enumerate(zip(groups, outliers))]

    # Needles: rare, highly capable sanctions-screening agents hidden in legal/fraud groups.
    legalish = [a for a in agents if a.group in ("legal_review", "fraud_detection")
                and not a.is_outlier]
    for a in rng.choice(legalish, size=cfg.needle_count, replace=False):
        a.is_needle = True
        a.capabilities.append(NEEDLE_CAPABILITY)
        a.tools.append("sanctions_list_api")
        a.categories.append(NEEDLE_CATEGORY)
        a.true_quality[NEEDLE_CATEGORY] = _beta(rng, 0.9, 40)
        a.description = "Agent that " + ", ".join(NEEDLE_PHRASES) + ", " + a.description[11:]
        a.data_clearance = "restricted"

    # Planted vulnerabilities. Half are "loud" (risky tools); half are latent - only
    # over-privilege and a noisy rise in guardrail violations give them away.
    n_vuln = int(cfg.vulnerable_rate * cfg.n_agents)
    for n, a in enumerate(rng.choice(agents, size=n_vuln, replace=False)):
        a.vulnerable = True
        if n % 2 == 0:
            a.tools.extend(rng.choice(RISKY_TOOLS, size=int(rng.integers(1, 3)),
                                      replace=False))
        else:
            a.tools.extend(rng.choice(sorted((PII_TOOLS | WRITE_TOOLS) - BLOCKED_TOOLS),
                                      size=2, replace=False))
        a.tools = list(dict.fromkeys(a.tools))
    # Decoys: legitimate ops/engineering agents that are approved to use risky tools.
    ops = [a for a in agents if a.group in ("it_ops", "code_generation") and not a.vulnerable]
    for a in rng.choice(ops, size=int(cfg.decoy_risky_rate * cfg.n_agents), replace=False):
        a.tools.append("shell_exec")

    # "Lucky" agents: mediocre true quality, but will show a tiny perfect track record.
    candidates = [a for a in agents if not a.is_needle and not a.vulnerable]
    for a in rng.choice(candidates, size=cfg.lucky_count, replace=False):
        a.is_lucky = True
        for c in a.true_quality:
            a.true_quality[c] = _beta(rng, 0.35, 30)
    return agents


# ---------------------------------------------------------------------------
# Infrastructure artefacts
# ---------------------------------------------------------------------------

def _name_variant(rng, slug: str) -> str:
    """How humans and tools mangle a name across systems."""
    r = rng.random()
    if r < 0.3:
        return slug.replace("-", " ").title()
    if r < 0.5:
        return slug.replace("-", "_").upper()
    if r < 0.65:
        return slug + str(rng.choice(["-svc", "-agent", "-v2", "-service"]))
    return slug


def build_infra(agents: list[TrueAgent], cfg: GenConfig) -> dict[str, list]:
    rng = np.random.default_rng(cfg.seed + 1)
    registrations, a2a_cards, k8s, cloud, cmdb, gateway, otel = [], [], [], [], [], [], []
    host_pool: dict[str, list[TrueAgent]] = {}

    for a in agents:
        registered = rng.random() < cfg.p_registration
        has_card = rng.random() < cfg.p_a2a
        if a.vulnerable and rng.random() < 0.6:  # risky agents skew towards shadow IT
            registered, has_card = False, False
        if registered:
            a.sources.append("registration")
            a.invocation, a.endpoint = "http", f"https://agents.internal/{a.slug}/invoke"
            registrations.append({
                "agent_id": a.agent_id, "name": a.name, "owner_team": a.owner_team,
                "department": a.department, "framework": a.framework, "model": a.model,
                "description": a.description, "capabilities": a.capabilities,
                "tools": a.tools, "data_clearance": a.data_clearance,
                "endpoint": a.endpoint, "protocol": "http",
            })
        if has_card:
            a.sources.append("a2a")
            a.invocation, a.endpoint = "a2a", f"https://{a.slug}.agents.internal/a2a"
            a2a_cards.append({
                "name": a.name, "description": a.description, "url": a.endpoint,
                "version": "1.0.0", "provider": {"organization": a.department},
                "capabilities": {"streaming": bool(rng.random() < 0.5)},
                "skills": [{"id": c, "name": c.replace("_", " "), "tags": [c]}
                           for c in a.capabilities],
            })
        if rng.random() < cfg.p_cmdb and not (a.vulnerable and rng.random() < 0.7):
            a.sources.append("cmdb")
            cmdb.append({"ci_id": f"CI{int(rng.integers(1e6, 1e7))}",
                         "ci_name": _name_variant(rng, a.slug), "owner_team": a.owner_team,
                         "department": a.department, "ci_class": "ai_agent",
                         "data_classification": a.data_clearance})

        if rng.random() < cfg.p_k8s:
            a.sources.append("k8s")
            if rng.random() < cfg.multi_agent_host_rate:
                host_pool.setdefault(a.department, []).append(a)
            else:
                clusters = ["prod-us-east"] + (["prod-eu-west"] if rng.random() < 0.15 else [])
                for cl in clusters:
                    ann = {}
                    if rng.random() < cfg.p_k8s_agent_annotation:
                        ann["agentatlas.io/agent-id"] = a.agent_id
                    k8s.append({
                        "cluster": cl, "namespace": a.department.lower(),
                        "name": _name_variant(rng, a.slug).lower().replace("_", "-")
                        .replace(" ", "-"),
                        "replicas": int(rng.integers(1, 7)),
                        "labels": {"app.kubernetes.io/name": a.slug,
                                   "app.kubernetes.io/part-of": a.framework},
                        "annotations": ann, "image": f"registry.internal/{a.framework}:"
                                                     f"{int(rng.integers(1, 30))}",
                        "env": ["LITELLM_BASE_URL", "OTEL_EXPORTER_OTLP_ENDPOINT"],
                    })
        elif rng.random() < cfg.p_cloud:
            a.sources.append("cloud")
            cloud.append({"provider": str(rng.choice(["aws", "gcp", "azure"])),
                          "resource_type": str(rng.choice(["lambda", "cloud_run",
                                                           "container_app"])),
                          "resource_id": f"res-{int(rng.integers(1e8, 1e9))}",
                          "name": _name_variant(rng, a.slug),
                          "tags": ({"agent-id": a.agent_id} if rng.random() < 0.5 else
                                   {"ai-workload": "true"})})

        if rng.random() < cfg.p_gateway:
            a.sources.append("gateway")
            gateway.append({"service_name": _name_variant(rng, a.slug),
                            "api_key_alias": f"key-{a.slug}", "models": [a.model],
                            "requests": int(rng.lognormal(7, 1.2)),
                            "tool_call_requests": int(rng.lognormal(6, 1.2)),
                            "window_days": cfg.telemetry_days})
        if rng.random() < cfg.p_otel:
            a.sources.append("otel")
            otel.extend(_otel_rows(rng, a, cfg))

    # Multi-agent hosts: one LangGraph deployment serving several logical agents.
    for dept, members in host_pool.items():
        for chunk_start in range(0, len(members), 12):
            chunk = members[chunk_start:chunk_start + 12]
            k8s.append({
                "cluster": "prod-us-east", "namespace": dept.lower(),
                "name": f"{dept.lower()}-agent-host-{chunk_start // 12}",
                "replicas": int(rng.integers(3, 10)),
                "labels": {"app.kubernetes.io/name": f"{dept.lower()}-agent-host",
                           "app.kubernetes.io/part-of": "langgraph"},
                "annotations": {"agentatlas.io/agents": ",".join(m.slug for m in chunk)},
                "image": "registry.internal/langgraph-platform:12",
                "env": ["LITELLM_BASE_URL", "OTEL_EXPORTER_OTLP_ENDPOINT"],
            })

    # Things that look agent-ish but are not agents.
    for i in range(cfg.n_plain_services):
        k8s.append({"cluster": "prod-us-east", "namespace": "platform",
                    "name": f"svc-{i:04d}", "replicas": int(rng.integers(1, 5)),
                    "labels": {"app.kubernetes.io/name": f"svc-{i:04d}"}, "annotations": {},
                    "image": "registry.internal/java-service:3", "env": ["DATABASE_URL"]})
    for i in range(cfg.n_mcp_servers):
        k8s.append({"cluster": "prod-us-east", "namespace": "tools",
                    "name": f"mcp-{i:04d}", "replicas": 2,
                    "labels": {"app.kubernetes.io/name": f"mcp-{i:04d}",
                               "agentatlas.io/kind": "mcp-server"}, "annotations": {},
                    "image": "registry.internal/mcp-server:1", "env": ["LITELLM_BASE_URL"]})
    for i in range(cfg.n_batch_jobs):
        gateway.append({"service_name": f"nightly-batch-summariser-{i:03d}",
                        "api_key_alias": f"key-batch-{i:03d}", "models": ["gpt-4o-mini"],
                        "requests": int(rng.lognormal(8, 1)), "tool_call_requests": 0,
                        "window_days": cfg.telemetry_days})

    return {"registrations": registrations, "a2a_cards": a2a_cards, "k8s_deployments": k8s,
            "cloud_inventory": cloud, "cmdb": cmdb, "gateway_usage": gateway,
            "otel_agent_stats": otel}


def _otel_rows(rng, a: TrueAgent, cfg: GenConfig) -> list[dict]:
    """Aggregated OTel GenAI spans (gen_ai.agent.*) per agent x task category."""
    rows = []
    for c in a.categories:
        if a.is_lucky:
            runs = int(rng.integers(2, 5))
            succ = runs  # perfect, tiny track record
        else:
            runs = int(np.clip(rng.lognormal(4.0, 1.6), 0, 20_000))
            succ = int(rng.binomial(runs, a.true_quality[c])) if runs else 0
        if runs == 0:
            continue
        tools_used = {t: int(rng.poisson(max(1, runs) * rng.uniform(0.3, 2.0)))
                      for t in a.tools}
        viol_rate = BASE_VIOLATION_RATE.get(a.group, 0.002) * (6 if a.vulnerable else 1)
        rows.append({
            "gen_ai.agent.id": a.agent_id, "gen_ai.agent.name": a.name,
            "service.name": _name_variant(rng, a.slug), "task.category": c,
            "runs": runs, "successes": succ,
            "errors": int(rng.binomial(runs, a.error_rate)),
            "p95_latency_ms": round(a.p95_latency_ms * float(rng.lognormal(0, 0.1)), 1),
            "cost_usd_total": round(a.cost_per_task_usd * runs, 4),
            "retries": int(rng.poisson(runs * a.error_rate * 2)),
            "guardrail_violations": int(rng.binomial(runs, viol_rate)),
            "tools_used": tools_used, "window_days": cfg.telemetry_days,
        })
    return rows


def build_tasks(agents: list[TrueAgent]) -> list[dict]:
    tasks = []
    for arche in ARCHETYPES:
        for cat in arche.categories:
            tasks.append({"task_id": f"task-{cat}", "description": arche.task_phrasings[cat],
                          "category": cat, "required_capabilities": [arche.capabilities[0]],
                          "data_classification": arche.data_classification})
    tasks.append({"task_id": f"task-{NEEDLE_CATEGORY}", "description": NEEDLE_TASK,
                  "category": NEEDLE_CATEGORY, "required_capabilities": [NEEDLE_CAPABILITY],
                  "data_classification": "restricted"})
    return tasks


def _kubectl_item(d: dict) -> dict:
    return {
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": d["name"], "namespace": d["namespace"], "labels": d["labels"],
                     "annotations": d["annotations"]},
        "spec": {"replicas": d["replicas"], "template": {"spec": {"containers": [
            {"name": "main", "image": d["image"],
             "env": [{"name": e, "value": "redacted"} for e in d["env"]]}]}}},
    }


def write_dataset(out: Path, cfg: GenConfig | None = None) -> dict:
    cfg = cfg or GenConfig()
    agents = generate_agents(cfg)
    infra = build_infra(agents, cfg)
    tasks = build_tasks(agents)
    (out / "infra").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)

    for key in ("registrations", "gateway_usage", "otel_agent_stats"):
        with open(out / "infra" / f"{key}.jsonl", "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in infra[key])
    for key in ("a2a_cards", "cloud_inventory"):
        (out / "infra" / f"{key}.json").write_text(json.dumps(infra[key]))
    # One `kubectl get deployments -A -o json` export per cluster.
    by_cluster: dict[str, list] = {}
    for d in infra["k8s_deployments"]:
        by_cluster.setdefault(d["cluster"], []).append(_kubectl_item(d))
    for cluster, items in by_cluster.items():
        (out / "infra" / f"k8s.{cluster}.json").write_text(
            json.dumps({"apiVersion": "v1", "kind": "List", "items": items}))
    with open(out / "infra" / "cmdb.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(infra["cmdb"][0].keys()))
        w.writeheader()
        w.writerows(infra["cmdb"])

    (out / "truth" / "agents.json").write_text(json.dumps([asdict(a) for a in agents]))
    (out / "truth" / "tasks.json").write_text(json.dumps(tasks, indent=1))
    (out / "truth" / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    return {"agents": len(agents), **{k: len(v) for k, v in infra.items()},
            "tasks": len(tasks)}
