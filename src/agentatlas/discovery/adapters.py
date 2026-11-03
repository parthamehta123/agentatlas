"""Discovery adapters. Only :class:`A2AAdapter` depends on the A2A protocol.

| adapter       | evidence family      | what it can tell us                                 |
|---------------|----------------------|-----------------------------------------------------|
| registration  | declared             | identity, owner, capabilities, endpoint (SDK/API)   |
| a2a           | declared             | Agent Card: skills, endpoint - only for A2A agents  |
| kubernetes    | runtime (+declared)  | workloads, replicas, multi-agent hosts, MCP servers |
| cloud         | runtime (+declared)  | serverless / container workloads by tag             |
| cmdb          | ownership            | owning team, department, data classification        |
| gateway       | behavioral           | who calls LLMs, and whether they use tools          |
| otel          | behavioral           | gen_ai.agent.* spans: runs, outcomes, tools, cost   |
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path

import httpx

from agentatlas.discovery.base import DiscoveryAdapter, slugify
from agentatlas.registry.models import Candidate, Family, Kind

LLM_ENV_HINTS = ("LITELLM_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                 "AZURE_OPENAI_ENDPOINT", "BEDROCK_REGION")
AGENT_ID_ANNOTATION = "agentatlas.io/agent-id"
AGENTS_ANNOTATION = "agentatlas.io/agents"
KIND_LABEL = "agentatlas.io/kind"


def _jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class RegistrationAdapter(DiscoveryAdapter):
    """Agents that self-registered via the SDK / ``POST /agents/register``."""

    name = "registration"

    def __init__(self, path: Path | None = None, records: list[dict] | None = None):
        self.path, self.records = path, records or []

    def collect(self) -> Iterator[Candidate]:
        rows = list(self.records) + (list(_jsonl(self.path)) if self.path else [])
        for r in rows:
            yield Candidate(
                source=self.name, source_ref=r["agent_id"], kind=Kind.AGENT,
                families={Family.DECLARED},
                identity_keys={f"id:{r['agent_id']}", f"slug:{slugify(r['name'])}"},
                attributes={k: r.get(k) for k in (
                    "name", "description", "owner_team", "department", "framework", "model",
                    "capabilities", "tools", "data_clearance", "endpoint", "protocol")},
            )


class A2AAdapter(DiscoveryAdapter):
    """A2A Agent Cards, from a file of cards or fetched from ``/.well-known`` URLs.

    Optional: the rest of the system works without any agent speaking A2A.
    """

    name = "a2a"
    WELL_KNOWN = ("/.well-known/agent-card.json", "/.well-known/agent.json")

    def __init__(self, path: Path | None = None, base_urls: list[str] | None = None,
                 timeout: float = 3.0):
        self.path, self.base_urls, self.timeout = path, base_urls or [], timeout

    def _fetch(self) -> Iterator[dict]:
        with httpx.Client(timeout=self.timeout) as client:
            for base in self.base_urls:
                for suffix in self.WELL_KNOWN:
                    try:
                        resp = client.get(base.rstrip("/") + suffix)
                    except httpx.HTTPError:
                        continue
                    if resp.status_code == 200:
                        yield resp.json()
                        break

    def collect(self) -> Iterator[Candidate]:
        cards = json.loads(self.path.read_text()) if self.path else []
        for card in [*cards, *self._fetch()]:
            skills = card.get("skills", [])
            caps = sorted({t for s in skills for t in (s.get("tags") or [s.get("id")]) if t})
            yield Candidate(
                source=self.name, source_ref=card["url"], kind=Kind.AGENT,
                families={Family.DECLARED},
                identity_keys={f"url:{card['url']}", f"slug:{slugify(card['name'])}"},
                attributes={"name": card["name"], "description": card.get("description", ""),
                            "capabilities": caps, "endpoint": card["url"], "protocol": "a2a",
                            "department": (card.get("provider") or {}).get("organization")},
            )


class KubernetesAdapter(DiscoveryAdapter):
    """Parses ``kubectl get deployments -A -o json`` exports (one per cluster).

    * ``agentatlas.io/agent-id`` annotation -> declared agent
    * ``agentatlas.io/agents: a,b,c``      -> multi-agent host, expanded per agent
    * ``agentatlas.io/kind: mcp-server``   -> MCP server (tools, not an agent)
    * LLM env vars, no annotation          -> llm_workload (needs more evidence)
    """

    name = "kubernetes"

    def __init__(self, paths: list[Path]):
        self.paths = paths

    def collect(self) -> Iterator[Candidate]:
        for path in self.paths:
            cluster = path.stem.split(".", 1)[-1]
            for item in json.loads(path.read_text()).get("items", []):
                yield from self._item(cluster, item)

    def _item(self, cluster: str, item: dict) -> Iterator[Candidate]:
        meta = item["metadata"]
        labels, ann = meta.get("labels") or {}, meta.get("annotations") or {}
        ref = f"{cluster}/{meta['namespace']}/{meta['name']}"
        replicas = int(item.get("spec", {}).get("replicas") or 0)
        containers = item["spec"]["template"]["spec"].get("containers", [])
        env = {e["name"] for c in containers for e in c.get("env", [])}
        image = " ".join(c.get("image", "") for c in containers)
        app = labels.get("app.kubernetes.io/name") or meta["name"]
        attrs = {"name": app, "framework": labels.get("app.kubernetes.io/part-of"),
                 "department": meta["namespace"].title()}

        if labels.get(KIND_LABEL) == "mcp-server" or "mcp-server" in image:
            yield Candidate(source=self.name, source_ref=ref, kind=Kind.MCP_SERVER,
                            families={Family.RUNTIME}, identity_keys={f"k8s:{ref}"},
                            attributes=attrs, replicas=replicas)
            return
        if AGENTS_ANNOTATION in ann:
            hosted = [s.strip() for s in ann[AGENTS_ANNOTATION].split(",") if s.strip()]
            for slug in hosted:
                yield Candidate(
                    source=self.name, source_ref=f"{ref}#{slug}", kind=Kind.AGENT,
                    families={Family.RUNTIME, Family.DECLARED},
                    identity_keys={f"slug:{slugify(slug)}"},
                    attributes={**attrs, "name": slug, "host": ref},
                    replicas=replicas)
            return
        keys = {f"slug:{slugify(app)}"}
        if AGENT_ID_ANNOTATION in ann:
            keys.add(f"id:{ann[AGENT_ID_ANNOTATION]}")
            kind, fams = Kind.AGENT, {Family.RUNTIME, Family.DECLARED}
        elif env & set(LLM_ENV_HINTS):
            kind, fams = Kind.LLM_WORKLOAD, {Family.RUNTIME}
        else:
            kind, fams = Kind.SERVICE, {Family.RUNTIME}
        yield Candidate(source=self.name, source_ref=ref, kind=kind, families=fams,
                        identity_keys=keys, attributes=attrs, replicas=replicas)


class CloudInventoryAdapter(DiscoveryAdapter):
    """Serverless / container inventory (AWS Resource Groups, GCP Asset Inventory...)."""

    name = "cloud"

    def __init__(self, path: Path):
        self.path = path

    def collect(self) -> Iterator[Candidate]:
        for r in json.loads(self.path.read_text()):
            tags = r.get("tags") or {}
            keys = {f"slug:{slugify(r['name'])}"}
            if "agent-id" in tags:
                keys.add(f"id:{tags['agent-id']}")
                kind, fams = Kind.AGENT, {Family.RUNTIME, Family.DECLARED}
            elif tags.get("ai-workload") == "true":
                kind, fams = Kind.LLM_WORKLOAD, {Family.RUNTIME}
            else:
                kind, fams = Kind.SERVICE, {Family.RUNTIME}
            yield Candidate(source=self.name, source_ref=f"{r['provider']}:{r['resource_id']}",
                            kind=kind, families=fams, identity_keys=keys,
                            attributes={"name": r["name"]}, replicas=1)


class CMDBAdapter(DiscoveryAdapter):
    name = "cmdb"

    def __init__(self, path: Path):
        self.path = path

    def collect(self) -> Iterator[Candidate]:
        with open(self.path, newline="") as f:
            for r in csv.DictReader(f):
                kind = Kind.AGENT if r.get("ci_class") == "ai_agent" else Kind.SERVICE
                yield Candidate(
                    source=self.name, source_ref=r["ci_id"], kind=kind,
                    families={Family.OWNERSHIP},
                    identity_keys={f"slug:{slugify(r['ci_name'])}"},
                    attributes={"name": r["ci_name"], "owner_team": r.get("owner_team"),
                                "department": r.get("department"),
                                "data_clearance": r.get("data_classification")})


class GatewayAdapter(DiscoveryAdapter):
    """LLM gateway (LiteLLM / API gateway) usage logs, aggregated per caller.

    Tool-calling traffic is behavioural evidence of an agent loop; plain completion
    traffic (batch summarisers, enrichment jobs) is an LLM client, not an agent.
    """

    name = "gateway"

    def __init__(self, path: Path):
        self.path = path

    def collect(self) -> Iterator[Candidate]:
        for r in _jsonl(self.path):
            agentic = r.get("tool_call_requests", 0) > 0
            yield Candidate(
                source=self.name, source_ref=r["api_key_alias"],
                kind=Kind.LLM_WORKLOAD if agentic else Kind.LLM_CLIENT,
                families={Family.BEHAVIORAL},
                identity_keys={f"slug:{slugify(r['service_name'])}"},
                attributes={"name": r["service_name"], "model": (r.get("models") or [None])[0],
                            "llm_requests": r.get("requests", 0),
                            "tool_call_requests": r.get("tool_call_requests", 0)})


class OTelAdapter(DiscoveryAdapter):
    """Aggregated OpenTelemetry GenAI spans (``gen_ai.agent.id`` / ``gen_ai.agent.name``).

    Also the source of the performance telemetry the ML layer consumes.
    """

    name = "otel"

    def __init__(self, path: Path):
        self.path = path

    def collect(self) -> Iterator[Candidate]:
        for r in _jsonl(self.path):
            keys = {f"slug:{slugify(r['gen_ai.agent.name'])}",
                    f"slug:{slugify(r['service.name'])}"}
            if r.get("gen_ai.agent.id"):
                keys.add(f"id:{r['gen_ai.agent.id']}")
            yield Candidate(
                source=self.name, source_ref=f"{r['gen_ai.agent.id']}/{r['task.category']}",
                kind=Kind.AGENT, families={Family.BEHAVIORAL}, identity_keys=keys,
                attributes={"name": r["gen_ai.agent.name"], "telemetry": {
                    r["task.category"]: {k: r[k] for k in (
                        "runs", "successes", "errors", "p95_latency_ms", "cost_usd_total",
                        "retries", "guardrail_violations")}},
                    "tools_used": r.get("tools_used", {})})


def adapters_from_dir(infra: Path, include_a2a: bool = True) -> list[DiscoveryAdapter]:
    """Wire every adapter whose input exists in an infra directory."""
    out: list[DiscoveryAdapter] = []
    if (p := infra / "registrations.jsonl").exists():
        out.append(RegistrationAdapter(p))
    if include_a2a and (p := infra / "a2a_cards.json").exists():
        out.append(A2AAdapter(p))
    if k8s := sorted(infra.glob("k8s.*.json")):
        out.append(KubernetesAdapter(k8s))
    if (p := infra / "cloud_inventory.json").exists():
        out.append(CloudInventoryAdapter(p))
    if (p := infra / "cmdb.csv").exists():
        out.append(CMDBAdapter(p))
    if (p := infra / "gateway_usage.jsonl").exists():
        out.append(GatewayAdapter(p))
    if (p := infra / "otel_agent_stats.jsonl").exists():
        out.append(OTelAdapter(p))
    return out
