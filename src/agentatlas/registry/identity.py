"""Identity resolution: many raw candidates -> one logical agent.

Union-find over identity keys (``id:``, ``slug:``, ``url:``). Two candidates that share
any key are the same logical agent - except when that would merge two different
explicit agent IDs, which is recorded as a conflict instead of silently merged.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from agentatlas.registry.models import (
    AgentRecord,
    Candidate,
    CategoryStats,
    Evidence,
    Family,
    Kind,
)

# Which source wins for scalar attributes (first non-empty wins).
PRECEDENCE = ("registration", "a2a", "cmdb", "otel", "kubernetes", "cloud", "gateway")
SCALAR_FIELDS = ("name", "description", "owner_team", "department", "framework", "model",
                 "data_clearance", "endpoint", "protocol")
KIND_PRIORITY = (Kind.MCP_SERVER, Kind.LLM_CLIENT, Kind.LLM_WORKLOAD, Kind.SERVICE)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.ids: list[set[str]] = [set() for _ in range(n)]

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.ids[ra] and self.ids[rb] and self.ids[ra] != self.ids[rb]:
            return False
        self.parent[rb] = ra
        self.ids[ra] |= self.ids[rb]
        return True


def resolve(candidates: list[Candidate]) -> tuple[list[AgentRecord], list[str]]:
    """Return one record per logical entity (agents *and* non-agents) plus conflicts."""
    # explicit IDs first so that slug-only candidates attach to an ID-bearing root
    candidates = sorted(candidates, key=lambda c: not any(k.startswith("id:")
                                                          for k in c.identity_keys))
    uf = _UnionFind(len(candidates))
    for i, c in enumerate(candidates):
        uf.ids[i] = {k for k in c.identity_keys if k.startswith("id:")}
    owner: dict[str, int] = {}
    conflicts: list[str] = []
    for i, c in enumerate(candidates):
        for key in sorted(c.identity_keys):
            if key not in owner:
                owner[key] = i
            elif not uf.union(owner[key], i):
                conflicts.append(f"{key}: {c.source}:{c.source_ref} links two agent ids")

    groups: dict[int, list[Candidate]] = defaultdict(list)
    for i, c in enumerate(candidates):
        groups[uf.find(i)].append(c)
    return [_merge(members) for members in groups.values()], conflicts


def _merge(members: list[Candidate]) -> AgentRecord:
    members = sorted(members, key=lambda c: PRECEDENCE.index(c.source)
                     if c.source in PRECEDENCE else len(PRECEDENCE))
    keys = [k for c in members for k in c.identity_keys]
    ids = sorted({k[3:] for k in keys if k.startswith("id:")})
    slugs = Counter(k[5:] for k in keys if k.startswith("slug:"))
    slug = slugs.most_common(1)[0][0] if slugs else members[0].source_ref

    kinds = {c.kind for c in members}
    inferred = False
    if Kind.AGENT in kinds:
        kind = Kind.AGENT
    elif any(c.source == "gateway" and c.kind == Kind.LLM_WORKLOAD for c in members):
        kind, inferred = Kind.AGENT, True  # tool-calling LLM traffic => agent loop
    else:
        kind = next(k for k in KIND_PRIORITY if k in kinds)

    scalars: dict[str, object] = {}
    caps: list[str] = []
    tools: list[str] = []
    for c in members:
        for f in SCALAR_FIELDS:
            if not scalars.get(f) and c.attributes.get(f):
                scalars[f] = c.attributes[f]
        caps += c.attributes.get("capabilities") or []
        tools += c.attributes.get("tools") or []

    telemetry: dict[str, CategoryStats] = {}
    tools_used: Counter = Counter()
    llm_requests = 0
    for c in members:
        for cat, st in (c.attributes.get("telemetry") or {}).items():
            cur = telemetry.setdefault(cat, CategoryStats())
            cur.runs += st["runs"]
            cur.successes += st["successes"]
            cur.errors += st["errors"]
            cur.cost_usd_total += st["cost_usd_total"]
            cur.retries += st["retries"]
            cur.guardrail_violations += st["guardrail_violations"]
            cur.p95_latency_ms = max(cur.p95_latency_ms or 0, st["p95_latency_ms"] or 0)
        tools_used.update(c.attributes.get("tools_used") or {})
        llm_requests += int(c.attributes.get("llm_requests") or 0)

    record = AgentRecord(
        agent_id=ids[0] if len(ids) == 1 else f"inferred:{slug}",
        slug=slug, name=str(scalars.pop("name", slug)), kind=kind, inferred=inferred,
        capabilities=list(dict.fromkeys(caps)),
        tools=list(dict.fromkeys(tools + list(tools_used))),
        instances=sum(c.replicas for c in members if Family.RUNTIME in c.families),
        evidence=[Evidence(source=c.source, source_ref=c.source_ref,
                           families=sorted(c.families)) for c in members],
        telemetry=telemetry, tools_used=dict(tools_used), llm_requests=llm_requests,
        conflicts=[f"multiple ids: {ids}"] if len(ids) > 1 else [],
        **{k: v for k, v in scalars.items() if v is not None},
    )
    return record
