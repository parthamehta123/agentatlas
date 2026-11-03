"""Discovery plane: adapters -> identity resolution -> verification -> registry."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

from agentatlas.discovery.base import DiscoveryAdapter
from agentatlas.registry.identity import resolve
from agentatlas.registry.models import AgentRecord, Candidate, Kind
from agentatlas.registry.repository import Registry
from agentatlas.registry.verification import Policy, assess


@dataclass
class DiscoveryReport:
    candidates_by_source: dict[str, int]
    entities: int
    agents: int
    by_status: dict[str, int]
    by_kind: dict[str, int]
    conflicts: list[str]
    seconds: float
    records: list[AgentRecord] = field(repr=False, default_factory=list)


def run_discovery(adapters: list[DiscoveryAdapter], registry: Registry | None = None,
                  policy: Policy | None = None) -> DiscoveryReport:
    t0 = time.perf_counter()
    policy = policy or Policy()
    candidates: list[Candidate] = [c for a in adapters for c in a.collect()]
    records, conflicts = resolve(candidates)
    for r in records:
        assess(r, policy)
    if registry is not None:
        registry.upsert_many(records)
    agents = [r for r in records if r.kind == Kind.AGENT]
    return DiscoveryReport(
        candidates_by_source=dict(Counter(c.source for c in candidates)),
        entities=len(records), agents=len(agents),
        by_status=dict(Counter(r.status.name for r in agents)),
        by_kind=dict(Counter(r.kind.value for r in records)),
        conflicts=conflicts, seconds=round(time.perf_counter() - t0, 2), records=records)
