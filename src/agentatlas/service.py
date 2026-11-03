"""Atlas: the stateful facade used by the API and CLI.

discovery plane  -> registry (SQLite)   -> ML plane (features, clusters) -> selectors
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from agentatlas.discovery.adapters import RegistrationAdapter, adapters_from_dir
from agentatlas.discovery.base import DiscoveryAdapter
from agentatlas.discovery.pipeline import DiscoveryReport, run_discovery
from agentatlas.ml.clustering.algorithms import ClusterResult, hierarchical, kmeans
from agentatlas.ml.features import FeatureStore, build_features
from agentatlas.registry.models import Status
from agentatlas.registry.repository import Registry
from agentatlas.registry.verification import Policy
from agentatlas.selection.engine import Selection, Selector, TaskConfig, TaskSpec


class Atlas:
    def __init__(self, db_path: Path | str = ":memory:", policy: Policy | None = None):
        self.registry = Registry(db_path)
        self.policy = policy or Policy()
        self.registrations: list[dict] = []
        self.features: FeatureStore | None = None
        self.clusters: ClusterResult | None = None
        self.taxonomy: ClusterResult | None = None
        self.selector: Selector | None = None
        self.last_report: DiscoveryReport | None = None

    # ------------------------------------------------------------- discovery
    def register(self, payload: dict) -> None:
        """SDK / API self-registration; reconciled on the next discovery run."""
        self.registrations = [r for r in self.registrations
                              if r["agent_id"] != payload["agent_id"]] + [payload]

    def discover(self, infra: Path | None = None, include_a2a: bool = True,
                 extra: list[DiscoveryAdapter] | None = None) -> DiscoveryReport:
        adapters = adapters_from_dir(infra, include_a2a) if infra else []
        if self.registrations:
            adapters.append(RegistrationAdapter(records=self.registrations))
        adapters += extra or []
        self.last_report = run_discovery(adapters, self.registry, self.policy)
        self.features = self.selector = None
        return self.last_report

    # -------------------------------------------------------------------- ML
    def build(self, k: int = 12) -> FeatureStore:
        agents = self.registry.agents(Status.DISCOVERED)
        if len(agents) < k:
            raise ValueError(f"need at least {k} discovered agents, have {len(agents)}")
        self.features = build_features(agents)
        self.clusters = kmeans(self.features.views["capability"], k)
        self.taxonomy = hierarchical(self.features.views["capability"], k)
        self.selector = Selector(self.features, self.clusters)
        return self.features

    def _ready(self) -> Selector:
        if self.selector is None:
            self.build()
        return self.selector

    def cluster_summary(self) -> list[dict]:
        sel = self._ready()
        fs, labels = sel.fs, self.clusters.labels
        out = []
        for c in range(self.clusters.n_clusters):
            members = np.where(labels == c)[0]
            caps = Counter(cap for i in members for cap in fs.records[i].all_capabilities())
            depts = Counter(fs.records[i].department for i in members)
            rep = fs.records[self.clusters.representatives[c]]
            out.append({
                "cluster": c, "size": len(members),
                "top_capabilities": [k for k, _ in caps.most_common(4)],
                "top_departments": [k for k, _ in depts.most_common(2) if k],
                "authorized_share": round(float(
                    (fs.frame["status"][members] >= Status.AUTHORIZED).mean()), 3),
                "representative": {"agent_id": rep.agent_id, "name": rep.name},
            })
        return sorted(out, key=lambda r: -r["size"])

    # ------------------------------------------------------------- selection
    def select_task(self, spec: TaskSpec, cfg: TaskConfig | None = None) -> Selection:
        return self._ready().select_for_task(spec, cfg)

    def select_representative(self, k: int = 10, outlier_slots: int = 1) -> Selection:
        return self._ready().select_representative(k, outlier_slots)

    def select_security(self, k: int = 10) -> Selection:
        return self._ready().select_for_security(k)
