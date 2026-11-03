"""Three independent top-k selection strategies over the feature store.

1. ``select_for_task``      - best k agents for a task:
       hybrid retrieval -> hard eligibility filters -> weighted ranking -> MMR diversity
2. ``select_representative`` - k agents that best represent the estate (audit sampling):
       K-Medoids over the fused view, with reserved slots for HDBSCAN outliers
3. ``select_for_security``  - k agents to security-test first:
       exposure score + IsolationForest anomaly, diversified across attack surfaces

Security authorisation and mandatory capabilities are hard constraints, never weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import normalize

from agentatlas.ml.clustering.algorithms import ClusterResult, hdbscan, kmedoids
from agentatlas.ml.features import FeatureStore
from agentatlas.registry.models import Status
from agentatlas.synthetic.archetypes import CLASSIFICATION_ORDER

DEFAULT_WEIGHTS = {"Q": 0.40, "C": 0.20, "R": 0.15, "L": 0.15, "K": 0.10}
# Exposure weights over FeatureStore.frame["risk_raw"] columns (see features.py).
RISK_WEIGHTS = np.array([3.0, 0.5, 0.7, 0.7, 1.5, 1.0, 1.0, 0.5, 0.3, 0.2, 0.2])


@dataclass
class TaskSpec:
    description: str
    task_id: str = "adhoc"
    category: str | None = None
    required_capabilities: list[str] = field(default_factory=list)
    data_classification: str = "internal"
    max_p95_latency_ms: float | None = None
    max_cost_per_task: float | None = None


@dataclass
class TaskConfig:
    k: int = 10
    retrieve_n: int = 100
    # Union the semantic top-n with every agent that has all required capabilities
    # (an inverted index lookup). Hard requirements should not depend on vector recall.
    capability_index: bool = True
    max_candidates: int = 2000
    min_status: Status = Status.AUTHORIZED
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    mmr_lambda: float = 0.75
    use_lower_bound: bool = True
    # Cluster-pruned retrieval (IVF-style): only scan the n closest clusters.
    cluster_probe: int | None = None


@dataclass
class Selection:
    mode: str
    agent_ids: list[str]
    rows: list[dict]
    funnel: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _pct_rank(x: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    if len(x) <= 1:
        return np.ones_like(x, dtype=float)
    r = np.argsort(np.argsort(x if higher_is_better else -x)).astype(float)
    return r / (len(x) - 1)


def mmr(scores: np.ndarray, vectors: np.ndarray, k: int, lam: float) -> list[int]:
    """Maximal Marginal Relevance over candidate positions."""
    V = normalize(vectors)
    chosen: list[int] = []
    max_sim = np.zeros(len(scores))
    remaining = np.ones(len(scores), dtype=bool)
    for _ in range(min(k, len(scores))):
        obj = np.where(remaining, lam * scores - (1 - lam) * max_sim, -np.inf)
        j = int(obj.argmax())
        chosen.append(j)
        remaining[j] = False
        max_sim = np.maximum(max_sim, V @ V[j])
    return chosen


class Selector:
    def __init__(self, fs: FeatureStore, clusters: ClusterResult | None = None):
        self.fs = fs
        self.clusters = clusters
        self.diversity_space = fs.fused()
        if clusters is not None:
            labels = clusters.labels
            self.cluster_centroids = {
                c: normalize(fs.embeddings[labels == c].mean(0, keepdims=True))[0]
                for c in set(labels.tolist()) if c != -1}

    # ------------------------------------------------------------------ task mode
    def capability_match(self, required: list[str]) -> np.ndarray:
        if not required:
            return np.ones(len(self.fs.records))
        cols = [self.fs.capability_names.index(c) for c in required
                if c in self.fs.capability_names]
        if not cols:
            return np.zeros(len(self.fs.records))
        return self.fs.capability_matrix[:, cols].sum(1) / len(required)

    def prefilter(self, task: TaskSpec, cfg: TaskConfig) -> tuple[np.ndarray, dict[str, int]]:
        """Cheap metadata constraints applied *before* the vector scan (a WHERE clause in
        pgvector/Milvus), so ineligible agents cannot crowd out the retrieval budget."""
        fr = self.fs.frame
        need = CLASSIFICATION_ORDER.index(task.data_classification)
        checks = {
            "status_below_" + cfg.min_status.name.lower(): fr["status"] < cfg.min_status,
            "insufficient_data_clearance": fr["clearance"] < need,
        }
        if task.max_p95_latency_ms:
            checks["latency_budget"] = fr["p95_latency_ms"] > task.max_p95_latency_ms
        if task.max_cost_per_task:
            checks["cost_budget"] = fr["cost_per_task"] > task.max_cost_per_task
        drop = np.zeros(len(self.fs.records), dtype=bool)
        reasons = {}
        for name, failed in checks.items():
            reasons[name] = int((failed & ~drop).sum())
            drop |= failed
        return np.where(~drop)[0], reasons

    def retrieve(self, task: TaskSpec, n: int, cluster_probe: int | None = None,
                 scope: np.ndarray | None = None, capability_index: bool = False,
                 max_candidates: int = 2000
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        q = self.fs.embedder.transform([task.description + " " +
                                        " ".join(task.required_capabilities)])[0]
        scope = np.arange(len(self.fs.records)) if scope is None else scope
        if cluster_probe and self.clusters is not None:
            sims = {c: float(v @ q) for c, v in self.cluster_centroids.items()}
            top = sorted(sims, key=sims.get, reverse=True)[:cluster_probe]
            scope = scope[np.isin(self.clusters.labels[scope], top)]
        sem = self.fs.embeddings[scope] @ q
        cap = self.capability_match(task.required_capabilities)[scope]
        hybrid = 0.6 * sem + 0.4 * cap
        order = np.argsort(-hybrid)
        top = order[:n]
        if capability_index and task.required_capabilities:
            exact = order[cap[order] >= 1.0][:max_candidates]
            top = np.concatenate([top, np.setdiff1d(exact, top, assume_unique=True)])
        return scope[top], sem[top], cap[top], len(scope)

    def select_for_task(self, task: TaskSpec, cfg: TaskConfig | None = None) -> Selection:
        cfg = cfg or TaskConfig()
        fr = self.fs.frame
        scope, reasons = self.prefilter(task, cfg)
        funnel = [{"stage": "registry", "count": len(self.fs.records)},
                  {"stage": "policy_eligible", "count": len(scope), "dropped": reasons}]
        idx, sem, cap, scanned = self.retrieve(task, cfg.retrieve_n, cfg.cluster_probe, scope,
                                                cfg.capability_index, cfg.max_candidates)
        funnel += [{"stage": "scanned", "count": scanned},
                   {"stage": "retrieved", "count": len(idx)}]
        idx_all = idx
        keep = cap >= 1.0  # mandatory capabilities are a hard constraint
        funnel.append({"stage": "capable", "count": int(keep.sum()),
                       "dropped": {"missing_required_capability": int((~keep).sum())}})
        idx, sem, cap = idx[keep], sem[keep], cap[keep]
        if len(idx) == 0:
            return Selection("task", [], [], funnel, {"task": task.task_id})

        if task.category:
            mean, lb, runs = (a[idx] for a in self.fs.category_quality(task.category))
        else:
            mean, lb, runs = fr["quality"][idx], fr["quality_lb"][idx], fr["runs"][idx]
        # Every component is rank-normalised within the candidate set so the weights mean
        # what they say; raw posteriors live in ~[0.55, 0.85] and would be swamped.
        Q = lb if cfg.use_lower_bound else mean
        sem_n = (sem - sem.min()) / (np.ptp(sem) or 1)
        parts = {
            "Q": _pct_rank(Q),
            "C": 0.5 * sem_n + 0.5 * cap,
            "R": _pct_rank(fr["error_rate"][idx], higher_is_better=False),
            "L": _pct_rank(fr["p95_latency_ms"][idx], higher_is_better=False),
            "K": _pct_rank(fr["cost_per_task"][idx], higher_is_better=False),
        }
        score = sum(cfg.weights[k] * v for k, v in parts.items())
        order = mmr(score, self.diversity_space[idx], cfg.k, cfg.mmr_lambda)
        funnel.append({"stage": "selected", "count": len(order)})

        rows = []
        for rank, j in enumerate(order, 1):
            r = self.fs.records[idx[j]]
            rows.append({
                "rank": rank, "agent_id": r.agent_id, "name": r.name,
                "score": round(float(score[j]), 4),
                **{k: round(float(v[j]), 3) for k, v in parts.items()},
                "quality_lb": round(float(lb[j]), 3), "quality_mean": round(float(mean[j]), 3),
                "runs": int(runs[j]), "status": r.status.name,
                "cluster": self._cluster_of(idx[j]),
            })
        return Selection("task", [r["agent_id"] for r in rows], rows, funnel,
                         {"task": task.task_id, "weights": cfg.weights,
                          "retrieved": idx_all.tolist()})

    def _cluster_of(self, i: int) -> int | None:
        return int(self.clusters.labels[i]) if self.clusters is not None else None

    # -------------------------------------------------------- representative mode
    def select_representative(self, k: int = 10, outlier_slots: int = 1,
                              min_status: Status = Status.DISCOVERED) -> Selection:
        pool = np.where(self.fs.frame["status"] >= min_status)[0]
        X = self.diversity_space[pool]
        res = kmedoids(X, k - outlier_slots)
        chosen = list(res.representatives)
        sizes = np.bincount(res.labels, minlength=k - outlier_slots)
        why = [f"medoid of cluster covering {s} agents" for s in sizes]
        if outlier_slots:
            noise = np.where(hdbscan(X).labels == -1)[0]
            for _ in range(min(outlier_slots, len(noise))):
                # farthest-point: the outlier least like anything already chosen
                d = np.linalg.norm(X[noise][:, None] - X[chosen][None], axis=2).min(1)
                pick = int(noise[d.argmax()])
                chosen.append(pick)
                why.append("most atypical HDBSCAN outlier")
        dist = np.linalg.norm(X[:, None] - X[chosen][None], axis=2).min(1)
        rows = [{"rank": n + 1, "agent_id": self.fs.records[pool[i]].agent_id,
                 "name": self.fs.records[pool[i]].name, "why": why[n]}
                for n, i in enumerate(chosen)]
        return Selection("representative", [r["agent_id"] for r in rows], rows,
                         meta={"mean_distance_to_nearest_rep": round(float(dist.mean()), 4),
                               "pool": len(pool)})

    # -------------------------------------------------------------- security mode
    def security_priority(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        risk_raw = self.fs.frame["risk_raw"]
        z = (risk_raw - risk_raw.mean(0)) / (risk_raw.std(0) + 1e-9)
        exposure = z @ RISK_WEIGHTS
        X = np.hstack([self.fs.views["risk"], self.fs.views["behavior"]])
        anomaly = -IsolationForest(n_estimators=200, random_state=0).fit(X).score_samples(X)
        priority = 0.7 * _pct_rank(exposure) + 0.3 * _pct_rank(anomaly)
        return priority, exposure, anomaly

    def select_for_security(self, k: int = 10, mmr_lambda: float = 0.85,
                            shortlist: int = 200) -> Selection:
        priority, exposure, anomaly = self.security_priority()
        top = np.argsort(-priority)[:shortlist]
        order = mmr(priority[top], self.fs.views["capability"][top], k, mmr_lambda)
        rows = []
        for rank, j in enumerate(order, 1):
            i = top[j]
            r = self.fs.records[i]
            rows.append({"rank": rank, "agent_id": r.agent_id, "name": r.name,
                         "priority": round(float(priority[i]), 4),
                         "exposure": round(float(exposure[i]), 2),
                         "anomaly": round(float(anomaly[i]), 3), "status": r.status.name,
                         "flags": r.policy_notes[:3]})
        return Selection("security", [r["agent_id"] for r in rows], rows,
                         meta={"population": len(self.fs.records)})
