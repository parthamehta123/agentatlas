"""End-to-end benchmark against synthetic ground truth.

Ground truth is only read here. Gains for task selection are defined so that a
selection is only worth something if the platform could actually run it:

    gain(a, t) = true_quality(a, category)  if a truly has the required capability,
                                             its clearance covers the task data, and
                                             the registry has it AUTHORIZED
               = 0                           otherwise
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from agentatlas.discovery.adapters import adapters_from_dir
from agentatlas.discovery.pipeline import run_discovery
from agentatlas.ml.clustering.algorithms import kmeans
from agentatlas.ml.clustering.compare import compare
from agentatlas.ml.features import FeatureStore, build_features
from agentatlas.registry.models import AgentRecord, Kind, Status
from agentatlas.selection.engine import Selector, TaskConfig, TaskSpec
from agentatlas.synthetic.archetypes import CLASSIFICATION_ORDER

K = 10


@dataclass
class Truth:
    agents: dict[str, dict]
    by_slug: dict[str, dict]
    tasks: list[dict]

    @classmethod
    def load(cls, data: Path) -> Truth:
        agents = {a["agent_id"]: a for a in json.loads((data / "truth/agents.json").read_text())}
        return cls(agents, {a["slug"]: a for a in agents.values()},
                   json.loads((data / "truth/tasks.json").read_text()))

    def of(self, r: AgentRecord) -> dict | None:
        return self.agents.get(r.agent_id) or self.by_slug.get(r.slug)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discovery_metrics(records: list[AgentRecord], truth: Truth) -> dict:
    agents = [r for r in records if r.kind == Kind.AGENT]
    matched = [truth.of(r) for r in agents]
    found = {t["agent_id"] for t in matched if t}
    false_pos = sum(1 for t in matched if t is None)
    dupes = sum(1 for t in matched if t) - len(found)
    n_true = len(truth.agents)
    status = np.array([int(r.status) for r in agents])
    return {
        "true_agents": n_true, "agents_found": len(agents),
        "recall": round(len(found) / n_true, 4),
        "precision": round((len(agents) - false_pos) / max(len(agents), 1), 4),
        "duplicates": dupes,
        "non_agents_rejected": sum(1 for r in records if r.kind != Kind.AGENT),
        "verified_share": round(float((status >= Status.VERIFIED).mean()), 3),
        "reachable_share": round(float((status >= Status.REACHABLE).mean()), 3),
        "authorized_share": round(float((status >= Status.AUTHORIZED).mean()), 3),
        "inferred_from_behaviour": sum(r.inferred for r in agents),
        "shadow_agents_found": sum(1 for r, t in zip(agents, matched) if t and not set(
            t["sources"]) & {"registration", "a2a", "cmdb"}),
    }


def discovery_ablation(infra: Path, truth: Truth) -> list[dict]:
    full = adapters_from_dir(infra)
    rows = []
    variants = [("all adapters", full), ("without A2A", [a for a in full if a.name != "a2a"]),
                ("A2A only", [a for a in full if a.name == "a2a"])]
    for a in full:
        if a.name != "a2a":
            variants.append((f"without {a.name}", [x for x in full if x is not a]))
    for label, adapters in variants:
        rep = run_discovery(adapters)
        rows.append({"variant": label, **discovery_metrics(rep.records, truth),
                     "seconds": rep.seconds})
    return rows


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------

def _gain_vector(fs: FeatureStore, truth: Truth, task: dict) -> np.ndarray:
    need = CLASSIFICATION_ORDER.index(task["data_classification"])
    gains = np.zeros(len(fs.records))
    for i, r in enumerate(fs.records):
        t = truth.of(r)
        if (t and set(task["required_capabilities"]) <= set(t["capabilities"])
                and CLASSIFICATION_ORDER.index(t["data_clearance"]) >= need
                and r.status >= Status.AUTHORIZED):
            gains[i] = t["true_quality"].get(task["category"], 0.0)
    return gains


def _ndcg(picked: list[int], gains: np.ndarray, k: int = K) -> float:
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = sum(gains[i] * disc[n] for n, i in enumerate(picked[:k]))
    ideal = np.sort(gains)[::-1][:k]
    idcg = float((ideal * disc[:len(ideal)]).sum())
    return dcg / idcg if idcg else 0.0


def _baselines(sel: Selector, fs: FeatureStore, task: TaskSpec, rng) -> dict[str, list[int]]:
    n = len(fs.records)
    fr = fs.frame
    out: dict[str, list[int]] = {"random": list(rng.choice(n, K, replace=False))}
    q = fs.embedder.transform([task.description + " " + " ".join(task.required_capabilities)])
    out["vector_only"] = list(np.argsort(-(fs.embeddings @ q[0]))[:K])
    need = CLASSIFICATION_ORDER.index(task.data_classification)
    eligible = np.where((sel.capability_match(task.required_capabilities) >= 1)
                        & (fr["status"] >= Status.AUTHORIZED) & (fr["clearance"] >= need))[0]
    out["metadata_only"] = (list(rng.choice(eligible, K, replace=False))
                            if len(eligible) >= K else list(eligible))
    s = np.array([r.telemetry[task.category].successes if task.category in r.telemetry else 0
                  for r in fs.records], float)
    runs = np.array([r.telemetry[task.category].runs if task.category in r.telemetry else 0
                     for r in fs.records], float)
    raw = np.where(runs > 0, s / np.maximum(runs, 1), -1)
    out["performance_only"] = list(np.lexsort((-runs, -raw))[:K])
    return out


def task_benchmark(fs: FeatureStore, sel: Selector, truth: Truth, seed: int = 0
                   ) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    V = normalize(sel.diversity_space)
    methods: dict[str, list[dict]] = {}
    needle_ranks: dict[str, int | None] = {}

    def record(method: str, picked: list[int], task: dict, gains: np.ndarray, secs: float,
               retrieved: np.ndarray | None = None):
        picked = [int(i) for i in picked]
        ts = [truth.of(fs.records[i]) for i in picked]
        capable = [bool(t and set(task["required_capabilities"]) <= set(t["capabilities"]))
                   for t in ts]
        sims = V[picked] @ V[picked].T if len(picked) > 1 else np.zeros((1, 1))
        row = {
            "ndcg@10": _ndcg(picked, gains),
            "precision@10": float(np.mean(gains[picked] > 0)) if picked else 0.0,
            "capable@10": float(np.mean(capable)) if picked else 0.0,
            "true_quality": float(np.mean([t["true_quality"].get(task["category"], 0)
                                           if t else 0 for t in ts])) if picked else 0.0,
            "unauthorized_rate": float(np.mean([fs.records[i].status < Status.AUTHORIZED
                                                for i in picked])) if picked else 0.0,
            "vulnerable_picked": sum(bool(t and t["vulnerable"]) for t in ts),
            "lucky_picked": sum(bool(t and t["is_lucky"]) for t in ts),
            "redundancy": float(sims[np.triu_indices(len(picked), 1)].mean())
            if len(picked) > 1 else 0.0,
            "ms": secs * 1000,
        }
        if retrieved is not None:
            ideal = set(np.argsort(-gains)[:K][gains[np.argsort(-gains)[:K]] > 0].tolist())
            row["oracle_top10_recall"] = (len(ideal & set(retrieved.tolist())) / len(ideal)
                                              if ideal else 1.0)
        methods.setdefault(method, []).append(row)
        if task["category"] == "sanctions_screening":
            ranks = [n + 1 for n, t in enumerate(ts) if t and t["is_needle"]]
            needle_ranks[method] = ranks[0] if ranks else None

    for task in truth.tasks:
        spec = TaskSpec(description=task["description"], task_id=task["task_id"],
                        category=task["category"],
                        required_capabilities=task["required_capabilities"],
                        data_classification=task["data_classification"])
        gains = _gain_vector(fs, truth, task)
        for name, picked in _baselines(sel, fs, spec, rng).items():
            record(name, picked, task, gains, 0.0)
        for name, cfg in (("hybrid_vector_top100_only", TaskConfig(capability_index=False)),
                          ("hybrid_no_shrinkage", TaskConfig(use_lower_bound=False)),
                          ("hybrid_no_mmr", TaskConfig(mmr_lambda=1.0)),
                          ("agentatlas_hybrid", TaskConfig()),
                          ("agentatlas_hybrid_ivf3", TaskConfig(cluster_probe=3))):
            t0 = time.perf_counter()
            res = sel.select_for_task(spec, cfg)
            secs = time.perf_counter() - t0
            record(name, [fs.index[a] for a in res.agent_ids], task, gains, secs,
                   np.array(res.meta.get("retrieved", []), dtype=int))
            methods[name][-1]["scanned_frac"] = next(
                f["count"] for f in res.funnel if f["stage"] == "scanned") / len(fs.records)

    summary = []
    for name, rows in methods.items():
        agg = {"method": name}
        for key in dict.fromkeys(k for r in rows for k in r):
            agg[key] = round(float(np.mean([r[key] for r in rows])), 3)
        agg["needle_rank"] = needle_ranks.get(name)
        summary.append(agg)
    return summary, needle_ranks


# ---------------------------------------------------------------------------
# Representative & security selection
# ---------------------------------------------------------------------------

def representative_benchmark(fs: FeatureStore, sel: Selector, truth: Truth,
                             seed: int = 0) -> list[dict]:
    X = sel.diversity_space
    rng = np.random.default_rng(seed)
    groups = np.array([truth.of(r)["group"] for r in fs.records])
    outl = np.array([truth.of(r)["is_outlier"] for r in fs.records])

    def evaluate(name: str, picked) -> dict:
        picked = [int(i) for i in picked]
        d = np.linalg.norm(X[:, None] - X[picked][None], axis=2).min(1)
        return {"method": name, "groups_covered": len(set(groups[picked])),
                "mean_dist_to_nearest_rep": round(float(d.mean()), 4),
                "p95_dist_to_nearest_rep": round(float(np.quantile(d, 0.95)), 4),
                "outliers_included": int(outl[picked].sum())}

    rand = [evaluate("random", rng.choice(len(X), K, replace=False)) for _ in range(20)]
    rows = [{"method": "random (mean of 20)",
             **{k: round(float(np.mean([r[k] for r in rand])), 4)
                for k in rand[0] if k != "method"}}]
    rows.append(evaluate("kmeans_nearest_centroid", kmeans(X, K).representatives))
    for slots, label in ((0, "kmedoids"), (1, "kmedoids + 1 outlier slot")):
        res = sel.select_representative(K, outlier_slots=slots)
        rows.append(evaluate(label, [fs.index[a] for a in res.agent_ids]))
    return rows


def security_benchmark(fs: FeatureStore, sel: Selector, truth: Truth, seed: int = 0
                       ) -> list[dict]:
    rng = np.random.default_rng(seed)
    vuln = np.array([truth.of(r)["vulnerable"] for r in fs.records])
    groups = np.array([truth.of(r)["group"] for r in fs.records])
    status = fs.frame["status"]
    priority, exposure, anomaly = sel.security_priority()

    n_vuln = int(vuln.sum())

    def evaluate(name, ranking):
        ranking = [int(i) for i in ranking]
        hits = vuln[ranking]
        return {"method": name,
                "precision@10": round(float(hits[:K].mean()), 3),
                "precision@50": round(float(hits[:50].mean()), 3),
                "r_precision": round(float(hits[:n_vuln].mean()), 3)
                if len(ranking) >= n_vuln else None,
                "groups_in_top10": len(set(groups[ranking[:K]]))}

    unverified = np.where(status < Status.VERIFIED)[0]
    rows = [
        evaluate("random", rng.permutation(len(vuln))),
        evaluate("unverified agents, random order", rng.permutation(unverified)),
        evaluate("anomaly only (IsolationForest)", np.argsort(-anomaly)),
        evaluate("exposure only", np.argsort(-exposure)),
        evaluate("exposure+anomaly", np.argsort(-priority)),
    ]
    res = sel.select_for_security(50, shortlist=400)
    rows.append(evaluate("agentatlas (exposure+anomaly+MMR)",
                         [fs.index[a] for a in res.agent_ids]))
    rows.append({"method": f"base rate ({n_vuln} vulnerable)",
                 "precision@10": round(float(vuln.mean()), 3)})
    return rows


def capability_inference_metrics(fs: FeatureStore, truth: Truth) -> dict:
    tp = fp = fn = 0
    n = 0
    for r in fs.records:
        if r.capabilities:
            continue
        n += 1
        true = set(truth.of(r)["capabilities"])
        pred = set(r.inferred_capabilities)
        tp, fp, fn = tp + len(pred & true), fp + len(pred - true), fn + len(true - pred)
    return {"agents_without_declared_capabilities": n,
            "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)}


# ---------------------------------------------------------------------------

def run_benchmark(data: Path) -> dict:
    truth = Truth.load(data)
    t0 = time.perf_counter()
    disc = discovery_ablation(data / "infra", truth)
    rep = run_discovery(adapters_from_dir(data / "infra"))
    agents = [r for r in rep.records if r.kind == Kind.AGENT]
    fs = build_features(agents)
    groups = np.array([truth.of(r)["group"] for r in agents])
    outl = np.array([truth.of(r)["is_outlier"] for r in agents])
    clustering = {view: compare(X, 12, groups, outl)[0]
                  for view, X in (("capability", fs.views["capability"]),
                                  ("fused", fs.fused()))}
    clusters = kmeans(fs.views["capability"], 12)
    sel = Selector(fs, clusters)
    tasks, _ = task_benchmark(fs, sel, truth)
    return {
        "discovery": disc,
        "capability_inference": capability_inference_metrics(fs, truth),
        "clustering": clustering,
        "task_selection": tasks,
        "representative_selection": representative_benchmark(fs, sel, truth),
        "security_selection": security_benchmark(fs, sel, truth),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def to_markdown(res: dict) -> str:
    def table(rows: list[dict], cols: list[str] | None = None) -> str:
        cols = cols or list(dict.fromkeys(k for r in rows for k in r))
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols)
                + " |" for r in rows]
        return "\n".join(out)

    disc_cols = ["variant", "recall", "precision", "duplicates", "verified_share",
                 "reachable_share", "authorized_share", "shadow_agents_found"]
    return "\n\n".join([
        "# AgentAtlas benchmark (synthetic estate)",
        f"Total runtime: {res['seconds']}s",
        "## 1. Discovery (adapter ablation)", table(res["discovery"], disc_cols),
        "## 2. Capability inference for undeclared agents",
        table([res["capability_inference"]]),
        "## 3. Clustering (k=12, latent groups=12)",
        "### Capability view", table(res["clustering"]["capability"]),
        "### Fused capability + behavior view", table(res["clustering"]["fused"]),
        "## 4. Task-aware top-10 selection (mean over 25 tasks)",
        table(res["task_selection"]),
        "## 5. Representative top-10", table(res["representative_selection"]),
        "## 6. Security-priority top-10", table(res["security_selection"]),
    ]) + "\n"
