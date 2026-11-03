"""Feature engineering: AgentRecord -> separate capability / behavior / risk views.

Design notes
* The text embedding is pluggable (``Embedder``). The default TF-IDF + SVD embedder is
  dependency-free and deterministic; ``SentenceTransformerEmbedder`` is a drop-in.
* Success rates are noisy for agents with few runs, so quality is an empirical-Bayes
  Beta posterior (shrunk towards the population prior) with a lower credible bound.
  An agent with 3/3 successes must not outrank one with 9,400/10,000.
* Agents discovered only from behaviour have no declared capabilities; we infer them
  from their name and observed tool usage with a model trained on declared agents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, normalize

from agentatlas.registry.models import AgentRecord, Family, Status
from agentatlas.synthetic.archetypes import CLASSIFICATION_ORDER
from agentatlas.tool_catalog import BLOCKED_TOOLS, EXTERNAL_TOOLS, PII_TOOLS, WRITE_TOOLS


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> Embedder: ...
    def transform(self, texts: list[str]) -> np.ndarray: ...


class TfidfSvdEmbedder:
    def __init__(self, dim: int = 96, seed: int = 0):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                   stop_words="english")
        self.svd = TruncatedSVD(n_components=dim, random_state=seed)

    def fit(self, texts: list[str]) -> TfidfSvdEmbedder:
        self.svd.fit(self.vec.fit_transform(texts))
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        return normalize(self.svd.transform(self.vec.transform(texts)))


class SentenceTransformerEmbedder:
    """Optional: ``pip install sentence-transformers``."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy

        self.model = SentenceTransformer(model)

    def fit(self, texts: list[str]) -> SentenceTransformerEmbedder:
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _tokens(s: str) -> str:
    return re.sub(r"[_\-\d]+", " ", s).strip()


def agent_text(r: AgentRecord) -> str:
    caps = " ".join(_tokens(c) for c in r.all_capabilities())
    tools = " ".join(_tokens(t) for t in r.tools)
    return f"{_tokens(r.slug)}. {r.description} {caps} {caps} {tools}"


# ---------------------------------------------------------------------------
# Empirical-Bayes success-rate shrinkage
# ---------------------------------------------------------------------------

@dataclass
class BetaPrior:
    alpha: float
    beta: float

    @classmethod
    def fit(cls, successes: np.ndarray, runs: np.ndarray, min_runs: int = 30) -> BetaPrior:
        """Method of moments over well-observed cells."""
        mask = runs >= min_runs
        p = successes[mask] / runs[mask]
        m, v = float(p.mean()), float(p.var())
        common = max(m * (1 - m) / max(v, 1e-6) - 1, 2.0)
        return cls(alpha=m * common, beta=(1 - m) * common)

    def posterior_mean(self, s, n):
        return (s + self.alpha) / (n + self.alpha + self.beta)

    def lower_bound(self, s, n, q: float = 0.1):
        return stats.beta.ppf(q, s + self.alpha, n - s + self.beta)


# ---------------------------------------------------------------------------
# Capability inference for agents with no declared capabilities
# ---------------------------------------------------------------------------

class CapabilityInferer:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.vec = TfidfVectorizer(token_pattern=r"[a-z]{3,}|[a-z_]+_[a-z_]+")
        self.mlb = MultiLabelBinarizer()
        self.clf = OneVsRestClassifier(LogisticRegression(max_iter=500, C=4.0))

    @staticmethod
    def _text(r: AgentRecord) -> str:
        return f"{_tokens(r.slug)} {' '.join(r.tools)}"

    def fit(self, records: list[AgentRecord]) -> CapabilityInferer:
        train = [r for r in records if r.capabilities]
        y = self.mlb.fit_transform([r.capabilities for r in train])
        keep = y.sum(0) >= 2  # need positives to learn a label
        self.labels = self.mlb.classes_[keep]
        self.clf.fit(self.vec.fit_transform([self._text(r) for r in train]), y[:, keep])
        return self

    def apply(self, records: list[AgentRecord]) -> int:
        todo = [r for r in records if not r.capabilities]
        if not todo:
            return 0
        proba = self.clf.predict_proba(self.vec.transform([self._text(r) for r in todo]))
        for r, p in zip(todo, proba):
            r.inferred_capabilities = [str(self.labels[j]) for j in np.where(
                p >= self.threshold)[0]]
        return len(todo)


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------

@dataclass
class FeatureStore:
    records: list[AgentRecord]
    embedder: Embedder
    embeddings: np.ndarray
    capability_matrix: np.ndarray
    capability_names: list[str]
    views: dict[str, np.ndarray]
    frame: dict[str, np.ndarray]  # named per-agent scalar columns
    prior: BetaPrior
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.index = {r.agent_id: i for i, r in enumerate(self.records)}

    def fused(self, weights: dict[str, float] | None = None) -> np.ndarray:
        weights = weights or {"capability": 1.0, "behavior": 0.5}
        parts = []
        for name, w in weights.items():
            v = self.views[name]
            scale = np.sqrt((v ** 2).sum(1).mean()) or 1.0
            parts.append(w * v / scale)
        return np.hstack(parts)

    def category_quality(self, category: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior mean, lower bound and runs for one task category."""
        s = np.array([r.telemetry[category].successes if category in r.telemetry else 0
                      for r in self.records], dtype=float)
        n = np.array([r.telemetry[category].runs if category in r.telemetry else 0
                      for r in self.records], dtype=float)
        return self.prior.posterior_mean(s, n), self.prior.lower_bound(s, n), n


def build_features(records: list[AgentRecord], embedder: Embedder | None = None,
                   infer_capabilities: bool = True) -> FeatureStore:
    if infer_capabilities:
        CapabilityInferer().fit(records).apply(records)

    texts = [agent_text(r) for r in records]
    embedder = embedder or TfidfSvdEmbedder()
    emb = embedder.fit(texts).transform(texts)

    mlb = MultiLabelBinarizer()
    caps = mlb.fit_transform([r.all_capabilities() for r in records]).astype(float)

    # --- performance & operations -------------------------------------------------
    all_s = np.array([s.successes for r in records for s in r.telemetry.values()], float)
    all_n = np.array([s.runs for r in records for s in r.telemetry.values()], float)
    prior = BetaPrior.fit(all_s, all_n)
    runs = np.array([r.total_runs() for r in records], float)
    succ = np.array([sum(s.successes for s in r.telemetry.values()) for r in records], float)
    errs = np.array([sum(s.errors for s in r.telemetry.values()) for r in records], float)
    viol = np.array([sum(s.guardrail_violations for s in r.telemetry.values())
                     for r in records], float)
    retries = np.array([sum(s.retries for s in r.telemetry.values()) for r in records], float)
    cost = np.array([sum(s.cost_usd_total for s in r.telemetry.values()) for r in records])
    lat = np.array([max((s.p95_latency_ms or 0) for s in r.telemetry.values())
                    if r.telemetry else np.nan for r in records], float)
    has_tel = runs > 0
    lat_f = np.where(np.isnan(lat), np.nanmedian(lat), lat)
    cost_per = np.where(has_tel, cost / np.maximum(runs, 1), np.nan)
    cost_per = np.where(np.isnan(cost_per), np.nanmedian(cost_per), cost_per)
    quality = prior.posterior_mean(succ, runs)
    quality_lb = prior.lower_bound(succ, runs)
    error_rate = (errs + 1) / (runs + 50)  # shrunk towards ~2%
    viol_rate = (viol + 0.1) / (runs + 50)
    tool_calls = np.array([sum(r.tools_used.values()) for r in records], float)

    tool_vocab = sorted({t for r in records for t in r.tools_used})
    tool_dist = np.zeros((len(records), len(tool_vocab)))
    tpos = {t: j for j, t in enumerate(tool_vocab)}
    for i, r in enumerate(records):
        tot = sum(r.tools_used.values())
        for t, c in r.tools_used.items():
            tool_dist[i, tpos[t]] = c / tot if tot else 0

    behavior_num = np.column_stack([
        np.log1p(lat_f), np.log(cost_per + 1e-4), error_rate, retries / np.maximum(runs, 1),
        np.log1p(tool_calls / np.maximum(runs, 1)), np.log1p(runs), has_tel.astype(float),
    ])
    behavior = np.hstack([StandardScaler().fit_transform(behavior_num), tool_dist * 2.0])

    # --- risk ---------------------------------------------------------------------
    def count(r, s):
        return len(set(r.tools) & s)

    clearance = np.array([CLASSIFICATION_ORDER.index(r.data_clearance)
                          if r.data_clearance in CLASSIFICATION_ORDER else 2 for r in records])
    risk_num = np.column_stack([
        [count(r, BLOCKED_TOOLS) for r in records],
        [count(r, WRITE_TOOLS) for r in records],
        [count(r, PII_TOOLS) for r in records],
        [count(r, EXTERNAL_TOOLS) for r in records],
        viol_rate * 100,
        [float(r.status < Status.VERIFIED) for r in records],
        [float(not r.owner_team) for r in records],
        [float(Family.DECLARED not in r.families) for r in records],
        clearance,
        np.log1p([r.instances for r in records]),
        np.log1p([r.llm_requests for r in records]),
    ]).astype(float)
    risk = StandardScaler().fit_transform(risk_num)

    capability_view = np.hstack([emb, caps / np.sqrt(np.maximum(caps.sum(1, keepdims=True),
                                                                1))])
    frame = {
        "runs": runs, "quality": quality, "quality_lb": quality_lb, "error_rate": error_rate,
        "violation_rate": viol_rate, "p95_latency_ms": lat_f, "cost_per_task": cost_per,
        "has_telemetry": has_tel, "status": np.array([int(r.status) for r in records]),
        "clearance": clearance, "risk_raw": risk_num,
    }
    return FeatureStore(records=records, embedder=embedder, embeddings=emb,
                        capability_matrix=caps, capability_names=list(mlb.classes_),
                        views={"capability": capability_view, "behavior": behavior,
                               "risk": risk},
                        frame=frame, prior=prior)
