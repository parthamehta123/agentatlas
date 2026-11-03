"""Four clustering algorithms behind one interface, all scalable to 10k+ agents.

* K-Means / MiniBatchKMeans - fast baseline; centroids are synthetic points.
* K-Medoids (CLARA)          - representatives are real agents; PAM on samples so we never
                               build a 10k x 10k distance matrix.
* HDBSCAN                    - variable density, no forced k, labels outliers (-1).
* Hierarchical (two-stage)   - micro-cluster with MiniBatchKMeans, then Ward linkage on
                               the micro-centroids -> a browsable taxonomy at any depth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin


@dataclass
class ClusterResult:
    algorithm: str
    labels: np.ndarray
    representatives: np.ndarray  # one real agent index per cluster, ordered by label
    seconds: float
    extra: dict = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        return len(set(self.labels.tolist()) - {-1})


def _nearest_members(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    reps = []
    for k, c in enumerate(centers):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            reps.append(int(pairwise_distances_argmin(c[None], X)[0]))
            continue
        reps.append(int(idx[np.argmin(((X[idx] - c) ** 2).sum(1))]))
    return np.array(reps)


def kmeans(X: np.ndarray, k: int, minibatch: bool = False, seed: int = 0) -> ClusterResult:
    t = time.perf_counter()
    model = (MiniBatchKMeans(k, random_state=seed, n_init=5, batch_size=2048) if minibatch
             else KMeans(k, random_state=seed, n_init=5))
    labels = model.fit_predict(X)
    reps = _nearest_members(X, labels, model.cluster_centers_)
    return ClusterResult("minibatch_kmeans" if minibatch else "kmeans", labels, reps,
                         time.perf_counter() - t, {"inertia": float(model.inertia_)})


def _pam_alternate(D: np.ndarray, k: int, rng, iters: int = 30) -> np.ndarray:
    """k-medoids++ init then alternating (Voronoi) medoid updates on a distance matrix."""
    n = D.shape[0]
    medoids = [int(rng.integers(n))]
    for _ in range(1, k):
        d = D[:, medoids].min(1) ** 2
        medoids.append(int(rng.choice(n, p=d / d.sum())))
    medoids = np.array(medoids)
    for _ in range(iters):
        assign = D[:, medoids].argmin(1)
        new = medoids.copy()
        for j in range(k):
            members = np.where(assign == j)[0]
            if len(members):
                new[j] = members[D[np.ix_(members, members)].sum(1).argmin()]
        if np.array_equal(new, medoids):
            break
        medoids = new
    return medoids


def _refine_medoids(X: np.ndarray, medoids: np.ndarray, rng, iters: int = 5,
                    candidates: int = 400) -> tuple[np.ndarray, float]:
    """Alternating updates on the *full* data, scoring sampled medoid candidates per
    cluster against all of its members (memory stays O(candidates x cluster size))."""
    medoids = medoids.copy()
    for _ in range(iters):
        labels = pairwise_distances(X, X[medoids]).argmin(1)
        changed = False
        for j in range(len(medoids)):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                continue
            cand = members if len(members) <= candidates else rng.choice(
                members, candidates, replace=False)
            cand = np.union1d(cand, [medoids[j]])
            best = cand[pairwise_distances(X[cand], X[members]).sum(1).argmin()]
            if best != medoids[j]:
                medoids[j], changed = best, True
        if not changed:
            break
    return medoids, float(pairwise_distances(X, X[medoids]).min(1).sum())


def kmedoids(X: np.ndarray, k: int, sample_size: int = 2000, n_samples: int = 5,
             seed: int = 0) -> ClusterResult:
    """CLARA: PAM on several random samples, keep the medoids with lowest full cost."""
    t = time.perf_counter()
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for _ in range(n_samples):
        idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
        med = idx[_pam_alternate(pairwise_distances(X[idx]), k, rng)]
        cost = pairwise_distances(X, X[med]).min(1).sum()
        if cost < best_cost:
            best, best_cost = med, cost
    best, best_cost = _refine_medoids(X, best, rng)
    labels = pairwise_distances(X, X[best]).argmin(1)
    return ClusterResult("kmedoids_clara", labels, best, time.perf_counter() - t,
                         {"cost": float(best_cost)})


def hdbscan(X: np.ndarray, min_cluster_size: int = 40, pca_dims: int = 16,
            seed: int = 0) -> ClusterResult:
    t = time.perf_counter()
    Z = PCA(pca_dims, random_state=seed).fit_transform(X) if X.shape[1] > pca_dims else X
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10, store_centers="medoid",
                    copy=True)
    labels = model.fit_predict(Z)
    reps = _nearest_members(Z, labels, model.medoids_) if len(model.medoids_) else np.array([])
    return ClusterResult("hdbscan", labels, reps, time.perf_counter() - t,
                         {"noise_fraction": float((labels == -1).mean()),
                          "outlier_score": 1 - model.probabilities_})


def hierarchical(X: np.ndarray, k: int, micro_clusters: int = 300,
                 taxonomy_levels: tuple[int, ...] = (4, 12, 40), seed: int = 0
                 ) -> ClusterResult:
    t = time.perf_counter()
    micro = MiniBatchKMeans(micro_clusters, random_state=seed, n_init=3, batch_size=2048)
    micro_labels = micro.fit_predict(X)
    C = micro.cluster_centers_

    def cut(n: int) -> np.ndarray:
        return AgglomerativeClustering(n_clusters=n, linkage="ward").fit_predict(C)[micro_labels]

    labels = cut(k)
    centers = np.array([X[labels == j].mean(0) for j in range(k)])
    taxonomy = {n: cut(n) for n in taxonomy_levels}
    return ClusterResult("hierarchical", labels, _nearest_members(X, labels, centers),
                         time.perf_counter() - t, {"taxonomy": taxonomy})


ALGORITHMS = {
    "kmeans": lambda X, k: kmeans(X, k),
    "minibatch_kmeans": lambda X, k: kmeans(X, k, minibatch=True),
    "kmedoids": lambda X, k: kmedoids(X, k),
    "hdbscan": lambda X, k: hdbscan(X),
    "hierarchical": lambda X, k: hierarchical(X, k),
}
