"""Compare clustering algorithms with internal and (synthetic) external metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from agentatlas.ml.clustering.algorithms import ALGORITHMS, ClusterResult


def score(X: np.ndarray, res: ClusterResult, truth: np.ndarray | None = None,
          outliers: np.ndarray | None = None, seed: int = 0) -> dict:
    labels = res.labels
    mask = labels != -1
    out = {"algorithm": res.algorithm, "clusters": res.n_clusters,
           "seconds": round(res.seconds, 2), "noise_fraction": round(float((~mask).mean()), 3)}
    if res.n_clusters >= 2:
        n = min(3000, int(mask.sum()))
        out["silhouette"] = round(float(silhouette_score(
            X[mask], labels[mask], sample_size=n, random_state=seed)), 3)
        out["davies_bouldin"] = round(float(davies_bouldin_score(X[mask], labels[mask])), 3)
    if truth is not None:
        out["ari"] = round(float(adjusted_rand_score(truth, labels)), 3)
        out["nmi"] = round(float(normalized_mutual_info_score(truth, labels)), 3)
        if outliers is not None and outliers.any():
            normal = ~outliers
            out["ari_typical_only"] = round(float(adjusted_rand_score(
                truth[normal], labels[normal])), 3)
    if outliers is not None and res.algorithm == "hdbscan":
        noise = labels == -1
        out["outlier_recall_in_noise"] = round(float(noise[outliers].mean()), 3)
        out["normal_in_noise"] = round(float(noise[~outliers].mean()), 3)
    return out


def compare(X: np.ndarray, k: int, truth: np.ndarray | None = None,
            outliers: np.ndarray | None = None,
            algorithms: tuple[str, ...] = tuple(ALGORITHMS)) -> tuple[list[dict], dict]:
    rows, results = [], {}
    for name in algorithms:
        res = ALGORITHMS[name](X, k)
        results[name] = res
        rows.append(score(X, res, truth, outliers))
    return rows, results
