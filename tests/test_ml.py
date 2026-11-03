import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from agentatlas.ml.clustering.algorithms import ALGORITHMS
from agentatlas.ml.features import BetaPrior


def test_shrinkage_prefers_large_track_record_over_tiny_perfect_one():
    prior = BetaPrior(alpha=9.5, beta=4.8)
    lucky = prior.lower_bound(3, 3)
    proven = prior.lower_bound(9_400, 10_000)
    assert proven > lucky
    assert prior.posterior_mean(3, 3) < 1.0


@pytest.mark.parametrize("name", ["kmeans", "minibatch_kmeans", "kmedoids", "hierarchical"])
def test_k_algorithms_recover_blobs_with_real_representatives(name):
    X, y = make_blobs(n_samples=1500, centers=6, n_features=8, cluster_std=0.6,
                      random_state=1)
    res = ALGORITHMS[name](X, 6)
    assert adjusted_rand_score(y, res.labels) > 0.9
    assert len(res.representatives) == 6
    assert all(0 <= i < len(X) for i in res.representatives)  # real agents, not centroids


def test_hdbscan_flags_scattered_points_as_noise():
    X, _ = make_blobs(n_samples=1200, centers=4, n_features=8, cluster_std=0.5,
                      random_state=2)
    rng = np.random.default_rng(0)
    noise = rng.uniform(X.min(), X.max(), size=(60, 8))
    res = ALGORITHMS["hdbscan"](np.vstack([X, noise]), 4)
    assert (res.labels[-60:] == -1).mean() > 0.8
    assert (res.labels[:-60] == -1).mean() < 0.1


def test_capability_inference_on_shadow_agents(atlas, data_dir):
    from agentatlas.evaluation.benchmark import Truth, capability_inference_metrics

    m = capability_inference_metrics(atlas.features, Truth.load(data_dir))
    assert m["agents_without_declared_capabilities"] > 0
    assert m["precision"] > 0.8 and m["recall"] > 0.8
