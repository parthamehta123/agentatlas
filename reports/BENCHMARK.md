# AgentAtlas benchmark (synthetic estate)

Total runtime: 18.6s

## 1. Discovery (adapter ablation)

| variant | recall | precision | duplicates | verified_share | reachable_share | authorized_share | shadow_agents_found |
|---|---|---|---|---|---|---|---|
| all adapters | 0.9984 | 1.0 | 0 | 0.94 | 0.593 | 0.525 | 2086 |
| without A2A | 0.9979 | 1.0 | 0 | 0.932 | 0.541 | 0.5 | 2086 |
| A2A only | 0.1161 | 1.0 | 0 | 0.0 | 0.0 | 0.0 | 0 |
| without registration | 0.9961 | 1.0 | 0 | 0.876 | 0.116 | 0.055 | 2086 |
| without kubernetes | 0.9976 | 1.0 | 0 | 0.806 | 0.588 | 0.52 | 2078 |
| without cloud | 0.9979 | 1.0 | 0 | 0.912 | 0.591 | 0.523 | 2081 |
| without cmdb | 0.9972 | 1.0 | 0 | 0.886 | 0.593 | 0.5 | 2086 |
| without gateway | 0.9825 | 1.0 | 0 | 0.916 | 0.578 | 0.511 | 1927 |
| without otel | 0.9874 | 1.0 | 0 | 0.926 | 0.585 | 0.551 | 1976 |

## 2. Capability inference for undeclared agents

| agents_without_declared_capabilities | precision | recall |
|---|---|---|
| 4031 | 0.9 | 0.949 |

## 3. Clustering (k=12, latent groups=12)

### Capability view

| algorithm | clusters | seconds | noise_fraction | silhouette | davies_bouldin | ari | nmi | ari_typical_only | outlier_recall_in_noise | normal_in_noise |
|---|---|---|---|---|---|---|---|---|---|---|
| kmeans | 12 | 0.13 | 0.0 | 0.495 | 0.917 | 0.979 | 0.984 | 0.982 |  |  |
| minibatch_kmeans | 12 | 0.07 | 0.0 | 0.445 | 1.141 | 0.87 | 0.948 | 0.871 |  |  |
| kmedoids_clara | 12 | 0.15 | 0.0 | 0.451 | 1.102 | 0.882 | 0.928 | 0.886 |  |  |
| hdbscan | 63 | 0.29 | 0.136 | 0.275 | 1.761 | 0.273 | 0.679 | 0.364 | 0.957 | 0.048 |
| hierarchical | 12 | 0.67 | 0.0 | 0.494 | 0.913 | 0.975 | 0.978 | 0.981 |  |  |

### Fused capability + behavior view

| algorithm | clusters | seconds | noise_fraction | silhouette | davies_bouldin | ari | nmi | ari_typical_only | outlier_recall_in_noise | normal_in_noise |
|---|---|---|---|---|---|---|---|---|---|---|
| kmeans | 12 | 0.08 | 0.0 | 0.403 | 1.108 | 0.979 | 0.984 | 0.981 |  |  |
| minibatch_kmeans | 12 | 0.07 | 0.0 | 0.37 | 1.318 | 0.906 | 0.942 | 0.91 |  |  |
| kmedoids_clara | 12 | 0.21 | 0.0 | 0.386 | 1.275 | 0.887 | 0.93 | 0.905 |  |  |
| hdbscan | 39 | 0.33 | 0.107 | 0.308 | 1.237 | 0.596 | 0.78 | 0.713 | 0.803 | 0.033 |
| hierarchical | 12 | 0.6 | 0.0 | 0.355 | 1.566 | 0.786 | 0.854 | 0.824 |  |  |

## 4. Task-aware top-10 selection (mean over 25 tasks)

| method | ndcg@10 | precision@10 | capable@10 | true_quality | unauthorized_rate | vulnerable_picked | lucky_picked | redundancy | ms | needle_rank | oracle_top10_recall | scanned_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.022 | 0.032 | 0.096 | 0.053 | 0.48 | 0.32 | 0.0 | 0.079 | 0.0 |  |  |  |
| vector_only | 0.375 | 0.512 | 0.832 | 0.531 | 0.392 | 0.32 | 0.0 | 0.689 | 0.0 | 3 |  |  |
| metadata_only | 0.64 | 0.88 | 1.0 | 0.587 | 0.0 | 0.08 | 0.04 | 0.551 | 0.0 | 1 |  |  |
| performance_only | 0.475 | 0.524 | 0.972 | 0.763 | 0.408 | 0.16 | 0.4 | 0.689 | 0.0 | 1 |  |  |
| hybrid_vector_top100_only | 0.812 | 0.964 | 1.0 | 0.747 | 0.0 | 0.0 | 0.0 | 0.73 | 11.742 | 1 | 0.272 | 0.345 |
| hybrid_no_shrinkage | 0.777 | 0.912 | 1.0 | 0.716 | 0.0 | 0.0 | 0.04 | 0.543 | 12.302 | 1 | 1.0 | 0.345 |
| hybrid_no_mmr | 0.864 | 0.988 | 1.0 | 0.793 | 0.0 | 0.04 | 0.0 | 0.786 | 12.418 | 1 | 1.0 | 0.345 |
| agentatlas_hybrid | 0.817 | 0.952 | 1.0 | 0.752 | 0.0 | 0.0 | 0.0 | 0.575 | 12.055 | 1 | 1.0 | 0.345 |
| agentatlas_hybrid_ivf3 | 0.819 | 0.948 | 1.0 | 0.755 | 0.0 | 0.0 | 0.0 | 0.643 | 11.895 | 1 | 0.96 | 0.091 |

## 5. Representative top-10

| method | groups_covered | mean_dist_to_nearest_rep | p95_dist_to_nearest_rep | outliers_included |
|---|---|---|---|---|
| random (mean of 20) | 7.0 | 1.0586 | 1.4936 | 1.25 |
| kmeans_nearest_centroid | 10 | 0.7816 | 1.3871 | 0 |
| kmedoids | 10 | 0.7754 | 1.3836 | 0 |
| kmedoids + 1 outlier slot | 10 | 0.8324 | 1.4612 | 1 |

## 6. Security-priority top-10

| method | precision@10 | precision@50 | r_precision | groups_in_top10 |
|---|---|---|---|---|
| random | 0.0 | 0.0 | 0.03 | 8 |
| unverified agents, random order | 0.0 | 0.1 | 0.09 | 6 |
| anomaly only (IsolationForest) | 0.8 | 0.58 | 0.295 | 5 |
| exposure only | 1.0 | 1.0 | 0.51 | 6 |
| exposure+anomaly | 1.0 | 0.9 | 0.455 | 6 |
| agentatlas (exposure+anomaly+MMR) | 1.0 | 0.78 |  | 10 |
| base rate (200 vulnerable) | 0.02 |  |  |  |
