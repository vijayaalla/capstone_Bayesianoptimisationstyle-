#!/usr/bin/env python3
"""Generate Week 11 BBO queries with a clustering-aware decision rule.

Approach:
- Tune SVR/MLP surrogate families using CV MAE.
- Fit a bootstrap ensemble for uncertainty estimates.
- Cluster each function's observed inputs to identify local regions.
- Rank clusters by local quality, centroid trend, uncertainty, and separation.
- Score candidates using prediction, uncertainty, novelty, and cluster-aware cues.
- Report which cluster each query targets and what distance cue guided it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from generate_week7_queries_tuned import (
    find_function_dirs,
    fit_ensemble,
    format_query,
    load_xy,
    tune_model_family,
)
from generate_week8_queries_llm_strategy import normalize
from generate_week10_queries_interpretable import local_sensitivity, top_dimensions


@dataclass(frozen=True)
class ClusterSummary:
    label: int
    size: int
    centroid: np.ndarray
    best_idx: int
    radius_p80: float
    quality: float
    centroid_pred: float
    centroid_unc: float
    centroid_gap: float
    score: float
    quality_norm: float
    pred_norm: float
    unc_norm: float
    gap_norm: float
    tight_norm: float


def ensemble_stats(models, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    preds = np.column_stack([m.predict(x) for m in models])
    return preds.mean(axis=1), preds.std(axis=1)


def choose_n_clusters(n_obs: int) -> int:
    if n_obs <= 12:
        return 2
    if n_obs <= 24:
        return 3
    if n_obs <= 36:
        return 4
    return 5


def summarize_clusters(
    x: np.ndarray,
    y: np.ndarray,
    models,
    seed: int,
) -> tuple[list[ClusterSummary], np.ndarray]:
    n_clusters = min(choose_n_clusters(x.shape[0]), x.shape[0])
    km = KMeans(n_clusters=n_clusters, n_init=12, random_state=seed)
    labels = km.fit_predict(x)
    centroids = km.cluster_centers_

    centroid_pred, centroid_unc = ensemble_stats(models, centroids)
    centroid_gap = np.full(n_clusters, 1.0, dtype=float)
    if n_clusters > 1:
        deltas = centroids[:, None, :] - centroids[None, :, :]
        dists = np.sqrt((deltas * deltas).sum(axis=2))
        dists[dists == 0.0] = np.inf
        centroid_gap = dists.min(axis=1)

    sizes: list[int] = []
    radii: list[float] = []
    qualities: list[float] = []
    best_indices: list[int] = []
    for label in range(n_clusters):
        idx = np.flatnonzero(labels == label)
        sizes.append(int(idx.size))
        pts = x[idx]
        local_d = np.sqrt(((pts - centroids[label]) ** 2).sum(axis=1))
        radii.append(float(np.percentile(local_d, 80)) if idx.size > 1 else 0.02)

        cluster_y = y[idx]
        top_take = min(3, idx.size)
        top_vals = np.sort(cluster_y)[-top_take:]
        qualities.append(float(0.65 * np.max(cluster_y) + 0.35 * np.mean(top_vals)))
        best_indices.append(int(idx[np.argmax(cluster_y)]))

    quality_norm = normalize(np.asarray(qualities, dtype=float))
    pred_norm = normalize(centroid_pred)
    unc_norm = normalize(centroid_unc)
    gap_norm = normalize(centroid_gap)
    tight_norm = normalize(-np.asarray(radii, dtype=float))

    cluster_scores = (
        0.34 * quality_norm
        + 0.24 * pred_norm
        + 0.18 * unc_norm
        + 0.14 * gap_norm
        + 0.10 * tight_norm
    )

    summaries: list[ClusterSummary] = []
    for label in range(n_clusters):
        summaries.append(
            ClusterSummary(
                label=label,
                size=sizes[label],
                centroid=centroids[label],
                best_idx=best_indices[label],
                radius_p80=radii[label],
                quality=qualities[label],
                centroid_pred=float(centroid_pred[label]),
                centroid_unc=float(centroid_unc[label]),
                centroid_gap=float(centroid_gap[label]),
                score=float(cluster_scores[label]),
                quality_norm=float(quality_norm[label]),
                pred_norm=float(pred_norm[label]),
                unc_norm=float(unc_norm[label]),
                gap_norm=float(gap_norm[label]),
                tight_norm=float(tight_norm[label]),
            )
        )

    return summaries, labels


def choose_target_cluster(clusters: list[ClusterSummary]) -> ClusterSummary:
    return max(clusters, key=lambda c: c.score)


def nearest_cluster(target: ClusterSummary, clusters: list[ClusterSummary]) -> ClusterSummary:
    others = [c for c in clusters if c.label != target.label]
    if not others:
        return target
    return min(
        others,
        key=lambda c: float(np.linalg.norm(c.centroid - target.centroid)),
    )


def choose_cue(target: ClusterSummary) -> str:
    if (
        target.quality_norm >= 0.55
        and target.gap_norm >= 0.45
        and (target.tight_norm >= 0.35 or target.pred_norm >= 0.70)
    ):
        return "boundary_tightening"
    if target.unc_norm > target.quality_norm + 0.10:
        return "bridge_probe"
    return "centroid_trend"


def base_candidate_pool(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    dim = x.shape[1]
    n_obs = x.shape[0]

    n_global = 28_000 if dim <= 4 else 42_000
    n_local = 7_000 if dim <= 4 else 10_000

    global_candidates = rng.random((n_global, dim))

    top_k = min(max(5, n_obs // 3), n_obs)
    anchor_idx = np.argsort(y)[-top_k:]
    anchors = x[anchor_idx]
    per_anchor = int(np.ceil(n_local / top_k))
    local_scale = 0.026 + 0.045 / np.sqrt(dim)
    local_parts: list[np.ndarray] = []
    for anchor in anchors:
        noise = rng.normal(0.0, local_scale, size=(per_anchor, dim))
        local_parts.append(np.clip(anchor + noise, 0.0, 1.0))

    local_candidates = np.vstack(local_parts)[:n_local]
    return np.vstack([global_candidates, local_candidates])


def cluster_candidate_pool(
    x: np.ndarray,
    y: np.ndarray,
    target: ClusterSummary,
    neighbor: ClusterSummary,
    cue: str,
    rng: np.random.Generator,
) -> np.ndarray:
    dim = x.shape[1]
    base = base_candidate_pool(x=x, y=y, rng=rng)

    n_center = 4_500 if dim <= 4 else 6_500
    n_elite = 3_500 if dim <= 4 else 5_000
    n_frontier = 4_000 if dim <= 4 else 5_500

    best_point = x[target.best_idx]
    center_scale = max(0.018, 0.55 * target.radius_p80 + 0.015)
    elite_scale = max(0.014, 0.35 * target.radius_p80 + 0.010)

    center_noise = rng.normal(0.0, center_scale, size=(n_center, dim))
    center_cands = np.clip(target.centroid + center_noise, 0.0, 1.0)

    elite_noise = rng.normal(0.0, elite_scale, size=(n_elite, dim))
    elite_cands = np.clip(best_point + elite_noise, 0.0, 1.0)

    if neighbor.label == target.label:
        frontier_cands = center_cands[:n_frontier]
    else:
        if cue == "boundary_tightening":
            blend = rng.beta(2.2, 5.0, size=(n_frontier, 1))
        elif cue == "bridge_probe":
            blend = rng.beta(2.4, 2.4, size=(n_frontier, 1))
        else:
            blend = rng.beta(4.5, 2.2, size=(n_frontier, 1)) * 0.45
        line = target.centroid + blend * (neighbor.centroid - target.centroid)
        frontier_scale = max(0.016, 0.35 * target.radius_p80 + 0.012)
        frontier_noise = rng.normal(0.0, frontier_scale, size=(n_frontier, dim))
        frontier_cands = np.clip(line + frontier_noise, 0.0, 1.0)

    return np.vstack([base, center_cands, elite_cands, frontier_cands])


def frontier_alignment(
    candidates: np.ndarray,
    target: ClusterSummary,
    neighbor: ClusterSummary,
    cue: str,
) -> np.ndarray:
    if neighbor.label == target.label:
        return np.zeros(candidates.shape[0], dtype=float)

    vec = neighbor.centroid - target.centroid
    denom = float(np.dot(vec, vec))
    if denom < 1e-12:
        return np.zeros(candidates.shape[0], dtype=float)

    rel = candidates - target.centroid
    proj = rel @ vec / denom
    proj_clip = np.clip(proj, 0.0, 1.0)
    nearest = target.centroid + proj_clip[:, None] * vec
    ortho = np.sqrt(((candidates - nearest) ** 2).sum(axis=1))

    desired = 0.25
    if cue == "bridge_probe":
        desired = 0.50
    elif cue == "centroid_trend":
        desired = 0.12

    proj_term = np.exp(-((proj_clip - desired) ** 2) / 0.020)
    ortho_scale = max(0.018, 0.60 * target.radius_p80 + 0.020)
    ortho_term = np.exp(-ortho / ortho_scale)
    return proj_term * ortho_term


def affinity_to_cluster(candidates: np.ndarray, target: ClusterSummary) -> np.ndarray:
    dist = np.sqrt(((candidates - target.centroid) ** 2).sum(axis=1))
    scale = max(0.025, target.radius_p80 + 0.04)
    return np.exp(-dist / scale)


def propose_query(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_svr: int,
    n_mlp: int,
    ensemble_size: int,
) -> tuple[np.ndarray, str]:
    family, params, cv_mae_score = tune_model_family(
        x=x,
        y=y,
        rng=rng,
        n_svr=n_svr,
        n_mlp=n_mlp,
    )

    models = fit_ensemble(
        x=x,
        y=y,
        family=family,
        params=params,
        rng=rng,
        ensemble_size=ensemble_size,
    )

    cluster_seed = int(rng.integers(0, 2**31 - 1))
    clusters, _labels = summarize_clusters(
        x=x,
        y=y,
        models=models,
        seed=cluster_seed,
    )
    target = choose_target_cluster(clusters)
    neighbor = nearest_cluster(target, clusters)
    cue = choose_cue(target)

    candidates = cluster_candidate_pool(
        x=x,
        y=y,
        target=target,
        neighbor=neighbor,
        cue=cue,
        rng=rng,
    )

    mean, std = ensemble_stats(models, candidates)
    deltas = candidates[:, None, :] - x[None, :, :]
    min_dist = np.sqrt((deltas * deltas).sum(axis=2)).min(axis=1)
    novelty = np.log1p(min_dist)
    affinity = affinity_to_cluster(candidates, target)
    frontier = frontier_alignment(candidates, target, neighbor, cue)

    pred_score = normalize(mean)
    unc_score = normalize(std)
    nov_score = normalize(novelty)
    aff_score = normalize(affinity)
    frontier_score = normalize(frontier)

    if cue == "boundary_tightening":
        score = (
            0.42 * pred_score
            + 0.18 * unc_score
            + 0.12 * nov_score
            + 0.14 * aff_score
            + 0.14 * frontier_score
        )
    elif cue == "bridge_probe":
        score = (
            0.34 * pred_score
            + 0.28 * unc_score
            + 0.12 * nov_score
            + 0.08 * aff_score
            + 0.18 * frontier_score
        )
    else:
        score = (
            0.48 * pred_score
            + 0.18 * unc_score
            + 0.12 * nov_score
            + 0.18 * aff_score
            + 0.04 * frontier_score
        )

    eps = 0.023 * np.sqrt(x.shape[1])
    score[min_dist < eps] = -np.inf

    chosen_idx = int(np.argmax(score))
    query = candidates[chosen_idx]

    grads = local_sensitivity(models=models, x0=query)
    report = (
        f"model={family} cv_mae={cv_mae_score:.6f} clusters={len(clusters)} "
        f"target=c{target.label} size={target.size} cue={cue} "
        f"cluster_score={target.score:.4f} gap={target.centroid_gap:.4f} "
        f"radius={target.radius_p80:.4f} pred={pred_score[chosen_idx]:.4f} "
        f"unc={unc_score[chosen_idx]:.4f} nov={nov_score[chosen_idx]:.4f} "
        f"aff={aff_score[chosen_idx]:.4f} frontier={frontier_score[chosen_idx]:.4f} "
        f"top_dims={top_dimensions(grads)}"
    )
    return query, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("initial_data"),
        help="Directory containing function_*/initial_inputs.npy and initial_outputs.npy",
    )
    parser.add_argument("--seed", type=int, default=1119, help="Random seed")
    parser.add_argument("--ensemble-size", type=int, default=8, help="Ensemble size")
    parser.add_argument("--n-svr", type=int, default=8, help="SVR tuning trials")
    parser.add_argument("--n-mlp", type=int, default=5, help="MLP tuning trials")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/module22_week11_queries.txt"),
        help="Output query file",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("src/module22_week11_cluster_report.txt"),
        help="Output clustering report",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    rng = np.random.default_rng(args.seed)
    query_lines: list[str] = []
    report_lines: list[str] = []

    for function_dir in find_function_dirs(args.data_dir):
        func_id = int(function_dir.name.split("_")[1])
        x, y = load_xy(function_dir)
        query, report = propose_query(
            x=x,
            y=y,
            rng=rng,
            n_svr=args.n_svr,
            n_mlp=args.n_mlp,
            ensemble_size=args.ensemble_size,
        )
        query_lines.append(f"function_{func_id}: {format_query(query)}")
        report_lines.append(f"function_{func_id}: {report}")

    args.output.write_text("\n".join(query_lines) + "\n")
    args.report.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
