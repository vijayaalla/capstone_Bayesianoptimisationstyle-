#!/usr/bin/env python3
"""Generate Week 12 BBO queries with a PCA-guided strategy.

Approach:
- Tune SVR/MLP surrogate families using CV MAE.
- Fit a bootstrap ensemble for mean/uncertainty estimation.
- Fit PCA on observed inputs to identify dominant variation directions.
- Generate targeted candidates by perturbing strong anchors along top PCs.
- Score candidates with prediction, uncertainty, novelty, and PCA alignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from generate_week7_queries_tuned import (
    find_function_dirs,
    fit_ensemble,
    format_query,
    load_xy,
    tune_model_family,
)
from generate_week8_queries_llm_strategy import candidate_pool, normalize


def pca_candidate_pool(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, PCA, StandardScaler]:
    dim = x.shape[1]
    scaler = StandardScaler()
    xz = scaler.fit_transform(x)
    pca = PCA(n_components=min(dim, x.shape[0]))
    pca.fit(xz)

    n_pc = min(2, dim)
    top_k = min(max(4, x.shape[0] // 4), x.shape[0])
    anchors = x[np.argsort(y)[-top_k:]]

    per_anchor = 3500 if dim <= 4 else 5000
    parts: list[np.ndarray] = []
    for anchor in anchors:
        z0 = scaler.transform(anchor[None, :])[0]
        coeffs = rng.normal(0.0, 0.55, size=(per_anchor, n_pc))
        z = np.repeat(z0[None, :], per_anchor, axis=0)
        for j in range(n_pc):
            scale = np.sqrt(max(pca.explained_variance_[j], 1e-9))
            z += coeffs[:, [j]] * scale * pca.components_[j][None, :]
        x_pc = np.clip(scaler.inverse_transform(z), 0.0, 1.0)
        parts.append(x_pc)

    return np.vstack(parts), pca, scaler


def top_loading_dims(pca: PCA, limit: int = 3) -> str:
    if pca.components_.size == 0:
        return ""
    load = np.abs(pca.components_[0])
    order = np.argsort(load)[::-1][: min(limit, len(load))]
    return ",".join([f"x{i + 1}:{load[i]:.3f}" for i in order])


def pca_alignment(candidates: np.ndarray, x: np.ndarray, pca: PCA, scaler: StandardScaler) -> np.ndarray:
    z = scaler.transform(candidates)
    xz = scaler.transform(x)
    deltas = z[:, None, :] - xz[None, :, :]
    nearest = np.argmin(np.sum(deltas * deltas, axis=2), axis=1)
    base = xz[nearest]
    move = z - base
    total = np.sqrt(np.sum(move * move, axis=1)) + 1e-12

    n_pc = min(2, pca.components_.shape[0])
    proj = np.zeros(len(candidates), dtype=float)
    for j in range(n_pc):
        comp = pca.components_[j]
        coeff = np.abs(move @ comp)
        proj += coeff
    return proj / total


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

    global_candidates = candidate_pool(x=x, y=y, rng=rng)
    pc_candidates, pca, scaler = pca_candidate_pool(x=x, y=y, rng=rng)
    candidates = np.vstack([global_candidates, pc_candidates])

    preds = np.column_stack([m.predict(candidates) for m in models])
    mean = preds.mean(axis=1)
    std = preds.std(axis=1)

    deltas = candidates[:, None, :] - x[None, :, :]
    min_dist = np.sqrt((deltas * deltas).sum(axis=2)).min(axis=1)
    novelty = np.log1p(min_dist)
    align = pca_alignment(candidates=candidates, x=x, pca=pca, scaler=scaler)

    pred_score = normalize(mean)
    unc_score = normalize(std)
    nov_score = normalize(novelty)
    align_score = normalize(align)

    score = (
        0.58 * pred_score
        + 0.20 * unc_score
        + 0.12 * nov_score
        + 0.10 * align_score
    )

    eps = 0.024 * np.sqrt(x.shape[1])
    score[min_dist < eps] = -np.inf

    chosen_idx = int(np.argmax(score))
    query = candidates[chosen_idx]

    evr = pca.explained_variance_ratio_
    evr1 = float(evr[0]) if len(evr) > 0 else 0.0
    evr2 = float(evr[1]) if len(evr) > 1 else 0.0

    report = (
        f"model={family} cv_mae={cv_mae_score:.6f} "
        f"pc1={evr1:.3f} pc2={evr2:.3f} "
        f"pred={pred_score[chosen_idx]:.4f} unc={unc_score[chosen_idx]:.4f} "
        f"nov={nov_score[chosen_idx]:.4f} align={align_score[chosen_idx]:.4f} "
        f"pc1_dims={top_loading_dims(pca)}"
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
    parser.add_argument("--seed", type=int, default=1222, help="Random seed")
    parser.add_argument("--ensemble-size", type=int, default=10, help="Ensemble size")
    parser.add_argument("--n-svr", type=int, default=12, help="SVR tuning trials")
    parser.add_argument("--n-mlp", type=int, default=8, help="MLP tuning trials")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/module23_week12_queries.txt"),
        help="Output query file",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("src/module23_week12_pca_report.txt"),
        help="Output PCA report",
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
