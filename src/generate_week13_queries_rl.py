#!/usr/bin/env python3
"""Generate final-round BBO queries with an RL-inspired policy chooser.

Each function compares three policy-style candidate generators:
- exploit arm: refine near best observed regions
- explore arm: search globally with uncertainty emphasis
- pca arm: move along dominant variation directions

The chosen arm is the one with the strongest policy-conditioned reward proxy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from generate_week7_queries_tuned import (
    find_function_dirs,
    fit_ensemble,
    format_query,
    load_xy,
    tune_model_family,
)
from generate_week8_queries_llm_strategy import normalize
from generate_week12_queries_pca import pca_alignment, pca_candidate_pool


def exploit_pool(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    dim = x.shape[1]
    top_k = min(max(5, x.shape[0] // 4), x.shape[0])
    anchors = x[np.argsort(y)[-top_k:]]
    per_anchor = 8000 if dim <= 4 else 12000
    scale = 0.025 + 0.045 / np.sqrt(dim)
    parts: list[np.ndarray] = []
    for anchor in anchors:
        noise = rng.normal(0.0, scale, size=(per_anchor, dim))
        parts.append(np.clip(anchor + noise, 0.0, 1.0))
    return np.vstack(parts)


def explore_pool(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    dim = x.shape[1]
    n = 90000 if dim <= 4 else 130000
    return rng.random((n, dim))


def score_common(
    candidates: np.ndarray,
    x: np.ndarray,
    models,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preds = np.column_stack([m.predict(candidates) for m in models])
    mean = preds.mean(axis=1)
    std = preds.std(axis=1)
    deltas = candidates[:, None, :] - x[None, :, :]
    min_dist = np.sqrt((deltas * deltas).sum(axis=2)).min(axis=1)
    novelty = np.log1p(min_dist)
    return mean, std, novelty, min_dist


def choose_best(score: np.ndarray, min_dist: np.ndarray, dim: int) -> int:
    score = score.copy()
    eps = 0.024 * np.sqrt(dim)
    score[min_dist < eps] = -np.inf
    return int(np.argmax(score))


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

    exp_cands = exploit_pool(x=x, y=y, rng=rng)
    exp_mean, exp_std, exp_nov, exp_dist = score_common(exp_cands, x, models)
    exp_score = 0.70 * normalize(exp_mean) + 0.15 * normalize(exp_std) + 0.15 * normalize(exp_nov)
    exp_idx = choose_best(exp_score, exp_dist, x.shape[1])

    explore_cands = explore_pool(x=x, rng=rng)
    ex_mean, ex_std, ex_nov, ex_dist = score_common(explore_cands, x, models)
    explore_score = 0.40 * normalize(ex_mean) + 0.40 * normalize(ex_std) + 0.20 * normalize(ex_nov)
    explore_idx = choose_best(explore_score, ex_dist, x.shape[1])

    pca_cands, pca, scaler = pca_candidate_pool(x=x, y=y, rng=rng)
    p_mean, p_std, p_nov, p_dist = score_common(pca_cands, x, models)
    p_align = pca_alignment(candidates=pca_cands, x=x, pca=pca, scaler=scaler)
    pca_score = (
        0.55 * normalize(p_mean)
        + 0.20 * normalize(p_std)
        + 0.10 * normalize(p_nov)
        + 0.15 * normalize(p_align)
    )
    pca_idx = choose_best(pca_score, p_dist, x.shape[1])

    arm_candidates = {
        "exploit": (exp_cands[exp_idx], float(exp_score[exp_idx]), float(exp_mean[exp_idx]), float(exp_std[exp_idx])),
        "explore": (
            explore_cands[explore_idx],
            float(explore_score[explore_idx]),
            float(ex_mean[explore_idx]),
            float(ex_std[explore_idx]),
        ),
        "pca": (pca_cands[pca_idx], float(pca_score[pca_idx]), float(p_mean[pca_idx]), float(p_std[pca_idx])),
    }

    chosen_arm = max(arm_candidates.items(), key=lambda kv: kv[1][1])[0]
    query, arm_score, arm_mean, arm_std = arm_candidates[chosen_arm]

    report = (
        f"model={family} cv_mae={cv_mae_score:.6f} chosen_arm={chosen_arm} "
        f"exploit={arm_candidates['exploit'][1]:.4f} "
        f"explore={arm_candidates['explore'][1]:.4f} "
        f"pca={arm_candidates['pca'][1]:.4f} "
        f"mean={arm_mean:.6f} std={arm_std:.6f}"
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
    parser.add_argument("--seed", type=int, default=1323, help="Random seed")
    parser.add_argument("--ensemble-size", type=int, default=10, help="Ensemble size")
    parser.add_argument("--n-svr", type=int, default=12, help="SVR tuning trials")
    parser.add_argument("--n-mlp", type=int, default=8, help="MLP tuning trials")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/module24_week13_queries.txt"),
        help="Output query file",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("src/module24_week13_rl_report.txt"),
        help="Output RL-style report",
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
