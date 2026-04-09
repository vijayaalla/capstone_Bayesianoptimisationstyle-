#!/usr/bin/env python3
"""Generate Week 10 BBO queries with a transparency-first decision rule.

Approach:
- Tune SVR/MLP surrogate families using CV MAE.
- Fit a bootstrap ensemble for uncertainty estimates.
- Score candidates with an explicit, decomposed formula:
  score = 0.55 * predicted_value + 0.30 * uncertainty + 0.15 * novelty
- Produce a per-function explanation report with local sensitivity estimates.
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
from generate_week8_queries_llm_strategy import candidate_pool, normalize


def ensemble_mean(models, x: np.ndarray) -> np.ndarray:
    preds = np.column_stack([m.predict(x) for m in models])
    return preds.mean(axis=1)


def local_sensitivity(models, x0: np.ndarray) -> np.ndarray:
    dim = x0.shape[0]
    grads = np.zeros(dim, dtype=float)
    for j in range(dim):
        step = np.zeros(dim, dtype=float)
        step[j] = 0.01
        xp = np.clip(x0 + step, 0.0, 1.0)[None, :]
        xm = np.clip(x0 - step, 0.0, 1.0)[None, :]
        yp = float(ensemble_mean(models, xp)[0])
        ym = float(ensemble_mean(models, xm)[0])
        grads[j] = (yp - ym) / 0.02
    return grads


def top_dimensions(grads: np.ndarray, limit: int = 3) -> str:
    order = np.argsort(np.abs(grads))[::-1]
    top = order[: min(limit, len(order))]
    parts = [f"x{idx + 1}:{grads[idx]:+.4f}" for idx in top]
    return ",".join(parts)


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

    candidates = candidate_pool(x=x, y=y, rng=rng)
    preds = np.column_stack([m.predict(candidates) for m in models])
    mean = preds.mean(axis=1)
    std = preds.std(axis=1)

    deltas = candidates[:, None, :] - x[None, :, :]
    min_dist = np.sqrt((deltas * deltas).sum(axis=2)).min(axis=1)
    novelty = np.log1p(min_dist)

    pred_score = normalize(mean)
    unc_score = normalize(std)
    nov_score = normalize(novelty)

    score = 0.55 * pred_score + 0.30 * unc_score + 0.15 * nov_score

    eps = 0.024 * np.sqrt(x.shape[1])
    score[min_dist < eps] = -np.inf

    chosen_idx = int(np.argmax(score))
    query = candidates[chosen_idx]

    grads = local_sensitivity(models=models, x0=query)
    mode = "balanced"
    if pred_score[chosen_idx] - unc_score[chosen_idx] > 0.18:
        mode = "exploit"
    elif unc_score[chosen_idx] - pred_score[chosen_idx] > 0.18:
        mode = "explore"

    report = (
        f"model={family} cv_mae={cv_mae_score:.6f} mode={mode} "
        f"pred={pred_score[chosen_idx]:.4f} unc={unc_score[chosen_idx]:.4f} "
        f"nov={nov_score[chosen_idx]:.4f} top_dims={top_dimensions(grads)}"
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
    parser.add_argument("--seed", type=int, default=1021, help="Random seed")
    parser.add_argument("--ensemble-size", type=int, default=10, help="Ensemble size")
    parser.add_argument("--n-svr", type=int, default=12, help="SVR tuning trials")
    parser.add_argument("--n-mlp", type=int, default=8, help="MLP tuning trials")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/module21_week10_queries.txt"),
        help="Output query file",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("src/module21_week10_interpretability_report.txt"),
        help="Output explanation report",
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
