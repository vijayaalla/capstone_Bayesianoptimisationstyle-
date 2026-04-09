#!/usr/bin/env python3
"""Generate Week 9 BBO queries with scaling/emergence-aware decision logic.

This extends the LLM-inspired Week 8 strategy by comparing:
- short-context model views
- medium-context model views
- full-context model views

The final score blends:
- stable performance across scales
- gains that only appear at larger context scales
- uncertainty and novelty
- constrained decoding for controlled diversity
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
from generate_week8_queries_llm_strategy import candidate_pool, normalize, select_with_decoding


def propose_query(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_svr: int,
    n_mlp: int,
    ensemble_size: int,
    base_temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
) -> tuple[np.ndarray, str]:
    family, params, cv_mae_score = tune_model_family(
        x=x,
        y=y,
        rng=rng,
        n_svr=n_svr,
        n_mlp=n_mlp,
    )

    n_obs = x.shape[0]
    dim = x.shape[1]

    small_window = min(n_obs, max(4, max_tokens // 16))
    medium_window = min(n_obs, max(small_window + 1, max_tokens // 10))

    full_models = fit_ensemble(
        x=x,
        y=y,
        family=family,
        params=params,
        rng=rng,
        ensemble_size=ensemble_size,
    )
    medium_models = fit_ensemble(
        x=x[-medium_window:],
        y=y[-medium_window:],
        family=family,
        params=params,
        rng=rng,
        ensemble_size=max(4, ensemble_size // 2),
    )
    small_models = fit_ensemble(
        x=x[-small_window:],
        y=y[-small_window:],
        family=family,
        params=params,
        rng=rng,
        ensemble_size=max(4, ensemble_size // 2),
    )

    candidates = candidate_pool(x=x, y=y, rng=rng)

    full_preds = np.column_stack([m.predict(candidates) for m in full_models])
    medium_preds = np.column_stack([m.predict(candidates) for m in medium_models])
    small_preds = np.column_stack([m.predict(candidates) for m in small_models])

    full_mean = full_preds.mean(axis=1)
    medium_mean = medium_preds.mean(axis=1)
    small_mean = small_preds.mean(axis=1)

    full_std = full_preds.std(axis=1)
    medium_std = medium_preds.std(axis=1)
    small_std = small_preds.std(axis=1)

    stacked_means = np.column_stack([small_mean, medium_mean, full_mean])
    scale_disagreement = stacked_means.std(axis=1)
    scale_mean = 0.20 * small_mean + 0.35 * medium_mean + 0.45 * full_mean

    emergence_gain = np.maximum(medium_mean - small_mean, 0.0) + np.maximum(
        full_mean - medium_mean, 0.0
    )
    uncertainty = 0.30 * small_std + 0.30 * medium_std + 0.40 * full_std

    deltas = candidates[:, None, :] - x[None, :, :]
    min_dist = np.sqrt((deltas * deltas).sum(axis=2)).min(axis=1)
    novelty = np.log1p(min_dist)

    edge_like = np.sum((candidates < 0.002) | (candidates > 0.998), axis=1).astype(float)
    edge_penalty = 0.035 * edge_like

    score = (
        0.34 * normalize(scale_mean)
        + 0.18 * normalize(emergence_gain)
        + 0.16 * normalize(uncertainty)
        + 0.12 * normalize(scale_disagreement)
        + 0.10 * normalize(novelty)
        + 0.10 * normalize(full_mean)
        - edge_penalty
    )

    eps = 0.024 * np.sqrt(dim)
    score[min_dist < eps] = -np.inf

    y_std = float(np.std(y))
    norm_cv = cv_mae_score / (y_std + 1e-9)
    temperature = base_temperature + 0.10 * np.clip(norm_cv, 0.0, 1.5)

    chosen_idx = select_with_decoding(
        scores=score,
        rng=rng,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    query = candidates[chosen_idx]

    report = (
        f"model={family} cv_mae={cv_mae_score:.6f} strategy=scale_emergence "
        f"temperature={temperature:.2f} top_k={top_k} top_p={top_p:.2f} "
        f"max_tokens={max_tokens} small_window={small_window} medium_window={medium_window} "
        f"emergence={emergence_gain[chosen_idx]:.6f} "
        f"disagreement={scale_disagreement[chosen_idx]:.6f}"
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
    parser.add_argument("--seed", type=int, default=920, help="Random seed")
    parser.add_argument("--ensemble-size", type=int, default=10, help="Ensemble size")
    parser.add_argument("--n-svr", type=int, default=12, help="SVR tuning trials")
    parser.add_argument("--n-mlp", type=int, default=8, help="MLP tuning trials")
    parser.add_argument("--temperature", type=float, default=0.58, help="Base decoder temperature")
    parser.add_argument("--top-k", type=int, default=112, help="Decoder top-k")
    parser.add_argument("--top-p", type=float, default=0.90, help="Decoder top-p")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=96,
        help="Context budget used to derive short/medium windows",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/module20_week9_queries.txt"),
        help="Output query file",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("src/module20_week9_scaling_report.txt"),
        help="Output scaling diagnostics file",
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
            base_temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        query_lines.append(f"function_{func_id}: {format_query(query)}")
        report_lines.append(f"function_{func_id}: {report}")

    args.output.write_text("\n".join(query_lines) + "\n")
    args.report.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
