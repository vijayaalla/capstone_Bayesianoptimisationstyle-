#!/usr/bin/env python3
"""Generate Week 8 BBO queries with an LLM-inspired decision layer.

This script keeps the tuned surrogate idea from Week 7, then adds:
- full-context and finite-context model views (attention/window analogy)
- structured scoring fields instead of one raw acquisition value
- decoding controls (temperature, top-k, top-p) for controlled diversity
- light penalties for boundary-heavy strings to reduce formatting artefacts
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


def normalize(v: np.ndarray) -> np.ndarray:
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if vmax - vmin < 1e-12:
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)


def candidate_pool(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    dim = x.shape[1]
    n_obs = x.shape[0]

    n_global = 80_000 if dim <= 4 else 120_000
    n_local = 16_000 if dim <= 4 else 24_000

    global_candidates = rng.random((n_global, dim))

    top_k = min(max(6, n_obs // 3), n_obs)
    anchor_idx = np.argsort(y)[-top_k:]
    anchors = x[anchor_idx]
    per_anchor = int(np.ceil(n_local / top_k))
    local_scale = 0.03 + 0.05 / np.sqrt(dim)
    local_parts: list[np.ndarray] = []
    for anchor in anchors:
        noise = rng.normal(0.0, local_scale, size=(per_anchor, dim))
        local_parts.append(np.clip(anchor + noise, 0.0, 1.0))

    local_candidates = np.vstack(local_parts)[:n_local]
    return np.vstack([global_candidates, local_candidates])


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    ex = np.exp(shifted)
    denom = np.sum(ex)
    if denom <= 0:
        return np.full_like(ex, 1.0 / len(ex))
    return ex / denom


def select_with_decoding(
    scores: np.ndarray,
    rng: np.random.Generator,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    top_k = min(max(1, top_k), scores.shape[0])
    top_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_scores = scores[top_idx]

    order = np.argsort(top_scores)[::-1]
    sorted_idx = top_idx[order]
    sorted_scores = top_scores[order]

    probs = softmax(sorted_scores / max(temperature, 1e-6))
    cdf = np.cumsum(probs)
    keep = cdf <= top_p
    if not np.any(keep):
        keep[0] = True
    else:
        first_excluded = int(np.argmax(~keep)) if np.any(~keep) else -1
        if first_excluded >= 0:
            keep[first_excluded] = True

    kept_idx = sorted_idx[keep]
    kept_scores = sorted_scores[keep]
    kept_probs = softmax(kept_scores / max(temperature, 1e-6))
    chosen_local = int(rng.choice(len(kept_idx), p=kept_probs))
    return int(kept_idx[chosen_local])


def propose_query(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_svr: int,
    n_mlp: int,
    ensemble_size: int,
    temperature: float,
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

    full_models = fit_ensemble(
        x=x,
        y=y,
        family=family,
        params=params,
        rng=rng,
        ensemble_size=ensemble_size,
    )

    # Finite-context view: treat max_tokens as a cap on how many summarized
    # observations can influence the prompt-like decision layer.
    context_window = min(x.shape[0], max(4, max_tokens // 8))
    x_focus = x[-context_window:]
    y_focus = y[-context_window:]
    focus_models = fit_ensemble(
        x=x_focus,
        y=y_focus,
        family=family,
        params=params,
        rng=rng,
        ensemble_size=max(4, ensemble_size // 2),
    )

    candidates = candidate_pool(x=x, y=y, rng=rng)

    full_preds = np.column_stack([m.predict(candidates) for m in full_models])
    focus_preds = np.column_stack([m.predict(candidates) for m in focus_models])

    full_mean = full_preds.mean(axis=1)
    full_std = full_preds.std(axis=1)
    focus_mean = focus_preds.mean(axis=1)
    focus_std = focus_preds.std(axis=1)

    deltas = candidates[:, None, :] - x[None, :, :]
    min_dist = np.sqrt((deltas * deltas).sum(axis=2)).min(axis=1)

    # Structured prompt analogue: combine multiple named fields instead of one scalar.
    few_shot_score = 0.65 * focus_mean + 0.35 * full_mean
    disagreement = np.abs(focus_mean - full_mean)
    uncertainty = 0.55 * full_std + 0.45 * focus_std
    novelty = np.log1p(min_dist)

    # Penalize strings with many edge-case coordinates to reduce formatting artefacts.
    edge_like = np.sum((candidates < 0.002) | (candidates > 0.998), axis=1).astype(float)
    edge_penalty = 0.04 * edge_like

    score = (
        0.46 * normalize(few_shot_score)
        + 0.18 * normalize(full_mean)
        + 0.18 * normalize(uncertainty)
        + 0.12 * normalize(disagreement)
        + 0.10 * normalize(novelty)
        - edge_penalty
    )

    eps = 0.024 * np.sqrt(x.shape[1])
    score[min_dist < eps] = -np.inf

    chosen_idx = select_with_decoding(
        scores=score,
        rng=rng,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    query = candidates[chosen_idx]

    report = (
        f"model={family} cv_mae={cv_mae_score:.6f} prompt=structured_few_shot "
        f"temperature={temperature:.2f} top_k={top_k} top_p={top_p:.2f} "
        f"max_tokens={max_tokens} context_window={context_window} "
        f"edge_penalty={edge_like[chosen_idx]:.0f}"
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
    parser.add_argument("--seed", type=int, default=819, help="Random seed")
    parser.add_argument("--ensemble-size", type=int, default=10, help="Ensemble size")
    parser.add_argument("--n-svr", type=int, default=12, help="SVR tuning trials")
    parser.add_argument("--n-mlp", type=int, default=8, help="MLP tuning trials")
    parser.add_argument("--temperature", type=float, default=0.65, help="Decoder temperature")
    parser.add_argument("--top-k", type=int, default=96, help="Decoder top-k")
    parser.add_argument("--top-p", type=float, default=0.88, help="Decoder top-p")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Prompt-like context budget used to limit focused-context history",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/module19_week8_queries.txt"),
        help="Output query file",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("src/module19_week8_prompt_report.txt"),
        help="Output prompt/decoding diagnostics file",
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
            temperature=args.temperature,
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
