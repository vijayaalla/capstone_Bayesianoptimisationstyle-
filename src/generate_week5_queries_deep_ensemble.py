#!/usr/bin/env python3
"""Generate Week 5 BBO query candidates with a deep-ensemble surrogate.

Strategy:
- Train a diverse ensemble of MLP regressors on bootstrap resamples.
- Build a mixed candidate pool (global random + local perturbations).
- Score candidates with: mean + beta * std + novelty_bonus.
- Exclude candidates too close to previously sampled points.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def format_query(x: np.ndarray) -> str:
    return "-".join(f"{v:.6f}" for v in x)


def find_function_dirs(data_dir: Path) -> list[Path]:
    dirs = [p for p in data_dir.glob("function_*") if p.is_dir()]
    return sorted(dirs, key=lambda p: int(p.name.split("_")[1]))


def load_xy(function_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(function_dir / "initial_inputs.npy")
    y = np.load(function_dir / "initial_outputs.npy")
    return x, y


def sample_hidden_layers(dim: int, rng: np.random.Generator) -> tuple[int, ...]:
    depth = int(rng.integers(2, 4))  # 2 or 3 layers
    base = int(np.clip(20 + 12 * dim + rng.integers(-8, 25), 24, 160))
    widths: list[int] = []
    current = base
    for _ in range(depth):
        widths.append(int(np.clip(current, 16, 160)))
        current = int(max(16, round(current * rng.uniform(0.65, 0.9))))
    return tuple(widths)


def fit_deep_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    ensemble_size: int,
    rng: np.random.Generator,
) -> list[Pipeline]:
    models: list[Pipeline] = []
    n = x.shape[0]
    dim = x.shape[1]

    for _ in range(ensemble_size):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]

        hidden_layers = sample_hidden_layers(dim=dim, rng=rng)
        activation = "relu" if rng.random() < 0.8 else "tanh"
        alpha = float(10 ** rng.uniform(-6.0, -2.5))
        lr_init = float(10 ** rng.uniform(-4.2, -2.2))
        early_stopping = bool(rng.random() < 0.7)

        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                solver="adam",
                alpha=alpha,
                learning_rate_init=lr_init,
                batch_size="auto",
                early_stopping=early_stopping,
                validation_fraction=0.15 if early_stopping else 0.1,
                n_iter_no_change=30,
                max_iter=3000,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(xb, yb)
        models.append(model)

    return models


def score_candidates(
    candidates: np.ndarray,
    x: np.ndarray,
    models: list[Pipeline],
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    preds = np.column_stack([m.predict(candidates) for m in models])
    mean = preds.mean(axis=1)
    std = preds.std(axis=1)

    deltas = candidates[:, None, :] - x[None, :, :]
    min_dist = np.sqrt((deltas * deltas).sum(axis=2)).min(axis=1)
    novelty_bonus = 0.05 * np.log1p(min_dist)

    acquisition = mean + beta * std + novelty_bonus
    return acquisition, min_dist


def propose_query(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    ensemble_size: int,
    beta: float | None,
) -> np.ndarray:
    dim = x.shape[1]
    n_obs = x.shape[0]

    global_candidates = 120_000 if dim <= 4 else 180_000
    local_candidates = 24_000 if dim <= 4 else 36_000
    candidates = rng.random((global_candidates, dim))

    models = fit_deep_ensemble(x=x, y=y, ensemble_size=ensemble_size, rng=rng)

    beta_value = beta
    if beta_value is None:
        beta_value = 1.0 + 0.12 * dim + 6.0 / (n_obs + 10.0)

    acquisition, min_dist = score_candidates(
        candidates=candidates,
        x=x,
        models=models,
        beta=beta_value,
    )

    # Keep only high-value seeds, then refine locally around those seeds.
    top_k = min(256, candidates.shape[0])
    top_idx = np.argpartition(acquisition, -top_k)[-top_k:]
    anchors = candidates[top_idx]

    per_anchor = int(np.ceil(local_candidates / top_k))
    local_list: list[np.ndarray] = []
    local_scale = 0.025 + 0.07 / np.sqrt(dim)
    for anchor in anchors:
        noise = rng.normal(loc=0.0, scale=local_scale, size=(per_anchor, dim))
        local_points = np.clip(anchor + noise, 0.0, 1.0)
        local_list.append(local_points)

    local_pool = np.vstack(local_list)[:local_candidates]
    all_candidates = np.vstack([candidates, local_pool])

    all_acq, all_min_dist = score_candidates(
        candidates=all_candidates,
        x=x,
        models=models,
        beta=beta_value,
    )

    eps = 0.028 * np.sqrt(dim)
    all_acq[all_min_dist < eps] = -np.inf

    best_idx = int(np.argmax(all_acq))
    return all_candidates[best_idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("initial_data"),
        help="Directory containing function_*/initial_inputs.npy and initial_outputs.npy",
    )
    parser.add_argument("--seed", type=int, default=512, help="Random seed")
    parser.add_argument("--ensemble-size", type=int, default=16, help="Number of MLP models")
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Exploration weight in acquisition. If omitted, uses adaptive value by dim/data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/module16_week5_queries.txt"),
        help="Output file path",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    rng = np.random.default_rng(args.seed)
    lines: list[str] = []

    for function_dir in find_function_dirs(args.data_dir):
        func_id = int(function_dir.name.split("_")[1])
        x, y = load_xy(function_dir)
        x_next = propose_query(
            x=x,
            y=y,
            rng=rng,
            ensemble_size=args.ensemble_size,
            beta=args.beta,
        )
        lines.append(f"function_{func_id}: {format_query(x_next)}")

    output_text = "\n".join(lines) + "\n"
    args.output.write_text(output_text)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
