#!/usr/bin/env python3
"""Generate Week 4 BBO query candidates using a neural-network surrogate.

Approach:
- Train an ensemble of MLP regressors on bootstrap resamples.
- Score candidate points with: mean_prediction + beta * prediction_std.
- Exclude candidates too close to already sampled points.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
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


def fit_mlp_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    ensemble_size: int,
    rng: np.random.Generator,
) -> list:
    models = []
    n = x.shape[0]

    for _ in range(ensemble_size):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]

        width = int(rng.integers(24, 96))
        alpha = float(10 ** rng.uniform(-6.0, -2.0))
        lr_init = float(10 ** rng.uniform(-4.0, -2.0))

        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(width, width),
                activation="relu",
                solver="adam",
                alpha=alpha,
                learning_rate_init=lr_init,
                early_stopping=True,
                validation_fraction=0.2,
                max_iter=2500,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(xb, yb)
        models.append(model)

    return models


def propose_query(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    ensemble_size: int,
    beta: float,
) -> np.ndarray:
    dim = x.shape[1]
    n_candidates = 80_000 if dim <= 4 else 120_000
    candidates = rng.random((n_candidates, dim))

    models = fit_mlp_ensemble(x, y, ensemble_size=ensemble_size, rng=rng)
    preds = np.column_stack([m.predict(candidates) for m in models])
    mean = preds.mean(axis=1)
    std = preds.std(axis=1)
    acquisition = mean + beta * std

    eps = 0.03 * np.sqrt(dim)
    deltas = candidates[:, None, :] - x[None, :, :]
    min_dist = np.sqrt((deltas * deltas).sum(axis=2)).min(axis=1)
    acquisition[min_dist < eps] = -np.inf

    best_idx = int(np.argmax(acquisition))
    return candidates[best_idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("initial_data"),
        help="Directory containing function_*/initial_inputs.npy and initial_outputs.npy",
    )
    parser.add_argument("--seed", type=int, default=321, help="Random seed")
    parser.add_argument("--ensemble-size", type=int, default=12, help="Number of MLP models")
    parser.add_argument(
        "--beta",
        type=float,
        default=1.6,
        help="Exploration weight in acquisition = mean + beta * std",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. If omitted, prints to stdout.",
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
    if args.output is None:
        print(output_text, end="")
    else:
        args.output.write_text(output_text)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
