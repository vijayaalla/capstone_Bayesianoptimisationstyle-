#!/usr/bin/env python3
"""Generate Week 3 BBO query candidates using an SVM-based surrogate strategy.

Approach:
- Fit an ensemble of SVR models on bootstrap resamples.
- Score random candidates by mean prediction + beta * predictive std.
- Exclude points too close to existing observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def format_query(x: np.ndarray) -> str:
    return "-".join(f"{v:.6f}" for v in x)


def find_function_dirs(data_dir: Path) -> list[Path]:
    dirs = [p for p in data_dir.glob("function_*") if p.is_dir()]
    return sorted(dirs, key=lambda p: int(p.name.split("_")[1]))


def load_xy(function_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(function_dir / "initial_inputs.npy")
    y = np.load(function_dir / "initial_outputs.npy")
    return x, y


def fit_svr_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    ensemble_size: int,
    rng: np.random.Generator,
) -> list:
    models = []
    n = x.shape[0]
    for _ in range(ensemble_size):
        # Bootstrap resample for diversity and uncertainty proxy.
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]

        # Randomized hyperparameters within conservative ranges.
        c = 10 ** rng.uniform(-1.0, 2.0)
        gamma = 10 ** rng.uniform(-2.5, 0.5)
        epsilon = 10 ** rng.uniform(-4.0, -1.0)

        model = make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", C=float(c), gamma=float(gamma), epsilon=float(epsilon)),
        )
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
    n_candidates = 70_000 if dim <= 4 else 90_000
    candidates = rng.random((n_candidates, dim))

    models = fit_svr_ensemble(x, y, ensemble_size=ensemble_size, rng=rng)
    preds = np.column_stack([m.predict(candidates) for m in models])
    mean = preds.mean(axis=1)
    std = preds.std(axis=1)
    acquisition = mean + beta * std

    # Keep some distance from already queried points.
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
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--ensemble-size", type=int, default=15, help="Number of SVR models")
    parser.add_argument(
        "--beta",
        type=float,
        default=1.8,
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
            x,
            y,
            rng=rng,
            ensemble_size=args.ensemble_size,
            beta=args.beta,
        )
        lines.append(f"function_{func_id}: {format_query(x_next)}")

    text = "\n".join(lines) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
