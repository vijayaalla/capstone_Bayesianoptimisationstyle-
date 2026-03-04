#!/usr/bin/env python3
"""Generate one candidate query per function for the BBO capstone portal.

Strategy:
- Fit a Gaussian Process surrogate for each function.
- Score random candidates with a UCB acquisition value.
- Return the best non-duplicate candidate.

Output format matches the portal style: x1-x2-...-xn with 6 decimals.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def format_query(x: np.ndarray) -> str:
    return "-".join(f"{v:.6f}" for v in x)


def load_xy(func_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(func_dir / "initial_inputs.npy")
    y = np.load(func_dir / "initial_outputs.npy")
    return x, y


def propose_query(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    dim = x.shape[1]
    n_candidates = 60_000 if dim <= 4 else 80_000
    candidates = rng.random((n_candidates, dim))

    y_scale = float(np.std(y)) if float(np.std(y)) > 1e-9 else 1.0
    y_norm = (y - float(np.mean(y))) / y_scale

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(dim), nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=2,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        gp.fit(x, y_norm)

    mean, std = gp.predict(candidates, return_std=True)
    kappa = 1.7 + 0.15 * dim
    acquisition = mean + kappa * std

    # Avoid choosing points that are too close to existing observations.
    eps = 0.03 * np.sqrt(dim)
    deltas = candidates[:, None, :] - x[None, :, :]
    min_dist = np.sqrt((deltas * deltas).sum(axis=2)).min(axis=1)
    acquisition[min_dist < eps] = -np.inf

    best_idx = int(np.argmax(acquisition))
    return candidates[best_idx]


def find_function_dirs(data_dir: Path) -> list[Path]:
    dirs = [p for p in data_dir.glob("function_*") if p.is_dir()]
    return sorted(dirs, key=lambda p: int(p.name.split("_")[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("initial_data"),
        help="Directory containing function_*/initial_inputs.npy and initial_outputs.npy",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    rng = np.random.default_rng(args.seed)
    lines: list[str] = []
    for func_dir in find_function_dirs(data_dir):
        func_id = int(func_dir.name.split("_")[1])
        x, y = load_xy(func_dir)
        x_next = propose_query(x, y, rng)
        lines.append(f"function_{func_id}: {format_query(x_next)}")

    output_text = "\n".join(lines) + "\n"
    if args.output is None:
        print(output_text, end="")
    else:
        args.output.write_text(output_text)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
