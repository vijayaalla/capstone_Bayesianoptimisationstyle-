#!/usr/bin/env python3
"""Generate Week 6 BBO query candidates with a HEBO-inspired hybrid strategy.

Strategy:
- Apply simple per-dimension input warping (power transform selected by correlation).
- Fit a GP surrogate on warped inputs.
- Score candidates with a blended acquisition:
  score = w_ei * EI + w_pi * PI + w_ucb * UCB + w_std * uncertainty + novelty
- Reject candidates too close to existing observations.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
from scipy.special import erf
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


def format_query(x: np.ndarray) -> str:
    # Keep portal-safe formatting where each coordinate is represented as 0.xxxxxx.
    x_safe = np.clip(x, 0.0, 0.999999)
    return "-".join(f"{v:.6f}" for v in x_safe)


def find_function_dirs(data_dir: Path) -> list[Path]:
    dirs = [p for p in data_dir.glob("function_*") if p.is_dir()]
    return sorted(dirs, key=lambda p: int(p.name.split("_")[1]))


def load_xy(function_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(function_dir / "initial_inputs.npy")
    y = np.load(function_dir / "initial_outputs.npy")
    return x, y


def normalize(v: np.ndarray) -> np.ndarray:
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if vmax - vmin < 1e-12:
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)


def choose_warp_exponents(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    candidates = np.array([0.6, 0.8, 1.0, 1.25, 1.6], dtype=float)
    exponents = np.ones(x.shape[1], dtype=float)
    y_centered = y - float(np.mean(y))
    y_std = float(np.std(y_centered))
    if y_std < 1e-12:
        return exponents

    for j in range(x.shape[1]):
        xj = x[:, j]
        best_gamma = 1.0
        best_score = -np.inf
        for gamma in candidates:
            w = np.power(np.clip(xj, 1e-9, 1.0), gamma)
            w_centered = w - float(np.mean(w))
            denom = float(np.std(w_centered)) * y_std
            corr = 0.0 if denom < 1e-12 else float(np.mean(w_centered * y_centered) / denom)
            score = abs(corr)
            if score > best_score:
                best_score = score
                best_gamma = float(gamma)
        exponents[j] = best_gamma
    return exponents


def apply_warp(x: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    return np.power(np.clip(x, 1e-9, 1.0), exponents[None, :])


def std_norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def std_norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


def compute_ei(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float) -> np.ndarray:
    improvement = mu - y_best - xi
    safe_sigma = np.maximum(sigma, 1e-12)
    z = improvement / safe_sigma
    ei = improvement * std_norm_cdf(z) + safe_sigma * std_norm_pdf(z)
    ei[sigma < 1e-12] = 0.0
    return ei


def compute_pi(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float) -> np.ndarray:
    safe_sigma = np.maximum(sigma, 1e-12)
    z = (mu - y_best - xi) / safe_sigma
    pi = std_norm_cdf(z)
    pi[sigma < 1e-12] = 0.0
    return pi


def fit_gp(x: np.ndarray, y: np.ndarray) -> tuple[GaussianProcessRegressor, StandardScaler]:
    x_scaler = StandardScaler()
    xz = x_scaler.fit_transform(x)

    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    y_scale = y_std if y_std > 1e-12 else 1.0
    yz = (y - y_mean) / y_scale

    dim = x.shape[1]
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(dim), nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=3,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        gp.fit(xz, yz)
    return gp, x_scaler


def propose_query(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    dim = x.shape[1]
    n_obs = x.shape[0]
    n_global = 90_000 if dim <= 4 else 140_000
    n_local = 20_000 if dim <= 4 else 30_000

    global_candidates = rng.random((n_global, dim))

    # Local refinements around current top observations.
    top_obs = min(max(6, n_obs // 3), n_obs)
    top_idx = np.argsort(y)[-top_obs:]
    anchors = x[top_idx]
    per_anchor = int(np.ceil(n_local / top_obs))
    scale = 0.04 + 0.06 / np.sqrt(dim)
    local_list: list[np.ndarray] = []
    for anchor in anchors:
        noise = rng.normal(0.0, scale, size=(per_anchor, dim))
        local_points = np.clip(anchor + noise, 0.0, 1.0)
        local_list.append(local_points)
    local_candidates = np.vstack(local_list)[:n_local]

    candidates = np.vstack([global_candidates, local_candidates])

    exponents = choose_warp_exponents(x=x, y=y)
    x_warp = apply_warp(x=x, exponents=exponents)
    c_warp = apply_warp(x=candidates, exponents=exponents)

    gp, x_scaler = fit_gp(x=x_warp, y=y)
    c_warp_z = x_scaler.transform(c_warp)
    mu, sigma = gp.predict(c_warp_z, return_std=True)

    # Adaptive exploration tuning: more uncertainty focus at higher dim / lower data.
    kappa = 1.4 + 0.15 * dim + 5.0 / (n_obs + 12.0)
    xi = 0.01 + 0.02 * dim / (dim + n_obs)
    y_best = float(np.max((y - float(np.mean(y))) / (float(np.std(y)) if float(np.std(y)) > 1e-12 else 1.0)))

    ei = compute_ei(mu=mu, sigma=sigma, y_best=y_best, xi=xi)
    pi = compute_pi(mu=mu, sigma=sigma, y_best=y_best, xi=xi)
    ucb = mu + kappa * sigma

    ei_n = normalize(ei)
    pi_n = normalize(pi)
    ucb_n = normalize(ucb)
    std_n = normalize(sigma)

    # Shift weights toward uncertainty when data coverage is lower.
    scarcity = np.clip((25.0 - n_obs) / 20.0, 0.0, 1.0)
    w_std = 0.10 + 0.20 * scarcity
    w_ei = 0.45 - 0.10 * scarcity
    w_pi = 0.15
    w_ucb = 1.0 - (w_ei + w_pi + w_std)

    acquisition = w_ei * ei_n + w_pi * pi_n + w_ucb * ucb_n + w_std * std_n

    deltas = candidates[:, None, :] - x[None, :, :]
    min_dist = np.sqrt((deltas * deltas).sum(axis=2)).min(axis=1)
    novelty = 0.05 * np.log1p(min_dist)
    acquisition += novelty

    eps = 0.026 * np.sqrt(dim)
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
    parser.add_argument("--seed", type=int, default=617, help="Random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/module17_week6_queries.txt"),
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
        x_next = propose_query(x=x, y=y, rng=rng)
        lines.append(f"function_{func_id}: {format_query(x_next)}")

    output_text = "\n".join(lines) + "\n"
    args.output.write_text(output_text)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
