#!/usr/bin/env python3
"""Generate Week 7 BBO query candidates with tuned surrogate hyperparameters.

Approach:
- Random-search tune two surrogate families (SVR and MLP) via CV MAE.
- Select the best family/config per function.
- Fit a bootstrap ensemble around the tuned configuration.
- Score candidates with mean + beta * std + novelty, and avoid near-duplicates.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def format_query(x: np.ndarray) -> str:
    # Keep portal-safe formatting in 0.xxxxxx form.
    x_safe = np.clip(x, 0.0, 0.999999)
    return "-".join(f"{v:.6f}" for v in x_safe)


def find_function_dirs(data_dir: Path) -> list[Path]:
    dirs = [p for p in data_dir.glob("function_*") if p.is_dir()]
    return sorted(dirs, key=lambda p: int(p.name.split("_")[1]))


def load_xy(function_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(function_dir / "initial_inputs.npy")
    y = np.load(function_dir / "initial_outputs.npy")
    return x, y


def sample_svr_params(rng: np.random.Generator) -> dict[str, float]:
    return {
        "C": float(10 ** rng.uniform(-1.0, 3.0)),
        "gamma": float(10 ** rng.uniform(-3.0, 0.3)),
        "epsilon": float(10 ** rng.uniform(-4.0, -0.2)),
    }


def sample_mlp_params(dim: int, rng: np.random.Generator) -> dict:
    depth = int(rng.integers(2, 4))
    base = int(np.clip(18 + 10 * dim + rng.integers(-6, 30), 24, 180))
    widths: list[int] = []
    w = base
    for _ in range(depth):
        widths.append(int(np.clip(w, 16, 180)))
        w = int(max(16, round(w * rng.uniform(0.65, 0.9))))
    return {
        "hidden_layer_sizes": tuple(widths),
        "activation": "relu" if rng.random() < 0.8 else "tanh",
        "alpha": float(10 ** rng.uniform(-6.0, -2.3)),
        "learning_rate_init": float(10 ** rng.uniform(-4.2, -2.0)),
        "early_stopping": bool(rng.random() < 0.65),
    }


def build_svr(params: dict[str, float]) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        SVR(
            kernel="rbf",
            C=float(params["C"]),
            gamma=float(params["gamma"]),
            epsilon=float(params["epsilon"]),
        ),
    )


def build_mlp(params: dict, rng: np.random.Generator, n_samples: int) -> Pipeline:
    early_stopping = bool(params["early_stopping"]) and n_samples >= 24
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            activation=str(params["activation"]),
            solver="adam",
            alpha=float(params["alpha"]),
            learning_rate_init=float(params["learning_rate_init"]),
            early_stopping=early_stopping,
            validation_fraction=0.15 if early_stopping else 0.1,
            n_iter_no_change=30,
            max_iter=2500,
            random_state=int(rng.integers(0, 2**31 - 1)),
        ),
    )


def cv_mae(model_factory, x: np.ndarray, y: np.ndarray, kf: KFold) -> float:
    scores: list[float] = []
    for tr, va in kf.split(x):
        model = model_factory()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model.fit(x[tr], y[tr])
        except ValueError:
            return float("inf")
        pred = model.predict(x[va])
        scores.append(float(mean_absolute_error(y[va], pred)))
    return float(np.mean(scores))


def tune_model_family(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_svr: int,
    n_mlp: int,
) -> tuple[str, dict, float]:
    n = x.shape[0]
    n_splits = 3 if n < 30 else 4
    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )

    best_family = "svr"
    best_params = sample_svr_params(rng)
    best_score = float("inf")

    for _ in range(n_svr):
        params = sample_svr_params(rng)
        score = cv_mae(
            model_factory=lambda p=params: build_svr(p),
            x=x,
            y=y,
            kf=kf,
        )
        if score < best_score:
            best_family = "svr"
            best_params = params
            best_score = score

    dim = x.shape[1]
    for _ in range(n_mlp):
        params = sample_mlp_params(dim=dim, rng=rng)
        score = cv_mae(
            model_factory=lambda p=params, n=n: build_mlp(p, rng, n),
            x=x,
            y=y,
            kf=kf,
        )
        if score < best_score:
            best_family = "mlp"
            best_params = params
            best_score = score

    return best_family, best_params, best_score


def jitter_params(family: str, params: dict, rng: np.random.Generator, dim: int) -> dict:
    if family == "svr":
        return {
            "C": float(np.clip(params["C"] * (10 ** rng.uniform(-0.3, 0.3)), 1e-2, 1e4)),
            "gamma": float(np.clip(params["gamma"] * (10 ** rng.uniform(-0.35, 0.35)), 1e-4, 5.0)),
            "epsilon": float(
                np.clip(params["epsilon"] * (10 ** rng.uniform(-0.35, 0.35)), 1e-5, 0.8)
            ),
        }

    # MLP jitter
    layers = list(params["hidden_layer_sizes"])
    new_layers: list[int] = []
    for w in layers:
        w_new = int(np.clip(round(w * rng.uniform(0.8, 1.2)), 16, 220))
        new_layers.append(w_new)
    if len(new_layers) == 0:
        new_layers = [max(24, 8 * dim), max(24, 8 * dim)]
    return {
        "hidden_layer_sizes": tuple(new_layers),
        "activation": params["activation"],
        "alpha": float(np.clip(params["alpha"] * (10 ** rng.uniform(-0.4, 0.4)), 1e-7, 1e-1)),
        "learning_rate_init": float(
            np.clip(params["learning_rate_init"] * (10 ** rng.uniform(-0.35, 0.35)), 1e-5, 2e-2)
        ),
        "early_stopping": params["early_stopping"],
    }


def fit_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    family: str,
    params: dict,
    rng: np.random.Generator,
    ensemble_size: int,
) -> list[Pipeline]:
    models: list[Pipeline] = []
    n = x.shape[0]
    dim = x.shape[1]
    for _ in range(ensemble_size):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        p = jitter_params(family=family, params=params, rng=rng, dim=dim)
        model = build_svr(p) if family == "svr" else build_mlp(p, rng, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(xb, yb)
        models.append(model)
    return models


def propose_query(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_svr: int,
    n_mlp: int,
    ensemble_size: int,
) -> tuple[np.ndarray, str]:
    family, best_params, cv_mae_score = tune_model_family(
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
        params=best_params,
        rng=rng,
        ensemble_size=ensemble_size,
    )

    dim = x.shape[1]
    n_obs = x.shape[0]
    n_global = 90_000 if dim <= 4 else 130_000
    n_local = 18_000 if dim <= 4 else 26_000

    global_candidates = rng.random((n_global, dim))

    # Local refinements around top points (exploitation).
    top_k = min(max(6, n_obs // 3), n_obs)
    anchor_idx = np.argsort(y)[-top_k:]
    anchors = x[anchor_idx]
    per_anchor = int(np.ceil(n_local / top_k))
    local_scale = 0.032 + 0.06 / np.sqrt(dim)
    local_parts: list[np.ndarray] = []
    for anchor in anchors:
        noise = rng.normal(0.0, local_scale, size=(per_anchor, dim))
        local_parts.append(np.clip(anchor + noise, 0.0, 1.0))
    local_candidates = np.vstack(local_parts)[:n_local]
    candidates = np.vstack([global_candidates, local_candidates])

    preds = np.column_stack([m.predict(candidates) for m in models])
    mean = preds.mean(axis=1)
    std = preds.std(axis=1)

    y_std = float(np.std(y))
    norm_cv = cv_mae_score / (y_std + 1e-9)
    # Less confident fit -> more exploration weight.
    beta = 1.2 + 0.12 * dim + 0.45 * np.clip(norm_cv, 0.0, 2.0)
    acquisition = mean + beta * std

    deltas = candidates[:, None, :] - x[None, :, :]
    min_dist = np.sqrt((deltas * deltas).sum(axis=2)).min(axis=1)
    novelty = 0.045 * np.log1p(min_dist)
    acquisition += novelty

    eps = 0.025 * np.sqrt(dim)
    acquisition[min_dist < eps] = -np.inf

    best_idx = int(np.argmax(acquisition))
    query = candidates[best_idx]

    report = (
        f"model={family} cv_mae={cv_mae_score:.6f} "
        f"norm_cv={norm_cv:.4f} beta={beta:.4f}"
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
    parser.add_argument("--seed", type=int, default=718, help="Random seed")
    parser.add_argument("--ensemble-size", type=int, default=10, help="Ensemble size")
    parser.add_argument("--n-svr", type=int, default=14, help="SVR random-search trials")
    parser.add_argument("--n-mlp", type=int, default=10, help="MLP random-search trials")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/module18_week7_queries.txt"),
        help="Output query file",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("src/module18_week7_tuning_report.txt"),
        help="Output tuning diagnostics file",
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
        x_next, report = propose_query(
            x=x,
            y=y,
            rng=rng,
            n_svr=args.n_svr,
            n_mlp=args.n_mlp,
            ensemble_size=args.ensemble_size,
        )
        query_lines.append(f"function_{func_id}: {format_query(x_next)}")
        report_lines.append(f"function_{func_id}: {report}")

    args.output.write_text("\n".join(query_lines) + "\n")
    args.report.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
