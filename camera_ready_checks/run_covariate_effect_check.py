"""
Illustrative covariate-effect target check for the MLHC camera-ready appendix.

This is intentionally a small appendix-only experiment.  It asks whether the
same summary-based NPE idea can target a simple current-study covariate effect
when the summary vector is made target-aware by including source-specific OLS
slope estimates.

The target is beta_1 in the concurrent outcome model

    Y_c = theta + X_c beta + error.

The external sample may have covariate shift and/or additive outcome drift.
The network is trained on simulated current/external pairs and returns a
Gaussian posterior approximation for beta_1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainConfig:
    n_sims: int = 5000
    n_epochs: int = 100
    batch_size: int = 64
    n_external: int = 200
    p: int = 6
    target_index: int = 0
    seed: int = 123


def compound_sigma(p: int, rho: float = 0.5) -> np.ndarray:
    sigma = np.ones((p, p)) * rho
    np.fill_diagonal(sigma, 1.0)
    return sigma


def fit_ols_with_se(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return OLS coefficients, standard errors, and residual SD."""
    design = np.column_stack([np.ones(X.shape[0]), X])
    xtx = design.T @ design
    xty = design.T @ y
    beta_hat = np.linalg.pinv(xtx) @ xty
    resid = y - design @ beta_hat
    dof = max(design.shape[0] - design.shape[1], 1)
    sigma2 = float((resid @ resid) / dof)
    cov = sigma2 * np.linalg.pinv(xtx)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    return beta_hat, se, float(np.sqrt(sigma2))


def overlap_summaries(X_c: np.ndarray, X_e: np.ndarray) -> np.ndarray:
    """Simple propensity-overlap summaries for source membership."""
    X = np.vstack([X_c, X_e])
    s = np.r_[np.ones(X_c.shape[0]), np.zeros(X_e.shape[0])]
    try:
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X, s)
        e_external = clf.predict_proba(X_e)[:, 1]
    except Exception:
        e_external = np.repeat(X_c.shape[0] / (X_c.shape[0] + X_e.shape[0]), X_e.shape[0])
    w = 2.0 * np.sqrt(np.clip(e_external * (1.0 - e_external), 1e-8, None))
    ess = (w.sum() ** 2) / np.maximum((w**2).sum(), 1e-8)
    return np.array([
        float(np.mean(e_external)),
        float(np.std(e_external)),
        float(np.mean(w)),
        float(ess),
        float(ess / max(X_e.shape[0], 1)),
    ])


def compute_summary(X_c: np.ndarray, y_c: np.ndarray, X_e: np.ndarray, y_e: np.ndarray) -> np.ndarray:
    """
    Target-aware summary vector.

    We include source-specific OLS slopes because the target is a slope.
    This differs from the main scalar-risk experiment, where the target was
    an intercept/study-effect parameter.
    """
    beta_c, se_c, resid_c = fit_ols_with_se(X_c, y_c)
    beta_e, se_e, resid_e = fit_ols_with_se(X_e, y_e)
    beta_diff = beta_c - beta_e

    parts = [
        np.array([X_c.shape[0], X_e.shape[0], np.log1p(X_c.shape[0]), np.log1p(X_e.shape[0])]),
        X_c.mean(axis=0),
        X_c.std(axis=0),
        X_e.mean(axis=0),
        X_e.std(axis=0),
        np.array([y_c.mean(), y_c.std(), y_e.mean(), y_e.std(), y_c.mean() - y_e.mean()]),
        beta_c,
        beta_e,
        beta_diff,
        se_c,
        se_e,
        np.array([resid_c, resid_e]),
        overlap_summaries(X_c, X_e),
    ]
    return np.concatenate(parts).astype(float)


def sample_training_pair(rng: np.random.Generator, p: int, n_external: int) -> Tuple[np.ndarray, float]:
    """Generate one training pair and return (summary, beta_1 target)."""
    sigma_c = compound_sigma(p)
    n_c = int(rng.choice([50, 100, 200, 500]))
    mu_c_scalar = rng.uniform(0.5, 1.5)
    mu_c = np.ones(p) * mu_c_scalar

    theta = rng.normal(1.0, 1.0)
    beta_c = rng.normal(0.3, 0.2, size=p)
    sigma = rng.uniform(0.5, 2.0)

    shift_type = int(rng.choice([0, 1, 2, 3, 4], p=[0.25, 0.25, 0.25, 0.15, 0.10]))
    mu_e = mu_c.copy()
    sigma_e_mat = sigma_c.copy()
    beta_e = beta_c.copy()
    delta_e = 0.0
    sigma_e = sigma

    if shift_type in (1, 3):
        direction = rng.choice([-1.0, 1.0])
        magnitude = rng.uniform(0.5, 3.0)
        mu_e = mu_c * magnitude if direction > 0 else mu_c / magnitude
    if shift_type in (2, 3):
        delta_e = rng.uniform(-2.0, 2.0)
    if shift_type == 4:
        diag_scale = np.exp(rng.uniform(-0.7, 0.7, size=p))
        d = np.diag(diag_scale)
        sigma_e_mat = d @ sigma_c @ d
        beta_e = beta_c + rng.normal(0.0, 0.3, size=p)
        sigma_e = sigma * np.exp(rng.uniform(-0.5, 0.5))

    X_c = rng.multivariate_normal(mu_c, sigma_c, size=n_c)
    y_c = theta + X_c @ beta_c + rng.normal(0.0, sigma, size=n_c)
    X_e = rng.multivariate_normal(mu_e, sigma_e_mat, size=n_external)
    y_e = theta + delta_e + X_e @ beta_e + rng.normal(0.0, sigma_e, size=n_external)

    return compute_summary(X_c, y_c, X_e, y_e), float(beta_c[0])


class GaussianNPE(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gaussian_nll(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mean = out[:, 0]
    log_sd = torch.clamp(out[:, 1], min=-6.0, max=3.0)
    sd = torch.exp(log_sd)
    return 0.5 * ((target - mean) / sd) ** 2 + log_sd


def train_npe(cfg: TrainConfig) -> Tuple[GaussianNPE, StandardScaler, float]:
    rng = np.random.default_rng(cfg.seed)
    summaries: List[np.ndarray] = []
    targets: List[float] = []
    for i in range(cfg.n_sims):
        s, beta_target = sample_training_pair(rng, cfg.p, cfg.n_external)
        summaries.append(s)
        targets.append(beta_target)
        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1}/{cfg.n_sims} training pairs")

    X = np.vstack(summaries)
    y = np.asarray(targets)
    n_train = int(0.8 * len(y))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:n_train])
    X_val = scaler.transform(X[n_train:])
    y_train = y[:n_train]
    y_val = y[n_train:]

    model = GaussianNPE(X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val = np.inf
    patience = 20
    bad = 0

    for epoch in range(cfg.n_epochs):
        model.train()
        idx = rng.permutation(n_train)
        losses = []
        for start in range(0, n_train, cfg.batch_size):
            batch = idx[start : start + cfg.batch_size]
            xb = torch.tensor(X_train[batch], dtype=torch.float32)
            yb = torch.tensor(y_train[batch], dtype=torch.float32)
            loss = gaussian_nll(model(xb), yb).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val, dtype=torch.float32)
            yv = torch.tensor(y_val, dtype=torch.float32)
            val = float(gaussian_nll(model(xv), yv).mean())
            pred = model(xv)[:, 0]
            rmse = float(torch.sqrt(torch.mean((pred - yv) ** 2)))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train={np.mean(losses):.4f}, val={val:.4f}, val_rmse={rmse:.4f}")

        if val < best_val:
            best_val = val
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.eval()
    with torch.no_grad():
        xv = torch.tensor(X_val, dtype=torch.float32)
        out = model(xv).numpy()
    val_mean = out[:, 0]
    val_sd = np.exp(np.clip(out[:, 1], -6.0, 3.0))
    val_z = np.abs(y_val - val_mean) / np.maximum(val_sd, 1e-8)
    sd_scale = max(1.0, float(np.quantile(val_z, 0.95) / 1.96))
    print(f"Validation SD calibration factor: {sd_scale:.3f}")

    return model, scaler, sd_scale


def scenario_params(scenario: str, mu_c: float) -> Tuple[float, float]:
    if scenario == "SC1":
        return mu_c, 0.0
    if scenario == "SC2":
        return 1.5 * mu_c, 0.0
    if scenario == "SC3":
        return mu_c, 0.75
    if scenario == "SC4":
        return 2.0 * mu_c, 0.0
    if scenario == "SC5":
        return mu_c, 1.5
    if scenario == "SC6":
        return 2.0 * mu_c, 1.5
    raise ValueError(f"Unknown scenario {scenario}")


def evaluate(
    model: GaussianNPE,
    scaler: StandardScaler,
    n_reps: int,
    n_concurrent_values: List[int],
    n_external: int,
    p: int,
    seed: int,
    sd_scale: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sigma_mat = compound_sigma(p)
    beta_true = np.array([0.5, 0.3, 0.2, 0.4, 0.3, 0.25])[:p]
    theta_true = 1.0
    sigma = 1.0
    mu_c_scalar = 1.0
    mu_c = np.ones(p) * mu_c_scalar
    target = float(beta_true[0])

    records = []
    for n_c in n_concurrent_values:
        for scenario in ["SC1", "SC2", "SC3", "SC4", "SC5", "SC6"]:
            mu_e_scalar, delta_e = scenario_params(scenario, mu_c_scalar)
            mu_e = np.ones(p) * mu_e_scalar
            for rep in range(n_reps):
                X_c = rng.multivariate_normal(mu_c, sigma_mat, size=n_c)
                y_c = theta_true + X_c @ beta_true + rng.normal(0.0, sigma, size=n_c)
                X_e = rng.multivariate_normal(mu_e, sigma_mat, size=n_external)
                y_e = theta_true + delta_e + X_e @ beta_true + rng.normal(0.0, sigma, size=n_external)

                summary = compute_summary(X_c, y_c, X_e, y_e)
                xs = scaler.transform(summary.reshape(1, -1))
                model.eval()
                with torch.no_grad():
                    out = model(torch.tensor(xs, dtype=torch.float32)).numpy()[0]
                mean = float(out[0])
                sd = float(np.exp(np.clip(out[1], -6.0, 3.0)) * sd_scale)
                ci_low = mean - 1.96 * sd
                ci_high = mean + 1.96 * sd

                ols_beta, ols_se, _ = fit_ols_with_se(X_c, y_c)
                ols_mean = float(ols_beta[1])
                ols_sd = float(ols_se[1])

                records.append({
                    "method": "NPE-beta1",
                    "n_concurrent": n_c,
                    "scenario": scenario,
                    "rep": rep,
                    "target": target,
                    "estimate": mean,
                    "posterior_sd": sd,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "covered": ci_low <= target <= ci_high,
                })
                records.append({
                    "method": "Concurrent OLS",
                    "n_concurrent": n_c,
                    "scenario": scenario,
                    "rep": rep,
                    "target": target,
                    "estimate": ols_mean,
                    "posterior_sd": ols_sd,
                    "ci_low": ols_mean - 1.96 * ols_sd,
                    "ci_high": ols_mean + 1.96 * ols_sd,
                    "covered": (ols_mean - 1.96 * ols_sd) <= target <= (ols_mean + 1.96 * ols_sd),
                })
    return pd.DataFrame.from_records(records)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in raw.groupby(["method", "n_concurrent", "scenario"]):
        method, n_c, scenario = keys
        err = g["estimate"].to_numpy() - g["target"].to_numpy()
        rows.append({
            "method": method,
            "n_concurrent": n_c,
            "scenario": scenario,
            "bias": float(np.mean(err)),
            "abs_bias": float(abs(np.mean(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "coverage": float(g["covered"].mean()),
            "mean_interval_width": float(np.mean(g["ci_high"] - g["ci_low"])),
        })
    return pd.DataFrame(rows).sort_values(["n_concurrent", "scenario", "method"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=5000)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--b-reps", type=int, default=50)
    parser.add_argument("--n-concurrent-values", type=int, nargs="+", default=[100])
    parser.add_argument("--n-external", type=int, default=200)
    parser.add_argument("--p", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="covariate_effect_results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        n_sims=args.n_sims,
        n_epochs=args.n_epochs,
        n_external=args.n_external,
        p=args.p,
        seed=args.seed,
    )
    model, scaler, sd_scale = train_npe(cfg)
    raw = evaluate(
        model=model,
        scaler=scaler,
        n_reps=args.b_reps,
        n_concurrent_values=args.n_concurrent_values,
        n_external=args.n_external,
        p=args.p,
        seed=args.seed + 1000,
        sd_scale=sd_scale,
    )
    summary = summarize(raw)

    raw_path = out_dir / "covariate_effect_raw.csv"
    summary_path = out_dir / "covariate_effect_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nCovariate-effect target summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
