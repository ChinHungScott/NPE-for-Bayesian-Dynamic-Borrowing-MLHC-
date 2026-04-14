from __future__ import annotations

"""
ADNI borrowing baseline comparison on the same target scale as NPE.

This script loads the ADNI subject-level application CSV and compares several
borrowing estimators for the current-cohort log-odds risk parameter.  All
methods use the same prognostic offset model

    Y_i ~ Bernoulli(sigmoid(X_i beta_anchor + theta)).

The first output keeps the previous weighted-offset MLE baselines.  The second
output adds posterior-based Bayesian comparators:

  - concurrent-only posterior for theta_c;
  - pooled posterior with one common theta;
  - fixed power-prior posterior with external likelihood weight a0;
  - commensurate joint posterior for (theta_c, theta_e);
  - NPE posterior mean and interval from the main ADNI run.

The commensurate posterior is computed on a deterministic two-dimensional grid,
so the resulting joint posterior sample is reproducible and does not require
MCMC tuning.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


COVARIATES = [
    "age",
    "female",
    "education",
    "apoe4_count",
    "mmse_bl",
    "adas13_bl",
    "cdrsb_bl",
    "faq_bl",
]


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x)
    out = np.empty_like(x_arr, dtype=float)
    pos = x_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    if np.ndim(x) == 0:
        return float(out)
    return out


def safe_logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def log_one_minus_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, x)


def load_subject_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    missing = [c for c in COVARIATES + ["conversion_24m", "source"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    is_concurrent = df["source"].astype(str).str.lower().eq("concurrent").to_numpy()
    is_external = df["source"].astype(str).str.lower().eq("external").to_numpy()
    if not is_concurrent.any() or not is_external.any():
        raise ValueError("The CSV must contain both source='concurrent' and source='external'.")

    X = df[COVARIATES].to_numpy(dtype=float)
    y = df["conversion_24m"].to_numpy(dtype=float)
    return X[is_concurrent], y[is_concurrent], X[is_external], y[is_external]


def fit_beta_anchor(
    X_c: np.ndarray,
    y_c: np.ndarray,
    X_e: np.ndarray,
    y_e: np.ndarray,
) -> Tuple[np.ndarray, SimpleImputer]:
    """Match the ADNI NPE beta-anchor convention."""
    X_all = np.vstack([X_c, X_e])
    y_all = np.concatenate([y_c, y_e])

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, fit_intercept=True)),
    ])
    model.fit(X_all, y_all)

    imputer: SimpleImputer = model.named_steps["imputer"]
    scaler: StandardScaler = model.named_steps["scaler"]
    clf: LogisticRegression = model.named_steps["clf"]
    beta_anchor = clf.coef_.ravel() / scaler.scale_
    return beta_anchor.astype(float), imputer


def fit_offset_theta(
    y: np.ndarray,
    offset: np.ndarray,
    weights: np.ndarray | None = None,
    init: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> Dict[str, float]:
    """Weighted logistic offset MLE for one scalar theta."""
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if weights is None:
        weights = np.ones_like(y, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if init is None:
        init = safe_logit(np.average(y, weights=weights))

    theta = float(init)
    for _ in range(max_iter):
        p = sigmoid(offset + theta)
        grad = float(np.sum(weights * (y - p)))
        info = float(np.sum(weights * p * (1.0 - p)) + 1e-10)
        step = grad / info
        theta_new = theta + step
        if abs(theta_new - theta) < tol:
            theta = float(theta_new)
            break
        theta = float(theta_new)

    p = sigmoid(offset + theta)
    info = float(np.sum(weights * p * (1.0 - p)) + 1e-10)
    se = float(np.sqrt(1.0 / info))
    return {
        "theta": theta,
        "std": se,
        "ci_low": theta - 1.96 * se,
        "ci_high": theta + 1.96 * se,
    }


def fit_ps_scores(X_c: np.ndarray, X_e: np.ndarray, clip: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    X_all = np.vstack([X_c, X_e])
    source = np.concatenate([np.ones(X_c.shape[0]), np.zeros(X_e.shape[0])])
    ps_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000)),
    ])
    ps_model.fit(X_all, source)
    ps = ps_model.predict_proba(X_all)[:, 1]
    ps = np.clip(ps, clip, 1.0 - clip)
    return ps[: X_c.shape[0]], ps[X_c.shape[0] :]


def pspower_external_weights(X_c: np.ndarray, X_e: np.ndarray, k: int = 5) -> np.ndarray:
    n_c = X_c.shape[0]
    n_e = X_e.shape[0]
    e_c, e_e = fit_ps_scores(X_c, X_e)

    quantiles = np.quantile(e_c, np.linspace(0.0, 1.0, k + 1))
    quantiles[0] -= 1e-9
    quantiles[-1] += 1e-9
    strata_c = np.clip(np.digitize(e_c, quantiles) - 1, 0, k - 1)
    strata_e = np.clip(np.digitize(e_e, quantiles) - 1, 0, k - 1)

    weights_e = np.zeros(n_e, dtype=float)
    for s in range(k):
        n_c_s = int(np.sum(strata_c == s))
        n_e_s = int(np.sum(strata_e == s))
        alpha_s = min(1.0, (n_c_s / n_c) / (n_e_s / n_e + 1e-9))
        weights_e[strata_e == s] = alpha_s
    return weights_e


def iw_external_weights(X_c: np.ndarray, X_e: np.ndarray, clip_min: float = 0.01) -> np.ndarray:
    mu_c = X_c.mean(axis=0)
    cov_c = np.cov(X_c.T) + 1e-6 * np.eye(X_c.shape[1])
    try:
        cov_inv = np.linalg.inv(cov_c)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_c)

    diff = X_e - mu_c
    d2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    d = np.sqrt(np.maximum(d2, 0.0))
    h = float(np.median(d)) + 1e-9
    return np.clip(np.exp(-d / h), clip_min, 1.0)


def empirical_logit_ci(y: np.ndarray) -> Dict[str, float]:
    p = float(np.mean(y))
    theta = safe_logit(p)
    n = len(y)
    se = float(np.sqrt(1.0 / max(n * p * (1.0 - p), 1e-10)))
    return {
        "theta": theta,
        "std": se,
        "ci_low": theta - 1.96 * se,
        "ci_high": theta + 1.96 * se,
    }


def add_weighted_row(
    rows: list[Dict[str, float | str]],
    method: str,
    est: Dict[str, float],
    target_theta: float,
    external_weight_sum: float | None = None,
    external_weight_mean: float | None = None,
) -> None:
    rows.append({
        "method": method,
        "theta": est["theta"],
        "std": est["std"],
        "ci_low": est["ci_low"],
        "ci_high": est["ci_high"],
        "implied_risk": float(sigmoid(est["theta"])),
        "abs_diff_from_concurrent_logit": abs(est["theta"] - target_theta),
        "external_weight_sum": np.nan if external_weight_sum is None else external_weight_sum,
        "external_weight_mean": np.nan if external_weight_mean is None else external_weight_mean,
    })


def make_weighted_comparison(
    y_c: np.ndarray,
    y_e: np.ndarray,
    X_c: np.ndarray,
    X_e: np.ndarray,
    offset_c: np.ndarray,
    offset_e: np.ndarray,
    npe_mean: float,
    npe_std: float,
    npe_ci_low: float,
    npe_ci_high: float,
) -> pd.DataFrame:
    target = empirical_logit_ci(y_c)
    target_theta = target["theta"]

    rows: list[Dict[str, float | str]] = []
    add_weighted_row(rows, "Empirical concurrent logit", target, target_theta)
    add_weighted_row(rows, "Empirical external logit", empirical_logit_ci(y_e), target_theta)
    add_weighted_row(rows, "Empirical pooled logit", empirical_logit_ci(np.concatenate([y_c, y_e])), target_theta)
    add_weighted_row(
        rows,
        "Offset MLE, concurrent",
        fit_offset_theta(y_c, offset_c, init=target_theta),
        target_theta,
    )

    y_all = np.concatenate([y_c, y_e])
    offset_all = np.concatenate([offset_c, offset_e])
    add_weighted_row(
        rows,
        "Offset MLE, pooled",
        fit_offset_theta(y_all, offset_all, init=safe_logit(np.mean(y_all))),
        target_theta,
        external_weight_sum=float(len(y_e)),
        external_weight_mean=1.0,
    )

    w_ps = pspower_external_weights(X_c, X_e)
    add_weighted_row(
        rows,
        "PSPower-weighted offset MLE",
        fit_offset_theta(
            y_all,
            offset_all,
            weights=np.concatenate([np.ones_like(y_c), w_ps]),
            init=target_theta,
        ),
        target_theta,
        external_weight_sum=float(np.sum(w_ps)),
        external_weight_mean=float(np.mean(w_ps)),
    )

    w_iw = iw_external_weights(X_c, X_e)
    add_weighted_row(
        rows,
        "IW-weighted offset MLE",
        fit_offset_theta(
            y_all,
            offset_all,
            weights=np.concatenate([np.ones_like(y_c), w_iw]),
            init=target_theta,
        ),
        target_theta,
        external_weight_sum=float(np.sum(w_iw)),
        external_weight_mean=float(np.mean(w_iw)),
    )

    add_weighted_row(
        rows,
        "NPE posterior mean",
        {
            "theta": npe_mean,
            "std": npe_std,
            "ci_low": npe_ci_low,
            "ci_high": npe_ci_high,
        },
        target_theta,
    )
    return pd.DataFrame(rows)


def logistic_offset_loglik_grid(theta_grid: np.ndarray, y: np.ndarray, offset: np.ndarray) -> np.ndarray:
    eta = theta_grid[:, None] + offset[None, :]
    return (y[None, :] * log_sigmoid(eta) + (1.0 - y[None, :]) * log_one_minus_sigmoid(eta)).sum(axis=1)


def summarize_grid(theta_grid: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    probs = probs / probs.sum()
    cdf = np.cumsum(probs)
    mean = float(np.sum(theta_grid * probs))
    std = float(np.sqrt(np.sum((theta_grid - mean) ** 2 * probs)))
    ci_low = float(np.interp(0.025, cdf, theta_grid))
    ci_high = float(np.interp(0.975, cdf, theta_grid))
    return {
        "theta": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "implied_risk": float(sigmoid(mean)),
    }


def posterior_1d(
    theta_grid: np.ndarray,
    loglik: np.ndarray,
    prior_sd: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    log_prior = -0.5 * (theta_grid / prior_sd) ** 2
    log_post = loglik + log_prior
    log_post -= np.max(log_post)
    probs = np.exp(log_post)
    probs /= probs.sum()
    return summarize_grid(theta_grid, probs), probs


def commensurate_joint_posterior(
    theta_grid: np.ndarray,
    loglik_c: np.ndarray,
    loglik_e: np.ndarray,
    prior_sd: float,
    commensurate_sd: float,
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray]:
    theta_c = theta_grid[:, None]
    theta_e = theta_grid[None, :]

    log_prior_c = -0.5 * (theta_grid / prior_sd) ** 2
    log_prior_e_given_c = -0.5 * ((theta_e - theta_c) / commensurate_sd) ** 2
    log_post = (
        loglik_c[:, None]
        + loglik_e[None, :]
        + log_prior_c[:, None]
        + log_prior_e_given_c
    )
    log_post -= np.max(log_post)
    probs = np.exp(log_post)
    probs /= probs.sum()

    probs_c = probs.sum(axis=1)
    probs_e = probs.sum(axis=0)
    return summarize_grid(theta_grid, probs_c), summarize_grid(theta_grid, probs_e), probs


def add_posterior_row(
    rows: list[Dict[str, float | str]],
    method: str,
    parameter: str,
    summary: Dict[str, float],
    target_theta: float,
    note: str = "",
) -> None:
    rows.append({
        "method": method,
        "parameter": parameter,
        "theta_mean": summary["theta"],
        "posterior_sd": summary["std"],
        "ci_low": summary["ci_low"],
        "ci_high": summary["ci_high"],
        "implied_risk": summary["implied_risk"],
        "abs_diff_from_concurrent_logit": abs(summary["theta"] - target_theta),
        "note": note,
    })


def make_bayesian_comparison(
    y_c: np.ndarray,
    y_e: np.ndarray,
    offset_c: np.ndarray,
    offset_e: np.ndarray,
    prior_sd: float,
    power_a0: float,
    commensurate_sd: float,
    grid_min: float,
    grid_max: float,
    grid_n: int,
    rng: np.random.Generator,
    posterior_sample_n: int,
    npe_mean: float,
    npe_std: float,
    npe_ci_low: float,
    npe_ci_high: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    theta_grid = np.linspace(grid_min, grid_max, grid_n)
    target_theta = empirical_logit_ci(y_c)["theta"]
    loglik_c = logistic_offset_loglik_grid(theta_grid, y_c, offset_c)
    loglik_e = logistic_offset_loglik_grid(theta_grid, y_e, offset_e)

    rows: list[Dict[str, float | str]] = []
    densities: list[pd.DataFrame] = []
    samples: list[pd.DataFrame] = []

    summary_c, probs_c = posterior_1d(theta_grid, loglik_c, prior_sd=prior_sd)
    add_posterior_row(rows, "Concurrent-only posterior", "theta_c", summary_c, target_theta)
    densities.append(pd.DataFrame({"method": "Concurrent-only posterior", "theta": theta_grid, "density": probs_c}))
    samples.append(pd.DataFrame({
        "method": "Concurrent-only posterior",
        "theta_c": rng.choice(theta_grid, size=posterior_sample_n, replace=True, p=probs_c),
        "theta_e": np.nan,
    }))

    summary_pooled, probs_pooled = posterior_1d(theta_grid, loglik_c + loglik_e, prior_sd=prior_sd)
    add_posterior_row(rows, "Pooled posterior", "theta_c = theta_e", summary_pooled, target_theta)
    densities.append(pd.DataFrame({"method": "Pooled posterior", "theta": theta_grid, "density": probs_pooled}))
    samples.append(pd.DataFrame({
        "method": "Pooled posterior",
        "theta_c": rng.choice(theta_grid, size=posterior_sample_n, replace=True, p=probs_pooled),
        "theta_e": np.nan,
    }))

    summary_power, probs_power = posterior_1d(theta_grid, loglik_c + power_a0 * loglik_e, prior_sd=prior_sd)
    add_posterior_row(
        rows,
        f"Power-prior posterior (a0={power_a0:g})",
        "theta_c = theta_e",
        summary_power,
        target_theta,
        note="external likelihood downweighted",
    )
    densities.append(pd.DataFrame({
        "method": f"Power-prior posterior (a0={power_a0:g})",
        "theta": theta_grid,
        "density": probs_power,
    }))
    samples.append(pd.DataFrame({
        "method": f"Power-prior posterior (a0={power_a0:g})",
        "theta_c": rng.choice(theta_grid, size=posterior_sample_n, replace=True, p=probs_power),
        "theta_e": np.nan,
    }))

    summary_comm_c, summary_comm_e, probs_joint = commensurate_joint_posterior(
        theta_grid,
        loglik_c,
        loglik_e,
        prior_sd=prior_sd,
        commensurate_sd=commensurate_sd,
    )
    add_posterior_row(
        rows,
        f"Commensurate posterior (sd={commensurate_sd:g})",
        "theta_c",
        summary_comm_c,
        target_theta,
        note="joint posterior over current and external risk",
    )
    add_posterior_row(
        rows,
        f"Commensurate posterior (sd={commensurate_sd:g})",
        "theta_e",
        summary_comm_e,
        target_theta,
        note="external-risk marginal",
    )
    densities.append(pd.DataFrame({
        "method": f"Commensurate posterior theta_c (sd={commensurate_sd:g})",
        "theta": theta_grid,
        "density": probs_joint.sum(axis=1),
    }))

    flat_probs = probs_joint.ravel()
    flat_idx = rng.choice(flat_probs.size, size=posterior_sample_n, replace=True, p=flat_probs)
    idx_c, idx_e = np.unravel_index(flat_idx, probs_joint.shape)
    samples.append(pd.DataFrame({
        "method": f"Commensurate posterior (sd={commensurate_sd:g})",
        "theta_c": theta_grid[idx_c],
        "theta_e": theta_grid[idx_e],
    }))

    npe_summary = {
        "theta": npe_mean,
        "std": npe_std,
        "ci_low": npe_ci_low,
        "ci_high": npe_ci_high,
        "implied_risk": float(sigmoid(npe_mean)),
    }
    add_posterior_row(rows, "NPE posterior", "theta_c", npe_summary, target_theta)
    npe_mass = np.exp(-0.5 * ((theta_grid - npe_mean) / npe_std) ** 2)
    npe_mass /= npe_mass.sum()
    densities.append(pd.DataFrame({
        "method": "NPE posterior normal approximation",
        "theta": theta_grid,
        "density": npe_mass,
    }))
    samples.append(pd.DataFrame({
        "method": "NPE posterior normal approximation",
        "theta_c": rng.normal(npe_mean, npe_std, size=posterior_sample_n),
        "theta_e": np.nan,
    }))

    return pd.DataFrame(rows), pd.concat(densities, ignore_index=True), pd.concat(samples, ignore_index=True)


def make_risk_logit_bayesian_comparison(
    y_c: np.ndarray,
    y_e: np.ndarray,
    prior_sd: float,
    power_a0: float,
    commensurate_sd: float,
    grid_min: float,
    grid_max: float,
    grid_n: int,
    rng: np.random.Generator,
    posterior_sample_n: int,
    npe_mean: float,
    npe_std: float,
    npe_ci_low: float,
    npe_ci_high: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bayesian comparison on cohort-level logit-risk scale.

    This is the cleanest comparator for the ADNI application if the NPE output
    is interpreted as the current cohort's risk-level logit.  It avoids mixing
    that NPE quantity with a covariate-offset intercept.
    """
    theta_grid = np.linspace(grid_min, grid_max, grid_n)
    offset_c = np.zeros_like(y_c, dtype=float)
    offset_e = np.zeros_like(y_e, dtype=float)
    target_theta = empirical_logit_ci(y_c)["theta"]

    loglik_c = logistic_offset_loglik_grid(theta_grid, y_c, offset_c)
    loglik_e = logistic_offset_loglik_grid(theta_grid, y_e, offset_e)

    rows: list[Dict[str, float | str]] = []
    densities: list[pd.DataFrame] = []
    samples: list[pd.DataFrame] = []

    summary_c, probs_c = posterior_1d(theta_grid, loglik_c, prior_sd=prior_sd)
    add_posterior_row(rows, "Concurrent-only risk-logit posterior", "theta_c", summary_c, target_theta)
    densities.append(pd.DataFrame({
        "method": "Concurrent-only risk-logit posterior",
        "theta": theta_grid,
        "density": probs_c,
    }))
    samples.append(pd.DataFrame({
        "method": "Concurrent-only risk-logit posterior",
        "theta_c": rng.choice(theta_grid, size=posterior_sample_n, replace=True, p=probs_c),
        "theta_e": np.nan,
    }))

    summary_pooled, probs_pooled = posterior_1d(theta_grid, loglik_c + loglik_e, prior_sd=prior_sd)
    add_posterior_row(rows, "Pooled risk-logit posterior", "theta_c = theta_e", summary_pooled, target_theta)
    densities.append(pd.DataFrame({
        "method": "Pooled risk-logit posterior",
        "theta": theta_grid,
        "density": probs_pooled,
    }))
    samples.append(pd.DataFrame({
        "method": "Pooled risk-logit posterior",
        "theta_c": rng.choice(theta_grid, size=posterior_sample_n, replace=True, p=probs_pooled),
        "theta_e": np.nan,
    }))

    summary_power, probs_power = posterior_1d(theta_grid, loglik_c + power_a0 * loglik_e, prior_sd=prior_sd)
    add_posterior_row(
        rows,
        f"Power-prior risk-logit posterior (a0={power_a0:g})",
        "theta_c = theta_e",
        summary_power,
        target_theta,
        note="external likelihood downweighted",
    )
    densities.append(pd.DataFrame({
        "method": f"Power-prior risk-logit posterior (a0={power_a0:g})",
        "theta": theta_grid,
        "density": probs_power,
    }))
    samples.append(pd.DataFrame({
        "method": f"Power-prior risk-logit posterior (a0={power_a0:g})",
        "theta_c": rng.choice(theta_grid, size=posterior_sample_n, replace=True, p=probs_power),
        "theta_e": np.nan,
    }))

    summary_comm_c, summary_comm_e, probs_joint = commensurate_joint_posterior(
        theta_grid,
        loglik_c,
        loglik_e,
        prior_sd=prior_sd,
        commensurate_sd=commensurate_sd,
    )
    add_posterior_row(
        rows,
        f"Commensurate risk-logit posterior (sd={commensurate_sd:g})",
        "theta_c",
        summary_comm_c,
        target_theta,
        note="joint posterior over current and external risk logits",
    )
    add_posterior_row(
        rows,
        f"Commensurate risk-logit posterior (sd={commensurate_sd:g})",
        "theta_e",
        summary_comm_e,
        target_theta,
        note="external-risk marginal",
    )
    densities.append(pd.DataFrame({
        "method": f"Commensurate risk-logit posterior theta_c (sd={commensurate_sd:g})",
        "theta": theta_grid,
        "density": probs_joint.sum(axis=1),
    }))

    flat_probs = probs_joint.ravel()
    flat_idx = rng.choice(flat_probs.size, size=posterior_sample_n, replace=True, p=flat_probs)
    idx_c, idx_e = np.unravel_index(flat_idx, probs_joint.shape)
    samples.append(pd.DataFrame({
        "method": f"Commensurate risk-logit posterior (sd={commensurate_sd:g})",
        "theta_c": theta_grid[idx_c],
        "theta_e": theta_grid[idx_e],
    }))

    npe_summary = {
        "theta": npe_mean,
        "std": npe_std,
        "ci_low": npe_ci_low,
        "ci_high": npe_ci_high,
        "implied_risk": float(sigmoid(npe_mean)),
    }
    add_posterior_row(rows, "NPE posterior", "theta_c", npe_summary, target_theta)
    npe_mass = np.exp(-0.5 * ((theta_grid - npe_mean) / npe_std) ** 2)
    npe_mass /= npe_mass.sum()
    densities.append(pd.DataFrame({
        "method": "NPE posterior normal approximation",
        "theta": theta_grid,
        "density": npe_mass,
    }))
    samples.append(pd.DataFrame({
        "method": "NPE posterior normal approximation",
        "theta_c": rng.normal(npe_mean, npe_std, size=posterior_sample_n),
        "theta_e": np.nan,
    }))

    return pd.DataFrame(rows), pd.concat(densities, ignore_index=True), pd.concat(samples, ignore_index=True)


def write_weighted_latex_table(df: pd.DataFrame, out_path: Path) -> None:
    table = df.copy()
    table["Estimate"] = table["theta"].map(lambda x: f"{x:.3f}")
    table["95\\% interval"] = table.apply(lambda r: f"({r['ci_low']:.3f}, {r['ci_high']:.3f})", axis=1)
    table["Risk"] = table["implied_risk"].map(lambda x: f"{x:.3f}")
    table["Abs. diff."] = table["abs_diff_from_concurrent_logit"].map(lambda x: f"{x:.3f}")
    table["External weight"] = table["external_weight_sum"].map(
        lambda x: "" if pd.isna(x) else f"{x:.1f}"
    )
    table = table[["method", "Estimate", "95\\% interval", "Risk", "Abs. diff.", "External weight"]]
    table = table.rename(columns={"method": "Method"})
    out_path.write_text(table.to_latex(index=False, escape=False), encoding="utf-8")


def write_bayesian_latex_table(df: pd.DataFrame, out_path: Path) -> None:
    table = df.copy()
    table["Mean"] = table["theta_mean"].map(lambda x: f"{x:.3f}")
    table["SD"] = table["posterior_sd"].map(lambda x: f"{x:.3f}")
    table["95\\% interval"] = table.apply(lambda r: f"({r['ci_low']:.3f}, {r['ci_high']:.3f})", axis=1)
    table["Risk"] = table["implied_risk"].map(lambda x: f"{x:.3f}")
    table["Abs. diff."] = table["abs_diff_from_concurrent_logit"].map(lambda x: f"{x:.3f}")
    table = table[["method", "parameter", "Mean", "SD", "95\\% interval", "Risk", "Abs. diff."]]
    table = table.rename(columns={"method": "Method", "parameter": "Parameter"})
    out_path.write_text(table.to_latex(index=False, escape=False), encoding="utf-8")


def maybe_write_figures(
    density_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    out_dir: Path,
    target_theta: float,
    prefix: str = "adni_bayesian",
    title: str = "ADNI posterior comparison on the current-cohort risk parameter",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping posterior figures.")
        return

    plt.figure(figsize=(7.2, 4.2))
    for method, sub in density_df.groupby("method", sort=False):
        density = sub["density"].to_numpy()
        theta = sub["theta"].to_numpy()
        # Convert grid probability mass to an approximate density for plotting.
        dx = theta[1] - theta[0]
        plt.plot(theta, density / dx, label=method)
    plt.axvline(target_theta, color="black", linestyle="--", linewidth=1.0, label="Concurrent empirical logit")
    plt.xlabel(r"$\theta_c$")
    plt.ylabel("Posterior density")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_theta_posterior_overlay.pdf")
    plt.close()

    joint = sample_df[sample_df["theta_e"].notna()].copy()
    if not joint.empty:
        plt.figure(figsize=(4.8, 4.4))
        plot_df = joint.sample(n=min(2500, len(joint)), random_state=1)
        plt.scatter(plot_df["theta_c"], plot_df["theta_e"], s=4, alpha=0.18)
        plt.axvline(target_theta, color="black", linestyle="--", linewidth=1.0)
        plt.axhline(safe_logit(0.376855), color="gray", linestyle=":", linewidth=1.0)
        plt.xlabel(r"Current parameter $\theta_c$")
        plt.ylabel(r"External parameter $\theta_e$")
        plt.title("Commensurate joint posterior")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_commensurate_joint_posterior.pdf")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ADNI borrowing baselines on the NPE theta scale."
    )
    parser.add_argument(
        "--subjects-csv",
        default="adni_application_subjects (1).csv",
        help="Subject-level ADNI application CSV.",
    )
    parser.add_argument("--results-dir", default="adni_weighted_baseline_results")
    parser.add_argument("--npe-mean", type=float, default=-1.5990)
    parser.add_argument("--npe-std", type=float, default=0.8585)
    parser.add_argument("--npe-ci-low", type=float, default=-3.3009)
    parser.add_argument("--npe-ci-high", type=float, default=0.1082)
    parser.add_argument("--prior-sd", type=float, default=5.0)
    parser.add_argument("--power-a0", type=float, default=0.5)
    parser.add_argument("--commensurate-sd", type=float, default=1.0)
    parser.add_argument("--grid-min", type=float, default=-5.0)
    parser.add_argument("--grid-max", type=float, default=2.0)
    parser.add_argument("--grid-n", type=int, default=1201)
    parser.add_argument("--posterior-sample-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260414)
    args = parser.parse_args()

    subjects_csv = Path(args.subjects_csv)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    X_c_raw, y_c, X_e_raw, y_e = load_subject_csv(subjects_csv)
    beta_anchor, imputer = fit_beta_anchor(X_c_raw, y_c, X_e_raw, y_e)
    X_c = imputer.transform(X_c_raw)
    X_e = imputer.transform(X_e_raw)
    offset_c = X_c @ beta_anchor
    offset_e = X_e @ beta_anchor

    weighted = make_weighted_comparison(
        y_c,
        y_e,
        X_c,
        X_e,
        offset_c,
        offset_e,
        npe_mean=args.npe_mean,
        npe_std=args.npe_std,
        npe_ci_low=args.npe_ci_low,
        npe_ci_high=args.npe_ci_high,
    )
    bayes, densities, samples = make_bayesian_comparison(
        y_c,
        y_e,
        offset_c,
        offset_e,
        prior_sd=args.prior_sd,
        power_a0=args.power_a0,
        commensurate_sd=args.commensurate_sd,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        grid_n=args.grid_n,
        rng=rng,
        posterior_sample_n=args.posterior_sample_n,
        npe_mean=args.npe_mean,
        npe_std=args.npe_std,
        npe_ci_low=args.npe_ci_low,
        npe_ci_high=args.npe_ci_high,
    )
    risk_bayes, risk_densities, risk_samples = make_risk_logit_bayesian_comparison(
        y_c,
        y_e,
        prior_sd=args.prior_sd,
        power_a0=args.power_a0,
        commensurate_sd=args.commensurate_sd,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        grid_n=args.grid_n,
        rng=rng,
        posterior_sample_n=args.posterior_sample_n,
        npe_mean=args.npe_mean,
        npe_std=args.npe_std,
        npe_ci_low=args.npe_ci_low,
        npe_ci_high=args.npe_ci_high,
    )

    weighted_csv = results_dir / "adni_weighted_borrowing_baselines.csv"
    weighted_tex = results_dir / "adni_weighted_borrowing_baselines.tex"
    bayes_csv = results_dir / "adni_bayesian_borrowing_posteriors.csv"
    bayes_tex = results_dir / "adni_bayesian_borrowing_posteriors.tex"
    density_csv = results_dir / "adni_bayesian_theta_posterior_densities.csv"
    sample_csv = results_dir / "adni_bayesian_posterior_samples.csv"
    risk_bayes_csv = results_dir / "adni_risk_logit_bayesian_posteriors.csv"
    risk_bayes_tex = results_dir / "adni_risk_logit_bayesian_posteriors.tex"
    risk_density_csv = results_dir / "adni_risk_logit_theta_posterior_densities.csv"
    risk_sample_csv = results_dir / "adni_risk_logit_posterior_samples.csv"

    weighted.to_csv(weighted_csv, index=False)
    write_weighted_latex_table(weighted, weighted_tex)
    bayes.to_csv(bayes_csv, index=False)
    write_bayesian_latex_table(bayes, bayes_tex)
    densities.to_csv(density_csv, index=False)
    samples.to_csv(sample_csv, index=False)
    risk_bayes.to_csv(risk_bayes_csv, index=False)
    write_bayesian_latex_table(risk_bayes, risk_bayes_tex)
    risk_densities.to_csv(risk_density_csv, index=False)
    risk_samples.to_csv(risk_sample_csv, index=False)
    maybe_write_figures(
        densities,
        samples,
        results_dir,
        target_theta=empirical_logit_ci(y_c)["theta"],
        prefix="adni_offset_bayesian",
        title="ADNI offset-intercept posterior comparison",
    )
    maybe_write_figures(
        risk_densities,
        risk_samples,
        results_dir,
        target_theta=empirical_logit_ci(y_c)["theta"],
        prefix="adni_risk_logit_bayesian",
        title="ADNI risk-logit posterior comparison",
    )

    print("\nADNI weighted-borrowing baseline comparison")
    print("=" * 78)
    print(weighted[[
        "method",
        "theta",
        "ci_low",
        "ci_high",
        "implied_risk",
        "abs_diff_from_concurrent_logit",
        "external_weight_sum",
    ]].to_string(index=False))

    print("\nADNI Bayesian posterior comparison")
    print("=" * 78)
    print(bayes[[
        "method",
        "parameter",
        "theta_mean",
        "posterior_sd",
        "ci_low",
        "ci_high",
        "implied_risk",
        "abs_diff_from_concurrent_logit",
    ]].to_string(index=False))

    print("\nADNI risk-logit Bayesian posterior comparison")
    print("=" * 78)
    print(risk_bayes[[
        "method",
        "parameter",
        "theta_mean",
        "posterior_sd",
        "ci_low",
        "ci_high",
        "implied_risk",
        "abs_diff_from_concurrent_logit",
    ]].to_string(index=False))

    print("\nSaved outputs:")
    for path in [
        weighted_csv,
        weighted_tex,
        bayes_csv,
        bayes_tex,
        density_csv,
        sample_csv,
        risk_bayes_csv,
        risk_bayes_tex,
        risk_density_csv,
        risk_sample_csv,
    ]:
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
