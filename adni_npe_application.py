from __future__ import annotations

"""
ADNI-specific NPE application built on top of the real ADNI covariates.

Main idea
---------
Use the real ADNI covariates as fixed design points, then simulate binary outcomes
under a simple logistic borrowing model:

    Y_c ~ Bernoulli(sigmoid(X_c beta + theta))
    Y_e ~ Bernoulli(sigmoid(X_e beta + theta + delta))

where
    theta = concurrent target intercept / risk-level parameter of interest
    delta = external outcome shift (nuisance heterogeneity parameter)

This keeps the NPE spirit while avoiding an overclaim that ADNI is a randomized
treatment-effect application.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adni_npe_prep import (
    ADNIPrepConfig,
    load_adni_table,
    prepare_adni_npe_application,
    print_adni_application_summary,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


class PosteriorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout_rate)])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, 1)
        self.log_std_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mean = self.mean_head(h).squeeze(-1)
        log_std = torch.clamp(self.log_std_head(h).squeeze(-1), min=-3.0, max=2.0)
        return mean, log_std


@dataclass
class ADNINPEConfig:
    hidden_dims: Tuple[int, ...] = (128, 128, 64)
    learning_rate: float = 1e-3
    dropout_rate: float = 0.1
    validation_fraction: float = 0.2
    random_seed: Optional[int] = None
    theta_range: Tuple[float, float] = (-3.0, 0.0)
    delta_sd: float = 0.75
    beta_jitter_sd: float = 0.15


class ADNINPEApplication:
    def __init__(self, config: Optional[ADNINPEConfig] = None, device: Optional[str] = None):
        self.config = config or ADNINPEConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.network: Optional[PosteriorMLP] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.training_history = {"train_loss": [], "val_loss": []}
        self.beta_anchor: Optional[np.ndarray] = None

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)

    @staticmethod
    def _gaussian_nll(theta_true: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        std = torch.exp(log_std) + 1e-6
        std_sg = std.detach()
        mean_sg = mean.detach()
        sq_err = (theta_true - mean) ** 2
        sq_err_sg = (theta_true - mean_sg) ** 2
        mean_term = 0.5 * sq_err / (std_sg ** 2 + 1e-6)
        sigma_term = 0.5 * sq_err_sg / (std ** 2 + 1e-6) + log_std
        return torch.mean(mean_term + sigma_term)

    def fit_beta_anchor(self, X_c: np.ndarray, y_c: np.ndarray, X_e: np.ndarray, y_e: np.ndarray) -> np.ndarray:
        X_all = np.vstack([X_c, X_e])
        y_all = np.concatenate([y_c, y_e])
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, fit_intercept=True)),
        ])
        model.fit(X_all, y_all)
        clf = model.named_steps["clf"]
        scaler = model.named_steps["scaler"]
        coef_on_original_scale = clf.coef_.ravel() / scaler.scale_
        self.beta_anchor = coef_on_original_scale.astype(float)
        return self.beta_anchor

    @staticmethod
    def _fit_offset_intercept(
        y: np.ndarray,
        offset: np.ndarray,
        init: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> float:
        theta = float(init)
        for _ in range(max_iter):
            p = sigmoid(offset + theta)
            grad = float(np.sum(y - p))
            hess = float(-np.sum(p * (1.0 - p)) - 1e-8)
            step = grad / hess
            theta_new = theta - step
            if abs(theta_new - theta) < tol:
                return float(theta_new)
            theta = float(theta_new)
        return float(theta)

    def estimate_theta_comparators(
        self,
        X_c: np.ndarray,
        y_c: np.ndarray,
        X_e: np.ndarray,
        y_e: np.ndarray,
    ) -> Dict[str, float]:
        if self.beta_anchor is None:
            self.fit_beta_anchor(X_c, y_c, X_e, y_e)
        assert self.beta_anchor is not None

        offset_c = X_c @ self.beta_anchor
        offset_e = X_e @ self.beta_anchor
        offset_all = np.concatenate([offset_c, offset_e])
        y_all = np.concatenate([y_c, y_e])

        theta_offset_concurrent = self._fit_offset_intercept(y_c, offset_c, init=safe_logit(np.mean(y_c)))
        theta_offset_pooled = self._fit_offset_intercept(y_all, offset_all, init=safe_logit(np.mean(y_all)))
        delta_offset_external = self._fit_offset_intercept(
            y_e,
            offset_e + theta_offset_concurrent,
            init=safe_logit(np.mean(y_e)) - theta_offset_concurrent,
        )

        return {
            "theta_empirical_logit_concurrent": safe_logit(np.mean(y_c)),
            "theta_empirical_logit_external": safe_logit(np.mean(y_e)),
            "theta_empirical_logit_pooled": safe_logit(np.mean(y_all)),
            "theta_offset_mle_concurrent": theta_offset_concurrent,
            "theta_offset_mle_pooled": theta_offset_pooled,
            "delta_offset_mle_external": delta_offset_external,
        }

    def compute_summary_statistics(
        self,
        X_c: np.ndarray,
        y_c: np.ndarray,
        X_e: np.ndarray,
        y_e: np.ndarray,
    ) -> np.ndarray:
        p = X_c.shape[1]
        n_c = len(y_c)
        n_e = len(y_e)

        X_c_mean = X_c.mean(axis=0)
        X_c_std = X_c.std(axis=0) + 1e-6
        X_e_mean = X_e.mean(axis=0)
        X_e_std = X_e.std(axis=0) + 1e-6

        y_c_mean = float(y_c.mean())
        y_e_mean = float(y_e.mean())
        y_c_std = float(y_c.std() + 1e-6)
        y_e_std = float(y_e.std() + 1e-6)

        diff_mean = X_c_mean - X_e_mean
        diff_std = X_c_std - X_e_std
        y_diff = y_c_mean - y_e_mean
        n_ratio = float(n_c / max(n_e, 1))

        summary = np.concatenate([
            X_c_mean, X_c_std, [y_c_mean], [y_c_std], [np.log(n_c + 1.0)],
            X_e_mean, X_e_std, [y_e_mean], [y_e_std], [np.log(n_e + 1.0)],
            diff_mean, diff_std, [y_diff], [n_ratio],
        ])

        if self.feature_names is None:
            names = []
            names += [f"Xc_mean_{j}" for j in range(p)]
            names += [f"Xc_std_{j}" for j in range(p)]
            names += ["yc_mean", "yc_std", "log_nc"]
            names += [f"Xe_mean_{j}" for j in range(p)]
            names += [f"Xe_std_{j}" for j in range(p)]
            names += ["ye_mean", "ye_std", "log_ne"]
            names += [f"Xdiff_mean_{j}" for j in range(p)]
            names += [f"Xdiff_std_{j}" for j in range(p)]
            names += ["ydiff_mean", "n_ratio"]
            self.feature_names = names

        return summary.astype(np.float32)

    def simulate_one_dataset(
        self,
        X_c: np.ndarray,
        X_e: np.ndarray,
        beta_anchor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        theta = float(np.random.uniform(*self.config.theta_range))
        delta = float(np.random.normal(0.0, self.config.delta_sd))
        beta = beta_anchor + np.random.normal(0.0, self.config.beta_jitter_sd, size=beta_anchor.shape[0])

        p_c = sigmoid(X_c @ beta + theta)
        p_e = sigmoid(X_e @ beta + theta + delta)
        y_c = np.random.binomial(1, p_c).astype(float)
        y_e = np.random.binomial(1, p_e).astype(float)
        return y_c, y_e, theta, delta

    def generate_training_data(
        self,
        X_c: np.ndarray,
        X_e: np.ndarray,
        y_c_obs: np.ndarray,
        y_e_obs: np.ndarray,
        n_simulations: int = 2000,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.beta_anchor is None:
            self.fit_beta_anchor(X_c, y_c_obs, X_e, y_e_obs)
        assert self.beta_anchor is not None

        summaries = []
        thetas = []
        for i in range(n_simulations):
            y_c_sim, y_e_sim, theta, delta = self.simulate_one_dataset(X_c, X_e, self.beta_anchor)
            summary = self.compute_summary_statistics(X_c, y_c_sim, X_e, y_e_sim)
            summaries.append(summary)
            thetas.append(theta)
            if verbose and (i + 1) % max(1, n_simulations // 10) == 0:
                print(f"  Generated {i + 1}/{n_simulations} simulations...")

        return np.vstack(summaries).astype(np.float32), np.asarray(thetas, dtype=np.float32)

    def build_network(self, input_dim: int) -> PosteriorMLP:
        self.network = PosteriorMLP(
            input_dim=input_dim,
            hidden_dims=list(self.config.hidden_dims),
            dropout_rate=self.config.dropout_rate,
        ).to(self.device)
        return self.network

    def train(
        self,
        X_c: np.ndarray,
        y_c_obs: np.ndarray,
        X_e: np.ndarray,
        y_e_obs: np.ndarray,
        n_simulations: int = 2000,
        epochs: int = 50,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        if verbose:
            print("=" * 72)
            print("ADNI-SPECIFIC NPE TRAINING")
            print("=" * 72)

        summaries, thetas = self.generate_training_data(
            X_c=X_c,
            X_e=X_e,
            y_c_obs=y_c_obs,
            y_e_obs=y_e_obs,
            n_simulations=n_simulations,
            verbose=verbose,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            summaries,
            thetas,
            test_size=self.config.validation_fraction,
            random_state=self.config.random_seed,
        )

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        self.build_network(X_train.shape[1])
        assert self.network is not None
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        n_train = len(X_train_t)
        self.training_history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.network.train()
            perm = torch.randperm(n_train, device=self.device)
            train_losses = []

            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                xb = X_train_t[idx]
                yb = y_train_t[idx]

                optimizer.zero_grad()
                mean, log_std = self.network(xb)
                loss = self._gaussian_nll(yb, mean, log_std)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))

            self.network.eval()
            with torch.no_grad():
                mean_val, log_std_val = self.network(X_val_t)
                val_loss = float(self._gaussian_nll(y_val_t, mean_val, log_std_val).item())
                mae = float(torch.mean(torch.abs(mean_val - y_val_t)).item())
                rmse = float(torch.sqrt(torch.mean((mean_val - y_val_t) ** 2)).item())

            train_loss = float(np.mean(train_losses))
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)

            if verbose and ((epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0):
                print(
                    f"Epoch {epoch + 1:3d}/{epochs}: "
                    f"Train Loss = {train_loss:.4f}, "
                    f"Val Loss = {val_loss:.4f}, "
                    f"MAE = {mae:.4f}, RMSE = {rmse:.4f}"
                )

        return self.training_history

    def infer(
        self,
        X_c: np.ndarray,
        y_c_obs: np.ndarray,
        X_e: np.ndarray,
        y_e_obs: np.ndarray,
        n_posterior_samples: int = 5000,
    ) -> Dict[str, object]:
        if self.network is None or self.scaler is None:
            raise RuntimeError("Model is not trained.")

        summary = self.compute_summary_statistics(X_c, y_c_obs, X_e, y_e_obs).reshape(1, -1)
        summary_scaled = self.scaler.transform(summary)
        x = torch.tensor(summary_scaled, dtype=torch.float32, device=self.device)

        self.network.eval()
        with torch.no_grad():
            mean, log_std = self.network(x)
            theta_mean = float(mean.item())
            theta_std = float(torch.exp(log_std).item())

        samples = np.random.normal(theta_mean, theta_std, size=n_posterior_samples)
        comparators = self.estimate_theta_comparators(X_c, y_c_obs, X_e, y_e_obs)
        return {
            "posterior_mean": theta_mean,
            "posterior_std": theta_std,
            "ci_low": float(np.percentile(samples, 2.5)),
            "ci_high": float(np.percentile(samples, 97.5)),
            "posterior_samples": samples,
            "summary_vector": summary.ravel(),
            "feature_names": self.feature_names,
            "observed_concurrent_rate": float(np.mean(y_c_obs)),
            "observed_external_rate": float(np.mean(y_e_obs)),
            **comparators,
        }


def load_adni_application_from_files(
    adsl: str,
    dxsum: str,
    ptdemog: str,
    mmse: str,
    adas: str,
    cdr: str,
    faq: str,
    apoe: str,
    allow_impute: bool = True,
) -> Dict[str, object]:
    tables = {
        "ADSL": load_adni_table(adsl),
        "DXSUM": load_adni_table(dxsum),
        "PTDEMOG": load_adni_table(ptdemog),
        "MMSE": load_adni_table(mmse),
        "ADAS": load_adni_table(adas),
        "CDR": load_adni_table(cdr),
        "FAQ": load_adni_table(faq),
        "APOERES": load_adni_table(apoe),
    }
    prep_cfg = ADNIPrepConfig(require_complete_cases=not allow_impute)
    data = prepare_adni_npe_application(
        adsl=tables["ADSL"],
        dxsum=tables["DXSUM"],
        ptdemog=tables["PTDEMOG"],
        mmse=tables["MMSE"],
        adas=tables["ADAS"],
        cdr=tables["CDR"],
        faq=tables["FAQ"],
        apoe=tables["APOERES"],
        config=prep_cfg,
    )
    return data


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run an ADNI-specific NPE application on real ADNI covariates.")
    parser.add_argument("--adsl", required=True)
    parser.add_argument("--dxsum", required=True)
    parser.add_argument("--ptdemog", required=True)
    parser.add_argument("--mmse", required=True)
    parser.add_argument("--adas", required=True)
    parser.add_argument("--cdr", required=True)
    parser.add_argument("--faq", required=True)
    parser.add_argument("--apoe", required=True)
    parser.add_argument("--allow-impute", action="store_true")
    parser.add_argument("--n-simulations", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--results-dir", default="adni_npe_results")
    parser.add_argument("--seed", type=int, default=20260401)
    args = parser.parse_args()

    data = load_adni_application_from_files(
        adsl=args.adsl,
        dxsum=args.dxsum,
        ptdemog=args.ptdemog,
        mmse=args.mmse,
        adas=args.adas,
        cdr=args.cdr,
        faq=args.faq,
        apoe=args.apoe,
        allow_impute=args.allow_impute,
    )
    print_adni_application_summary(data)

    app = ADNINPEApplication(config=ADNINPEConfig(random_seed=args.seed))
    app.train(
        X_c=data["X_concurrent"],
        y_c_obs=data["Y_concurrent"],
        X_e=data["X_external"],
        y_e_obs=data["Y_external"],
        n_simulations=args.n_simulations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
    )
    result = app.infer(
        X_c=data["X_concurrent"],
        y_c_obs=data["Y_concurrent"],
        X_e=data["X_external"],
        y_e_obs=data["Y_external"],
    )

    print("\nADNI-SPECIFIC NPE POSTERIOR")
    print(f"Posterior mean theta: {result['posterior_mean']:.4f}")
    print(f"Posterior std theta:  {result['posterior_std']:.4f}")
    print(f"95% CI: ({result['ci_low']:.4f}, {result['ci_high']:.4f})")
    print(f"Observed concurrent event rate: {result['observed_concurrent_rate']:.4f}")
    print(f"Observed external event rate:   {result['observed_external_rate']:.4f}")
    print("Comparator estimates:")
    print(f"  empirical logit concurrent: {result['theta_empirical_logit_concurrent']:.4f}")
    print(f"  empirical logit external:   {result['theta_empirical_logit_external']:.4f}")
    print(f"  empirical logit pooled:     {result['theta_empirical_logit_pooled']:.4f}")
    print(f"  offset-MLE theta (conc):    {result['theta_offset_mle_concurrent']:.4f}")
    print(f"  offset-MLE theta (pooled):  {result['theta_offset_mle_pooled']:.4f}")
    print(f"  offset-MLE delta (ext):     {result['delta_offset_mle_external']:.4f}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "posterior_mean": [result["posterior_mean"]],
        "posterior_std": [result["posterior_std"]],
        "ci_low": [result["ci_low"]],
        "ci_high": [result["ci_high"]],
        "observed_concurrent_rate": [result["observed_concurrent_rate"]],
        "observed_external_rate": [result["observed_external_rate"]],
        "theta_empirical_logit_concurrent": [result["theta_empirical_logit_concurrent"]],
        "theta_empirical_logit_external": [result["theta_empirical_logit_external"]],
        "theta_empirical_logit_pooled": [result["theta_empirical_logit_pooled"]],
        "theta_offset_mle_concurrent": [result["theta_offset_mle_concurrent"]],
        "theta_offset_mle_pooled": [result["theta_offset_mle_pooled"]],
        "delta_offset_mle_external": [result["delta_offset_mle_external"]],
    }).to_csv(results_dir / "adni_npe_posterior_summary.csv", index=False)
    pd.DataFrame({"feature_name": app.feature_names, "summary_value": result["summary_vector"]}).to_csv(
        results_dir / "adni_npe_summary_vector.csv", index=False
    )
    pd.DataFrame({
        "epoch": np.arange(1, len(app.training_history["train_loss"]) + 1),
        "train_loss": app.training_history["train_loss"],
        "val_loss": app.training_history["val_loss"],
    }).to_csv(results_dir / "adni_npe_training_history.csv", index=False)


if __name__ == "__main__":
    main()
