"""
Bayesian Dynamic Borrowing Baseline Methods
============================================

Baseline methods for borrowing from external (historical) data:
1. PropensityScoreMethod         — IPW/ATT weighting (causallib)
2. PowerPriorMethod              — Full history (a0=1) power prior
3. CommensurateMethod            — Commensurate prior (Hobbs 2011), Rao-Blackwellized MCMC
4. OverlapWeightPSMethod         — Overlap-weight PS (Li et al. 2018)
5. IndividualOverlapPriorMethod  — IOP (Li & Xu 2022)
6. PSPowerMethod                 — PS-stratified power prior (Wang et al. 2019)
7. IndividualizedWeightsMethod   — Mahalanobis IW (Golchi 2021)

OT-augmented NPE methods (NPEMethod, OTNPEMethod) are in methods/npe.py.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from causallib.estimation import IPW


# ============================================================
# PART 1: Propensity Score Method
# ============================================================

class PropensityScoreMethod:
    """
    Propensity score weighting/matching for combining external and concurrent data.
    
    The propensity score is P(S=1|X) where S indicates concurrent (S=1) vs external (S=0).
    Weighting external observations by odds ratio helps balance covariate distributions.
    """
    
    def __init__(self, method: str = 'weighting'):
        """
        Args:
            method: 'weighting' or 'matching'
        """
        self.method = method
        self.ipw_model = None
        self.feature_names = None
        
    def fit_propensity_model(self, X_concurrent: np.ndarray, X_external: np.ndarray):
        """
        Fit propensity score model P(S=1|X).
        
        Args:
            X_concurrent: Covariates from concurrent trial (n_c x p)
            X_external: Covariates from external trial (n_e x p)
        """
        # Combine data: S=1 concurrent, S=0 external
        X = np.vstack([X_concurrent, X_external])
        S = np.concatenate([
            np.ones(len(X_concurrent), dtype=int),
            np.zeros(len(X_external), dtype=int)
        ])

        self.feature_names = [f"x{i+1}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=self.feature_names)
        S_series = pd.Series(S, name="S")

        # Causallib IPW model (with sklearn learner pipeline)
        learner = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0)
        )
        self.ipw_model = IPW(learner=learner, clip_min=0.05, clip_max=0.95, use_stabilized=False)
        self.ipw_model.fit(X_df, S_series)

        # Propensity scores P(S=1|X) from causallib model
        propensity_s1 = self.ipw_model.compute_propensity(X_df, S_series, treatment_values=1)
        e_all = propensity_s1.to_numpy(dtype=float)

        n_c = len(X_concurrent)
        ps_concurrent = e_all[:n_c]
        ps_external = e_all[n_c:]

        # ATT weights: concurrent = 1 (they ARE the target population, no reweighting),
        # external = e/(1-e) (PS odds ratio reweights external to look like concurrent).
        # Using ATE weights (1/e for concurrent, 1/(1-e) for external) would target the
        # pooled population, not the concurrent population — wrong for the borrowing problem.
        w_concurrent = np.ones(n_c)
        w_external = ps_external / (1.0 - ps_external)

        return ps_concurrent, ps_external, w_concurrent, w_external
    
    def fit_weighted_regression(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
        weights_all: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit weighted linear regression combining both datasets.
        
        Model: Y = X*beta + Z*theta + epsilon
        where Z=1 for all (single arm trial)
        
        Returns:
            beta: Coefficient estimates
            cov: Covariance matrix
        """
        # Combine data
        X = np.vstack([X_concurrent, X_external])
        Y = np.concatenate([Y_concurrent, Y_external])
        
        # Add intercept only. In a single-arm setup, theta is represented by intercept.
        n_c = len(X_concurrent)
        n_e = len(X_external)
        
        # Design matrix: [1, X1, ..., Xp]
        design = np.column_stack([np.ones(n_c + n_e), X])
        
        # Weights from causallib IPW for all observations
        weights = weights_all
        W = np.diag(weights)
        
        # Weighted least squares: beta = (X'WX)^{-1} X'WY
        XtWX = design.T @ W @ design
        XtWY = design.T @ W @ Y
        
        # Add small ridge penalty to prevent singularity
        ridge_penalty = 1e-6
        XtWX_reg = XtWX + ridge_penalty * np.eye(XtWX.shape[0])
        
        try:
            beta = np.linalg.solve(XtWX_reg, XtWY)
            
            # Estimate residual variance
            residuals = Y - design @ beta
            sigma2 = np.sum(weights * residuals**2) / np.sum(weights)
            
            # Covariance matrix
            cov = sigma2 * np.linalg.inv(XtWX_reg)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse as fallback
            beta = np.linalg.pinv(XtWX) @ XtWY
            residuals = Y - design @ beta
            sigma2 = np.sum(weights * residuals**2) / np.sum(weights)
            cov = sigma2 * np.linalg.pinv(XtWX)
        
        return beta, cov
    
    def estimate_treatment_effect(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
        n_bootstrap: int = 1000
    ) -> Dict:
        """
        Estimate treatment effect with uncertainty via bootstrap.
        
        Returns:
            Dictionary with 'mean', 'std', 'ci_low', 'ci_high', 'samples'
        """
        # Fit causallib propensity model and obtain package-computed IPW weights
        ps_concurrent, ps_external, w_concurrent, w_external = self.fit_propensity_model(
            X_concurrent, X_external
        )
        weights_all = np.concatenate([w_concurrent, w_external])
        
        # Fit regression
        beta, cov = self.fit_weighted_regression(
            X_concurrent, Y_concurrent,
            X_external, Y_external,
            weights_all
        )
        
        # In single-arm setup, intercept represents theta
        theta_hat = beta[0]
        theta_se = np.sqrt(cov[0, 0])
        
        # Bootstrap for better CI
        theta_boot = []
        n_c = len(X_concurrent)
        n_e = len(X_external)
        
        for _ in range(n_bootstrap):
            # Resample concurrent
            idx_c = np.random.choice(n_c, n_c, replace=True)
            X_c_boot = X_concurrent[idx_c]
            Y_c_boot = Y_concurrent[idx_c]
            
            # Resample external
            idx_e = np.random.choice(n_e, n_e, replace=True)
            X_e_boot = X_external[idx_e]
            Y_e_boot = Y_external[idx_e]
            
            # Refit
            try:
                ps_c_boot, ps_e_boot, w_c_boot, w_e_boot = self.fit_propensity_model(X_c_boot, X_e_boot)
                w_all_boot = np.concatenate([w_c_boot, w_e_boot])
                beta_boot, _ = self.fit_weighted_regression(
                    X_c_boot, Y_c_boot, X_e_boot, Y_e_boot, w_all_boot
                )
                theta_boot.append(beta_boot[0])
            except:
                continue
        
        theta_boot = np.array(theta_boot)
        if theta_boot.size == 0:
            theta_boot = np.array([theta_hat])
        
        return {
            'mean': theta_hat,
            'std': theta_se,
            'ci_low': np.percentile(theta_boot, 2.5),
            'ci_high': np.percentile(theta_boot, 97.5),
            'samples': theta_boot,
            'weights_external': w_external,
            'effective_n_external': np.sum(w_external)**2 / np.sum(w_external**2)
        }


# ============================================================
# PART 2: Power Prior Method
# ============================================================

class PowerPriorMethod:
    """
    Power prior with full borrowing (a0 = 1).
    
    Posterior ∝ L(θ|D_curr) * L(θ|D_ext)^{a0} * π(θ)
    
    Using a0 = 1 (full borrowing): pools concurrent and external data.
    """
    
    def __init__(self, a0: float = 1.0):
        """
        Args:
            a0: Borrowing parameter (default: 1.0 for full borrowing)
        """
        self.a0 = a0
    
    def log_likelihood(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        beta: np.ndarray,
        sigma: float
    ) -> float:
        """
        Log-likelihood for linear model Y = X*beta + epsilon.
        
        Args:
            X: Design matrix (n x p), includes intercept and treatment
            Y: Outcomes (n,)
            beta: Coefficients (p,)
            sigma: Noise std
        """
        n = len(Y)
        mu = X @ beta
        resid = Y - mu
        
        ll = -0.5 * n * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(resid**2) / (sigma**2)
        return ll
    
    def fit_mle(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Fit MLE for linear model.
        
        Returns:
            beta: MLE coefficients
            sigma: MLE noise std
        """
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ beta
        sigma = np.sqrt(np.mean(resid**2))
        return beta, sigma
    
    def compute_dic(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        beta_samples: np.ndarray,
        sigma: float
    ) -> float:
        """
        Compute Deviance Information Criterion.
        
        DIC = D_bar + p_D
        where D_bar = average deviance, p_D = effective number of parameters
        """
        # Average deviance
        D_samples = []
        for beta in beta_samples:
            ll = self.log_likelihood(X, Y, beta, sigma)
            D_samples.append(-2 * ll)
        D_bar = np.mean(D_samples)
        
        # Deviance at posterior mean
        beta_mean = beta_samples.mean(axis=0)
        ll_mean = self.log_likelihood(X, Y, beta_mean, sigma)
        D_hat = -2 * ll_mean
        
        # Effective number of parameters
        p_D = D_bar - D_hat
        
        # DIC
        dic = D_bar + p_D
        
        return dic
    
    def posterior_with_power_prior(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
        a0: float,
        prior_mean: Optional[np.ndarray] = None,
        prior_cov: Optional[np.ndarray] = None,
        n_samples: int = 5000
    ) -> np.ndarray:
        """
        Sample from posterior with power prior.
        
        For conjugate normal-normal model, this has closed form.
        For simplicity, we use Laplace approximation.
        """
        # Design matrices
        n_c = len(Y_concurrent)
        n_e = len(Y_external)
        p = X_concurrent.shape[1]
        
        # Design: [intercept(=theta), X_covariates]
        # In a single-arm trial all Z=1, so theta IS the intercept.
        # Previous version had two identical columns of ones -> rank-deficient (fixed).
        X_c_full = np.column_stack([np.ones(n_c), X_concurrent])  # (n_c, 1+p)
        X_e_full = np.column_stack([np.ones(n_e), X_external])    # (n_e, 1+p)

        n_params = 1 + p  # intercept(theta) + p covariates
        
        # Prior
        if prior_mean is None:
            prior_mean = np.zeros(n_params)
        if prior_cov is None:
            prior_cov = np.eye(n_params) * 25.0  # Vague prior
        
        # Estimate sigma from concurrent data
        beta_c, sigma_c = self.fit_mle(X_c_full, Y_concurrent)
        sigma = sigma_c  # Use as fixed
        
        # Posterior precision and mean (conjugate)
        # Precision: Sigma^{-1} = Sigma_0^{-1} + X_c'X_c/sigma^2 + a0 * X_e'X_e/sigma^2
        prior_prec = np.linalg.inv(prior_cov)
        
        X_c_prec = (X_c_full.T @ X_c_full) / (sigma**2)
        X_e_prec = (X_e_full.T @ X_e_full) / (sigma**2)
        
        post_prec = prior_prec + X_c_prec + a0 * X_e_prec
        post_cov = np.linalg.inv(post_prec)
        
        # Posterior mean
        post_mean = post_cov @ (
            prior_prec @ prior_mean +
            (X_c_full.T @ Y_concurrent) / (sigma**2) +
            a0 * (X_e_full.T @ Y_external) / (sigma**2)
        )
        
        # Sample from posterior
        samples = np.random.multivariate_normal(post_mean, post_cov, size=n_samples)
        
        return samples
    
    def select_a0(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
        criterion: str = 'dic'
    ) -> Tuple[float, Dict]:
        """
        Deprecated: Using fixed a0 for full borrowing.
        Returns the configured a0 value.
        """
        return self.a0, {}
    
    def estimate_treatment_effect(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray
    ) -> Dict:
        """
        Estimate treatment effect with full borrowing (a0=1).
        """
        # Get posterior samples with fixed a0
        samples = self.posterior_with_power_prior(
            X_concurrent, Y_concurrent,
            X_external, Y_external,
            self.a0, 
            n_samples=5000
        )
        
        # Treatment effect is the intercept (index 0) in design [1, X_covariates]
        theta_samples = samples[:, 0]
        
        return {
            'mean': theta_samples.mean(),
            'std': theta_samples.std(),
            'ci_low': np.percentile(theta_samples, 2.5),
            'ci_high': np.percentile(theta_samples, 97.5),
            'samples': theta_samples,
            'a0': self.a0
        }


# ============================================================
# PART 2b: Normalized Power Prior (Joint theta, a0 via MCMC)
# ============================================================

class NormalizedPowerPriorMethod:
    """
    Normalized Power Prior (NPP) with joint MCMC over (theta, a0).

    Target:
      p(eta, a0 | D, D0) ∝ L(D | eta) * L(D0 | eta)^a0 * pi0(eta) * pi(a0) / c(a0)

    where eta = [theta, beta_1, ..., beta_p] and
      c(a0) = ∫ L(D0 | eta)^a0 pi0(eta) d eta.

    This implementation uses a Gaussian linear model with known sigma (plug-in estimate)
    and computes log c(a0) in closed form, while sampling (eta, a0) by MH.
    """

    def __init__(
        self,
        prior_var: float = 25.0,
        a0_alpha: float = 1.0,
        a0_beta: float = 1.0,
        step_eta: float = 0.08,
        step_logit_a0: float = 0.20,
    ):
        self.prior_var = float(prior_var)
        self.a0_alpha = float(a0_alpha)
        self.a0_beta = float(a0_beta)
        self.step_eta = float(step_eta)
        self.step_logit_a0 = float(step_logit_a0)

    @staticmethod
    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    def _build_design(self, X: np.ndarray) -> np.ndarray:
        # eta = [theta(intercept), beta_covariates]
        return np.column_stack([np.ones(X.shape[0]), X])

    def _fit_sigma(self, Xc: np.ndarray, yc: np.ndarray) -> float:
        eta_hat = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        resid = yc - Xc @ eta_hat
        sigma = np.sqrt(np.mean(resid ** 2))
        return float(max(sigma, 1e-6))

    def _log_likelihood(self, X: np.ndarray, y: np.ndarray, eta: np.ndarray, sigma: float) -> float:
        n = y.shape[0]
        resid = y - X @ eta
        return -0.5 * n * np.log(2.0 * np.pi * sigma ** 2) - 0.5 * np.sum(resid ** 2) / (sigma ** 2)

    def _log_prior_eta(self, eta: np.ndarray) -> float:
        d = eta.shape[0]
        return -0.5 * d * np.log(2.0 * np.pi * self.prior_var) - 0.5 * np.sum(eta ** 2) / self.prior_var

    def _log_prior_a0(self, a0: float) -> float:
        if a0 <= 0.0 or a0 >= 1.0:
            return -np.inf
        # Beta(a0_alpha, a0_beta) up to normalizing const via scipy
        return stats.beta.logpdf(a0, self.a0_alpha, self.a0_beta)

    def _log_c_a0(
        self,
        a0: float,
        Xh: np.ndarray,
        yh: np.ndarray,
        sigma: float,
    ) -> float:
        """
        Closed-form log normalizing constant:
          c(a0) = ∫ L_h(eta)^a0 pi0(eta) d eta
        with pi0(eta)=N(0, prior_var * I).
        """
        if a0 < 0.0 or a0 > 1.0:
            return -np.inf

        n_h, d = Xh.shape
        sigma2 = sigma ** 2

        V0 = self.prior_var * np.eye(d)
        V0_inv = np.eye(d) / self.prior_var

        XtX = Xh.T @ Xh
        Xty = Xh.T @ yh
        yty = float(yh.T @ yh)

        A = (a0 / sigma2) * XtX + V0_inv
        b = (a0 / sigma2) * Xty

        sign_A, logdet_A = np.linalg.slogdet(A)
        if sign_A <= 0:
            return -np.inf

        sign_V0, logdet_V0 = np.linalg.slogdet(V0)
        if sign_V0 <= 0:
            return -np.inf

        A_inv_b = np.linalg.solve(A, b)
        const_term = (a0 / sigma2) * yty - float(b.T @ A_inv_b)

        log_c = (
            -0.5 * a0 * n_h * np.log(2.0 * np.pi * sigma2)
            -0.5 * logdet_V0
            -0.5 * logdet_A
            -0.5 * const_term
        )
        return float(log_c)

    def _log_joint(
        self,
        eta: np.ndarray,
        a0: float,
        Xc: np.ndarray,
        yc: np.ndarray,
        Xh: np.ndarray,
        yh: np.ndarray,
        sigma: float,
    ) -> float:
        if a0 <= 0.0 or a0 >= 1.0:
            return -np.inf

        ll_c = self._log_likelihood(Xc, yc, eta, sigma)
        ll_h = self._log_likelihood(Xh, yh, eta, sigma)
        lp_eta = self._log_prior_eta(eta)
        lp_a0 = self._log_prior_a0(a0)
        log_c = self._log_c_a0(a0, Xh, yh, sigma)

        return ll_c + a0 * ll_h + lp_eta + lp_a0 - log_c

    def mcmc_sample(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
        n_samples: int = 5000,
        n_burnin: int = 1000,
    ) -> Dict:
        Xc = self._build_design(X_concurrent)
        Xh = self._build_design(X_external)
        yc = Y_concurrent.astype(float)
        yh = Y_external.astype(float)

        sigma = self._fit_sigma(Xc, yc)
        d = Xc.shape[1]

        # Initialize eta at concurrent OLS, a0 at 0.5
        eta = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        logit_a0 = 0.0

        samples_theta = []
        samples_a0 = []
        samples_eta = []

        logp_curr = self._log_joint(
            eta, self._sigmoid(logit_a0), Xc, yc, Xh, yh, sigma
        ) + np.log(self._sigmoid(logit_a0) * (1.0 - self._sigmoid(logit_a0)))

        for i in range(n_samples + n_burnin):
            # 1) Propose eta block
            eta_prop = eta + np.random.randn(d) * self.step_eta
            a0_curr = self._sigmoid(logit_a0)
            logp_prop = self._log_joint(
                eta_prop, a0_curr, Xc, yc, Xh, yh, sigma
            ) + np.log(a0_curr * (1.0 - a0_curr))
            if np.log(np.random.rand()) < (logp_prop - logp_curr):
                eta = eta_prop
                logp_curr = logp_prop

            # 2) Propose a0 in logit space
            logit_a0_prop = logit_a0 + np.random.randn() * self.step_logit_a0
            a0_prop = self._sigmoid(logit_a0_prop)
            logp_prop = self._log_joint(
                eta, a0_prop, Xc, yc, Xh, yh, sigma
            ) + np.log(a0_prop * (1.0 - a0_prop))
            if np.log(np.random.rand()) < (logp_prop - logp_curr):
                logit_a0 = logit_a0_prop
                logp_curr = logp_prop

            if i >= n_burnin:
                a0_keep = self._sigmoid(logit_a0)
                samples_theta.append(float(eta[0]))
                samples_a0.append(float(a0_keep))
                samples_eta.append(eta.copy())

        return {
            'theta': np.array(samples_theta),
            'a0': np.array(samples_a0),
            'eta': np.array(samples_eta),
            'sigma': sigma,
        }

    def estimate_treatment_effect(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
        n_samples: int = 5000,
        n_burnin: int = 1000,
    ) -> Dict:
        samples = self.mcmc_sample(
            X_concurrent,
            Y_concurrent,
            X_external,
            Y_external,
            n_samples=n_samples,
            n_burnin=n_burnin,
        )

        theta_samples = samples['theta']
        a0_samples = samples['a0']

        return {
            'mean': theta_samples.mean(),
            'std': theta_samples.std(),
            'ci_low': np.percentile(theta_samples, 2.5),
            'ci_high': np.percentile(theta_samples, 97.5),
            'samples': theta_samples,
            'a0_mean': a0_samples.mean(),
            'a0_ci_low': np.percentile(a0_samples, 2.5),
            'a0_ci_high': np.percentile(a0_samples, 97.5),
            'a0_samples': a0_samples,
            # keep compatibility with existing summary formatter
            'a0': a0_samples.mean(),
        }


# ============================================================
# PART 3: Commensurate Prior Method
# ============================================================

class CommensurateMethod:
    """Commensurate prior (Hobbs et al. 2011) — Rao-Blackwellized MCMC.

    Model (theta_e marginalized out via Hobbs 2011, Sec. 4):
        Y_c | eta, sigma       ~ N(Xd_c @ eta, sigma^2 I)
        Y_e | eta, sigma, tau  ~ N(Xd_e @ eta, (sigma^2 + tau^2) I)
        eta = [theta_c, beta_1, ..., beta_p]
        eta ~ N(0, prior_var * I)
        tau ~ HalfNormal(tau_scale)

    The key insight: marginalizing theta_e ~ N(theta_c, tau^2) over the external
    likelihood inflates external variance to sigma^2 + tau^2.
      tau -> 0 : full borrowing (external precision -> 1/sigma^2, like FH a0=1)
      tau -> inf: no borrowing  (external precision -> 0)

    Algorithm (Rao-Blackwellized — avoids double-counting):
      1. Estimate sigma from concurrent data OLS (plug-in, used only once).
      2. Run 1D MH-MCMC over log(tau) sampling the MARGINAL posterior of tau.
      3. For each tau sample, draw eta from the EXACT conjugate Gaussian posterior.
         This avoids the initialization artifact of the previous implementation
         and provides properly calibrated credible intervals.

    Prior choices:
        eta   ~ N(0, 25 * I)  — vague prior
        tau   ~ HalfNormal(1) — places prior mass on moderate borrowing
    """

    def __init__(
        self,
        tau_scale: float = 1.0,
        prior_var: float = 25.0,
        step_log_tau: float = 0.5,
    ):
        self.tau_scale = tau_scale
        self.prior_var = prior_var
        self.step_log_tau = step_log_tau

    def _log_post_log_tau(
        self,
        log_tau: float,
        Xd_c: np.ndarray,
        Y_c: np.ndarray,
        Xd_e: np.ndarray,
        Y_e: np.ndarray,
        sigma: float,
    ) -> float:
        """Log p(tau | data) up to a constant (marginal likelihood + prior + Jacobian).

        Derivation (conjugate Gaussian marginal):
          log p(Y_c, Y_e | tau) = const
              - 0.5 * n_e * log(sigma^2 + tau^2)
              - 0.5 * Y_e'Y_e / (sigma^2 + tau^2)
              + 0.5 * rhs' Lam_post^{-1} rhs
              - 0.5 * log|Lam_post|
        where Lam_post = I/prior_var + Xd_c'Xd_c/sigma^2 + Xd_e'Xd_e/(sigma^2+tau^2).
        """
        tau = np.exp(log_tau)
        sigma2 = sigma ** 2
        sigma2_e = sigma2 + tau ** 2
        d = Xd_c.shape[1]

        Lam_post = (
            np.eye(d) / self.prior_var
            + Xd_c.T @ Xd_c / sigma2
            + Xd_e.T @ Xd_e / sigma2_e
        )
        rhs = Xd_c.T @ Y_c / sigma2 + Xd_e.T @ Y_e / sigma2_e

        sign, logdet_Lam = np.linalg.slogdet(Lam_post)
        if sign <= 0:
            return -np.inf
        try:
            mu_post = np.linalg.solve(Lam_post, rhs)
        except np.linalg.LinAlgError:
            return -np.inf

        n_e = len(Y_e)
        log_mlik = (
            -0.5 * n_e * np.log(sigma2_e)          # normalization (tau-dependent)
            - 0.5 * np.dot(Y_e, Y_e) / sigma2_e    # external data quadratic
            + 0.5 * np.dot(rhs, mu_post)            # posterior quadratic
            - 0.5 * logdet_Lam                      # posterior log-det
        )
        log_prior_tau = stats.halfnorm.logpdf(tau, scale=self.tau_scale)
        # Jacobian for sampling in log space: d(log tau) -> multiply by tau
        return log_mlik + log_prior_tau + log_tau

    def _posterior_given_tau(
        self,
        tau: float,
        Xd_c: np.ndarray,
        Y_c: np.ndarray,
        Xd_e: np.ndarray,
        Y_e: np.ndarray,
        sigma: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Exact conjugate Gaussian posterior mean and covariance for eta given tau."""
        sigma2 = sigma ** 2
        sigma2_e = sigma ** 2 + tau ** 2
        d = Xd_c.shape[1]

        Lam_post = (
            np.eye(d) / self.prior_var
            + Xd_c.T @ Xd_c / sigma2
            + Xd_e.T @ Xd_e / sigma2_e
        )
        rhs = Xd_c.T @ Y_c / sigma2 + Xd_e.T @ Y_e / sigma2_e
        Sigma_post = np.linalg.inv(Lam_post)
        mu_post = Sigma_post @ rhs
        return mu_post, Sigma_post

    def _log_posterior_stub(
        self,
        theta_c: float,
        theta_e: float,
        tau: float,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
        beta: np.ndarray,
        sigma: float
    ) -> float:
        """Legacy stub — kept for API compatibility. Not used in sampling.

        Args:
            theta_c: Current treatment effect
            tau: Commensurability parameter (ignored — legacy stub)
            beta: Covariate coefficients (ignored — legacy stub)
            sigma: Noise std (ignored — legacy stub)
        """
        return 0.0

    def mcmc_sample(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
        n_samples: int = 2000,
        n_burnin: int = 500,
    ) -> Dict:
        """Rao-Blackwellized MCMC: sample tau (1D MH), draw eta analytically.

        Returns:
            Dictionary with theta_c samples, tau samples, sigma, acceptance_rate
        """
        n_c, p = X_concurrent.shape
        n_e = len(Y_external)
        d = 1 + p

        Xd_c = np.column_stack([np.ones(n_c), X_concurrent])
        Xd_e = np.column_stack([np.ones(n_e), X_external])
        Y_c = Y_concurrent.astype(float)
        Y_e = Y_external.astype(float)

        # sigma estimated from concurrent data only (no double-counting: eta not plug-in)
        sigma = _estimate_sigma_from_concurrent(X_concurrent, Y_concurrent)
        sigma2 = sigma ** 2

        # Initialize tau at prior mode (tau_scale)
        log_tau = np.log(self.tau_scale)
        log_p_curr = self._log_post_log_tau(log_tau, Xd_c, Y_c, Xd_e, Y_e, sigma)

        theta_samples: list = []
        tau_chain: list = []
        n_accept = 0

        for i in range(n_samples + n_burnin):
            # 1D MH proposal for log(tau)
            log_tau_prop = log_tau + np.random.randn() * self.step_log_tau
            log_p_prop = self._log_post_log_tau(log_tau_prop, Xd_c, Y_c, Xd_e, Y_e, sigma)
            if np.log(np.random.rand() + 1e-300) < log_p_prop - log_p_curr:
                log_tau = log_tau_prop
                log_p_curr = log_p_prop
                if i >= n_burnin:
                    n_accept += 1

            if i >= n_burnin:
                tau_curr = float(np.exp(log_tau))
                tau_chain.append(tau_curr)
                # Draw eta from exact conjugate posterior given tau (Rao-Blackwellization)
                try:
                    mu_post, Sigma_post = self._posterior_given_tau(
                        tau_curr, Xd_c, Y_c, Xd_e, Y_e, sigma
                    )
                    eta_samp = np.random.multivariate_normal(mu_post, Sigma_post)
                    theta_samples.append(float(eta_samp[0]))
                except np.linalg.LinAlgError:
                    # Fallback: posterior mean only (no sampling noise)
                    Lam = (np.eye(d) / self.prior_var
                           + Xd_c.T @ Xd_c / sigma2
                           + Xd_e.T @ Xd_e / (sigma2 + tau_curr ** 2))
                    rhs = Xd_c.T @ Y_c / sigma2 + Xd_e.T @ Y_e / (sigma2 + tau_curr ** 2)
                    theta_samples.append(float(np.linalg.lstsq(Lam, rhs, rcond=None)[0][0]))

        return {
            'theta_c': np.array(theta_samples),
            'tau': np.array(tau_chain),
            'sigma': sigma,
            'acceptance_rate': n_accept / max(n_samples, 1),
        }

    def estimate_treatment_effect(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
        n_samples: int = 2000,
        n_burnin: int = 500,
    ) -> Dict:
        """Estimate treatment effect using Rao-Blackwellized commensurate prior MCMC."""
        samples = self.mcmc_sample(
            X_concurrent, Y_concurrent,
            X_external, Y_external,
            n_samples=n_samples, n_burnin=n_burnin,
        )
        valid = samples['theta_c'][np.isfinite(samples['theta_c'])]
        if len(valid) == 0:
            valid = np.array([0.0])

        return {
            'mean': float(np.mean(valid)),
            'std': float(np.std(valid)),
            'ci_low': float(np.percentile(valid, 2.5)),
            'ci_high': float(np.percentile(valid, 97.5)),
            'samples': valid,
            'tau_mean': float(np.mean(samples['tau'])),
            'tau_samples': samples['tau'],
            'acceptance_rate': samples['acceptance_rate'],
        }


# ============================================================
# PART 4: Shared Conjugate Gaussian Helper
# ============================================================

def _gaussian_borrowing_posterior(
    X_c: np.ndarray,
    Y_c: np.ndarray,
    X_e: np.ndarray,
    Y_e: np.ndarray,
    weights_e: np.ndarray,
    sigma: float,
    prior_var: float = 25.0,
    n_samples: int = 5000,
) -> Dict:
    """Conjugate Gaussian posterior for the single-arm dynamic borrowing model.

    Model: Y = [1, X] eta + eps,  eps ~ N(0, sigma^2 I)
    where eta = [theta, beta_1, ..., beta_p], theta = intercept = treatment effect.

    External data is incorporated with individual weights:
      each obs i contributes L(y_e,i | eta)^{w_i}, equivalent to variance sigma^2 / w_i.

    Posterior:
      Lam_post = (1/prior_var)*I + Xd_c'Xd_c/sigma^2 + Xd_e' diag(w) Xd_e/sigma^2
      mu_post  = Lam_post^{-1} (Xd_c'Yc/sigma^2 + Xd_e'(w*Ye)/sigma^2)
      theta ~ N(mu_post[0], (Lam_post^{-1})[0,0])

    Args:
        X_c: (n_c, p) concurrent covariates
        Y_c: (n_c,) concurrent outcomes
        X_e: (n_e, p) external covariates
        Y_e: (n_e,) external outcomes
        weights_e: (n_e,) individual borrowing weights for external obs
          - weights_e = ones(n_e):      power prior alpha=1 (full history)
          - weights_e = f(e_i):         Individual Overlap Prior (Li & Xu 2022)
          - weights_e = alpha_k:        PS-stratified power prior (Wang et al. 2019)
          - weights_e = exp(-d_i/h):    Mahalanobis individualized weights (Golchi 2021)
        sigma: noise std (estimated from concurrent data)
        prior_var: prior variance for eta ~ N(0, prior_var * I)

    Returns:
        dict with mean, std, ci_low, ci_high, samples (for theta)
    """
    n_c, p = X_c.shape
    n_e = len(Y_e)
    d = 1 + p

    Xd_c = np.column_stack([np.ones(n_c), X_c])   # (n_c, d)
    Xd_e = np.column_stack([np.ones(n_e), X_e])   # (n_e, d)
    sigma2 = sigma ** 2

    Lam_prior = np.eye(d) / prior_var
    Lam_c = Xd_c.T @ Xd_c / sigma2
    Lam_e = Xd_e.T @ (weights_e[:, None] * Xd_e) / sigma2   # weighted by w_i

    Lam_post = Lam_prior + Lam_c + Lam_e
    rhs = Xd_c.T @ Y_c / sigma2 + Xd_e.T @ (weights_e * Y_e) / sigma2

    try:
        mu_post = np.linalg.solve(Lam_post, rhs)
        Sigma_post = np.linalg.inv(Lam_post)
    except np.linalg.LinAlgError:
        mu_post = np.linalg.lstsq(Lam_post, rhs, rcond=None)[0]
        Sigma_post = np.linalg.pinv(Lam_post)

    theta_mean = float(mu_post[0])
    theta_var = float(np.abs(Sigma_post[0, 0]))
    theta_std = float(np.sqrt(theta_var))

    theta_samples = np.random.normal(theta_mean, theta_std, size=n_samples)

    return {
        'mean': theta_mean,
        'std': theta_std,
        'ci_low': theta_mean - 1.96 * theta_std,
        'ci_high': theta_mean + 1.96 * theta_std,
        'samples': theta_samples,
    }


def _fit_ps_logistic(
    X_c: np.ndarray,
    X_e: np.ndarray,
    clip: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit logistic PS model P(S=1|X). Returns (e_c, e_e), clipped to [clip, 1-clip]."""
    X_all = np.vstack([X_c, X_e])
    S_all = np.concatenate([np.ones(len(X_c)), np.zeros(len(X_e))])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X_all, S_all)
    e_all = np.clip(clf.predict_proba(X_all)[:, 1], clip, 1.0 - clip)
    return e_all[:len(X_c)], e_all[len(X_c):]


def _estimate_sigma_from_concurrent(X_c: np.ndarray, Y_c: np.ndarray) -> float:
    """OLS estimate of sigma from concurrent data using design [1, X_c]."""
    Xd = np.column_stack([np.ones(len(Y_c)), X_c])
    eta = np.linalg.lstsq(Xd, Y_c, rcond=None)[0]
    resid = Y_c - Xd @ eta
    return float(max(np.sqrt(np.mean(resid ** 2)), 1e-6))


# ============================================================
# PART 6: Overlap-Weight PS Method
# ============================================================

class OverlapWeightPSMethod:
    """Propensity score method using overlap weights (Li, Morgan, Zaslavsky 2018).

    Overlap weight for external obs i: w_i = e_i * (1 - e_i)
    where e_i = P(S=1 | X_i), S=1 concurrent, S=0 external.

    h(x) = e(x)(1 - e(x)) is maximized at e=0.5 (perfect overlap) and approaches
    zero for observations far from the common support region.

    Estimator: WLS on pooled data [Y_c, Y_e] ~ [1, X] with weights
      [ones(n_c), w_e]. Concurrent observations get unit weight (targeting
      the concurrent population). External observations are down-weighted
      proportional to their overlap with the concurrent covariate distribution.
    """

    def __init__(self, clip: float = 0.01):
        self.clip = clip

    def estimate_treatment_effect(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
    ) -> Dict:
        n_c = len(Y_concurrent)
        n_e = len(Y_external)

        _, e_e = _fit_ps_logistic(X_concurrent, X_external, self.clip)
        w_e = e_e * (1.0 - e_e)   # overlap weights, in (0, 0.25]

        # WLS: pool datasets
        Xd = np.column_stack([np.ones(n_c + n_e),
                               np.vstack([X_concurrent, X_external])])
        Y = np.concatenate([Y_concurrent, Y_external])
        w = np.concatenate([np.ones(n_c), w_e])

        W_diag = w
        XtW = Xd.T * W_diag           # (d, n)
        XtWX = XtW @ Xd + 1e-8 * np.eye(Xd.shape[1])
        XtWY = XtW @ Y
        try:
            beta = np.linalg.solve(XtWX, XtWY)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtWX, XtWY, rcond=None)[0]

        # Sandwich (Huber-White) SE
        resid = Y - Xd @ beta
        XtWX_inv = np.linalg.inv(XtWX)
        # meat = X'W diag(r^2) WX  -- here W is diagonal so W^2 = diag(w^2)
        meat = (Xd * (w ** 2 * resid ** 2)[:, None]).T @ Xd
        var_beta = XtWX_inv @ meat @ XtWX_inv
        theta_mean = float(beta[0])
        theta_std = float(np.sqrt(np.abs(var_beta[0, 0])))

        n_samples = 5000
        theta_samples = np.random.normal(theta_mean, theta_std, size=n_samples)

        return {
            'mean': theta_mean,
            'std': theta_std,
            'ci_low': theta_mean - 1.96 * theta_std,
            'ci_high': theta_mean + 1.96 * theta_std,
            'samples': theta_samples,
            'weights_external': w_e,
        }


# ============================================================
# PART 7: Individual Overlap Prior (Li & Xu 2022)
# ============================================================

class IndividualOverlapPriorMethod:
    """Individual Overlap Prior (IOP) — Li & Xu (2022).

    Prior:
      p0(theta | D_ext) proportional to p0(theta) * prod_i L(y_e,i | theta)^{f(e_i)}
    where f(e_i) = 2 * sqrt(e_i * (1 - e_i))  and  e_i = P(S=1 | X_i).

    f(e) is motivated by the overlap weighting literature (Li, Morgan, Zaslavsky 2018):
      - proportional to harmonic mean of e_i and 1-e_i
      - maximized at e=0.5 (perfect overlap), zero at e=0 or e=1 (no overlap)
      - leads to optimal covariate overlap between external and concurrent samples

    For a Gaussian linear model this is analytically conjugate; no MCMC required.
    Individual weights weights_e = f(e_i) are passed to _gaussian_borrowing_posterior.

    DESIGN LIMITATION: IOP protects against X-shift only.
    When X is exchangeable (mu_e = mu_c) but Y is not (delta_e != 0, e.g. SC5),
    the PS model assigns e_i ≈ n_c/(n_c+n_e) to everyone (base rate), giving
    f(e) ≈ 0.8 → IOP borrows ~80% of external data and inherits the Y-bias.
    This is expected per Li & Xu (2022): IOP is designed to handle covariate
    distribution shift, not outcome heterogeneity.
    """

    def __init__(self, prior_var: float = 25.0, clip: float = 0.01):
        """
        Args:
            prior_var: Prior variance for eta ~ N(0, prior_var * I)
            clip: PS clipping for numerical stability
        """
        self.prior_var = prior_var
        self.clip = clip

    def estimate_treatment_effect(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
    ) -> Dict:
        _, e_e = _fit_ps_logistic(X_concurrent, X_external, self.clip)
        # IOP weight: f(e) = 2 * sqrt(e * (1 - e))  (Eq. 1 in Li & Xu 2022)
        f_e = 2.0 * np.sqrt(e_e * (1.0 - e_e))

        sigma = _estimate_sigma_from_concurrent(X_concurrent, Y_concurrent)
        result = _gaussian_borrowing_posterior(
            X_concurrent, Y_concurrent, X_external, Y_external,
            f_e, sigma, self.prior_var,
        )
        result['weights_external'] = f_e
        return result


# ============================================================
# PART 8: PS-Stratified Power Prior (Wang et al. 2019)
# ============================================================

class PSPowerMethod:
    """Stratified PS + Power Prior (Wang et al. 2019).

    Stratify all observations into K=5 PS strata based on concurrent PS quantiles.
    Within stratum k, the borrowing weight is:
      alpha_k = min(1,  (n_c_k / n_c) / (n_e_k / n_e + 1e-9))

    This gives alpha_k near 1 when the stratum is well-represented in the concurrent
    data (similar representation), and near 0 when the stratum has many more external
    than concurrent observations (poor overlap in that stratum).

    Individual weights weights_e[i] = alpha_{k(i)} are passed to
    _gaussian_borrowing_posterior for a closed-form conjugate update.
    """

    def __init__(self, K: int = 5, prior_var: float = 25.0, clip: float = 0.01):
        """
        Args:
            K: Number of PS strata (quintiles by default)
            prior_var: Prior variance for eta ~ N(0, prior_var * I)
            clip: PS clipping for numerical stability
        """
        self.K = K
        self.prior_var = prior_var
        self.clip = clip

    def estimate_treatment_effect(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
    ) -> Dict:
        n_c = len(Y_concurrent)
        n_e = len(Y_external)

        e_c, e_e = _fit_ps_logistic(X_concurrent, X_external, self.clip)

        # Strata boundaries from concurrent PS K-quantiles
        quantiles = np.quantile(e_c, np.linspace(0.0, 1.0, self.K + 1))
        quantiles[0] -= 1e-9
        quantiles[-1] += 1e-9

        strata_c = np.digitize(e_c, quantiles) - 1
        strata_c = np.clip(strata_c, 0, self.K - 1)
        strata_e = np.digitize(e_e, quantiles) - 1
        strata_e = np.clip(strata_e, 0, self.K - 1)

        weights_e = np.zeros(n_e)
        for k in range(self.K):
            n_c_k = int(np.sum(strata_c == k))
            n_e_k = int(np.sum(strata_e == k))
            # alpha_k: proportion of concurrent in stratum / proportion of external
            alpha_k = min(1.0, (n_c_k / n_c) / (n_e_k / n_e + 1e-9))
            weights_e[strata_e == k] = alpha_k

        sigma = _estimate_sigma_from_concurrent(X_concurrent, Y_concurrent)
        result = _gaussian_borrowing_posterior(
            X_concurrent, Y_concurrent, X_external, Y_external,
            weights_e, sigma, self.prior_var,
        )
        result['alpha_strata'] = [
            min(1.0, (np.sum(strata_c == k) / n_c) / (np.sum(strata_e == k) / n_e + 1e-9))
            for k in range(self.K)
        ]
        return result


# ============================================================
# PART 9: Individualized Weights Method (Golchi 2021)
# ============================================================

class IndividualizedWeightsMethod:
    """Individualized Weights (IW) prior — Golchi (2021).

    Each external observation gets borrowing weight:
      alpha_i = exp(-d_i / h)
    where d_i = Mahalanobis distance from x_{e,i} to the concurrent sample
    and h = median(d_i) is the bandwidth (data-adaptive).

    Observations with d_i >> h are far from the concurrent covariate distribution
    and receive negligible weight. Weights are clipped to [clip_min, 1].

    Individual weights are passed to _gaussian_borrowing_posterior for
    a closed-form conjugate update.
    """

    def __init__(self, prior_var: float = 25.0, clip_min: float = 0.01):
        """
        Args:
            prior_var: Prior variance for eta ~ N(0, prior_var * I)
            clip_min: Minimum weight (prevents zero contribution)
        """
        self.prior_var = prior_var
        self.clip_min = clip_min

    def estimate_treatment_effect(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
    ) -> Dict:
        n_c = len(Y_concurrent)
        p = X_concurrent.shape[1]

        mu_c = X_concurrent.mean(axis=0)
        # Regularized sample covariance to handle near-singular cases (p=6 is fine)
        S_c = np.cov(X_concurrent.T) + 1e-6 * np.eye(p)
        try:
            S_inv = np.linalg.inv(S_c)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S_c)

        diff = X_external - mu_c                        # (n_e, p)
        d2 = np.einsum('ij,jk,ik->i', diff, S_inv, diff)  # (n_e,) Mahalanobis^2
        d = np.sqrt(np.maximum(d2, 0.0))

        h = float(np.median(d)) + 1e-9                 # bandwidth = median distance
        weights_e = np.clip(np.exp(-d / h), self.clip_min, 1.0)

        sigma = _estimate_sigma_from_concurrent(X_concurrent, Y_concurrent)
        result = _gaussian_borrowing_posterior(
            X_concurrent, Y_concurrent, X_external, Y_external,
            weights_e, sigma, self.prior_var,
        )
        result['weights_external'] = weights_e
        result['mahal_distances'] = d
        return result


# ============================================================
# Utility Functions
# ============================================================

def compare_methods_summary(results: Dict[str, Dict]) -> str:
    """
    Create a summary table comparing all methods.
    
    Args:
        results: Dictionary mapping method name to result dict
        
    Returns:
        Formatted summary string
    """
    summary = []
    summary.append("\n" + "=" * 80)
    summary.append("COMPARISON OF METHODS")
    summary.append("=" * 80)
    summary.append(f"{'Method':<20} {'Mean':>10} {'Std':>10} {'95% CI':>25} {'Extra Info':>15}")
    summary.append("-" * 80)
    
    for method_name, result in results.items():
        mean = result['mean']
        std = result['std']
        ci = f"[{result['ci_low']:.3f}, {result['ci_high']:.3f}]"
        
        # Method-specific extra info
        extra = ""
        if 'a0' in result:
            extra = f"a0={result['a0']:.3f}"
        elif 'a0_mean' in result:
            extra = f"a0={result['a0_mean']:.3f}"
        elif 'tau_mean' in result:
            extra = f"τ={result['tau_mean']:.3f}"
        elif 'effective_n_external' in result:
            extra = f"n_eff={result['effective_n_external']:.1f}"
        
        summary.append(f"{method_name:<20} {mean:>10.3f} {std:>10.3f} {ci:>25} {extra:>15}")
    
    summary.append("=" * 80)
    
    return "\n".join(summary)
