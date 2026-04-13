"""
Neural Posterior Estimation (NPE) for Bayesian Dynamic Borrowing
================================================================

Two methods:

1. NPEMethod — Base amortized posterior estimator.
   Train once offline on diverse simulated scenarios spanning all shift types.
   At inference: single forward pass → posterior (μ, σ) for θ.

2. OTNPEMethod — Covariate-Wasserstein-Augmented NPE.
   Extends NPEMethod by appending 4 Sinkhorn OT features to the summary
   statistics vector.  All distances are computed on X ONLY (pooled-standardized).

   Why X-only?  Y_e = X_e@beta + θ + noise, Y_c = X_c@beta + noise.
   OT distances on raw Y therefore encode θ (the prediction target), not shift.
   Y-shift is already captured by the base summary stats (Y_diff_mean, t_like).

   OT features (Sinkhorn regularised transport, ε=0.05 adaptive):
     W_X         = Sinkhorn-W₂(X_std_c, X_std_e)
                   Joint covariate distributional shift.
                   Large in SC4/SC6 (X not exch.), near-zero in SC1/SC3/SC5.

     W_X_max     = max_j Sinkhorn-W₁(X_std_c[:,j], X_std_e[:,j])
                   Worst-case per-dimension shift; catches localised shift.

     ot_ess_ratio = ESS(col-marginals of X Sinkhorn plan) / n_e  ∈ (0, 1]
                   Effective external coverage under the X transport.

     ot_coverage = fraction of X_e units with weight > 1/(2·n_e)
                   Hard coverage indicator complementing the soft ESS.

   Decomposition the network can exploit:
     SC1 (all exch.)  : W_X ≈ 0, ot_ess ≈ 1, ot_cov ≈ 1
     SC4 (X↑, Y OK)   : W_X large, ot_ess/cov ≈ 0
     SC3/SC5 (Y shift): W_X ≈ 0 → correctly signals X is fine

   All other components (network, training loop, NLL loss) are inherited unchanged.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import ot  # POT library


# ============================================================
# NPEMethod — Base class (moved from methods/baselines.py)
# ============================================================

class NPEMethod:
    """
    Neural Posterior Estimation for Bayesian Dynamic Borrowing.

    Key Innovation: Amortized inference
    ===================================
    Instead of running MCMC for each new dataset (slow), we:
    1. Train a neural network on many simulated scenarios
    2. Network learns to map (D_c, D_e) → posterior p(θ | D_c, D_e)
    3. At inference: one forward pass through network (fast!)

    Training Phase (once, offline):
    - Simulate N scenarios with varying parameters
    - For each: generate (D_c, D_e, θ_true)
    - Train network to predict θ from summary statistics

    Inference Phase (per dataset):
    - Compute summary statistics from observed data
    - Single forward pass → posterior parameters
    - Sample from predicted posterior

    Network Architecture:
    - Input: Summary statistics from both datasets
    - Hidden: MLP with ReLU activations
    - Output: (μ_posterior, σ_posterior) for Gaussian posterior
    """

    def __init__(
        self,
        hidden_dims: List[int] = [128, 128, 64],
        learning_rate: float = 0.001,
        dropout_rate: float = 0.1
    ):
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.network = None
        self.scaler = None
        self.training_history = {'train_loss': [], 'val_loss': []}

    # ----------------------------------------------------------
    # Overlap-prior inspired helpers
    # ----------------------------------------------------------
    def _fit_propensity_and_overlap_weights(
        self,
        X_concurrent: np.ndarray,
        X_external: np.ndarray,
        clip: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit e(x)=P(S=1|X=x) and compute overlap weights for external units.

        S=1: concurrent, S=0: external.
        Overlap weight (Li 2022): f(e) = 2*sqrt(e*(1-e)).
        """
        X = np.vstack([X_concurrent, X_external])
        S = np.concatenate([np.ones(X_concurrent.shape[0]), np.zeros(X_external.shape[0])])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X, S)
        e_all = clf.predict_proba(X)[:, 1]
        e_all = np.clip(e_all, clip, 1.0 - clip)
        e_e = e_all[X_concurrent.shape[0]:]
        w_e = 2.0 * np.sqrt(e_e * (1.0 - e_e))
        return e_e, w_e

    def _wmean(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        w = w / (w.sum() + 1e-12)
        if x.ndim == 2:
            return (w[:, None] * x).sum(axis=0)
        return (w * x).sum()

    def _wvar(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        mu = self._wmean(x, w)
        w = np.asarray(w, dtype=float)
        w = w / (w.sum() + 1e-12)
        if x.ndim == 2:
            return (w[:, None] * (x - mu) ** 2).sum(axis=0)
        return (w * (x - mu) ** 2).sum()

    def _ess(self, w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        return float((w.sum() ** 2) / (np.sum(w ** 2) + 1e-12))

    # ----------------------------------------------------------
    # Summary statistics
    # ----------------------------------------------------------
    def compute_summary_statistics(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray
    ) -> np.ndarray:
        """Compute fixed-length summary statistics for NPE.

        Overlap-prior inspiration (Li 2022):
        - Fit propensity e(x)=P(S=1|X=x) where S indicates data source.
        - Compute overlap weights for external units: w_i = 2*sqrt(e_i*(1-e_i)).
        - Use overlap-weighted external summaries so that non-overlapping external
          units contribute less to the network input.

        Returns: 1D array of (6p+15) features for p=6 covariates → 51 features.
        """
        p = X_concurrent.shape[1]
        n_c = len(Y_concurrent)
        n_e = len(Y_external)

        # Concurrent stats (unweighted)
        X_c_mean = X_concurrent.mean(axis=0)
        X_c_std = X_concurrent.std(axis=0) + 1e-6
        Y_c_mean = float(Y_concurrent.mean())
        Y_c_std = float(Y_concurrent.std() + 1e-6)
        try:
            X_c_centered = X_concurrent - X_c_mean
            beta_c = np.linalg.lstsq(X_c_centered, Y_concurrent - Y_c_mean, rcond=None)[0]
        except Exception:
            beta_c = np.zeros(p)

        # External stats (overlap-weighted)
        try:
            e_e, w_e = self._fit_propensity_and_overlap_weights(X_concurrent, X_external)
        except Exception:
            e_e = np.full(n_e, 0.5, dtype=float)
            w_e = np.ones(n_e, dtype=float)

        w_mean = float(np.mean(w_e))
        w_std = float(np.std(w_e) + 1e-6)
        ess_e = self._ess(w_e)
        ess_ratio = float(ess_e / (n_e + 1e-12))
        e_min = float(np.min(e_e))
        e_max = float(np.max(e_e))

        X_e_mean = self._wmean(X_external, w_e)
        X_e_std = np.sqrt(self._wvar(X_external, w_e)) + 1e-6
        Y_e_mean = float(self._wmean(Y_external, w_e))
        Y_e_std = float(np.sqrt(self._wvar(Y_external, w_e)) + 1e-6)
        try:
            X_e_centered = X_external - X_e_mean
            Y_e_centered = Y_external - Y_e_mean
            sw = np.sqrt(w_e + 1e-12)
            Xw = X_e_centered * sw[:, None]
            Yw = Y_e_centered * sw
            beta_e = np.linalg.lstsq(Xw, Yw, rcond=None)[0]
        except Exception:
            beta_e = np.zeros(p)

        # Heterogeneity indicators
        X_diff_mean = X_c_mean - X_e_mean
        X_diff_std = X_c_std - X_e_std  # noqa: F841 (computed for completeness)
        Y_diff_mean = Y_c_mean - Y_e_mean
        Y_diff_std = Y_c_std - Y_e_std  # noqa: F841
        beta_diff = beta_c - beta_e
        n_ratio = float(n_c / (n_e + 1.0))
        ess_ratio_log = float(np.log(ess_e + 1.0))
        t_like = float(
            np.abs(Y_c_mean - Y_e_mean) /
            np.sqrt((Y_c_std ** 2) / (n_c + 1e-12) + (Y_e_std ** 2) / (max(ess_e, 1.0)) + 1e-6)
        )

        summary = np.concatenate([
            # Concurrent stats: 2p + 3
            X_c_mean, X_c_std, [Y_c_mean], [Y_c_std], [np.log(n_c + 1.0)],
            # External stats (overlap-weighted): 2p + 3
            X_e_mean, X_e_std, [Y_e_mean], [Y_e_std], [np.log(n_e + 1.0)],
            # Overlap diagnostics: 6
            [w_mean], [w_std], [ess_ratio_log], [ess_ratio], [e_min], [e_max],
            # Heterogeneity: p + 1 + p + 2
            X_diff_mean,
            [Y_diff_mean / (Y_c_std + Y_e_std + 1e-6)],
            beta_diff, [n_ratio], [t_like],
        ])
        return summary

    # ----------------------------------------------------------
    # Training data generation
    # ----------------------------------------------------------
    def generate_training_data(
        self,
        n_simulations: int = 1000,
        n_concurrent: int = 100,
        n_external: int = 200,
        p_covariates: int = 6,
        random_seed: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate large-scale simulation dataset for offline training.

        Balanced shift taxonomy — each scenario is independently drawn as one of:

          Type 0 (25%): No shift — all exchangeable.
          Type 1 (25%): X location shift only.
                        mu_e offset sampled continuously; Y generated with same beta.
          Type 2 (25%): Y additive shift only.
                        delta_e ~ Uniform(-2, 2); X exchangeable.
          Type 3 (15%): Both X location + Y additive shift (SC6 analogue).
          Type 4 (15%): Structural shift — covariate scale/correlation or
                        regression-beta heterogeneity.

        Additional design choices vs original:
          - n_c sampled from {50, 100, 200, 500} (adds small-sample scenarios)
          - mu_e location shift: continuous Uniform(-2, 2) × mu_c scale
          - delta_e: continuous Uniform(-2, 2) rather than discrete {0, 0.5, 1, 1.5}
          - Type 4 adds: per-dimension covariance scale shift and beta perturbation
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        summaries = []
        thetas = []

        if verbose:
            print(f"Generating {n_simulations} training scenarios...")

        # Base concurrent covariance: compound symmetry ρ=0.5
        Sigma_c = np.ones((p_covariates, p_covariates)) * 0.5
        np.fill_diagonal(Sigma_c, 1.0)

        # Shift-type boundaries for balanced sampling
        shift_probs = np.array([0.25, 0.25, 0.25, 0.15, 0.10])
        shift_cumul = np.cumsum(shift_probs)

        for i in range(n_simulations):
            # --- Sample sizes ---
            Nc_i = int(np.random.choice([50, 100, 200, 500]))
            Ne_i = n_external

            if verbose and (i + 1) % 200 == 0:
                print(f"  Generated {i + 1}/{n_simulations} scenarios...")

            # --- Base parameters ---
            theta_true = np.random.normal(1.0, 1.0)
            beta = np.random.normal(0.3, 0.2, size=p_covariates)
            sigma = np.random.uniform(0.5, 2.0)
            mu_c = np.random.uniform(0.5, 1.5)

            # --- Shift type ---
            u = np.random.uniform()
            shift_type = int(np.searchsorted(shift_cumul, u))

            # Defaults (no shift)
            mu_e = mu_c * np.ones(p_covariates)
            mu_c_vec = mu_c * np.ones(p_covariates)
            delta_e = 0.0
            Sigma_e = Sigma_c.copy()
            beta_e = beta.copy()
            sigma_e = sigma

            if shift_type == 1:
                # X location shift: continuous multiplier in [0.5, 3] and direction
                direction = np.random.choice([-1.0, 1.0])
                magnitude = np.random.uniform(0.5, 3.0)
                mu_e = mu_c_vec * magnitude * direction if direction > 0 else mu_c_vec / magnitude

            elif shift_type == 2:
                # Y additive shift: continuous in [-2, 2]
                delta_e = np.random.uniform(-2.0, 2.0)

            elif shift_type == 3:
                # Both X location + Y additive
                direction = np.random.choice([-1.0, 1.0])
                magnitude = np.random.uniform(0.5, 3.0)
                mu_e = mu_c_vec * magnitude * direction if direction > 0 else mu_c_vec / magnitude
                delta_e = np.random.uniform(-2.0, 2.0)

            elif shift_type == 4:
                # Structural shift: covariance scale + beta perturbation
                # Per-dimension covariance scale shift (log-uniform)
                diag_scale = np.exp(np.random.uniform(-0.7, 0.7, size=p_covariates))
                D = np.diag(diag_scale)
                Sigma_e = D @ Sigma_c @ D
                # Beta perturbation: changes X-Y relationship for external
                beta_e = beta + np.random.normal(0.0, 0.3, size=p_covariates)
                # Optional Y scale shift
                sigma_e = sigma * np.exp(np.random.uniform(-0.5, 0.5))

            # --- Generate data ---
            X_c = np.random.multivariate_normal(mu_c_vec, Sigma_c, size=Nc_i)
            Y_c = X_c @ beta + theta_true + np.random.randn(Nc_i) * sigma

            X_e = np.random.multivariate_normal(mu_e, Sigma_e, size=Ne_i)
            Y_e = X_e @ beta_e + theta_true + delta_e + np.random.randn(Ne_i) * sigma_e

            summary = self.compute_summary_statistics(X_c, Y_c, X_e, Y_e)
            summaries.append(summary)
            thetas.append(theta_true)

        if verbose:
            print(f"Training data generation complete.")
            print(f"  Scenarios: {len(summaries)}, feature dim: {summaries[0].shape[0]}")
            print(f"  theta range: [{np.min(thetas):.3f}, {np.max(thetas):.3f}]")

        return np.array(summaries), np.array(thetas)

    # ----------------------------------------------------------
    # Network
    # ----------------------------------------------------------
    def build_network(self, input_dim: int) -> nn.Module:
        """Build MLP for posterior estimation. Output: (mean, log_std)."""
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        return nn.Sequential(*layers)

    @staticmethod
    def _nll(pred_mean, pred_log_std, targets):
        """Gaussian negative log-likelihood with two-component stop-gradient.

        Loss = L_mean + L_sigma where:
          L_mean  = 0.5*(y - μ)² / σ_sg²     (σ detached; trains mean only)
          L_sigma = 0.5*(y - μ_sg)² / σ² + log(σ)  (μ detached; trains σ only)

        Fixed-point analysis:
          dL_mean/dμ = 0  ↔  μ = y  (correct mean)
          dL_sigma/d(log σ) = 1 - (y-μ_sg)²/σ² = 0  ↔  σ = |y-μ_sg| = RMSE  (calibrated)

        Why not standard NLL alone:
          Standard NLL has coupling: σ affects the mean gradient through 1/σ².
          Large initial σ → small mean gradient → slow mean convergence.
          The two-component form decouples the update directions.

        Why not the naive stop-gradient L = 0.5*(y-μ)²/σ_sg² + log_std:
          The log_std term has gradient +1 always → σ → 0 (collapse).
          The restoring force must come from L_sigma's (y-μ_sg)²/σ² term.

        Reference: Seitzer et al. (2022) NeurIPS, Section 3.
        """
        pred_std = torch.exp(pred_log_std)
        pred_std_sg = pred_std.detach()
        pred_mean_sg = pred_mean.detach()
        sq_err = (targets - pred_mean) ** 2
        sq_err_sg = (targets - pred_mean_sg) ** 2
        L_mean  = 0.5 * sq_err / (pred_std_sg ** 2 + 1e-6)
        L_sigma = 0.5 * sq_err_sg / (pred_std ** 2 + 1e-6) + pred_log_std
        return L_mean + L_sigma

    def train(
        self,
        n_simulations: int = 1000,
        n_epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        random_seed: Optional[int] = None,
        verbose: bool = True
    ):
        """Train NPE network on simulated data (stop-gradient NLL objective)."""
        if verbose:
            print("=" * 60)
            print("NPE TRAINING")
            print("=" * 60)

        summaries, thetas = self.generate_training_data(
            n_simulations=n_simulations, random_seed=random_seed, verbose=verbose
        )

        n_train = int(len(summaries) * (1 - validation_split))
        train_summaries, val_summaries = summaries[:n_train], summaries[n_train:]
        train_thetas, val_thetas = thetas[:n_train], thetas[n_train:]

        if verbose:
            print(f"\nData split: {len(train_summaries)} train / {len(val_summaries)} val")

        self.scaler = StandardScaler()
        train_summaries = self.scaler.fit_transform(train_summaries)
        val_summaries = self.scaler.transform(val_summaries)

        input_dim = train_summaries.shape[1]
        self.network = self.build_network(input_dim)

        if verbose:
            n_params = sum(p.numel() for p in self.network.parameters())
            print(f"\nNetwork: input={input_dim}, hidden={self.hidden_dims}, params={n_params:,}")

        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 20

        for epoch in range(n_epochs):
            self.network.train()
            epoch_loss, n_batches = 0.0, 0
            indices = np.random.permutation(len(train_summaries))

            for k in range(0, len(train_summaries), batch_size):
                batch_idx = indices[k:k + batch_size]
                batch_s = torch.FloatTensor(train_summaries[batch_idx])
                batch_t = torch.FloatTensor(train_thetas[batch_idx])

                out = self.network(batch_s)
                pred_mean, pred_log_std = out[:, 0], out[:, 1]

                loss = self._nll(pred_mean, pred_log_std, batch_t).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            self.training_history['train_loss'].append(avg_train_loss)

            self.network.eval()
            with torch.no_grad():
                val_s = torch.FloatTensor(val_summaries)
                val_t = torch.FloatTensor(val_thetas)
                out = self.network(val_s)
                pred_mean, pred_log_std = out[:, 0], out[:, 1]
                val_loss = self._nll(pred_mean, pred_log_std, val_t).mean().item()
                mae = torch.mean(torch.abs(val_t - pred_mean))
                rmse = torch.sqrt(torch.mean((val_t - pred_mean) ** 2))

            self.training_history['val_loss'].append(val_loss)
            scheduler.step(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{n_epochs}: "
                      f"Train={avg_train_loss:.4f}, Val={val_loss:.4f}, "
                      f"MAE={mae:.4f}, RMSE={rmse:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1} (best val={best_val_loss:.4f})")
                    break

        if verbose:
            print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    # ----------------------------------------------------------
    # Persistence: save / load
    # ----------------------------------------------------------
    def _extra_config(self) -> dict:
        """Subclass hook: return extra __init__ kwargs to persist."""
        return {}

    def save(self, path: str) -> None:
        """Save trained model to disk.

        Stores: network weights, StandardScaler, all __init__ hyperparameters,
        and training history.  Subclass extra params captured via _extra_config().

        Usage:
            npe.save('models/npe_20260306.pt')
            npe2 = load_npe('models/npe_20260306.pt')
        """
        if self.network is None:
            raise RuntimeError("Model not trained. Call train() before save().")

        checkpoint = {
            'class': self.__class__.__name__,
            'config': {
                'hidden_dims':    self.hidden_dims,
                'learning_rate':  self.learning_rate,
                'dropout_rate':   self.dropout_rate,
                **self._extra_config(),
            },
            'input_dim':         self.scaler.n_features_in_,
            'network_state':     self.network.state_dict(),
            'scaler':            self.scaler,
            'training_history':  self.training_history,
        }
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Model saved → {path}")

    def estimate_treatment_effect(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
        n_samples: int = 5000
    ) -> Dict:
        """Estimate treatment effect via single forward pass through trained network."""
        if self.network is None:
            raise RuntimeError("Network not trained. Call train() first.")

        import time
        t0 = time.time()
        summary = self.compute_summary_statistics(X_concurrent, Y_concurrent, X_external, Y_external)
        summary_norm = self.scaler.transform(summary.reshape(1, -1))

        self.network.eval()
        with torch.no_grad():
            out = self.network(torch.FloatTensor(summary_norm))
            pred_mean = out[0, 0].item()
            pred_std = float(np.exp(out[0, 1].item()))

        samples = np.random.normal(pred_mean, pred_std, size=n_samples)
        return {
            'mean': pred_mean,
            'std': pred_std,
            'ci_low': float(np.percentile(samples, 2.5)),
            'ci_high': float(np.percentile(samples, 97.5)),
            'samples': samples,
            'inference_time': time.time() - t0,
        }


# ============================================================
# OTNPEMethod — Wasserstein-Augmented NPE
# ============================================================


class KLNPEMethod(NPEMethod):
    """NPE augmented with Gaussian-approx KL divergence features on X distribution.

    Approximates X_c and X_e as multivariate Gaussians after pooled per-dimension
    standardization (same standardization as OT-NPE), then appends:

      KL_c_e = KL(N_c || N_e)
      KL_e_c = KL(N_e || N_c)
      SKL    = KL_c_e + KL_e_c

    These are X-only features (no Y) to avoid leaking θ via outcome shift.
    """

    def __init__(self, ridge: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.ridge = ridge

    def _extra_config(self):
        return {"ridge": self.ridge}

    @staticmethod
    def _gaussian_kl(mu0: np.ndarray, S0: np.ndarray, mu1: np.ndarray, S1: np.ndarray) -> float:
        """KL(N0 || N1) where N0=N(mu0,S0), N1=N(mu1,S1)."""
        p = mu0.shape[0]
        # Use slogdet for stability
        sign0, logdet0 = np.linalg.slogdet(S0)
        sign1, logdet1 = np.linalg.slogdet(S1)
        if sign0 <= 0 or sign1 <= 0:
            # Fallback (should be rare with ridge)
            return 0.0
        # Compute S1^{-1} via solve
        try:
            S1_inv_S0 = np.linalg.solve(S1, S0)
            diff = (mu1 - mu0).reshape(-1, 1)
            quad = float(diff.T @ np.linalg.solve(S1, diff))
        except Exception:
            return 0.0
        tr_term = float(np.trace(S1_inv_S0))
        return 0.5 * (logdet1 - logdet0 - p + tr_term + quad)

    def _kl_features(
        self,
        X_c: np.ndarray,
        Y_c: np.ndarray,
        X_e: np.ndarray,
        Y_e: np.ndarray,
    ) -> np.ndarray:
        """Compute 3 Gaussian-approx KL features on pooled-standardized X."""
        # --- Pooled per-dimension standardization of X (match OT-NPE) ---
        X_pool = np.vstack([X_c, X_e])
        X_std_scale = X_pool.std(axis=0) + 1e-8  # (p,)
        Xc_std = X_c / X_std_scale
        Xe_std = X_e / X_std_scale

        mu_c = Xc_std.mean(axis=0)
        mu_e = Xe_std.mean(axis=0)

        # Sample covariances
        S_c = np.cov(Xc_std, rowvar=False, bias=False)
        S_e = np.cov(Xe_std, rowvar=False, bias=False)

        # Ensure 2D (p=1 case)
        if S_c.ndim == 0:
            S_c = np.array([[float(S_c)]])
        if S_e.ndim == 0:
            S_e = np.array([[float(S_e)]])

        p = mu_c.shape[0]
        ridge = float(self.ridge)
        S_c = S_c + ridge * np.eye(p)
        S_e = S_e + ridge * np.eye(p)

        kl_c_e = self._gaussian_kl(mu_c, S_c, mu_e, S_e)
        kl_e_c = self._gaussian_kl(mu_e, S_e, mu_c, S_c)
        skl = kl_c_e + kl_e_c
        # Clip to avoid wild values if numerical issues
        return np.array([np.clip(kl_c_e, 0.0, 1e6),
                         np.clip(kl_e_c, 0.0, 1e6),
                         np.clip(skl,    0.0, 1e6)], dtype=float)

    def compute_summary_statistics(
        self,
        X_c: np.ndarray,
        Y_c: np.ndarray,
        X_e: np.ndarray,
        Y_e: np.ndarray,
    ) -> np.ndarray:
        base = super().compute_summary_statistics(X_c, Y_c, X_e, Y_e)
        kl_feats = self._kl_features(X_c, Y_c, X_e, Y_e)
        return np.concatenate([base, kl_feats])


class OTNPEMethod(NPEMethod):
    """NPE augmented with 4 Sinkhorn OT features on the covariate (X) distribution.

    Extends NPEMethod by overriding compute_summary_statistics() to append
    4 OT features computed via Sinkhorn regularised transport (POT library).

    All distances are computed on POOLED-STANDARDIZED X so that each covariate
    dimension contributes equally.

    Why X-only (not joint X+Y)?
    ---------------------------
    In the single-arm concurrent design, Y_e includes the treatment effect θ
    (the target variable), while Y_c does not.  OT distances on raw Y therefore
    encode θ itself, not distributional shift — the model would learn to use a
    feature that is a noisy proxy for its own prediction target.  Y-shift
    information (mean difference, variance, t-statistic) is already captured by
    the base summary statistics; the OT layer should add what is missing: a
    geometry-aware measure of covariate shift.

    OT features:
      W_X         = Sinkhorn-W₂(X_std_c, X_std_e)
                    Joint covariate distributional shift (all p dims together).
                    Large in SC4/SC6 (X not exchangeable), near-zero in SC1/SC3/SC5.

      W_X_max     = max_{j=1..p} Sinkhorn-W₁(X_std_c[:,j], X_std_e[:,j])
                    Worst-case per-dimension shift.
                    Catches localised shift in one covariate even when W_X is diluted.

      ot_ess_ratio = ESS(col-marginals of X Sinkhorn plan) / n_e  ∈ (0, 1]
                    Effective external coverage: how many external units are
                    well-matched to concurrent units under the transport plan.
                    Near 1.0 when X distributions overlap fully; near 0 when X_e
                    lies outside the support of X_c.

      ot_coverage = fraction of X_e units with normalised weight > 1/(2·n_e)
                    Hard coverage indicator: how many X_e units receive any
                    meaningful weight.  Complement to the soft ESS.

    Shift-type decomposition the network can exploit:
      SC1 (all exch.)  : W_X ≈ W_X_max ≈ 0, ot_ess ≈ 1, ot_cov ≈ 1
      SC2 (X partial)  : W_X moderate, ot_ess/cov reduced
      SC4 (X no exch.) : W_X large, W_X_max large, ot_ess/cov ≈ 0
      SC3/SC5 (Y shift): W_X ≈ 0 — correctly indicating X is fine
      SC6 (both)       : W_X large and ot_ess/cov ≈ 0

    Design notes:
    - Sinkhorn (ε=0.05 adaptive) is always numerically stable; no EMD needed.
    - Fallback to safe defaults on any numerical exception.
    - Overhead: ~3–5ms per scenario × 5000 training ≈ 15–25s extra. Acceptable.
    - All other components (network, training loop, NLL loss) unchanged.
    """

    def __init__(self, epsilon: float = 0.05, **kwargs):
        """
        Args:
            epsilon: Sinkhorn regularisation. 0.05 balances accuracy and speed
                     for n_c=50, n_e=200 (≈30 iterations to convergence).
            **kwargs: Passed to NPEMethod (hidden_dims, learning_rate, etc.)
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _extra_config(self) -> dict:
        return {'epsilon': self.epsilon}

    def _ot_features(
        self,
        X_c: np.ndarray,
        Y_c: np.ndarray,
        X_e: np.ndarray,
        Y_e: np.ndarray,
    ) -> np.ndarray:
        """Compute 4 X-only OT features via Sinkhorn regularised transport.

        Uses X-only (not Y) because Y_e includes θ (the target), which would
        make W_Y a noisy proxy for the prediction target rather than a genuine
        shift signal.  Y-shift is already captured by the base summary stats.

        Regularisation: reg = epsilon * M.mean() (adaptive) keeps Sinkhorn
        well-conditioned regardless of the scale of X.

        Returns:
            np.ndarray of shape (4,): [W_X, W_X_max, ot_ess_ratio, ot_coverage]
        """
        n_c = X_c.shape[0]
        n_e = X_e.shape[0]
        a = np.ones(n_c) / n_c
        b = np.ones(n_e) / n_e

        # --- Pooled per-dimension standardization of X ---
        X_pool = np.vstack([X_c, X_e])
        X_std_scale = X_pool.std(axis=0) + 1e-8   # (p,)
        Xc_std = X_c / X_std_scale                 # (n_c, p)
        Xe_std = X_e / X_std_scale                 # (n_e, p)

        M_X = None

        # --- W_X: Sinkhorn-W₂ on joint X (all dims together) ---
        try:
            M_X = ot.dist(Xc_std, Xe_std, metric='sqeuclidean')
            reg_x = max(self.epsilon * float(M_X.mean()), 1e-9)
            W_X = float(np.clip(
                ot.sinkhorn2(a, b, M_X, reg=reg_x, numItermax=1000),
                0.0, 1e9
            ))
        except Exception:
            W_X = 0.0

        # --- W_X_max: max per-dimension Sinkhorn-W₁ ---
        try:
            p = Xc_std.shape[1]
            dim_dists = []
            for j in range(p):
                xc_j = Xc_std[:, j].reshape(-1, 1)
                xe_j = Xe_std[:, j].reshape(-1, 1)
                M_j = ot.dist(xc_j, xe_j, metric='euclidean')
                reg_j = max(self.epsilon * float(M_j.mean()), 1e-9)
                w_j = float(np.clip(
                    ot.sinkhorn2(a, b, M_j, reg=reg_j, numItermax=500),
                    0.0, 1e9
                ))
                dim_dists.append(w_j)
            W_X_max = float(max(dim_dists))
        except Exception:
            W_X_max = 0.0

        # --- ot_ess_ratio and ot_coverage from X Sinkhorn plan ---
        try:
            if M_X is None:
                M_X = ot.dist(Xc_std, Xe_std, metric='sqeuclidean')
            reg_x2 = max(self.epsilon * float(M_X.mean()), 1e-9)
            plan = ot.sinkhorn(a, b, M_X, reg=reg_x2, numItermax=1000)
            # col_marg[j] = total weight assigned to external unit j
            col_marg = plan.sum(axis=0) * n_e   # rescale: uniform = 1.0
            col_marg = np.clip(col_marg, 1e-12, None)
            ess = float((col_marg.sum() ** 2) / np.sum(col_marg ** 2))
            ot_ess_ratio = float(np.clip(ess / n_e, 0.0, 1.0))
            # coverage: fraction with non-trivial weight (> half of uniform)
            ot_coverage = float(np.mean(col_marg > 0.5))
        except Exception:
            ot_ess_ratio = 1.0
            ot_coverage = 1.0

        return np.array([W_X, W_X_max, ot_ess_ratio, ot_coverage])

    def compute_summary_statistics(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
    ) -> np.ndarray:
        """Base summary statistics + 4 X-only OT features appended."""
        base = super().compute_summary_statistics(X_concurrent, Y_concurrent, X_external, Y_external)
        ot_feats = self._ot_features(X_concurrent, Y_concurrent, X_external, Y_external)
        return np.concatenate([base, ot_feats])


# ============================================================
# Module-level loader (dispatches to NPEMethod or OTNPEMethod)
# ============================================================

def load_npe(path: str):
    """Load a saved NPEMethod or OTNPEMethod from disk.

    Automatically dispatches to the correct class based on the saved class name.

    Usage:
        npe  = load_npe('models/npe_20260306.pt')
        otnpe = load_npe('models/otnpe_20260306.pt')
    """
    _REGISTRY = {'NPEMethod': NPEMethod, 'OTNPEMethod': OTNPEMethod, 'KLNPEMethod': KLNPEMethod, 'KLKSDistNPEMethod': KLKSDistNPEMethod, 'JSDNPEMethod': JSDNPEMethod}

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    class_name = checkpoint['class']
    if class_name not in _REGISTRY:
        raise ValueError(f"Unknown class '{class_name}'. Expected one of {list(_REGISTRY)}")

    cls = _REGISTRY[class_name]
    obj = cls(**checkpoint['config'])
    obj.scaler = checkpoint['scaler']
    obj.network = obj.build_network(checkpoint['input_dim'])
    obj.network.load_state_dict(checkpoint['network_state'])
    obj.network.eval()
    obj.training_history = checkpoint['training_history']
    print(f"Loaded {class_name} from {path} (input_dim={checkpoint['input_dim']})")
    return obj


class KLKSDistNPEMethod(KLNPEMethod):
    """
    KL+KS-NPE: Gaussian-KL(X) features + 1D KS distance on residualized outcomes.

    Summary dims:
      - KL-NPE summary: base(51) + KL(3) = 54
      - Append KS distance on residualized outcomes: +1
      => total dim = 55

    Residual KS (leakage-aware):
      - Split concurrent indices into I_fit (70%) and I_sum (30%).
      - Fit beta_hat on I_fit via OLS of centered Y on centered X.
      - Residuals:
          r_c = Y_c(I_sum) - (X_c(I_sum)-mean(X_fit))^T beta_hat
          r_e = Y_e        - (X_e       -mean(X_fit))^T beta_hat
      - KS distance compares empirical CDFs of r_c vs r_e.
    """

    def __init__(self, *args, ks_split_ratio=0.7, ks_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ks_split_ratio = float(ks_split_ratio)
        self.ks_seed = int(ks_seed)

    @staticmethod
    def _ks_statistic(x, y):
        import numpy as np
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size == 0 or y.size == 0:
            return float("nan")
        xs = np.sort(x)
        ys = np.sort(y)
        allv = np.sort(np.concatenate([xs, ys]))
        cdfx = np.searchsorted(xs, allv, side="right") / xs.size
        cdfy = np.searchsorted(ys, allv, side="right") / ys.size
        return float(np.max(np.abs(cdfx - cdfy)))

    def _residual_ks_feature(self, Xc, Yc, Xe, Ye):
        import numpy as np
        n = Xc.shape[0]
        rng = np.random.default_rng(self.ks_seed)
        perm = rng.permutation(n)
        n_fit = max(10, int(self.ks_split_ratio * n))
        fit_idx = perm[:n_fit]
        sum_idx = perm[n_fit:]
        if sum_idx.size < 5:
            fit_idx = np.arange(n)
            sum_idx = np.arange(n)

        X_fit = Xc[fit_idx]
        Y_fit = Yc[fit_idx]
        muX = X_fit.mean(axis=0)
        muY = float(Y_fit.mean())

        try:
            beta_hat = np.linalg.lstsq(X_fit - muX, Y_fit - muY, rcond=None)[0]
        except Exception:
            beta_hat = np.zeros(Xc.shape[1])

        r_c = Yc[sum_idx] - ((Xc[sum_idx] - muX) @ beta_hat)
        r_e = Ye          - ((Xe - muX) @ beta_hat)

        return self._ks_statistic(r_c, r_e)

    def compute_summary_statistics(self, X_concurrent, Y_concurrent, X_external, Y_external):
        base_kl = super().compute_summary_statistics(X_concurrent, Y_concurrent, X_external, Y_external)
        try:
            ks = self._residual_ks_feature(X_concurrent, Y_concurrent, X_external, Y_external)
        except Exception:
            ks = float("nan")
        return np.concatenate([base_kl, np.array([ks], dtype=float)])



class JSDNPEMethod(NPEMethod):
    """
    JSD-NPE: base NPE summary + Jensen-Shannon divergence features on X.

    Uses a Gaussian approximation on pooled-standardized covariates:
      P = N(mu_c, S_c), Q = N(mu_e, S_e),
      M ≈ N(mu_m, S_m), where the 50/50 mixture is moment-matched by
          mu_m = 0.5 * (mu_c + mu_e)
          S_m  = 0.5*(S_c + S_e) + 0.25*(mu_c-mu_e)(mu_c-mu_e)^T

    Then:
      JSD(P, Q) ≈ 0.5 * KL(P || M) + 0.5 * KL(Q || M)

    Appended features:
      jsd_full    : multivariate Gaussian-approx JSD on all X dimensions
      jsd_max_dim : maximum 1D marginal JSD across covariates

    These are X-only features to avoid outcome-target leakage.
    """

    def __init__(self, ridge: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.ridge = ridge

    def _extra_config(self):
        return {"ridge": self.ridge}

    @staticmethod
    def _gaussian_kl(mu0: np.ndarray, S0: np.ndarray, mu1: np.ndarray, S1: np.ndarray) -> float:
        p = mu0.shape[0]
        sign0, logdet0 = np.linalg.slogdet(S0)
        sign1, logdet1 = np.linalg.slogdet(S1)
        if sign0 <= 0 or sign1 <= 0:
            return 0.0
        try:
            S1_inv_S0 = np.linalg.solve(S1, S0)
            diff = (mu1 - mu0).reshape(-1, 1)
            quad = float(diff.T @ np.linalg.solve(S1, diff))
        except Exception:
            return 0.0
        tr_term = float(np.trace(S1_inv_S0))
        return 0.5 * (logdet1 - logdet0 - p + tr_term + quad)

    def _jsd_features(
        self,
        X_c: np.ndarray,
        Y_c: np.ndarray,
        X_e: np.ndarray,
        Y_e: np.ndarray,
    ) -> np.ndarray:
        # pooled standardization (same spirit as OT/KL versions)
        X_pool = np.vstack([X_c, X_e])
        X_std_scale = X_pool.std(axis=0) + 1e-8
        Xc_std = X_c / X_std_scale
        Xe_std = X_e / X_std_scale

        mu_c = Xc_std.mean(axis=0)
        mu_e = Xe_std.mean(axis=0)

        S_c = np.cov(Xc_std, rowvar=False, bias=False)
        S_e = np.cov(Xe_std, rowvar=False, bias=False)
        if np.ndim(S_c) == 0:
            S_c = np.array([[float(S_c)]])
        if np.ndim(S_e) == 0:
            S_e = np.array([[float(S_e)]])

        p = mu_c.shape[0]
        ridge = float(self.ridge)
        S_c = S_c + ridge * np.eye(p)
        S_e = S_e + ridge * np.eye(p)

        # moment-matched midpoint mixture M
        mu_m = 0.5 * (mu_c + mu_e)
        diff = (mu_c - mu_e).reshape(-1, 1)
        S_m = 0.5 * (S_c + S_e) + 0.25 * (diff @ diff.T) + ridge * np.eye(p)

        jsd_full = 0.5 * self._gaussian_kl(mu_c, S_c, mu_m, S_m) + \
                   0.5 * self._gaussian_kl(mu_e, S_e, mu_m, S_m)

        # marginal (1D) JSDs
        jsd_dims = []
        for j in range(Xc_std.shape[1]):
            mc = float(np.mean(Xc_std[:, j]))
            me = float(np.mean(Xe_std[:, j]))
            vc = float(np.var(Xc_std[:, j], ddof=1) + ridge)
            ve = float(np.var(Xe_std[:, j], ddof=1) + ridge)

            mm = 0.5 * (mc + me)
            vm = 0.5 * (vc + ve) + 0.25 * (mc - me) ** 2 + ridge

            mu0 = np.array([mc], dtype=float)
            mu1 = np.array([me], dtype=float)
            mum = np.array([mm], dtype=float)
            S0 = np.array([[vc]], dtype=float)
            S1 = np.array([[ve]], dtype=float)
            Sm = np.array([[vm]], dtype=float)

            jsd_j = 0.5 * self._gaussian_kl(mu0, S0, mum, Sm) + \
                    0.5 * self._gaussian_kl(mu1, S1, mum, Sm)
            jsd_dims.append(jsd_j)

        jsd_max_dim = float(np.max(jsd_dims)) if jsd_dims else 0.0

        return np.array([
            np.clip(jsd_full, 0.0, 1e6),
            np.clip(jsd_max_dim, 0.0, 1e6),
        ], dtype=float)

    def compute_summary_statistics(
        self,
        X_concurrent: np.ndarray,
        Y_concurrent: np.ndarray,
        X_external: np.ndarray,
        Y_external: np.ndarray,
    ) -> np.ndarray:
        base = super().compute_summary_statistics(X_concurrent, Y_concurrent, X_external, Y_external)
        jsd_feats = self._jsd_features(X_concurrent, Y_concurrent, X_external, Y_external)
        return np.concatenate([base, jsd_feats])
