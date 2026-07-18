import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


DEFAULT_OTCODE_DIR = "."
DEFAULT_OUTPUT_DIR = Path("ols_perturbation_results")


def add_otcode_to_path(otcode_dir: str) -> None:
    if otcode_dir not in sys.path:
        sys.path.insert(0, otcode_dir)


def get_beta_diff_slice(p_covariates: int) -> slice:
    # Summary layout in npe.py:
    # concurrent: 2p + 3
    # external:   2p + 3
    # overlap:    6
    # x diff:     p
    # y diff:     1
    # beta diff:  p
    start = (2 * p_covariates + 3) + (2 * p_covariates + 3) + 6 + p_covariates + 1
    end = start + p_covariates
    return slice(start, end)


def predict_from_summary(npe_model, summary_vec: np.ndarray, n_samples: int, rng: np.random.Generator):
    summary_norm = npe_model.scaler.transform(summary_vec.reshape(1, -1))
    npe_model.network.eval()
    with torch.no_grad():
        out = npe_model.network(torch.FloatTensor(summary_norm))
        pred_mean = out[0, 0].item()
        pred_std = float(np.exp(out[0, 1].item()))

    samples = rng.normal(pred_mean, pred_std, size=n_samples)
    return {
        "theta_hat": float(pred_mean),
        "posterior_std": pred_std,
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
    }


def train_or_load_npe(args, NPEMethod, load_npe):
    checkpoint_path = Path(args.checkpoint_path)
    if checkpoint_path.exists() and not args.pretrain:
        print(f"Loading pretrained NPE from {checkpoint_path}")
        return load_npe(str(checkpoint_path))

    print("Pretraining NPE for OLS perturbation experiment...")
    npe = NPEMethod()
    npe.train(
        n_simulations=args.n_sims,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        random_seed=args.seed,
        verbose=True,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    npe.save(str(checkpoint_path))
    return npe


def run_experiment(args):
    add_otcode_to_path(args.otcode_dir)
    from simulation import SimulationConfig, SimulationScenario
    from npe import NPEMethod, load_npe

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npe = train_or_load_npe(args, NPEMethod, load_npe)

    p_covariates = 6
    beta_slice = get_beta_diff_slice(p_covariates)
    rng_global = np.random.default_rng(args.seed)

    scenario_sub_idx = {
        "sc1": 0,
        "sc2": 1,
        "sc3": 1,
        "sc4": 0,
        "sc5": 0,
        "sc6": 0,
    }.get(args.scenario, 0)

    records = []

    for rep in range(args.b_reps):
        rep_seed = int(rng_global.integers(0, 2**31 - 1))
        np.random.seed(rep_seed)
        torch.manual_seed(rep_seed)

        config = SimulationConfig(
            n_concurrent=args.n_concurrent,
            n_external=args.n_external,
            theta_true=args.theta_true,
            random_seed=rep_seed,
        )
        scenario = SimulationScenario(args.scenario, config)
        data = scenario.generate_data(sub_scenario_idx=scenario_sub_idx)

        base_summary = npe.compute_summary_statistics(
            data["X_concurrent"],
            data["Y_concurrent"],
            data["X_external"],
            data["Y_external"],
        )

        for noise_sd in args.noise_sds:
            noisy_summary = base_summary.copy()
            if noise_sd > 0:
                noisy_summary[beta_slice] += rng_global.normal(
                    loc=0.0, scale=noise_sd, size=beta_slice.stop - beta_slice.start
                )

            pred = predict_from_summary(
                npe_model=npe,
                summary_vec=noisy_summary,
                n_samples=args.posterior_samples,
                rng=rng_global,
            )

            records.append(
                {
                    "rep": rep,
                    "scenario": args.scenario,
                    "n_concurrent": args.n_concurrent,
                    "n_external": args.n_external,
                    "theta_true": args.theta_true,
                    "noise_sd": noise_sd,
                    "theta_hat": pred["theta_hat"],
                    "posterior_std": pred["posterior_std"],
                    "ci_low": pred["ci_low"],
                    "ci_high": pred["ci_high"],
                    "covered": float(pred["ci_low"] <= args.theta_true <= pred["ci_high"]),
                    "interval_width": float(pred["ci_high"] - pred["ci_low"]),
                }
            )

        if (rep + 1) % 10 == 0 or (rep + 1) == args.b_reps:
            print(f"Completed {rep + 1}/{args.b_reps} replications")

    raw_df = pd.DataFrame(records)
    raw_path = output_dir / "npe_ols_perturbation_raw.csv"
    raw_df.to_csv(raw_path, index=False)

    summary_rows = []
    for noise_sd, grp in raw_df.groupby("noise_sd"):
        theta_hat = grp["theta_hat"].to_numpy()
        bias = float(theta_hat.mean() - args.theta_true)
        rmse = float(np.sqrt(np.mean((theta_hat - args.theta_true) ** 2)))
        coverage = float(grp["covered"].mean())
        posterior_std_mean = float(grp["posterior_std"].mean())
        interval_width_mean = float(grp["interval_width"].mean())
        ci_low_mean = float(grp["ci_low"].mean())
        ci_high_mean = float(grp["ci_high"].mean())

        summary_rows.append(
            {
                "noise_sd": noise_sd,
                "bias": bias,
                "abs_bias": abs(bias),
                "rmse": rmse,
                "coverage": coverage,
                "posterior_std_mean": posterior_std_mean,
                "ci_low_mean": ci_low_mean,
                "ci_high_mean": ci_high_mean,
                "interval_width_mean": interval_width_mean,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("noise_sd")
    summary_path = output_dir / "npe_ols_perturbation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nOLS perturbation summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sensitivity check for perturbations of the OLS-derived beta_diff summary block in NPE."
    )
    parser.add_argument("--otcode-dir", default=DEFAULT_OTCODE_DIR, help="Path to the OTCode folder.")
    parser.add_argument(
        "--checkpoint-path",
        default=str(DEFAULT_OUTPUT_DIR / "npe_pretrained_for_ols_perturbation.pt"),
        help="Path to save/load the pretrained NPE checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV outputs.",
    )
    parser.add_argument("--pretrain", action="store_true", help="Force retraining even if checkpoint exists.")
    parser.add_argument("--n-sims", type=int, default=5000, help="Number of synthetic training instances.")
    parser.add_argument("--n-epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split for NPE training.")
    parser.add_argument("--scenario", default="sc5", choices=["sc1", "sc2", "sc3", "sc4", "sc5", "sc6"])
    parser.add_argument("--n-concurrent", type=int, default=100)
    parser.add_argument("--n-external", type=int, default=200)
    parser.add_argument("--theta-true", type=float, default=1.0)
    parser.add_argument("--b-reps", type=int, default=50, help="Number of Monte Carlo replications.")
    parser.add_argument(
        "--noise-sds",
        type=float,
        nargs="+",
        default=[0.0, 0.10, 0.25, 0.50],
        help="Noise SDs added to the beta_diff summary block.",
    )
    parser.add_argument("--posterior-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260606)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
