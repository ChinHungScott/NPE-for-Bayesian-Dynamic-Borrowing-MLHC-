from __future__ import annotations

"""
ADNI NPE sensitivity analysis for the external-shift prior scale.

This script keeps the ADNI cohort construction and ADNI-specific NPE model fixed,
then varies only `delta_sd`, the prior/training scale for the external-vs-
concurrent outcome shift term:

    Y_c ~ Bernoulli(sigmoid(X_c beta + theta))
    Y_e ~ Bernoulli(sigmoid(X_e beta + theta + delta)),
    delta ~ Normal(0, delta_sd^2).

The goal is to check whether the ADNI posterior for theta is stable across
reasonable assumptions about source mismatch.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adni_npe_application import ADNINPEApplication, ADNINPEConfig
from adni_npe_prep import (
    ADNIPrepConfig,
    load_adni_table,
    prepare_adni_npe_application,
    print_adni_application_summary,
)


def load_adni_application(args) -> Dict[str, object]:
    tables = {
        "ADSL": load_adni_table(args.adsl),
        "DXSUM": load_adni_table(args.dxsum),
        "PTDEMOG": load_adni_table(args.ptdemog),
        "MMSE": load_adni_table(args.mmse),
        "ADAS": load_adni_table(args.adas),
        "CDR": load_adni_table(args.cdr),
        "FAQ": load_adni_table(args.faq),
        "APOERES": load_adni_table(args.apoe),
    }

    prep_cfg = ADNIPrepConfig(
        target_month=args.target_month,
        target_month_window=args.target_window,
        require_complete_cases=not args.allow_impute,
    )
    return prepare_adni_npe_application(
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


def run_one_fit(
    data: Dict[str, object],
    delta_sd: float,
    seed: int,
    n_simulations: int,
    epochs: int,
    batch_size: int,
    posterior_samples: int,
    verbose: bool,
) -> Dict[str, float]:
    app = ADNINPEApplication(config=ADNINPEConfig(random_seed=seed, delta_sd=delta_sd))
    app.train(
        X_c=data["X_concurrent"],
        y_c_obs=data["Y_concurrent"],
        X_e=data["X_external"],
        y_e_obs=data["Y_external"],
        n_simulations=n_simulations,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )
    result = app.infer(
        X_c=data["X_concurrent"],
        y_c_obs=data["Y_concurrent"],
        X_e=data["X_external"],
        y_e_obs=data["Y_external"],
        n_posterior_samples=posterior_samples,
    )
    return {
        "delta_sd": delta_sd,
        "seed": seed,
        "posterior_mean": result["posterior_mean"],
        "posterior_std": result["posterior_std"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "interval_width": result["ci_high"] - result["ci_low"],
        "observed_concurrent_rate": result["observed_concurrent_rate"],
        "observed_external_rate": result["observed_external_rate"],
        "theta_empirical_logit_concurrent": result["theta_empirical_logit_concurrent"],
        "theta_empirical_logit_external": result["theta_empirical_logit_external"],
        "theta_empirical_logit_pooled": result["theta_empirical_logit_pooled"],
        "theta_offset_mle_concurrent": result["theta_offset_mle_concurrent"],
        "theta_offset_mle_pooled": result["theta_offset_mle_pooled"],
        "delta_offset_mle_external": result["delta_offset_mle_external"],
        "final_train_loss": app.training_history["train_loss"][-1],
        "final_val_loss": app.training_history["val_loss"][-1],
    }


def summarize_repeats(results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for delta_sd, grp in results.groupby("delta_sd"):
        rows.append({
            "delta_sd": delta_sd,
            "n_repeats": len(grp),
            "posterior_mean_mean": grp["posterior_mean"].mean(),
            "posterior_mean_sd": grp["posterior_mean"].std(ddof=1) if len(grp) > 1 else 0.0,
            "posterior_std_mean": grp["posterior_std"].mean(),
            "ci_low_mean": grp["ci_low"].mean(),
            "ci_high_mean": grp["ci_high"].mean(),
            "interval_width_mean": grp["interval_width"].mean(),
            "interval_width_sd": grp["interval_width"].std(ddof=1) if len(grp) > 1 else 0.0,
            "final_val_loss_mean": grp["final_val_loss"].mean(),
        })
    return pd.DataFrame(rows).sort_values("delta_sd").reset_index(drop=True)


def save_latex_table(summary: pd.DataFrame, out_path: Path) -> None:
    table = summary.copy()
    table["delta_sd"] = table["delta_sd"].map(lambda x: f"{x:.2f}")
    table["posterior_mean"] = table.apply(
        lambda r: f"{r['posterior_mean_mean']:.3f} ({r['posterior_mean_sd']:.3f})",
        axis=1,
    )
    table["posterior_sd"] = table["posterior_std_mean"].map(lambda x: f"{x:.3f}")
    table["ci"] = table.apply(lambda r: f"({r['ci_low_mean']:.3f}, {r['ci_high_mean']:.3f})", axis=1)
    table["interval_width"] = table.apply(
        lambda r: f"{r['interval_width_mean']:.3f} ({r['interval_width_sd']:.3f})",
        axis=1,
    )
    table = table[["delta_sd", "posterior_mean", "posterior_sd", "ci", "interval_width", "n_repeats"]]
    table = table.rename(columns={
        "delta_sd": "$\\delta$ SD",
        "posterior_mean": "Posterior mean",
        "posterior_sd": "Posterior SD",
        "ci": "95\\% interval",
        "interval_width": "Interval width",
        "n_repeats": "Repeats",
    })
    out_path.write_text(table.to_latex(index=False, escape=False), encoding="utf-8")


def make_figures(summary: pd.DataFrame, results_dir: Path) -> None:
    x = summary["delta_sd"].to_numpy()
    mean = summary["posterior_mean_mean"].to_numpy()
    low = summary["ci_low_mean"].to_numpy()
    high = summary["ci_high_mean"].to_numpy()
    yerr = np.vstack([mean - low, high - mean])

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.errorbar(x, mean, yerr=yerr, fmt="o-", color="#8c564b", capsize=4, linewidth=2)
    ax.axhline(summary["theta_empirical_logit_concurrent"].iloc[0]
               if "theta_empirical_logit_concurrent" in summary.columns else np.nan,
               color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_xlabel(r"External-shift training scale $\delta_{\mathrm{sd}}$")
    ax.set_ylabel(r"Posterior mean of $\theta$ with 95\% interval")
    ax.set_title("ADNI NPE Sensitivity to External-Shift Scale")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    out = results_dir / "adni_delta_sd_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.plot(x, summary["interval_width_mean"], "o-", color="#1f77b4", linewidth=2)
    ax.set_xlabel(r"External-shift training scale $\delta_{\mathrm{sd}}$")
    ax.set_ylabel("Mean 95% interval width")
    ax.set_title("Posterior Precision Across Sensitivity Settings")
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    out = results_dir / "adni_delta_sd_interval_width.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ADNI NPE sensitivity over delta_sd values.")
    parser.add_argument("--adsl", required=True)
    parser.add_argument("--dxsum", required=True)
    parser.add_argument("--ptdemog", required=True)
    parser.add_argument("--mmse", required=True)
    parser.add_argument("--adas", required=True)
    parser.add_argument("--cdr", required=True)
    parser.add_argument("--faq", required=True)
    parser.add_argument("--apoe", required=True)
    parser.add_argument("--target-month", type=float, default=24.0)
    parser.add_argument("--target-window", type=float, default=6.0)
    parser.add_argument("--allow-impute", action="store_true")
    parser.add_argument("--delta-sd-values", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25])
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--n-simulations", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--posterior-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260401)
    parser.add_argument("--results-dir", default="adni_delta_sensitivity_results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_adni_application(args)
    print_adni_application_summary(data)

    records = []
    for delta_sd in args.delta_sd_values:
        for repeat in range(args.n_repeats):
            seed = args.seed + 1000 * repeat
            print("\n" + "=" * 72)
            print(f"delta_sd={delta_sd:.3f}, repeat={repeat + 1}/{args.n_repeats}, seed={seed}")
            print("=" * 72)
            row = run_one_fit(
                data=data,
                delta_sd=delta_sd,
                seed=seed,
                n_simulations=args.n_simulations,
                epochs=args.epochs,
                batch_size=args.batch_size,
                posterior_samples=args.posterior_samples,
                verbose=not args.quiet,
            )
            records.append(row)
            print(
                f"Posterior mean={row['posterior_mean']:.4f}, "
                f"SD={row['posterior_std']:.4f}, "
                f"CI=({row['ci_low']:.4f}, {row['ci_high']:.4f}), "
                f"width={row['interval_width']:.4f}"
            )

    results = pd.DataFrame(records)
    raw_path = results_dir / "adni_delta_sd_sensitivity_raw.csv"
    results.to_csv(raw_path, index=False)

    summary = summarize_repeats(results)
    # Carry fixed comparator columns into the summary for easier plotting/reporting.
    for col in [
        "theta_empirical_logit_concurrent",
        "theta_empirical_logit_external",
        "theta_empirical_logit_pooled",
        "theta_offset_mle_concurrent",
        "theta_offset_mle_pooled",
        "delta_offset_mle_external",
        "observed_concurrent_rate",
        "observed_external_rate",
    ]:
        summary[col] = results[col].iloc[0]

    summary_path = results_dir / "adni_delta_sd_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    save_latex_table(summary, results_dir / "adni_delta_sd_sensitivity_table.tex")
    make_figures(summary, results_dir)

    print("\nSensitivity summary:")
    print(summary[[
        "delta_sd",
        "posterior_mean_mean",
        "posterior_mean_sd",
        "posterior_std_mean",
        "ci_low_mean",
        "ci_high_mean",
        "interval_width_mean",
    ]].to_string(index=False))
    print(f"\nSaved results to {results_dir.resolve()}")


if __name__ == "__main__":
    main()
