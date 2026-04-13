from __future__ import annotations

"""
Run simple predictive baselines on the ADNI real-world borrowing setup.

Goal
----
Create a paper-friendly benchmark table before plugging in more complex methods.

Evaluation target
-----------------
Concurrent population only.

Methods
-------
- External-only logistic regression
- Concurrent-only logistic regression
- Pooled logistic regression

Protocol
--------
- Split the concurrent cohort into train/test repeatedly.
- Train each baseline and evaluate on the held-out concurrent test set.
- External cohort is used as-is for the external-only and pooled methods.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adni_npe_prep import (
    ADNIPrepConfig,
    load_adni_table,
    prepare_adni_npe_application,
    print_adni_application_summary,
)


def _build_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000)),
    ])


def _evaluate_binary(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(float)
    return {
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, prob)),
        "logloss": float(log_loss(y_true, np.column_stack([1.0 - prob, prob]), labels=[0.0, 1.0])),
        "mean_pred": float(np.mean(prob)),
        "mean_obs": float(np.mean(y_true)),
    }


def run_benchmarks(
    X_concurrent: np.ndarray,
    Y_concurrent: np.ndarray,
    X_external: np.ndarray,
    Y_external: np.ndarray,
    n_splits: int = 30,
    test_size: float = 0.3,
    random_seed: int = 20260331,
) -> pd.DataFrame:
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_seed)
    records: List[Dict[str, float]] = []

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X_concurrent, Y_concurrent), start=1):
        Xc_train = X_concurrent[train_idx]
        yc_train = Y_concurrent[train_idx]
        Xc_test = X_concurrent[test_idx]
        yc_test = Y_concurrent[test_idx]

        train_sets = {
            "ExternalOnly": (X_external, Y_external),
            "ConcurrentOnly": (Xc_train, yc_train),
            "Pooled": (np.vstack([X_external, Xc_train]), np.concatenate([Y_external, yc_train])),
        }

        for method_name, (X_train, y_train) in train_sets.items():
            model = _build_model()
            model.fit(X_train, y_train)
            prob = model.predict_proba(Xc_test)[:, 1]
            metrics = _evaluate_binary(yc_test, prob)
            records.append({
                "split": split_id,
                "method": method_name,
                **metrics,
            })

    return pd.DataFrame(records)


def summarize_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, grp in df.groupby("method"):
        rows.append({
            "method": method,
            "AUC_mean": grp["auc"].mean(),
            "AUC_sd": grp["auc"].std(ddof=1),
            "Accuracy_mean": grp["accuracy"].mean(),
            "Accuracy_sd": grp["accuracy"].std(ddof=1),
            "Brier_mean": grp["brier"].mean(),
            "Brier_sd": grp["brier"].std(ddof=1),
            "LogLoss_mean": grp["logloss"].mean(),
            "LogLoss_sd": grp["logloss"].std(ddof=1),
            "MeanPred_mean": grp["mean_pred"].mean(),
            "MeanObs_mean": grp["mean_obs"].mean(),
            "n_splits": len(grp),
        })
    out = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    return out


def print_summary(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 96)
    print("ADNI BASELINE BENCHMARKS  (test set: held-out concurrent cohort)")
    print("=" * 96)
    print(
        f"{'Method':<16} {'AUC':>14} {'Accuracy':>14} {'Brier':>14} {'LogLoss':>14} {'MeanPred':>10} {'MeanObs':>10}"
    )
    print("-" * 96)
    for _, row in summary.iterrows():
        print(
            f"{row['method']:<16} "
            f"{row['AUC_mean']:.3f} ({row['AUC_sd']:.3f})".rjust(14) + " "
            f"{row['Accuracy_mean']:.3f} ({row['Accuracy_sd']:.3f})".rjust(14) + " "
            f"{row['Brier_mean']:.3f} ({row['Brier_sd']:.3f})".rjust(14) + " "
            f"{row['LogLoss_mean']:.3f} ({row['LogLoss_sd']:.3f})".rjust(14) + " "
            f"{row['MeanPred_mean']:.3f}".rjust(10) + " "
            f"{row['MeanObs_mean']:.3f}".rjust(10)
        )


def save_latex_table(summary: pd.DataFrame, out_path: Path) -> None:
    table = summary.copy()
    for metric in ["AUC", "Accuracy", "Brier", "LogLoss"]:
        table[metric] = table.apply(
            lambda r: f"{r[f'{metric}_mean']:.3f} ({r[f'{metric}_sd']:.3f})",
            axis=1,
        )
    table["MeanPred"] = table["MeanPred_mean"].map(lambda x: f"{x:.3f}")
    table["MeanObs"] = table["MeanObs_mean"].map(lambda x: f"{x:.3f}")
    table = table[["method", "AUC", "Accuracy", "Brier", "LogLoss", "MeanPred", "MeanObs"]]
    table = table.rename(columns={"method": "Method"})
    latex = table.to_latex(index=False, escape=False)
    out_path.write_text(latex, encoding="utf-8")


def make_figures(benchmark_df: pd.DataFrame, summary_df: pd.DataFrame, results_dir: Path) -> None:
    metric_specs = [
        ("auc", "AUC", True),
        ("accuracy", "Accuracy", True),
        ("brier", "Brier Score", False),
        ("logloss", "Log Loss", False),
    ]

    # Split-wise boxplots
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    methods = ["ExternalOnly", "ConcurrentOnly", "Pooled"]
    colors = ["#8c564b", "#1f77b4", "#2ca02c"]
    for ax, (metric_col, metric_label, higher_better) in zip(axes, metric_specs):
        data = [benchmark_df.loc[benchmark_df["method"] == m, metric_col].to_numpy() for m in methods]
        bp = ax.boxplot(data, patch_artist=True, labels=methods)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(metric_label, fontsize=10)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        if metric_col in {"auc", "accuracy"}:
            ax.set_ylim(0.0, 1.0)
    fig.suptitle("ADNI Baseline Benchmark Distribution Across Splits", fontsize=12)
    fig.tight_layout()
    fig.savefig(results_dir / "adni_baseline_benchmark_boxplots.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Mean metric bar chart
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (metric_col, metric_label, higher_better) in zip(axes, metric_specs):
        mean_col = metric_label.replace(" ", "") + "_mean"
        sd_col = metric_label.replace(" ", "") + "_sd"
        if metric_label == "Brier Score":
            mean_col, sd_col = "Brier_mean", "Brier_sd"
        elif metric_label == "Log Loss":
            mean_col, sd_col = "LogLoss_mean", "LogLoss_sd"
        elif metric_label == "Accuracy":
            mean_col, sd_col = "Accuracy_mean", "Accuracy_sd"
        elif metric_label == "AUC":
            mean_col, sd_col = "AUC_mean", "AUC_sd"
        x = np.arange(len(summary_df))
        ax.bar(x, summary_df[mean_col], yerr=summary_df[sd_col], color=colors, alpha=0.8, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df["method"], rotation=20, fontsize=8)
        ax.set_title(metric_label, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        if metric_col in {"auc", "accuracy"}:
            ax.set_ylim(0.0, 1.0)
    fig.suptitle("ADNI Baseline Benchmark Means", fontsize=12)
    fig.tight_layout()
    fig.savefig(results_dir / "adni_baseline_benchmark_means.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple predictive baselines for the ADNI real-world application.")
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
    parser.add_argument("--n-splits", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260331)
    parser.add_argument("--results-dir", default="adni_results")
    args = parser.parse_args()

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
    print_adni_application_summary(data)

    X_c = data["X_concurrent"]
    y_c = data["Y_concurrent"]
    X_e = data["X_external"]
    y_e = data["Y_external"]

    if len(X_c) == 0 or len(X_e) == 0:
        raise RuntimeError("No subjects available after cohort construction. Cannot run benchmarks.")

    benchmark_df = run_benchmarks(
        X_concurrent=X_c,
        Y_concurrent=y_c,
        X_external=X_e,
        Y_external=y_e,
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_seed=args.seed,
    )
    summary_df = summarize_benchmarks(benchmark_df)
    print_summary(summary_df)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(results_dir / "adni_baseline_benchmark_splits.csv", index=False)
    summary_df.to_csv(results_dir / "adni_baseline_benchmark_summary.csv", index=False)
    data["analytic_subjects"].to_csv(results_dir / "adni_application_subjects.csv", index=False)
    save_latex_table(summary_df, results_dir / "adni_baseline_benchmark_summary.tex")
    make_figures(benchmark_df, summary_df, results_dir)
    print(f"\nSaved results to {results_dir.resolve()}")


if __name__ == "__main__":
    main()
