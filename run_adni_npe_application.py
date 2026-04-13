from __future__ import annotations

"""
Run a simple real-data ADNI application for the baseline NPE-BDB pipeline.

Expected inputs
---------------
Paths to the following ADNI tables, typically as `.rda` files:
    ADSL, DXSUM, PTDEMOG, MMSE, ADAS, CDR, FAQ, APOERES

This script:
1. Loads the ADNI tables.
2. Builds a simple baseline-MCI application dataset.
3. Prints a concise data summary.
4. Fits a plain logistic regression baseline for a quick real-data benchmark.

Note
----
This script does not "train NPE on ADNI". Your NPE remains trained on simulated data.
The ADNI dataset prepared here is the real-world application dataset that you would
feed into your downstream application workflow.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adni_npe_prep import (
    ADNIPrepConfig,
    load_adni_table,
    prepare_adni_npe_application,
    print_adni_application_summary,
)


def _fit_simple_baseline(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000)),
    ])
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(float)

    out = {
        "auc": float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) > 1 else np.nan,
        "accuracy": float(np.mean(pred == y_test)),
        "mean_pred": float(np.mean(prob)),
        "mean_obs": float(np.mean(y_test)),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and summarize an ADNI application dataset for baseline NPE-BDB.")
    parser.add_argument("--adsl", required=True, help="Path to ADSL table (.rda/.csv/.parquet).")
    parser.add_argument("--dxsum", required=True, help="Path to DXSUM table (.rda/.csv/.parquet).")
    parser.add_argument("--ptdemog", required=True, help="Path to PTDEMOG table (.rda/.csv/.parquet).")
    parser.add_argument("--mmse", required=True, help="Path to MMSE table (.rda/.csv/.parquet).")
    parser.add_argument("--adas", required=True, help="Path to ADAS table (.rda/.csv/.parquet).")
    parser.add_argument("--cdr", required=True, help="Path to CDR table (.rda/.csv/.parquet).")
    parser.add_argument("--faq", required=True, help="Path to FAQ table (.rda/.csv/.parquet).")
    parser.add_argument("--apoe", required=True, help="Path to APOERES table (.rda/.csv/.parquet).")
    parser.add_argument("--target-month", type=float, default=24.0, help="Target follow-up month for conversion outcome.")
    parser.add_argument("--target-window", type=float, default=6.0, help="Allowed month gap around the target follow-up.")
    parser.add_argument("--allow-impute", action="store_true", help="Use median imputation instead of complete-case filtering.")
    parser.add_argument("--save-analytic-csv", default=None, help="Optional path to save the final analytic subject-level table.")
    args = parser.parse_args()

    print("=" * 72)
    print("ADNI REAL-WORLD APPLICATION PREP")
    print("=" * 72)

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

    print("Loaded tables:")
    for name, df in tables.items():
        print(f"  {name:<8} shape={df.shape}")

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
    print()
    print_adni_application_summary(data)

    X_c = data["X_concurrent"]
    y_c = data["Y_concurrent"]
    X_e = data["X_external"]
    y_e = data["Y_external"]

    print("\nQuick baseline check")
    print("-" * 72)
    if len(X_c) == 0 or len(X_e) == 0:
        print("Not enough subjects in concurrent or external cohort after filtering.")
    else:
        # Train on external, evaluate on concurrent to reflect borrowing direction.
        metrics = _fit_simple_baseline(X_train=X_e, y_train=y_e, X_test=X_c, y_test=y_c)
        print(f"External -> Concurrent AUC:      {metrics['auc']:.3f}")
        print(f"External -> Concurrent accuracy: {metrics['accuracy']:.3f}")
        print(f"Mean predicted risk:             {metrics['mean_pred']:.3f}")
        print(f"Observed concurrent risk:        {metrics['mean_obs']:.3f}")

    if args.save_analytic_csv:
        out_path = Path(args.save_analytic_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data["analytic_subjects"].to_csv(out_path, index=False)
        print(f"\nSaved analytic subject-level table to {out_path}")


if __name__ == "__main__":
    main()
