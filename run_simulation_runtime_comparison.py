"""
Runtime comparison for the main-paper no-OT simulation methods.

This script is intentionally separate from the main simulation runner so that
runtime experiments do not overwrite the main result files used in the paper.

Methods compared:
  - PSPower
  - IW
  - Commensurate
  - NPE

The runtime story is reported in two parts:
  1. Per-dataset fit time for each method.
  2. Cumulative time across repeated analyses, where NPE includes a one-time
     training cost plus fast amortized inference.
"""

import argparse
import os
import platform
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running from project root or scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

warnings.filterwarnings("ignore")

from methods.simulation import SimulationConfig, SimulationScenario
from methods.baselines import (
    CommensurateMethod,
    IndividualizedWeightsMethod,
    PSPowerMethod,
)
from methods.npe import NPEMethod, load_npe


SEED = 20260306
DEFAULT_NC_VALUES = [50, 100]
DEFAULT_NE = 200

SCENARIO_SPECS = {
    "sc1": {"sub_idx": 0, "label": "SC1"},
    "sc2": {"sub_idx": 1, "label": "SC2"},
    "sc3": {"sub_idx": 1, "label": "SC3"},
    "sc4": {"sub_idx": 0, "label": "SC4"},
    "sc5": {"sub_idx": 0, "label": "SC5"},
    "sc6": {"sub_idx": 0, "label": "SC6"},
}
SCENARIO_ORDER = list(SCENARIO_SPECS.keys())
METHOD_NAMES = ["PSPower", "IW", "Commensurate", "NPE"]
METHOD_COLORS = {
    "PSPower": "#d62728",
    "IW": "#9467bd",
    "Commensurate": "#7f7f7f",
    "NPE": "#8c564b",
}


def build_results_dir(base_dir: str) -> str:
    out_dir = os.path.join(base_dir, "results_runtime_no_ot")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def build_model_dir(base_dir: str) -> str:
    out_dir = os.path.join(base_dir, "results_main_no_ot_model")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _build_methods(npe_instance=None):
    methods = {
        "PSPower": PSPowerMethod(K=5),
        "IW": IndividualizedWeightsMethod(),
        "Commensurate": CommensurateMethod(),
    }
    if npe_instance is not None:
        methods["NPE"] = npe_instance
    return methods


def _run_method(name, method, data):
    Xc = data["X_concurrent"]
    Yc = data["Y_concurrent"]
    Xe = data["X_external"]
    Ye = data["Y_external"]
    if name == "Commensurate":
        return method.estimate_treatment_effect(Xc, Yc, Xe, Ye, n_samples=2000, n_burnin=500)
    return method.estimate_treatment_effect(Xc, Yc, Xe, Ye)


def run_runtime_study(
    B: int,
    theta_true: float,
    npe_instance,
    results_dir: str,
    nc_values,
    ne: int,
) -> pd.DataFrame:
    rng_global = np.random.default_rng(SEED)
    records = []
    total_cells = len(nc_values) * len(SCENARIO_SPECS) * B
    done = 0
    active_methods = METHOD_NAMES[:]
    if npe_instance is None:
        active_methods = [m for m in active_methods if m != "NPE"]

    for nc in nc_values:
        for sc_id, sc_spec in SCENARIO_SPECS.items():
            sub_idx = sc_spec["sub_idx"]
            rep_seeds = rng_global.integers(0, 2**31, size=B)

            for rep in range(B):
                seed = int(rep_seeds[rep])
                np.random.seed(seed)
                config = SimulationConfig(
                    n_concurrent=nc,
                    n_external=ne,
                    theta_true=theta_true,
                    random_seed=seed,
                )
                scenario = SimulationScenario(sc_id, config)
                data = scenario.generate_data(sub_scenario_idx=sub_idx)
                methods = _build_methods(npe_instance)

                for mname in active_methods:
                    t0 = time.perf_counter()
                    success = True
                    try:
                        result = _run_method(mname, methods[mname], data)
                        theta_hat = result["mean"]
                    except Exception:
                        success = False
                        theta_hat = np.nan
                    elapsed = time.perf_counter() - t0

                    records.append({
                        "Nc": nc,
                        "scenario": sc_id,
                        "rep": rep,
                        "method": mname,
                        "theta_hat": theta_hat,
                        "fit_time_sec": elapsed,
                        "fit_time_ms": 1000.0 * elapsed,
                        "success": success,
                    })

                done += 1
                if done % 20 == 0 or done == total_cells:
                    pct = 100.0 * done / total_cells
                    print(f"  {done}/{total_cells} ({pct:.1f}%)  Nc={nc} {sc_id} rep={rep + 1}")
                    sys.stdout.flush()

    df = pd.DataFrame(records)
    out = os.path.join(results_dir, "runtime_results.csv")
    df.to_csv(out, index=False)
    print(f"\nRuntime records saved to {out}")
    return df


def summarize_runtime(df: pd.DataFrame, results_dir: str) -> pd.DataFrame:
    rows = []
    for (method, nc, scenario), grp in df.groupby(["method", "Nc", "scenario"]):
        rows.append({
            "method": method,
            "Nc": nc,
            "scenario": scenario,
            "n_runs": len(grp),
            "success_rate": float(np.mean(grp["success"])),
            "mean_fit_time_sec": float(grp["fit_time_sec"].mean()),
            "median_fit_time_sec": float(grp["fit_time_sec"].median()),
            "sd_fit_time_sec": float(grp["fit_time_sec"].std(ddof=1)) if len(grp) > 1 else 0.0,
        })

    summary = pd.DataFrame(rows).sort_values(["method", "Nc", "scenario"]).reset_index(drop=True)
    out = os.path.join(results_dir, "runtime_summary_by_cell.csv")
    summary.to_csv(out, index=False)
    print(f"Runtime cell summary saved to {out}")

    overall = (
        df.groupby("method")
        .agg(
            n_runs=("fit_time_sec", "size"),
            success_rate=("success", "mean"),
            mean_fit_time_sec=("fit_time_sec", "mean"),
            median_fit_time_sec=("fit_time_sec", "median"),
            sd_fit_time_sec=("fit_time_sec", "std"),
        )
        .reset_index()
        .sort_values("method")
    )

    if "Commensurate" in set(overall["method"]):
        comm_mean = float(
            overall.loc[overall["method"] == "Commensurate", "mean_fit_time_sec"].iloc[0]
        )
        overall["speedup_vs_commensurate"] = comm_mean / overall["mean_fit_time_sec"]
    else:
        overall["speedup_vs_commensurate"] = np.nan

    out_overall = os.path.join(results_dir, "runtime_summary_overall.csv")
    overall.to_csv(out_overall, index=False)
    print(f"Runtime overall summary saved to {out_overall}")
    return overall


def _latex_num(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def write_online_runtime_table(overall: pd.DataFrame, results_dir: str) -> pd.DataFrame:
    table = overall.copy()
    table["mean_fit_time_ms"] = 1000.0 * table["mean_fit_time_sec"]
    table["median_fit_time_ms"] = 1000.0 * table["median_fit_time_sec"]
    table = table.sort_values("mean_fit_time_sec").reset_index(drop=True)

    out_csv = os.path.join(results_dir, "runtime_online_table.csv")
    table[
        [
            "method",
            "n_runs",
            "mean_fit_time_ms",
            "median_fit_time_ms",
            "speedup_vs_commensurate",
        ]
    ].to_csv(out_csv, index=False)
    print(f"Online runtime table saved to {out_csv}")

    out_tex = os.path.join(results_dir, "runtime_online_table.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Runs & Mean ms & Median ms & Speedup vs. Commensurate \\\\\n")
        f.write("\\midrule\n")
        for _, row in table.iterrows():
            f.write(
                f"{row['method']} & {int(row['n_runs'])} & "
                f"{_latex_num(row['mean_fit_time_ms'])} & "
                f"{_latex_num(row['median_fit_time_ms'])} & "
                f"{_latex_num(row['speedup_vs_commensurate'], digits=1)}$\\times$ \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"Online runtime LaTeX table saved to {out_tex}")
    return table


def write_amortized_runtime_tables(
    overall: pd.DataFrame,
    training_time_sec: float,
    results_dir: str,
    max_analyses: int,
) -> None:
    analysis_counts = sorted(
        {
            1,
            10,
            50,
            100,
            min(500, max_analyses),
            max_analyses,
        }
    )
    analysis_counts = [n for n in analysis_counts if n > 0]

    rows = []
    for n in analysis_counts:
        row = {"n_analyses": n}
        for _, method_row in overall.iterrows():
            method = method_row["method"]
            total = n * float(method_row["mean_fit_time_sec"])
            if method == "NPE":
                total += training_time_sec
            row[method] = total
        rows.append(row)

    cumulative = pd.DataFrame(rows)
    out_csv = os.path.join(results_dir, "runtime_amortized_table.csv")
    cumulative.to_csv(out_csv, index=False)
    print(f"Amortized runtime table saved to {out_csv}")

    out_tex = os.path.join(results_dir, "runtime_amortized_table.tex")
    methods = [m for m in METHOD_NAMES if m in set(overall["method"])]
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l" + "c" * len(methods) + "}\n")
        f.write("\\toprule\n")
        f.write("Number of analyses & " + " & ".join(methods) + " \\\\\n")
        f.write("\\midrule\n")
        for _, row in cumulative.iterrows():
            vals = [_latex_num(row[m], digits=2) for m in methods]
            f.write(f"{int(row['n_analyses'])} & " + " & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"Amortized runtime LaTeX table saved to {out_tex}")

    if "NPE" in set(overall["method"]) and "Commensurate" in set(overall["method"]):
        npe_mean = float(overall.loc[overall["method"] == "NPE", "mean_fit_time_sec"].iloc[0])
        comm_mean = float(
            overall.loc[overall["method"] == "Commensurate", "mean_fit_time_sec"].iloc[0]
        )
        denom = comm_mean - npe_mean
        break_even = np.ceil(training_time_sec / denom) if denom > 0 else np.inf
        break_even_df = pd.DataFrame(
            [
                {
                    "comparison": "NPE vs Commensurate",
                    "npe_training_time_sec": training_time_sec,
                    "npe_online_time_sec": npe_mean,
                    "commensurate_time_sec": comm_mean,
                    "break_even_analyses": break_even,
                }
            ]
        )
        out_break = os.path.join(results_dir, "runtime_break_even.csv")
        break_even_df.to_csv(out_break, index=False)
        print(f"Break-even summary saved to {out_break}")


def load_previous_training_time(results_dir: str) -> float:
    meta_path = os.path.join(results_dir, "runtime_metadata.csv")
    if not os.path.exists(meta_path):
        return 0.0
    try:
        meta = pd.read_csv(meta_path)
        vals = meta.get("npe_training_time_sec", pd.Series(dtype=float)).dropna()
        vals = vals[vals > 0]
        if len(vals) > 0:
            return float(vals.iloc[-1])
    except Exception:
        return 0.0
    return 0.0


def save_metadata(results_dir: str, training_time_sec: float, npe_path: str, args) -> None:
    meta = pd.DataFrame([{
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "theta_true": args.theta_true,
        "b_reps": args.b_reps,
        "nc_values": ",".join(str(x) for x in args.nc_values),
        "n_external": args.ne,
        "max_analyses": args.max_analyses,
        "n_sims": args.n_sims,
        "n_epochs": args.n_epochs,
        "npe_training_time_override_sec": args.npe_training_time_sec,
        "npe_path": npe_path,
        "npe_training_time_sec": training_time_sec,
        "seed": SEED,
    }])
    out = os.path.join(results_dir, "runtime_metadata.csv")
    meta.to_csv(out, index=False)
    print(f"Runtime metadata saved to {out}")


def make_runtime_figures(
    overall: pd.DataFrame,
    training_time_sec: float,
    results_dir: str,
    max_analyses: int,
) -> None:
    ordered = overall.set_index("method").loc[METHOD_NAMES].reset_index()

    # Per-fit runtime bar chart
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(
        ordered["method"],
        ordered["mean_fit_time_sec"],
        color=[METHOD_COLORS[m] for m in ordered["method"]],
        alpha=0.85,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Mean fit time (seconds, log scale)")
    ax.set_title("Per-dataset Runtime Comparison")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out1 = os.path.join(results_dir, "runtime_per_fit.pdf")
    fig.savefig(out1, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out1}")

    # Cumulative runtime plot
    analyses = np.arange(1, max_analyses + 1)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for _, row in ordered.iterrows():
        method = row["method"]
        mean_fit = row["mean_fit_time_sec"]
        cumulative = analyses * mean_fit
        if method == "NPE":
            cumulative = cumulative + training_time_sec
        ax.plot(
            analyses,
            cumulative,
            label=method,
            color=METHOD_COLORS[method],
            linewidth=2.2 if method == "NPE" else 1.7,
        )
    ax.set_xlabel("Number of dataset analyses")
    ax.set_ylabel("Cumulative runtime (seconds)")
    ax.set_title("Amortized Runtime Comparison")
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    out2 = os.path.join(results_dir, "runtime_cumulative.pdf")
    fig.savefig(out2, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out2}")


def main():
    parser = argparse.ArgumentParser(description="Runtime comparison for main-paper methods without OT.")
    parser.add_argument("--pretrain", action="store_true", help="Force re-training NPE even if saved model exists.")
    parser.add_argument("--load", action="store_true", help="Load NPE from disk; error if missing.")
    parser.add_argument("--npe-path", default=None, metavar="PATH",
                        help="Path to NPE .pt file (default: results_main_no_ot_model/npe_pretrained.pt).")
    parser.add_argument("--n-sims", type=int, default=5000, metavar="N",
                        help="Number of training simulations for NPE pretraining.")
    parser.add_argument("--n-epochs", type=int, default=100, metavar="N",
                        help="Number of NPE training epochs.")
    parser.add_argument("--b-reps", type=int, default=10, metavar="B",
                        help="Number of runtime replications per scenario/cell.")
    parser.add_argument("--theta-true", type=float, default=1.0,
                        help="True theta used to generate runtime study datasets.")
    parser.add_argument("--nc-values", type=int, nargs="+", default=DEFAULT_NC_VALUES,
                        help="Concurrent sample sizes to include in the runtime study.")
    parser.add_argument("--ne", type=int, default=DEFAULT_NE,
                        help="External sample size used in the runtime study.")
    parser.add_argument("--max-analyses", type=int, default=500,
                        help="Maximum number of repeated analyses shown in the cumulative runtime figure.")
    parser.add_argument("--npe-training-time-sec", type=float, default=None,
                        help="Optional one-time NPE training time to use in amortized runtime summaries.")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    results_dir = build_results_dir(base_dir)
    model_dir = build_model_dir(base_dir)
    npe_path = args.npe_path or os.path.join(model_dir, "npe_pretrained.pt")

    print("=" * 72)
    print("MAIN-PAPER RUNTIME COMPARISON (NO OT)")
    print(f"B={args.b_reps} replications, Nc in {args.nc_values}, Ne={args.ne}, theta_true={args.theta_true}")
    print(f"Methods: {METHOD_NAMES}")
    print("=" * 72)

    def _should_train(path: str) -> bool:
        if args.pretrain:
            return True
        if args.load:
            if not os.path.exists(path):
                raise FileNotFoundError(f"--load specified but model not found: {path}")
            return False
        return not os.path.exists(path)

    training_time_sec = 0.0
    seed_train = SEED + 1
    train_cfg = {
        "n_simulations": args.n_sims,
        "n_epochs": args.n_epochs,
        "batch_size": 32,
        "random_seed": seed_train,
        "verbose": True,
    }

    if _should_train(npe_path):
        print("\nPre-training NPE for runtime measurement...")
        npe = NPEMethod()
        t0 = time.perf_counter()
        npe.train(**train_cfg)
        training_time_sec = time.perf_counter() - t0
        npe.save(npe_path)
        print(f"NPE saved to {npe_path}")
        print(f"NPE training time: {training_time_sec:.3f} seconds")
    else:
        print(f"\nLoading pre-trained NPE from {npe_path} ...")
        npe = load_npe(npe_path)
        print("Training time not re-measured in this run.")

    if args.npe_training_time_sec is not None:
        training_time_sec = float(args.npe_training_time_sec)
        print(f"Using supplied NPE training time: {training_time_sec:.3f} seconds")
    elif training_time_sec == 0.0:
        previous_training_time_sec = load_previous_training_time(results_dir)
        if previous_training_time_sec > 0:
            training_time_sec = previous_training_time_sec
            print(f"Using previous recorded NPE training time: {training_time_sec:.3f} seconds")
        else:
            print("No NPE training time available; amortized NPE totals will use online time only.")

    df = run_runtime_study(
        B=args.b_reps,
        theta_true=args.theta_true,
        npe_instance=npe,
        results_dir=results_dir,
        nc_values=args.nc_values,
        ne=args.ne,
    )
    overall = summarize_runtime(df, results_dir=results_dir)
    write_online_runtime_table(overall, results_dir=results_dir)
    write_amortized_runtime_tables(
        overall,
        training_time_sec=training_time_sec,
        results_dir=results_dir,
        max_analyses=args.max_analyses,
    )
    save_metadata(results_dir, training_time_sec, npe_path, args)
    make_runtime_figures(
        overall,
        training_time_sec,
        results_dir=results_dir,
        max_analyses=args.max_analyses,
    )

    print("\nOverall mean fit times (seconds):")
    print(overall[["method", "mean_fit_time_sec", "median_fit_time_sec", "success_rate"]].to_string(index=False))
    print(f"\nDone. Outputs in {results_dir}")


if __name__ == "__main__":
    main()
