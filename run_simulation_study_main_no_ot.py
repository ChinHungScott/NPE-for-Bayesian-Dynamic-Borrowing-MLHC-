"""
Main-text Monte Carlo Simulation Study for Bayesian Dynamic Borrowing
=====================================================================

This version is intentionally restricted to the baseline main-paper methods:
  - PSPower
  - IW
  - Commensurate
  - NPE

OT-augmented variants are excluded here and can be reported separately
in an appendix if desired.

Metrics reported:
  - Bias
  - RMSE
  - Coverage      = fraction of 95% CIs containing theta_true
  - RejectNull    = fraction of 95% CIs excluding 0
  - Power         = RejectNull when theta_true != 0, else NaN
  - TypeIError    = RejectNull when theta_true == 0, else NaN
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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
NC_VALUES = [50, 100]
NE = 200

SCENARIO_SPECS = {
    "sc1": {"sub_idx": 0, "label": "SC1\n(All exch.)"},
    "sc2": {"sub_idx": 1, "label": "SC2\n(X partial)"},
    "sc3": {"sub_idx": 1, "label": "SC3\n(Y partial)"},
    "sc4": {"sub_idx": 0, "label": "SC4\n(X no exch.)"},
    "sc5": {"sub_idx": 0, "label": "SC5\n(Y no exch.)"},
    "sc6": {"sub_idx": 0, "label": "SC6\n(None exch.)"},
}
SCENARIO_ORDER = list(SCENARIO_SPECS.keys())
SCENARIO_LABELS = [SCENARIO_SPECS[sc]["label"] for sc in SCENARIO_ORDER]

METHOD_NAMES = ["PSPower", "IW", "Commensurate", "NPE"]

METHOD_COLORS = {
    "PSPower": "#d62728",
    "IW": "#9467bd",
    "Commensurate": "#7f7f7f",
    "NPE": "#8c564b",
}
METHOD_MARKERS = {
    "PSPower": "D",
    "IW": "v",
    "Commensurate": "P",
    "NPE": "X",
}
METHOD_LS = {
    "PSPower": ":",
    "IW": (0, (3, 1, 1, 1)),
    "Commensurate": "-",
    "NPE": "-",
}


def build_results_dir(base_dir: str, theta_true: float) -> str:
    suffix = "null" if np.isclose(theta_true, 0.0) else f"theta_{str(theta_true).replace('.', 'p')}"
    out_dir = os.path.join(base_dir, f"results_main_no_ot_{suffix}")
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


def run_simulation(B: int, theta_true: float, npe_instance, results_dir: str):
    rng_global = np.random.default_rng(SEED)
    records = []
    total_cells = len(NC_VALUES) * len(SCENARIO_SPECS) * B
    done = 0
    active_methods = METHOD_NAMES[:]
    if npe_instance is None:
        active_methods = [m for m in active_methods if m != "NPE"]

    for nc in NC_VALUES:
        for sc_id, sc_spec in SCENARIO_SPECS.items():
            sub_idx = sc_spec["sub_idx"]
            rep_seeds = rng_global.integers(0, 2**31, size=B)
            errors = {m: 0 for m in active_methods}

            for rep in range(B):
                seed = int(rep_seeds[rep])
                np.random.seed(seed)
                config = SimulationConfig(
                    n_concurrent=nc,
                    n_external=NE,
                    theta_true=theta_true,
                    random_seed=seed,
                )
                scenario = SimulationScenario(sc_id, config)
                data = scenario.generate_data(sub_scenario_idx=sub_idx)
                methods = _build_methods(npe_instance)

                for mname in active_methods:
                    try:
                        result = _run_method(mname, methods[mname], data)
                        records.append({
                            "Nc": nc,
                            "scenario": sc_id,
                            "rep": rep,
                            "method": mname,
                            "theta_hat": result["mean"],
                            "ci_low": result["ci_low"],
                            "ci_high": result["ci_high"],
                            "theta_true": theta_true,
                        })
                    except Exception:
                        errors[mname] += 1
                        records.append({
                            "Nc": nc,
                            "scenario": sc_id,
                            "rep": rep,
                            "method": mname,
                            "theta_hat": np.nan,
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                            "theta_true": theta_true,
                        })

                done += 1
                if done % 100 == 0 or done == total_cells:
                    pct = 100.0 * done / total_cells
                    print(f"  {done}/{total_cells} ({pct:.1f}%)  Nc={nc} {sc_id} rep={rep + 1}")
                    sys.stdout.flush()

            for mname in active_methods:
                if errors[mname] > 0:
                    print(f"    [WARN] {mname} errors in Nc={nc} {sc_id}: {errors[mname]}/{B}")

    df = pd.DataFrame(records)
    out = os.path.join(results_dir, "simulation_results.csv")
    df.to_csv(out, index=False)
    print(f"\nRaw results saved to {out}")
    return df


def compute_summary(df: pd.DataFrame, theta_true: float, results_dir: str) -> pd.DataFrame:
    rows = []
    for (nc, sc_id, method), grp in df.groupby(["Nc", "scenario", "method"]):
        grp = grp.dropna(subset=["theta_hat", "ci_low", "ci_high"]).copy()
        if grp.empty:
            continue

        th = grp["theta_hat"].to_numpy()
        ci_l = grp["ci_low"].to_numpy()
        ci_h = grp["ci_high"].to_numpy()
        covered = (ci_l <= theta_true) & (theta_true <= ci_h)
        reject_null = (ci_l > 0.0) | (ci_h < 0.0)

        bias = float(np.mean(th) - theta_true)
        rmse = float(np.sqrt(np.mean((th - theta_true) ** 2)))
        coverage = float(np.mean(covered))
        reject_rate = float(np.mean(reject_null))
        power = float(reject_rate) if not np.isclose(theta_true, 0.0) else np.nan
        type1 = float(reject_rate) if np.isclose(theta_true, 0.0) else np.nan

        rows.append({
            "Nc": nc,
            "scenario": sc_id,
            "method": method,
            "Bias": round(bias, 4),
            "RMSE": round(rmse, 4),
            "Coverage": round(coverage, 4),
            "RejectNull": round(reject_rate, 4),
            "Power": round(power, 4) if not np.isnan(power) else np.nan,
            "TypeIError": round(type1, 4) if not np.isnan(type1) else np.nan,
            "n_valid": len(grp),
        })

    summary = pd.DataFrame(rows)
    out = os.path.join(results_dir, "simulation_summary.csv")
    summary.to_csv(out, index=False)
    print(f"Summary saved to {out}")
    return summary


def _setup_fig():
    fig, axes = plt.subplots(1, 6, figsize=(14, 3.8), sharey=False, gridspec_kw={"wspace": 0.35})
    return fig, axes


def _plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, out_path: str, hline=None, ylim=None):
    fig, axes = _setup_fig()
    for ax, sc_id, sc_label in zip(axes, SCENARIO_ORDER, SCENARIO_LABELS):
        for mname in METHOD_NAMES:
            sub = summary[(summary["scenario"] == sc_id) & (summary["method"] == mname)]
            if sub.empty:
                continue
            sub = sub.sort_values("Nc")
            ax.plot(
                sub["Nc"], sub[metric],
                color=METHOD_COLORS[mname],
                linestyle=METHOD_LS[mname],
                marker=METHOD_MARKERS[mname],
                markersize=5,
                linewidth=2.2 if mname == "NPE" else 1.5,
                label=mname,
            )
        if hline is not None:
            ax.axhline(hline, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(sc_label, fontsize=8)
        ax.set_xlabel("$N_c$", fontsize=8)
        ax.set_xticks(NC_VALUES)
        ax.tick_params(labelsize=7)
        if ylim is not None:
            ax.set_ylim(ylim)

    axes[0].set_ylabel(ylabel, fontsize=9)
    handles = [
        Line2D(
            [0], [0],
            color=METHOD_COLORS[m],
            linestyle=METHOD_LS[m],
            marker=METHOD_MARKERS[m],
            markersize=5,
            linewidth=2.2 if m == "NPE" else 1.5,
            label=m,
        )
        for m in METHOD_NAMES
    ]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=True)
    fig.tight_layout(rect=[0, 0, 0.87, 1.0])
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def make_figures(summary: pd.DataFrame, theta_true: float, results_dir: str):
    bias_max = max(summary["Bias"].abs().max(), 0.2)
    rmse_max = max(summary["RMSE"].max(), 0.3)

    _plot_metric(summary, "Bias", "Bias", os.path.join(results_dir, "fig_bias.pdf"), hline=0.0,
                 ylim=(-bias_max * 1.15, bias_max * 1.15))
    _plot_metric(summary, "RMSE", "RMSE", os.path.join(results_dir, "fig_rmse.pdf"),
                 ylim=(0.0, rmse_max * 1.15))
    _plot_metric(summary, "Coverage", "Coverage", os.path.join(results_dir, "fig_coverage.pdf"),
                 hline=0.95, ylim=(0.0, 1.05))

    if np.isclose(theta_true, 0.0):
        _plot_metric(summary, "TypeIError", "Type I Error", os.path.join(results_dir, "fig_type1_error.pdf"),
                     hline=0.05, ylim=(0.0, 1.05))
    else:
        _plot_metric(summary, "Power", "Power", os.path.join(results_dir, "fig_power.pdf"),
                     ylim=(0.0, 1.05))


def print_summary_table(summary: pd.DataFrame, theta_true: float):
    print("\n" + "=" * 90)
    if np.isclose(theta_true, 0.0):
        print("SIMULATION SUMMARY  (Bias / RMSE / Coverage / Type I Error)")
    else:
        print("SIMULATION SUMMARY  (Bias / RMSE / Coverage / Power)")
    print("=" * 90)

    for sc_id in SCENARIO_ORDER:
        print(f"\n--- {sc_id} ({SCENARIO_SPECS[sc_id]['label'].replace(chr(10), ' ')}) ---")
        for nc in NC_VALUES:
            sub = summary[(summary["scenario"] == sc_id) & (summary["Nc"] == nc)]
            if sub.empty:
                continue
            sub = sub.set_index("method")
            if nc == NC_VALUES[0]:
                print(f"         {'  '.join(f'{m:>12}' for m in METHOD_NAMES)}")
            vals_b = "  ".join(f"{sub.at[m, 'Bias']:12.3f}" if m in sub.index else f"{np.nan:12.3f}" for m in METHOD_NAMES)
            vals_r = "  ".join(f"{sub.at[m, 'RMSE']:12.3f}" if m in sub.index else f"{np.nan:12.3f}" for m in METHOD_NAMES)
            vals_c = "  ".join(f"{sub.at[m, 'Coverage']:12.3f}" if m in sub.index else f"{np.nan:12.3f}" for m in METHOD_NAMES)
            metric_col = "TypeIError" if np.isclose(theta_true, 0.0) else "Power"
            vals_last = "  ".join(f"{sub.at[m, metric_col]:12.3f}" if m in sub.index else f"{np.nan:12.3f}" for m in METHOD_NAMES)
            print(f"  Nc={nc}  Bias      {vals_b}")
            print(f"         RMSE      {vals_r}")
            print(f"         Coverage  {vals_c}")
            print(f"         {metric_col:<9}{vals_last}")


def main():
    parser = argparse.ArgumentParser(description="Main-text simulation study without OT methods.")
    parser.add_argument("--pretrain", action="store_true", help="Force re-training NPE even if saved model exists.")
    parser.add_argument("--load", action="store_true", help="Load NPE from disk; error if missing.")
    parser.add_argument("--npe-path", default=None, metavar="PATH",
                        help="Path to NPE .pt file (default: results_main_no_ot_*/npe_pretrained.pt).")
    parser.add_argument("--n-sims", type=int, default=5000, metavar="N",
                        help="Number of training simulations for NPE pretraining.")
    parser.add_argument("--n-epochs", type=int, default=100, metavar="N",
                        help="Number of NPE training epochs.")
    parser.add_argument("--b-reps", type=int, default=50, metavar="B",
                        help="Number of Monte Carlo replications.")
    parser.add_argument("--theta-true", type=float, default=1.0,
                        help="True theta used in the simulation. Use 0.0 for Type I error runs.")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    results_dir = build_results_dir(base_dir, args.theta_true)
    model_dir = build_model_dir(base_dir)
    npe_path = args.npe_path or os.path.join(model_dir, "npe_pretrained.pt")

    print("=" * 72)
    print("MAIN-TEXT MONTE CARLO SIMULATION STUDY (NO OT)")
    print(f"B={args.b_reps} replications, Nc in {NC_VALUES}, Ne={NE}, theta_true={args.theta_true}")
    print(f"Scenarios: {list(SCENARIO_SPECS.keys())}")
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

    seed_train = SEED + 1
    train_cfg = {
        "n_simulations": args.n_sims,
        "n_epochs": args.n_epochs,
        "batch_size": 32,
        "random_seed": seed_train,
        "verbose": True,
    }

    if _should_train(npe_path):
        print("\nPre-training NPE...")
        npe = NPEMethod()
        npe.train(**train_cfg)
        npe.save(npe_path)
        print(f"NPE saved to {npe_path}")
    else:
        print(f"\nLoading pre-trained NPE from {npe_path} ...")
        npe = load_npe(npe_path)

    df = run_simulation(B=args.b_reps, theta_true=args.theta_true, npe_instance=npe, results_dir=results_dir)
    summary = compute_summary(df, theta_true=args.theta_true, results_dir=results_dir)
    print_summary_table(summary, theta_true=args.theta_true)
    make_figures(summary, theta_true=args.theta_true, results_dir=results_dir)

    print(f"\nDone. Outputs in {results_dir}")


if __name__ == "__main__":
    main()
