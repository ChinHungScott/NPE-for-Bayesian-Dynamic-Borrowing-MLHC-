"""
Monte Carlo Simulation Study for Bayesian Dynamic Borrowing Methods
====================================================================

B=200 replications across 6 scenarios and Nc in {50, 100}.

Metrics reported:
  - Bias  = mean(theta_hat) - theta_true
  - RMSE  = sqrt(mean((theta_hat - theta_true)^2))
  - Power = fraction of 95% CIs that exclude 0

Methods:
  IPW          PropensityScoreMethod (causallib)
  Overlap      OverlapWeightPSMethod
  FH           PowerPriorMethod (a0=1, full-history)
  PSPower      PSPowerMethod
  IW           IndividualizedWeightsMethod
  Commensurate CommensurateMethod
  IOP          IndividualOverlapPriorMethod  (Li & Xu 2022)
  OT-IOP       OTIOPMethod  [NEW — OT Monge-map weights]
  NPE          NPEMethod    [NEW — trained once offline]
  OT-NPE       OTNPEMethod  [NEW — NPE + Wasserstein features, trained once]

Outputs:
  results/simulation_results.csv   -- raw per-rep results
  results/simulation_summary.csv   -- Bias / RMSE / Power table
  results/fig_bias.pdf
  results/fig_rmse.pdf
  results/fig_power.pdf
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Allow running from project root or scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Suppress sklearn and causallib chatter
warnings.filterwarnings("ignore")

from methods.simulation import SimulationConfig, SimulationScenario
from methods.baselines import (
    PropensityScoreMethod,
    PowerPriorMethod,
    CommensurateMethod,
    OverlapWeightPSMethod,
    IndividualOverlapPriorMethod,
    PSPowerMethod,
    IndividualizedWeightsMethod,
)
from methods.npe import NPEMethod, OTNPEMethod, KLNPEMethod, load_npe

# ============================================================
# Configuration
# ============================================================
SEED = 20260306
B = 50
NC_VALUES = [50, 100]
NE = 200
THETA_TRUE = 1.0

# Scenarios to run; sub_idx=1 for sc2/sc3 to activate the non-exchangeable sub-case
SCENARIO_SPECS = {
    "sc1": {"sub_idx": 0, "label": "SC1\n(All exch.)"},
    "sc2": {"sub_idx": 1, "label": "SC2\n(X partial)"},
    "sc3": {"sub_idx": 1, "label": "SC3\n(Y partial)"},
    "sc4": {"sub_idx": 0, "label": "SC4\n(X no exch.)"},
    "sc5": {"sub_idx": 0, "label": "SC5\n(Y no exch.)"},
    "sc6": {"sub_idx": 0, "label": "SC6\n(None exch.)"},
}

METHOD_NAMES = ["PSPower", "IW", "Commensurate",
 "NPE", "OT-NPE", "KL-NPE",
]

# NeurIPS-style palette
METHOD_COLORS = {
    "IPW":          "#1f77b4",   # blue
    "Overlap":      "#ff7f0e",   # orange
    "FH":           "#2ca02c",   # green
    "PSPower":      "#d62728",   # red
    "IW":           "#9467bd",   # purple
    "Commensurate": "#7f7f7f",   # gray
    "IOP":          "#e377c2",   # pink
    "NPE":          "#8c564b",   # brown  [new]
    "OT-NPE":       "#17becf",   # cyan   [new]
    "KL-NPE":       "#bcbd22",   # olive  [new]
}
METHOD_MARKERS = {
    "IPW": "o", "Overlap": "s", "FH": "^", "PSPower": "D",
    "IW": "v", "Commensurate": "P", "IOP": "*",
    "NPE": "X", "OT-NPE": "p", "KL-NPE": "h",
}
METHOD_LS = {
    "IPW": "-", "Overlap": "--", "FH": "-.",
    "PSPower": ":", "IW": (0, (3, 1, 1, 1)), "Commensurate": "-", "IOP": "--",
    "NPE": "-", "OT-NPE": "--", "KL-NPE": "-.",
}
# Bold lines for the two new proposed methods
NEW_METHODS = {"NPE", "OT-NPE", "KL-NPE"}
IOP_LINEWIDTH = 2.5

# Results always written to project-root/results/ regardless of working directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Method factory (returns a fresh instance each rep;
# pre-trained NPE instances are reused across all reps)
# ============================================================

def _build_methods(npe_instance=None, otnpe_instance=None, klnpe_instance=None):
    """Build method dict.  NPE/OT-NPE are pre-trained singletons passed in."""
    methods = {
        "IPW":          PropensityScoreMethod(method="weighting"),
        "Overlap":      OverlapWeightPSMethod(),
        "FH":           PowerPriorMethod(a0=1.0),
        "PSPower":      PSPowerMethod(K=5),
        "IW":           IndividualizedWeightsMethod(),
        "Commensurate": CommensurateMethod(),
        "IOP":          IndividualOverlapPriorMethod(),
    }
    if npe_instance is not None:
        methods["NPE"] = npe_instance
    if otnpe_instance is not None:
        methods["OT-NPE"] = otnpe_instance
    if klnpe_instance is not None:
        methods["KL-NPE"] = klnpe_instance
    return methods


def _run_method(name, method, data):
    """Call estimate_treatment_effect with MCMC fast-settings for Commensurate."""
    Xc = data["X_concurrent"]
    Yc = data["Y_concurrent"]
    Xe = data["X_external"]
    Ye = data["Y_external"]
    if name == "Commensurate":
        return method.estimate_treatment_effect(
            Xc, Yc, Xe, Ye, n_samples=2000, n_burnin=500
        )
    elif name == "IPW":
        return method.estimate_treatment_effect(
            Xc, Yc, Xe, Ye, n_bootstrap=100   # faster bootstrap
        )
    else:
        return method.estimate_treatment_effect(Xc, Yc, Xe, Ye)


# ============================================================
# Main simulation loop
# ============================================================

def run_simulation(npe_instance=None, otnpe_instance=None, klnpe_instance=None):
    rng_global = np.random.default_rng(SEED)

    records = []

    total_cells = len(NC_VALUES) * len(SCENARIO_SPECS) * B
    done = 0

    active_methods = METHOD_NAMES[:]
    if npe_instance is None:
        active_methods = [m for m in active_methods if m != "NPE"]
    if otnpe_instance is None:
        active_methods = [m for m in active_methods if m != "OT-NPE"]

    for nc in NC_VALUES:
        for sc_id, sc_spec in SCENARIO_SPECS.items():
            sub_idx = sc_spec["sub_idx"]

            rep_seeds = rng_global.integers(0, 2**31, size=B)

            theta_hats = {m: [] for m in active_methods}
            ci_lows    = {m: [] for m in active_methods}
            ci_highs   = {m: [] for m in active_methods}
            errors     = {m: 0  for m in active_methods}

            for rep in range(B):
                seed = int(rep_seeds[rep])
                np.random.seed(seed)
                config = SimulationConfig(
                    n_concurrent=nc,
                    n_external=NE,
                    theta_true=THETA_TRUE,
                    random_seed=seed,
                )
                scenario = SimulationScenario(sc_id, config)
                data = scenario.generate_data(sub_scenario_idx=sub_idx)

                methods = _build_methods(npe_instance, otnpe_instance, klnpe_instance)
                for mname in active_methods:
                    try:
                        result = _run_method(mname, methods[mname], data)
                        theta_hats[mname].append(result["mean"])
                        ci_lows[mname].append(result["ci_low"])
                        ci_highs[mname].append(result["ci_high"])
                    except Exception as exc:
                        errors[mname] += 1
                        theta_hats[mname].append(np.nan)
                        ci_lows[mname].append(np.nan)
                        ci_highs[mname].append(np.nan)

                done += 1
                if done % 100 == 0 or done == total_cells:
                    pct = 100.0 * done / total_cells
                    print(f"  {done}/{total_cells} ({pct:.1f}%)  Nc={nc} {sc_id} rep={rep+1}")
                    sys.stdout.flush()

            # Store raw records
            for rep in range(B):
                for mname in active_methods:
                    records.append({
                        "Nc": nc,
                        "scenario": sc_id,
                        "rep": rep,
                        "method": mname,
                        "theta_hat": theta_hats[mname][rep],
                        "ci_low": ci_lows[mname][rep],
                        "ci_high": ci_highs[mname][rep],
                        "theta_true": THETA_TRUE,
                    })

            # Print error counts
            for mname in active_methods:
                if errors[mname] > 0:
                    print(f"    [WARN] {mname} errors in Nc={nc} {sc_id}: {errors[mname]}/{B}")

    df = pd.DataFrame(records)
    out = os.path.join(RESULTS_DIR, "simulation_results.csv")
    df.to_csv(out, index=False)
    print(f"\nRaw results saved to {out}")
    return df


# ============================================================
# Summary statistics
# ============================================================

def compute_summary(df):
    rows = []
    for (nc, sc_id, method), grp in df.groupby(["Nc", "scenario", "method"]):
        th = grp["theta_hat"].dropna().values
        ci_l = grp["ci_low"].dropna().values
        ci_h = grp["ci_high"].dropna().values
        theta_true = grp["theta_true"].iloc[0]

        if len(th) == 0:
            continue

        bias  = float(np.mean(th) - theta_true)
        rmse  = float(np.sqrt(np.mean((th - theta_true) ** 2)))
        # Power = fraction of CIs that exclude 0
        valid = min(len(ci_l), len(ci_h))
        power = float(np.mean((ci_l[:valid] > 0) | (ci_h[:valid] < 0))) if valid > 0 else np.nan

        rows.append({
            "Nc": nc, "scenario": sc_id, "method": method,
            "Bias": round(bias, 4), "RMSE": round(rmse, 4),
            "Power": round(power, 4), "n_valid": len(th),
        })

    summary = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "simulation_summary.csv")
    summary.to_csv(out, index=False)
    print(f"Summary saved to {out}")
    return summary


# ============================================================
# Figures
# ============================================================

SCENARIO_ORDER = list(SCENARIO_SPECS.keys())
SCENARIO_LABELS = [SCENARIO_SPECS[sc]["label"] for sc in SCENARIO_ORDER]


def _setup_fig():
    fig, axes = plt.subplots(
        1, 6, figsize=(14, 3.8), sharey=False,
        gridspec_kw={"wspace": 0.35}
    )
    return fig, axes


def _plot_metric(summary, metric, ylabel, out_path, hline=None, ylim=None):
    fig, axes = _setup_fig()

    for ax, sc_id, sc_label in zip(axes, SCENARIO_ORDER, SCENARIO_LABELS):
        for mname in METHOD_NAMES:
            sub = summary[(summary["scenario"] == sc_id) & (summary["method"] == mname)]
            if sub.empty:
                continue
            sub_sorted = sub.sort_values("Nc")
            lw = IOP_LINEWIDTH if mname in NEW_METHODS or mname == "IOP" else 1.5
            ax.plot(
                sub_sorted["Nc"], sub_sorted[metric],
                color=METHOD_COLORS[mname],
                linestyle=METHOD_LS[mname],
                marker=METHOD_MARKERS[mname],
                markersize=5, linewidth=lw,
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

    # Single legend outside right
    handles = [
        Line2D([0], [0],
               color=METHOD_COLORS[m],
               linestyle=METHOD_LS[m],
               marker=METHOD_MARKERS[m],
               markersize=5,
               linewidth=IOP_LINEWIDTH if m in NEW_METHODS or m == "IOP" else 1.5,
               label=m)
        for m in METHOD_NAMES
    ]
    fig.legend(
        handles=handles,
        loc="center right", bbox_to_anchor=(1.0, 0.5),
        fontsize=7, frameon=True, framealpha=0.9,
        title="Method", title_fontsize=8,
    )
    fig.tight_layout(rect=[0, 0, 0.87, 1.0])
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def make_figures(summary):
    # Determine sensible y-limits
    bias_max = summary["Bias"].abs().max()
    rmse_max = summary["RMSE"].max()

    _plot_metric(
        summary, "Bias", "Bias",
        os.path.join(RESULTS_DIR, "fig_bias.pdf"),
        hline=0.0,
        ylim=(-max(bias_max * 1.15, 0.2), max(bias_max * 1.15, 0.2)),
    )
    _plot_metric(
        summary, "RMSE", "RMSE",
        os.path.join(RESULTS_DIR, "fig_rmse.pdf"),
        hline=None,
        ylim=(0, max(rmse_max * 1.15, 0.3)),
    )
    _plot_metric(
        summary, "Power", "Power (fraction CI excludes 0)",
        os.path.join(RESULTS_DIR, "fig_power.pdf"),
        hline=None,
        ylim=(0.0, 1.05),
    )


# ============================================================
# Print summary table
# ============================================================

def print_summary_table(summary):
    print("\n" + "=" * 80)
    print("SIMULATION SUMMARY  (Bias / RMSE / Power)")
    print("=" * 80)
    pivot_bias  = summary.pivot_table("Bias",  index=["scenario","Nc"], columns="method")
    pivot_rmse  = summary.pivot_table("RMSE",  index=["scenario","Nc"], columns="method")
    pivot_power = summary.pivot_table("Power", index=["scenario","Nc"], columns="method")

    for sc_id in SCENARIO_ORDER:
        print(f"\n--- {sc_id} ({SCENARIO_SPECS[sc_id]['label'].replace(chr(10),' ')}) ---")
        for nc in NC_VALUES:
            idx = (sc_id, nc)
            if idx not in pivot_bias.index:
                continue
            row_b = pivot_bias.loc[idx]
            row_r = pivot_rmse.loc[idx]
            row_p = pivot_power.loc[idx]
            header = f"  Nc={nc:3d}  {'Method':<12}"
            vals_b = "  ".join(f"{row_b.get(m,np.nan):6.3f}" for m in METHOD_NAMES)
            vals_r = "  ".join(f"{row_r.get(m,np.nan):6.3f}" for m in METHOD_NAMES)
            vals_p = "  ".join(f"{row_p.get(m,np.nan):6.3f}" for m in METHOD_NAMES)
            if nc == NC_VALUES[0]:
                print(f"         {'  '.join(f'{m:>6}' for m in METHOD_NAMES)}")
            print(f"  Nc={nc}  Bias   {vals_b}")
            print(f"         RMSE   {vals_r}")
            print(f"         Power  {vals_p}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monte Carlo Simulation Study — Bayesian Dynamic Borrowing"
    )
    npe_group = parser.add_mutually_exclusive_group()
    npe_group.add_argument(
        "--pretrain", action="store_true",
        help="Force re-training NPE and OT-NPE even if saved models exist."
    )
    npe_group.add_argument(
        "--load", action="store_true",
        help="Load NPE and OT-NPE from disk; error if files are missing."
    )
    parser.add_argument(
        "--npe-path", default=None, metavar="PATH",
        help="Path to NPE .pt file (default: results/npe_pretrained.pt)."
    )
    parser.add_argument(
        "--otnpe-path", default=None, metavar="PATH",
        help="Path to OT-NPE .pt file (default: results/otnpe_pretrained.pt)."
    )
    parser.add_argument(
        "--klnpe-path", default=None, metavar="PATH",
        help="Path to KL-NPE .pt file (default: results/klnpe_pretrained.pt)."
    )
    parser.add_argument(
        "--n-sims", type=int, default=5000, metavar="N",
        help="Number of training simulations (default: 5000)."
    )
    parser.add_argument(
        "--n-epochs", type=int, default=100, metavar="N",
        help="Training epochs (default: 100)."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Monte Carlo Simulation Study — Bayesian Dynamic Borrowing")
    print(f"B={B} replications, Nc in {NC_VALUES}, Ne={NE}")
    print(f"Scenarios: {list(SCENARIO_SPECS.keys())}")
    print(f"Methods: {METHOD_NAMES}")
    print("=" * 70)

    # ----------------------------------------------------------
    # Pre-train NPE and OT-NPE once (amortized — reused per rep).
    # Training seed is deliberately different from benchmarking SEED to prevent
    # overlap between the simulated training corpus and evaluation reps.
    # ----------------------------------------------------------
    SEED_TRAIN = SEED + 1   # 20260307
    NPE_TRAIN_CONFIG = {
        "n_simulations": args.n_sims,
        "n_epochs": args.n_epochs,
        "batch_size": 32,
        "random_seed": SEED_TRAIN,
        "verbose": True,
    }
    NPE_SAVE_PATH   = args.npe_path   or os.path.join(RESULTS_DIR, "npe_pretrained.pt")
    OTNPE_SAVE_PATH = args.otnpe_path or os.path.join(RESULTS_DIR, "otnpe_pretrained.pt")
    KLNPE_SAVE_PATH = args.klnpe_path or os.path.join(RESULTS_DIR, "klnpe_pretrained.pt")

    # Determine whether to train or load for each model
    def _should_train(path):
        if args.pretrain:
            return True
        if args.load:
            if not os.path.exists(path):
                raise FileNotFoundError(f"--load specified but model not found: {path}")
            return False
        # Default: load if exists, train otherwise
        return not os.path.exists(path)

    if _should_train(NPE_SAVE_PATH):
        print("\nPre-training NPE...")
        npe = NPEMethod()
        npe.train(**NPE_TRAIN_CONFIG)
        npe.save(NPE_SAVE_PATH)
        print(f"NPE saved → {NPE_SAVE_PATH}")
    else:
        print(f"\nLoading pre-trained NPE from {NPE_SAVE_PATH} ...")
        npe = load_npe(NPE_SAVE_PATH)

    if _should_train(OTNPE_SAVE_PATH):
        print("\nPre-training OT-NPE...")
        otnpe = OTNPEMethod(epsilon=0.05)
        otnpe.train(**NPE_TRAIN_CONFIG)
        otnpe.save(OTNPE_SAVE_PATH)
        print(f"OT-NPE saved → {OTNPE_SAVE_PATH}")
    else:
        print(f"\nLoading pre-trained OT-NPE from {OTNPE_SAVE_PATH} ...")
        otnpe = load_npe(OTNPE_SAVE_PATH)
    if _should_train(KLNPE_SAVE_PATH):
        print("\nPre-training KL-NPE...")
        klnpe = KLNPEMethod(ridge=1e-3)
        klnpe.train(**NPE_TRAIN_CONFIG)
        klnpe.save(KLNPE_SAVE_PATH)
        print(f"KL-NPE saved → {KLNPE_SAVE_PATH}")
    else:
        print(f"\nLoading pre-trained KL-NPE from {KLNPE_SAVE_PATH} ...")
        klnpe = load_npe(KLNPE_SAVE_PATH)



    # ----------------------------------------------------------
    # Monte Carlo loop
    # ----------------------------------------------------------
    df = run_simulation(npe_instance=npe, otnpe_instance=otnpe, klnpe_instance=klnpe)
    summary = compute_summary(df)
    print_summary_table(summary)
    make_figures(summary)

    print("\nDone. Outputs in results/")
