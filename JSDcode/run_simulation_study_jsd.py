"""
Standalone Monte Carlo Simulation Study for Bayesian Dynamic Borrowing Methods
==============================================================================

Upload these files into the SAME folder in Colab:
  - baselines.py
  - simulation.py
  - npe_jsd.py
  - run_simulation_study_jsd.py

Outputs:
  results_jsd/simulation_results_jsd.csv
  results_jsd/simulation_summary_jsd.csv
  results_jsd/fig_bias_jsd.pdf
  results_jsd/fig_rmse_jsd.pdf
  results_jsd/fig_power_jsd.pdf
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

from simulation import SimulationConfig, SimulationScenario
from baselines import (
    PropensityScoreMethod,
    PowerPriorMethod,
    CommensurateMethod,
    OverlapWeightPSMethod,
    IndividualOverlapPriorMethod,
    PSPowerMethod,
    IndividualizedWeightsMethod,
)
from npe_jsd import (
    NPEMethod,
    OTNPEMethod,
    KLNPEMethod,
    KLKSDistNPEMethod,
    JSDNPEMethod,
    load_npe,
)

# ============================================================
# Configuration
# ============================================================
SEED = 20260316
B = 50
NC_VALUES = [50, 100]
NE = 200
THETA_TRUE = 1.0

SCENARIO_SPECS = {
    "sc1": {"sub_idx": 0, "label": "SC1\n(All exch.)"},
    "sc2": {"sub_idx": 1, "label": "SC2\n(X partial)"},
    "sc3": {"sub_idx": 1, "label": "SC3\n(Y partial)"},
    "sc4": {"sub_idx": 0, "label": "SC4\n(X no exch.)"},
    "sc5": {"sub_idx": 0, "label": "SC5\n(Y no exch.)"},
    "sc6": {"sub_idx": 0, "label": "SC6\n(None exch.)"},
}

METHOD_NAMES = [
    "PSPower", "IW", "Commensurate",
    "NPE", "OT-NPE", "KL-NPE", "KL+KS-NPE", "JSD-NPE",
]

METHOD_COLORS = {
    "IPW":          "#1f77b4",
    "Overlap":      "#ff7f0e",
    "FH":           "#2ca02c",
    "PSPower":      "#d62728",
    "IW":           "#9467bd",
    "Commensurate": "#7f7f7f",
    "IOP":          "#e377c2",
    "NPE":          "#8c564b",
    "OT-NPE":       "#17becf",
    "KL-NPE":       "#bcbd22",
    "KL+KS-NPE":    "#e377c2",
    "JSD-NPE":      "#1b9e77",
}
METHOD_MARKERS = {
    "IPW": "o", "Overlap": "s", "FH": "^", "PSPower": "D",
    "IW": "v", "Commensurate": "P", "IOP": "*",
    "NPE": "X", "OT-NPE": "p", "KL-NPE": "h", "KL+KS-NPE": "s", "JSD-NPE": "8",
}
METHOD_LS = {
    "IPW": "-", "Overlap": "--", "FH": "-.",
    "PSPower": ":", "IW": (0, (3, 1, 1, 1)), "Commensurate": "-", "IOP": "--",
    "NPE": "-", "OT-NPE": "--", "KL-NPE": "-.", "KL+KS-NPE": (0, (5, 2)), "JSD-NPE": (0, (2, 2)),
}
NEW_METHODS = {"NPE", "OT-NPE", "KL-NPE", "KL+KS-NPE", "JSD-NPE"}
BASELINE_LINEWIDTH = 1.5
NEW_LINEWIDTH = 2.5

RESULTS_DIR = "results_jsd"
os.makedirs(RESULTS_DIR, exist_ok=True)


def _build_methods(npe_instance=None, otnpe_instance=None, klnpe_instance=None,
                   klksnpe_instance=None, jsdnpe_instance=None):
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
    if klksnpe_instance is not None:
        methods["KL+KS-NPE"] = klksnpe_instance
    if jsdnpe_instance is not None:
        methods["JSD-NPE"] = jsdnpe_instance
    return methods


def _run_method(name, method, data):
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
            Xc, Yc, Xe, Ye, n_bootstrap=100
        )
    else:
        return method.estimate_treatment_effect(Xc, Yc, Xe, Ye)


def run_simulation(npe_instance=None, otnpe_instance=None, klnpe_instance=None,
                   klksnpe_instance=None, jsdnpe_instance=None):
    rng_global = np.random.default_rng(SEED)
    records = []
    total_cells = len(NC_VALUES) * len(SCENARIO_SPECS) * B
    done = 0

    active_methods = METHOD_NAMES[:]
    if npe_instance is None:
        active_methods.remove("NPE")
    if otnpe_instance is None:
        active_methods.remove("OT-NPE")
    if klnpe_instance is None:
        active_methods.remove("KL-NPE")
    if klksnpe_instance is None:
        active_methods.remove("KL+KS-NPE")
    if jsdnpe_instance is None:
        active_methods.remove("JSD-NPE")

    for nc in NC_VALUES:
        for sc_id, sc_spec in SCENARIO_SPECS.items():
            sub_idx = sc_spec["sub_idx"]
            rep_seeds = rng_global.integers(0, 2**31, size=B)

            theta_hats = {m: [] for m in active_methods}
            ci_lows = {m: [] for m in active_methods}
            ci_highs = {m: [] for m in active_methods}
            errors = {m: 0 for m in active_methods}

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

                methods = _build_methods(
                    npe_instance=npe_instance,
                    otnpe_instance=otnpe_instance,
                    klnpe_instance=klnpe_instance,
                    klksnpe_instance=klksnpe_instance,
                    jsdnpe_instance=jsdnpe_instance,
                )

                for mname in active_methods:
                    try:
                        result = _run_method(mname, methods[mname], data)
                        theta_hats[mname].append(result["mean"])
                        ci_lows[mname].append(result["ci_low"])
                        ci_highs[mname].append(result["ci_high"])
                    except Exception:
                        errors[mname] += 1
                        theta_hats[mname].append(np.nan)
                        ci_lows[mname].append(np.nan)
                        ci_highs[mname].append(np.nan)

                done += 1
                if done % 100 == 0 or done == total_cells:
                    pct = 100.0 * done / total_cells
                    print(f"  {done}/{total_cells} ({pct:.1f}%)  Nc={nc} {sc_id} rep={rep+1}", flush=True)

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

            for mname in active_methods:
                if errors[mname] > 0:
                    print(f"    [WARN] {mname} errors in Nc={nc} {sc_id}: {errors[mname]}/{B}")

    df = pd.DataFrame(records)
    out = os.path.join(RESULTS_DIR, "simulation_results_jsd.csv")
    df.to_csv(out, index=False)
    print(f"\nRaw results saved to {out}")
    return df


def compute_summary(df):
    rows = []
    for (nc, sc_id, method), grp in df.groupby(["Nc", "scenario", "method"]):
        th = grp["theta_hat"].dropna().values
        ci_l = grp["ci_low"].dropna().values
        ci_h = grp["ci_high"].dropna().values
        theta_true = grp["theta_true"].iloc[0]

        if len(th) == 0:
            continue

        bias = float(np.mean(th) - theta_true)
        rmse = float(np.sqrt(np.mean((th - theta_true) ** 2)))
        valid = min(len(ci_l), len(ci_h))
        power = float(np.mean((ci_l[:valid] > 0) | (ci_h[:valid] < 0))) if valid > 0 else np.nan

        rows.append({
            "Nc": nc,
            "scenario": sc_id,
            "method": method,
            "Bias": round(bias, 4),
            "RMSE": round(rmse, 4),
            "Power": round(power, 4),
            "n_valid": len(th),
        })

    summary = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "simulation_summary_jsd.csv")
    summary.to_csv(out, index=False)
    print(f"Summary saved to {out}")
    return summary


SCENARIO_ORDER = list(SCENARIO_SPECS.keys())
SCENARIO_LABELS = [SCENARIO_SPECS[sc]["label"] for sc in SCENARIO_ORDER]


def _setup_fig():
    fig, axes = plt.subplots(1, 6, figsize=(15, 4), sharey=False, gridspec_kw={"wspace": 0.35})
    return fig, axes


def _plot_metric(summary, metric, ylabel, out_path, hline=None, ylim=None):
    fig, axes = _setup_fig()

    for ax, sc_id, sc_label in zip(axes, SCENARIO_ORDER, SCENARIO_LABELS):
        for mname in METHOD_NAMES:
            sub = summary[(summary["scenario"] == sc_id) & (summary["method"] == mname)]
            if sub.empty:
                continue
            sub_sorted = sub.sort_values("Nc")
            lw = NEW_LINEWIDTH if mname in NEW_METHODS else BASELINE_LINEWIDTH
            ax.plot(
                sub_sorted["Nc"], sub_sorted[metric],
                color=METHOD_COLORS[mname],
                linestyle=METHOD_LS[mname],
                marker=METHOD_MARKERS[mname],
                markersize=5,
                linewidth=lw,
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
        Line2D([0], [0],
               color=METHOD_COLORS[m],
               linestyle=METHOD_LS[m],
               marker=METHOD_MARKERS[m],
               markersize=5,
               linewidth=NEW_LINEWIDTH if m in NEW_METHODS else BASELINE_LINEWIDTH,
               label=m)
        for m in METHOD_NAMES
    ]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.02, 0.5),
               fontsize=7, frameon=True, framealpha=0.9, title="Method", title_fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.86, 1.0])
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def make_figures(summary):
    bias_max = summary["Bias"].abs().max()
    rmse_max = summary["RMSE"].max()

    _plot_metric(
        summary, "Bias", "Bias",
        os.path.join(RESULTS_DIR, "fig_bias_jsd.pdf"),
        hline=0.0,
        ylim=(-max(bias_max * 1.15, 0.2), max(bias_max * 1.15, 0.2)),
    )
    _plot_metric(
        summary, "RMSE", "RMSE",
        os.path.join(RESULTS_DIR, "fig_rmse_jsd.pdf"),
        hline=None,
        ylim=(0.0, max(rmse_max * 1.15, 0.3)),
    )
    _plot_metric(
        summary, "Power", "Power (fraction CI excludes 0)",
        os.path.join(RESULTS_DIR, "fig_power_jsd.pdf"),
        hline=None,
        ylim=(0.0, 1.05),
    )


def print_summary_table(summary):
    print("\n" + "=" * 90)
    print("SIMULATION SUMMARY  (Bias / RMSE / Power)")
    print("=" * 90)
    pivot_bias = summary.pivot_table("Bias", index=["scenario", "Nc"], columns="method")
    pivot_rmse = summary.pivot_table("RMSE", index=["scenario", "Nc"], columns="method")
    pivot_power = summary.pivot_table("Power", index=["scenario", "Nc"], columns="method")

    for sc_id in SCENARIO_ORDER:
        print(f"\n--- {sc_id} ({SCENARIO_SPECS[sc_id]['label'].replace(chr(10), ' ')}) ---")
        for nc in NC_VALUES:
            idx = (sc_id, nc)
            if idx not in pivot_bias.index:
                continue
            row_b = pivot_bias.loc[idx]
            row_r = pivot_rmse.loc[idx]
            row_p = pivot_power.loc[idx]
            if nc == NC_VALUES[0]:
                print("         " + "  ".join(f"{m:>10}" for m in METHOD_NAMES))
            print(f"  Nc={nc:<3} Bias   " + "  ".join(f"{row_b.get(m, np.nan):10.3f}" for m in METHOD_NAMES))
            print("        RMSE   " + "  ".join(f"{row_r.get(m, np.nan):10.3f}" for m in METHOD_NAMES))
            print("        Power  " + "  ".join(f"{row_p.get(m, np.nan):10.3f}" for m in METHOD_NAMES))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone simulation study with JSD-NPE")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pretrain", action="store_true", help="Force retraining all NPE variants.")
    group.add_argument("--load", action="store_true", help="Load saved models; error if missing.")
    parser.add_argument("--n-sims", type=int, default=5000, help="Training simulations per NPE variant.")
    parser.add_argument("--n-epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--npe-path", default=os.path.join(RESULTS_DIR, "npe_pretrained.pt"))
    parser.add_argument("--otnpe-path", default=os.path.join(RESULTS_DIR, "otnpe_pretrained.pt"))
    parser.add_argument("--klnpe-path", default=os.path.join(RESULTS_DIR, "klnpe_pretrained.pt"))
    parser.add_argument("--klksnpe-path", default=os.path.join(RESULTS_DIR, "klksnpe_pretrained.pt"))
    parser.add_argument("--jsdnpe-path", default=os.path.join(RESULTS_DIR, "jsdnpe_pretrained.pt"))
    args = parser.parse_args()

    print("=" * 72)
    print("Standalone Monte Carlo Simulation Study — Bayesian Dynamic Borrowing")
    print(f"B={B} replications, Nc in {NC_VALUES}, Ne={NE}")
    print(f"Scenarios: {list(SCENARIO_SPECS.keys())}")
    print(f"Methods: {METHOD_NAMES}")
    print("=" * 72)

    seed_train = SEED + 1
    train_cfg = {
        "n_simulations": args.n_sims,
        "n_epochs": args.n_epochs,
        "batch_size": 32,
        "random_seed": seed_train,
        "verbose": True,
    }

    def should_train(path):
        if args.pretrain:
            return True
        if args.load:
            if not os.path.exists(path):
                raise FileNotFoundError(f"--load specified but model not found: {path}")
            return False
        return not os.path.exists(path)

    if should_train(args.npe_path):
        print("\nPre-training NPE...")
        npe = NPEMethod()
        npe.train(**train_cfg)
        npe.save(args.npe_path)
    else:
        print(f"\nLoading NPE from {args.npe_path} ...")
        npe = load_npe(args.npe_path)

    if should_train(args.otnpe_path):
        print("\nPre-training OT-NPE...")
        otnpe = OTNPEMethod(epsilon=0.05)
        otnpe.train(**train_cfg)
        otnpe.save(args.otnpe_path)
    else:
        print(f"\nLoading OT-NPE from {args.otnpe_path} ...")
        otnpe = load_npe(args.otnpe_path)

    if should_train(args.klnpe_path):
        print("\nPre-training KL-NPE...")
        klnpe = KLNPEMethod(ridge=1e-3)
        klnpe.train(**train_cfg)
        klnpe.save(args.klnpe_path)
    else:
        print(f"\nLoading KL-NPE from {args.klnpe_path} ...")
        klnpe = load_npe(args.klnpe_path)

    if should_train(args.klksnpe_path):
        print("\nPre-training KL+KS-NPE...")
        klksnpe = KLKSDistNPEMethod(ridge=1e-3, ks_split_ratio=0.7, ks_seed=0)
        klksnpe.train(**train_cfg)
        klksnpe.save(args.klksnpe_path)
    else:
        print(f"\nLoading KL+KS-NPE from {args.klksnpe_path} ...")
        klksnpe = load_npe(args.klksnpe_path)

    if should_train(args.jsdnpe_path):
        print("\nPre-training JSD-NPE...")
        jsdnpe = JSDNPEMethod(ridge=1e-3)
        jsdnpe.train(**train_cfg)
        jsdnpe.save(args.jsdnpe_path)
    else:
        print(f"\nLoading JSD-NPE from {args.jsdnpe_path} ...")
        jsdnpe = load_npe(args.jsdnpe_path)

    df = run_simulation(
        npe_instance=npe,
        otnpe_instance=otnpe,
        klnpe_instance=klnpe,
        klksnpe_instance=klksnpe,
        jsdnpe_instance=jsdnpe,
    )
    summary = compute_summary(df)
    print_summary_table(summary)
    make_figures(summary)

    print("\nDone. Outputs are in results_jsd/")
