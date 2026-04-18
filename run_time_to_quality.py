"""
Time-to-quality benchmark for joint-posterior MCMC methods.

Question answered in the paper:
    "How long does each method take to deliver a *high-quality* joint posterior
     p(theta, phi | D_c, D_e)?"

"High quality" is operationalized as:
    min( ESS(theta), ESS(phi) )  >=  ESS_TARGET   (default: 1000)

where phi is the borrowing-strength parameter of each method:
    - NormalizedPowerPrior (NPP)  -> phi = a0   (discount factor)
    - Commensurate                -> phi = tau  (commensurability precision)

Protocol (per method x scenario x replication):
    1. Run chain of length n_0 (burn-in fixed).
    2. Compute ESS(theta), ESS(phi).
    3. If min-ESS < target, double n_samples and rerun. Repeat.
    4. Record cumulative wall-clock (honest user experience) AND final-run
       wall-clock (chain length needed "in hindsight"). Both reported.

NPE is reported as a reference point: single amortized forward pass
(marginal theta only, no phi — flagged in table).

Outputs under  results/time_to_quality/ :
    time_to_quality_raw.csv       (per-rep records)
    time_to_quality_summary.csv   (method x scenario medians + IQR)
    time_to_quality_table.tex     (paper table)
    time_to_quality_box.pdf       (boxplot of cumulative time)
"""

import argparse
import os
import sys
import time
import warnings
from typing import Callable, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 20260417
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from methods.simulation import SimulationConfig, SimulationScenario
from methods.baselines import NormalizedPowerPriorMethod, CommensurateMethod
from methods.npe import load_npe


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
CONFIG = {
    "ess_target": 1000,
    "n_init": 500,
    "n_max": 32000,
    "burnin": 500,
    "gold_samples": 20000,
    "gold_burnin": 5000,
    "npe_samples": 5000,
    "n_reps": 5,
    "n_concurrent": 100,
    "n_external": 200,
    "theta_true": 1.0,
}

SCENARIO_SPECS = {
    "sc1": {"sub_idx": 0, "label": "SC1 (exchangeable)"},
    "sc3": {"sub_idx": 1, "label": "SC3 (covariate shift)"},
    "sc5": {"sub_idx": 0, "label": "SC5 (outcome drift)"},
    "sc6": {"sub_idx": 0, "label": "SC6 (joint mismatch)"},
}

METHOD_COLORS = {
    "NPP":          "#9467bd",
    "Commensurate": "#d62728",
    "NPE":          "#1f77b4",
}


# ------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------
def autocorr_ess(x: np.ndarray) -> float:
    """Effective sample size via Geyer initial-positive-sequence estimator."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4 or np.std(x) == 0.0:
        return float(n)
    x = x - x.mean()
    f = np.fft.fft(x, n=2 * n)
    acf = np.real(np.fft.ifft(f * np.conj(f)))[:n]
    acf /= acf[0]
    rho_sum = 1.0
    k = 1
    while k + 1 < n:
        pair = acf[k] + acf[k + 1]
        if pair < 0:
            break
        rho_sum += 2.0 * pair
        k += 2
    ess = n / max(rho_sum, 1e-6)
    return float(min(max(ess, 1.0), n))


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """1D Wasserstein-1 via sorted quantile matching."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    m = max(len(a), len(b))
    qa = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, len(a)), a)
    qb = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, len(b)), b)
    return float(np.mean(np.abs(qa - qb)))


# ------------------------------------------------------------------
# Method runners
# ------------------------------------------------------------------
def _data_tuple(data):
    return (data["X_concurrent"], data["Y_concurrent"],
            data["X_external"],   data["Y_external"])


def make_npp_runner(data) -> Callable:
    Xc, Yc, Xe, Ye = _data_tuple(data)
    m = NormalizedPowerPriorMethod()

    def runner(n_samples: int, n_burnin: int) -> Dict:
        res = m.estimate_treatment_effect(Xc, Yc, Xe, Ye,
                                          n_samples=n_samples, n_burnin=n_burnin)
        return {"theta_samples": np.asarray(res["samples"]),
                "phi_samples":   np.asarray(res["a0_samples"]),
                "phi_name": "a0"}
    return runner


def make_commensurate_runner(data) -> Callable:
    Xc, Yc, Xe, Ye = _data_tuple(data)
    m = CommensurateMethod()

    def runner(n_samples: int, n_burnin: int) -> Dict:
        res = m.estimate_treatment_effect(Xc, Yc, Xe, Ye,
                                          n_samples=n_samples, n_burnin=n_burnin)
        return {"theta_samples": np.asarray(res["samples"]),
                "phi_samples":   np.asarray(res["tau_samples"]),
                "phi_name": "tau"}
    return runner


# ------------------------------------------------------------------
# Doubling loop
# ------------------------------------------------------------------
def time_to_quality(
    runner: Callable,
    ess_target: int,
    n_init: int,
    n_max: int,
    burnin: int,
) -> Dict:
    """Run the chain, doubling n_samples until min(ESS_theta, ESS_phi) >= target.

    Returns a dict with cumulative + final-run wall-clock, final ESS, and the
    chain (theta, phi samples) from the run that succeeded (or the longest run
    if the cap was hit).
    """
    n = n_init
    total_time = 0.0
    history = []
    last_out = None

    while n <= n_max:
        t0 = time.perf_counter()
        out = runner(n_samples=n, n_burnin=burnin)
        dt = time.perf_counter() - t0
        total_time += dt
        last_out = out

        ess_t = autocorr_ess(out["theta_samples"])
        ess_p = autocorr_ess(out["phi_samples"])
        history.append((n, dt, ess_t, ess_p))

        if min(ess_t, ess_p) >= ess_target:
            return {
                "theta_samples": out["theta_samples"],
                "phi_samples":   out["phi_samples"],
                "phi_name":      out["phi_name"],
                "cumulative_time_s":  total_time,
                "final_run_time_s":   dt,
                "final_n_samples":    n,
                "final_ess_theta":    ess_t,
                "final_ess_phi":      ess_p,
                "hit_target":         True,
                "history":            history,
            }
        n *= 2

    # Did not hit target within cap
    ess_t = autocorr_ess(last_out["theta_samples"])
    ess_p = autocorr_ess(last_out["phi_samples"])
    return {
        "theta_samples": last_out["theta_samples"],
        "phi_samples":   last_out["phi_samples"],
        "phi_name":      last_out["phi_name"],
        "cumulative_time_s":  total_time,
        "final_run_time_s":   history[-1][1],
        "final_n_samples":    history[-1][0],
        "final_ess_theta":    ess_t,
        "final_ess_phi":      ess_p,
        "hit_target":         False,
        "history":            history,
    }


def run_npe_reference(data, npe, n_samples: int) -> Dict:
    Xc, Yc, Xe, Ye = _data_tuple(data)
    t0 = time.perf_counter()
    res = npe.estimate_treatment_effect(Xc, Yc, Xe, Ye, n_samples=n_samples)
    dt = time.perf_counter() - t0
    theta = np.asarray(res["samples"])
    return {
        "theta_samples": theta,
        "phi_samples":   None,
        "phi_name":      None,
        "cumulative_time_s": dt,
        "final_run_time_s":  dt,
        "final_n_samples":   n_samples,
        "final_ess_theta":   float(n_samples),  # i.i.d. Gaussian draws
        "final_ess_phi":     np.nan,
        "hit_target":        True,
        "history":           [(n_samples, dt, float(n_samples), np.nan)],
    }


# ------------------------------------------------------------------
# Gold reference for W1 comparison
# ------------------------------------------------------------------
def run_gold_npp(data, n_samples: int, n_burnin: int) -> np.ndarray:
    Xc, Yc, Xe, Ye = _data_tuple(data)
    m = NormalizedPowerPriorMethod()
    res = m.estimate_treatment_effect(Xc, Yc, Xe, Ye,
                                      n_samples=n_samples, n_burnin=n_burnin)
    return np.asarray(res["samples"])


# ------------------------------------------------------------------
# One replication
# ------------------------------------------------------------------
def run_one_rep(scenario_id, sub_idx, seed, cfg, npe) -> list:
    np.random.seed(seed)
    sim_cfg = SimulationConfig(n_concurrent=cfg["n_concurrent"],
                               n_external=cfg["n_external"],
                               theta_true=cfg["theta_true"],
                               random_seed=seed)
    scenario = SimulationScenario(scenario_id, sim_cfg)
    data = scenario.generate_data(sub_scenario_idx=sub_idx)

    gold_theta = run_gold_npp(data, n_samples=cfg["gold_samples"],
                              n_burnin=cfg["gold_burnin"])

    results = {
        "NPP": time_to_quality(
            make_npp_runner(data),
            ess_target=cfg["ess_target"], n_init=cfg["n_init"],
            n_max=cfg["n_max"], burnin=cfg["burnin"],
        ),
        "Commensurate": time_to_quality(
            make_commensurate_runner(data),
            ess_target=cfg["ess_target"], n_init=cfg["n_init"],
            n_max=cfg["n_max"], burnin=cfg["burnin"],
        ),
    }
    if npe is not None:
        results["NPE"] = run_npe_reference(data, npe, n_samples=cfg["npe_samples"])

    records = []
    for mname, r in results.items():
        w1 = wasserstein1(r["theta_samples"], gold_theta)
        records.append({
            "scenario":            scenario_id,
            "seed":                seed,
            "method":              mname,
            "ess_target":          cfg["ess_target"],
            "hit_target":          r["hit_target"],
            "final_n_samples":     r["final_n_samples"],
            "final_ess_theta":     r["final_ess_theta"],
            "final_ess_phi":       r["final_ess_phi"],
            "cumulative_time_s":   r["cumulative_time_s"],
            "final_run_time_s":    r["final_run_time_s"],
            "theta_w1_vs_gold":    w1,
            "phi_name":            r["phi_name"],
        })
    return records


# ------------------------------------------------------------------
# Aggregation + tables + plots
# ------------------------------------------------------------------
def summarize(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    def _iqr_low(x):  return float(np.percentile(x, 25))
    def _iqr_high(x): return float(np.percentile(x, 75))

    agg = (df.groupby(["method", "scenario"])
             .agg(
                 n_reps=("seed", "size"),
                 hit_rate=("hit_target", "mean"),
                 median_cumulative_s=("cumulative_time_s", "median"),
                 q25_cumulative_s=("cumulative_time_s", _iqr_low),
                 q75_cumulative_s=("cumulative_time_s", _iqr_high),
                 median_final_run_s=("final_run_time_s", "median"),
                 median_final_n=("final_n_samples", "median"),
                 median_ess_theta=("final_ess_theta", "median"),
                 median_ess_phi=("final_ess_phi", "median"),
                 median_w1=("theta_w1_vs_gold", "median"),
             )
             .reset_index())
    agg.to_csv(os.path.join(out_dir, "time_to_quality_summary.csv"), index=False)

    overall = (df.groupby("method")
                 .agg(
                     n_reps=("seed", "size"),
                     hit_rate=("hit_target", "mean"),
                     median_cumulative_s=("cumulative_time_s", "median"),
                     q25_cumulative_s=("cumulative_time_s", _iqr_low),
                     q75_cumulative_s=("cumulative_time_s", _iqr_high),
                     median_final_run_s=("final_run_time_s", "median"),
                     median_w1=("theta_w1_vs_gold", "median"),
                 )
                 .reset_index())
    overall.to_csv(os.path.join(out_dir, "time_to_quality_overall.csv"), index=False)
    return overall


def write_latex_table(overall: pd.DataFrame, ess_target: int, out_dir: str) -> None:
    path = os.path.join(out_dir, "time_to_quality_table.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("% Time-to-quality table: median cumulative wall-clock to reach\n")
        f.write(f"% min(ESS(theta), ESS(phi)) >= {ess_target}.\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write(
            "Method & Median time (s) & IQR (s) & Final run (s) & "
            "$W_1$ vs.\\ gold \\\\\n"
        )
        f.write("\\midrule\n")
        for _, r in overall.iterrows():
            iqr_str = f"[{r['q25_cumulative_s']:.2f}, {r['q75_cumulative_s']:.2f}]"
            f.write(
                f"{r['method']} & {r['median_cumulative_s']:.2f} & {iqr_str} & "
                f"{r['median_final_run_s']:.2f} & {r['median_w1']:.3f} "
                "\\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")


def make_plot(df: pd.DataFrame, ess_target: int, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    methods = list(df["method"].unique())
    data = [df.loc[df["method"] == m, "cumulative_time_s"].values for m in methods]
    bp = ax.boxplot(data, labels=methods, showfliers=False, patch_artist=True,
                    widths=0.55)
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS.get(m, "#7f7f7f"))
        patch.set_alpha(0.65)
    ax.set_yscale("log")
    ax.set_ylabel("Wall-clock to reach quality target (s, log)")
    ax.set_title(
        fr"Time to min(ESS($\theta$), ESS($\phi$)) $\geq$ {ess_target}"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "time_to_quality_box.pdf"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ess-target", type=int, default=CONFIG["ess_target"])
    p.add_argument("--n-reps", type=int, default=CONFIG["n_reps"])
    p.add_argument("--n-init", type=int, default=CONFIG["n_init"])
    p.add_argument("--n-max", type=int, default=CONFIG["n_max"])
    p.add_argument("--burnin", type=int, default=CONFIG["burnin"])
    p.add_argument("--gold-samples", type=int, default=CONFIG["gold_samples"])
    p.add_argument("--gold-burnin", type=int, default=CONFIG["gold_burnin"])
    p.add_argument("--npe-samples", type=int, default=CONFIG["npe_samples"])
    p.add_argument("--npe-path", default="results/npe_pretrained.pt")
    p.add_argument("--out-dir", default="results/time_to_quality")
    args = p.parse_args()

    cfg = dict(CONFIG)
    cfg.update({
        "ess_target":   args.ess_target,
        "n_reps":       args.n_reps,
        "n_init":       args.n_init,
        "n_max":        args.n_max,
        "burnin":       args.burnin,
        "gold_samples": args.gold_samples,
        "gold_burnin":  args.gold_burnin,
        "npe_samples":  args.npe_samples,
    })

    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    npe = None
    if args.npe_path and os.path.exists(args.npe_path):
        print(f"Loading NPE from {args.npe_path}")
        npe = load_npe(args.npe_path)
    else:
        print("NPE model not provided; reporting MCMC methods only.")

    rng = np.random.default_rng(SEED)
    records = []
    total = len(SCENARIO_SPECS) * cfg["n_reps"]
    done = 0
    for sc_id, spec in SCENARIO_SPECS.items():
        seeds = rng.integers(0, 2**31 - 1, size=cfg["n_reps"])
        for rep, seed in enumerate(seeds):
            recs = run_one_rep(sc_id, spec["sub_idx"], int(seed), cfg, npe)
            for r in recs:
                r["rep"] = rep
            records.extend(recs)
            done += 1
            print(f"  [{done}/{total}] {sc_id} rep={rep + 1} done "
                  f"(NPP hit={recs[0]['hit_target']}, "
                  f"t_cum={recs[0]['cumulative_time_s']:.2f}s)")
            sys.stdout.flush()

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, "time_to_quality_raw.csv"), index=False)
    overall = summarize(df, out_dir=out_dir)
    write_latex_table(overall, ess_target=cfg["ess_target"], out_dir=out_dir)
    make_plot(df, ess_target=cfg["ess_target"], out_dir=out_dir)

    print("\nOverall summary (median cumulative time to hit ESS target):")
    print(overall.to_string(index=False))
    print(f"\nAll outputs saved under {out_dir}/")


if __name__ == "__main__":
    main()
