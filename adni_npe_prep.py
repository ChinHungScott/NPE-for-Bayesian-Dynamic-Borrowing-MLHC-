from __future__ import annotations

"""
Prepare a simple ADNI real-world application dataset for the baseline NPE-BDB method.

Design goal
-----------
Keep the real-data application as close as possible to the original NPE setup:

    concurrent dataset  -> X_concurrent, Y_concurrent
    external dataset    -> X_external,   Y_external

Recommended first application
-----------------------------
- Population: baseline MCI subjects
- External source: ADNI1
- Concurrent source: ADNIGO + ADNI2
- Outcome: conversion to dementia/AD by ~24 months
- Covariates: baseline-only clinical features

This gives a simple, defensible first real-world borrowing example for the main paper,
without requiring transformer models or raw sequence modeling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PHASE_COL_CANDIDATES = ["COLPROT", "ORIGPROT", "PHASE"]
RID_COL_CANDIDATES = ["RID", "PTID", "USUBJID"]
VISIT_COL_CANDIDATES = ["VISCODE2", "VISCODE"]
DATE_COL_CANDIDATES = ["VISDATE", "EXAMDATE", "USERDATE", "USERDATE2"]


def load_adni_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".rda":
        try:
            import pyreadr
        except ImportError as exc:
            raise ImportError("Reading .rda files requires `pyreadr`. Install it with `pip install pyreadr`.") from exc
        result = pyreadr.read_r(path)
        if not result:
            raise ValueError(f"No object found in {path}")
        return next(iter(result.values()))
    raise ValueError(f"Unsupported file type: {path}")


def _first_existing(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    cols = set(columns)
    for candidate in candidates:
        if candidate in cols:
            return candidate
    return None


def _require_column(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    col = _first_existing(df.columns, candidates)
    if col is None:
        raise KeyError(f"Could not find {label}. Tried columns: {list(candidates)}")
    return col


def _normalize_rid_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _standardize_phase(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.upper().str.strip()
    out = out.replace({
        "ADNI GO": "ADNIGO",
        "GO": "ADNIGO",
    })
    return out


def _parse_month(viscode: object) -> float:
    if pd.isna(viscode):
        return np.nan
    s = str(viscode).strip().lower()
    if s in {"bl", "m0", "sc", "init", "v00", "f", "screening"}:
        return 0.0
    if s.startswith("m") and s[1:].isdigit():
        return float(s[1:])
    if s.startswith("y") and s[1:].isdigit():
        return 12.0 * float(s[1:])
    if s.replace(".", "", 1).isdigit():
        return float(s)
    return np.nan


def _best_visit_per_subject(
    df: pd.DataFrame,
    rid_col: str,
    visit_col: str,
    target_month: float,
    max_gap: float,
) -> pd.DataFrame:
    out = df.copy()
    out["_month"] = out[visit_col].map(_parse_month)
    out = out.loc[out["_month"].notna()].copy()
    out["_gap"] = (out["_month"] - target_month).abs()
    out = out.loc[out["_gap"] <= max_gap].copy()
    if out.empty:
        return out
    out = out.sort_values([rid_col, "_gap", "_month"])
    return out.groupby(rid_col, as_index=False).first()


def _baseline_visit_per_subject(df: pd.DataFrame, rid_col: str, visit_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_month"] = out[visit_col].map(_parse_month)
    out = out.loc[out["_month"].notna()].copy()
    out = out.sort_values([rid_col, "_month"])
    return out.groupby(rid_col, as_index=False).first()


def _find_numeric_col(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    col = _require_column(df, candidates, label)
    return col


def _extract_baseline_score(
    df: pd.DataFrame,
    score_candidates: Sequence[str],
    new_name: str,
) -> pd.DataFrame:
    rid_col = _require_column(df, RID_COL_CANDIDATES, "subject id")
    visit_col = _require_column(df, VISIT_COL_CANDIDATES, "visit code")
    score_col = _find_numeric_col(df, score_candidates, new_name)

    base = _baseline_visit_per_subject(df[[rid_col, visit_col, score_col]].copy(), rid_col, visit_col)
    out = base[[rid_col, score_col]].rename(columns={rid_col: "RID", score_col: new_name})
    out["RID"] = _normalize_rid_series(out["RID"])
    return out


def _extract_phase_frame(adsl: pd.DataFrame) -> pd.DataFrame:
    rid_col = _require_column(adsl, ["RID"], "subject id")
    phase_col = _require_column(adsl, PHASE_COL_CANDIDATES, "phase")
    out = adsl[[rid_col, phase_col]].copy()
    out[rid_col] = _normalize_rid_series(out[rid_col])
    out[phase_col] = _standardize_phase(out[phase_col])
    out = out.drop_duplicates(subset=[rid_col])
    return out.rename(columns={phase_col: "phase", rid_col: "RID"})


def _extract_demographics(ptdemog: pd.DataFrame) -> pd.DataFrame:
    rid_col = _require_column(ptdemog, RID_COL_CANDIDATES, "subject id")
    visit_col = _require_column(ptdemog, VISIT_COL_CANDIDATES, "visit code")

    base = _baseline_visit_per_subject(ptdemog.copy(), rid_col, visit_col)

    col_map: Dict[str, Sequence[str]] = {
        "age": ["AGE", "PTAGE"],
        "sex": ["PTGENDER", "SEX", "GENDER"],
        "education": ["PTEDUCAT", "EDUC", "EDUCATION"],
    }

    keep = [rid_col]
    rename = {}
    for new_name, candidates in col_map.items():
        col = _first_existing(base.columns, candidates)
        if col is not None:
            keep.append(col)
            rename[col] = new_name

    out = base[keep].rename(columns=rename)
    out = out.rename(columns={rid_col: "RID"})
    out["RID"] = _normalize_rid_series(out["RID"])

    if "age" not in out.columns:
        dob_year_col = _first_existing(base.columns, ["PTDOBYY", "BIRTHYR", "BIRTHYEAR"])
        date_col = _first_existing(base.columns, DATE_COL_CANDIDATES)
        if dob_year_col is not None and date_col is not None:
            dob_year = pd.to_numeric(base[dob_year_col], errors="coerce")
            visit_year = pd.to_datetime(base[date_col], errors="coerce").dt.year
            out["age"] = visit_year - dob_year

    if "sex" in out.columns:
        out["female"] = out["sex"].astype(str).str.upper().str.startswith("F").astype(float)
        out = out.drop(columns=["sex"])

    return out


def _extract_apoe(apoe: pd.DataFrame) -> pd.DataFrame:
    rid_col = _require_column(apoe, RID_COL_CANDIDATES, "subject id")
    candidates = ["APOE4", "APGEN1", "NUME4", "APOE4NUM"]
    col = _first_existing(apoe.columns, candidates)
    if col is None and "GENOTYPE" in apoe.columns:
        out = apoe[[rid_col, "GENOTYPE"]].drop_duplicates(subset=[rid_col]).copy()
        out = out.rename(columns={rid_col: "RID"})
        out["RID"] = _normalize_rid_series(out["RID"])
        geno = out["GENOTYPE"].astype(str)
        out["apoe4_count"] = geno.str.count("4").astype(float)
        return out[["RID", "apoe4_count"]]
    if col is None:
        out = apoe[[rid_col]].drop_duplicates().copy()
        out["apoe4_count"] = np.nan
        out = out.rename(columns={rid_col: "RID"})
        out["RID"] = _normalize_rid_series(out["RID"])
        return out

    out = apoe[[rid_col, col]].drop_duplicates(subset=[rid_col]).copy()
    out = out.rename(columns={rid_col: "RID", col: "apoe4_count"})
    out["RID"] = _normalize_rid_series(out["RID"])
    out["apoe4_count"] = pd.to_numeric(out["apoe4_count"], errors="coerce")
    return out


def _extract_baseline_dx(dxsum: pd.DataFrame) -> pd.DataFrame:
    rid_col = _require_column(dxsum, RID_COL_CANDIDATES, "subject id")
    visit_col = _require_column(dxsum, VISIT_COL_CANDIDATES, "visit code")
    dx_col = _first_existing(dxsum.columns, ["DIAGNOSIS", "DX", "DX_bl", "DXCHANGE"])
    if dx_col is None:
        raise KeyError("Could not find a diagnosis column in DXSUM.")

    base = _baseline_visit_per_subject(dxsum[[rid_col, visit_col, dx_col]].copy(), rid_col, visit_col)
    out = base[[rid_col, dx_col]].rename(columns={rid_col: "RID", dx_col: "baseline_dx_raw"})
    out["RID"] = _normalize_rid_series(out["RID"])
    out["baseline_dx"] = out["baseline_dx_raw"].astype(str).str.upper()
    out["baseline_dx"] = out["baseline_dx"].replace({
        "EMCI": "MCI",
        "LMCI": "MCI",
        "SMC": "CN",
        "NL": "CN",
        "NORMAL": "CN",
        "DEMENTIA": "DEM",
        "AD": "DEM",
    })
    return out


def _extract_conversion_outcome(dxsum: pd.DataFrame, target_month: float = 24.0, max_gap: float = 6.0) -> pd.DataFrame:
    rid_col = _require_column(dxsum, RID_COL_CANDIDATES, "subject id")
    visit_col = _require_column(dxsum, VISIT_COL_CANDIDATES, "visit code")
    dx_col = _first_existing(dxsum.columns, ["DIAGNOSIS", "DX", "DX_bl", "DXCHANGE"])
    if dx_col is None:
        raise KeyError("Could not find a diagnosis column in DXSUM.")

    fu = _best_visit_per_subject(dxsum[[rid_col, visit_col, dx_col]].copy(), rid_col, visit_col, target_month, max_gap)
    out = fu[[rid_col, dx_col]].rename(columns={rid_col: "RID", dx_col: "followup_dx_raw"})
    out["RID"] = _normalize_rid_series(out["RID"])
    out["followup_dx"] = out["followup_dx_raw"].astype(str).str.upper()
    out["followup_dx"] = out["followup_dx"].replace({
        "EMCI": "MCI",
        "LMCI": "MCI",
        "SMC": "CN",
        "NL": "CN",
        "NORMAL": "CN",
        "DEMENTIA": "DEM",
        "AD": "DEM",
    })
    out["conversion_24m"] = (out["followup_dx"] == "DEM").astype(float)
    return out


@dataclass
class ADNIPrepConfig:
    external_phases: Tuple[str, ...] = ("ADNI1",)
    concurrent_phases: Tuple[str, ...] = ("ADNIGO", "ADNI2")
    target_month: float = 24.0
    target_month_window: float = 6.0
    require_complete_cases: bool = True


def prepare_adni_npe_application(
    adsl: pd.DataFrame,
    dxsum: pd.DataFrame,
    ptdemog: pd.DataFrame,
    mmse: pd.DataFrame,
    adas: pd.DataFrame,
    cdr: pd.DataFrame,
    faq: pd.DataFrame,
    apoe: pd.DataFrame,
    config: ADNIPrepConfig | None = None,
) -> Dict[str, object]:
    """
    Build a simple subject-level ADNI dataset for the baseline NPE-BDB pipeline.

    Returns a dictionary with:
    - X_concurrent, Y_concurrent
    - X_external, Y_external
    - feature_names
    - analytic_subjects (subject-level DataFrame used for the split)
    """
    if config is None:
        config = ADNIPrepConfig()

    # Use a table with a true RID for cohort/phase extraction.
    phase_source = dxsum if "RID" in dxsum.columns else ptdemog
    phase_df = _extract_phase_frame(phase_source)
    demo_df = _extract_demographics(ptdemog)
    apoe_df = _extract_apoe(apoe)
    dx_base_df = _extract_baseline_dx(dxsum)
    dx_outcome_df = _extract_conversion_outcome(
        dxsum,
        target_month=config.target_month,
        max_gap=config.target_month_window,
    )

    mmse_df = _extract_baseline_score(mmse, ["MMSCORE", "MMSE", "TOTSCORE"], "mmse_bl")
    adas_df = _extract_baseline_score(adas, ["TOTAL13", "TOTSCORE", "ADAS13"], "adas13_bl")
    cdr_df = _extract_baseline_score(cdr, ["CDRSB", "CDGLOBAL"], "cdrsb_bl")
    faq_df = _extract_baseline_score(faq, ["FAQTOTAL", "FAQ_TOTAL"], "faq_bl")

    subject_df = phase_df.merge(demo_df, on="RID", how="left")
    for frame in [apoe_df, dx_base_df, dx_outcome_df, mmse_df, adas_df, cdr_df, faq_df]:
        subject_df = subject_df.merge(frame, on="RID", how="left")

    subject_df["phase"] = _standardize_phase(subject_df["phase"])

    allowed_phases = set(config.external_phases).union(config.concurrent_phases)
    subject_df = subject_df.loc[subject_df["phase"].isin(allowed_phases)].copy()
    subject_df = subject_df.loc[subject_df["baseline_dx"] == "MCI"].copy()
    subject_df = subject_df.loc[subject_df["conversion_24m"].notna()].copy()

    subject_df["source"] = np.where(
        subject_df["phase"].isin(config.concurrent_phases),
        "concurrent",
        np.where(subject_df["phase"].isin(config.external_phases), "external", "drop"),
    )
    subject_df = subject_df.loc[subject_df["source"] != "drop"].copy()

    feature_names = [
        "age",
        "female",
        "education",
        "apoe4_count",
        "mmse_bl",
        "adas13_bl",
        "cdrsb_bl",
        "faq_bl",
    ]

    for col in feature_names:
        if col not in subject_df.columns:
            subject_df[col] = np.nan

    if config.require_complete_cases:
        subject_df = subject_df.dropna(subset=feature_names + ["conversion_24m"]).copy()
    else:
        for col in feature_names:
            subject_df[col] = subject_df[col].astype(float)
            subject_df[col] = subject_df[col].fillna(subject_df[col].median())

    concurrent = subject_df.loc[subject_df["source"] == "concurrent"].copy()
    external = subject_df.loc[subject_df["source"] == "external"].copy()

    X_concurrent = concurrent[feature_names].to_numpy(dtype=float)
    Y_concurrent = concurrent["conversion_24m"].to_numpy(dtype=float)
    X_external = external[feature_names].to_numpy(dtype=float)
    Y_external = external["conversion_24m"].to_numpy(dtype=float)

    return {
        "X_concurrent": X_concurrent,
        "Y_concurrent": Y_concurrent,
        "X_external": X_external,
        "Y_external": Y_external,
        "feature_names": feature_names,
        "analytic_subjects": subject_df.reset_index(drop=True),
        "concurrent_subjects": concurrent.reset_index(drop=True),
        "external_subjects": external.reset_index(drop=True),
        "config": config,
    }


def print_adni_application_summary(data: Dict[str, object]) -> None:
    concurrent = data["concurrent_subjects"]
    external = data["external_subjects"]

    print("=" * 72)
    print("ADNI NPE APPLICATION SUMMARY")
    print("=" * 72)
    print(f"Concurrent subjects: {len(concurrent)}")
    print(f"External subjects:   {len(external)}")
    print(f"Feature dimension:   {len(data['feature_names'])}")
    if len(concurrent) > 0:
        print(f"Concurrent conversion rate: {concurrent['conversion_24m'].mean():.3f}")
    if len(external) > 0:
        print(f"External conversion rate:   {external['conversion_24m'].mean():.3f}")
    print("Features:", ", ".join(data["feature_names"]))


if __name__ == "__main__":
    print(
        "This module prepares a simple ADNI real-world application dataset for the\n"
        "baseline NPE-BDB pipeline. Import the functions and pass DataFrames loaded\n"
        "from ADSL, DXSUM, PTDEMOG, MMSE, ADAS, CDR, FAQ, and APOERES."
    )
