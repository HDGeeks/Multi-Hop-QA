# src/report/build_from_metrics.py
"""
Build paper tables & figures from metrics/ folders.

Inputs (per model under <results_dir>/<model>/metrics/>):
  1) per_run_v2.csv                         --- used for latency & stability (Tables 2, 3, 4)
  2) aggregated_items_v2.csv                --- used for item-level aggregation (Table 5)
  3) summary_v2.csv                         --- used for per-setting summary (Table 1)
  4) <model>_bertscore_per_run_v2.csv       [not directly used]
  5) <model>_bertscore_aggregated_items_v2.csv  [BERT column in Table 1]
  6) <model>_bertscore_by_domain_v2.csv     [not directly used]
  7) <model>_bertscore_summary_v2.json      [not directly used]
  8) <model>_scoring_v2.json                [not directly used]
  9) <model>_scoring_v2_extended.json       [fallback, not required]

Outputs (into <reports_dir>/tables and <reports_dir>/figures):
  Tables (CSV):
    - table1_per_setting.csv
    - table2_robust_latency.csv
    - table3_stability.csv
    - table4_leaderboard.csv
    - table5_domain_gold.csv
  Figures (PNG):
    - bar_em_by_setting.png
    - bar_f1_by_setting.png
    - bar_delta_f1.png
    - bar_latency.png

Notes / conventions:
- All EM/F1 values written to tables are in PERCENT (%).
- Table 1 F1 uses the summary_v2.csv convention (mean of per-item medians) — unchanged.
- Table 5 (domain breakdown, Gold only) uses MEAN EM and MEAN F1 across items BY DESIGN.
  This is intentional to emphasize average domain performance and to align EM/F1 aggregation
  within the table. The code comment inside build_table5 documents this explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Canonical settings we want to report in this order
SETTINGS = ["Gold", "Para", "Dist", "Para+Dist"]

def _norm_setting(val: str) -> str:
    """Normalize various setting spellings to canonical labels."""
    if val is None:
        return val
    s = str(val).strip().lower().replace("-", "_").replace("+", "_")
    if s in ("gold",):
        return "Gold"
    if s in ("para", "paraphrase", "paraphrased"):
        return "Para"
    if s in ("dist", "distractor", "distractors"):
        return "Dist"
    if s in ("para_dist", "paradist", "para__dist", "paraanddist"):
        return "Para+Dist"
    return val  # leave as-is (filtered later if not canonical)

# -------------------------------
# Utilities
# -------------------------------

def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)

def _median(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(np.median(s)) if len(s) else float("nan")

def _p(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(np.percentile(s, q)) if len(s) else float("nan")

def _p50(series: pd.Series) -> float:
    return _p(series, 50)

def _p90(series: pd.Series) -> float:
    return _p(series, 90)

def _ensure_setting_order(df: pd.DataFrame, col: str = "Setting") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=SETTINGS, ordered=True)
        df = df.sort_values([col])
    return df

def _ensure_pct(series: pd.Series) -> pd.Series:
    """
    Ensure series is in percent.
    Heuristic: if max <= 1.5 → treat as fraction and *100, else assume already %.
    """
    s = pd.to_numeric(series, errors="coerce")
    s_nonan = s.dropna()
    if not len(s_nonan):
        return s
    mx = s_nonan.max()
    if mx <= 1.5:
        return s * 100.0
    return s

def _pick_ci(df: pd.DataFrame, cand: List[str]) -> str:
    """
    Case-insensitive column picker. Returns the actual column name present in df.
    """
    lower = {c.lower(): c for c in df.columns}
    for c in cand:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"None of {cand} found in columns: {list(df.columns)}")

# -------------------------------
# Loaders
# -------------------------------

def load_summary(results_dir: Path, model: str) -> pd.DataFrame:
    """
    Load per-setting summary, robust to fraction-vs-percent.
    Output columns (always in %):
      Model | Setting | EM | F1
    """
    path = results_dir / model / "metrics" / "summary_v2.csv"
    df = _read_csv(path)
    cols_l = {c.lower(): c for c in df.columns}

    # model/setting
    col_model   = cols_l.get("model", cols_l.get("Model", None))
    col_setting = cols_l.get("setting", cols_l.get("Setting", None))
    if col_model is None or col_setting is None:
        raise KeyError(f"{path} missing model/setting. got {list(df.columns)}")

    # EM: prefer percent, else fraction
    em_key = None
    for key in ["em_mean_percent","em_percent","em_%","em (mean items)","em_mean"]:
        if key in cols_l:
            em_key = cols_l[key]
            break
    if em_key is None:
        raise KeyError(f"{path} missing EM column; got {list(df.columns)}")

    # F1: prefer percent, else fraction
    f1_key = None
    for key in ["f1_median_percent","f1_percent","f1_%","f1 (median items)","f1_median"]:
        if key in cols_l:
            f1_key = cols_l[key]
            break
    if f1_key is None:
        raise KeyError(f"{path} missing F1 column; got {list(df.columns)}")

    out = df[[col_model, col_setting, em_key, f1_key]].copy()
    out.columns = ["Model", "Setting", "EM", "F1"]

    # normalize settings & keep canonical 4
    out["Setting"] = out["Setting"].apply(_norm_setting)
    out = out[out["Setting"].isin(SETTINGS)].copy()

    # SCALE DEFENSIVELY to percent
    out["EM"] = _ensure_pct(out["EM"])
    out["F1"] = _ensure_pct(out["F1"])

    # Consistent ordering
    out["Setting"] = pd.Categorical(out["Setting"], categories=SETTINGS, ordered=True)
    out.sort_values(["Model", "Setting"], inplace=True)
    return out.reset_index(drop=True)

def load_per_run(results_dir: Path, model: str) -> pd.DataFrame:
    """
    Load per-run rows (each model call), used for latency percentiles and stability.
    Ensures: Model | Setting | QID | Domain | EM_run | F1_run | latency_ms (F1_run is fraction 0–1 in source).
    """
    path = results_dir / model / "metrics" / "per_run_v2.csv"
    df = _read_csv(path)

    # common columns: model, setting, qid, em, f1, latency_ms, domain
    colmap = {}
    for src, want in [
        ("model", "Model"),
        ("setting", "Setting"),
        ("qid", "QID"),
        ("em", "EM_run"),
        ("f1", "F1_run"),
        ("latency_ms", "latency_ms"),
        ("domain", "Domain"),
    ]:
        if src in df.columns:
            colmap[src] = want
        elif src.upper() in df.columns:
            colmap[src.upper()] = want
        elif src.capitalize() in df.columns:
            colmap[src.capitalize()] = want

    df = df.rename(columns=colmap)
    need = ["Model", "Setting", "QID", "EM_run", "F1_run", "latency_ms", "Domain"]
    for n in need:
        if n not in df.columns:
            raise KeyError(f"Column '{n}' not found in {path}. Columns={list(df.columns)}")

    # normalize settings and keep canonical
    df["Setting"] = df["Setting"].apply(_norm_setting)
    df = df[df["Setting"].isin(SETTINGS)].copy()

    # ensure numerics
    df["EM_run"] = pd.to_numeric(df["EM_run"], errors="coerce")
    df["F1_run"] = pd.to_numeric(df["F1_run"], errors="coerce")   # fraction 0–1
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")

    return df.reset_index(drop=True)

def load_aggregated_items(results_dir: Path, model: str) -> pd.DataFrame:
    """
    aggregated_items_v2.csv columns:
      ['qid','domain','model','setting','em_majority','f1_median',
       'latency_median_ms','refusal_any','invalid_any','stability_mad_f1']
    Output: Model | Setting | QID | Domain | EM_item | F1_item
    """
    path = results_dir / model / "metrics" / "aggregated_items_v2.csv"
    df = _read_csv(path)

    mapping = {
        "model": "Model",
        "setting": "Setting",
        "qid": "QID",
        "domain": "Domain",
        "em_majority": "EM_item",
        "f1_median": "F1_item",
    }
    for src, dst in mapping.items():
        if src not in df.columns:
            raise KeyError(f"Expected '{src}' in {path}; got {list(df.columns)}")
    out = df[list(mapping.keys())].rename(columns=mapping)

    out["Setting"] = out["Setting"].apply(_norm_setting)
    out = out[out["Setting"].isin(SETTINGS)].copy()

    out["EM_item"] = pd.to_numeric(out["EM_item"], errors="coerce")   # fraction 0–1
    out["F1_item"] = pd.to_numeric(out["F1_item"], errors="coerce")   # fraction 0–1
    return out.reset_index(drop=True)

def load_berts_items(results_dir: Path, model: str) -> pd.DataFrame:
    """
    Load <model>_bertscore_aggregated_items_v2.csv

    Supports two layouts:
      (A) long/tidy with columns like: Model|Setting|QID|BERT_item (0–1 or %)
      (B) wide with per-setting columns:
          berts* or berts*_percent (any case), e.g.:
            ['qid','domain','bertscore_gold','bertscore_gold_percent', ...]
    Returns columns: Model, Setting, QID, BERT_item  (always in %)
    """
    path = results_dir / model / "metrics" / f"{model}_bertscore_aggregated_items_v2.csv"
    df = _read_csv(path)

    # ---------- Case B: wide ----------
    cols_lower = {c.lower(): c for c in df.columns}
    wide_keys_raw = {
        "bertscore_gold": "Gold",
        "bertscore_para": "Para",
        "bertscore_dist": "Dist",
        "bertscore_para_dist": "Para+Dist",
    }
    # Prefer *_percent columns if present
    wide_keys_pct = {k + "_percent": v for k, v in wide_keys_raw.items()}
    has_raw = set(wide_keys_raw.keys()).issubset(set(cols_lower.keys()))
    has_pct = set(wide_keys_pct.keys()).issubset(set(cols_lower.keys()))
    if has_raw or has_pct:
        # id column
        qid_col = None
        for c in ["qid", "QID"]:
            if c in df.columns:
                qid_col = c
                break
        if qid_col is None:
            qid_col = _pick_ci(df, ["qid", "QID"])

        use_map = wide_keys_pct if has_pct else wide_keys_raw  # choose percent if available
        use_cols = [qid_col] + [cols_lower[k] for k in use_map.keys()]
        m = df[use_cols].melt(
            id_vars=[qid_col],
            value_vars=[cols_lower[k] for k in use_map.keys()],
            var_name="setting_key",
            value_name="BERT_item"
        )
        m["Setting"] = m["setting_key"].str.lower().map(use_map)
        m.drop(columns=["setting_key"], inplace=True)
        m.rename(columns={qid_col: "QID"}, inplace=True)
        m["Model"] = model
        m = m[m["Setting"].isin(SETTINGS)].copy()
        # ensure percent
        m["BERT_item"] = _ensure_pct(m["BERT_item"])
        return m[["Model","Setting","QID","BERT_item"]].reset_index(drop=True)

    # ---------- Case A: long ----------
    model_col = None
    for c in ["Model", "model"]:
        if c in df.columns:
            model_col = c
            break
    set_col = None
    for c in ["Setting", "setting"]:
        if c in df.columns:
            set_col = c
            break
    qid_col = None
    for c in ["QID", "qid"]:
        if c in df.columns:
            qid_col = c
            break

    # pick BERT column (prefer percent if exists)
    bert_col = None
    for c in [
        "BERT_item_percent","bert_item_percent","BERTScore_F1_median_percent",
        "bertscore_f1_median_percent","bertscore_f1_percent",
        "BERT_item","bert_item","bert_f1_median","bertscore_f1_median","bertscore_f1","bert_f1","bert_median","BERTScore_F1_median"
    ]:
        if c in df.columns:
            bert_col = c
            break

    if model_col and set_col and qid_col and bert_col:
        out = df[[model_col, set_col, qid_col, bert_col]].copy()
        out.columns = ["Model","Setting","QID","BERT_item"]
        # normalize + filter just once (fix: avoid double-apply)
        out["Setting"] = out["Setting"].apply(_norm_setting)
        out = out[out["Setting"].isin(SETTINGS)].copy()
        # ensure percent
        out["BERT_item"] = _ensure_pct(out["BERT_item"])
        return out.reset_index(drop=True)

    raise KeyError(f"Unrecognized BERTScore schema in {path}. Columns: {list(df.columns)}")

# --------------------------
# Build Table 1: Per-setting
# --------------------------
def build_table1(summary_by_model: Dict[str, pd.DataFrame],
                 bert_items_by_model: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Table 1: Per-setting EM / F1 (from summary) and BERT (median across items).
    F1 here follows summary_v2.csv semantics (mean of per-item medians).
    """
    rows = []
    for model, sumdf in summary_by_model.items():
        bdf = bert_items_by_model.get(model)
        for setting in SETTINGS:
            sub = sumdf[(sumdf["Setting"] == setting)]
            if not len(sub):
                continue
            em = float(sub["EM"].iloc[0])   # %
            f1 = float(sub["F1"].iloc[0])   # %
            bert = float("nan")
            if bdf is not None:
                # bdf["BERT_item"] already in %
                bert = _median(bdf[bdf["Setting"] == setting]["BERT_item"])
            rows.append({"Model": model, "Setting": setting, "EM": em, "F1": f1, "BERT": bert})
    t1 = pd.DataFrame(rows)
    t1 = _ensure_setting_order(t1)
    # Present wide format: one row per model, groups of 3 cols per setting
    out = []
    for model, g in t1.groupby("Model"):
        row = {"Model": model}
        for _, r in g.sort_values("Setting").iterrows():
            s = r["Setting"]
            row[f"{s}_EM"] = r["EM"]
            row[f"{s}_F1"] = r["F1"]
            row[f"{s}_BERT"] = r["BERT"]
        out.append(row)
    return pd.DataFrame(out).sort_values("Model").reset_index(drop=True)

# ----------------------------------------
# Build Table 2: Δ vs Gold + latency p50/90
# ----------------------------------------
def build_table2(summary_by_model: Dict[str, pd.DataFrame],
                 per_run_by_model: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Table 2: For each model, ΔEM/ΔF1 (setting minus Gold) and latency p50/p90.
    """
    rows = []
    for model, sumdf in summary_by_model.items():
        gold = sumdf[sumdf["Setting"] == "Gold"].iloc[0]
        em_gold = float(gold["EM"])
        f1_gold = float(gold["F1"])

        pr = per_run_by_model[model]
        p50 = _p50(pr["latency_ms"])
        p90 = _p90(pr["latency_ms"])

        def _delta(setting: str, col: str) -> float:
            srow = sumdf[sumdf["Setting"] == setting]
            if not len(srow):
                return float("nan")
            val = float(srow[col].iloc[0])
            base = em_gold if col == "EM" else f1_gold
            return val - base

        rows.append({
            "Model": model,
            "ΔEM_Para": _delta("Para", "EM"),
            "ΔEM_Dist": _delta("Dist", "EM"),
            "ΔEM_Both": _delta("Para+Dist", "EM"),
            "ΔF1_Para": _delta("Para", "F1"),
            "ΔF1_Dist": _delta("Dist", "F1"),
            "ΔF1_Both": _delta("Para+Dist", "F1"),
            "p50_ms": p50,
            "p90_ms": p90,
        })
    return pd.DataFrame(rows).sort_values("Model").reset_index(drop=True)

# -----------------------------------
# Build Table 3: run-to-run stability
# -----------------------------------
def _mode_agreement(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if len(vals) == 0:
        return float("nan")
    uniq, counts = np.unique(vals, return_counts=True)
    mode_val = uniq[np.argmax(counts)]
    return float(np.mean(vals == mode_val))

def build_table3(per_run_by_model: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Table 3: Stability: EM mode-agreement (%) and F1 MAD (pp), plus latency p50.
    NOTE: per_run F1 is in fraction (0–1); we convert MAD to percentage points.
    """
    rows = []
    for model, pr in per_run_by_model.items():
        em_agrees = []
        f1_mads_pp = []
        # group by Setting × QID across multiple runs
        for (_, _), g in pr.groupby(["Setting", "QID"]):
            em_agrees.append(_mode_agreement(g["EM_run"]))
            # MAD in fraction → convert to percentage points
            f1_mad_frac = np.median(np.abs(g["F1_run"] - np.median(g["F1_run"])))
            f1_mads_pp.append(100.0 * float(f1_mad_frac))
        em_aggr = float(np.nanmean(em_agrees) * 100.0) if len(em_agrees) else float("nan")  # %
        f1_mad_aggr = float(np.nanmedian(f1_mads_pp)) if len(f1_mads_pp) else float("nan")  # pp
        p50 = _p50(pr["latency_ms"])
        rows.append({"Model": model, "EM-Agree (%)": em_aggr, "F1-MAD (pp)": f1_mad_aggr, "Latency p50 (ms)": p50})
    return pd.DataFrame(rows).sort_values("Model").reset_index(drop=True)

# -----------------------------------------
# Build Table 4: Composite Leaderboard
# -----------------------------------------
def build_table4(summary_by_model: Dict[str, pd.DataFrame],
                 per_run_by_model: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Table 4: Composite ranking combining accuracy and robustness with latency.
    """
    rows = []
    for model, sumdf in summary_by_model.items():
        gold_f1 = float(sumdf[sumdf["Setting"] == "Gold"]["F1"].iloc[0])
        f1_vals = []
        for s in SETTINGS:
            srow = sumdf[sumdf["Setting"] == s]
            if len(srow):
                f1_vals.append(float(srow["F1"].iloc[0]))
        avg_f1 = float(np.mean(f1_vals)) if len(f1_vals) else float("nan")
        drops = []
        for s in ["Para", "Dist", "Para+Dist"]:
            srow = sumdf[sumdf["Setting"] == s]
            if len(srow):
                drops.append(abs(float(srow["F1"].iloc[0]) - gold_f1))
        avg_abs_drop = float(np.mean(drops)) if len(drops) else float("nan")

        p50_ms = _p50(per_run_by_model[model]["latency_ms"])
        efficiency = gold_f1 / (p50_ms / 1000.0) if p50_ms and not np.isnan(p50_ms) else float("nan")
        composite = avg_f1 - 0.5 * avg_abs_drop if not (np.isnan(avg_f1) or np.isnan(avg_abs_drop)) else float("nan")

        rows.append({
            "Model": model,
            "Gold F1": gold_f1,
            "Avg F1": avg_f1,
            "Avg |ΔF1| (pp)": avg_abs_drop,
            "p50 (ms)": p50_ms,
            "Efficiency (F1/s)": efficiency,
            "Composite ↑": composite,
        })
    t4 = pd.DataFrame(rows).sort_values("Composite ↑", ascending=False).reset_index(drop=True)
    return t4

# -----------------------------------------
# Build Table 5: Domain breakdown (Gold only)
# -----------------------------------------
def build_table5(aggregated_items_by_model: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Table 5: Domain-wise Gold performance.
    Aggregation is intentionally:
      - EM: MEAN across items (×100)
      - F1: MEAN across items (×100)
    Rationale: For a domain breakdown we emphasize average performance (not robustness),
    and we want EM and F1 to share the same aggregation notion within the table.
    """
    domains: List[str] = []
    rows = []
    for model, df in aggregated_items_by_model.items():
        g = df[df["Setting"] == "Gold"].copy()
        if g.empty:
            continue

        # collect domain universe
        domains = sorted(list(set(domains) | set(g["Domain"].unique())))

        # aggregate (convert fractions -> percent)
        agg = g.groupby("Domain").agg(
            EM=("EM_item",  lambda s: 100.0 * float(np.mean(pd.to_numeric(s, errors="coerce")))),
            # Intentionally using MEAN F1 here (not median). See module docstring + comment above.
            F1=("F1_item",  lambda s: 100.0 * float(np.mean(pd.to_numeric(s, errors="coerce")))),
        ).reset_index()

        rows.append((model, agg))

    # wide layout
    all_models = []
    for model, agg in rows:
        row = {"Model": model}
        for _, r in agg.iterrows():
            d = str(r["Domain"])
            row[f"{d}_EM"] = float(r["EM"])
            row[f"{d}_F1"] = float(r["F1"])
        all_models.append(row)

    # consistent column order
    cols = ["Model"]
    for d in sorted(domains):
        cols.extend([f"{d}_EM", f"{d}_F1"])
    out = pd.DataFrame(all_models)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols].sort_values("Model").reset_index(drop=True)

    # Avg. row across domains (simple mean of the per-domain % columns)
    avg_row = {"Model": "Avg."}
    for d in sorted(domains):
        avg_row[f"{d}_EM"] = float(np.nanmean(out[f"{d}_EM"]))
        avg_row[f"{d}_F1"] = float(np.nanmean(out[f"{d}_F1"]))
    out = pd.concat([out, pd.DataFrame([avg_row])], ignore_index=True)
    return out

# -------------------------------
# Orchestration
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="Root results dir (e.g., src/results)")
    ap.add_argument("--reports-dir", required=True, help="Output dir for tables/ and figures/")
    ap.add_argument("--models", nargs="+", required=True, help="Models to include")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    reports_dir = Path(args.reports_dir)
    tables_dir = reports_dir / "tables"
    figs_dir = reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Load all inputs
    summary_by_model: Dict[str, pd.DataFrame] = {}
    per_run_by_model: Dict[str, pd.DataFrame] = {}
    aggregated_items_by_model: Dict[str, pd.DataFrame] = {}
    bert_items_by_model: Dict[str, pd.DataFrame] = {}

    for model in args.models:
        summary_by_model[model] = load_summary(results_dir, model)
        per_run_by_model[model] = load_per_run(results_dir, model)
        aggregated_items_by_model[model] = load_aggregated_items(results_dir, model)
        bert_items_by_model[model] = load_berts_items(results_dir, model)

    # ---- Build tables ----
    # Table 1 (per setting; EM/F1 from summary, BERT from BERT aggregated items)
    t1_wide = build_table1(summary_by_model, bert_items_by_model)
    t1_wide.to_csv(tables_dir / "table1_per_setting.csv", index=False)

    # Table 2 (deltas + latency)
    t2 = build_table2(summary_by_model, per_run_by_model)
    t2.to_csv(tables_dir / "table2_robust_latency.csv", index=False)

    # Table 3 (stability)
    t3 = build_table3(per_run_by_model)
    t3.to_csv(tables_dir / "table3_stability.csv", index=False)

    # Table 4 (leaderboard)
    t4 = build_table4(summary_by_model, per_run_by_model)
    t4.to_csv(tables_dir / "table4_leaderboard.csv", index=False)

    # Table 5 (domain, Gold only; MEAN EM & MEAN F1 intentionally)
    t5 = build_table5(aggregated_items_by_model)
    t5.to_csv(tables_dir / "table5_domain_gold.csv", index=False)

    # Quick console summary
    print("✅ Built tables:")
    for p in sorted(tables_dir.glob("*.csv")):
        print("  -", p)
    print("✅ Figures directory ready (figures are built by make_bar_charts.py):")
    for p in sorted(figs_dir.glob("*.png")):
        print("  -", p)

if __name__ == "__main__":
    main()


