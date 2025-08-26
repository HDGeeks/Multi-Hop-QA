#!/usr/bin/env python3
"""
Builds publication-ready tables & figures from scoring outputs.

Inputs (per model under --results-root/<model>/metrics/; any of these names work):
  Summary (pick first that exists):
    - <model>_summary.csv            (v1)
    - summary_v2.csv                 (v2)
    - <model>_scoring_v2_extended.json (v2 JSON fallback; EM/F1 per setting)

  Aggregated items (pick first that exists):
    - <model>_aggregated_items.csv   (v1)
    - aggregated_items_v2.csv        (v2)

  Per-run (pick first that exists):
    - <model>_per_run.csv            (v1)
    - per_run_v2.csv                 (v2)

  Per-domain (optional):
    - <model>_per_domain.json

BERTScore files are not required for these tables/figures.

Outputs:
  Tables → <out-root>/tables/*.tex
  Figures → <out-root>/figures/*.pdf
"""

import argparse, json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Helpers ----------
SETTINGS = ["gold", "para", "dist", "para_dist"]
SETTING_LABEL = {
    "gold":"Gold",
    "para":"Gold+Paraphrase",
    "dist":"Gold+Distractor",
    "para_dist":"Gold+Paraphrase+Distractor",
}

def ensure_out_dirs(root: Path):
    (root/"tables").mkdir(parents=True, exist_ok=True)
    (root/"figures").mkdir(parents=True, exist_ok=True)

def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

# ---------- Loaders ----------
def read_summary(base: Path, model: str) -> Optional[pd.DataFrame]:
    """
    Loads a model's summary as a normalized DataFrame with columns at least:
      ['model', 'setting', 'em_mean', 'f1_median', ...]
    Accepts either the v1 CSV summary or v2 JSON (scoring_v2_extended.json).
    Also normalizes the 'setting' column to {gold, para, dist, para_dist}.
    """
    # Try CSV summaries first
    cand = _first_existing([
        base / f"{model}_summary.csv",
        base / "summary_v2.csv",
    ])
    if cand:
        df = pd.read_csv(cand)
        # Make columns consistent
        df = _normalize_summary_cols(df)

        # Remove duplicate-named columns if any (pivot-safe)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        # Ensure we have a 'setting' column; try common fallbacks
        if "setting" not in df.columns:
            for alt in ["condition", "config", "variant"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "setting"})
                    break

        # Normalize setting values
        if "setting" in df.columns:
            m = {
                "gold": "gold",
                "para": "para",
                "paraphrase": "para",
                "dist": "dist",
                "distractor": "dist",
                "both": "para_dist",
                "para_dist": "para_dist",
                "gold+paraphrase": "para",
                "gold+distractor": "dist",
                "gold+paraphrase+distractor": "para_dist",
            }
            df["setting"] = df["setting"].astype(str).str.strip().str.lower().map(lambda s: m.get(s, s))
        else:
            # If still no setting column, we can't build T1; return None so caller can skip
            return None

        # Ensure 'model' column is set to current model
        if "model" not in df.columns:
            df["model"] = model
        else:
            df["model"] = df["model"].fillna(model).replace("", model)

        return df

    # Fallback: v2 JSON (scoring_v2_extended.json)
    cand_json = _first_existing([
        base / f"{model}_scoring_v2_extended.json",
        base / "scoring_v2_extended.json",
    ])
    if cand_json and cand_json.exists():
        with open(cand_json, "r", encoding="utf-8") as f:
            js = json.load(f)

        by_setting = js.get("by_setting", {})
        rows = []
        # keys are already expected to be 'gold','para','dist','para_dist'
        for setting, vals in by_setting.items():
            rows.append({
                "model": model,
                "setting": str(setting).strip().lower(),
                "em_mean": vals.get("em_percent", np.nan),    # already in %
                "f1_median": vals.get("f1_percent", np.nan),  # already in %
            })
        if rows:
            df = pd.DataFrame(rows)
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]
            return df

    return None

def _normalize_summary_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Handle both raw % and 0..1 scales; coerce into % (0..100).
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"model"}: colmap[c] = "model"
        elif lc in {"setting"}: colmap[c] = "setting"
        elif "em" in lc and "mean" in lc and "drop" not in lc: colmap[c] = "em_mean"
        elif "f1" in lc and ("median" in lc or "mean" in lc) and "drop" not in lc: colmap[c] = "f1_median"
        elif "latency" in lc and ("median" in lc or "p50" in lc): colmap[c] = "latency_p50"
        elif "latency" in lc and "p90" in lc: colmap[c] = "latency_p90"
    out = df.rename(columns=colmap).copy()
    # If EM/F1 look like fractions, multiply by 100
    for k in ("em_mean","f1_median"):
        if k in out.columns:
            s = out[k]
            # Heuristic: if max ≤ 1.0, treat as fraction
            try:
                if pd.to_numeric(s, errors="coerce").max() is not None and pd.to_numeric(s, errors="coerce").max() <= 1.0:
                    out[k] = pd.to_numeric(s, errors="coerce") * 100.0
            except Exception:
                pass
    return out

def read_items_csv(base: Path, model: str) -> Optional[pd.DataFrame]:
    p = _first_existing([
        base / f"{model}_aggregated_items.csv",
        base / "aggregated_items_v2.csv",
    ])
    if not p: return None
    df = pd.read_csv(p)
    df.columns = [c.lower() for c in df.columns]
    return df

def read_per_run_csv(base: Path, model: str) -> Optional[pd.DataFrame]:
    p = _first_existing([
        base / f"{model}_per_run.csv",
        base / "per_run_v2.csv",
    ])
    if not p: return None
    df = pd.read_csv(p)
    df.columns = [c.lower() for c in df.columns]
    # Normalize col names used in behavior table
    if "latency" in df.columns and "latency_ms" not in df.columns:
        df = df.rename(columns={"latency":"latency_ms"})
    return df

def read_per_domain_json(base: Path, model: str) -> Dict:
    p = base / f"{model}_per_domain.json"
    if not p.exists(): return {}
    with open(p, "r") as f:
        return json.load(f)

# ---------- Tables ----------
def table_T1_main_accuracy(all_summaries: pd.DataFrame, out_tex: Path):
    """
    Writes a LaTeX table with EM and F1 per setting × model.
    Handles input where EM/F1 are proportions or already in %.
    Robust to missing/extra columns and empty subsets.
    """
    df = all_summaries.copy()

    # Drop duplicate-named columns (e.g., after multiple merges)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # Must have 'model', 'setting', 'em_mean', 'f1_median'
    needed = {"model", "setting", "em_mean", "f1_median"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"[T1] Missing required columns: {sorted(missing)} in summaries")

    # Normalize setting values to canonical set
    m = {
        "gold": "gold",
        "para": "para",
        "paraphrase": "para",
        "dist": "dist",
        "distractor": "dist",
        "both": "para_dist",
        "para_dist": "para_dist",
        "gold+paraphrase": "para",
        "gold+distractor": "dist",
        "gold+paraphrase+distractor": "para_dist",
    }
    df["setting"] = df["setting"].astype(str).str.strip().str.lower().map(lambda s: m.get(s, s))

    # Filter to known settings and add display label
    df = df[df["setting"].isin(SETTINGS)].copy()
    df["setting_label"] = df["setting"].map(SETTING_LABEL)

    # If nothing to show (e.g., only one model finished), emit a tiny valid table and return
    if df.empty:
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        with open(out_tex, "w", encoding="utf-8") as f:
            f.write("\\begin{table}[t]\\centering\n")
            f.write("\\caption{Main accuracy (no data available).}\n")
            f.write("\\label{tab:main-accuracy}\n")
            f.write("\\begin{tabular}{l}\n\\toprule\nSetting \\\\\n\\midrule\n-- \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        return

    # Macro-average per setting across models
    macro = (df.groupby("setting", as_index=False)
               .agg(em_mean=("em_mean", "mean"),
                    f1_median=("f1_median", "median")))
    macro["model"] = "Macro-Avg"
    macro["setting_label"] = macro["setting"].map(SETTING_LABEL)

    big = pd.concat([df, macro], ignore_index=True, sort=False)

    # Pivot to Setting × Model
    em_pvt = big.pivot(index="setting_label", columns="model", values="em_mean")
    f1_pvt = big.pivot(index="setting_label", columns="model", values="f1_median")

    # Sort rows by canonical setting order
    em_pvt = em_pvt.reindex([SETTING_LABEL[s] for s in SETTINGS if SETTING_LABEL[s] in em_pvt.index])
    f1_pvt = f1_pvt.reindex([SETTING_LABEL[s] for s in SETTINGS if SETTING_LABEL[s] in f1_pvt.index])

    # Helper to format either proportions or percentages
    def _fmt(v):
        if pd.isna(v):
            return "NaN"
        try:
            vv = float(v)
        except Exception:
            return "NaN"
        return "{:.1f}".format(100 * vv) if 0.0 <= vv <= 1.0 else "{:.1f}".format(vv)

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\\centering\n")
        f.write("\\caption{Main accuracy by setting and model (EM and F1; \\%).}\n")
        f.write("\\label{tab:main-accuracy}\n")
        f.write("\\begin{tabular}{l" + "r" * len(em_pvt.columns) + "}\n\\toprule\n")
        f.write("Setting & " + " & ".join(em_pvt.columns) + " \\\\\n\\midrule\n")
        for idx, row in em_pvt.iterrows():
            vals = [_fmt(row[c]) for c in em_pvt.columns]
            f.write(f"{idx} (EM) & " + " & ".join(vals) + " \\\\\n")
        for idx, row in f1_pvt.iterrows():
            vals = [_fmt(row[c]) for c in f1_pvt.columns]
            f.write(f"{idx} (F1) & " + " & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

def table_T2_drops(all_summaries: pd.DataFrame, out_tex: Path):
    want = []
    for model, sdf in all_summaries.groupby("model"):
        sdf = sdf.set_index("setting")
        row = {"model": model}
        try:
            base_f1 = sdf.at["gold","f1_median"]
            row["ΔF1_para_pp"] = sdf.at["para","f1_median"] - base_f1
            row["ΔF1_dist_pp"] = sdf.at["dist","f1_median"] - base_f1
            row["ΔF1_both_pp"] = sdf.at["para_dist","f1_median"] - base_f1
        except Exception:
            row["ΔF1_para_pp"]=row["ΔF1_dist_pp"]=row["ΔF1_both_pp"]=np.nan
        try:
            base_em = sdf.at["gold","em_mean"]
            row["ΔEM_para_pp"] = sdf.at["para","em_mean"] - base_em
            row["ΔEM_dist_pp"] = sdf.at["dist","em_mean"] - base_em
            row["ΔEM_both_pp"] = sdf.at["para_dist","em_mean"] - base_em
        except Exception:
            row["ΔEM_para_pp"]=row["ΔEM_dist_pp"]=row["ΔEM_both_pp"]=np.nan
        want.append(row)
    tbl = pd.DataFrame(want).set_index("model").loc[sorted(set(all_summaries["model"]))]

    with open(out_tex, "w") as f:
        f.write("\\begin{table}[t]\\centering\n")
        f.write("\\caption{Perturbation drops relative to Gold (percentage points). Negative = improvement.}\n")
        f.write("\\label{tab:drops}\n")
        cols = ["ΔEM_para_pp","ΔEM_dist_pp","ΔEM_both_pp","ΔF1_para_pp","ΔF1_dist_pp","ΔF1_both_pp"]
        f.write("\\begin{tabular}{l" + "r"*len(cols) + "}\n\\toprule\n")
        f.write("Model & " + " & ".join(cols) + " \\\\\n\\midrule\n")
        for m, row in tbl.iterrows():
            f.write(m + " & " + " & ".join(
                ("{:+.1f}".format(row[c]) if pd.notna(row[c]) else "–") for c in cols
            ) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

def table_T3_domain(per_domain_jsons: Dict[str, Dict], items_map: Dict[str, pd.DataFrame], out_tex: Path):
    """
    Build per-model domain table. Prefer per_domain_json if present.
    If missing, compute from items_map[model] (aggregated_items_v2.csv):
      - EM = mean(em_majority)*100
      - F1 = median(f1_median)*100
    Shows Gold vs Gold+Distractor.
    """
    models_done = 0
    with open(out_tex, "w") as f:
        for model in sorted(set(list(per_domain_jsons.keys()) + list(items_map.keys()))):
            payload = per_domain_jsons.get(model)

            # Fallback: compute from items if payload is absent
            if not payload:
                df = items_map.get(model)
                if df is None or df.empty:
                    continue
                df = df.copy()
                df["domain"] = df["domain"].str.capitalize()
                def agg_for(setting):
                    sdf = df[df["setting"] == setting]
                    if sdf.empty:
                        return {}
                    stats = (sdf.groupby("domain")
                               .agg(em=("em_majority","mean"),
                                    f1=("f1_median","median"))
                               .reset_index())
                    out = {}
                    for _, r in stats.iterrows():
                        out[str(r["domain"])] = {
                            "em": float(r["em"])*100.0,
                            "f1": float(r["f1"])*100.0,
                        }
                    return out
                payload = {
                    "gold": agg_for("gold"),
                    "dist": agg_for("dist"),
                }

            gold = payload.get("gold") or payload.get("Gold") or {}
            dist = payload.get("dist") or payload.get("Gold+Distractor") or {}

            # domains to show (sorted for stable output)
            doms = sorted(set(list(gold.keys()) + list(dist.keys())))
            if not doms:
                continue

            f.write("\\begin{table}[t]\\centering\n")
            f.write(f"\\caption{{Domain breakdown for \\texttt{{{model}}}: Gold vs. Gold+Distractor. Values are percent.}}\n")
            f.write(f"\\label{{tab:domain-{model}}}\n")
            f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
            f.write("Domain & EM$_{Gold}$ & F1$_{Gold}$ & EM$_{Gold+Dist}$ & F1$_{Gold+Dist}$ \\\\\n\\midrule\n")
            def fmt(v): return "--" if v is None else f"{v:.1f}"
            for dom in doms:
                g = gold.get(dom, {})
                d = dist.get(dom, {})
                f.write(f"{dom} & {fmt(g.get('em'))} & {fmt(g.get('f1'))} & {fmt(d.get('em'))} & {fmt(d.get('f1'))} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")
            models_done += 1

    if models_done == 0:
        print("[WARN] T3: no domain data available (neither JSON nor items fallback).")

def table_T4_behavior(all_per_run: Dict[str, pd.DataFrame], out_tex: Path):
    rows = []
    for model, df in all_per_run.items():
        dd = df.copy()
        ref = 100*dd.get("refusal", pd.Series([0]*len(dd))).mean()
        fmt_rate = 100*dd.get("format_violation", pd.Series([0]*len(dd))).mean()
        p50 = dd["latency_ms"].median() if "latency_ms" in dd.columns else np.nan
        p90 = dd["latency_ms"].quantile(0.90) if "latency_ms" in dd.columns else np.nan
        rows.append({"model": model, "refusal_pct": ref, "format_pct": fmt_rate, "lat_p50_ms": p50, "lat_p90_ms": p90})
    tbl = pd.DataFrame(rows).set_index("model").loc[sorted(all_per_run.keys())]

    with open(out_tex, "w") as f:
        f.write("\\begin{table}[t]\\centering\n")
        f.write("\\caption{Behavior signals: refusal/format rates and latency.}\n")
        f.write("\\label{tab:behavior}\n")
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Model & Refusal (\\%) & Format (\\%) & p50 (ms) & p90 (ms) \\\\\n\\midrule\n")
        for m, row in tbl.iterrows():
            fmtv = lambda v: "--" if pd.isna(v) else f"{v:.1f}"
            f.write(f"{m} & {fmtv(row['refusal_pct'])} & {fmtv(row['format_pct'])} & {fmtv(row['lat_p50_ms'])} & {fmtv(row['lat_p90_ms'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------- Figures ----------
def fig_F1_drops(all_summaries: pd.DataFrame, out_pdf: Path):
    models = sorted(all_summaries["model"].unique())
    drops = []
    for m in models:
        sdf = all_summaries[all_summaries["model"] == m].set_index("setting")
        try:
            base = sdf.at["gold","f1_median"]
            dv = [
                sdf.at["para","f1_median"] - base,
                sdf.at["dist","f1_median"] - base,
                sdf.at["para_dist","f1_median"] - base,
            ]
        except Exception:
            dv = [np.nan, np.nan, np.nan]
        drops.append(dv)
    drops = np.array(drops)  # % points

    plt.figure(figsize=(7.0, 4.0))
    x = np.arange(len(models))
    w = 0.25
    labels = ["Para", "Dist", "Both"]
    for i in range(3):
        plt.bar(x + (i-1)*w, drops[:,i], width=w, label=labels[i])
    plt.axhline(0, linewidth=1)
    plt.xticks(x, models, rotation=20, ha='right')
    plt.ylabel("ΔF1 vs Gold (pp)")
    plt.title("F1 Drops by Perturbation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

def fig_F1_distribution(items_map: Dict[str, pd.DataFrame], out_pdf: Path):
    plt.figure(figsize=(10.5, 4.5))
    n = len(items_map)
    for i, (model, df) in enumerate(sorted(items_map.items())):
        ax = plt.subplot(1, n, i+1)
        dd = df[df["setting"].isin(SETTINGS)]
        groups = [dd[dd["setting"]==s]["f1_median"].dropna().values for s in SETTINGS]
        # assume f1_median already in %; if not, multiply here
        if max((g.max() if len(g) else 0) for g in groups) <= 1.0:
            groups = [g*100 for g in groups]
        ax.boxplot(groups, showfliers=False)
        ax.set_xticklabels([SETTING_LABEL[s] for s in SETTINGS], rotation=60, ha='right', fontsize=8)
        ax.set_title(model)
        ax.set_ylabel("F1 ( % )" if i==0 else "")
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "gpt4o","gpt4o_mini","gemini_pro","llama31_8b","mistral7b"
    ])
    ap.add_argument("--results-root", default="src/results")      # set to src/results_50 for your new run
    ap.add_argument("--out-root", default="src/report")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_root = Path(args.out_root)
    ensure_out_dirs(out_root)

    all_summaries = []
    per_run_map = {}
    items_map = {}
    per_domain_map = {}

    for m in args.models:
        base = results_root / m / "metrics"
        if not base.exists():
            print(f"[WARN] {base} missing; skipping {m}")
            continue

        sdf = read_summary(base, m)
        items = read_items_csv(base, m)
        perr  = read_per_run_csv(base, m)
        domj  = read_per_domain_json(base, m)

        if sdf is None or items is None or perr is None:
            need = []
            if sdf is None: need.append("summary")
            if items is None: need.append("aggregated_items")
            if perr is None: need.append("per_run")
            print(f"[WARN] Missing {', '.join(need)} for {m}; skipping.")
            continue

        sdf["model"] = m
        # guard: ensure columns exist
        if "em_mean" not in sdf.columns or "f1_median" not in sdf.columns or "setting" not in sdf.columns:
            print(f"[WARN] Summary cols incomplete for {m}; skipping.")
            continue

        all_summaries.append(sdf)
        per_run_map[m] = perr
        items["model"] = m
        items_map[m] = items
        if domj:
            per_domain_map[m] = domj

    if not all_summaries:
        raise SystemExit("No summaries loaded. Abort.")

    all_summaries = pd.concat(all_summaries, ignore_index=True, sort=False)

    # ---------- Tables ----------
    # ---------- Tables ----------
    table_T1_main_accuracy(all_summaries, out_root/"tables"/"T1_main_accuracy.tex")
    table_T2_drops(all_summaries, out_root/"tables"/"T2_drops.tex")
    table_T3_domain(per_domain_map, items_map, out_root/"tables"/"T3_domain_breakdown.tex")  # <— pass items_map too
    table_T4_behavior(per_run_map, out_root/"tables"/"T4_behavior.tex")

    # ---------- Figures ----------
    fig_F1_drops(all_summaries, out_root/"figures"/"F1_drops.pdf")
    fig_F1_distribution(items_map, out_root/"figures"/"F1_distribution.pdf")

    print("✅ Built tables →", out_root/"tables")
    print("✅ Built figures →", out_root/"figures")

if __name__ == "__main__":
    main()