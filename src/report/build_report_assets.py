#!/usr/bin/env python3
"""
Builds publication-ready tables & figures from scoring outputs.

Inputs (already produced by your pipeline):
  src/results/<model>/metrics/<model>_summary.csv
  src/results/<model>/metrics/<model>_aggregated_items.csv
  src/results/<model>/metrics/<model>_per_run.csv
  src/results/<model>/metrics/<model>_per_domain.json           (optional but used for T3)
  src/results/<model>/metrics/<model>_bertscore_summary.csv     (optional)
  src/results/<model>/metrics/<model>_bertscore_aggregated_items.csv (optional)

Outputs:
  Tables → src/report/tables/*.tex
  Figures → src/report/figures/*.pdf
"""

import argparse, json
from pathlib import Path
from typing import Dict, List
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

def read_summary_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # We expect: columns like ['model','setting','n_items','em_mean','f1_median',
    # 'latency_median_ms','drop_paraphrase_pp','drop_distractor_pp','drop_combined_pp', ...]
    # Make it robust to slight naming variations:
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"model"}: colmap[c] = "model"
        elif lc in {"setting"}: colmap[c] = "setting"
        elif "em" in lc and "mean" in lc and "drop" not in lc: colmap[c] = "em_mean"
        elif "f1" in lc and "median" in lc and "drop" not in lc: colmap[c] = "f1_median"
        elif "latency" in lc and ("median" in lc or "p50" in lc): colmap[c] = "latency_p50"
        elif "latency" in lc and "p90" in lc: colmap[c] = "latency_p90"
        elif "drop" in lc and "para_dist" in lc: colmap[c] = "drop_both_pp"
        elif "drop" in lc and "distractor" in lc: colmap[c] = "drop_dist_pp"
        elif "drop" in lc and ("para" in lc and "para_dist" not in lc): colmap[c] = "drop_para_pp"
    df = df.rename(columns=colmap)
    return df

def read_per_run_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # expected: run_id, qid, domain, model, setting, em, f1, latency_ms, refusal, format_violation
    # Make robust:
    df.columns = [c.lower() for c in df.columns]
    return df

def read_items_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # expected: qid, domain, model, setting, em_majority, f1_median, latency_median_ms, stability_abs_dev, ...
    df.columns = [c.lower() for c in df.columns]
    return df

def read_per_domain_json(p: Path) -> Dict:
    if not p.exists(): return {}
    with open(p, "r") as f:
        return json.load(f)

def safe_pct(x): 
    return None if pd.isna(x) else float(x)

def ensure_out_dirs(root: Path):
    (root/"tables").mkdir(parents=True, exist_ok=True)
    (root/"figures").mkdir(parents=True, exist_ok=True)

# ---------- Tables ----------
def table_T1_main_accuracy(all_summaries: pd.DataFrame, out_tex: Path):
    # Pivot to rows=settings, cols=model, values em/f1 per setting per model
    # But the results section plan wants EM/F1 per model PER setting; To keep it compact we produce
    # one table per metric (EM and F1) stacked vertically in a single tabular.
    df = all_summaries.copy()
    # keep only known settings ordering
    df = df[df['setting'].isin(SETTINGS)]
    df['setting_label'] = df['setting'].map(SETTING_LABEL)

    # macro-average per setting across models (mean of model-level aggregates)
    macro = (df.groupby('setting')
               .agg(em_mean=('em_mean','mean'),
                    f1_median=('f1_median','median'))
               .reset_index())
    macro['model'] = 'Macro-Avg'
    macro['setting_label'] = macro['setting'].map(SETTING_LABEL)
    big = pd.concat([df, macro], ignore_index=True, sort=False)

    # Build LaTeX table: rows = setting, cols = Model(EM,F1) pairs? To save space, do two simple tables.
    em_pvt = big.pivot(index='setting_label', columns='model', values='em_mean')
    f1_pvt = big.pivot(index='setting_label', columns='model', values='f1_median')

    with open(out_tex, "w") as f:
        f.write("\\begin{table}[t]\\centering\n")
        f.write("\\caption{Main accuracy by setting and model (EM and F1; \\%).}\n")
        f.write("\\label{tab:main-accuracy}\n")
        f.write("\\begin{tabular}{l" + "r"*(len(em_pvt.columns)) + "}\n\\toprule\n")
        f.write("Setting & " + " & ".join(em_pvt.columns) + " \\\\\n\\midrule\n")
        for idx, row in em_pvt.iterrows():
            vals = ["{:.1f}".format(100*row[c]) if pd.notna(row[c]) and row[c] <= 1.0 else "{:.1f}".format(row[c] if pd.notna(row[c]) else np.nan) for c in em_pvt.columns]
            f.write(f"{idx} (EM) & " + " & ".join(vals) + " \\\\\n")
        for idx, row in f1_pvt.iterrows():
            vals = ["{:.1f}".format(100*row[c]) if pd.notna(row[c]) and row[c] <= 1.0 else "{:.1f}".format(row[c] if pd.notna(row[c]) else np.nan) for c in f1_pvt.columns]
            f.write(f"{idx} (F1) & " + " & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

def table_T2_drops(all_summaries: pd.DataFrame, out_tex: Path):
    # Rows: models; Cols: ΔEM/ΔF1 for Paraphrase, Distractor, Both (pp)
    # From summary we likely have drop_para_pp, drop_dist_pp, drop_both_pp (on F1); EM drops sometimes too.
    # We’ll compute drops from per-setting EM/F1 if drop columns are missing.
    want = []
    for model, sdf in all_summaries.groupby("model"):
        sdf = sdf.set_index("setting")
        row = {"model": model}
        # If drop columns present, prefer them.
        for key in ["drop_para_pp", "drop_dist_pp", "drop_both_pp"]:
            if key in sdf.columns:
                pass
        # Compute F1 drops:
        try:
            base_f1 = sdf.at["gold", "f1_median"]
            row["ΔF1_para_pp"] = 100*(sdf.at["para","f1_median"] - base_f1)
            row["ΔF1_dist_pp"] = 100*(sdf.at["dist","f1_median"] - base_f1)
            row["ΔF1_both_pp"] = 100*(sdf.at["para_dist","f1_median"] - base_f1)
        except Exception:
            row["ΔF1_para_pp"]=row["ΔF1_dist_pp"]=row["ΔF1_both_pp"]=np.nan
        # Compute EM drops:
        try:
            base_em = sdf.at["gold", "em_mean"]
            row["ΔEM_para_pp"] = 100*(sdf.at["para","em_mean"] - base_em)
            row["ΔEM_dist_pp"] = 100*(sdf.at["dist","em_mean"] - base_em)
            row["ΔEM_both_pp"] = 100*(sdf.at["para_dist","em_mean"] - base_em)
        except Exception:
            row["ΔEM_para_pp"]=row["ΔEM_dist_pp"]=row["ΔEM_both_pp"]=np.nan
        want.append(row)
    tbl = pd.DataFrame(want).set_index("model").loc[sorted(set(all_summaries["model"]))]

    with open(out_tex, "w") as f:
        f.write("\\begin{table}[t]\\centering\n")
        f.write("\\caption{Perturbation drops relative to Gold (pp). Negative = improvement, Positive = degradation.}\n")
        f.write("\\label{tab:drops}\n")
        cols = ["ΔEM_para_pp","ΔEM_dist_pp","ΔEM_both_pp","ΔF1_para_pp","ΔF1_dist_pp","ΔF1_both_pp"]
        f.write("\\begin{tabular}{l" + "r"*len(cols) + "}\n\\toprule\n")
        f.write("Model & " + " & ".join(cols) + " \\\\\n\\midrule\n")
        for m, row in tbl.iterrows():
            f.write(m + " & " + " & ".join("{:+.1f}".format(row[c]) if pd.notna(row[c]) else "–" for c in cols) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

def table_T3_domain(per_domain_jsons: Dict[str, Dict], out_tex: Path):
    # Expect per-domain JSON with structure similar to what your domain_scoring writes:
    # { "Gold": {"History":{"em":..,"f1":..}, ...}, "Gold+Distractor": {...} }
    # Build a compact table: rows=domain, cols=EM_gold,F1_gold,EM_dist,F1_dist per model (make one table per model).
    with open(out_tex, "w") as f:
        for model, payload in per_domain_jsons.items():
            gold = payload.get("gold", payload.get("Gold", {}))
            dist = payload.get("dist", payload.get("Gold+Distractor", {}))
            f.write("\\begin{table}[t]\\centering\n")
            f.write(f"\\caption{{Domain breakdown for \\texttt{{{model}}}: Gold vs. Gold+Distractor. Values are percent.}}\n")
            f.write(f"\\label{{tab:domain-{model}}}\n")
            f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
            f.write("Domain & EM$_{Gold}$ & F1$_{Gold}$ & EM$_{Gold+Dist}$ & F1$_{Gold+Dist}$ \\\\\n\\midrule\n")
            for dom in ["History","Science","Politics"]:
                g = gold.get(dom, {})
                d = dist.get(dom, {})
                em_g = g.get("em", None); f1_g = g.get("f1", None)
                em_d = d.get("em", None); f1_d = d.get("f1", None)
                fmt = lambda v: "--" if v is None else f"{v:.1f}"
                f.write(f"{dom} & {fmt(em_g)} & {fmt(f1_g)} & {fmt(em_d)} & {fmt(f1_d)} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

def table_T4_behavior(all_per_run: Dict[str, pd.DataFrame], out_tex: Path):
    # Rows: model; Cols: refusal %, format %, latency p50, latency p90 (across all settings)
    rows = []
    for model, df in all_per_run.items():
        dd = df.copy()
        if "latency_ms" not in dd.columns:
            # try 'latency' fallback
            if "latency" in dd.columns: dd = dd.rename(columns={"latency":"latency_ms"})
        # refusal/format columns may be ints 0/1 or missing
        ref = 100*dd.get("refusal", pd.Series([0]*len(dd))).mean()
        fmt = 100*dd.get("format_violation", pd.Series([0]*len(dd))).mean()
        p50 = dd["latency_ms"].median() if "latency_ms" in dd.columns else np.nan
        p90 = dd["latency_ms"].quantile(0.90) if "latency_ms" in dd.columns else np.nan
        rows.append({"model": model, "refusal_pct": ref, "format_pct": fmt, "lat_p50_ms": p50, "lat_p90_ms": p90})
    tbl = pd.DataFrame(rows).set_index("model").loc[sorted(all_per_run.keys())]

    with open(out_tex, "w") as f:
        f.write("\\begin{table}[t]\\centering\n")
        f.write("\\caption{Behavior signals: refusal/format rates and latency.}\n")
        f.write("\\label{tab:behavior}\n")
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Model & Refusal (\\%) & Format (\\%) & p50 (ms) & p90 (ms) \\\\\n\\midrule\n")
        for m, row in tbl.iterrows():
            fmt = lambda v: "--" if pd.isna(v) else f"{v:.1f}"
            f.write(f"{m} & {fmt(row['refusal_pct'])} & {fmt(row['format_pct'])} & {fmt(row['lat_p50_ms'])} & {fmt(row['lat_p90_ms'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------- Figures ----------
def fig_F1_drops(all_summaries: pd.DataFrame, out_pdf: Path):
    # Bar chart: ΔF1 paraphrase, distractor, both per model
    models = sorted(all_summaries["model"].unique())
    drops = []
    for m in models:
        sdf = all_summaries[all_summaries["model"] == m].set_index("setting")
        try:
            base = sdf.at["gold","f1_median"]
            dv = [
                100*(sdf.at["para","f1_median"] - base),
                100*(sdf.at["dist","f1_median"] - base),
                100*(sdf.at["para_dist","f1_median"] - base),
            ]
        except Exception:
            dv = [np.nan, np.nan, np.nan]
        drops.append(dv)
    drops = np.array(drops)  # shape Mx3

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

def fig_F1_distribution(all_items: Dict[str, pd.DataFrame], out_pdf: Path):
    # Box plots of item-level F1_median by setting per model (stacked horizontally)
    plt.figure(figsize=(10.5, 4.5))
    n = len(all_items)
    for i, (model, df) in enumerate(sorted(all_items.items())):
        ax = plt.subplot(1, n, i+1)
        dd = df[df["setting"].isin(SETTINGS)]
        # df is aggregated_items with column 'f1_median' at item-level
        groups = [dd[dd["setting"]==s]["f1_median"].dropna().values*100 for s in SETTINGS]
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
    ap.add_argument("--results-root", default="src/results")
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
        summ = base / f"{m}_summary.csv"
        items = base / f"{m}_aggregated_items.csv"
        perr = base / f"{m}_per_run.csv"
        domj = base / f"{m}_per_domain.json"

        if not summ.exists() or not items.exists() or not perr.exists():
            print(f"[WARN] Missing metrics for {m}; skipping. Needed: {summ.name}, {items.name}, {perr.name}")
            continue

        sdf = read_summary_csv(summ)
        # carry model id consistently
        sdf["model"] = m
        all_summaries.append(sdf)

        per_run_map[m] = read_per_run_csv(perr)
        items_df = read_items_csv(items)
        items_df["model"] = m
        items_map[m] = items_df

        pdj = read_per_domain_json(domj)
        if pdj:
            per_domain_map[m] = pdj

    if not all_summaries:
        raise SystemExit("No summaries loaded. Abort.")

    all_summaries = pd.concat(all_summaries, ignore_index=True, sort=False)

    # ---------- Tables ----------
    table_T1_main_accuracy(all_summaries, out_root/"tables"/"T1_main_accuracy.tex")
    table_T2_drops(all_summaries, out_root/"tables"/"T2_drops.tex")
    table_T3_domain(per_domain_map, out_root/"tables"/"T3_domain_breakdown.tex")
    table_T4_behavior(per_run_map, out_root/"tables"/"T4_behavior.tex")

    # ---------- Figures ----------
    fig_F1_drops(all_summaries, out_root/"figures"/"F1_drops.pdf")
    fig_F1_distribution({k:v for k,v in items_map.items()}, out_root/"figures"/"F1_distribution.pdf")

    print("✅ Built tables →", out_root/"tables")
    print("✅ Built figures →", out_root/"figures")

if __name__ == "__main__":
    main()