#!/usr/bin/env python3
"""
Builds a dense 'Run-to-Run & Perturbation Stability' table from per_run_v2.csv
for all models under --results-root/<model>/metrics/per_run_v2.csv.

Outputs:
  <out-root>/tables/T_stability.tex
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

SETTINGS = ["gold", "para", "dist", "para_dist"]
SETTING_LABEL = {
    "gold": "Gold",
    "para": "Para",
    "dist": "Dist",
    "para_dist": "Para+Dist",
}

def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def read_per_run_csv(base: Path, model: str) -> pd.DataFrame | None:
    p = _first_existing([
        base / f"{model}_per_run.csv",
        base / "per_run_v2.csv",
    ])
    if not p:
        return None
    df = pd.read_csv(p)
    df.columns = [c.lower() for c in df.columns]
    # Expected minimal columns: run_id, qid, setting, em, f1
    # Optional: latency_ms
    if "setting" not in df.columns:
        # try common fallbacks
        for alt in ["condition", "variant", "config"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "setting"})
                break
    # normalize setting labels
    norm = {
        "gold":"gold",
        "para":"para","paraphrase":"para","gold+paraphrase":"para",
        "dist":"dist","distractor":"dist","gold+distractor":"dist",
        "para_dist":"para_dist","both":"para_dist","gold+paraphrase+distractor":"para_dist",
    }
    df["setting"] = df["setting"].astype(str).str.strip().str.lower().map(lambda s: norm.get(s, s))
    # latency fallback
    if "latency_ms" not in df.columns and "latency" in df.columns:
        df = df.rename(columns={"latency": "latency_ms"})
    # store model name
    df["model"] = model
    return df

def per_model_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute for each model × setting:
      - EM_mean: mean of EM over all rows (items×runs)
      - F1_mean: mean of F1 over all rows (items×runs)
      - EM_agree: macro-average over items of (max run agreement on EM)
      - F1_MAD: macro-average over items of median absolute deviation across runs
      - lat_p50: median of latency_ms over rows
    """
    out_rows = []
    for (model, setting), g in df.groupby(["model", "setting"]):
        # Skip unknown settings
        if setting not in SETTINGS:
            continue

        # Accuracy over runs (micro over rows)
        em_mean = g["em"].mean() * 100.0 if "em" in g.columns else np.nan
        f1_mean = g["f1"].mean() * 100.0 if "f1" in g.columns else np.nan

        # Run-consistency (EM)
        em_agree = np.nan
        if {"qid","run_id","em"}.issubset(g.columns):
            # per-item across runs
            def agree_frac(x):
                # fraction of runs that match the modal EM
                counts = x.value_counts(dropna=False)
                return counts.max() / counts.sum() if counts.sum() > 0 else np.nan
            em_agree = (g.groupby("qid")["em"].apply(agree_frac).mean()) * 100.0

        # F1 MAD (per item across runs → median abs dev, then macro-avg)
        f1_mad = np.nan
        if {"qid","run_id","f1"}.issubset(g.columns):
            def mad(x):
                med = np.median(x.values)
                return np.median(np.abs(x.values - med))
            f1_mad = g.groupby("qid")["f1"].apply(mad).mean() * 100.0  # scale to %

        # Latency p50
        lat_p50 = np.nan
        if "latency_ms" in g.columns:
            lat_p50 = g["latency_ms"].median()

        out_rows.append({
            "model": model,
            "setting": setting,
            "EM_mean": em_mean,
            "F1_mean": f1_mean,
            "EM_agree": em_agree,
            "F1_MAD": f1_mad,
            "lat_p50": lat_p50,
        })
    return pd.DataFrame(out_rows)

def fmt(x, digits=1):
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"

def to_latex_table(stab: pd.DataFrame, out_tex: Path):
    """
    Dense table with blocks per setting. Each cell shows:
      EM / F1 / EM-Agree / F1-MAD / p50
    """
    models = sorted(stab["model"].unique())
    # Build per-setting blocks
    blocks = {}
    for s in SETTINGS:
        sd = stab[stab["setting"] == s].set_index("model")
        rows = []
        for m in models:
            r = sd.loc[m] if m in sd.index else None
            if r is None or isinstance(r, pd.Series) == False:
                cell = "— / — / — / — / —"
            else:
                cell = f"{fmt(r['EM_mean'])} / {fmt(r['F1_mean'])} / {fmt(r['EM_agree'])} / {fmt(r['F1_MAD'])} / {fmt(r['lat_p50'],0)}"
            rows.append((m, cell))
        blocks[s] = rows

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\\centering\n")
        f.write("\\caption{Run-to-Run and Perturbation Stability. "
                "Each cell is EM / F1 / EM-Agree / F1-MAD / p50(ms). "
                "EM-Agree: modal EM agreement across runs (\\%). F1-MAD: median abs. deviation of F1 across runs (pp).}\n")
        f.write("\\label{tab:stability}\n")
        f.write("\\small\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\renewcommand{\\arraystretch}{1.05}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Model & Gold & Para & Dist & Para+Dist \\\\\n\\midrule\n")
        for m in models:
            cells = []
            for s in SETTINGS:
                pairlist = blocks[s]
                # find the cell for model m
                val = next((c for (mm, c) in pairlist if mm == m), "— / — / — / — / —")
                cells.append(val)
            f.write(m + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="src/results_50")
    ap.add_argument("--models", nargs="+", default=["gpt4o","gpt4o_mini","gemini_pro","llama31_8b","mistral7b"])
    ap.add_argument("--out-root", default="src/report_50")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_root = Path(args.out_root)
    (out_root/"tables").mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for m in args.models:
        base = results_root / m / "metrics"
        if not base.exists():
            print(f"[WARN] {base} missing; skip {m}")
            continue
        pr = read_per_run_csv(base, m)
        if pr is None:
            print(f"[WARN] per_run_v2.csv missing for {m}; skip")
            continue
        # minimal sanitation
        need = {"model","setting","qid","run_id","em","f1"}
        if not need.issubset(set(pr.columns)):
            print(f"[WARN] {m} missing required cols for stability; has {sorted(pr.columns)}")
        all_dfs.append(pr)

    if not all_dfs:
        raise SystemExit("No per-run files loaded; abort.")

    per_run = pd.concat(all_dfs, ignore_index=True, sort=False)
    # numeric coercions
    for c in ["em","f1","latency_ms"]:
        if c in per_run.columns:
            per_run[c] = pd.to_numeric(per_run[c], errors="coerce")

    stab = per_model_metrics(per_run)
    to_latex_table(stab, out_root/"tables"/"T_stability.tex")
    print("✅ Wrote:", out_root/"tables"/"T_stability.tex")

if __name__ == "__main__":
    main()