# src/report/write_results_tex.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np


# ----- Display name mapping (ids -> pretty) -----
DISPLAY_NAME = {
    "gemini_pro": "Gemini Pro",
    "gpt4o_mini": "GPT-4o Mini",
    "gpt4o": "GPT-4o",
    "llama31_8b": "LLaMA-3.1-8B",
    "mistral7b": "Mistral-7B",
}

SETTINGS = ["Gold", "Para", "Dist", "Para+Dist"]
DOMAINS_ORDER = ["History", "Literature", "Politics", "Geography", "Science"]


def show_name(model: str) -> str:
    """Map possible model ids to display names."""
    m = str(model).strip()
    return DISPLAY_NAME.get(m, DISPLAY_NAME.get(m.lower(), m))


def fmt(x: float, dp: int) -> str:
    if pd.isna(x):
        return "-"
    return f"{float(x):.{dp}f}"


def must_read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing table: {p}")
    return pd.read_csv(p)


def pick_row(df: pd.DataFrame, model_display: str) -> pd.Series:
    """Pick a model row by display name or id (robust)."""
    # exact display match
    hit = df[df["Model"].apply(show_name) == model_display]
    if len(hit):
        return hit.iloc[0]
    # fallback by id (case-insensitive)
    hit = df[df["Model"].astype(str).str.lower() == model_display.lower()]
    if len(hit):
        return hit.iloc[0]
    # fallback by display literal already in table
    hit = df[df["Model"].astype(str) == model_display]
    if len(hit):
        return hit.iloc[0]
    raise KeyError(f"Model '{model_display}' not found in table. Available: {df['Model'].tolist()}")


# =========================
# Table 1: Per-setting (EM/F1/BERT)
# =========================
def table1_tex(t1: pd.DataFrame, order: List[str], em_dp: int, f1_dp: int, bert_dp: int) -> str:
    # Expect columns like "Gold_EM","Gold_F1","Gold_BERT", etc.
    for s in SETTINGS:
        for m in ["EM", "F1", "BERT"]:
            col = f"{s}_{m}"
            if col not in t1.columns:
                raise KeyError(f"table1_per_setting.csv is missing column '{col}'")

    header = r"""
\subsection{Per-setting Accuracy (EM / F1 / BERT)}
\begin{table}[H]\centering
\caption{Per-setting performance. Each cell is \textbf{EM / F1 / BERTScore F1 (median)}.}
\label{tab:mega-per-setting}
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.05}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l ccc ccc ccc ccc}
\toprule
& \multicolumn{3}{c}{Gold} & \multicolumn{3}{c}{Para} & \multicolumn{3}{c}{Dist} & \multicolumn{3}{c}{Para+Dist} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}
Model & EM & F1 & BERT & EM & F1 & BERT & EM & F1 & BERT & EM & F1 & BERT \\
\midrule
""".strip("\n")

    lines = [header]
    for name in order:
        r = pick_row(t1, name)
        vals: List[str] = []
        for s in SETTINGS:
            vals += [
                fmt(r[f"{s}_EM"], em_dp),
                fmt(r[f"{s}_F1"], f1_dp),
                fmt(r[f"{s}_BERT"], bert_dp),
            ]
        lines.append(f"{show_name(r['Model'])} \\\\")
        lines.append("& " + " & ".join(vals) + r" \\")
    tail = r"""
\bottomrule
\end{tabular}%
}
\caption*{\scriptsize
\textbf{What this shows.} Side-by-side accuracy per model for each setting.\;
\textbf{How computed.} EM = \emph{mean across items} of exact span match (after SQuAD-style normalization). 
F1 = \emph{median across items} of token-level precision/recall F1 (computed per item, max over golds).
BERTScore F1 = \emph{median across items} of contextual similarity (computed on original strings, max over golds).
All values are in \%.}
\par\medskip
\caption*{\scriptsize \textbf{Remark.} \textit{LLaMA-3.1-8B often returned full sentences instead of short spans, yielding 0\% EM despite F1≈55–58; strict span-EM penalizes formatting rather than content.}}
\end{table}
""".strip("\n")
    lines.append(tail)
    return "\n".join(lines)


# =========================
# Table 2: Δ vs Gold + latency
# =========================
def table2_tex(t2: pd.DataFrame, order: List[str], pp_dp: int, ms_dp: int) -> str:
    required = ["ΔEM_Para", "ΔEM_Dist", "ΔEM_Both",
                "ΔF1_Para", "ΔF1_Dist", "ΔF1_Both",
                "p50_ms", "p90_ms"]
    for c in required:
        if c not in t2.columns:
            raise KeyError(f"table2_robust_latency.csv missing column '{c}'")

    header = r"""
\subsection{Robustness and Latency}
\begin{table}[H]\centering
\caption{Robustness and behavior. Drops are relative to Gold (percentage points).}
\label{tab:robustness-behavior}
\scriptsize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{l rrr rrr rr}
\toprule
& \multicolumn{3}{c}{\(\Delta\)EM (pp)} & \multicolumn{3}{c}{\(\Delta\)F1 (pp)} & \multicolumn{2}{c}{Latency (ms)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}
Model & Para & Dist & Para+Dist & Para & Dist & Para+Dist & p50 & p90 \\
\midrule
""".strip("\n")

    lines = [header]
    for name in order:
        r = pick_row(t2, name)
        vals = [
            fmt(r["ΔEM_Para"], pp_dp), fmt(r["ΔEM_Dist"], pp_dp), fmt(r["ΔEM_Both"], pp_dp),
            fmt(r["ΔF1_Para"], pp_dp), fmt(r["ΔF1_Dist"], pp_dp), fmt(r["ΔF1_Both"], pp_dp),
            fmt(r["p50_ms"], ms_dp), fmt(r["p90_ms"], ms_dp),
        ]
        lines.append(f"{show_name(r['Model'])} & " + " & ".join(vals) + r" \\")
    tail = r"""
\bottomrule
\end{tabular}
\caption*{\scriptsize
\textbf{What this shows.} Sensitivity to paraphrases/distractors and speed.\;
\textbf{How computed.} For each model, $\Delta$EM/F1 are \emph{percentage-point} differences vs. Gold for that model. Latency p50/p90 are medians/90th percentiles over responses (ms).}
\end{table}
""".strip("\n")
    lines.append(tail)
    return "\n".join(lines)


# =========================
# Table 3: Stability
# =========================
def table3_tex(t3: pd.DataFrame, order: List[str], ms_dp: int) -> str:
    # Robust header detection
    emagree_cols = ["EM-Agree (%)", "EM-Agree_%", "EM_Agree_%"]
    f1mad_cols = ["F1-MAD (pp)", "F1-MAD_pp", "F1_MAD_pp"]
    p50_cols = ["Latency p50 (ms)", "Latency_p50_ms", "p50_ms"]

    def col(df, cands):
        for c in cands:
            if c in df.columns: return c
        raise KeyError(f"table3_stability.csv missing one of {cands}")

    c_em = col(t3, emagree_cols)
    c_mad = col(t3, f1mad_cols)
    c_p50 = col(t3, p50_cols)

    header = r"""
\subsection{Run-to-Run Stability (Aggregated)}
\begin{table}[H]\centering
\caption{Run-to-run stability aggregated across settings.}
\label{tab:stability}
\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{lccc}
\toprule
Model & EM-Agree (\%) & F1-MAD (pp) & Latency p50 (ms) \\
\midrule
""".strip("\n")

    lines = [header]
    for name in order:
        r = pick_row(t3, name)
        vals = [fmt(r[c_em], 1), fmt(r[c_mad], 1), fmt(r[c_p50], ms_dp)]
        lines.append(f"{show_name(r['Model'])} & " + " & ".join(vals) + r" \\")
    tail = r"""
\bottomrule
\end{tabular}
\caption*{\scriptsize
\textbf{What this shows.} Stability across $n=3$ repeated runs.\;
\textbf{How computed.} EM-Agree = average modal-agreement rate per item across runs. 
F1-MAD = median absolute deviation of per-item F1 across runs (pp). 
Latency p50 = overall median latency across all responses for the model (ms).}
\end{table}
""".strip("\n")
    lines.append(tail)
    return "\n".join(lines)


# =========================
# Table 4: Leaderboard
# =========================
def table4_tex(t4: pd.DataFrame, order: List[str], f1_dp: int, pp_dp: int, ms_dp: int) -> str:
    req = ["Gold F1", "Avg F1", "Avg |ΔF1| (pp)", "p50 (ms)", "Efficiency (F1/s)", "Composite ↑"]
    for c in req:
        if c not in t4.columns:
            raise KeyError(f"table4_leaderboard.csv missing column '{c}'")

    header = r"""
\subsection{Composite Leaderboard}
\begin{table}[H]\centering
\caption{Leaderboard combining accuracy, robustness, and speed.}
\label{tab:leaderboard}
\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular}{lrrrrrr}
\toprule
Model & Gold F1 & Avg F1 & Avg $|\Delta$F1| (pp) & p50 (ms) & Efficiency (F1/s) & Composite $\uparrow$ \\
\midrule
""".strip("\n")

    lines = [header]
    # Keep the order provided (even if t4 is sorted by Composite in csv)
    order_rows = []
    name_to_row = {show_name(r["Model"]): r for _, r in t4.iterrows()}
    for name in order:
        disp = show_name(name)
        if disp in name_to_row:
            order_rows.append(name_to_row[disp])
        else:
            # fall back by id
            r2 = t4[t4["Model"].astype(str).str.lower() == str(name).lower()]
            if len(r2):
                order_rows.append(r2.iloc[0])

    for r in order_rows:
        vals = [
            fmt(r["Gold F1"], f1_dp),
            fmt(r["Avg F1"], f1_dp),
            fmt(r["Avg |ΔF1| (pp)"], pp_dp),
            fmt(r["p50 (ms)"], ms_dp),
            fmt(r["Efficiency (F1/s)"], 1),
            r"\textbf{" + fmt(r["Composite ↑"], 1) + "}",
        ]
        lines.append(f"{show_name(r['Model'])} & " + " & ".join(vals) + r" \\")
    tail = r"""
\bottomrule
\end{tabular}
\caption*{\scriptsize
\textbf{What this shows.} A single ranking balancing accuracy, robustness, and speed.\;
\textbf{How computed.} Avg F1 = mean over \{Gold, Para, Dist, Para+Dist\}. 
Avg\,$|\Delta$F1| = mean \emph{absolute} drop vs.\ Gold over \{Para, Dist, Para+Dist\}. 
Efficiency = Gold F1 / (p50 latency in sec). 
Composite = Avg F1 $- \tfrac{1}{2}$\,Avg\,$|\Delta$F1|.}
\end{table}
""".strip("\n")
    lines.append(tail)
    return "\n".join(lines)


# =========================
# Table 5: Domain breakdown (Gold only)
# =========================
def table5_tex(t5: pd.DataFrame, order: List[str], em_dp: int, f1_dp: int) -> str:
    # Identify domains present (intersect with desired order)
    all_domains = []
    for c in t5.columns:
        if c.endswith("_EM"):
            all_domains.append(c[:-3])
    domains_present = [d for d in DOMAINS_ORDER if d in all_domains]
    if not domains_present:
        # fall back to any found
        domains_present = sorted(all_domains)

    # verify F1 columns exist
    for d in domains_present:
        for suffix in ["_EM", "_F1"]:
            if f"{d}{suffix}" not in t5.columns:
                raise KeyError(f"table5_domain_gold.csv missing column '{d}{suffix}'")

    # header
    head_domains = " & ".join([fr"\multicolumn{{2}}{{c}}{{{d}}}" for d in domains_present])
    cmids = []
    col_index = 2
    for _ in domains_present:
        cmids.append(fr"\cmidrule(lr){{{col_index}-{col_index+1}}}")
        col_index += 2
    cmid_str = "".join(cmids)

    header = rf"""
\subsection{{Domain Breakdown (Gold Only)}}
\begin{{table}}[H]\centering
\caption{{Domain breakdown (EM / F1; \%) on \textbf{{Gold}}.}}
\label{{tab:domain-breakdown}}
\scriptsize
\setlength{{\tabcolsep}}{{3.8pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l {'cc ' * len(domains_present)} }}
\toprule
& {head_domains} \\
{cmid_str}
Model & {' & '.join(['EM & F1' for _ in domains_present])} \\
\midrule
""".strip("\n")

    lines = [header]
    # normal rows
    for name in order:
        r = pick_row(t5, name)
        vals = []
        for d in domains_present:
            vals.append(fmt(r[f"{d}_EM"], em_dp))
            vals.append(fmt(r[f"{d}_F1"], f1_dp))
        lines.append(f"{show_name(r['Model'])} & " + " & ".join(vals) + r" \\")
    # Avg row if present
    avg = t5[t5["Model"].astype(str).str.strip().str.lower() == "avg."]
    if len(avg):
        r = avg.iloc[0]
        vals = []
        for d in domains_present:
            vals.append(fmt(r[f"{d}_EM"], em_dp))
            vals.append(fmt(r[f"{d}_F1"], f1_dp))
        lines.append(r"\midrule")
        lines.append("Avg. & " + " & ".join(vals) + r" \\")
    tail = r"""
\bottomrule
\end{tabular}%
}
\caption*{\scriptsize
\textbf{What this shows.} Which domains are easier/harder under \emph{Gold} (no perturbations).\;
\textbf{How computed.} Items grouped by domain; 
EM = \emph{mean of per-item EM} (per-item EM = mean across 3 runs). 
F1 = \emph{median of per-item F1} (per-item F1 = median across 3 runs). 
Both scaled to \%.}
\end{table}
""".strip("\n")
    lines.append(tail)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", required=True)
    ap.add_argument("--figs-dir", required=False)  # ignored (no figures here)
    ap.add_argument("--out-tex", required=True)
    ap.add_argument("--model-order", required=True,
                    help="Comma-separated display names (e.g., 'Gemini Pro,GPT-4o Mini,...')")
    ap.add_argument("--em-dp", type=int, default=1)
    ap.add_argument("--f1-dp", type=int, default=1)
    ap.add_argument("--bert-dp", type=int, default=6)
    ap.add_argument("--pp-dp", type=int, default=1)
    ap.add_argument("--ms-dp", type=int, default=1)
    args = ap.parse_args()

    tables_dir = Path(args.tables_dir)
    out_tex = Path(args.out_tex)
    order = [s.strip() for s in args.model_order.split(",") if s.strip()]

    # read tables
    t1 = must_read_csv(tables_dir / "table1_per_setting.csv")
    t2 = must_read_csv(tables_dir / "table2_robust_latency.csv")
    t3 = must_read_csv(tables_dir / "table3_stability.csv")
    t4 = must_read_csv(tables_dir / "table4_leaderboard.csv")
    t5 = must_read_csv(tables_dir / "table5_domain_gold.csv")

    # build sections
    parts = []
    parts.append(table1_tex(t1, order, args.em_dp, args.f1_dp, args.bert_dp))
    parts.append(table2_tex(t2, order, args.pp_dp, args.ms_dp))
    parts.append(table3_tex(t3, order, args.ms_dp))
    parts.append(table4_tex(t4, order, args.f1_dp, args.pp_dp, args.ms_dp))
    parts.append(table5_tex(t5, order, args.em_dp, args.f1_dp))

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n\n".join(parts))
    print(f"✅ Wrote LaTeX tables to: {out_tex}")


if __name__ == "__main__":
    main()