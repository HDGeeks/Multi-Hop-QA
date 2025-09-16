# src/report/write_results_tex.py
"""
write_results_tex.py
====================

Render LaTeX table sections for the paper from the CSV tables produced by
`build_from_metrics.py`.

Inputs (CSV files in --tables-dir, created earlier by build_from_metrics.py)
----------------------------------------------------------------------------
- table1_per_setting.csv
    Wide per-model table containing per-setting metrics (percent):
      columns: Model,
               Gold_EM, Gold_F1, Gold_BERT,
               Para_EM, Para_F1, Para_BERT,
               Dist_EM, Dist_F1, Dist_BERT,
               Para+Dist_EM, Para+Dist_F1, Para+Dist_BERT

- table2_robust_latency.csv
    Per-model deltas vs. Gold (pp) and latency p50/p90 (ms):
      columns (at minimum): Model,
          ΔEM_Para, ΔEM_Dist, ΔEM_Both,
          ΔF1_Para, ΔF1_Dist, ΔF1_Both,
          p50_ms, p90_ms

- table3_stability.csv
    Run-to-run stability and latency p50 (ms):
      columns include (with robust name detection):
          Model, "EM-Agree (%)", "F1-MAD (pp)", "Latency p50 (ms)"

- table4_leaderboard.csv
    Composite leaderboard metrics:
      columns: Model, "Gold F1", "Avg F1", "Avg |ΔF1| (pp)", "p50 (ms)",
               "Efficiency (F1/s)", "Composite ↑"

- table5_domain_gold.csv
    Domain breakdown for Gold only (percent):
      columns: Model, <Domain>_EM, <Domain>_F1, ... for the domains available

Outputs
-------
- A single LaTeX file constructed by concatenating the 5 tables:
  The file is written to --out-tex.

Usage
-----
$ python -m src.report.write_results_tex \
    --tables-dir src/reports/tables \
    --out-tex src/reports/tables/paper_tables.tex \
    --model-order "Gemini Pro,GPT-4o Mini,GPT-4o,Mistral-7B,LLaMA-3.1-8B" \
    --em-dp 1 --f1-dp 1 --bert-dp 6 --pp-dp 1 --ms-dp 1

Key design choices
------------------
- Model display names are mapped via DISPLAY_NAME, and all user-facing strings
  used inside LaTeX are escaped to avoid compilation issues.
- Numeric formatting is centralized in `fmt` (NaN -> "-" placeholder).
- The code is defensive with descriptive errors (e.g., missing columns).
- Table 1 row emission fixed: each row is now a *single* LaTeX line
  of the form `Model & ... \\` (previous version accidentally split rows).

Note
----
This module does not generate figures; it only assembles LaTeX for tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Sequence

import numpy as np
import pandas as pd


# ----- Display name mapping (ids -> pretty) -----
DISPLAY_NAME: Dict[str, str] = {
    "gemini_pro": "Gemini Pro",
    "gpt4o_mini": "GPT-4o Mini",
    "gpt4o": "GPT-4o",
    "llama31_8b": "LLaMA-3.1-8B",
    "mistral7b": "Mistral-7B",
}

# Canonical setting labels as used by upstream tables
SETTINGS: List[str] = ["Gold", "Para", "Dist", "Para+Dist"]

# Preferred order if present in Table 5 (domain breakdown)
DOMAINS_ORDER: List[str] = ["History", "Literature", "Politics", "Geography", "Science"]


# =========================
# Utilities
# =========================

def latex_escape(s: str) -> str:
    """
    Escape LaTeX-special characters in text fields (model names, etc.).
    This prevents common compilation failures.

    Parameters
    ----------
    s : str
        Raw string to be inserted into LaTeX tables.

    Returns
    -------
    str
        Escaped string safe for LaTeX.
    """
    if s is None:
        return ""
    # Order matters; apply backslash first to avoid double-escaping
    repl = [
        ("\\", r"\textbackslash{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("$", r"\$"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    out = str(s)
    for a, b in repl:
        out = out.replace(a, b)
    return out


def show_name(model: str) -> str:
    """
    Map (case-insensitive) model ids to a human-readable display name.

    Falls back to the original string if no mapping is found.

    Parameters
    ----------
    model : str
        Model id as stored in tables (e.g., "llama31_8b").

    Returns
    -------
    str
        Display name (e.g., "LLaMA-3.1-8B").
    """
    m = str(model).strip()
    return DISPLAY_NAME.get(m, DISPLAY_NAME.get(m.lower(), m))


def fmt(x: float, dp: int) -> str:
    """
    Format a floating-point number to the requested decimals, or '-' if NaN.

    Parameters
    ----------
    x : float
        Value to format.
    dp : int
        Decimal places.

    Returns
    -------
    str
        Formatted string or "-".
    """
    if pd.isna(x):
        return "-"
    try:
        return f"{float(x):.{dp}f}"
    except Exception:
        return "-"


def must_read_csv(p: Path) -> pd.DataFrame:
    """
    Read a CSV file or raise a clear FileNotFoundError.

    Parameters
    ----------
    p : Path
        CSV path.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    if not p.exists():
        raise FileNotFoundError(f"Missing table: {p}")
    return pd.read_csv(p)


def pick_row(df: pd.DataFrame, model_display: str) -> pd.Series:
    """
    Pick a model row by display name or id (robust).

    The function tries (in order):
      1) Match by normalized DISPLAY_NAME(show_name(.)) of df["Model"].
      2) Case-insensitive match on raw df["Model"] value.
      3) Exact match on df["Model"] value.

    Parameters
    ----------
    df : pd.DataFrame
        Table dataframe with a 'Model' column.
    model_display : str
        Desired model name as passed in --model-order (e.g., "GPT-4o Mini").

    Returns
    -------
    pd.Series
        Matching row.

    Raises
    ------
    KeyError
        If no matching row is found.
    """
    if "Model" not in df.columns:
        raise KeyError("Input table is missing required 'Model' column.")

    # 1) match by display mapping
    hit = df[df["Model"].apply(show_name) == model_display]
    if len(hit):
        return hit.iloc[0]

    # 2) case-insensitive match on raw id/display
    hit = df[df["Model"].astype(str).str.lower() == model_display.lower()]
    if len(hit):
        return hit.iloc[0]

    # 3) exact literal fallback
    hit = df[df["Model"].astype(str) == model_display]
    if len(hit):
        return hit.iloc[0]

    avail = [show_name(m) for m in df["Model"].astype(str).tolist()]
    raise KeyError(f"Model '{model_display}' not found in table. Available: {avail}")


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
        lines.append(f"{show_name(r['Model'])} & " + " & ".join(vals) + r" \\")
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
\caption*{\scriptsize \textbf{Remark.} 
\textit{BERTScore values saturate near 100\%; we retain up to 6 decimal places to highlight small but consistent differences across models. 
Some models occasionally return sentences or HTML-wrapped spans; EM is strict span match and may understate semantic correctness compared to F1/BERT.}}
\end{table}
""".strip("\n")
    lines.append(tail)
    return "\n".join(lines)

# =========================
# Table 2: Δ vs Gold + latency
# =========================

def table2_tex(t2: pd.DataFrame, order: List[str], pp_dp: int, ms_dp: int) -> str:
    """
    Render LaTeX for Table 2 (Drops vs Gold and latency).

    Required columns in `t2`:
      ["ΔEM_Para", "ΔEM_Dist", "ΔEM_Both",
       "ΔF1_Para", "ΔF1_Dist", "ΔF1_Both",
       "p50_ms", "p90_ms"]

    Parameters
    ----------
    t2 : pd.DataFrame
        Robustness/latency table.
    order : list[str]
        Model display order to render.
    pp_dp : int
        Decimal places for percentage-point deltas.
    ms_dp : int
        Decimal places for latency (ms).

    Returns
    -------
    str
        LaTeX block for the table.
    """
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

    lines: List[str] = [header]
    for name in order:
        r = pick_row(t2, name)
        vals = [
            fmt(r["ΔEM_Para"], pp_dp), fmt(r["ΔEM_Dist"], pp_dp), fmt(r["ΔEM_Both"], pp_dp),
            fmt(r["ΔF1_Para"], pp_dp), fmt(r["ΔF1_Dist"], pp_dp), fmt(r["ΔF1_Both"], pp_dp),
            fmt(r["p50_ms"], ms_dp), fmt(r["p90_ms"], ms_dp),
        ]
        row = latex_escape(show_name(r["Model"])) + " & " + " & ".join(vals) + r" \\"
        lines.append(row)

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
    """
    Render LaTeX for Table 3 (Run-to-run stability).

    The function detects column names robustly from candidates:
      - EM-Agree: one of ["EM-Agree (%)", "EM-Agree_%", "EM_Agree_%"]
      - F1-MAD  : one of ["F1-MAD (pp)", "F1-MAD_pp", "F1_MAD_pp"]
      - p50     : one of ["Latency p50 (ms)", "Latency_p50_ms", "p50_ms"]

    Parameters
    ----------
    t3 : pd.DataFrame
        Stability table.
    order : list[str]
        Model display order to render.
    ms_dp : int
        Decimal places for latency (ms).

    Returns
    -------
    str
        LaTeX block for the table.
    """
    emagree_cols = ["EM-Agree (%)", "EM-Agree_%", "EM_Agree_%"]
    f1mad_cols = ["F1-MAD (pp)", "F1-MAD_pp", "F1_MAD_pp"]
    p50_cols = ["Latency p50 (ms)", "Latency_p50_ms", "p50_ms"]

    def col(df: pd.DataFrame, cands: Sequence[str]) -> str:
        for c in cands:
            if c in df.columns:
                return c
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

    lines: List[str] = [header]
    for name in order:
        r = pick_row(t3, name)
        vals = [fmt(r[c_em], 1), fmt(r[c_mad], 1), fmt(r[c_p50], ms_dp)]
        row = latex_escape(show_name(r["Model"])) + " & " + " & ".join(vals) + r" \\"
        lines.append(row)

    tail = r"""
\bottomrule
\end{tabular}
\caption*{\scriptsize
\textbf{What this shows.} Stability across repeated runs.\;
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
    """
    Render LaTeX for Table 4 (Composite Leaderboard).

    Required columns in `t4`:
      ["Gold F1", "Avg F1", "Avg |ΔF1| (pp)", "p50 (ms)",
       "Efficiency (F1/s)", "Composite ↑"]

    Parameters
    ----------
    t4 : pd.DataFrame
        Leaderboard table.
    order : list[str]
        Model display order to render.
    f1_dp : int
        Decimals for F1 (%).
    pp_dp : int
        Decimals for percentage-point deltas.
    ms_dp : int
        Decimals for latency (ms).

    Returns
    -------
    str
        LaTeX block for the table.
    """
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

    lines: List[str] = [header]

    # preserve requested order where possible
    name_to_row = {show_name(r["Model"]): r for _, r in t4.iterrows()}
    order_rows = []
    for name in order:
        disp = show_name(name)
        if disp in name_to_row:
            order_rows.append(name_to_row[disp])
        else:
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
        row = latex_escape(show_name(r["Model"])) + " & " + " & ".join(vals) + r" \\"
        lines.append(row)

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
    """
    Render LaTeX for Table 5 (Gold-only domain breakdown).

    Expected columns in `t5`:
      - "Model"
      - Paired domain columns: <Domain>_EM, <Domain>_F1, for multiple domains.

    The function will use DOMAINS_ORDER if present; otherwise it falls back to
    the set of detected domains sorted alphabetically.

    Parameters
    ----------
    t5 : pd.DataFrame
        Domain breakdown table (wide).
    order : list[str]
        Model display order to render.
    em_dp : int
        Decimals for EM.
    f1_dp : int
        Decimals for F1.

    Returns
    -------
    str
        LaTeX block for the table.
    """
    # Identify domains present (intersect with desired order)
    all_domains: List[str] = []
    for c in t5.columns:
        if c.endswith("_EM"):
            all_domains.append(c[:-3])
    domains_present = [d for d in DOMAINS_ORDER if d in all_domains]
    if not domains_present:
        # fall back to any found
        domains_present = sorted(all_domains)
    if not domains_present:
        raise KeyError("table5_domain_gold.csv contains no <Domain>_EM columns.")

    # verify F1 columns exist
    for d in domains_present:
        for suffix in ["_EM", "_F1"]:
            col = f"{d}{suffix}"
            if col not in t5.columns:
                raise KeyError(f"table5_domain_gold.csv missing column '{col}'")

    # header
    head_domains = " & ".join([fr"\multicolumn{{2}}{{c}}{{{latex_escape(d)}}}" for d in domains_present])
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

    lines: List[str] = [header]
    # normal model rows
    for name in order:
        r = pick_row(t5, name)
        vals: List[str] = []
        for d in domains_present:
            vals.append(fmt(r[f"{d}_EM"], em_dp))
            vals.append(fmt(r[f"{d}_F1"], f1_dp))
        row = latex_escape(show_name(r["Model"])) + " & " + " & ".join(vals) + r" \\"
        lines.append(row)

    # optional Avg. row
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
EM = \emph{mean of per-item EM} (per-item EM = mean across runs). 
F1 = \emph{median of per-item F1} (per-item F1 = median across runs). 
Both scaled to \%.}
\end{table}
""".strip("\n")
    lines.append(tail)
    return "\n".join(lines)


# =========================
# Orchestration
# =========================

def main() -> None:
    """
    CLI entry point.

    Reads the 5 CSV tables from --tables-dir, renders LaTeX blocks for each,
    and writes the concatenated content to --out-tex. The model order for
    rows is controlled by --model-order (comma-separated display names).
    """
    ap = argparse.ArgumentParser(description="Write LaTeX tables from prebuilt CSV metrics tables.")
    ap.add_argument("--tables-dir", required=True, help="Directory containing table*.csv files")
    ap.add_argument("--figs-dir", required=False, help="(Unused) figures directory parameter for symmetry")
    ap.add_argument("--out-tex", required=True, help="Path to write the combined LaTeX")
    ap.add_argument("--model-order", required=True,
                    help="Comma-separated display names (e.g., 'Gemini Pro,GPT-4o Mini,GPT-4o,Mistral-7B,LLaMA-3.1-8B')")
    ap.add_argument("--em-dp", type=int, default=1)
    ap.add_argument("--f1-dp", type=int, default=1)
    ap.add_argument("--bert-dp", type=int, default=6)
    ap.add_argument("--pp-dp", type=int, default=1)
    ap.add_argument("--ms-dp", type=int, default=1)
    args = ap.parse_args()

    tables_dir = Path(args.tables_dir)
    out_tex = Path(args.out_tex)
    order: List[str] = [s.strip() for s in args.model_order.split(",") if s.strip()]

    # Read tables (explicit failures are helpful during paper builds)
    t1 = must_read_csv(tables_dir / "table1_per_setting.csv")
    t2 = must_read_csv(tables_dir / "table2_robust_latency.csv")
    t3 = must_read_csv(tables_dir / "table3_stability.csv")
    t4 = must_read_csv(tables_dir / "table4_leaderboard.csv")
    t5 = must_read_csv(tables_dir / "table5_domain_gold.csv")

    # Build sections
    parts: List[str] = []
    parts.append(table1_tex(t1, order, args.em_dP if hasattr(args, 'em_dP') else args.em_dp,
                            args.f1_dP if hasattr(args, 'f1_dP') else args.f1_dp,
                            args.bert_dP if hasattr(args, 'bert_dP') else args.bert_dp))
    parts.append(table2_tex(t2, order, args.pp_dp, args.ms_dp))
    parts.append(table3_tex(t3, order, args.ms_dp))
    parts.append(table4_tex(t4, order, args.f1_dp, args.pp_dp, args.ms_dp))
    parts.append(table5_tex(t5, order, args.em_dp, args.f1_dp))

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"✅ Wrote LaTeX tables to: {out_tex}")


if __name__ == "__main__":
    main()

# # src/report/write_results_tex.py
# from __future__ import annotations
# import argparse
# from pathlib import Path
# from typing import List, Dict
# import pandas as pd
# import numpy as np


# # ----- Display name mapping (ids -> pretty) -----
# DISPLAY_NAME = {
#     "gemini_pro": "Gemini Pro",
#     "gpt4o_mini": "GPT-4o Mini",
#     "gpt4o": "GPT-4o",
#     "llama31_8b": "LLaMA-3.1-8B",
#     "mistral7b": "Mistral-7B",
# }

# SETTINGS = ["Gold", "Para", "Dist", "Para+Dist"]
# DOMAINS_ORDER = ["History", "Literature", "Politics", "Geography", "Science"]


# def show_name(model: str) -> str:
#     """Map possible model ids to display names."""
#     m = str(model).strip()
#     return DISPLAY_NAME.get(m, DISPLAY_NAME.get(m.lower(), m))


# def fmt(x: float, dp: int) -> str:
#     if pd.isna(x):
#         return "-"
#     return f"{float(x):.{dp}f}"


# def must_read_csv(p: Path) -> pd.DataFrame:
#     if not p.exists():
#         raise FileNotFoundError(f"Missing table: {p}")
#     return pd.read_csv(p)


# def pick_row(df: pd.DataFrame, model_display: str) -> pd.Series:
#     """Pick a model row by display name or id (robust)."""
#     # exact display match
#     hit = df[df["Model"].apply(show_name) == model_display]
#     if len(hit):
#         return hit.iloc[0]
#     # fallback by id (case-insensitive)
#     hit = df[df["Model"].astype(str).str.lower() == model_display.lower()]
#     if len(hit):
#         return hit.iloc[0]
#     # fallback by display literal already in table
#     hit = df[df["Model"].astype(str) == model_display]
#     if len(hit):
#         return hit.iloc[0]
#     raise KeyError(f"Model '{model_display}' not found in table. Available: {df['Model'].tolist()}")


# # =========================
# # Table 1: Per-setting (EM/F1/BERT)
# # =========================
# def table1_tex(t1: pd.DataFrame, order: List[str], em_dp: int, f1_dp: int, bert_dp: int) -> str:
#     # Expect columns like "Gold_EM","Gold_F1","Gold_BERT", etc.
#     for s in SETTINGS:
#         for m in ["EM", "F1", "BERT"]:
#             col = f"{s}_{m}"
#             if col not in t1.columns:
#                 raise KeyError(f"table1_per_setting.csv is missing column '{col}'")

#     header = r"""
# \subsection{Per-setting Accuracy (EM / F1 / BERT)}
# \begin{table}[H]\centering
# \caption{Per-setting performance. Each cell is \textbf{EM / F1 / BERTScore F1 (median)}.}
# \label{tab:mega-per-setting}
# \scriptsize
# \setlength{\tabcolsep}{3.5pt}
# \renewcommand{\arraystretch}{1.05}
# \resizebox{\textwidth}{!}{%
# \begin{tabular}{l ccc ccc ccc ccc}
# \toprule
# & \multicolumn{3}{c}{Gold} & \multicolumn{3}{c}{Para} & \multicolumn{3}{c}{Dist} & \multicolumn{3}{c}{Para+Dist} \\
# \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}
# Model & EM & F1 & BERT & EM & F1 & BERT & EM & F1 & BERT & EM & F1 & BERT \\
# \midrule
# """.strip("\n")

#     lines = [header]
#     for name in order:
#         r = pick_row(t1, name)
#         vals: List[str] = []
#         for s in SETTINGS:
#             vals += [
#                 fmt(r[f"{s}_EM"], em_dp),
#                 fmt(r[f"{s}_F1"], f1_dp),
#                 fmt(r[f"{s}_BERT"], bert_dp),
#             ]
#         lines.append(f"{show_name(r['Model'])} \\\\")
#         lines.append("& " + " & ".join(vals) + r" \\")
#     tail = r"""
# \bottomrule
# \end{tabular}%
# }
# \caption*{\scriptsize
# \textbf{What this shows.} Side-by-side accuracy per model for each setting.\;
# \textbf{How computed.} EM = \emph{mean across items} of exact span match (after SQuAD-style normalization). 
# F1 = \emph{median across items} of token-level precision/recall F1 (computed per item, max over golds).
# BERTScore F1 = \emph{median across items} of contextual similarity (computed on original strings, max over golds).
# All values are in \%.}
# \par\medskip
# \caption*{\scriptsize \textbf{Remark.} \textit{LLaMA-3.1-8B often returned full sentences instead of short spans, yielding 0\% EM despite F1≈55–58; strict span-EM penalizes formatting rather than content.}}
# \end{table}
# """.strip("\n")
#     lines.append(tail)
#     return "\n".join(lines)


# # =========================
# # Table 2: Δ vs Gold + latency
# # =========================
# def table2_tex(t2: pd.DataFrame, order: List[str], pp_dp: int, ms_dp: int) -> str:
#     required = ["ΔEM_Para", "ΔEM_Dist", "ΔEM_Both",
#                 "ΔF1_Para", "ΔF1_Dist", "ΔF1_Both",
#                 "p50_ms", "p90_ms"]
#     for c in required:
#         if c not in t2.columns:
#             raise KeyError(f"table2_robust_latency.csv missing column '{c}'")

#     header = r"""
# \subsection{Robustness and Latency}
# \begin{table}[H]\centering
# \caption{Robustness and behavior. Drops are relative to Gold (percentage points).}
# \label{tab:robustness-behavior}
# \scriptsize
# \setlength{\tabcolsep}{4pt}
# \renewcommand{\arraystretch}{1.05}
# \begin{tabular}{l rrr rrr rr}
# \toprule
# & \multicolumn{3}{c}{\(\Delta\)EM (pp)} & \multicolumn{3}{c}{\(\Delta\)F1 (pp)} & \multicolumn{2}{c}{Latency (ms)} \\
# \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}
# Model & Para & Dist & Para+Dist & Para & Dist & Para+Dist & p50 & p90 \\
# \midrule
# """.strip("\n")

#     lines = [header]
#     for name in order:
#         r = pick_row(t2, name)
#         vals = [
#             fmt(r["ΔEM_Para"], pp_dp), fmt(r["ΔEM_Dist"], pp_dp), fmt(r["ΔEM_Both"], pp_dp),
#             fmt(r["ΔF1_Para"], pp_dp), fmt(r["ΔF1_Dist"], pp_dp), fmt(r["ΔF1_Both"], pp_dp),
#             fmt(r["p50_ms"], ms_dp), fmt(r["p90_ms"], ms_dp),
#         ]
#         lines.append(f"{show_name(r['Model'])} & " + " & ".join(vals) + r" \\")
#     tail = r"""
# \bottomrule
# \end{tabular}
# \caption*{\scriptsize
# \textbf{What this shows.} Sensitivity to paraphrases/distractors and speed.\;
# \textbf{How computed.} For each model, $\Delta$EM/F1 are \emph{percentage-point} differences vs. Gold for that model. Latency p50/p90 are medians/90th percentiles over responses (ms).}
# \end{table}
# """.strip("\n")
#     lines.append(tail)
#     return "\n".join(lines)


# # =========================
# # Table 3: Stability
# # =========================
# def table3_tex(t3: pd.DataFrame, order: List[str], ms_dp: int) -> str:
#     # Robust header detection
#     emagree_cols = ["EM-Agree (%)", "EM-Agree_%", "EM_Agree_%"]
#     f1mad_cols = ["F1-MAD (pp)", "F1-MAD_pp", "F1_MAD_pp"]
#     p50_cols = ["Latency p50 (ms)", "Latency_p50_ms", "p50_ms"]

#     def col(df, cands):
#         for c in cands:
#             if c in df.columns: return c
#         raise KeyError(f"table3_stability.csv missing one of {cands}")

#     c_em = col(t3, emagree_cols)
#     c_mad = col(t3, f1mad_cols)
#     c_p50 = col(t3, p50_cols)

#     header = r"""
# \subsection{Run-to-Run Stability (Aggregated)}
# \begin{table}[H]\centering
# \caption{Run-to-run stability aggregated across settings.}
# \label{tab:stability}
# \small
# \setlength{\tabcolsep}{5pt}
# \renewcommand{\arraystretch}{1.05}
# \begin{tabular}{lccc}
# \toprule
# Model & EM-Agree (\%) & F1-MAD (pp) & Latency p50 (ms) \\
# \midrule
# """.strip("\n")

#     lines = [header]
#     for name in order:
#         r = pick_row(t3, name)
#         vals = [fmt(r[c_em], 1), fmt(r[c_mad], 1), fmt(r[c_p50], ms_dp)]
#         lines.append(f"{show_name(r['Model'])} & " + " & ".join(vals) + r" \\")
#     tail = r"""
# \bottomrule
# \end{tabular}
# \caption*{\scriptsize
# \textbf{What this shows.} Stability across $n=3$ repeated runs.\;
# \textbf{How computed.} EM-Agree = average modal-agreement rate per item across runs. 
# F1-MAD = median absolute deviation of per-item F1 across runs (pp). 
# Latency p50 = overall median latency across all responses for the model (ms).}
# \end{table}
# """.strip("\n")
#     lines.append(tail)
#     return "\n".join(lines)


# # =========================
# # Table 4: Leaderboard
# # =========================
# def table4_tex(t4: pd.DataFrame, order: List[str], f1_dp: int, pp_dp: int, ms_dp: int) -> str:
#     req = ["Gold F1", "Avg F1", "Avg |ΔF1| (pp)", "p50 (ms)", "Efficiency (F1/s)", "Composite ↑"]
#     for c in req:
#         if c not in t4.columns:
#             raise KeyError(f"table4_leaderboard.csv missing column '{c}'")

#     header = r"""
# \subsection{Composite Leaderboard}
# \begin{table}[H]\centering
# \caption{Leaderboard combining accuracy, robustness, and speed.}
# \label{tab:leaderboard}
# \small
# \setlength{\tabcolsep}{5pt}
# \renewcommand{\arraystretch}{1.08}
# \begin{tabular}{lrrrrrr}
# \toprule
# Model & Gold F1 & Avg F1 & Avg $|\Delta$F1| (pp) & p50 (ms) & Efficiency (F1/s) & Composite $\uparrow$ \\
# \midrule
# """.strip("\n")

#     lines = [header]
#     # Keep the order provided (even if t4 is sorted by Composite in csv)
#     order_rows = []
#     name_to_row = {show_name(r["Model"]): r for _, r in t4.iterrows()}
#     for name in order:
#         disp = show_name(name)
#         if disp in name_to_row:
#             order_rows.append(name_to_row[disp])
#         else:
#             # fall back by id
#             r2 = t4[t4["Model"].astype(str).str.lower() == str(name).lower()]
#             if len(r2):
#                 order_rows.append(r2.iloc[0])

#     for r in order_rows:
#         vals = [
#             fmt(r["Gold F1"], f1_dp),
#             fmt(r["Avg F1"], f1_dp),
#             fmt(r["Avg |ΔF1| (pp)"], pp_dp),
#             fmt(r["p50 (ms)"], ms_dp),
#             fmt(r["Efficiency (F1/s)"], 1),
#             r"\textbf{" + fmt(r["Composite ↑"], 1) + "}",
#         ]
#         lines.append(f"{show_name(r['Model'])} & " + " & ".join(vals) + r" \\")
#     tail = r"""
# \bottomrule
# \end{tabular}
# \caption*{\scriptsize
# \textbf{What this shows.} A single ranking balancing accuracy, robustness, and speed.\;
# \textbf{How computed.} Avg F1 = mean over \{Gold, Para, Dist, Para+Dist\}. 
# Avg\,$|\Delta$F1| = mean \emph{absolute} drop vs.\ Gold over \{Para, Dist, Para+Dist\}. 
# Efficiency = Gold F1 / (p50 latency in sec). 
# Composite = Avg F1 $- \tfrac{1}{2}$\,Avg\,$|\Delta$F1|.}
# \end{table}
# """.strip("\n")
#     lines.append(tail)
#     return "\n".join(lines)


# # =========================
# # Table 5: Domain breakdown (Gold only)
# # =========================
# def table5_tex(t5: pd.DataFrame, order: List[str], em_dp: int, f1_dp: int) -> str:
#     # Identify domains present (intersect with desired order)
#     all_domains = []
#     for c in t5.columns:
#         if c.endswith("_EM"):
#             all_domains.append(c[:-3])
#     domains_present = [d for d in DOMAINS_ORDER if d in all_domains]
#     if not domains_present:
#         # fall back to any found
#         domains_present = sorted(all_domains)

#     # verify F1 columns exist
#     for d in domains_present:
#         for suffix in ["_EM", "_F1"]:
#             if f"{d}{suffix}" not in t5.columns:
#                 raise KeyError(f"table5_domain_gold.csv missing column '{d}{suffix}'")

#     # header
#     head_domains = " & ".join([fr"\multicolumn{{2}}{{c}}{{{d}}}" for d in domains_present])
#     cmids = []
#     col_index = 2
#     for _ in domains_present:
#         cmids.append(fr"\cmidrule(lr){{{col_index}-{col_index+1}}}")
#         col_index += 2
#     cmid_str = "".join(cmids)

#     header = rf"""
# \subsection{{Domain Breakdown (Gold Only)}}
# \begin{{table}}[H]\centering
# \caption{{Domain breakdown (EM / F1; \%) on \textbf{{Gold}}.}}
# \label{{tab:domain-breakdown}}
# \scriptsize
# \setlength{{\tabcolsep}}{{3.8pt}}
# \renewcommand{{\arraystretch}}{{1.05}}
# \resizebox{{\textwidth}}{{!}}{{%
# \begin{{tabular}}{{l {'cc ' * len(domains_present)} }}
# \toprule
# & {head_domains} \\
# {cmid_str}
# Model & {' & '.join(['EM & F1' for _ in domains_present])} \\
# \midrule
# """.strip("\n")

#     lines = [header]
#     # normal rows
#     for name in order:
#         r = pick_row(t5, name)
#         vals = []
#         for d in domains_present:
#             vals.append(fmt(r[f"{d}_EM"], em_dp))
#             vals.append(fmt(r[f"{d}_F1"], f1_dp))
#         lines.append(f"{show_name(r['Model'])} & " + " & ".join(vals) + r" \\")
#     # Avg row if present
#     avg = t5[t5["Model"].astype(str).str.strip().str.lower() == "avg."]
#     if len(avg):
#         r = avg.iloc[0]
#         vals = []
#         for d in domains_present:
#             vals.append(fmt(r[f"{d}_EM"], em_dp))
#             vals.append(fmt(r[f"{d}_F1"], f1_dp))
#         lines.append(r"\midrule")
#         lines.append("Avg. & " + " & ".join(vals) + r" \\")
#     tail = r"""
# \bottomrule
# \end{tabular}%
# }
# \caption*{\scriptsize
# \textbf{What this shows.} Which domains are easier/harder under \emph{Gold} (no perturbations).\;
# \textbf{How computed.} Items grouped by domain; 
# EM = \emph{mean of per-item EM} (per-item EM = mean across 3 runs). 
# F1 = \emph{median of per-item F1} (per-item F1 = median across 3 runs). 
# Both scaled to \%.}
# \end{table}
# """.strip("\n")
#     lines.append(tail)
#     return "\n".join(lines)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--tables-dir", required=True)
#     ap.add_argument("--figs-dir", required=False)  # ignored (no figures here)
#     ap.add_argument("--out-tex", required=True)
#     ap.add_argument("--model-order", required=True,
#                     help="Comma-separated display names (e.g., 'Gemini Pro,GPT-4o Mini,...')")
#     ap.add_argument("--em-dp", type=int, default=1)
#     ap.add_argument("--f1-dp", type=int, default=1)
#     ap.add_argument("--bert-dp", type=int, default=6)
#     ap.add_argument("--pp-dp", type=int, default=1)
#     ap.add_argument("--ms-dp", type=int, default=1)
#     args = ap.parse_args()

#     tables_dir = Path(args.tables_dir)
#     out_tex = Path(args.out_tex)
#     order = [s.strip() for s in args.model_order.split(",") if s.strip()]

#     # read tables
#     t1 = must_read_csv(tables_dir / "table1_per_setting.csv")
#     t2 = must_read_csv(tables_dir / "table2_robust_latency.csv")
#     t3 = must_read_csv(tables_dir / "table3_stability.csv")
#     t4 = must_read_csv(tables_dir / "table4_leaderboard.csv")
#     t5 = must_read_csv(tables_dir / "table5_domain_gold.csv")

#     # build sections
#     parts = []
#     parts.append(table1_tex(t1, order, args.em_dp, args.f1_dp, args.bert_dp))
#     parts.append(table2_tex(t2, order, args.pp_dp, args.ms_dp))
#     parts.append(table3_tex(t3, order, args.ms_dp))
#     parts.append(table4_tex(t4, order, args.f1_dp, args.pp_dp, args.ms_dp))
#     parts.append(table5_tex(t5, order, args.em_dp, args.f1_dp))

#     out_tex.parent.mkdir(parents=True, exist_ok=True)
#     out_tex.write_text("\n\n".join(parts))
#     print(f"✅ Wrote LaTeX tables to: {out_tex}")


# if __name__ == "__main__":
#     main()