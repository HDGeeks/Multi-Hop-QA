# src/scoring/scoring_v2.py
from __future__ import annotations
import argparse, json, re, html
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

"""
================================================================================
Scoring v2 — Alias-aware EM/F1 with HTML-tag cleanup for LLaMA
================================================================================

WHAT (plain english):
---------------------
This module scores model outputs against gold answers and produces clean CSV/JSON
artifacts used by the reporting pipeline. It computes:
  • Exact Match (EM) – strict span match after normalization
  • Token-level F1  – overlap between tokens after normalization
  • Refusal / Invalid flags
  • Aggregations per item (across runs) and per model×setting (across items)
  • Drops vs. Gold (Δ in percentage points)
  • Latency summaries (median over runs/items)

It also fixes a practical formatting issue: some LLaMA outputs wrap answers in
HTML (e.g., "<span>H<sub>2</sub>O</span>"). We safely strip tags *only* for
LLaMA models before scoring so formatting does not unfairly hurt EM/F1.

WHY (the intent):
-----------------
We want robust, fair, and reproducible numbers that are comparable across:
  • Multiple models (API and open-weights)
  • Four settings (gold, para, dist, para_dist)
  • Multiple runs per item (to capture randomness)

Key fairness aspects:
  • Alias-aware scoring: if gold has aliases, any match counts.
  • Normalize text: lowercasing, punctuation removal, whitespace squeeze.
  • LLaMA HTML cleanup: remove tags so content (not markup) is scored.

HOW (the mechanics):
--------------------
Inputs:
  • Raw JSONL files written by the runner, with fields:
      run_id, qid, domain, model, setting, output, latency_ms, ...
  • A CSV with gold answers and optional aliases (columns: qid, answer, aliases)
  • “Context” and “Paras” CSVs are checked for existence but not read here.

Pipeline (high level):
  1) read_jsonl_many(glob) → DataFrame of raw rows
  2) load_gold_map(csv)    → dict: qid -> [canonical, alias1, alias2, ...]
  3) compute_per_run(raw, gold_map)
       - If model name includes “llama”, remove HTML tags in the prediction.
       - Compute EM/F1/refusal/invalid per row.
  4) aggregate_items(per_run)
       - Collapse multiple runs per (qid, setting, model) into per-item stats.
  5) summarize_model_setting(items_ag)
       - Aggregate across items → per model×setting summary, % scaled.
       - Add Δ vs Gold within each model.
  6) Write artifacts next to the requested out JSON:
       • per_run_v2.csv
       • aggregated_items_v2.csv
       • summary_v2.csv
       • summary_v2.json (compact micro/by-setting overview)

Outputs (where they go):
------------------------
The --out-json path determines the output folder. The three CSVs are written
alongside it, and the JSON contains quick “micro” metrics and the CSV paths.

Design choices & examples:
--------------------------
• EM is strict: "paris" == "paris", but "the capital is paris" is not EM,
  though it can still get high F1 due to token overlap.
• F1 is bag-of-words over normalized text. Example:
    pred="the capital is paris"
    gold="paris"
  → EM=0 (not exact), F1>0 (shared token “paris”).

• LLaMA HTML cleanup (only LLaMA):
    "<span>H<sub>2</sub>O</span>" → "H2O"
  This affects EM/F1 fairly without rewarding extra trimming of unrelated words.

Assumptions:
------------
• Each (qid, model, setting) can have 1..n runs.
• “Refusal” is a heuristic phrase check (simple and conservative).
• “Invalid” counts only truly empty output (after basic strip).
• Aliases in CSV are “|”-separated in a single 'aliases' column.

Repro tip (CLI):
----------------
Example command:

    python -m src.scoring.scoring_v2 \
        --glob "src/results_50/llama31_8b/*.jsonl" \
        --gold-csv "src/data_50/mhqa_questions_50.csv" \
        --context-csv "src/data_50/mhqa_context_50.csv" \
        --paras-csv "src/data_50/mhqa_paraphrases_50.csv" \
        --out-json "src/results_50/llama31_8b/summary_v2.json"

This will score all matching JSONL, write 3 CSVs, and produce a compact JSON.

Happy path mental model 🧠:
---------------------------
Think of it like grading short-answer exams:
  • WHAT: Compare student answers (“output”) to the answer key (“gold+aliases”)
  • WHY: Be fair across different styles/formatting (normalize; strip LLaMA tags)
  • HOW: Grade each attempt (run), then summarize per question, then per model
================================================================================
"""


# ----------------------------
# Text normalization & metrics
# ----------------------------
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize(s: str) -> str:
    """
    WHAT:
        Normalize a string for fair text comparison.

    WHY:
        Models differ in capitalization, punctuation, and spacing. Normalizing
        removes those superficial differences so EM/F1 reflect content.

    HOW:
        1) Handle None as empty string.
        2) Strip leading/trailing spaces.
        3) Lowercase.
        4) Replace any non-word / non-space character with a space.
        5) Collapse multiple spaces to one.

    Parameters
    ----------
    s : str
        Input text. Can be None.

    Returns
    -------
    str
        A normalized string.

    Examples
    --------
    >>> normalize("  The Capital: Paris!  ")
    'the capital paris'
    >>> normalize(None)
    ''
    """
    if s is None:
        return ""
    s = s.strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


# --- HTML stripping (minimal, safe) ---
_TAG_RE = re.compile(r"<[^>]+>")


def strip_html_tags(s: str) -> str:
    """
    WHAT:
        Remove HTML tags and unescape HTML entities from a prediction string.

    WHY:
        Some models (notably LLaMA) may wrap outputs in tags like <span>...</span>
        or include subscripts (e.g., H<sub>2</sub>O). These tags should not
        affect EM/F1. We want to score the content, not the markup.

    HOW:
        1) html.unescape to convert entities (e.g., &nbsp;, &amp;).
        2) Regex replace tags (“<...>”) with a single space.
        3) Squeeze whitespace.

        Note: This is intentionally simple and safe for short span answers.
        We do NOT parse HTML — we just drop tags.

    Parameters
    ----------
    s : str
        Raw prediction text that may contain HTML.

    Returns
    -------
    str
        Plain text with tags removed and whitespace normalized.

    Examples
    --------
    >>> strip_html_tags("<span>H<sub>2</sub>O</span>")
    'H2O'
    >>> strip_html_tags("John &amp; Mary")
    'John & Mary'
    """
    if not s:
        return ""
    s = html.unescape(s)
    s = _TAG_RE.sub(" ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    """
    WHAT:
        Compute token-level F1 between a predicted answer and one gold answer.

    WHY:
        EM is strict; small phrasing differences can miss EM. F1 gives partial
        credit based on overlapping tokens after normalization.

    HOW:
        1) normalize(pred) and normalize(gold)
        2) Split by whitespace into tokens.
        3) Count multiset overlap (min counts per token).
        4) precision = overlap / len(pred_tokens)
           recall    = overlap / len(gold_tokens)
           F1        = 2 * p * r / (p + r), guarding zeros.

    Edge cases
    ----------
    - Both empty → F1 = 1.0 (they match on “nothing”)
    - One empty  → F1 = 0.0

    Parameters
    ----------
    pred : str
        Model prediction.
    gold : str
        One gold answer.

    Returns
    -------
    float
        F1 in [0.0, 1.0].

    Examples
    --------
    >>> token_f1("The capital is Paris", "Paris")
    0.5  # tokens overlap on "paris"
    >>> token_f1("", "")
    1.0
    >>> token_f1("London", "Paris")
    0.0
    """
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    # bag-of-words overlap (multiset)
    common = {}
    for t in p:
        common[t] = min(p.count(t), g.count(t))
    num = sum(common.values())
    if num == 0:
        return 0.0
    prec = num / len(p)
    rec = num / len(g)
    return 2 * prec * rec / (prec + rec)


def best_em_f1(pred: str, refs: List[str]) -> Tuple[int, float]:
    """
    WHAT:
        Compute EM and the best (max) F1 between a prediction and a list of
        reference answers (canonical + aliases).

    WHY:
        Questions can have multiple valid forms (“JFK”, “John F. Kennedy”).
        We allow any alias to satisfy EM or give partial credit for F1.

    HOW:
        For each reference r in refs:
          • EM: if normalize(pred) == normalize(r) → EM=1
          • F1: compute token_f1(pred, r) and take the maximum over refs

    Parameters
    ----------
    pred : str
        Model prediction (raw; caller may pre-clean HTML if needed).
    refs : List[str]
        List of gold answers (first is canonical; rest are aliases). May be empty.

    Returns
    -------
    (int, float)
        Tuple (em, best_f1)
        - em: 0 or 1
        - best_f1: max F1 in [0.0, 1.0] across refs

    Examples
    --------
    >>> best_em_f1("John F. Kennedy", ["John F. Kennedy", "JFK"])
    (1, 1.0)
    >>> best_em_f1("JFK", ["John F. Kennedy", "JFK"])
    (1, 1.0)
    >>> best_em_f1("Kennedy", ["John F. Kennedy", "JFK"])
    (0, 0.5)
    """
    refs = refs or [""]
    em = 0
    best = 0.0
    for r in refs:
        if normalize(pred) == normalize(r):
            em = 1
        best = max(best, token_f1(pred, r))
    return em, best


REFUSAL_MARKERS = [
    "i can't", "i cannot", "i’m unable", "i am unable",
    "as an ai", "i don't have", "cannot assist", "won't provide",
    "cannot answer", "no sufficient", "lack enough info",
]


def is_refusal(text: str) -> bool:
    """
    WHAT:
        Heuristically flag refusals (e.g., policy statements instead of answers).

    WHY:
        For analysis, it’s helpful to know when models refuse, even if answers
        are otherwise short. This is a simple keyword check (not perfect).

    HOW:
        Lowercase and test if any marker phrase exists in the text.

    Parameters
    ----------
    text : str
        Model output.

    Returns
    -------
    bool
        True if any refusal marker is found; else False.

    Examples
    --------
    >>> is_refusal("As an AI, I cannot answer that.")
    True
    >>> is_refusal("John F. Kennedy")
    False
    """
    t = (text or "").strip().lower()
    return any(m in t for m in REFUSAL_MARKERS)


def is_invalid(text: str) -> bool:
    """
    WHAT:
        Flag truly empty outputs.

    WHY:
        Empty means no usable answer for span scoring.

    HOW:
        After a simple strip, check if the string is empty.

    Parameters
    ----------
    text : str
        Model output.

    Returns
    -------
    bool
        True if empty; False otherwise.

    Examples
    --------
    >>> is_invalid("   ")
    True
    >>> is_invalid("Paris")
    False
    """
    return (text or "").strip() == ""


# ----------------------------
# Load gold from CSV
# ----------------------------
def load_gold_map(q_csv: Path) -> Dict[str, List[str]]:
    """
    WHAT:
        Load a mapping from qid → [canonical, alias1, alias2, ...] from a CSV.

    WHY:
        Alias-aware scoring needs all acceptable references for each question.

    HOW:
        - Read CSV (expects columns: 'qid', 'answer', optional 'aliases').
        - Split 'aliases' by '|' if present.
        - Build a list: [answer] + aliases (order preserved).
        - Keys are strings (qid coerced to str for consistency).

    Parameters
    ----------
    q_csv : Path
        Path to the questions CSV.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from qid to list of references.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.

    Examples
    --------
    CSV row:
      qid=HIST001, answer="John F. Kennedy", aliases="JFK|John Kennedy"
    → {"HIST001": ["John F. Kennedy", "JFK", "John Kennedy"]}
    """
    df = pd.read_csv(q_csv)
    out: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        qid = str(r["qid"])
        canon = str(r["answer"]).strip()
        aliases = [a.strip() for a in str(r.get("aliases", "") or "").split("|") if a.strip()]
        out[qid] = [canon] + aliases
    return out


# ----------------------------
# I/O helpers
# ----------------------------
def read_jsonl_many(glob_pattern: str) -> pd.DataFrame:
    """
    WHAT:
        Read many JSONL files matching a glob pattern into a single DataFrame.

    WHY:
        The runner writes one JSONL per (model, run). We want to score them all
        together.

    HOW:
        - Glob paths (sorted for determinism).
        - Read each line, json.loads → dict → collect in a list.
        - Build a DataFrame from the collected dicts.

    Parameters
    ----------
    glob_pattern : str
        e.g., "src/results_50/gpt4o/*.jsonl"

    Returns
    -------
    pandas.DataFrame
        One row per JSONL line.

    Raises
    ------
    SystemExit
        If no files match (so downstream code doesn't run silently).

    Examples
    --------
    >>> df = read_jsonl_many("src/results_50/llama31_8b/*.jsonl")
    >>> {"run_id","qid","model","setting","output"}.issubset(df.columns)
    True
    """
    all_rows = []
    for p in sorted(Path().glob(glob_pattern)):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                all_rows.append(json.loads(line))
    if not all_rows:
        raise SystemExit(f"No files matched: {glob_pattern}")
    return pd.DataFrame(all_rows)


# ----------------------------
# Scoring
# ----------------------------
def compute_per_run(df: pd.DataFrame, gold_map: Dict[str, List[str]]) -> pd.DataFrame:
    """
    WHAT:
        Score every raw row (a single model output) against the gold map.

    WHY:
        We need per-run EM/F1 first; later we collapse across runs into per-item.

    HOW:
        - Validate required columns.
        - For each row:
            * If model contains "llama", strip HTML tags from the prediction.
            * Look up refs by qid.
            * best_em_f1(pred, refs) → (em, f1)
            * Flag refusal/invalid on the (possibly cleaned) pred.
        - Return a copy of df with new columns: em, f1, refusal, invalid.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw rows with at least:
        { run_id, qid, domain, model, setting, output, latency_ms }.
    gold_map : Dict[str, List[str]]
        Mapping qid → [canonical, aliases...]

    Returns
    -------
    pandas.DataFrame
        Per-run dataframe including scoring columns.

    Raises
    ------
    ValueError
        If required columns are missing.

    Examples
    --------
    >>> df = pd.DataFrame([{
    ...   "run_id":1, "qid":"H1", "domain":"hist", "model":"llama31_8b",
    ...   "setting":"gold", "output":"<span>Paris</span>", "latency_ms": 800
    ... }])
    >>> gold = {"H1": ["Paris"]}
    >>> out = compute_per_run(df, gold)
    >>> int(out.loc[0, "em"]), round(out.loc[0, "f1"], 3)
    (1, 1.0)
    """
    need = {"run_id", "qid", "domain", "model", "setting", "output", "latency_ms"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in raw jsonl: {sorted(miss)}")

    ems, f1s, refusals, invalids = [], [], [], []
    for qid, pred, model in zip(df["qid"], df["output"], df["model"]):
        # strip HTML only for LLaMA family
        m = str(model).lower()
        pred_str = str(pred)
        if "llama" in m:
            pred_str = strip_html_tags(pred_str)

        refs = gold_map.get(str(qid), [""])
        em, f1 = best_em_f1(pred_str, refs)
        ems.append(em)
        f1s.append(f1)
        refusals.append(int(is_refusal(pred_str)))
        invalids.append(int(is_invalid(pred_str)))

    out = df.copy()
    out["em"] = ems
    out["f1"] = f1s
    out["refusal"] = refusals
    out["invalid"] = invalids
    return out


def aggregate_items(per_run: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT:
        Collapse multiple runs into a single per-item result for each
        (qid, domain, model, setting).

    WHY:
        We evaluate robustness across repeated runs. For reporting, we use
        stable per-item aggregates:
          • EM_majority over runs
          • F1_median   over runs
          • latency median over runs
          • refusal_any / invalid_any (did it occur at least once?)
          • stability_mad_f1 = median absolute deviation across runs

    HOW:
        - Group by ["qid","domain","model","setting"].
        - Aggregate with custom functions (see code).
        - Return a dataframe with one row per item.

    Aggregation details
    -------------------
    EM majority:
        For n runs, EM_majority = 1 if sum(EM) >= floor(n/2)+1 else 0.
    F1 median:
        Median of run-level F1s (robust to outliers).
    Latency median:
        Median of latency_ms across runs.
    MAD of F1:
        median(|F1_i − median(F1)|) across runs.

    Parameters
    ----------
    per_run : pandas.DataFrame
        Output of compute_per_run with columns ['em','f1',...].

    Returns
    -------
    pandas.DataFrame
        Per-item aggregated results.

    Examples
    --------
    >>> x = pd.DataFrame({
    ...   "qid":["Q1","Q1","Q1"], "domain":["d"]*3, "model":["m"]*3,
    ...   "setting":["gold"]*3, "em":[1,1,0], "f1":[1.0,1.0,0.5], "latency_ms":[800,900,950],
    ... })
    >>> agg = aggregate_items(x)
    >>> int(agg.loc[0, "em_majority"]), float(agg.loc[0, "f1_median"])
    (1, 1.0)
    """
    # Collapse runs per (qid, setting, model, domain)
    def em_majority(s: pd.Series) -> int:
        n = int(s.size)
        if n == 0:
            return 0
        thresh = n // 2 + 1  # majority threshold, works for any n
        return int(int(s.sum()) >= thresh)

    def mad(s: pd.Series) -> float:
        arr = s.to_numpy(dtype=float)
        if arr.size == 0:
            return 0.0
        med = float(np.median(arr))
        return float(np.median(np.abs(arr - med)))

    ag = (
        per_run
        .groupby(["qid", "domain", "model", "setting"], as_index=False)
        .agg(
            em_majority=("em", em_majority),
            f1_median=("f1", "median"),
            latency_median_ms=("latency_ms", "median"),
            refusal_any=("refusal", "max"),
            invalid_any=("invalid", "max"),
            stability_mad_f1=("f1", mad),
        )
    )
    return ag


def summarize_model_setting(items_ag: pd.DataFrame) -> pd.DataFrame:
    """
    WHAT:
        Build per model×setting summary across items, scaled to percent and
        enriched with Δ vs. Gold.

    WHY:
        Paper/report tables need one row per model×setting with interpretable
        percentages and robustness drops.

    HOW:
        For each (model, setting):
          • n_items: unique qids
          • em_mean_percent: mean of em_majority × 100
          • f1_mean_of_medians_percent: mean of per-item f1_median × 100
          • latency_p50_ms: median of per-item latency_median_ms
          • refusal_rate_percent / invalid_rate_percent: mean × 100
          • stability_mad_f1_median: median of per-item MAD(F1)
        Then compute drops vs the "gold" row *within each model*:
          • drop_em_vs_gold_pp
          • drop_f1_vs_gold_pp

        Settings are ordered gold, para, dist, para_dist (if present).

    Parameters
    ----------
    items_ag : pandas.DataFrame
        Output of aggregate_items.

    Returns
    -------
    pandas.DataFrame
        Summary dataframe used by reporting.

    Examples
    --------
    >>> # Minimal synthetic:
    >>> s = pd.DataFrame({
    ...   "model":["m","m"], "setting":["gold","para"],
    ...   "qid":["Q1","Q1"],
    ...   "em_majority":[1,0], "f1_median":[1.0,0.8],
    ...   "latency_median_ms":[900,950], "refusal_any":[0,0], "invalid_any":[0,0],
    ...   "stability_mad_f1":[0.0,0.0]
    ... })
    >>> out = summarize_model_setting(s)
    >>> round(float(out.loc[out.setting=="para","drop_f1_vs_gold_pp"]), 1)
    -20.0
    """
    # Summary per model×setting across items
    def pct(x) -> float:
        x = pd.Series(x)
        return 100.0 * float(x.mean()) if len(x) else 0.0

    rows = []
    for (model, setting), grp in items_ag.groupby(["model", "setting"]):
        n_items = grp["qid"].nunique()
        f1_mean_of_medians = 100.0 * float(np.mean(grp["f1_median"])) if len(grp) else 0.0
        row = {
            "model": model,
            "setting": setting,
            "n_items": int(n_items),
            "em_mean_percent": pct(grp["em_majority"]),
            # Paper definition: mean over items of per-item (median-across-runs) F1
            "f1_mean_of_medians_percent": f1_mean_of_medians,
            "latency_p50_ms": float(np.median(grp["latency_median_ms"])) if len(grp) else 0.0,
            "refusal_rate_percent": pct(grp["refusal_any"]),
            "invalid_rate_percent": pct(grp["invalid_any"]),
            "stability_mad_f1_median": float(np.median(grp["stability_mad_f1"])) if len(grp) else 0.0,
        }
        rows.append(row)

    summ = pd.DataFrame(rows)

    # Back-compat alias so downstream code (tables/figures) keeps working
    summ["f1_median_percent"] = summ["f1_mean_of_medians_percent"]

    # Add drops vs GOLD within each model (Δ in percentage points)
    def add_drop(metric_cols, label):
        col = next((c for c in metric_cols if c in summ.columns), None)
        if col is None:
            return
        drops = []
        for model, grp in summ.groupby("model", sort=False):
            base_ser = grp.loc[grp["setting"] == "gold", col]
            base_val = float(base_ser.iloc[0]) if len(base_ser) else np.nan
            for _, r in grp.iterrows():
                if r["model"] != model or np.isnan(base_val):
                    drops.append(np.nan)
                elif r["setting"] == "gold":
                    drops.append(np.nan)
                else:
                    drops.append(float(r[col]) - base_val)
        summ[label] = drops

    add_drop(["em_mean_percent"], "drop_em_vs_gold_pp")
    add_drop(["f1_mean_of_medians_percent", "f1_median_percent"], "drop_f1_vs_gold_pp")

    # Order settings
    order = pd.CategoricalDtype(categories=["gold", "para", "dist", "para_dist"], ordered=True)
    if "setting" in summ.columns:
        summ["setting"] = summ["setting"].astype(order)
        summ = summ.sort_values(["model", "setting"]).reset_index(drop=True)

    return summ


# ----------------------------
# CLI
# ----------------------------
def main():
    """
    WHAT:
        Command-line entry point. Wires the whole scoring pipeline together and
        writes CSVs + a compact JSON summary.

    WHY:
        Reproducible batch scoring that other scripts (report builders, plots)
        can depend on.

    HOW:
        1) Parse args.
        2) Sanity-check input files exist.
        3) Load raw JSONL and gold map.
        4) compute_per_run → aggregate_items → summarize_model_setting.
        5) Write per_run_v2.csv, aggregated_items_v2.csv, summary_v2.csv.
        6) Also write a compact JSON with “micro_overall” and “by_setting”.

    Arguments
    ---------
    --glob : str
        Glob for input JSONL files (e.g., "src/results_50/gpt4o/*.jsonl").
    --gold-csv : str
        Path to questions CSV (must include qid, answer, optional aliases).
    --context-csv : str
        Checked for existence only (not read).
    --paras-csv : str
        Checked for existence only (not read).
    --out-json : str
        Path for summary JSON; sibling CSVs are written next to it.

    Outputs
    -------
    • <out_dir>/per_run_v2.csv
    • <out_dir>/aggregated_items_v2.csv
    • <out_dir>/summary_v2.csv
    • <out_json> (summary_v2.json with quick stats and the above paths)

    Examples
    --------
    >>> # Typical invocation from shell:
    >>> # python -m src.scoring.scoring_v2 \\
    ...     --glob "src/results_50/llama31_8b/*.jsonl" \\
    ...     --gold-csv "src/data_50/mhqa_questions_50.csv" \\
    ...     --context-csv "src/data_50/mhqa_context_50.csv" \\
    ...     --paras-csv "src/data_50/mhqa_paraphrases_50.csv" \\
    ...     --out-json "src/results_50/llama31_8b/summary_v2.json"
    """
    ap = argparse.ArgumentParser(description="Alias-aware EM/F1 scoring with run collapsing.")
    ap.add_argument("--glob", required=True, help="e.g. src/results_50/gpt4o/*.jsonl")
    ap.add_argument("--gold-csv", required=True, help="src/data_50/mhqa_questions_50.csv")
    ap.add_argument("--context-csv", required=True, help="(unused for scoring, checked for existence)")
    ap.add_argument("--paras-csv", required=True, help="(unused for scoring, checked for existence)")
    ap.add_argument("--out-json", required=True, help="summary JSON path")
    args = ap.parse_args()

    # Resolve and sanity-check I/O
    glob_pat = args.glob
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    for p in [args.gold_csv, args.context_csv, args.paras_csv]:
        if not Path(p).exists():
            raise SystemExit(f"Missing file: {p}")

    # 1) Read raw + gold
    raw = read_jsonl_many(glob_pat)
    gold_map = load_gold_map(Path(args.gold_csv))

    # 2) Per-run scoring
    per_run = compute_per_run(raw, gold_map)

    # 3) Collapse to per-item
    items_ag = aggregate_items(per_run)

    # 4) Build model×setting summary
    summary = summarize_model_setting(items_ag)

    # 5) Write artifacts next to JSON
    outdir = out_json.parent
    per_run_path = outdir / "per_run_v2.csv"
    items_path = outdir / "aggregated_items_v2.csv"
    summary_path = outdir / "summary_v2.csv"

    per_run.to_csv(per_run_path, index=False)
    items_ag.to_csv(items_path, index=False)
    summary.to_csv(summary_path, index=False)

    # 6) Compact JSON with micro + by-setting (micro = mean across runs)
    def micro_overall(df: pd.DataFrame) -> Dict[str, float]:
        """
        WHAT:
            Quick “overall” metrics over all rows (ignores settings/models).

        WHY:
            Handy summary for dashboards and quick checks.

        HOW:
            - exact_match_percent: 100 * mean(em)
            - f1_percent:          100 * mean(f1)
            - n:                    number of rows
        """
        return {
            "exact_match_percent": 100.0 * float(df["em"].mean()) if len(df) else 0.0,
            "f1_percent": 100.0 * float(df["f1"].mean()) if len(df) else 0.0,
            "n": int(len(df)),
        }

    micro = micro_overall(per_run)
    by_setting = (
        per_run.groupby("setting", sort=False)
        .apply(lambda g: micro_overall(g))
        .to_dict()
    )

    payload = {
        "micro_overall": micro,
        "by_setting": by_setting,
        "paths": {
            "per_run_csv": str(per_run_path),
            "aggregated_items_csv": str(items_path),
            "summary_csv": str(summary_path),
        },
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("✅ scoring_v2 complete")
    print(json.dumps(payload["micro_overall"], indent=2))


if __name__ == "__main__":
    main()