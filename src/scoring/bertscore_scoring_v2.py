# src/scoring/bertscore_scoring_v2.py
"""
BERTScore scoring (v2) with percent outputs for paper/figure tables.

This module computes BERTScore F1 for QA predictions against canonical gold answers
(with alias support), then aggregates the results at several granularities and writes
clean artifacts suitable for downstream analysis, tables, and plots.

What this script does
---------------------
1) Reads raw model outputs from JSONL files (one JSON object per line).
2) Loads gold answers (and aliases) from a CSV.
3) For each prediction, computes BERTScore F1 against *each* gold reference and
   keeps the best score (max over references).
4) Aggregates per-item (median across runs within a qid×setting), pivots to a wide
   per-question table, and computes drops vs. GOLD in percentage points.
5) Aggregates by domain (medians).
6) Writes four artifacts:
   - <outdir>/<model>_bertscore_per_run_v2.csv
       One row per raw JSONL entry. Adds columns:
         • bertscore_f1              (float, 0–1 if rescaled; otherwise typically ~0.7–1.0)
         • bertscore_f1_percent      (float, 0–100; bertscore_f1 × 100)
   - <outdir>/<model>_bertscore_aggregated_items_v2.csv
       One row per (qid, domain). Wide columns:
         • bertscore_{gold,para,dist,para_dist}           (0–1)
         • bertscore_{gold,para,dist,para_dist}_percent   (0–100 mirror)
         • drop_bertscore_{para,dist,para_dist}_pp        (percentage points vs. GOLD)
   - <outdir>/<model>_bertscore_by_domain_v2.csv
       One row per domain with medians:
         • bertscore_gold_median, bertscore_dist_median                 (0–1)
         • bertscore_gold_median_percent, bertscore_dist_median_percent (0–100 mirror)
         • drop_dist_pp_median                                          (pp)
   - <outdir>/<model>_bertscore_summary_v2.json
       Compact JSON summary containing:
         • model, n_requests, n_items
         • bertscore_by_setting_median: medians in PERCENT for {gold, para, dist, para_dist}
         • bertscore_drop_vs_gold_median_pp: median drops in percentage points

Metric notes
------------
- We use Hugging Face `evaluate`'s "bertscore" implementation.
- `rescale_with_baseline=True` (default) yields scores closer to a human-interpretable 0–1
  range (often ~0.85–0.99 on short spans). When disabled, raw cosine-based scores may
  shift slightly; we surface a `--no-rescale` flag to control this.
- We intentionally compute BERTScore on the *original strings* (no lowercasing, no
  punctuation stripping). This matches common practice for semantic similarity metrics.

Terminology
-----------
- “Percent” vs “percentage points (pp)”
  • Percent columns (e.g., *_percent) are simply score × 100.
  • “Drops” (e.g., drop_dist_pp) are *differences* of percent values in *percentage points*.

Assumptions & invariants
------------------------
- The JSONL files contain at least: run_id, qid, setting, output.
- The gold CSV contains: qid, answer, (optional) aliases, domain.
- If a qid in JSONL is missing from gold CSV, we still score it using an empty reference
  string and mark its domain as "unknown".

Typical usage
-------------
python -m src.scoring.bertscore_scoring_v2 \
  --glob "src/results_50/gpt4o/*.jsonl" \
  --gold-csv "src/data_50/mhqa_questions_50.csv" \
  --model "gpt4o" \
  --outdir "src/results_50/gpt4o_bertscore"

Command-line arguments
----------------------
--glob               Glob for input JSONL files. Example: "src/results_50/gpt4o/*.jsonl"
--gold-csv           Path to the gold CSV with columns: qid, answer, [aliases], domain
--model              Model identifier used only for naming output files (e.g., "gpt4o")
--outdir             Directory to write outputs
--bertscore-model    HF backbone for BERTScore (default: "roberta-large")
--bertscore-lang     Language code for BERTScore (default: "en")
--no-rescale         If set, disables BERTScore rescaling with baseline

Implementation outline
----------------------
- We flatten (prediction, reference) pairs to evaluate all references in a single
  BERTScore call, then gather the best reference per prediction efficiently.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import evaluate


# -------------------------
# IO helpers
# -------------------------
def read_jsonl_many(glob_pattern: str) -> pd.DataFrame:
    """
    Read multiple JSONL files (matched by a glob) into a single DataFrame.

    Each matched file is read line-by-line. Each line must contain a valid JSON object.
    All parsed JSON objects from all files are concatenated into a single list and then
    converted to a pandas DataFrame.

    Parameters
    ----------
    glob_pattern : str
        A glob pattern (relative to current working directory) selecting one or more
        JSONL files to read. Example: "src/results_50/gpt4o/*.jsonl".

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per JSON object (per line) across all matched files.
        Columns reflect the keys present in the JSON objects. The function imposes no
        schema beyond “valid JSON per line”.

    Raises
    ------
    FileNotFoundError
        If the glob matches no files or if matched files contain no lines.
    JSONDecodeError
        If any line is not valid JSON.
    """
    rows = []
    for path in sorted(Path().glob(glob_pattern)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No JSONL matched: {glob_pattern}")
    return pd.DataFrame(rows)


def load_gold(gold_csv: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Load gold answers (with aliases) and domains from a CSV.

    The CSV is expected to contain:
      - qid (required): unique question identifier
      - answer (required): canonical gold answer string
      - aliases (optional): '|' separated list of acceptable alternative strings
      - domain (required): domain/category label for the question

    The canonical answer is stored under key "answer" as a **single-element list**,
    and aliases are stored under key "aliases" as a list of strings. This layout
    aligns with the code that later concatenates the two lists to form the reference
    set for each qid.

    Parameters
    ----------
    gold_csv : pathlib.Path
        Path to the CSV containing gold data.

    Returns
    -------
    Dict[str, Dict[str, List[str]]]
        Mapping:
            qid -> {
                "answer":  [canonical_answer_str],
                "aliases": [alias_1, alias_2, ...],
                "domain":  domain_str
            }

    Notes
    -----
    - Empty or missing "aliases" cells are treated as zero aliases.
    - Whitespace is stripped around entries. Empty aliases are dropped.
    """
    df = pd.read_csv(gold_csv)
    out: Dict[str, Dict[str, List[str]]] = {}
    for _, r in df.iterrows():
        qid = str(r["qid"])
        canon = str(r["answer"]).strip()
        aliases_col = str(r.get("aliases", "") or "")
        aliases = [a.strip() for a in aliases_col.split("|") if a.strip()]
        out[qid] = {"answer": [canon], "aliases": aliases, "domain": str(r["domain"]).strip()}
    return out


# -------------------------
# Scoring
# -------------------------
def best_ref_bertscore_f1(
    preds: List[str],
    refs_lists: List[List[str]],
    model_type: str = "roberta-large",
    lang: str = "en",
    rescale_with_baseline: bool = True,
) -> List[float]:
    """
    Compute BERTScore F1 for each prediction against a set of references and
    return the **maximum** F1 per prediction.

    Why "best reference"?
    ---------------------
    Many QA items admit multiple acceptable surface forms (aliases). BERTScore is
    computed for (prediction, each_reference) and the best score is taken for the
    item, mirroring how EM/F1 are often defined as a max over references.

    Efficiency detail
    -----------------
    Instead of calling the metric once per reference set, we **flatten** all
    (prediction, reference) pairs into two lists of the same length and call
    `evaluate.load("bertscore").compute(...)` exactly once. We then map results
    back to each prediction and take the max in its contiguous segment. This
    is significantly faster than many small metric calls.

    Parameters
    ----------
    preds : List[str]
        A list of prediction strings of length N.
    refs_lists : List[List[str]]
        A list (length N) where each element is the reference list for the
        corresponding prediction. For example, refs_lists[i] might be:
        ["John F. Kennedy", "JFK"].
        Empty lists are allowed (treated as [""]).
    model_type : str, optional
        Hugging Face backbone to use for BERTScore (default: "roberta-large").
    lang : str, optional
        Language code (default: "en"). Passed through to the metric.
    rescale_with_baseline : bool, optional
        Whether to apply BERTScore baseline rescaling (default: True). With rescaling,
        results are typically near [0, 1]. Without rescaling, raw similarity scores
        are used (often still high but not baseline-adjusted).

    Returns
    -------
    List[float]
        A list of length N. Each element is the maximum BERTScore F1 across the
        reference set for that prediction (float, commonly ~0.85–0.99 when rescaled).

    Raises
    ------
    ValueError
        If `preds` and `refs_lists` lengths differ.

    Notes
    -----
    - BERTScore is computed on original strings (no normalization).
    - If a reference list is empty, we substitute a single empty string to keep shapes valid.
    - The `evaluate` library will handle model loading and caching.
    """
    if len(preds) != len(refs_lists):
        raise ValueError("preds and refs_lists must have the same length")

    metric = evaluate.load("bertscore")

    # Flatten (prediction, reference) pairs into parallel lists
    flat_preds: List[str] = []
    flat_refs: List[str] = []
    bounds: List[tuple[int, int]] = []  # segment bounds for each original prediction
    cursor = 0

    for p, refs in zip(preds, refs_lists):
        refs = refs or [""]
        n = len(refs)
        flat_preds.extend([p or ""] * n)
        flat_refs.extend([(r or "") for r in refs])
        bounds.append((cursor, cursor + n))
        cursor += n

    # Single metric call over the flattened pairs
    res = metric.compute(
        predictions=flat_preds,
        references=flat_refs,
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
    )
    f1 = res["f1"]  # list[float] aligned to flat pairs

    # Take the max F1 within each prediction's segment
    best: List[float] = []
    for a, b in bounds:
        seg = f1[a:b]
        best.append(max(seg) if seg else 0.0)
    return best


def pp(x: float) -> float:
    """
    Convert a 0–1 score to a 0–100 "percent" value.

    Parameters
    ----------
    x : float
        Score on [0, 1], e.g., BERTScore F1.

    Returns
    -------
    float
        100 * x. If x is already a percent value and you call this again by mistake,
        the result will be 100× too large—use with care.
    """
    return 100.0 * x


# -------------------------
# Main
# -------------------------
def main():
    """
    CLI entry point for BERTScore scoring with v2 aggregation.

    This function:
      1) Parses CLI args (glob, gold-csv, model, outdir, metric config).
      2) Loads raw predictions (JSONL) and gold data (CSV).
      3) Computes per-request best-reference BERTScore F1.
      4) Writes per-run CSV with F1 and F1-percent.
      5) Aggregates to per-item medians across runs within (qid × setting),
         pivots to wide, computes percent mirrors and drops vs GOLD; writes CSV.
      6) Aggregates by domain (medians for GOLD and DIST and the median drop); writes CSV.
      7) Writes a compact JSON summary with medians by setting (percent) and median drops.

    Inputs (required)
    -----------------
    --glob : str
        Glob selecting JSONL files. Each line is a JSON object with keys including:
        run_id, qid, setting, output. Additional keys (domain, model, latency_ms) are
        tolerated and preserved in the per-run CSV if present.
    --gold-csv : str
        Path to gold CSV with columns: qid, answer, [aliases], domain.
    --model : str
        Model identifier (purely for naming output files).
    --outdir : str
        Output directory. Created if it does not exist.

    Optional metric args
    --------------------
    --bertscore-model : str, default "roberta-large"
    --bertscore-lang  : str, default "en"
    --no-rescale      : flag; if present, disables baseline rescaling.

    Outputs
    -------
    Writes four artifacts into --outdir:
      • <model>_bertscore_per_run_v2.csv
      • <model>_bertscore_aggregated_items_v2.csv
      • <model>_bertscore_by_domain_v2.csv
      • <model>_bertscore_summary_v2.json

    Exit status
    -----------
    Prints paths and exits normally on success. Raises exceptions on missing
    inputs or schema violations (to fail fast in pipelines).
    """
    ap = argparse.ArgumentParser(description="BERTScore v2 (per-run, per-item, domain, JSON) with percent outputs.")
    ap.add_argument("--glob", required=True, help='Glob for raw jsonl, e.g. "src/results_50/gpt4o/*.jsonl"')
    ap.add_argument("--gold-csv", required=True, help='e.g. "src/data_50/mhqa_questions_50.csv"')
    ap.add_argument("--model", required=True, help="model id for filenames (e.g., gpt4o)")
    ap.add_argument("--outdir", required=True, help="directory to write outputs")
    ap.add_argument("--bertscore-model", default="roberta-large")
    ap.add_argument("--bertscore-lang", default="en")
    ap.add_argument("--no-rescale", action="store_true", help="disable rescale_with_baseline")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    gold = load_gold(Path(args.gold_csv))
    df = read_jsonl_many(args.glob)

    # Expect columns: run_id, qid, domain, model, setting, output, latency_ms (domain/model optional)
    needed = {"run_id", "qid", "setting", "output"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in jsonl: {missing}")

    # Reference lists and domains (from gold)
    refs_lists: List[List[str]] = []
    domains: List[str] = []
    for q in df["qid"]:
        g = gold.get(str(q))
        if not g:
            refs_lists.append([""])
            domains.append("unknown")
        else:
            refs_lists.append(g["answer"] + g["aliases"])
            domains.append(g["domain"])
    df["domain_gold"] = domains

    # --- Per-request BERTScore-F1 (best ref) ---
    df["bertscore_f1"] = best_ref_bertscore_f1(
        preds=df["output"].astype(str).tolist(),
        refs_lists=refs_lists,
        model_type=args.bertscore_model,
        lang=args.bertscore_lang,
        rescale_with_baseline=not args.no_rescale,
    )
    # Percent mirror for paper
    df["bertscore_f1_percent"] = df["bertscore_f1"].map(pp)

    # Save per-run
    per_run_csv = outdir / f"{args.model}_bertscore_per_run_v2.csv"
    df.to_csv(per_run_csv, index=False)

    # --- Aggregate to per-item (median across runs within qid×setting) ---
    ag = (
        df.groupby(["qid", "domain_gold", "setting"], as_index=False)
          .agg(bertscore_f1_median=("bertscore_f1", "median"))
    )

    # Pivot to wide (one row per qid with four settings)
    wide = ag.pivot(index=["qid", "domain_gold"], columns="setting", values="bertscore_f1_median")

    # Ensure all four columns exist
    for s in ["gold", "para", "dist", "para_dist"]:
        if s not in wide.columns:
            wide[s] = None

    # Reset index and standardize column names
    wide = wide.reset_index().rename(columns={"domain_gold": "domain"})
    renamed = wide.rename(columns={
        "gold": "bertscore_gold",
        "para": "bertscore_para",
        "dist": "bertscore_dist",
        "para_dist": "bertscore_para_dist",
    })

    # Percent mirrors for paper/tables
    for col in ["bertscore_gold", "bertscore_para", "bertscore_dist", "bertscore_para_dist"]:
        renamed[col + "_percent"] = renamed[col].map(lambda v: pp(v) if v is not None else None)

    # Drops vs GOLD (percentage points); underlying values are 0–1
    renamed["drop_bertscore_para_pp"]      = (renamed["bertscore_para"]      - renamed["bertscore_gold"]).fillna(0.0) * 100.0
    renamed["drop_bertscore_dist_pp"]      = (renamed["bertscore_dist"]      - renamed["bertscore_gold"]).fillna(0.0) * 100.0
    renamed["drop_bertscore_para_dist_pp"] = (renamed["bertscore_para_dist"] - renamed["bertscore_gold"]).fillna(0.0) * 100.0

    # Save per-item CSV
    per_item_csv = outdir / f"{args.model}_bertscore_aggregated_items_v2.csv"
    renamed.to_csv(per_item_csv, index=False)

    # --- By-domain (median GOLD/DIST and drop) ---
    def med(series):
        """
        Helper: median ignoring NaNs.

        Parameters
        ----------
        series : Iterable[float] or pandas.Series
            Values for which to compute the median.

        Returns
        -------
        float or None
            Median value if available; otherwise None when all values are NaN/empty.
        """
        s = pd.Series(series).dropna()
        return float(s.median()) if not s.empty else None

    domain_rows = []
    for dom, sub in renamed.groupby("domain"):
        domain_rows.append({
            "domain": dom,
            "bertscore_gold_median": med(sub["bertscore_gold"]),
            "bertscore_dist_median": med(sub["bertscore_dist"]),
            "drop_dist_pp_median":   med(sub["drop_bertscore_dist_pp"]),
        })
    by_domain_df = pd.DataFrame(domain_rows).sort_values("domain")

    # Percent mirrors for domain medians
    for col in ["bertscore_gold_median", "bertscore_dist_median"]:
        by_domain_df[col + "_percent"] = by_domain_df[col].map(lambda v: pp(v) if v is not None else None)

    by_domain_csv = outdir / f"{args.model}_bertscore_by_domain_v2.csv"
    by_domain_df.to_csv(by_domain_csv, index=False)

    # --- JSON summary (compact; medians in percent, drops in pp) ---
    by_setting = {
        "gold":      med(renamed["bertscore_gold"]),
        "para":      med(renamed["bertscore_para"]),
        "dist":      med(renamed["bertscore_dist"]),
        "para_dist": med(renamed["bertscore_para_dist"]),
    }
    drops = {
        "para":      med(renamed["drop_bertscore_para_pp"]),
        "dist":      med(renamed["drop_bertscore_dist_pp"]),
        "para_dist": med(renamed["drop_bertscore_para_dist_pp"]),
    }

    summary = {
        "model": args.model,
        "n_requests": int(len(df)),
        "n_items": int(renamed.shape[0]),
        "bertscore_by_setting_median": {k: (pp(v) if v is not None else None) for k, v in by_setting.items()},
        "bertscore_drop_vs_gold_median_pp": drops,  # already percentage points
    }
    summary_json = outdir / f"{args.model}_bertscore_summary_v2.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("✅ BERTScore v2 done")
    print("  Per-run:   ", per_run_csv)
    print("  Per-item:  ", per_item_csv)
    print("  By-domain: ", by_domain_csv)
    print("  Summary:   ", summary_json)


if __name__ == "__main__":
    main()
