# src/scoring/domain_scoring.py
"""
domain_scoring.py
=================

Aggregate **per-domain** performance metrics from an item-level CSV into a
clean, nested JSON summary. This is intended to sit downstream of your
item-level aggregation (e.g., the output of `scoring_v2.py`), where each row
represents one (qid × model × setting) with already-computed metrics such as:

- `em_majority` (int {0,1} collapsed across runs via majority)
- `f1_median`   (float 0–1, per-item median across runs)
- `latency_median_ms` (float or int in milliseconds) — optional, but used if present

What this script does
---------------------
1) Loads a CSV containing columns at least:
      `qid`, `domain`, `model`, `setting`, `em_majority`, `f1_median`
   (If `latency_median_ms` exists, it will also be summarized.)
2) Groups rows by (`model`, `domain`, `setting`).
3) For each group, computes:
      • **em_mean**                — mean of `em_majority` × 100 (percent)
      • **f1_median**              — median of `f1_median` × 100 (percent)
      • **latency_median**         — median of `latency_median_ms` (if available)
      • **n_items**                — number of rows in the group
4) Writes a nested JSON:
      {
        "<model>": {
          "<domain>": {
            "<setting>": {
              "em_mean": <float percent>,
              "f1_median": <float percent>,
              "latency_median": <int ms or null>,
              "n_items": <int>
            },
            ...
          },
          ...
        },
        ...
      }

Design choices & notes
----------------------
- We report **EM as a percent** by averaging 0/1 over items (i.e., micro mean),
  which is a standard, stable summary.
- We report **F1 as the median of per-item medians** (then ×100 to percent).
  Using medians helps reduce sensitivity to outliers if there are multiple runs.
- `latency_median_ms` is OPTIONAL. If absent, `latency_median` is `null` in JSON.
- We round EM and F1 to 2 decimals, and latency to an integer (ms).

Usage
-----
python -m src.scoring.domain_scoring \
  --in-csv  path/to/aggregated_items_v2.csv \
  --out-json path/to/domain_summary.json

Inputs
------
--in-csv   : CSV produced by an upstream item aggregator (e.g., scoring_v2).
--out-json : Output JSON path (directories created if missing).

Common pitfalls
---------------
- Missing required columns: you must have at least
  `qid, domain, model, setting, em_majority, f1_median`.
- Ensure `em_majority` is numeric (0/1) and `f1_median` is in [0,1].

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Set

import pandas as pd


def _validate_required_columns(df: pd.DataFrame, required: Set[str], src: str) -> None:
    """
    Validate that the DataFrame contains all required columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame loaded from the CSV.
    required : set[str]
        Column names that must be present in `df`.
    src : str
        A friendly label for the source (usually the CLI argument path) used in
        error messages to help locate the problematic file.

    Raises
    ------
    ValueError
        If any required column is missing from `df`.
    """
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {src}: {sorted(missing)}")


def _aggregate_domain_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Group by (model, domain, setting) and compute summary statistics.

    For each group:
      - em_mean        : mean of `em_majority` × 100 (percent)
      - f1_median      : median of `f1_median` × 100 (percent)
      - latency_median : median of `latency_median_ms` (if present), else None
      - n_items        : count of rows in the group

    Parameters
    ----------
    df : pandas.DataFrame
        Item-level aggregated DataFrame with at least:
        `qid`, `domain`, `model`, `setting`, `em_majority`, `f1_median`.
        If present, `latency_median_ms` will be summarized.

    Returns
    -------
    dict
        Nested dictionary keyed by model → domain → setting → metrics dict.

    Notes
    -----
    - EM is averaged as a fraction then scaled to percent.
    - F1 is median-of-item-median (already per-item), then scaled to percent.
    - Latency is median in milliseconds and cast to int; if column is missing,
      latency_median is `None`.
    """
    has_latency = "latency_median_ms" in df.columns

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (model, domain, setting), group in df.groupby(["model", "domain", "setting"], dropna=False):
        # Coerce to numeric just in case, errors='coerce' turns bad cells to NaN
        em_series = pd.to_numeric(group["em_majority"], errors="coerce")
        f1_series = pd.to_numeric(group["f1_median"], errors="coerce")

        em_mean_pct = float(em_series.mean() * 100.0) if not em_series.empty else 0.0
        f1_median_pct = float(f1_series.median() * 100.0) if not f1_series.empty else 0.0

        latency_value = None
        if has_latency:
            lat_series = pd.to_numeric(group["latency_median_ms"], errors="coerce")
            if lat_series.notna().any():
                # median then cast to int ms
                latency_value = int(float(lat_series.median()))
            else:
                latency_value = None

        metrics = {
            "em_mean": round(em_mean_pct, 2),
            "f1_median": round(f1_median_pct, 2),
            "latency_median": latency_value,
            "n_items": int(len(group)),
        }

        results.setdefault(str(model), {}).setdefault(str(domain), {})[str(setting)] = metrics

    return results


def main() -> None:
    """
    CLI entry point.

    Reads a CSV of item-level scores (e.g., output from `scoring_v2.py`),
    aggregates metrics by (model, domain, setting), and writes a nested JSON
    summary to the path provided by `--out-json`.

    Command-line arguments
    ----------------------
    --in-csv : str (required)
        Path to the input CSV. Must contain:
        `qid`, `domain`, `model`, `setting`, `em_majority`, `f1_median`.
        If available, `latency_median_ms` will be summarized as well.
    --out-json : str (required)
        Path to the output JSON. Parent directories will be created if needed.

    Output JSON schema
    ------------------
    {
      "<model>": {
        "<domain>": {
          "<setting>": {
            "em_mean": <float percent>,        # e.g., 83.33
            "f1_median": <float percent>,      # e.g., 78.57
            "latency_median": <int ms|null>,   # e.g., 912
            "n_items": <int>                   # number of rows aggregated
          },
          ...
        },
        ...
      },
      ...
    }

    Raises
    ------
    ValueError
        If the input CSV is missing required columns.
    """
    parser = argparse.ArgumentParser(description="Aggregate per-domain scores from aggregated items CSV into JSON.")
    parser.add_argument("--in-csv", required=True, help="Path to aggregated items CSV (e.g., aggregated_items_v2.csv)")
    parser.add_argument("--out-json", required=True, help="Where to save per-domain JSON summary")
    args = parser.parse_args()

    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)

    # Validate required columns
    required_cols = {"qid", "domain", "model", "setting", "em_majority", "f1_median"}
    _validate_required_columns(df, required_cols, src=args.in_csv)

    # Aggregate
    results = _aggregate_domain_metrics(df)

    # Write JSON
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Wrote per-domain summary to {out_path.resolve()}")


if __name__ == "__main__":
    main()

