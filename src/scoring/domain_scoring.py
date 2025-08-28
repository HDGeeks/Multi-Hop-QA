# src/scoring/domain_scoring.py
"""
domain_scoring.py

This module aggregates per-domain performance metrics from a CSV file containing item-level scores, and outputs a structured JSON summary.
It is designed for evaluating models across different domains and settings in multi-hop question answering tasks.

Main Functionality:
-------------------
- Reads a CSV file with expected columns: 'qid', 'domain', 'model', 'setting', 'em_majority', 'f1_median', 'latency_median_ms', etc.
- Groups data by 'model', 'domain', and 'setting'.
- Computes:
    - Mean Exact Match (EM) score (as a percentage)
    - Median F1 score (as a percentage)
    - Median latency (in milliseconds)
    - Number of items per group
- Outputs a nested JSON file summarizing these metrics per model, domain, and setting.

Functions:
----------
- main(): Parses command-line arguments, performs aggregation, and writes the output JSON.

Usage:
------
Run from the command line:

    python domain_scoring.py --in-csv path/to/scores_aggregated_items.csv --out-json path/to/output_summary.json

Arguments:
----------
--in-csv      (required) : Path to the input CSV file containing aggregated item scores.
--out-json    (required) : Path to the output JSON file for saving the per-domain summary.

Inputs:
-------
- CSV file with required columns: 'qid', 'domain', 'model', 'setting', 'em_majority', 'f1_median', 'latency_median_ms'.

Outputs:
--------
- JSON file containing nested dictionaries with aggregated metrics per model, domain, and setting.

Example:
--------
Suppose you have a CSV file at './data/scores_aggregated_items.csv' and want to save the summary to './results/domain_summary.json':

    python domain_scoring.py --in-csv ./data/scores_aggregated_items.csv --out-json ./results/domain_summary.json

Notes:
------
- The script will raise an error if required columns are missing from the input CSV.
- Output JSON will be created with parent directories if they do not exist.
"""


import pandas as pd
from pathlib import Path
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Aggregate per-domain scores from aggregated items")
    parser.add_argument("--in-csv", required=True, help="Path to scores_aggregated_items.csv")
    parser.add_argument("--out-json", required=True, help="Where to save per-domain JSON summary")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    # Expected columns: qid, domain, model, setting, em_majority, f1_median, latency_median_ms, ...
    required_cols = {"qid", "domain", "model", "setting", "em_majority", "f1_median"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in {args.in_csv}: {required_cols - set(df.columns)}")

    results = {}
    for (model, domain, setting), group in df.groupby(["model", "domain", "setting"]):
        em_mean = group["em_majority"].mean() * 100  # %
        f1_median = group["f1_median"].median() * 100  # %
        latency_median = group["latency_median_ms"].median()

        results.setdefault(model, {}).setdefault(domain, {})[setting] = {
            "em_mean": round(em_mean, 2),
            "f1_median": round(f1_median, 2),
            "latency_median": int(latency_median),
            "n_items": len(group),
        }

    # Save JSON
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Wrote per-domain summary to {out_path.resolve()}")

if __name__ == "__main__":
    main()