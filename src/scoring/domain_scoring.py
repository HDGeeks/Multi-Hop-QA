# src/scoring/domain_scoring.py
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