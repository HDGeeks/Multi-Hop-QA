#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

DOM_ORDER = ["history","science","politics","geography","literature"]

def cap(dom: str) -> str:
    return dom[:1].upper() + dom[1:] if dom else dom

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-csv", required=True, help=".../metrics/aggregated_items_v2.csv")
    ap.add_argument("--out-json",  required=True, help=".../metrics/<model>_per_domain.json")
    args = ap.parse_args()

    df = pd.read_csv(args.items_csv)
    # expected cols: qid,domain,model,setting,em_majority,f1_median,...
    df["domain"] = df["domain"].str.lower()

    out = { "gold": {}, "para": {}, "dist": {}, "para_dist": {} }
    for setting, sdf in df.groupby("setting"):
        dom_stats = (sdf.groupby("domain")
                       .agg(em=("em_majority","mean"),
                            f1=("f1_median","median"))
                       .reset_index())
        # to percentages
        for _, r in dom_stats.iterrows():
            out.setdefault(setting, {})
            out[setting][cap(r["domain"])] = {
                "em": float(r["em"])*100.0,
                "f1": float(r["f1"])*100.0,
            }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✅ wrote {args.out_json}")

if __name__ == "__main__":
    main()