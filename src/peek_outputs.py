#!/usr/bin/env python3
"""
Peek top-N outputs from the first runs of multiple models.

Usage (default models & run=1, limit=50):
    python tools/peek_outputs.py

Custom:
    python tools/peek_outputs.py \
        --results-root src/results \
        --models gpt4o gpt4o_mini gemini_pro llama31_8b mistral7b \
        --run-id 1 \
        --limit 50 \
        --save-csv peek_run1_top50.csv
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import re

DEFAULT_MODELS = ["gpt4o", "gpt4o_mini", "gemini_pro", "llama31_8b", "mistral7b"]

TAG_RE = re.compile(r"<[^>]+>")

def read_jsonl_many(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        # skip bad lines but keep going
                        continue
        except FileNotFoundError:
            # ignore missing files silently
            continue
    return rows

def gather_model_rows(results_root: Path, model: str, run_id: int) -> pd.DataFrame:
    glob_pat = results_root / model / "*.jsonl"
    files = sorted(glob_pat.parent.glob(glob_pat.name))
    data = read_jsonl_many(files)
    if not data:
        return pd.DataFrame(columns=["model","qid","setting","output","ts","path"])

    df = pd.DataFrame(data)
    # only rows for the requested run_id
    if "run_id" in df.columns:
        df = df[df["run_id"] == run_id]

    # keep only a few columns
    keep = []
    for c in ["model","qid","setting","output","ts"]:
        if c in df.columns:
            keep.append(c)
    df = df[keep].copy() if keep else pd.DataFrame(columns=["model","qid","setting","output","ts"])

    # add a quick diagnostic: has_html
    df["has_html"] = df["output"].astype(str).str.contains(TAG_RE)
    return df

def main():
    ap = argparse.ArgumentParser(description="Peek top-N outputs of first runs across models.")
    ap.add_argument("--results-root", default="src/results_50", help="Root folder containing per-model JSONL outputs")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to include")
    ap.add_argument("--run-id", type=int, default=1, help="Which run_id to filter on")
    ap.add_argument("--limit", type=int, default=50, help="Max rows to show per model")
    ap.add_argument("--save-csv", default=None, help="Optional path to save the concatenated preview as CSV")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    all_previews = []

    for m in args.models:
        df = gather_model_rows(results_root, m, args.run_id)
        head = df.head(args.limit).copy()
        # Reorder for readability
        cols = [c for c in ["model","qid","setting","output","has_html","ts"] if c in head.columns]
        head = head[cols]
        all_previews.append(head)

        print("\n" + "="*80)
        print(f"{m} — run_id={args.run_id} — showing top {min(args.limit, len(df))} rows")
        print("="*80)
        if head.empty:
            print("(no rows found)")
        else:
            # pretty print with limited column width for output
            with pd.option_context(
                "display.max_rows", None,
                "display.max_colwidth", 120,
                "display.width", 160,
            ):
                print(head.to_string(index=False))

    if all_previews:
        concat = pd.concat(all_previews, ignore_index=True)
        if args.save_csv:
            outp = Path(args.save_csv)
            outp.parent.mkdir(parents=True, exist_ok=True)
            concat.to_csv(outp, index=False)
            print(f"\n✅ Saved preview CSV → {outp}")
        else:
            print("\n(Use --save-csv <path> to save this preview to a file.)")

if __name__ == "__main__":
    main()