# src/scoring/extended_scoring.py

import argparse
import glob
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

# ------------------------------
# Normalization helpers
# ------------------------------
import re
PUNCT_RE = re.compile(r"[,\.!?;:'\"()\[\]]")

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = PUNCT_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

# ------------------------------
# Main Scoring Logic
# ------------------------------

def score_runs(jsonl_files, model_id, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in jsonl_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                qid = ex["qid"]
                domain = ex["domain"]
                run_id = ex["run_id"]
                setting = ex["setting"]
                output = ex.get("output", "")
                answer = ex.get("answer", "")  # might need to merge from gold file if not logged
                latency_ms = ex.get("latency_ms", -1)
                error = ex.get("error")
                finish_reason = ex.get("finish_reason")

                norm_pred = normalize_text(output)
                norm_gold = normalize_text(answer)

                em = 1 if norm_pred == norm_gold and norm_gold != "" else 0
                f1 = f1_score(output, answer)

                refusal = 1 if (not output or "cannot answer" in output.lower()) else 0
                format_violation = 1 if len(output.split()) > 10 else 0  # heuristic span rule

                rows.append({
                    "run_id": run_id,
                    "qid": qid,
                    "domain": domain,
                    "model": model_id,
                    "setting": setting,
                    "output": output,
                    "answer": answer,
                    "em": em,
                    "f1": f1,
                    "latency_ms": latency_ms,
                    "refusal": refusal,
                    "format_violation": format_violation,
                })

    df = pd.DataFrame(rows)
    df.sort_values(["qid", "setting", "run_id"], inplace=True)

    # ------------------------------
    # Save per-run scores
    # ------------------------------
    per_run_path = outdir / f"{model_id}_per_run.csv"
    df.to_csv(per_run_path, index=False)

    # ------------------------------
    # Aggregate per item (collapse runs)
    # ------------------------------
    agg_rows = []
    for (qid, setting), group in df.groupby(["qid", "setting"]):
        em_majority = 1 if group["em"].sum() >= 2 else 0
        f1_median = group["f1"].median()
        latency_median = group["latency_ms"].median()
        refusal_any = 1 if group["refusal"].any() else 0
        format_any = 1 if group["format_violation"].any() else 0
        stability = np.median(np.abs(group["f1"] - f1_median))

        agg_rows.append({
            "qid": qid,
            "domain": group["domain"].iloc[0],
            "model": model_id,
            "setting": setting,
            "em_majority": em_majority,
            "f1_median": f1_median,
            "latency_median_ms": latency_median,
            "refusal_any": refusal_any,
            "format_any": format_any,
            "stability_abs_dev": stability,
        })

    df_items = pd.DataFrame(agg_rows)
    agg_items_path = outdir / f"{model_id}_aggregated_items.csv"
    df_items.to_csv(agg_items_path, index=False)

    # ------------------------------
    # Aggregate per setting summary
    # ------------------------------
    summary_rows = []
    for setting, group in df_items.groupby("setting"):
        em_mean = group["em_majority"].mean() * 100
        f1_median = group["f1_median"].median() * 100
        latency_median = group["latency_median_ms"].median()
        refusal_rate = group["refusal_any"].mean() * 100
        format_rate = group["format_any"].mean() * 100
        stability_median = group["stability_abs_dev"].median()

        # Bootstrap CIs
        rng = np.random.default_rng(seed=42)
        em_ci = bootstrap((group["em_majority"].values,), np.mean, n_resamples=1000, random_state=rng).confidence_interval
        f1_ci = bootstrap((group["f1_median"].values,), np.median, n_resamples=1000, random_state=rng).confidence_interval

        summary_rows.append({
            "model": model_id,
            "setting": setting,
            "n_items": len(group),
            "em_mean": em_mean,
            "f1_median": f1_median,
            "latency_median_ms": latency_median,
            "refusal_rate": refusal_rate,
            "format_rate": format_rate,
            "stability_median": stability_median,
            "ci_em_low": em_ci.low * 100,
            "ci_em_high": em_ci.high * 100,
            "ci_f1_low": f1_ci.low * 100,
            "ci_f1_high": f1_ci.high * 100,
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = outdir / f"{model_id}_summary.csv"
    df_summary.to_csv(summary_path, index=False)

    print(f"✅ Scoring complete for {model_id}")
    print(f"  Per-run:   {per_run_path}")
    print(f"  Items:     {agg_items_path}")
    print(f"  Summary:   {summary_path}")


# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True, help="Glob for input JSONL files (3 runs)")
    parser.add_argument("--model", required=True, help="Model id (e.g., gpt4o)")
    parser.add_argument("--outdir", required=True, help="Directory to save outputs")
    args = parser.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files found with glob {args.glob}")

    score_runs(files, args.model, Path(args.outdir))

if __name__ == "__main__":
    main()