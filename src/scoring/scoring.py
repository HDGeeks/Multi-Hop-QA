# src/scoring/score_runs.py
import json
from pathlib import Path
import argparse
from collections import defaultdict, Counter
import statistics as stats
from typing import Dict, List, Tuple

import evaluate  # Hugging Face's metrics

# Our repo loaders (to get gold answers by qid)
from src.data.load_data import load_items

# -------------------------
# Helpers
# -------------------------
REFUSAL_MARKERS = [
    "i can't", "i cannot", "i’m unable", "i am unable",
    "as an ai", "i don't have", "cannot assist", "won't provide"
]

def is_refusal(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    return any(marker in t for marker in REFUSAL_MARKERS)

def is_invalid(text: str) -> bool:
    # For one-span answers, treat only truly empty outputs as invalid.
    # (We allow punctuation like '.' because some completions returned it.)
    return text.strip() == ""

def load_gold_map() -> Dict[str, List[str]]:
    """
    Build qid -> list-of-acceptable-answers
    from our data CSVs via load_items.
    """
    base = Path("src/data")
    questions_csv = base / "mhqa_questions.csv"
    context_csv   = base / "mhqa_context.csv"
    paras_csv     = base / "mhqa_paraphrases.csv"
    items = load_items(questions_csv, context_csv, paras_csv)

    gold_map: Dict[str, List[str]] = {}
    for it in items:
        # Try to be generous: if the dataclass exposes .answer or .answers/aliases, include them.
        # Fallback to a single canonical answer string if that’s all we have.
        # Adjust these field names if your Item schema differs.
        answers: List[str] = []
        if hasattr(it, "answers") and it.answers:
            # e.g., list[str]
            answers = [a for a in it.answers if a]
        elif hasattr(it, "aliases") and it.aliases:
            answers = [a for a in it.aliases if a]
        elif hasattr(it, "answer") and it.answer:
            answers = [it.answer]
        else:
            # Last resort: try 'gold_answer'
            if hasattr(it, "gold_answer") and it.gold_answer:
                answers = [it.gold_answer]

        if not answers:
            # If truly missing, set an impossible placeholder to avoid KeyErrors
            answers = ["__MISSING_GOLD__"]

        gold_map[it.qid] = answers
    return gold_map

def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def aggregate_latency(ms_list: List[int]) -> Dict[str, float]:
    if not ms_list:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0}
    ms_sorted = sorted(ms_list)
    n = len(ms_sorted)
    def pct(p):  # p in [0..100]
        if n == 1:
            return float(ms_sorted[0])
        k = max(0, min(n - 1, int(round((p/100.0)*(n-1)))))
        return float(ms_sorted[k])
    return {
        "avg_ms": float(sum(ms_sorted)/n),
        "p50_ms": pct(50),
        "p90_ms": pct(90),
    }

# -------------------------
# Core Scoring
# -------------------------
def score_files(
    files: List[Path],
    gold_map: Dict[str, List[str]],
) -> Dict:
    """
    Returns a nested dict with:
      - overall and by-setting EM/F1 (via HF 'squad')
      - refusal/invalid rates
      - latency stats
      - robustness delta across settings (per qid, EM difference gold vs others)
    """
    # Group predictions by (run_id, setting)
    # Also track model name consistency from the files
    predictions_by_setting: Dict[str, List[dict]] = defaultdict(list)
    golds_by_setting: Dict[str, List[dict]] = defaultdict(list)

    # For extra stats
    refusals_by_setting: Counter = Counter()
    invalids_by_setting: Counter = Counter()
    count_by_setting: Counter = Counter()
    latencies_by_setting: Dict[str, List[int]] = defaultdict(list)

    # Also build a per-qid map of correctness for robustness deltas
    # correctness[(setting)][qid] = 1 if EM==1 else 0
    correctness: Dict[str, Dict[str, int]] = defaultdict(dict)

    # Read all files and accumulate examples
    model_name_seen = set()
    runs_seen = set()

    for f in files:
        rows = read_jsonl(f)
        for r in rows:
            qid = r["qid"]
            setting = (r.get("setting") or "").lower()
            output = (r.get("output") or "").strip()
            model = r.get("model", "")
            latency_ms = int(r.get("latency_ms") or 0)
            run_id = r.get("run_id")
            model_name_seen.add(model)
            runs_seen.add(run_id)

            # gold(s)
            ref_answers = gold_map.get(qid, ["__MISSING_GOLD__"])

            # Build SQuAD-compatible dicts
            predictions_by_setting[setting].append(
                {"id": qid, "prediction_text": output}
            )
            golds_by_setting[setting].append(
                {"id": qid, "answers": {"text": ref_answers, "answer_start": [0]*len(ref_answers)}}
            )

            # Counters
            count_by_setting[setting] += 1
            latencies_by_setting[setting].append(latency_ms)
            if is_invalid(output):
                invalids_by_setting[setting] += 1
            if is_refusal(output):
                refusals_by_setting[setting] += 1

    # Compute EM/F1 per setting via HF 'squad'
    squad_metric = evaluate.load("squad")
    em_f1_by_setting = {}
    for setting, preds in predictions_by_setting.items():
        refs = golds_by_setting[setting]
        if not preds:
            em_f1_by_setting[setting] = {"exact_match": 0.0, "f1": 0.0}
            continue
        res = squad_metric.compute(predictions=preds, references=refs)
        em_f1_by_setting[setting] = res

        # Also populate correctness per qid for robustness delta
        # Recompute example-level EM (simple: 1 if pred matches any gold after SQuAD normalization).
        # We approximate by doing a per-example run (cheap for our size).
        # NOTE: evaluate.squad doesn't expose per-example; quick re-check:
        for p, g in zip(preds, refs):
            single = squad_metric.compute(predictions=[p], references=[g])
            correctness[setting][p["id"]] = 1 if single["exact_match"] == 100.0 else 0

    # Robustness deltas: Compare each non-GOLD against GOLD, average ΔEM per qid
    if "gold" in correctness:
        gold_corr = correctness["gold"]
    else:
        # If no gold key (should not happen), baseline zeros
        gold_corr = {qid: 0 for qid in set().union(*[set(d.keys()) for d in correctness.values()])}

    robustness_deltas = {}
    for setting in correctness.keys():
        if setting == "gold":
            continue
        deltas = []
        for qid, base in gold_corr.items():
            if qid in correctness[setting]:
                deltas.append(correctness[setting][qid] - base)
        # Average delta (in percentage points)
        robustness_deltas[setting] = 100.0 * (sum(deltas) / len(deltas)) if deltas else 0.0

    # Overall (micro) across settings: pool all predictions
    all_preds = []
    all_refs = []
    for setting in predictions_by_setting:
        all_preds.extend(predictions_by_setting[setting])
        all_refs.extend(golds_by_setting[setting])
    overall = {"exact_match": 0.0, "f1": 0.0}
    if all_preds:
        overall = squad_metric.compute(predictions=all_preds, references=all_refs)

    # Build summary
    model_name = ",".join(sorted(x for x in model_name_seen if x))
    summary = {
        "model": model_name,
        "num_runs": len(runs_seen),
        "num_files": len(files),
        "counts_by_setting": dict(count_by_setting),
        "em_f1_by_setting": em_f1_by_setting,  # {'gold': {'exact_match':..,'f1':..}, ...}
        "overall": overall,
        "invalid_rate_by_setting": {
            s: (100.0 * invalids_by_setting[s] / count_by_setting[s]) if count_by_setting[s] else 0.0
            for s in count_by_setting
        },
        "refusal_rate_by_setting": {
            s: (100.0 * refusals_by_setting[s] / count_by_setting[s]) if count_by_setting[s] else 0.0
            for s in count_by_setting
        },
        "latency_by_setting": {
            s: aggregate_latency(latencies_by_setting[s]) for s in latencies_by_setting
        },
        "robustness_delta_vs_gold_pp": robustness_deltas,  # ΔEM (percentage points) relative to GOLD
    }
    return summary

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Score run files with SQuAD EM/F1 + extra metrics.")
    parser.add_argument("--glob", required=True,
                        help="Glob for input JSONL files, e.g., 'src/results/raw/gpt4o_run*.jsonl'")
    parser.add_argument("--out-json", default=None,
                        help="Path to write JSON summary, e.g., src/results/metrics/gpt4o_summary.json")
    parser.add_argument("--out-csv", default=None,
                        help="Optional: write a wide CSV with per-setting EM/F1/invalid/refusal/latency.")
    args = parser.parse_args()

    files = sorted(Path().glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    gold_map = load_gold_map()
    summary = score_files(files, gold_map)

    # Ensure results dir
    default_json = Path("src/results/metrics/summary.json")
    out_json = Path(args.out_json) if args.out_json else default_json
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote JSON summary → {out_json}")

    # Optional CSV (compact, one row per setting)
    if args.out_csv:
        import csv
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        # Collect rows
        settings = sorted(summary["em_f1_by_setting"].keys())
        rows = []
        for s in settings:
            em_f1 = summary["em_f1_by_setting"][s]
            inv = summary["invalid_rate_by_setting"].get(s, 0.0)
            ref = summary["refusal_rate_by_setting"].get(s, 0.0)
            lat = summary["latency_by_setting"].get(s, {"avg_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0})
            row = {
                "model": summary["model"],
                "setting": s,
                "count": summary["counts_by_setting"].get(s, 0),
                "EM": em_f1.get("exact_match", 0.0),
                "F1": em_f1.get("f1", 0.0),
                "InvalidRate(%)": inv,
                "RefusalRate(%)": ref,
                "Latency.avg_ms": lat["avg_ms"],
                "Latency.p50_ms": lat["p50_ms"],
                "Latency.p90_ms": lat["p90_ms"],
                "ΔEM_vs_GOLD_pp": summary["robustness_delta_vs_gold_pp"].get(s, 0.0) if s != "gold" else 0.0,
            }
            rows.append(row)

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV summary → {out_csv}")

if __name__ == "__main__":
    main()