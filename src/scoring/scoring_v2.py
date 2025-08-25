# src/scoring/scoring_v2.py
import argparse, json, re
from pathlib import Path
from collections import defaultdict, Counter
import statistics as stats
from typing import Dict, List

from src.data.load_data import load_items

# -------------------------
# Normalization
# -------------------------
PUNCT_RE = re.compile(r"[.,!?;:'\"()“”’]")

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = PUNCT_RE.sub("", s)
    s = " ".join(s.split())
    return s

def tokenize(s: str) -> List[str]:
    return normalize_text(s).split()

# -------------------------
# EM + F1 helpers
# -------------------------
def exact_match(pred: str, golds: List[str]) -> bool:
    norm_pred = normalize_text(pred)
    return any(norm_pred == normalize_text(g) for g in golds)

def f1_score(pred: str, golds: List[str]) -> float:
    """
    Token-level F1 between prediction and the best-matching gold.
    """
    pred_toks = tokenize(pred)
    if not pred_toks:
        return 0.0

    def f1_pair(gt: str) -> float:
        gold_toks = tokenize(gt)
        if not gold_toks:
            return 0.0
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)
        return 2 * precision * recall / (precision + recall)

    return max(f1_pair(g) for g in golds)

# -------------------------
# Load gold map
# -------------------------
def load_gold_map(gold_csv: Path, context_csv: Path, paras_csv: Path) -> Dict[str, List[str]]:
    items = load_items(gold_csv, context_csv, paras_csv)
    gold_map: Dict[str, List[str]] = {}
    for it in items:
        refs = [it.answer]
        if it.aliases:
            refs.extend(list(it.aliases))
        gold_map[it.qid] = refs
    return gold_map

# -------------------------
# Core scoring
# -------------------------
def score_jsonl_files(files: List[Path], gold_map: Dict[str, List[str]]) -> Dict:
    counts_by_setting = Counter()
    em_by_setting = defaultdict(list)
    f1_by_setting = defaultdict(list)
    latencies_by_setting = defaultdict(list)

    refusals_by_setting = Counter()
    invalids_by_setting = Counter()

    for f in files:
        for line in f.open("r", encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row["qid"]
            setting = row["setting"].lower()
            pred = row.get("output", "").strip()
            refs = gold_map.get(qid, ["__MISSING__"])

            # Metrics
            em = 1.0 if exact_match(pred, refs) else 0.0
            f1 = f1_score(pred, refs)

            em_by_setting[setting].append(em)
            f1_by_setting[setting].append(f1)

            counts_by_setting[setting] += 1
            latencies_by_setting[setting].append(int(row.get("latency_ms", 0)))

            # Invalid/refusal
            if pred.strip() == "":
                invalids_by_setting[setting] += 1
            if pred.lower().startswith("i cannot") or pred.lower().startswith("as an ai"):
                refusals_by_setting[setting] += 1

    # Aggregate
    summary = {
        "counts_by_setting": dict(counts_by_setting),
        "em_f1_by_setting": {
            s: {
                "exact_match": 100.0 * (sum(em_by_setting[s]) / len(em_by_setting[s])),
                "f1": 100.0 * (sum(f1_by_setting[s]) / len(f1_by_setting[s]))
            }
            for s in counts_by_setting
        },
        "invalid_rate_by_setting": {
            s: 100.0 * invalids_by_setting[s] / counts_by_setting[s]
            for s in counts_by_setting
        },
        "refusal_rate_by_setting": {
            s: 100.0 * refusals_by_setting[s] / counts_by_setting[s]
            for s in counts_by_setting
        },
        "latency_by_setting": {
            s: {
                "avg_ms": stats.mean(latencies_by_setting[s]) if latencies_by_setting[s] else 0.0,
                "p50_ms": stats.median(latencies_by_setting[s]) if latencies_by_setting[s] else 0.0,
                "p90_ms": sorted(latencies_by_setting[s])[int(0.9*len(latencies_by_setting[s]))] if latencies_by_setting[s] else 0.0,
            }
            for s in counts_by_setting
        }
    }

    # Overall
    all_em = [x for xs in em_by_setting.values() for x in xs]
    all_f1 = [x for xs in f1_by_setting.values() for x in xs]
    summary["overall"] = {
        "exact_match": 100.0 * sum(all_em)/len(all_em) if all_em else 0.0,
        "f1": 100.0 * sum(all_f1)/len(all_f1) if all_f1 else 0.0
    }

    return summary

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Glob for jsonl files, e.g. src/results_50/gpt4o/*.jsonl")
    ap.add_argument("--gold-csv", required=True)
    ap.add_argument("--context-csv", required=True)
    ap.add_argument("--paras-csv", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    files = sorted(Path().glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    gold_map = load_gold_map(Path(args.gold_csv), Path(args.context_csv), Path(args.paras_csv))
    summary = score_jsonl_files(files, gold_map)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ scoring_v2 complete → {out_path}")

if __name__ == "__main__":
    main()

    
# import argparse, json, re, string
# from pathlib import Path
# import pandas as pd
# from collections import Counter

# # ---------------- Normalization helpers ----------------
# PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# def normalize(text: str) -> str:
#     if text is None:
#         return ""
#     return text.lower().translate(PUNCT_TABLE).strip()

# def tokenize(text: str):
#     return normalize(text).split()

# def f1_score(pred: str, golds: list[str]) -> float:
#     """
#     Compute token-level F1 between prediction and the *best* matching gold/alias.
#     """
#     pred_tokens = tokenize(pred)
#     if not pred_tokens:
#         return 0.0

#     best = 0.0
#     for g in golds:
#         gold_tokens = tokenize(g)
#         if not gold_tokens:
#             continue
#         common = Counter(pred_tokens) & Counter(gold_tokens)
#         num_same = sum(common.values())
#         if num_same == 0:
#             continue
#         precision = num_same / len(pred_tokens)
#         recall = num_same / len(gold_tokens)
#         f1 = (2 * precision * recall) / (precision + recall)
#         best = max(best, f1)
#     return best

# def exact_match(pred: str, golds: list[str]) -> int:
#     """
#     Exact match: 1 if normalized pred == any normalized gold/alias.
#     """
#     pred_norm = normalize(pred)
#     for g in golds:
#         if pred_norm == normalize(g):
#             return 1
#     return 0

# # ---------------- Main ----------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--glob", required=True, help="Glob for jsonl results")
#     ap.add_argument("--gold-csv", required=True, help="Questions file with answers+aliases")
#     ap.add_argument("--out-json", required=True, help="Where to write summary JSON")
#     args = ap.parse_args()

#     # Load golds
#     gold = {}
#     df_gold = pd.read_csv(args.gold_csv)
#     for _, r in df_gold.iterrows():
#         qid = r["qid"]
#         canon = str(r["answer"]).strip()
#         aliases = []
#         if "aliases" in r and isinstance(r["aliases"], str):
#             aliases = [a.strip() for a in r["aliases"].split("|") if a.strip()]
#         gold[qid] = [canon] + aliases

#     # Load predictions
#     rows = []
#     for path in Path().glob(args.glob):
#         with open(path, "r", encoding="utf-8") as f:
#             for line in f:
#                 rows.append(json.loads(line))
#     df = pd.DataFrame(rows)

#     # Compute EM/F1
#     ems, f1s = [], []
#     for _, r in df.iterrows():
#         qid = r["qid"]
#         pred = str(r["output"])
#         refs = gold.get(qid, [""])
#         em = exact_match(pred, refs)
#         f1 = f1_score(pred, refs)
#         ems.append(em)
#         f1s.append(f1)

#     df["em_v2"] = ems
#     df["f1_v2"] = f1s

#     # Aggregate
#     summary = {
#         "exact_match_percent": 100 * df["em_v2"].mean(),
#         "f1_percent": 100 * df["f1_v2"].mean(),
#         "n": len(df),
#     }

#     with open(args.out_json, "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2)

#     print("✅ scoring_v2 complete")
#     print(json.dumps(summary, indent=2))

# if __name__ == "__main__":
#     main()