import argparse, json, re, string
from pathlib import Path
import pandas as pd
from collections import Counter

# ---------------- Normalization helpers ----------------
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def normalize(text: str) -> str:
    if text is None:
        return ""
    return text.lower().translate(PUNCT_TABLE).strip()

def tokenize(text: str):
    return normalize(text).split()

def f1_score(pred: str, golds: list[str]) -> float:
    """
    Compute token-level F1 between prediction and the *best* matching gold/alias.
    """
    pred_tokens = tokenize(pred)
    if not pred_tokens:
        return 0.0

    best = 0.0
    for g in golds:
        gold_tokens = tokenize(g)
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        best = max(best, f1)
    return best

def exact_match(pred: str, golds: list[str]) -> int:
    """
    Exact match: 1 if normalized pred == any normalized gold/alias.
    """
    pred_norm = normalize(pred)
    for g in golds:
        if pred_norm == normalize(g):
            return 1
    return 0

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Glob for jsonl results")
    ap.add_argument("--gold-csv", required=True, help="Questions file with answers+aliases")
    ap.add_argument("--out-json", required=True, help="Where to write summary JSON")
    args = ap.parse_args()

    # Load golds
    gold = {}
    df_gold = pd.read_csv(args.gold_csv)
    for _, r in df_gold.iterrows():
        qid = r["qid"]
        canon = str(r["answer"]).strip()
        aliases = []
        if "aliases" in r and isinstance(r["aliases"], str):
            aliases = [a.strip() for a in r["aliases"].split("|") if a.strip()]
        gold[qid] = [canon] + aliases

    # Load predictions
    rows = []
    for path in Path().glob(args.glob):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    # Compute EM/F1
    ems, f1s = [], []
    for _, r in df.iterrows():
        qid = r["qid"]
        pred = str(r["output"])
        refs = gold.get(qid, [""])
        em = exact_match(pred, refs)
        f1 = f1_score(pred, refs)
        ems.append(em)
        f1s.append(f1)

    df["em_v2"] = ems
    df["f1_v2"] = f1s

    # Aggregate
    summary = {
        "exact_match_percent": 100 * df["em_v2"].mean(),
        "f1_percent": 100 * df["f1_v2"].mean(),
        "n": len(df),
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("✅ scoring_v2 complete")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()