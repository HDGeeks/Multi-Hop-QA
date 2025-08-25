# src/scoring/scoring_v2_extended.py
import argparse, json
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict

# ---------- Normalization ----------
def normalize(text: str) -> str:
    if text is None:
        return ""
    t = text.lower().strip()
    t = re.sub(r"[.,!?;:'\"()\[\]]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def f1_score(pred: str, gold: str) -> float:
    p_tokens = pred.split()
    g_tokens = gold.split()
    if not p_tokens or not g_tokens:
        return 0.0
    common = set(p_tokens) & set(g_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(p_tokens)
    recall = len(common) / len(g_tokens)
    return 2 * precision * recall / (precision + recall)

# ---------- Gold Loader ----------
def load_gold(gold_csv: Path):
    df = pd.read_csv(gold_csv)
    gold_map = {}
    for _, row in df.iterrows():
        qid = str(row["qid"])
        ans = normalize(str(row["answer"]))
        aliases = [normalize(a) for a in str(row.get("aliases", "")).split("|") if a.strip()]
        refs = [ans] + aliases
        gold_map[qid] = {"answer": ans, "aliases": aliases, "refs": refs, "domain": row["domain"]}
    return gold_map

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Glob pattern for jsonl files")
    ap.add_argument("--gold-csv", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    # Load
    paths = sorted(Path().glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No files matched {args.glob}")

    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    gold_map = load_gold(Path(args.gold_csv))

    # Score
    ems, f1s = [], []
    results = []
    for _, row in df.iterrows():
        qid = str(row["qid"])
        pred = normalize(str(row.get("output", "")))
        refs = gold_map.get(qid, {}).get("refs", [""])
        best_em = 0
        best_f1 = 0.0
        for ref in refs:
            if pred == ref:
                best_em = 1
            best_f1 = max(best_f1, f1_score(pred, ref))
        results.append({
            "qid": qid,
            "domain": gold_map.get(qid, {}).get("domain", "unknown"),
            "setting": row["setting"],
            "em": best_em,
            "f1": best_f1,
        })
        ems.append(best_em); f1s.append(best_f1)

    scored = pd.DataFrame(results)

    # ---------- Aggregates ----------
    summary = {}

    # Overall
    summary["overall"] = {
        "exact_match_percent": 100 * sum(ems) / len(ems),
        "f1_percent": 100 * (sum(f1s) / len(f1s)),
        "n": len(ems),
    }

    # By setting
    by_setting = scored.groupby("setting").agg(
        em_mean=("em", "mean"),
        f1_mean=("f1", "mean")
    ).reset_index()
    summary["by_setting"] = {
        row["setting"]: {
            "em_percent": 100 * row["em_mean"],
            "f1_percent": 100 * row["f1_mean"],
        }
        for _, row in by_setting.iterrows()
    }

    # By domain
    by_domain = scored.groupby("domain").agg(
        em_mean=("em", "mean"),
        f1_mean=("f1", "mean")
    ).reset_index()
    summary["by_domain"] = {
        row["domain"]: {
            "em_percent": 100 * row["em_mean"],
            "f1_percent": 100 * row["f1_mean"],
        }
        for _, row in by_domain.iterrows()
    }

    # Perturbation drops (relative to gold)
    drops = {}
    gold_row = summary["by_setting"].get("gold", {})
    gold_em = gold_row.get("em_percent", 0)
    gold_f1 = gold_row.get("f1_percent", 0)
    for setting in ["para", "dist", "para_dist"]:
        if setting in summary["by_setting"]:
            drops[setting] = {
                "em_drop_pp": summary["by_setting"][setting]["em_percent"] - gold_em,
                "f1_drop_pp": summary["by_setting"][setting]["f1_percent"] - gold_f1,
            }
    summary["drops_vs_gold"] = drops

    # Save
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ scoring_v2_extended complete → {args.out_json}")

if __name__ == "__main__":
    main()