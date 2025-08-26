# src/scoring/scoring_v2_extended.py
import argparse, json, re
from pathlib import Path
from collections import Counter
import pandas as pd

# ---------- Normalization ----------
def normalize(text: str) -> str:
    if text is None:
        return ""
    t = text.lower().strip()
    t = re.sub(r"[.,!?;:'\"()\[\]]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def f1_score(pred: str, gold: str) -> float:
    """Token F1 using multiset overlap (proper precision/recall)."""
    p = pred.split()
    g = gold.split()
    if not p or not g:
        return 0.0
    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall    = overlap / len(g)
    return 2 * precision * recall / (precision + recall)

# ---------- Gold Loader ----------
def load_gold(gold_csv: Path):
    df = pd.read_csv(gold_csv)
    gold_map = {}
    for _, row in df.iterrows():
        qid = str(row["qid"])
        ans = normalize(str(row["answer"]))
        aliases_raw = str(row.get("aliases", "") or "")
        aliases = [normalize(a) for a in aliases_raw.split("|") if a.strip()]
        refs = [x for x in [ans, *aliases] if x] or [""]
        gold_map[qid] = {
            "answer": ans,
            "aliases": aliases,
            "refs": refs,
            "domain": row["domain"],
        }
    return gold_map

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Lightweight scoring (EM/F1), drops, domains.")
    ap.add_argument("--glob", required=True, help='e.g. "src/results_50/gpt4o/*.jsonl"')
    ap.add_argument("--gold-csv", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    paths = sorted(Path().glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No files matched {args.glob}")

    # Load raw rows
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("No rows found in matched JSONL files.")
    df = pd.DataFrame(rows)

    # Basic column sanity
    for col in ["qid", "setting", "output"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in input JSONL.")

    gold_map = load_gold(Path(args.gold_csv))

    # Per-response scoring
    scored_rows = []
    for _, r in df.iterrows():
        qid = str(r["qid"])
        setting = str(r["setting"]).lower().strip()
        pred = normalize(str(r.get("output", "")))
        gold_entry = gold_map.get(qid, {"refs": [""], "domain": "unknown"})
        refs = gold_entry["refs"] or [""]
        best_em = 0
        best_f1 = 0.0
        for ref in refs:
            if pred == ref:
                best_em = 1
            best_f1 = max(best_f1, f1_score(pred, ref))
        scored_rows.append({
            "qid": qid,
            "domain": gold_entry["domain"],
            "setting": setting,
            "em": best_em,
            "f1": best_f1,
        })
    scored = pd.DataFrame(scored_rows)

    # ---------- Aggregates ----------
    summary = {}

    # Overall micro
    summary["overall"] = {
        "exact_match_percent": 100.0 * scored["em"].mean(),
        "f1_percent":         100.0 * scored["f1"].mean(),
        "n": int(len(scored)),
    }

    # By setting (ensure all 4 keys exist)
    by_setting = scored.groupby("setting").agg(
        em_mean=("em", "mean"),
        f1_mean=("f1", "mean"),
        n=("qid", "count")
    )
    settings = ["gold", "para", "dist", "para_dist"]
    summary["by_setting"] = {}
    for s in settings:
        if s in by_setting.index:
            row = by_setting.loc[s]
            summary["by_setting"][s] = {
                "em_percent": 100.0 * float(row["em_mean"]),
                "f1_percent": 100.0 * float(row["f1_mean"]),
                "n": int(row["n"]),
            }
        else:
            summary["by_setting"][s] = {"em_percent": None, "f1_percent": None, "n": 0}

    # By domain (micro across all settings)
    by_domain = scored.groupby("domain").agg(
        em_mean=("em", "mean"),
        f1_mean=("f1", "mean"),
        n=("qid", "count")
    ).reset_index()
    summary["by_domain"] = {
        row["domain"]: {
            "em_percent": 100.0 * float(row["em_mean"]),
            "f1_percent": 100.0 * float(row["f1_mean"]),
            "n": int(row["n"]),
        }
        for _, row in by_domain.iterrows()
    }

    # Drops vs GOLD (positive = degradation in pp)
    drops = {}
    gold_em = summary["by_setting"]["gold"]["em_percent"] or 0.0
    gold_f1 = summary["by_setting"]["gold"]["f1_percent"] or 0.0
    for s in ["para", "dist", "para_dist"]:
        s_em = summary["by_setting"][s]["em_percent"]
        s_f1 = summary["by_setting"][s]["f1_percent"]
        if s_em is None or s_f1 is None:
            drops[s] = {"em_drop_pp": None, "f1_drop_pp": None}
        else:
            drops[s] = {
                "em_drop_pp": gold_em - s_em,  # positive is worse
                "f1_drop_pp": gold_f1 - s_f1,
            }
    summary["drops_vs_gold"] = drops

    # Save
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ scoring_v2_extended complete → {out_path}")

if __name__ == "__main__":
    main()