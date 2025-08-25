# src/scoring/bertscore_scoring.py
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import evaluate

# ---------------- Helpers ----------------
def load_gold(gold_csv: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns: qid -> {"answer": [canonical], "aliases": [alias1,...]}
    Aliases are parsed from pipe-separated 'aliases' column if present.
    """
    df = pd.read_csv(gold_csv)
    out = {}
    for _, r in df.iterrows():
        qid = str(r["qid"])
        canon = str(r["answer"]).strip()
        aliases_col = (str(r.get("aliases", "") or "")).strip()
        aliases = [a.strip() for a in aliases_col.split("|") if a.strip()] if aliases_col else []
        out[qid] = {"answer": [canon], "aliases": aliases}
    return out

def read_jsonl_many(glob_pattern: str) -> pd.DataFrame:
    rows = []
    for path in sorted(Path().glob(glob_pattern)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No JSONL matched: {glob_pattern}")
    return pd.DataFrame(rows)

def best_ref_bertscore_f1(preds: List[str], refs_lists: List[List[str]],
                          model_type="roberta-large", lang="en",
                          rescale_with_baseline=True) -> List[float]:
    m = evaluate.load("bertscore")
    flat_preds, flat_refs, idx = [], [], []
    cursor = 0
    for p, refs in zip(preds, refs_lists):
        refs = refs or [""]
        flat_preds.extend([p or ""] * len(refs))
        flat_refs.extend([r or "" for r in refs])
        idx.append((cursor, cursor + len(refs)))
        cursor += len(refs)
    res = m.compute(predictions=flat_preds, references=flat_refs,
                    model_type=model_type, lang=lang,
                    rescale_with_baseline=rescale_with_baseline)
    f1 = res["f1"]
    out = []
    for a, b in idx:
        seg = f1[a:b]
        out.append(max(seg) if seg else 0.0)
    return out

def pp(x: float) -> float:
    return 100.0 * x

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Independent BERTScore scoring")
    ap.add_argument("--glob", required=True, help="Glob for raw jsonl, e.g. src/results/gpt4o/*.jsonl")
    ap.add_argument("--gold-csv", required=True, help="data/mhqa_questions.csv")
    ap.add_argument("--model", required=True, help="model id (used in filenames)")
    ap.add_argument("--outdir", required=True, help="output directory for CSVs")
    ap.add_argument("--bertscore-model", default="roberta-large")
    ap.add_argument("--bertscore-lang", default="en")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    gold = load_gold(Path(args.gold_csv))
    df = read_jsonl_many(args.glob)

    # Expect columns from your runner: run_id, qid, domain, model, setting, output, latency_ms
    needed = {"run_id", "qid", "setting", "output"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in jsonl: {missing}")

    # Build references per row: canonical + aliases (best-match policy)
    refs_lists = []
    for q in df["qid"]:
        g = gold.get(str(q))
        if not g:
            refs_lists.append([""])
        else:
            refs_lists.append(g["answer"] + g["aliases"])

    # Per-request BERTScore-F1 (best reference)
    df["bertscore_f1_bestref"] = best_ref_bertscore_f1(
        preds=df["output"].astype(str).tolist(),
        refs_lists=refs_lists,
        model_type=args.bertscore_model,
        lang=args.bertscore_lang,
        rescale_with_baseline=True,
    )

    # Save per-run
    per_run_path = outdir / f"{args.model}_bertscore_per_run.csv"
    df.to_csv(per_run_path, index=False)

    # Aggregate to per-item (median across runs within setting)
    ag = (df.groupby(["qid", "setting"])
            .agg(bertscore_f1_median=("bertscore_f1_bestref", "median"))
            .reset_index())

    # Pivot to wide per item (one row per qid with four settings)
    wide = ag.pivot(index="qid", columns="setting", values="bertscore_f1_median")
    for s in ["gold", "para", "dist", "para_dist"]:
        if s not in wide.columns:
            wide[s] = None
    wide = wide.reset_index()

    # Drops vs GOLD (percentage points)
    wide["drop_bertscore_para_pp"] = pp((wide["para"] - wide["gold"]).fillna(0.0))
    wide["drop_bertscore_dist_pp"] = pp((wide["dist"] - wide["gold"]).fillna(0.0))
    wide["drop_bertscore_para_dist_pp"] = pp((wide["para_dist"] - wide["gold"]).fillna(0.0))

    per_item_path = outdir / f"{args.model}_bertscore_aggregated_items.csv"
    wide.rename(columns={
        "gold": "bertscore_gold",
        "para": "bertscore_para",
        "dist": "bertscore_dist",
        "para_dist": "bertscore_para_dist",
    }).to_csv(per_item_path, index=False)

    # Model×setting summary (median over items)
    def med(col): 
        return float(pd.Series(col).dropna().median()) if col is not None else None

    summary = pd.DataFrame([{
        "model": args.model,
        "bertscore_gold_median": med(wide["bertscore_gold"]),
        "bertscore_para_median": med(wide["bertscore_para"]),
        "bertscore_dist_median": med(wide["bertscore_dist"]),
        "bertscore_para_dist_median": med(wide["bertscore_para_dist"]),
        "drop_bertscore_para_median_pp": med(wide["drop_bertscore_para_pp"]),
        "drop_bertscore_dist_median_pp": med(wide["drop_bertscore_dist_pp"]),
        "drop_bertscore_para_dist_median_pp": med(wide["drop_bertscore_para_dist_pp"]),
    }])

    summary_path = outdir / f"{args.model}_bertscore_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("✅ BERTScore done")
    print("  Per-run:   ", per_run_path)
    print("  Per-item:  ", per_item_path)
    print("  Summary:   ", summary_path)

if __name__ == "__main__":
    main()