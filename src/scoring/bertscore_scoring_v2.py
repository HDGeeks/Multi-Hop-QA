# src/scoring/bertscore_scoring_v2.py
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import evaluate

# -------------------------
# Helpers
# -------------------------
def read_jsonl_many(glob_pattern: str) -> pd.DataFrame:
    rows = []
    for path in sorted(Path().glob(glob_pattern)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No JSONL matched: {glob_pattern}")
    return pd.DataFrame(rows)

def load_gold(gold_csv: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    qid -> { 'answer': [canonical], 'aliases': [a1, ...], 'domain': '...' }
    """
    df = pd.read_csv(gold_csv)
    out: Dict[str, Dict[str, List[str]]] = {}
    for _, r in df.iterrows():
        qid = str(r["qid"])
        canon = str(r["answer"]).strip()
        aliases_col = str(r.get("aliases", "") or "")
        aliases = [a.strip() for a in aliases_col.split("|") if a.strip()]
        out[qid] = {"answer": [canon], "aliases": aliases, "domain": str(r["domain"]).strip()}
    return out

def best_ref_bertscore_f1(
    preds: List[str],
    refs_lists: List[List[str]],
    model_type: str = "roberta-large",
    lang: str = "en",
    rescale_with_baseline: bool = True,
) -> List[float]:
    """
    For each prediction p and list of references [r1..rk], compute BERTScore F1
    for (p, ri) for all i and take the max. Returns list of length len(preds).
    """
    metric = evaluate.load("bertscore")
    flat_preds, flat_refs, idx = [], [], []
    cursor = 0
    for p, refs in zip(preds, refs_lists):
        refs = refs or [""]
        flat_preds.extend([p or ""] * len(refs))
        flat_refs.extend([r or ""] for r in refs)
        idx.append((cursor, cursor + len(refs)))
        cursor += len(refs)

    # evaluate expects references as list[str], not list[list[str]]
    flat_refs = [r for r in flat_refs]

    res = metric.compute(
        predictions=flat_preds,
        references=flat_refs,
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
    )
    f1 = res["f1"]  # list of floats aligned to flat pairs

    out = []
    for a, b in idx:
        seg = f1[a:b]
        out.append(max(seg) if seg else 0.0)
    return out

def pp(x: float) -> float:
    return 100.0 * x

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Independent BERTScore v2 (per-run, per-item, summaries + drops).")
    ap.add_argument("--glob", required=True, help='Glob for raw jsonl, e.g. "src/results_50/gpt4o/*.jsonl"')
    ap.add_argument("--gold-csv", required=True, help='e.g. "src/data_50/mhqa_questions_50.csv"')
    ap.add_argument("--model", required=True, help="model id for filenames (e.g., gpt4o)")
    ap.add_argument("--outdir", required=True, help="directory to write outputs")
    ap.add_argument("--bertscore-model", default="roberta-large")
    ap.add_argument("--bertscore-lang", default="en")
    ap.add_argument("--no-rescale", action="store_true", help="disable rescale_with_baseline")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    gold = load_gold(Path(args.gold_csv))
    df = read_jsonl_many(args.glob)

    # Expect columns: run_id, qid, domain, model, setting, output, latency_ms
    needed = {"run_id", "qid", "setting", "output"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in jsonl: {missing}")

    # Build reference lists (canonical + aliases) and attach domain
    refs_lists, domains = [], []
    for q in df["qid"]:
        g = gold.get(str(q))
        if not g:
            refs_lists.append([""])
            domains.append("unknown")
        else:
            refs_lists.append(g["answer"] + g["aliases"])
            domains.append(g["domain"])
    df["domain_gold"] = domains

    # --- Per-request BERTScore-F1 (best ref) ---
    df["bertscore_f1"] = best_ref_bertscore_f1(
        preds=df["output"].astype(str).tolist(),
        refs_lists=refs_lists,
        model_type=args.bertscore_model,
        lang=args.bertscore_lang,
        rescale_with_baseline=not args.no_rescale,
    )

    # Save per-run (raw rows + bertscore)
    per_run_csv = outdir / f"{args.model}_bertscore_per_run_v2.csv"
    df.to_csv(per_run_csv, index=False)

    # --- Aggregate to per-item (median across runs within qid×setting) ---
    ag = (
        df.groupby(["qid", "domain_gold", "setting"], as_index=False)
          .agg(bertscore_f1_median=("bertscore_f1", "median"))
    )

        # Pivot to wide (one row per qid with four settings)
    wide = ag.pivot(index=["qid", "domain_gold"], columns="setting", values="bertscore_f1_median")

    # Ensure all four columns exist
    for s in ["gold", "para", "dist", "para_dist"]:
        if s not in wide.columns:
            wide[s] = None

    # Reset index and standardize column names
    wide = wide.reset_index().rename(columns={"domain_gold": "domain"})
    renamed = wide.rename(columns={
        "gold": "bertscore_gold",
        "para": "bertscore_para",
        "dist": "bertscore_dist",
        "para_dist": "bertscore_para_dist",
    })

    # Drops vs GOLD (percentage points) — compute on *renamed*
    renamed["drop_bertscore_para_pp"]      = 100.0 * (renamed["bertscore_para"]      - renamed["bertscore_gold"]).fillna(0.0)
    renamed["drop_bertscore_dist_pp"]      = 100.0 * (renamed["bertscore_dist"]      - renamed["bertscore_gold"]).fillna(0.0)
    renamed["drop_bertscore_para_dist_pp"] = 100.0 * (renamed["bertscore_para_dist"] - renamed["bertscore_gold"]).fillna(0.0)

    # Save per-item CSV
    per_item_csv = outdir / f"{args.model}_bertscore_aggregated_items_v2.csv"
    renamed.to_csv(per_item_csv, index=False)

    # --- Summaries (use 'renamed') ---
    def med(series):
        s = pd.Series(series).dropna()
        return float(s.median()) if not s.empty else None

    # By setting (median over qids)
    by_setting = {
        "gold":      med(renamed["bertscore_gold"]),
        "para":      med(renamed["bertscore_para"]),
        "dist":      med(renamed["bertscore_dist"]),
        "para_dist": med(renamed["bertscore_para_dist"]),
    }
    # Drops (median over qids)
    drops = {
        "para":      med(renamed["drop_bertscore_para_pp"]),
        "dist":      med(renamed["drop_bertscore_dist_pp"]),
        "para_dist": med(renamed["drop_bertscore_para_dist_pp"]),
    }

    # By domain (median GOLD/DIST and drop)
    domain_rows = []
    for dom, sub in renamed.groupby("domain"):
        domain_rows.append({
            "domain": dom,
            "bertscore_gold_median": med(sub["bertscore_gold"]),
            "bertscore_dist_median": med(sub["bertscore_dist"]),
            "drop_dist_pp_median":   med(sub["drop_bertscore_dist_pp"]),
        })
    by_domain_df = pd.DataFrame(domain_rows).sort_values("domain")
    by_domain_csv = outdir / f"{args.model}_bertscore_by_domain_v2.csv"
    by_domain_df.to_csv(by_domain_csv, index=False)

    # JSON summary (compact)
    summary = {
        "model": args.model,
        "n_requests": int(len(df)),
        "n_items": int(renamed.shape[0]),
        "bertscore_by_setting_median": {k: (100.0 * v if v is not None else None) for k, v in by_setting.items()},
        "bertscore_drop_vs_gold_median_pp": drops,
    }
    summary_json = outdir / f"{args.model}_bertscore_summary_v2.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("✅ BERTScore v2 done")
    print("  Per-run:   ", per_run_csv)
    print("  Per-item:  ", per_item_csv)
    print("  By-domain: ", by_domain_csv)
    print("  Summary:   ", summary_json)

if __name__ == "__main__":
    main()