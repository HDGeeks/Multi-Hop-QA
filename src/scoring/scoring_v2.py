# src/scoring/scoring_v2.py
# v2: alias-aware EM + token F1 with sane normalization, repeat-collapsed items, and summaries.
import argparse, json, re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# -------- Normalization -------
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def f1_tokens(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    pc = Counter(p); gc = Counter(g)
    overlap = sum((pc & gc).values())
    if overlap == 0:   return 0.0
    prec = overlap / len(p)
    rec  = overlap / len(g)
    return 2 * prec * rec / (prec + rec)

def em_any_alias(pred: str, refs: List[str]) -> int:
    p = normalize_text(pred)
    for r in refs:
        if p == normalize_text(r):
            return 1
    return 0

def best_f1_over_aliases(pred: str, refs: List[str]) -> float:
    if not refs: return 0.0
    return max(f1_tokens(pred, r) for r in refs)

# -------- Data loading (gold refs from CSVs) -------
def load_gold(questions_csv: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(questions_csv)
    gold: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        qid = str(r["qid"])
        canon = str(r["answer"]).strip()
        aliases = str(r.get("aliases", "") or "")
        alias_list = [a.strip() for a in aliases.split("|") if a.strip()]
        gold[qid] = [canon] + alias_list
    return gold

# -------- Scoring core -------
def score_df(df: pd.DataFrame, gold_map: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Adds columns: em, f1, refusal, invalid
    Keeps: run_id, qid, domain, model, setting, output, latency_ms
    """
    def refs(qid: str) -> List[str]:
        return gold_map.get(str(qid), [])

    def is_refusal(txt: str) -> int:
        t = (txt or "").strip().lower()
        if not t: return 0
        markers = ["i can't", "i cannot", "i’m unable", "i am unable",
                   "as an ai", "i don't have", "cannot assist", "won't provide"]
        return int(any(m in t for m in markers))

    def is_invalid(txt: str) -> int:
        return int((txt or "").strip() == "")

    ems, f1s, ref_flags, inv_flags = [], [], [], []
    for _, r in df.iterrows():
        out = str(r.get("output", "") or "")
        R = refs(r["qid"])
        ems.append(em_any_alias(out, R))
        f1s.append(best_f1_over_aliases(out, R))
        ref_flags.append(is_refusal(out))
        inv_flags.append(is_invalid(out))

    df = df.copy()
    df["em"] = ems
    df["f1"] = f1s
    df["refusal"] = ref_flags
    df["invalid"] = inv_flags
    return df

def aggregate_items(per_run: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse repeats per (qid, setting) using:
      - em_majority (>=2/3 wins)
      - f1_median
      - latency_median_ms
      - stability_mad_f1 (median absolute deviation of F1 across runs)
      - refusal_any / invalid_any (any run)
    """
    def em_majority(s: pd.Series) -> int:
        # s is the 'em' Series for the group
        return int(s.sum() >= 2)  # majority over 3 runs

    def mad(s: pd.Series) -> float:
        # s is the 'f1' Series for the group
        vals = s.values
        med = np.median(vals)
        return float(np.median(np.abs(vals - med)))

    ag = (per_run
          .groupby(["qid","domain","model","setting"], as_index=False)
          .agg(em_majority=("em", em_majority),
               f1_median=("f1", "median"),
               latency_median_ms=("latency_ms", "median"),
               refusal_any=("refusal", "max"),
               invalid_any=("invalid", "max"),
               stability_mad_f1=("f1", mad)))
    return ag

def summarize_settings(items_ag: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (model, setting) with item-level medians:
      EM (mean of em_majority), F1 median, latency median.
      Drops vs GOLD (ΔEM pp, ΔF1 pp) are computed later per model by merge.
    """
    base = (items_ag
        .groupby(["model","setting"], as_index=False)
        .agg(em_mean=("em_majority","mean"),
             f1_median=("f1_median","median"),
             latency_median_ms=("latency_median_ms","median"),
             refusal_rate=("refusal_any","mean"),
             invalid_rate=("invalid_any","mean"),
             stability_median=("stability_mad_f1","median")))
    # Convert to percentages for readability
    for c in ["em_mean","refusal_rate","invalid_rate"]:
        base[c] = 100.0 * base[c]
    base["f1_median"] = 100.0 * base["f1_median"]
    return base

def add_drops(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add Δ vs GOLD per model (percentage points) for EM and F1.
    """
    wide = summary.pivot(index="model", columns="setting", values=["em_mean","f1_median"])
    rows = []
    for model, sub in wide.groupby(level=0):
        def v(metric, setting):
            try:
                return float(sub.loc[(model,), (metric, setting)])
            except Exception:
                return np.nan
        gold_em = v("em_mean","gold"); gold_f1 = v("f1_median","gold")
        for setting in ["para","dist","para_dist"]:
            rows.append({
                "model": model,
                "setting": setting,
                "em_mean": v("em_mean", setting),
                "f1_median": v("f1_median", setting),
                "drop_em_pp": (v("em_mean", setting) - gold_em) if pd.notna(gold_em) else np.nan,
                "drop_f1_pp": (v("f1_median", setting) - gold_f1) if pd.notna(gold_f1) else np.nan,
            })
    drops = pd.DataFrame(rows)
    # Merge drops back
    base = summary.merge(drops, on=["model","setting"], how="left")
    base["drop_em_pp"] = base["drop_em_pp"].fillna(0.0)
    base["drop_f1_pp"] = base["drop_f1_pp"].fillna(0.0)
    return base

# -------- CLI -------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help='e.g. "src/results_50/gpt4o/*.jsonl"')
    ap.add_argument("--gold-csv", required=True, help="questions CSV (with answer+aliases)")
    ap.add_argument("--context-csv", required=True, help="(unused for scoring; required for reproducibility)")
    ap.add_argument("--paras-csv", required=True, help="(unused for scoring; required for reproducibility)")
    ap.add_argument("--out-json", required=True, help="summary JSON path")
    args = ap.parse_args()

    files = sorted(Path().glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    gold_map = load_gold(Path(args.gold_csv))

    # -------- load all runs --------
    rows = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    raw = pd.DataFrame(rows)

    # Minimal required columns check
    need = {"run_id","qid","domain","model","setting","output","latency_ms"}
    miss = need - set(raw.columns)
    if miss:
        raise SystemExit(f"Missing columns in jsonl: {miss}")

    # -------- per-run scoring --------
    per_run = score_df(raw, gold_map)
    # write per-run CSV next to out-json
    outdir = Path(args.out_json).parent
    outdir.mkdir(parents=True, exist_ok=True)
    model = str(per_run["model"].iloc[0])
    per_run_csv = outdir / f"{model}_per_run_v2.csv"
    per_run.to_csv(per_run_csv, index=False)

    # -------- per-item aggregation (collapse repeats) --------
    items_ag = aggregate_items(per_run)
    items_csv = outdir / f"{model}_aggregated_items_v2.csv"
    items_ag.to_csv(items_csv, index=False)

    # -------- per-setting summary + drops --------
    summary0 = summarize_settings(items_ag)
    summary = add_drops(summary0)
    summary_csv = outdir / f"{model}_summary_v2.csv"
    summary.to_csv(summary_csv, index=False)

    # -------- compact JSON (for dashboards) --------
    # overall micro (for sanity)
    overall_em = 100.0 * per_run["em"].mean() if len(per_run) else 0.0
    overall_f1 = 100.0 * per_run["f1"].mean() if len(per_run) else 0.0

    out = {
        "model": model,
        "n_rows": int(len(per_run)),
        "overall": {"exact_match_percent": overall_em, "f1_percent": overall_f1},
        "by_setting": (
            summary
            .loc[:, ["setting","em_mean","f1_median","latency_median_ms","refusal_rate","invalid_rate",
                     "drop_em_pp","drop_f1_pp"]]
            .sort_values(["setting"])
            .to_dict(orient="records")
        ),
        "paths": {
            "per_run_csv": str(per_run_csv),
            "aggregated_items_csv": str(items_csv),
            "summary_csv": str(summary_csv),
        },
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("✅ scoring_v2 complete")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()