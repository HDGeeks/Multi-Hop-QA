# src/scoring/scoring_v2.py
from __future__ import annotations
import argparse, json, re ,html
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# ----------------------------
# Text normalization & metrics
# ----------------------------
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")

def normalize(s: str) -> str:
    """Lowercase, strip, remove punctuation, squeeze whitespace."""
    if s is None:
        return ""
    s = s.strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s

# --- HTML stripping (minimal, safe) ---
_TAG_RE = re.compile(r"<[^>]+>")

def strip_html_tags(s: str) -> str:
    """Unescape HTML entities, drop tags, squeeze whitespace. No extra trimming."""
    if not s:
        return ""
    s = html.unescape(s)
    s = _TAG_RE.sub(" ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 of prediction vs. gold."""
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    # bag-of-words overlap
    common = {}
    for t in p:
        common[t] = min(p.count(t), g.count(t))
    num = sum(common.values())
    if num == 0:
        return 0.0
    prec = num / len(p)
    rec  = num / len(g)
    return 2 * prec * rec / (prec + rec)

def best_em_f1(pred: str, refs: List[str]) -> Tuple[int, float]:
    refs = refs or [""]
    em = 0
    best = 0.0
    for r in refs:
        if normalize(pred) == normalize(r):
            em = 1
        best = max(best, token_f1(pred, r))
    return em, best

REFUSAL_MARKERS = [
    "i can't", "i cannot", "i’m unable", "i am unable",
    "as an ai", "i don't have", "cannot assist", "won't provide",
    "cannot answer", "no sufficient", "lack enough info",
]

def is_refusal(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(m in t for m in REFUSAL_MARKERS)

def is_invalid(text: str) -> bool:
    return (text or "").strip() == ""

# ----------------------------
# Load gold from CSV
# ----------------------------
def load_gold_map(q_csv: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(q_csv)
    out: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        qid = str(r["qid"])
        canon = str(r["answer"]).strip()
        aliases = [a.strip() for a in str(r.get("aliases","") or "").split("|") if a.strip()]
        out[qid] = [canon] + aliases
    return out

# ----------------------------
# I/O helpers
# ----------------------------
def read_jsonl_many(glob_pattern: str) -> pd.DataFrame:
    all_rows = []
    for p in sorted(Path().glob(glob_pattern)):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                all_rows.append(json.loads(line))
    if not all_rows:
        raise SystemExit(f"No files matched: {glob_pattern}")
    return pd.DataFrame(all_rows)

# ----------------------------
# Scoring
# ----------------------------
def compute_per_run(df: pd.DataFrame, gold_map: Dict[str, List[str]]) -> pd.DataFrame:
    # Expect: run_id, qid, domain, model, setting, output, latency_ms
    need = {"run_id","qid","domain","model","setting","output","latency_ms"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in raw jsonl: {sorted(miss)}")

    ems, f1s, refusals, invalids = [], [], [], []
    for qid, pred, model in zip(df["qid"], df["output"], df["model"]):
        # strip HTML only for LLaMA family
        m = str(model).lower()
        pred_str = str(pred)
        if "llama" in m:
            pred_str = strip_html_tags(pred_str)

        refs = gold_map.get(str(qid), [""])
        em, f1 = best_em_f1(pred_str, refs)
        ems.append(em)
        f1s.append(f1)
        refusals.append(int(is_refusal(pred_str)))
        invalids.append(int(is_invalid(pred_str)))

    out = df.copy()
    out["em"] = ems
    out["f1"] = f1s
    out["refusal"] = refusals
    out["invalid"] = invalids
    return out

def aggregate_items(per_run: pd.DataFrame) -> pd.DataFrame:
    # Collapse runs per (qid, setting, model, domain)
    def em_majority(s: pd.Series) -> int:
        n = int(s.size)
        if n == 0:
            return 0
        thresh = n // 2 + 1  # majority threshold, works for any n
        return int(int(s.sum()) >= thresh)

    def mad(s: pd.Series) -> float:
        arr = s.to_numpy(dtype=float)
        if arr.size == 0:
            return 0.0
        med = float(np.median(arr))
        return float(np.median(np.abs(arr - med)))

    ag = (
        per_run
        .groupby(["qid","domain","model","setting"], as_index=False)
        .agg(
            em_majority=("em", em_majority),
            f1_median=("f1", "median"),
            latency_median_ms=("latency_ms", "median"),
            refusal_any=("refusal", "max"),
            invalid_any=("invalid", "max"),
            stability_mad_f1=("f1", mad),
        )
    )
    return ag

def summarize_model_setting(items_ag: pd.DataFrame) -> pd.DataFrame:
    # Summary per model×setting across items
    def pct(x) -> float:
        x = pd.Series(x)
        return 100.0 * float(x.mean()) if len(x) else 0.0

    rows = []
    for (model, setting), grp in items_ag.groupby(["model","setting"]):
        n_items = grp["qid"].nunique()
        f1_mean_of_medians = 100.0 * float(np.mean(grp["f1_median"])) if len(grp) else 0.0
        row = {
            "model": model,
            "setting": setting,
            "n_items": int(n_items),
            "em_mean_percent": pct(grp["em_majority"]),
            # Paper definition: mean over items of per-item (median-across-runs) F1
            "f1_mean_of_medians_percent": f1_mean_of_medians,
            "latency_p50_ms": float(np.median(grp["latency_median_ms"])) if len(grp) else 0.0,
            "refusal_rate_percent": pct(grp["refusal_any"]),
            "invalid_rate_percent": pct(grp["invalid_any"]),
            "stability_mad_f1_median": float(np.median(grp["stability_mad_f1"])) if len(grp) else 0.0,
        }
        rows.append(row)

    summ = pd.DataFrame(rows)

    # Back-compat alias so downstream code (tables/figures) keeps working
    summ["f1_median_percent"] = summ["f1_mean_of_medians_percent"]

    # Add drops vs GOLD within each model (Δ in percentage points)
    def add_drop(metric_cols, label):
        col = next((c for c in metric_cols if c in summ.columns), None)
        if col is None:
            return
        drops = []
        for model, grp in summ.groupby("model", sort=False):
            base_ser = grp.loc[grp["setting"] == "gold", col]
            base_val = float(base_ser.iloc[0]) if len(base_ser) else np.nan
            for _, r in grp.iterrows():
                if r["model"] != model or np.isnan(base_val):
                    drops.append(np.nan)
                elif r["setting"] == "gold":
                    drops.append(np.nan)
                else:
                    drops.append(float(r[col]) - base_val)
        summ[label] = drops

    add_drop(["em_mean_percent"], "drop_em_vs_gold_pp")
    add_drop(["f1_mean_of_medians_percent", "f1_median_percent"], "drop_f1_vs_gold_pp")

    # Order settings
    order = pd.CategoricalDtype(categories=["gold","para","dist","para_dist"], ordered=True)
    if "setting" in summ.columns:
        summ["setting"] = summ["setting"].astype(order)
        summ = summ.sort_values(["model","setting"]).reset_index(drop=True)

    return summ

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Alias-aware EM/F1 scoring with run collapsing.")
    ap.add_argument("--glob", required=True, help="e.g. src/results_50/gpt4o/*.jsonl")
    ap.add_argument("--gold-csv", required=True, help="src/data_50/mhqa_questions_50.csv")
    ap.add_argument("--context-csv", required=True, help="(unused for scoring, checked for existence)")
    ap.add_argument("--paras-csv", required=True, help="(unused for scoring, checked for existence)")
    ap.add_argument("--out-json", required=True, help="summary JSON path")
    args = ap.parse_args()

    # Resolve and sanity-check I/O
    glob_pat = args.glob
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    for p in [args.gold_csv, args.context_csv, args.paras_csv]:
        if not Path(p).exists():
            raise SystemExit(f"Missing file: {p}")

    # 1) Read raw + gold
    raw = read_jsonl_many(glob_pat)
    gold_map = load_gold_map(Path(args.gold_csv))

    # 2) Per-run scoring
    per_run = compute_per_run(raw, gold_map)

    # 3) Collapse to per-item
    items_ag = aggregate_items(per_run)

    # 4) Build model×setting summary
    summary = summarize_model_setting(items_ag)

    # 5) Write artifacts next to JSON
    outdir = out_json.parent
    per_run_path = outdir / "per_run_v2.csv"
    items_path   = outdir / "aggregated_items_v2.csv"
    summary_path = outdir / "summary_v2.csv"

    per_run.to_csv(per_run_path, index=False)
    items_ag.to_csv(items_path, index=False)
    summary.to_csv(summary_path, index=False)

    # 6) Compact JSON with micro + by-setting (micro = mean across runs)
    def micro_overall(df: pd.DataFrame) -> Dict[str, float]:
        return {
            "exact_match_percent": 100.0 * float(df["em"].mean()) if len(df) else 0.0,
            "f1_percent": 100.0 * float(df["f1"].mean()) if len(df) else 0.0,
            "n": int(len(df)),
        }

    micro = micro_overall(per_run)
    by_setting = (
        per_run.groupby("setting", sort=False)
        .apply(lambda g: micro_overall(g))
        .to_dict()
    )

    payload = {
        "micro_overall": micro,
        "by_setting": by_setting,
        "paths": {
            "per_run_csv": str(per_run_path),
            "aggregated_items_csv": str(items_path),
            "summary_csv": str(summary_path),
        },
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("✅ scoring_v2 complete")
    print(json.dumps(payload["micro_overall"], indent=2))

if __name__ == "__main__":
    main()
