from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, DegenerateDataWarning


# ---------------------------
# Normalization / scoring
# ---------------------------

_PUNC_RE = re.compile(r"[,\.\!\?\:\;\'\"\(\)\[\]]")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = _PUNC_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s

def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    pc = Counter(p)
    gc = Counter(g)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, len(p))
    recall = overlap / max(1, len(g))
    return 2 * precision * recall / (precision + recall)

REFUSAL_PATTERNS = [
    "cannot answer",
    "can't answer",
    "as an ai",
    "i cannot",
    "i'm unable",
    "i can’t",
    "unable to comply",
    "refuse",
]
def is_refusal(s: str) -> bool:
    s_norm = s.lower()
    return any(p in s_norm for p in REFUSAL_PATTERNS) or (normalize_text(s) == "")

def is_format_violation(s: str) -> bool:
    # Heuristic: if output is long/multi‑sentence when a short span is requested.
    # More than 8 tokens or contains newline -> flag.
    toks = normalize_text(s).split()
    return (len(toks) > 8) or ("\n" in s)


# ---------------------------
# CI helper with fallback
# ---------------------------

# put near the top with imports
import numpy as np
import warnings

def ci_with_fallback(samples, stat_fn, conf=0.95, n_resamples=10_000):
    """
    Compute a bootstrap CI for a 1D array of samples using SciPy's bootstrap.
    Falls back to percentile CI if BCa is degenerate or SciPy not available.

    stat_fn: a numpy-like reducer (e.g., np.mean, np.median) that accepts axis=
    """
    arr = np.asarray(samples, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, np.nan, "empty"
    # Degenerate distribution guard: all values same or 1 sample
    if arr.size < 2 or np.allclose(arr, arr[0]):
        v = float(arr.mean())
        return v, v, "degenerate"

    # Adapter so scipy.bootstrap can call our stat with axis=
    def stat_with_axis(x, axis=None):
        return stat_fn(x, axis=axis)

    try:
        from scipy.stats import bootstrap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence DegenerateDataWarning, etc.
            res = bootstrap(
                (arr,),
                stat_with_axis,
                vectorized=False,
                paired=False,
                confidence_level=conf,
                n_resamples=n_resamples,
                method="BCa",
                random_state=0,
            )
        lo = float(res.confidence_interval.low)
        hi = float(res.confidence_interval.high)
        return lo, hi, "BCa"
    except Exception:
        lo, hi = np.percentile(arr, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
        return float(lo), float(hi), "percentile"


# ---------------------------
# Data classes
# ---------------------------

@dataclass
class GoldItem:
    qid: str
    domain: str | None
    answer: str
    aliases: List[str]

def load_gold(path: Path) -> Dict[str, GoldItem]:
    df = pd.read_csv(path)
    # Expected columns: qid, domain, answer, aliases (pipe-separated, may be empty)
    gold: Dict[str, GoldItem] = {}
    for _, r in df.iterrows():
        aliases = []
        if "aliases" in df.columns and isinstance(r.get("aliases", ""), str) and r["aliases"].strip():
            aliases = [a.strip() for a in str(r["aliases"]).split("|") if a.strip()]
        gold[str(r["qid"])] = GoldItem(
            qid=str(r["qid"]),
            domain=str(r["domain"]) if "domain" in df.columns else None,
            answer=str(r["answer"]),
            aliases=aliases,
        )
    return gold


# ---------------------------
# Scoring core
# ---------------------------

def best_match_metrics(pred: str, candidates: List[str]) -> Tuple[int, float]:
    """EM (max over candidates), F1 (max over candidates)."""
    ems = []
    f1s = []
    p_norm = normalize_text(pred)
    for cand in candidates:
        c_norm = normalize_text(cand)
        ems.append(1 if p_norm == c_norm else 0)
        f1s.append(token_f1(pred, cand))
    return int(max(ems)), float(max(f1s))

def mad(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


# ---------------------------
# Pipeline
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Glob for input JSONL logs (e.g., src/results/gpt4o/*.jsonl)")
    ap.add_argument("--model", required=True, help="Model id (e.g., gpt4o)")
    ap.add_argument("--outdir", required=True, help="Output directory for metrics CSVs")
    ap.add_argument("--gold-csv", default="data/mhqa_questions.csv", help="Gold answers CSV (qid, domain, answer, aliases)")
    args = ap.parse_args()

    files = sorted(glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load gold
    gold_map = load_gold(Path(args.gold_csv))

    # Read JSONL logs
    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                # Expected fields in logs: run_id, qid, domain, model, setting, output, latency_ms
                qid = rec.get("qid")
                if qid not in gold_map:
                    # allow but skip scoring; still keep row
                    gold = None
                    candidates = []
                    domain = rec.get("domain")
                else:
                    gold = gold_map[qid]
                    candidates = [gold.answer] + gold.aliases
                    domain = gold.domain if gold.domain is not None else rec.get("domain")

                output = rec.get("output", "")
                em, f1 = (np.nan, np.nan) if not candidates else best_match_metrics(output, candidates)

                rows.append({
                    "run_id": rec.get("run_id"),
                    "qid": qid,
                    "domain": domain,
                    "model": args.model,
                    "setting": rec.get("setting"),
                    "output": output,
                    "latency_ms": rec.get("latency_ms"),
                    "em": em,
                    "f1": f1,
                    "refusal": int(is_refusal(output)),
                    "format_violation": int(is_format_violation(output)),
                })

    df = pd.DataFrame(rows)
    # Persist per-run
    per_run_csv = outdir / f"{args.model}_per_run.csv"
    df.to_csv(per_run_csv, index=False)

    # Aggregate per item (collapse 3 runs)
    group_keys = ["qid", "domain", "model", "setting"]
    agg_rows = []
    for (qid, domain, model, setting), g in df.groupby(group_keys, dropna=False):
        em_majority = int(g["em"].sum() >= 2) if g["em"].notna().any() else np.nan
        f1_median = float(np.median(g["f1"])) if g["f1"].notna().any() else np.nan
        lat_med = float(np.median(g["latency_ms"])) if g["latency_ms"].notna().any() else np.nan
        refusal_any = int((g["refusal"] == 1).any())
        format_any = int((g["format_violation"] == 1).any())
        stab_mad = mad(g["f1"])
        agg_rows.append({
            "qid": qid,
            "domain": domain,
            "model": model,
            "setting": setting,
            "em_majority": em_majority,
            "f1_median": f1_median,
            "latency_median_ms": lat_med,
            "refusal_any": refusal_any,
            "format_any": format_any,
            "stability_abs_dev": stab_mad,
        })
    df_items = pd.DataFrame(agg_rows).sort_values(["qid", "setting"])
    items_csv = outdir / f"{args.model}_aggregated_items.csv"
    df_items.to_csv(items_csv, index=False)

    # Build per-setting summary across items
    # Compute drops vs Gold on a per-item basis first
    # (join each setting with the same qid's Gold)
    df_gold = df_items[df_items["setting"] == "gold"][["qid", "model", "em_majority", "f1_median"]]
    df_gold = df_gold.rename(columns={"em_majority": "em_gold", "f1_median": "f1_gold"})

    df_items = df_items.merge(df_gold, on=["qid", "model"], how="left")
    def drop_cols(sub: pd.DataFrame) -> pd.DataFrame:
        # Item-wise drops (Gold minus Setting); NaNs preserved
        sub = sub.copy()
        sub["drop_em"] = sub["em_gold"] - sub["em_majority"]
        sub["drop_f1"] = sub["f1_gold"] - sub["f1_median"]
        return sub

    df_items = drop_cols(df_items)

    summary_rows = []
    for setting, g in df_items.groupby("setting"):
        # Only items that exist for this setting
        n_items = int(g["qid"].nunique())

        # Aggregate core stats over items
        em_mean_items = g["em_majority"].dropna().astype(float).values
        f1_med_items = g["f1_median"].dropna().astype(float).values
        lat_med_items = g["latency_median_ms"].dropna().astype(float).values
        stab_items = g["stability_abs_dev"].dropna().astype(float).values

        em_mean = float(np.mean(em_mean_items)) if em_mean_items.size else float("nan")
        f1_median_of_medians = float(np.median(f1_med_items)) if f1_med_items.size else float("nan")
        lat_median_of_medians = float(np.median(lat_med_items)) if lat_med_items.size else float("nan")
        stab_median = float(np.median(stab_items)) if stab_items.size else float("nan")

        # Drops (compare to gold) — medians across items
        drops = g[["drop_em", "drop_f1"]].dropna()
        drop_em_median = float(np.median(drops["drop_em"])) if not drops.empty else float("nan")
        drop_f1_median = float(np.median(drops["drop_f1"])) if not drops.empty else float("nan")

        # CIs with fallback
        em_lo, em_hi, em_ci_method = ci_with_fallback(em_mean_items, np.mean)
        f1_lo, f1_hi, f1_ci_method = ci_with_fallback(f1_med_items, np.median)

        summary_rows.append({
            "model": args.model,
            "setting": setting,
            "n_items": n_items,
            "em_mean": em_mean,
            "f1_median": f1_median_of_medians,
            "latency_median_ms": lat_median_of_medians,
            "drop_em_median": drop_em_median,
            "drop_f1_median": drop_f1_median,
            "ci_em_low": em_lo, "ci_em_high": em_hi, "ci_em_method": em_ci_method,
            "ci_f1_low": f1_lo, "ci_f1_high": f1_hi, "ci_f1_method": f1_ci_method,
            "stability_median": stab_median,
        })

    df_sum = pd.DataFrame(summary_rows).sort_values(["model", "setting"])
    summary_csv = outdir / f"{args.model}_summary.csv"
    df_sum.to_csv(summary_csv, index=False)

    print("✅ Scoring complete for", args.model)
    print("  Per-run:  ", per_run_csv)
    print("  Items:    ", items_csv)
    print("  Summary:  ", summary_csv)


if __name__ == "__main__":
    main()