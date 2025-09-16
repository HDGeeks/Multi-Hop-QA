# src/scoring/scoring_v2_extended.py
"""
scoring_v2_extended.py

Lightweight, robust scoring for QA outputs using Exact Match (EM) and token-level F1.

Key features:
- Normalizes predictions and gold answers consistently.
- Auto-strips HTML tags (e.g., <span>…</span>, <sub>…</sub>) from predictions when needed:
  * If model name contains "llama", OR
  * If the prediction visibly contains tags.
  This fixes systematic 0 EM from LLaMA-style wrappers without over-correcting other models.
- Supports aliases in the gold file and uses the best match across (answer + aliases).
- Produces overall, per-setting, and per-domain micro aggregates, plus drops vs. Gold.

Inputs:
- JSONL result files from your runner (at least: qid, setting, output; model optional but recommended).
- Gold CSV with columns: qid, answer, aliases (optional, '|' separated), domain.

Outputs:
- Summary JSON with:
    overall: { exact_match_percent, f1_percent, n }
    by_setting: { gold/para/dist/para_dist: { em_percent, f1_percent, n } }
    by_domain: { <domain>: { em_percent, f1_percent, n } }
    drops_vs_gold: { para/dist/para_dist: { em_drop_pp, f1_drop_pp } }

Usage:
    python src/scoring/scoring_v2_extended.py \
        --glob "src/results_50/gpt4o/*.jsonl" \
        --gold-csv "src/data_50/mhqa_questions_50.csv" \
        --out-json "src/results_50/gpt4o/summary_extended.json"
"""
import argparse, json, re, html
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

# ---------- Normalization ----------
_PUNCT = re.compile(r"[^\w\s]")      # remove non-alnum, keep whitespace
_WS    = re.compile(r"\s+")
_TAGS  = re.compile(r"<[^>]+>")      # naive HTML tag stripper

def normalize(text: str) -> str:
    """Lowercase, trim, strip punctuation, and squeeze whitespace."""
    if text is None:
        return ""
    t = text.strip().lower()
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t

def strip_html(text: str) -> str:
    """
    Unescape HTML entities and drop tags; squeeze whitespace.
    Example: "<span>H<sub>2</sub>O</span>" -> "H 2 O" -> normalize() will remove extra spacing/punct.
    """
    if not text:
        return ""
    t = html.unescape(text)
    t = _TAGS.sub(" ", t)
    t = " ".join(t.split())
    return t

def f1_score(pred: str, gold: str) -> float:
    """
    Token F1 using multiset overlap (proper precision/recall).
    Inputs must already be normalized() strings.
    """
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    p = pred.split()
    g = gold.split()
    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall    = overlap / len(g)
    return (2 * precision * recall) / (precision + recall)

# ---------- Gold Loader ----------
def load_gold(gold_csv: Path):
    """
    Read gold CSV and build a mapping:
        qid -> {
          "answer": <normalized>,
          "aliases": [normalized...],
          "refs": [normalized answer + aliases],
          "domain": original domain string
        }
    """
    df = pd.read_csv(gold_csv)
    required = {"qid", "answer", "domain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Gold CSV missing columns: {sorted(missing)}")

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
    ap = argparse.ArgumentParser(description="Lightweight scoring (EM/F1), drops, domains, with auto HTML-strip for LLaMA.")
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
        with p.open("r", encoding="utf-8") as f:
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
    stripped_counts = defaultdict(int)
    scored_rows = []
    for _, r in df.iterrows():
        qid     = str(r["qid"])
        setting = str(r["setting"]).lower().strip()
        model   = str(r.get("model", "")).lower().strip()
        raw_out = str(r.get("output", ""))

        # Decide when to strip HTML:
        # - If model looks like LLaMA, or
        # - If we visibly see tags in the output
        needs_strip = ("llama" in model) or bool(_TAGS.search(raw_out))
        pred_raw_clean = strip_html(raw_out) if needs_strip else raw_out
        if needs_strip:
            stripped_counts[model or "<unknown>"] += 1

        pred = normalize(pred_raw_clean)

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
                "em_drop_pp": gold_em - s_em,  # positive = worse than gold
                "f1_drop_pp": gold_f1 - s_f1,
            }
    summary["drops_vs_gold"] = drops

    # Attach a tiny diagnostic about HTML stripping
    summary["html_stripping_diagnostics"] = {
        "by_model": dict(sorted(stripped_counts.items(), key=lambda kv: kv[0])),
        "total_rows_stripped": int(sum(stripped_counts.values())),
    }

    # Save
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ scoring_v2_extended complete → {out_path}")

    # Pretty-print the stripping diag (useful immediate feedback)
    if stripped_counts:
        print("ℹ️  HTML-stripped predictions (model → count):")
        for m, c in sorted(stripped_counts.items()):
            print(f"   - {m or '<unknown>'}: {c}")

if __name__ == "__main__":
    main()

