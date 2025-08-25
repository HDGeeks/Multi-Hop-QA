#!/usr/bin/env python3
"""
Validate the 50-question multi-hop dataset.

Checks:
- Schema columns for all three CSVs
- Row count and per-domain distribution
- qid uniqueness + cross-file alignment
- Empty fields and all-whitespace cells
- Paraphrase quality (not identical to gold, not trivially similar)
- Basic content sanity: snippet lengths, distractor not leaking answer
- Alias parsing sanity (no dup aliases after stripping)

Usage:
  python3 -m src.data_50.validate_50 \
    --base src/data_50 \
    --questions mhqa_questions_50.csv \
    --context mhqa_context_50.csv \
    --paraphrases mhqa_paraphrases_50.csv \
    --expect 50 \
    --domains history,science,politics,literature,geography
"""
import argparse
from pathlib import Path
import sys
import csv
from collections import Counter, defaultdict
from difflib import SequenceMatcher

EXPECTED_Q_COLS = ["qid","domain","question","answer","aliases"]
EXPECTED_C_COLS = ["qid","snippet_a","snippet_b","distractor"]
EXPECTED_P_COLS = ["qid","paraphrase"]

def read_csv(path, expected_cols):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing = [c for c in expected_cols if c not in cols]
        extra = [c for c in cols if c not in expected_cols]
        if missing:
            raise SystemExit(f"[ERROR] {path}: missing columns: {missing}")
        for r in reader:
            # normalize whitespace
            clean = {k: (r.get(k,"") or "").strip() for k in expected_cols}
            rows.append(clean)
    return rows

def warn(msg):
    print(f"[WARN] {msg}")

def error(msg):
    print(f"[ERROR] {msg}")

def ok(msg):
    print(f"[OK] {msg}")

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="directory containing the CSVs")
    ap.add_argument("--questions", default="mhqa_questions_50.csv")
    ap.add_argument("--context", default="mhqa_context_50.csv")
    ap.add_argument("--paraphrases", default="mhqa_paraphrases_50.csv")
    ap.add_argument("--expect", type=int, default=50, help="expected number of items")
    ap.add_argument("--domains", default="history,science,politics,literature,geography")
    ap.add_argument("--min_snippet_chars", type=int, default=50)
    ap.add_argument("--max_para_similarity", type=float, default=0.90, help="warn if paraphrase similarity to gold >= this")
    ap.add_argument("--min_para_similarity", type=float, default=0.35, help="warn if paraphrase similarity to gold < this (possibly off-topic)")
    args = ap.parse_args()

    base = Path(args.base)
    q_path = base / args.questions
    c_path = base / args.context
    p_path = base / args.paraphrases

    if not q_path.exists() or not c_path.exists() or not p_path.exists():
        error(f"File not found. Checked:\n  - {q_path}\n  - {c_path}\n  - {p_path}")
        sys.exit(1)

    Q = read_csv(q_path, EXPECTED_Q_COLS)
    C = read_csv(c_path, EXPECTED_C_COLS)
    P = read_csv(p_path, EXPECTED_P_COLS)

    # ---------- basic sizes ----------
    ok(f"Loaded: questions={len(Q)}, context={len(C)}, paraphrases={len(P)}")
    hard_fail = False

    if len(Q) != args.expect:
        warn(f"Questions count {len(Q)} != expected {args.expect}")
    if len(C) != len(Q) or len(P) != len(Q):
        error("Row count mismatch across files (Q, C, P).")
        hard_fail = True

    # ---------- qid uniqueness & alignment ----------
    qids_q = [r["qid"] for r in Q]
    qids_c = [r["qid"] for r in C]
    qids_p = [r["qid"] for r in P]

    def dupes(arr): 
        c = Counter(arr)
        return [k for k,v in c.items() if v>1]

    d_q = dupes(qids_q)
    d_c = dupes(qids_c)
    d_p = dupes(qids_p)
    if d_q: error(f"Duplicate qids in questions: {d_q}"); hard_fail=True
    if d_c: error(f"Duplicate qids in context:   {d_c}"); hard_fail=True
    if d_p: error(f"Duplicate qids in paraphrases: {d_p}"); hard_fail=True

    set_q, set_c, set_p = set(qids_q), set(qids_c), set(qids_p)
    if set_q != set_c or set_q != set_p:
        only_q = sorted(set_q - set_c - set_p)
        only_c = sorted(set_c - set_q - set_p)
        only_p = sorted(set_p - set_q - set_c)
        error(f"QID sets differ. Only in Q={only_q[:5]} Only in C={only_c[:5]} Only in P={only_p[:5]} (showing up to 5)")
        hard_fail = True
    else:
        ok("QIDs aligned across all files.")

    # ---------- domain distribution ----------
    domains = [d.strip().lower() for d in args.domains.split(",") if d.strip()]
    dom_counts = Counter([r["domain"].lower() for r in Q])
    for d in domains:
        if d not in dom_counts:
            warn(f"Domain '{d}' missing in questions.")
    ok(f"Domain counts: {dict(dom_counts)}")

    # ---------- empty field checks ----------
    def any_empty(rows, cols, label):
        bad = []
        for r in rows:
            for c in cols:
                if (r.get(c,"") or "").strip() == "":
                    bad.append((r.get("qid","?"), c))
        if bad:
            error(f"{label}: empty cells: {bad[:8]}{' ...' if len(bad)>8 else ''}")
            return True
        return False

    if any_empty(Q, ["qid","domain","question","answer"], "Questions"):
        hard_fail = True
    if any_empty(C, ["qid","snippet_a","snippet_b","distractor"], "Context"):
        hard_fail = True
    if any_empty(P, ["qid","paraphrase"], "Paraphrases"):
        hard_fail = True

    # ---------- build index by qid ----------
    Q_by = {r["qid"]: r for r in Q}
    C_by = {r["qid"]: r for r in C}
    P_by = {r["qid"]: r for r in P}

    # ---------- alias sanity & dedup ----------
    alias_dups = []
    for r in Q:
        aliases_raw = (r.get("aliases","") or "").strip()
        if aliases_raw:
            parts = [a.strip() for a in aliases_raw.split("|") if a.strip()]
            if len(parts) != len(set([a.lower() for a in parts])):
                alias_dups.append(r["qid"])
    if alias_dups:
        warn(f"Duplicate aliases after normalization for qids: {alias_dups[:10]}")

    # ---------- content heuristics ----------
    short_snips = []
    leaks = []
    para_too_close = []
    para_too_far = []

    for qid in set_q:
        q = Q_by[qid]
        c = C_by[qid]
        ptext = P_by[qid]["paraphrase"]
        gold_q = q["question"]
        ans = (q["answer"] or "").lower()

        # snippet lengths
        for key in ("snippet_a","snippet_b","distractor"):
            if len(c[key]) < args.min_snippet_chars:
                short_snips.append((qid, key, len(c[key])))

        # distractor leaking answer
        if ans and ans in c["distractor"].lower():
            leaks.append(qid)

        # paraphrase similarity checks
        sim = similarity(gold_q, ptext)
        if sim >= args.max_para_similarity:
            para_too_close.append((qid, round(sim,3)))
        if sim < args.min_para_similarity:
            para_too_far.append((qid, round(sim,3)))

    if short_snips:
        warn(f"Short snippets (<{args.min_snippet_chars} chars): {short_snips[:10]}{' ...' if len(short_snips)>10 else ''}")
    else:
        ok("Snippet lengths look reasonable.")

    if leaks:
        warn(f"Distractor contains gold answer (possible leakage) for qids: {leaks[:10]}{' ...' if len(leaks)>10 else ''}")

    if para_too_close:
        warn(f"Paraphrase too similar to gold (sim>={args.max_para_similarity}): {para_too_close[:10]}{' ...' if len(para_too_close)>10 else ''}")
    if para_too_far:
        warn(f"Paraphrase possibly off-topic (sim<{args.min_para_similarity}): {para_too_far[:10]}{' ...' if len(para_too_far)>10 else ''}")
    if not para_too_close and not para_too_far:
        ok("Paraphrase similarity within heuristic bounds.")

    # ---------- summary ----------
    print("\n=== SUMMARY ===")
    print(f"Items: {len(Q)} | Domains: {dict(dom_counts)}")
    print(f"Hard failures: {'YES' if hard_fail else 'NO'}")
    if hard_fail:
        sys.exit(2)
    else:
        ok("Validation passed with no hard failures.")
        sys.exit(0)

if __name__ == "__main__":
    main()