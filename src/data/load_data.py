# src/data/load_data.py
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set

# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Item:
    """
    One fully-materialized multi-hop QA example.
    """
    qid: str
    domain: str                   # 'history' | 'science' | 'politics' (free text allowed, validated)
    question: str                 # gold question
    paraphrase: Optional[str]     # may be None if not provided
    snippet_a: str                # gold supporting snippet A
    snippet_b: str                # gold supporting snippet B
    distractor: str               # plausible but incorrect snippet
    answer: str                   # canonical gold answer (not shown to model)
    aliases: Tuple[str, ...]      # canonical-alias list for scoring (may be empty)

    def to_preview(self) -> Dict[str, str]:
        """
        Short preview for debugging or manifests.
        """
        return {
            "qid": self.qid,
            "domain": self.domain,
            "question": self.question[:140] + ("…" if len(self.question) > 140 else ""),
            "paraphrase": (self.paraphrase[:140] + "…") if (self.paraphrase and len(self.paraphrase) > 140) else (self.paraphrase or ""),
            "answer": self.answer,
        }

# ---------------------------------------------------------------------
# CSV schemas (expected)
# ---------------------------------------------------------------------
# data/mhqa_questions.csv
#   qid, domain, question, answer, aliases
#
# data/mhqa_context.csv
#   qid, snippet_a, snippet_b, distractor
#
# data/mhqa_paraphrases.csv
#   qid, paraphrase
#
# Notes:
# - aliases: pipe-separated string, e.g. "George III|King George III|George the Third"
# - All files must have unique qid rows (no duplicates).
# - Every qid present in questions must be present in context; paraphrase is optional per qid.

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def load_items(
    questions_csv: Path,
    context_csv: Path,
    paraphrases_csv: Optional[Path] = None,
    *,
    required_domains: Optional[Iterable[str]] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
    sample_n: Optional[int] = None,
    strict_paraphrase: bool = False,
) -> List[Item]:
    """
    Load, join, validate, and return a list of Items.

    Args:
        questions_csv: path to data/mhqa_questions.csv
        context_csv: path to data/mhqa_context.csv
        paraphrases_csv: optional path to data/mhqa_paraphrases.csv
        required_domains: if provided, only keep items whose domain is in this set (case-insensitive)
        shuffle: shuffle the final list
        seed: RNG seed for deterministic shuffling / sampling
        sample_n: if provided, keep only the first N after shuffling/filtering
        strict_paraphrase: if True, require paraphrase for every qid (raise if missing)

    Returns:
        List[Item] suitable for prompt building and scoring.
    """
    q_rows = _read_questions(questions_csv)
    c_rows = _read_context(context_csv)
    p_rows = _read_paraphrases(paraphrases_csv) if paraphrases_csv else {}

    # Join on qid
    items: List[Item] = []
    for qid, q in q_rows.items():
        if qid not in c_rows:
            raise ValueError(f"[load_items] Missing context for qid={qid}")

        ctx = c_rows[qid]
        paraphrase = p_rows.get(qid)  # may be None

        if strict_paraphrase and (paraphrase is None or paraphrase.strip() == ""):
            raise ValueError(f"[load_items] strict_paraphrase=True but paraphrase is missing/empty for qid={qid}")

        item = Item(
            qid=qid,
            domain=q["domain"],
            question=_req(q["question"], f"questions[{qid}].question"),
            paraphrase=paraphrase.strip() if paraphrase else None,
            snippet_a=_req(ctx["snippet_a"], f"context[{qid}].snippet_a"),
            snippet_b=_req(ctx["snippet_b"], f"context[{qid}].snippet_b"),
            distractor=_req(ctx["distractor"], f"context[{qid}].distractor"),
            answer=_req(q["answer"], f"questions[{qid}].answer"),
            aliases=_parse_aliases(q["aliases"]),
        )
        _validate_item(item)
        items.append(item)

    # check for stray qids in context / paraphrase
    _warn_strays("context", set(c_rows.keys()) - set(q_rows.keys()))
    _warn_strays("paraphrases", set(p_rows.keys()) - set(q_rows.keys()))

    # domain filter
    if required_domains:
        req = {d.lower().strip() for d in required_domains}
        items = [it for it in items if it.domain.lower().strip() in req]
        if not items:
            raise ValueError("[load_items] After domain filtering, no items remain.")

    # shuffle/sample
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(items)
    if sample_n is not None:
        items = items[:sample_n]

    return items


def write_manifest_jsonl(items: List[Item], out_path: Path) -> None:
    """
    Convenience helper: write a compact JSONL manifest for quick inspection.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_preview(), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Internal readers + validators
# ---------------------------------------------------------------------

def _read_questions(path: Path) -> Dict[str, Dict[str, str]]:
    _assert_exists(path, "questions_csv")
    seen: Set[str] = set()
    rows: Dict[str, Dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _require_cols(reader.fieldnames, path, {"qid", "domain", "question", "answer", "aliases"})
        for i, row in enumerate(reader, start=2):
            qid = _req(row.get("qid"), f"{path.name}:{i}.qid")
            if qid in seen:
                raise ValueError(f"[{path.name}] Duplicate qid='{qid}' at line {i}")
            seen.add(qid)
            rows[qid] = {
                "domain": _req(row.get("domain"), f"{path.name}:{i}.domain"),
                "question": _req(row.get("question"), f"{path.name}:{i}.question"),
                "answer": _req(row.get("answer"), f"{path.name}:{i}.answer"),
                "aliases": (row.get("aliases") or "").strip(),
            }
    if not rows:
        raise ValueError(f"[{path.name}] No rows found.")
    return rows


def _read_context(path: Path) -> Dict[str, Dict[str, str]]:
    _assert_exists(path, "context_csv")
    seen: Set[str] = set()
    rows: Dict[str, Dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _require_cols(reader.fieldnames, path, {"qid", "snippet_a", "snippet_b", "distractor"})
        for i, row in enumerate(reader, start=2):
            qid = _req(row.get("qid"), f"{path.name}:{i}.qid")
            if qid in seen:
                raise ValueError(f"[{path.name}] Duplicate qid='{qid}' at line {i}")
            seen.add(qid)
            rows[qid] = {
                "snippet_a": _req(row.get("snippet_a"), f"{path.name}:{i}.snippet_a"),
                "snippet_b": _req(row.get("snippet_b"), f"{path.name}:{i}.snippet_b"),
                "distractor": _req(row.get("distractor"), f"{path.name}:{i}.distractor"),
            }
    if not rows:
        raise ValueError(f"[{path.name}] No rows found.")
    return rows


def _read_paraphrases(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    _assert_exists(path, "paraphrases_csv")
    seen: Set[str] = set()
    rows: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _require_cols(reader.fieldnames, path, {"qid", "paraphrase"})
        for i, row in enumerate(reader, start=2):
            qid = _req(row.get("qid"), f"{path.name}:{i}.qid")
            if qid in seen:
                raise ValueError(f"[{path.name}] Duplicate qid='{qid}' at line {i}")
            seen.add(qid)
            para = (row.get("paraphrase") or "").strip()
            if not para:
                # allowed unless strict_paraphrase=True, which we check later
                pass
            rows[qid] = para
    return rows


def _validate_item(it: Item) -> None:
    # minimal hard checks to catch curation mistakes
    if it.question.strip() == "":
        raise ValueError(f"[validate] Empty question for qid={it.qid}")
    if it.snippet_a.strip() == "" or it.snippet_b.strip() == "":
        raise ValueError(f"[validate] Missing gold snippets for qid={it.qid}")
    if it.distractor.strip() == "":
        raise ValueError(f"[validate] Empty distractor for qid={it.qid}")
    if it.answer.strip() == "":
        raise ValueError(f"[validate] Empty answer for qid={it.qid}")
    # guard against trivial distractors identical to gold snippets
    if it.distractor.strip() in (it.snippet_a.strip(), it.snippet_b.strip()):
        raise ValueError(f"[validate] Distractor duplicates a gold snippet for qid={it.qid}")
    # domains: we keep free text but enforce non-empty
    if it.domain.strip() == "":
        raise ValueError(f"[validate] Empty domain for qid={it.qid}")


def _parse_aliases(raw: str) -> Tuple[str, ...]:
    raw = (raw or "").strip()
    if not raw:
        return tuple()
    # split on pipe and drop empties; keep original capitalization for reporting
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    # dedupe preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            uniq.append(p)
            seen.add(key)
    return tuple(uniq)


def _require_cols(fieldnames: Optional[List[str]], path: Path, required: Set[str]) -> None:
    if not fieldnames:
        raise ValueError(f"[{path.name}] Missing header row.")
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"[{path.name}] Missing required columns: {sorted(missing)}")


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"[load_data] {label} not found at: {path}")


def _req(val: Optional[str], where: str) -> str:
    if val is None:
        raise ValueError(f"[{where}] value is missing")
    v = val.strip()
    if v == "":
        raise ValueError(f"[{where}] value is empty")
    return v


def _warn_strays(label: str, strays: Set[str]) -> None:
    if strays:
        # Non-fatal: context/paraphrase rows whose qid isn’t in questions.
        # You can upgrade this to an exception if you prefer hard fail.
        print(f"[load_data] Warning: {label} has {len(strays)} qid(s) not in questions: {sorted(strays)[:5]}{'…' if len(strays)>5 else ''}")

if __name__ == "__main__":
    from pathlib import Path

    # Adjust paths if needed
    # q_csv = Path("src/data/mhqa_questions.csv")
    # c_csv = Path("src/data/mhqa_context.csv")
    # p_csv = Path("src/data/mhqa_paraphrases.csv")

    q_csv = Path("src/data_50/mhqa_questions_50.csv")
    c_csv = Path("src/data_50/mhqa_context_50.csv")
    p_csv = Path("src/data_50/mhqa_paraphrases_50.csv")

    items = load_items(q_csv, c_csv, p_csv, shuffle=True, seed=42)

    out_path = Path("src/data_50/preview_all.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items: 
            f.write(f"QID: {it.qid}\n")
            f.write(f"Domain: {it.domain}\n")
            f.write(f"Question: {it.question}\n")
            f.write(f"Paraphrase: {it.paraphrase}\n")
            f.write(f"Snippet A: {it.snippet_a[:150]}...\n")
            f.write(f"Snippet B: {it.snippet_b[:150]}...\n")
            f.write(f"Distractor: {it.distractor[:150]}...\n")
            f.write(f"Answer: {it.answer}\n")
            f.write(f"Aliases: {', '.join(it.aliases)}\n")
            f.write("-" * 40 + "\n")
    print(f"Wrote preview to {out_path}")