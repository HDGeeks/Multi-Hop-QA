# src/data/load_data.py
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class Item:
    qid: str
    domain: str
    question: str
    paraphrase: Optional[str]
    snippet_a: str
    snippet_b: str
    distractor: str
    answer: str

def load_items(base_dir: Path) -> list[Item]:
    # Load questions
    questions = {}
    with (base_dir / "mhqa_questions.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            questions[row["qid"]] = {
                "qid": row["qid"],
                "domain": row["domain"],
                "question": row["question"],
                "answer": row["answer"],
            }

    # Load contexts
    contexts = {}
    with (base_dir / "mhqa_context.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            contexts[row["qid"]] = {
                "snippet_a": row["snippet_a"],
                "snippet_b": row["snippet_b"],
                "distractor": row["distractor"],
            }

    # Load paraphrases
    paras = {}
    with (base_dir / "mhqa_paraphrases.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            paras[row["qid"]] = row["paraphrase"]

    # Merge into Item objects
    items = []
    for qid, q in questions.items():
        ctx = contexts[qid]
        para = paras.get(qid, None)
        items.append(Item(
            qid=q["qid"],
            domain=q["domain"],
            question=q["question"],
            paraphrase=para,
            snippet_a=ctx["snippet_a"],
            snippet_b=ctx["snippet_b"],
            distractor=ctx["distractor"],
            answer=q["answer"],
        ))

    return items

if __name__ == "__main__":
    base = Path("src/data")
    items = load_items(base)
    print(f"Loaded {len(items)} items")
    print(items[0])  # peek at first