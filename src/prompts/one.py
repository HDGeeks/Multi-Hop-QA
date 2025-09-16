
# src/prompts/one.py
"""
one.py

This script provides a utility for displaying prompt variants for a specific question ID (qid) from a multi-hop QA dataset.
It loads question, context, and paraphrase data from a 50-question subset, finds the requested item, and prints out all prompt settings
("gold", "para", "dist", "para_dist") for inspection. Intended for debugging and prompt engineering.

Functions:
    show(qid: str):
        Loads the dataset, finds the item with the given qid, and prints all prompt variants for that item.

Usage:
    Run as a script with an optional QID argument:
        python one.py SCI002

# This file is a utility for inspecting prompt variants for individual questions in the 50-question multi-hop QA dataset.
"""
import sys
from pathlib import Path

# make repo root importable when running as a script
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.load_data import load_items
from src.prompts.prompts_builder import make_all_prompts, ALL_SETTINGS

def show(qid: str):
    repo_root = Path(__file__).resolve().parents[2]
    base = repo_root / "src" / "data_50"   # use your new 50-question dataset

    items = load_items(
        questions_csv=base / "mhqa_questions_50.csv",
        context_csv=base / "mhqa_context_50.csv",
        paraphrases_csv=base / "mhqa_paraphrases_50.csv",
    )

    # find the item
    item = next((it for it in items if it.qid == qid), None)
    if not item:
        print(f"[!] QID '{qid}' not found. Available: {[it.qid for it in items[:5]]} ...")
        return

    print(f"QID: {item.qid} | Domain: {item.domain}\n")

    prompts = make_all_prompts(item)
    for setting in ALL_SETTINGS:  # ["gold", "para", "dist", "para_dist"]
        print(f"--- {setting.upper()} ---")
        print(prompts[setting])
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    arg_qid = sys.argv[1] if len(sys.argv) > 1 else "HIST009"
    show(arg_qid)