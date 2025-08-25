# src/prompts/one.py
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
    arg_qid = sys.argv[1] if len(sys.argv) > 1 else "SCI002"
    show(arg_qid)