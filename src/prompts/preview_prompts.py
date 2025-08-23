# src/prompts/preview_prompts.py
from pathlib import Path
from data.load_data import load_items
from .prompts_builder import make_all_prompts, ALL_SETTINGS

def main():
    base = Path("src/data")
    items = load_items(base)
    print(f"Loaded {len(items)} items")

    out_file = Path("src/prompts/preview_prompts.txt")
    with out_file.open("w", encoding="utf-8") as f:
        for item in items[:3]:  # just first 3 for sanity check
            f.write(f"QID: {item.qid} | Domain: {item.domain}\n")
            for setting in ALL_SETTINGS:
                prompt = make_all_prompts(item)[setting]
                f.write(f"\n--- {setting.upper()} ---\n")
                f.write(prompt + "\n\n")
            f.write("="*80 + "\n\n")

    print(f"Preview written to {out_file.resolve()}")

if __name__ == "__main__":
    main()