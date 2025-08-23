import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.load_data import load_items
from src.prompts.prompts_builder import make_all_prompts, ALL_SETTINGS

def main():
    repo_root = Path(__file__).resolve().parents[2]
    base = repo_root / "src" / "data"
    items = load_items(base)
    print(f"Loaded {len(items)} items")

    out_file = repo_root / "src" / "prompts" / "preview_prompts.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        for item in items[:3]:
            f.write(f"QID: {item.qid} | Domain: {item.domain}\n")
            for setting in ALL_SETTINGS:
                prompt = make_all_prompts(item)[setting]
                f.write(f"\n--- {setting.upper()} ---\n{prompt}\n\n")
            f.write("="*80 + "\n\n")

    print(f"Preview written to {out_file.resolve()}")

if __name__ == "__main__":
    main()