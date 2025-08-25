# src/prompts/prompts_preview.py

import sys
import argparse
from pathlib import Path

# Make repo root importable
THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]
sys.path.append(str(ROOT))

from src.data.load_data import load_items
from src.prompts.prompts_builder import make_all_prompts, ALL_SETTINGS  # ["gold","para","dist","para_dist"]

def write_domain(items, domain: str, out_dir: Path, limit: int | None):
    items_d = [it for it in items if it.domain == domain]
    if limit is not None:
        items_d = items_d[:limit]

    out_path = out_dir / f"{domain}_prompts.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Domain: {domain} | Items: {len(items_d)}")
    lines.append(f"# Settings: {', '.join(ALL_SETTINGS)}")
    lines.append("=" * 80 + "\n")

    for it in items_d:
        lines.append(f"QID: {it.qid} | Domain: {it.domain}")
        lines.append("-" * 80)
        prompts = make_all_prompts(it)
        for setting in ALL_SETTINGS:
            lines.append(f"[{setting.upper()}]")
            lines.append(prompts[setting].rstrip() + "\n")
        lines.append("=" * 80 + "\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Wrote {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Dump constructed prompts per domain into text files.")
    ap.add_argument("--domain", choices=["history", "science", "politics", "all"], default="all",
                    help="Which domain to dump (default: all).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional max items per domain (for quick inspection).")
    ap.add_argument("--outdir", default=None,
                    help="Output directory (default: src/prompts/preview).")
    args = ap.parse_args()

    data_dir = ROOT / "src" / "data_50"
    items = load_items(
        questions_csv=data_dir / "mhqa_questions_50.csv",
        context_csv=data_dir / "mhqa_context_50.csv",
        paraphrases_csv=data_dir / "mhqa_paraphrases_50.csv",
    )

    out_dir = Path(args.outdir) if args.outdir else (ROOT / "src" / "prompts" / "preview")

    domains = ["history", "science", "politics","geography","literature"] if args.domain == "all" else [args.domain]
    for d in domains:
        write_domain(items, d, out_dir, args.limit)

if __name__ == "__main__":
    main()