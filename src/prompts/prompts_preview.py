
"""
prompts_preview.py

This script is designed to generate and export prompt previews for multi-hop question answering tasks, organized by domain.
It loads question, context, and paraphrase data, constructs prompts using various settings, and writes the results to text files for inspection or further use.

Main Features:
--------------
- Loads items from CSV files containing questions, contexts, and paraphrases.
- Constructs prompts for each item using multiple settings (e.g., "gold", "para", "dist", "para_dist").
- Filters items by domain and optionally limits the number of items processed.
- Writes formatted prompt previews to domain-specific text files in a specified output directory.

Functions:
----------
- write_domain(items, domain, out_dir, limit):
    Filters items by domain, constructs prompts for each, and writes them to a text file.
    Inputs:
        - items: List of loaded question items.
        - domain: Domain to filter (e.g., "history", "science").
        - out_dir: Output directory for the text files.
        - limit: Optional maximum number of items to process.
    Output:
        - Writes a text file named "{domain}_prompts.txt" in the output directory.

- main():
    Parses command-line arguments, loads data, and calls write_domain for each selected domain.

Usage:
------
Run this script from the command line to generate prompt previews:

    python prompts_preview.py --domain history --limit 10 --outdir /path/to/output

Arguments:
----------
--domain   : Domain to process ("history", "science", "politics", "all"). Default: "all".
--limit    : Optional maximum number of items per domain. Default: None (no limit).
--outdir   : Output directory for prompt files. Default: "src/prompts/preview" under repo root.

Inputs:
-------
- CSV files in "src/data_50" directory:
    - mhqa_questions_50.csv
    - mhqa_context_50.csv
    - mhqa_paraphrases_50.csv

Outputs:
--------
- Text files containing prompt previews for each domain, written to the specified output directory.

Example:
--------
To generate prompt previews for the "science" domain, limited to 5 items, and save them to "output/prompts":

    python prompts_preview.py --domain science --limit 5 --outdir output/prompts

"""
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