# src/runner.py

"""
This script orchestrates the evaluation of multiple language models on a set of multi-hop QA items using various prompt settings.
It loads data, builds prompts, queries models, and writes results to a JSONL file for further analysis.

Main Features:
--------------
- Loads multi-hop QA items and associated context/paraphrases from CSV files.
- Supports multiple prompt settings: GOLD, PARA, DIST, PARA_DIST.
- Dispatches queries to one of several supported model clients (OpenAI GPT-4o, Gemini Pro, Llama, Mistral).
- Records model outputs, latency, usage statistics, and errors for each item/setting combination.
- Writes results as JSONL, including full prompt for reproducibility.

Functions:
----------
- get_model_callable(model_id: str): Returns a callable for querying the specified model.
- ensure_dirs(path: Path): Ensures output directory exists.
- now_iso(): Returns current UTC timestamp in ISO format.
- main(): Entry point; parses arguments, runs evaluation, writes output.

Usage:
------
Run from the command line with required arguments:

    python src/runner.py --model gpt4o --run-id 1

Arguments:
----------
--model    (required) : Model identifier. One of: gpt4o, gpt4o_mini, gemini_pro, llama31_8b, mistral7b
--run-id   (default=1): Integer run index for repeatability.
--out      (optional) : Output path for JSONL results. Defaults to 'src/results_50/{model}/{model}_run{run_id}.jsonl'
--qid      (optional) : If provided, only runs the specified QID.

Inputs:
-------
- Loads QA items from CSV files in 'src/data_50/' (or 'src/data/' if uncommented).
- Requires model API keys/configuration as needed by model clients.

Outputs:
--------
- Writes one JSONL row per (item, setting) to the specified output file, including metadata and model response.

Example:
--------
To run GPT-4o on all items and settings, saving results to default location:

    python src/runner.py --model gpt4o --run-id 1

To run only QID 'Q123' and save to a custom file:

    python src/runner.py --model llama31_8b --run-id 2 --qid Q123 --out results/custom_run.jsonl

Notes:
------
- Ensure model API credentials are configured for the selected model.
- Output JSONL includes full prompt, model output, latency, and error info for reproducibility and debugging.

"""
import json
from pathlib import Path
from datetime import datetime
import argparse
import time

# data + prompts
from src.data.load_data import load_items
from src.prompts.prompts_builder import build_prompt, GOLD, PARA, DIST, PARA_DIST

# model clients
from src.models.openai_4o_client import query_openai_4o
from src.models.openai_4o_mini_client import query_openai_4o_mini
from src.models.gemini_pro_client import query_gemini_pro
from src.models.llama_client import query_llama
from src.models.mistral_client import query_mistral

SETTINGS = [GOLD, PARA, DIST, PARA_DIST]
# decoding defaults for all models
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 64

# ---- model dispatch (one model per run) ----
def get_model_callable(model_id: str):
    """
    Return a callable with signature: fn(prompt: str) -> dict
    Each fn must return a dict containing at least:
        { 'output': str, 'error': str|None, 'version': str, 'finish_reason': str|None, 'usage': dict, 'latency_ms': int? }
    """
    model_id = model_id.lower()
    if model_id == "gpt4o":
        # your openai 4o wrapper typically ignores top_p; keep temperature/max_tokens
        return lambda p: query_openai_4o(
            p, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS
        )
    if model_id == "gpt4o_mini":
        return lambda p: query_openai_4o_mini(
            p, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS, model_override="gpt-4o-mini"
        )
    if model_id == "gemini_pro":
        return lambda p: query_gemini_pro(
            p, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P,
            max_tokens=DEFAULT_MAX_TOKENS, model_override="gemini-1.5-pro"
        )
    if model_id == "llama31_8b":
        return lambda p: query_llama(
            p, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P,
            max_tokens=DEFAULT_MAX_TOKENS, model="meta-llama/Llama-3.1-8B-Instruct"
        )
    if model_id == "mistral7b":
        return lambda p: query_mistral(
            p, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P,
            max_tokens=DEFAULT_MAX_TOKENS, model="mistralai/Mistral-7B-Instruct-v0.3"
        )
    raise ValueError(f"Unknown model_id: {model_id}")
def ensure_dirs(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def now_iso():
    return datetime.utcnow().isoformat()

def main():
    parser = argparse.ArgumentParser(description="Run a single model over all items and 4 settings; write raw JSONL.")
    parser.add_argument("--model", required=True, help="One of: gpt4o, gpt4o_mini, gemini_pro, llama31_8b, mistral7b")
    parser.add_argument("--run-id", type=int, default=1, help="Repeat index (1..n)")
    parser.add_argument("--out", default=None, help="Optional override path for output JSONL")
    parser.add_argument("--qid", default=None, help="Optional: only run this QID")
    args = parser.parse_args()

    model_id = args.model
    run_id = args.run_id

    # load items
    # base = Path("src/data")
    # questions_csv = base / "mhqa_questions.csv"
    # context_csv   = base / "mhqa_context.csv"
    # paras_csv     = base / "mhqa_paraphrases.csv"
    # items = load_items(questions_csv, context_csv, paras_csv)

    base = Path("src/data")
    questions_csv = base / "mhqa_questions_50.csv"
    context_csv   = base / "mhqa_context_50.csv"
    paras_csv     = base / "mhqa_paraphrases_50.csv"
    items = load_items(questions_csv, context_csv, paras_csv)

    # optional: filter to a single QID
    if args.qid:
        items = [it for it in items if it.qid == args.qid]
    if not items:
        raise ValueError(f"No item with qid={args.qid}")
    # output path (per-model, per-run)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(f"src/results/{model_id}/{model_id}_run{run_id}.jsonl")

    ensure_dirs(out_path)

    call = get_model_callable(model_id)

    print(f"Running model={model_id} run_id={run_id} on {len(items)} items × {len(SETTINGS)} settings")
    t_start = time.time()
    n = 0

    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            for setting in SETTINGS:
                prompt = build_prompt(item, setting)
                t0 = time.time()
                res = call(prompt)  # dict from wrapper
                # Some wrappers include latency; if not, compute
                latency_ms = res.get("latency_ms")
                if latency_ms is None:
                    latency_ms = int((time.time() - t0) * 1000)

                row = {
                    "run_id": run_id,
                    "qid": item.qid,
                    "domain": item.domain,
                    "model": model_id,
                    "setting": setting,
                    "prompt": prompt,                 # keep full prompt for reproducibility
                    "output": res.get("output", ""),
                    "latency_ms": latency_ms,
                    "ts": now_iso(),
                    "version": res.get("version", ""),
                    "finish_reason": res.get("finish_reason"),
                    "usage": res.get("usage", {}),
                    "error": res.get("error"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1

                # lightweight progress
                if n % 20 == 0:
                    print(f"  wrote {n} rows...")

    elapsed = int((time.time() - t_start))
    print(f"Done. Wrote {n} rows to {out_path} in {elapsed}s.")

if __name__ == "__main__":
    main()