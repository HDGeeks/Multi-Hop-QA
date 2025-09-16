#!/bin/bash
# ============================================================
# run_models.sh
# ------------------------------------------------------------
# Runs all models (gpt4o, gpt4o_mini, gemini_pro, llama31_8b, mistral7b)
# across 3 runs each using runner.py.
#
# Usage:
#   ./src/scripts/run_models.sh <output_dir>
#
# Example:
#   ./src/scripts/run_models.sh src/results_test
#
# Results:
#   JSONL outputs will be stored under:
#   <output_dir>/<model>/<model>_run<id>.jsonl
# ============================================================

set -euo pipefail

# --- Arg check ---
if [ $# -lt 1 ]; then
  echo "Usage: $0 <output_dir>"
  exit 1
fi
OUT_DIR="$1"

# --- Compute paths ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"             # /.../Multi-Hop-QA/src/scripts
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"        # /.../Multi-Hop-QA   <-- REAL ROOT
cd "$PROJECT_ROOT"

echo ">>> DEBUG: PROJECT_ROOT"
pwd
echo ">>> DEBUG: ls -la (root)"
ls -la

# Sanity: ensure top-level src exists and contains runner.py
if [ ! -d "src" ]; then
  echo "ERROR: Top-level 'src' directory not found at $PROJECT_ROOT"
  exit 1
fi
if [ ! -f "src/runner.py" ]; then
  echo "ERROR: 'src/runner.py' not found at $PROJECT_ROOT/src/runner.py"
  exit 1
fi

echo ">>> DEBUG: PYTHONPATH will include PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
echo ">>> DEBUG: PYTHONPATH=$PYTHONPATH"

# --- Choose python executable (prefer venv if present) ---
if [ -x ".venv/bin/python3" ]; then
  PY=".venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "ERROR: No python interpreter found."
  exit 1
fi
echo ">>> DEBUG: Using python at: $PY"

# --- Models and runs ---
MODELS=("gpt4o" "gpt4o_mini" "gemini_pro" "llama31_8b" "mistral7b")
N_RUNS=3

# --- Create base output dir (relative to root) ---
mkdir -p "$OUT_DIR"

# --- Run loop ---
for model in "${MODELS[@]}"; do
  for run_id in $(seq 1 "$N_RUNS"); do
    MODEL_OUT_DIR="${OUT_DIR}/${model}"
    mkdir -p "$MODEL_OUT_DIR"
    OUT_FILE="${MODEL_OUT_DIR}/${model}_run${run_id}.jsonl"

    echo ">>> Running model=$model run_id=$run_id"
    echo ">>> Will write to: $OUT_FILE"

    # Try running as a module first (recommended)
    set +e
    "$PY" -m src.runner \
      --model "$model" \
      --run-id "$run_id" \
      --out "$OUT_FILE"
    STATUS=$?
    set -e

    # Fallback to direct script execution if module import fails
    if [ $STATUS -ne 0 ]; then
      echo ">>> WARNING: 'python -m src.runner' failed (status=$STATUS). Falling back to 'python src/runner.py'..."
      "$PY" src/runner.py \
        --model "$model" \
        --run-id "$run_id" \
        --out "$OUT_FILE"
    fi

    echo ">>> Finished model=$model run_id=$run_id"
    echo "----------------------------------------------"
  done
done

echo ">>> All model runs completed. Outputs are in ${OUT_DIR}/"