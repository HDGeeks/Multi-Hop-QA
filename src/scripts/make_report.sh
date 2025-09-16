#!/bin/bash
# ============================================================
# make_report.sh
# ------------------------------------------------------------
# 1) Build all tables (CSV) from per-model metrics
#    using src.report.build_from_metrics
# 2) Build all figures (PNG) from RAW metrics JSON/CSV
#    using src.report.make_bar_charts_from_raw
# 3) Emit LaTeX results section from the tables + figs
#
# Usage:
#   src/scripts/make_report.sh <RESULTS_DIR> <REPORTS_DIR>
# Example:
#   src/scripts/make_report.sh src/results src/reports
#
# Outputs:
#   <REPORTS_DIR>/tables/*.csv
#   <REPORTS_DIR>/figures/*.png
#   <REPORTS_DIR>/results.tex
# ============================================================

set -euo pipefail

RESULTS_DIR="${1:-}"
REPORTS_DIR="${2:-}"

if [[ -z "$RESULTS_DIR" || -z "$REPORTS_DIR" ]]; then
  echo "Usage: $0 <RESULTS_DIR> <REPORTS_DIR>" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

PY="$(command -v python3 || command -v python)"
if [[ -x "$PROJECT_ROOT/.venv/bin/python3" ]]; then
  PY="$PROJECT_ROOT/.venv/bin/python3"
fi

echo ">>> RESULTS_DIR=$RESULTS_DIR"
echo ">>> REPORTS_DIR=$REPORTS_DIR"
echo ">>> PROJECT_ROOT=$PROJECT_ROOT"
echo ">>> Using python: $PY"

# Ensure output dirs exist
mkdir -p "$REPORTS_DIR/tables" "$REPORTS_DIR/figures"

# Models to include (auto-detect if present)
DEFAULT_MODELS=("gpt4o" "gpt4o_mini" "gemini_pro" "llama31_8b" "mistral7b")
MODELS=()
for m in "${DEFAULT_MODELS[@]}"; do
  if [[ -d "$RESULTS_DIR/$m/metrics" ]]; then
    MODELS+=("$m")
  fi
done
if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "ERROR: No model metrics found under $RESULTS_DIR/<model>/metrics/" >&2
  exit 2
fi
echo ">>> Models: ${MODELS[*]}"

# 1) Build TABLES from metrics (no plotting inside)
set -x
"$PY" -m src.reporters.build_from_metrics \
  --results-dir "$RESULTS_DIR" \
  --reports-dir "$REPORTS_DIR" \
  --models "${MODELS[@]}"
set +x

# 2) Build FIGURES from RAW metrics (paper-consistent)
#    Remove old PNGs safely (no error if none)
shopt -s nullglob
rm -f "$REPORTS_DIR/figures"/*.png || true
set -x
"$PY" -m src.reporters.make_bar_charts \
  --results-root "$RESULTS_DIR" \
  --out-dir "$REPORTS_DIR/figures"
set +x

# 3) Write LaTeX results section
set -x
"$PY" -m src.reporters.write_results_tex \
  --tables-dir "$REPORTS_DIR/tables" \
  --figs-dir   "$REPORTS_DIR/figures" \
  --out-tex    "$REPORTS_DIR/results.tex" \
  --model-order "Gemini Pro,GPT-4o Mini,GPT-4o,Mistral-7B,LLaMA-3.1-8B" \
  --em-dp 1 --f1-dp 1 --bert-dp 6 --pp-dp 1 --ms-dp 1
set +x

echo ">>> Done."
echo ">>> Tables -> $REPORTS_DIR/tables"
ls -1 "$REPORTS_DIR/tables" || true
echo ">>> Figures -> $REPORTS_DIR/figures"
ls -1 "$REPORTS_DIR/figures" || true
echo ">>> LaTeX  -> $REPORTS_DIR/results.tex"