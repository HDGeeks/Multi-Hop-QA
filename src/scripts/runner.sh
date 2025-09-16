#!/bin/bash
# ============================================================
# runner.sh
# ------------------------------------------------------------
# Master pipeline script.
# Runs (optionally) model inference, scoring, report building,
# figure creation, and LaTeX results export.
#
# Usage:
#   src/scripts/runner.sh <RESULTS_DIR> <REPORTS_DIR>
# Example:
#   src/scripts/runner.sh src/results_test src/reports_test
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

echo "============================================================"
echo ">>> PROJECT_ROOT=$PROJECT_ROOT"
echo ">>> RESULTS_DIR=$RESULTS_DIR"
echo ">>> REPORTS_DIR=$REPORTS_DIR"
echo ">>> Using python=$PY"
echo "============================================================"

# ------------------------------------------------------------
# 0) Run models (optional) 
# ------------------------------------------------------------
# Uncomment if you want to regenerate raw model outputs
# echo ">>> [0/4] Running models..."
# bash "$PROJECT_ROOT/src/scripts/run_models.sh" "$RESULTS_DIR"
# echo ">>> Done running models."

# ------------------------------------------------------------
# 1) Score results
# ------------------------------------------------------------
echo ">>> [1/4] Scoring results..."
bash "$PROJECT_ROOT/src/scripts/score_results.sh" "$RESULTS_DIR"
echo ">>> Done scoring results. Metrics written under:"
echo "    $RESULTS_DIR/<model>/metrics/"
echo

# ------------------------------------------------------------
# 2) Build report (tables + figures + LaTeX)
# ------------------------------------------------------------
echo ">>> [2/4] Building report (tables + figures + LaTeX)..."
bash "$PROJECT_ROOT/src/scripts/make_report.sh" "$RESULTS_DIR" "$REPORTS_DIR"
echo ">>> Done building report."
echo

# ------------------------------------------------------------
# 3) Echo key outputs
# ------------------------------------------------------------
echo ">>> Final artifacts written:"
echo "Tables  -> $REPORTS_DIR/tables/"
ls -1 "$REPORTS_DIR/tables" || true
echo
echo "Figures -> $REPORTS_DIR/figures/"
ls -1 "$REPORTS_DIR/figures" || true
echo
echo "LaTeX   -> $REPORTS_DIR/results.tex"
echo "============================================================"
echo ">>> Runner pipeline complete!"
echo "============================================================"