#!/bin/bash
# ============================================================
# score_results.sh
# ------------------------------------------------------------
# Master scorer: runs all scoring phases to produce *all 9*
# metrics files per model.
#
# Usage:
#   src/scripts/score_results.sh <RESULTS_DIR>
# Example:
#   src/scripts/score_results.sh src/results
#
# Outputs per model (to <RESULTS_DIR>/<model>/metrics/):
#   1) per_run_v2.csv
#   2) aggregated_items_v2.csv
#   3) summary_v2.csv
#   4) <model>_bertscore_per_run_v2.csv
#   5) <model>_bertscore_aggregated_items_v2.csv
#   6) <model>_bertscore_by_domain_v2.csv
#   7) <model>_bertscore_summary_v2.json
#   8) <model>_scoring_v2.json
#   9) <model>_scoring_v2_extended.json
#
# Mapping to Python scripts:
#   - scoring_v2.py           → files 1–3 and 8
#   - bertscore_scoring_v2.py → files 4–7
#   - scoring_v2_extended.py  → file 9
# ============================================================

set -euo pipefail

RESULTS_DIR="${1:-}"
if [[ -z "$RESULTS_DIR" ]]; then
  echo "Usage: $0 <RESULTS_DIR>" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

PY="$(command -v python3 || command -v python)"
if [[ -x "$PROJECT_ROOT/.venv/bin/python3" ]]; then
  PY="$PROJECT_ROOT/.venv/bin/python3"
fi

MODELS=("gpt4o" "gpt4o_mini" "gemini_pro" "llama31_8b" "mistral7b")

QUESTIONS="$PROJECT_ROOT/src/data/mhqa_questions_50.csv"
CONTEXT="$PROJECT_ROOT/src/data/mhqa_context_50.csv"
PARAS="$PROJECT_ROOT/src/data/mhqa_paraphrases_50.csv"

echo ">>> PROJECT_ROOT=$PROJECT_ROOT"
echo ">>> RESULTS_DIR=$RESULTS_DIR"
echo ">>> Using python: $PY"
echo ">>> DATA:"
echo "    QUESTIONS=$QUESTIONS"
echo "    CONTEXT  =$CONTEXT"
echo "    PARAS    =$PARAS"

for model in "${MODELS[@]}"; do
  model_dir="$PROJECT_ROOT/$RESULTS_DIR/$model"
  metrics_dir="$model_dir/metrics"
  mkdir -p "$metrics_dir"

  echo "============================================================"
  echo ">>> MODEL: $model"
  echo "------------------------------------------------------------"

  RUN_GLOB="$RESULTS_DIR/$model/${model}_run*.jsonl"

  # 1) Core scoring: EM/F1/Latency (+ per_run/agg_items/summary CSVs + compact JSON)
  echo ">>> [1/3] scoring_v2.py → $metrics_dir/${model}_scoring_v2.json"
  set -x
  "$PY" -m src.scoring.scoring_v2 \
      --glob "$RUN_GLOB" \
      --gold-csv "$QUESTIONS" \
      --context-csv "$CONTEXT" \
      --paras-csv "$PARAS" \
      --out-json "$metrics_dir/${model}_scoring_v2.json"
  set +x

  # 2) BERTScore scoring
  echo ">>> [2/3] bertscore_scoring_v2.py (semantic similarity)"
  set -x
  "$PY" -m src.scoring.bertscore_scoring_v2 \
      --glob "$RUN_GLOB" \
      --gold-csv "$QUESTIONS" \
      --model "$model" \
      --outdir "$metrics_dir"
  set +x

  # 3) Extended summaries (uses RAW JSONL glob + gold CSV)
  echo ">>> [3/3] scoring_v2_extended.py (rollups)"
  set -x
  "$PY" -m src.scoring.scoring_v2_extended \
      --glob "$RUN_GLOB" \
      --gold-csv "$QUESTIONS" \
      --out-json "$metrics_dir/${model}_scoring_v2_extended.json"
  set +x

  echo ">>> Done with model=$model"
done

echo "============================================================"
echo ">>> All models scored. Full set of 9 files written under:"
echo "    $RESULTS_DIR/<model>/metrics/"
echo "============================================================"