#!/usr/bin/env bash
set -euo pipefail

# -------- CONFIG --------
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv/bin/python3"
RESULTS="$ROOT/src/results_50"
DATA="$ROOT/src/data_50"
MODELS=("gpt4o" "gemini_pro" "llama31_8b" "mistral7b" "gpt4o_mini")
RUNS=3

# -------- RUN MODELS --------
# -------- RUN MODELS --------
for model in "${MODELS[@]}"; do
  echo "=== Running $model ($RUNS runs) ==="
  for run_id in $(seq 1 $RUNS); do
    $VENV -m src.runner --model "$model" --run-id "$run_id"
  done
done

# -------- SCORING --------
for model in "${MODELS[@]}"; do
  echo "=== Scoring $model (v2) ==="
  METRICS_DIR="$RESULTS/$model/metrics"
  mkdir -p "$METRICS_DIR"

  $VENV -m src.scoring.scoring_v2 \
    --glob "$RESULTS/$model/*.jsonl" \
    --gold-csv "$DATA/mhqa_questions_50.csv" \
    --context-csv "$DATA/mhqa_context_50.csv" \
    --paras-csv "$DATA/mhqa_paraphrases_50.csv" \
    --out-json "$METRICS_DIR/${model}_scoring_v2.json"

  $VENV -m src.scoring.scoring_v2_extended \
    --glob "$RESULTS/$model/*.jsonl" \
    --gold-csv "$DATA/mhqa_questions_50.csv" \
    --out-json "$METRICS_DIR/${model}_scoring_v2_extended.json"

  $VENV -m src.scoring.bertscore_scoring_v2 \
    --glob "$RESULTS/$model/*.jsonl" \
    --gold-csv "$DATA/mhqa_questions_50.csv" \
    --model "$model" \
    --outdir "$METRICS_DIR" \
    --bertscore-model "roberta-large" --bertscore-lang "en"
done

# -------- REPORT --------
echo "=== Building report assets ==="
$VENV src/report/build_report_assets.py \
  --results-root "$RESULTS" \
  --out-root "$ROOT/src/report_50" \
  --models "${MODELS[@]}"

echo "✅ All done. Tables → src/report_50/tables, Figures → src/report_50/figures"