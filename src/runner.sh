#!/usr/bin/env bash
set -euo pipefail

# ===== Resolve repo root and use relative paths =====
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python3"
RESULTS_REL="src/results_50"
DATA_REL="src/data_50"

MODELS=("gpt4o" "gemini_pro" "llama31_8b" "mistral7b" "gpt4o_mini")
RUNS=3

# ===== Run models =====
# for model in "${MODELS[@]}"; do
#   echo "=== Running $model ($RUNS runs) ==="
#   for run_id in $(seq 1 "$RUNS"); do
#     "$PY" -m src.runner --model "$model" --run-id "$run_id"
#   done
# done

# ===== Scoring (v2) =====
for model in "${MODELS[@]}"; do
  echo "=== Scoring $model (v2) ==="
  METRICS_DIR="$RESULTS_REL/$model/metrics"
  mkdir -p "$METRICS_DIR"

  # Use RELATIVE glob so v2 scorers don't see absolute patterns
  GLOB_PATTERN="$RESULTS_REL/$model/*.jsonl"

  # Quick sanity: skip if no files (prevents confusing traceback)
  if ! compgen -G "$GLOB_PATTERN" > /dev/null; then
    echo "[WARN] No JSONL files for $model at $GLOB_PATTERN — skipping scoring."
    continue
  fi

  "$PY" -m src.scoring.scoring_v2 \
    --glob "$GLOB_PATTERN" \
    --gold-csv "$DATA_REL/mhqa_questions_50.csv" \
    --context-csv "$DATA_REL/mhqa_context_50.csv" \
    --paras-csv "$DATA_REL/mhqa_paraphrases_50.csv" \
    --out-json "$METRICS_DIR/${model}_scoring_v2.json"

  "$PY" -m src.scoring.scoring_v2_extended \
    --glob "$GLOB_PATTERN" \
    --gold-csv "$DATA_REL/mhqa_questions_50.csv" \
    --out-json "$METRICS_DIR/${model}_scoring_v2_extended.json"

  "$PY" -m src.scoring.bertscore_scoring_v2 \
    --glob "$GLOB_PATTERN" \
    --gold-csv "$DATA_REL/mhqa_questions_50.csv" \
    --model "$model" \
    --outdir "$METRICS_DIR" \
    --bertscore-model "roberta-large" --bertscore-lang "en"
done

# ===== Report assets =====
echo "=== Building report assets ==="
"$PY" -m src.report.build_report_assets \
  --results-root "$RESULTS_REL" \
  --out-root "src/report_50" \
  --models "${MODELS[@]}"

echo "✅ All done. Tables → src/report_50/tables, Figures → src/report_50/figures"