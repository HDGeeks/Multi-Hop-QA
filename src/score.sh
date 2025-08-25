#!/usr/bin/env bash
set -euo pipefail

# ---------- resolve repo root ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# ---------- venv ----------
if [ ! -f "$ROOT/.venv/bin/activate" ]; then
  echo "❌ venv not found at $ROOT/.venv. Create it and install deps first."
  exit 1
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

# ---------- args ----------
MODEL="${1:-gpt4o}"
RAW_DIR="$ROOT/src/results_50/$MODEL"
METRICS_DIR="$RAW_DIR/metrics"
mkdir -p "$METRICS_DIR"

# ---------- gold CSV (use the 50-item dataset) ----------
GOLD_CSV="src/data_50/mhqa_questions_50.csv"
if [ ! -f "$GOLD_CSV" ]; then
  echo "❌ Gold CSV not found at $GOLD_CSV"
  exit 1
fi

echo "Scoring $MODEL with gold=$GOLD_CSV"
echo "Looking for JSONL files at: $RAW_DIR/*.jsonl"

shopt -s nullglob
FILES=("$RAW_DIR"/*.jsonl)
shopt -u nullglob
if [ ${#FILES[@]} -eq 0 ]; then
  echo "❌ No raw jsonl files found in $RAW_DIR"
  exit 1
fi

# ---------- 1) Quick summary ----------
echo "[1/3] Quick scoring..."
python3 -m src.scoring.scoring \
  --glob "src/results_50/$MODEL/*.jsonl" \
  --out-json "src/results_50/$MODEL/metrics/${MODEL}_summary.json"
echo "✅ Wrote JSON summary → src/results_50/$MODEL/metrics/${MODEL}_summary.json"

# ---------- 2) Extended scoring ----------
echo "[2/3] Extended scoring..."
python3 -m src.scoring.extended_scoring \
  --glob "src/results_50/$MODEL/*.jsonl" \
  --model "$MODEL" \
  --outdir "src/results_50/$MODEL/metrics" \
  --gold-csv "$GOLD_CSV"

# ---------- 3) Domain-specific scoring ----------
echo "[3/3] Domain scoring..."
python3 -m src.scoring.domain_scoring \
  --in-csv "src/results_50/$MODEL/metrics/${MODEL}_aggregated_items.csv" \
  --out-json "src/results_50/$MODEL/metrics/${MODEL}_per_domain.json"

# ---------- 4) BERTScore ----------
echo "[4/4] BERTScore..."
python3 -m src.scoring.bertscore_scoring \
  --glob "src/results_50/$MODEL/*.jsonl" \
  --gold-csv "$GOLD_CSV" \
  --model "$MODEL" \
  --outdir "src/results_50/$MODEL/metrics" \
  --bertscore-model "roberta-large" \
  --bertscore-lang "en"

echo "✅ All scoring finished."
echo "   Metrics in: src/results_50/$MODEL/metrics/"
ls -1 "src/results_50/$MODEL/metrics/" || true