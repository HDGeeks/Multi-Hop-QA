#!/usr/bin/env bash
set -euo pipefail

# Activate venv
source "$(dirname "$0")/.venv/bin/activate"

# Directories
RESULTS_DIR="src/results"
RAW_DIR="$RESULTS_DIR/raw"
METRICS_DIR="$RESULTS_DIR/metrics"

mkdir -p "$RAW_DIR" "$METRICS_DIR"

# -------- 1. Quick scoring check (optional sanity) --------
echo "[1/3] Running quick scoring sanity check..."
python -m src.scoring.scoring \
  --glob "$RAW_DIR/gpt4o/*_run*.jsonl" \
  --out-json "$METRICS_DIR/gpt4o_summary.json"

# -------- 2. Extended scoring (full pipeline) --------
echo "[2/3] Running extended scoring..."
python -m src.scoring.extended_scoring \
  --glob "$RAW_DIR/gpt4o/*_run*.jsonl" \
  --out-dir "$METRICS_DIR"

# -------- 3. Domain-specific scoring --------
echo "[3/3] Running domain scoring..."
python -m src.scoring.domain_scoring \
  --in-csv "$METRICS_DIR/gpt4o_aggregated_items.csv" \
  --out-json "$METRICS_DIR/gpt4o_per_domain.json"

echo "✅ All scoring finished. Results in $METRICS_DIR"