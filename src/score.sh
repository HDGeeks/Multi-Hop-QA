#!/usr/bin/env bash
set -euo pipefail

# ---------- resolve repo root even if run from anywhere ----------
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

RAW_DIR="$ROOT/src/results/$MODEL"
METRICS_DIR="$ROOT/src/results/$MODEL/metrics"
mkdir -p "$METRICS_DIR"

# ---------- locate gold CSV (try common locations) ----------
CANDIDATES=(
  "$ROOT/data/mhqa_questions.csv"
  "$ROOT/src/data/mhqa_questions.csv"
  "$ROOT/datasets/mhqa_questions.csv"
)
GOLD_CSV=""
for p in "${CANDIDATES[@]}"; do
  if [ -f "$p" ]; then GOLD_CSV="$p"; break; fi
done
if [ -z "${GOLD_CSV:-}" ]; then
  echo "❌ Gold CSV not found. Expected at one of:"
  printf '   - %s\n' "${CANDIDATES[@]}"
  echo "Tip: move or symlink your file to: data/mhqa_questions.csv"
  exit 1
fi
# repo-root-relative for Python
GOLD_CSV_REL="${GOLD_CSV#$ROOT/}"

# ---------- sanity: raw files ----------
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
python -m src.scoring.scoring \
  --glob "src/results/$MODEL/*.jsonl" \
  --out-json "src/results/$MODEL/metrics/${MODEL}_summary.json"
echo "✅ Wrote JSON summary → src/results/$MODEL/metrics/${MODEL}_summary.json"

# ---------- 2) Extended scoring ----------
echo "[2/3] Extended scoring..."
python -m src.scoring.extended_scoring \
  --glob "src/results/$MODEL/*.jsonl" \
  --model "$MODEL" \
  --outdir "src/results/$MODEL/metrics" \
  --gold-csv "$GOLD_CSV_REL"

# ---------- 3) Domain-specific scoring ----------
echo "[3/3] Domain scoring..."
python -m src.scoring.domain_scoring \
  --in-csv "src/results/$MODEL/metrics/${MODEL}_aggregated_items.csv" \
  --out-json "src/results/$MODEL/metrics/${MODEL}_per_domain.json"

echo "✅ All scoring finished."
echo "   Metrics in: src/results/$MODEL/metrics/"
ls -1 "src/results/$MODEL/metrics/" || true
