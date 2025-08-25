#!/usr/bin/env bash
set -euo pipefail

# run from repo root or from src/report
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# activate venv
source ".venv/bin/activate"

# install deps if needed
python3 - <<'PY'
import sys, pkgutil
need = []
for m in ["pandas","numpy","matplotlib"]:
    if not pkgutil.find_loader(m):
        need.append(m)
if need:
    print("Installing:", need)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", *need])
else:
    print("Deps OK")
PY

# build assets
python3 src/report/build_report_assets.py \
  --models gpt4o gpt4o_mini gemini_pro llama31_8b mistral7b \
  --results-root src/results \
  --out-root src/report

echo "✅ Report assets are ready."
echo "  Tables:  src/report/tables/*.tex"
echo "  Figures: src/report/figures/*.pdf"