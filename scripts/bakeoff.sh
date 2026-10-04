#!/usr/bin/env bash
# Step 5: backbone x LR bake-off by 5-fold CV on train+val (decisions.md D-03, G18-G19). Hard cap: 6 configs.
# Results: reports/cv/<name>.json (+ _oof.csv); W&B group "cv-bakeoff". Compare with: python -m src.bakeoff_report
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

[ -f reports/cv/cv-tfidf-baseline.json ] || $PY -m src.baseline --cv
for spec in \
  "minilm  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
  "mpnet   sentence-transformers/paraphrase-multilingual-mpnet-base-v2" \
  "xlmr    FacebookAI/xlm-roberta-base"; do
  read -r short model <<<"$spec"
  for lr in 2e-5 5e-5; do
    name="cv-$short-lr$lr"
    if [ -f "reports/cv/$name.json" ]; then echo "[skip] $name already done"; continue; fi  # resumable
    $PY -m src.train --config configs/base.yaml --mode cv --set model="$model" lr="$lr" name="$name"
  done
done
