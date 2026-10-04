#!/usr/bin/env bash
# Stretch round 2 (added after the error analysis found the ID-prefix shortcut, decisions.md G43).
# Counterfactual ID-swap augmentation: CV macro-F1 (must not drop) + shortcut-probe flip rate on a single run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL=$($PY -c "import yaml;print(yaml.safe_load(open('configs/model_a.yaml'))['model'])")
LR=$($PY -c "import yaml;print(yaml.safe_load(open('configs/model_a.yaml'))['lr'])")
TAG=$(cat configs/WINNER_CV)
[ -f "reports/cv/$TAG-aug-idswap.json" ] || \
  $PY -m src.train --config configs/base.yaml --mode cv --set model="$MODEL" lr="$LR" name="$TAG-aug-idswap" 'augment=[idswap]'
[ -f outputs/model_a_idswap/metrics.json ] || \
  $PY -m src.train --config configs/model_a.yaml --mode single --set name=model_a_idswap 'augment=[idswap]'
$PY -m src.shortcut_probe --run model_a_idswap
