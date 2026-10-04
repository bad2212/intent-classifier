#!/usr/bin/env bash
# Core deliverables after the CV bake-off (plan.md §4.3–4.4). One MPS job at a time (decisions.md G39).
# Resumable: a step whose outputs exist is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

$PY -m src.select_winner                                   # pre-declared rule -> configs/model_{a,b,b_near}.yaml
WINNER_CV=$(cat configs/WINNER_CV)

single() {  # single <config> -> outputs/<name>/
  local name; name=$($PY -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['name'])" "$1")
  if [ -f "outputs/$name/metrics.json" ]; then echo "[skip] $name"; return; fi
  $PY -m src.train --config "$1" --mode single
}

echo "== Model A (12 classes) -> Track A";               single configs/model_a.yaml
$PY -m src.analysis --run model_a --cv-name "$WINNER_CV"
echo "== Model B (yard + documents held out) -> Track B"; single configs/model_b.yaml
$PY -m src.openset --run model_b --wandb
echo "== Model B-near (disruptions held out) -> Track B"; single configs/model_b_near.yaml
$PY -m src.openset --run model_b_near --wandb
echo "== drift monitor demo";                             $PY -m src.monitor --run model_b
echo "== translated-test stress test";                    $PY -m src.translated_eval --run model_a
