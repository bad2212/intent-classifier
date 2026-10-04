#!/usr/bin/env bash
# End-to-end reproduction. Every step is deterministic given the pinned split/folds (MPS adds small run-to-run noise;
# add `--set cpu=true` to train commands for bit-exact CPU runs). Each step is skippable: outputs are re-used if present.
#   ./scripts/run_all.sh            core pipeline (Track A + Track B + analysis + report)
#   STRETCH=1 ./scripts/run_all.sh  also the stretch experiments (OE, hierarchical, augmentation, AL, seeds, LLM)
#   PUSH=1 ./scripts/run_all.sh     also push Model A to the Hub (needs `hf auth login`)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "== 1. EDA, language ID, leakage check";      $PY -m src.eda
echo "== 2. pinned split + CV folds";              $PY -m src.split
echo "== 3. TF-IDF baseline (split + CV)";         $PY -m src.baseline && $PY -m src.baseline --cv
echo "== 4. backbone x LR bake-off (5-fold CV)";   ./scripts/bakeoff.sh
echo "== 5. unit checks";                          $PY -m pytest -q tests

echo "== 6-9. Model A / B / B-near, analysis, monitor"; ./scripts/core.sh

if [[ "${STRETCH:-0}" == "1" ]]; then
  echo "== S. stretch experiments";                ./scripts/stretch.sh
fi
if [[ "${PUSH:-0}" == "1" ]]; then
  echo "== 10. push Model A to the Hub";           $PY scripts/push_to_hub.py --run model_a --repo "$(cat configs/HUB_REPO)" \
                                                     --trackb-run model_b --cv-name "$(cat configs/WINNER_CV)"
fi
echo "== 11. report";                              $PY -m src.report
