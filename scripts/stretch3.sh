#!/usr/bin/env bash
# Stretch queue v3 (replaces stretch.sh + stretch2.sh after measuring real run times; decisions.md G44).
# Ordered by value to the report; resumable (finished runs are skipped); one MPS job at a time (G39).
# Heavy translation augmentation runs on folds 0-2 only and is compared PAIRED against the baseline's same folds.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL=$($PY -c "import yaml;print(yaml.safe_load(open('configs/model_a.yaml'))['model'])")
LR=$($PY -c "import yaml;print(yaml.safe_load(open('configs/model_a.yaml'))['lr'])")
TAG=$(cat configs/WINNER_CV)
OE="oe_lambda=0.5 oe_file=data/aux/oe_outliers_screened.csv"   # lambda fixed a priori (Hendrycks et al. 2019)

cv() {  # cv <name> <overrides...>
  local name=$1; shift
  if [ -f "reports/cv/$name.json" ]; then echo "[skip] $name"; return; fi
  $PY -m src.train --config configs/base.yaml --mode cv --set model="$MODEL" lr="$LR" name="$name" "$@"
}
single() {  # single <config> <name> <overrides...>
  local cfg=$1 name=$2; shift 2
  if [ -f "outputs/$name/metrics.json" ]; then echo "[skip] $name"; return; fi
  $PY -m src.train --config "$cfg" --mode single --set name="$name" "$@"
}
trackb() { [ -f "reports/trackB/$1/metrics.json" ] && echo "[skip] trackB $1" || $PY -m src.openset --run "$1" --wandb; }

echo "== S1b Outlier Exposure (Track A guardrail by CV, then Track B far + near)"
cv "$TAG-oe" $OE
single configs/model_b.yaml model_b_oe $OE;           trackb model_b_oe
single configs/model_b_near.yaml model_b_near_oe $OE; trackb model_b_near_oe
echo "== S10 counterfactual ID-swap augmentation (shortcut fix)"
cv "$TAG-aug-idswap" 'augment=[idswap]'
single configs/model_a.yaml model_a_idswap 'augment=[idswap]'
$PY -m src.shortcut_probe --run model_a_idswap
echo "== S3 hierarchical parent loss"
cv "$TAG-hier0.5" hier_lambda=0.5
cv "$TAG-hier1.0" hier_lambda=1.0
echo "== S8 extra seeds of the submitted recipe"
for s in 43 44; do single configs/model_a.yaml "model_a_s$s" seed=$s; done
echo "== S4 active learning (MiniLM, folds 0-2)"
[ -f reports/active_learning/summary.json ] || $PY -m src.active_learning --folds 0 1 2
echo "== S4 augmentation"
cv "$TAG-aug-noise" 'augment=[noise]' aug_copies=1
cv "$TAG-aug-translate-f3" 'augment=[translate]' cv_folds_limit=3
$PY -m src.stretch_summary --decide-aug-both > /tmp/aug_both_decision.txt || true
if grep -q RUN /tmp/aug_both_decision.txt; then cv "$TAG-aug-both-f3" 'augment=[noise,translate]' aug_copies=1 cv_folds_limit=3; fi
echo "== summary"; $PY -m src.stretch_summary
