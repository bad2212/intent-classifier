#!/usr/bin/env bash
# Stretch experiments (plan.md §7, S1–S9). All comparisons are made by 5-fold CV on train+val with the winning
# recipe from the bake-off; test is NOT used to choose anything (decisions.md G38). Resumable: finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MODEL=$($PY -c "import yaml;print(yaml.safe_load(open('configs/model_a.yaml'))['model'])")
LR=$($PY -c "import yaml;print(yaml.safe_load(open('configs/model_a.yaml'))['lr'])")
TAG=$(cat configs/WINNER_CV)          # e.g. cv-mpnet-lr5e-5
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

# S3 hierarchical auxiliary parent loss
cv "$TAG-hier0.5" hier_lambda=0.5
cv "$TAG-hier1.0" hier_lambda=1.0
# S4 augmentation (copies only from each fold's training rows)
cv "$TAG-aug-noise" 'augment=[noise]' aug_copies=1
cv "$TAG-aug-translate" 'augment=[translate]'
cv "$TAG-aug-both" 'augment=[noise,translate]' aug_copies=1
# S1b Outlier Exposure: Track A guardrail by CV, then Track B with OE
cv "$TAG-oe" $OE
single configs/model_b.yaml model_b_oe $OE
$PY -m src.openset --run model_b_oe --wandb
single configs/model_b_near.yaml model_b_near_oe $OE
$PY -m src.openset --run model_b_near_oe --wandb
# S8 multi-seed of the submitted recipe (seed 42 is the declared submission)
for s in 43 44; do single configs/model_a.yaml "model_a_s$s" seed=$s; done
# S6 translated test stress test + S4 active-learning simulation
$PY -m src.translated_eval --run model_a
[ -f reports/active_learning/summary.json ] || $PY -m src.active_learning --folds 0 1 2
# S2 LLM baseline: Gemini free tier (key: GEMINI_API_KEY in .env). Cached per row, so a daily-quota stop resumes.
for m in gemini-3.5-flash-lite; do   # free tier: flash = 20 req/day/model (too few); pro = 0
  for shots in 0 3; do $PY -m src.llm_baseline --model "$m" --mode closed --shots "$shots" || echo "[llm] stopped ($m $shots-shot)"; done
  $PY -m src.llm_baseline --model "$m" --mode open --shots 3 || echo "[llm] stopped ($m open)"
done
