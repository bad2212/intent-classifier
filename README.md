# Multilingual intent router with open-set rejection

A fine-tuned multilingual encoder that routes short logistics-chat messages to one of 12 intents and answers
`unknown` when it is not confident (temperature-calibrated confidence threshold stored in the model config).

- **Model:** https://huggingface.co/Badalt/intent-classifier-mpnet
- **Experiment tracking:** W&B report (view-only link shared separately) · project `badalthakur2212-iisc/intent-classifier`

## Results at a glance

| | Value |
|---|---|
| Backbone | `paraphrase-multilingual-mpnet-base-v2`, lr 5e-5 (chosen by 5-fold CV among 3 encoders × 2 learning rates) |
| 5-fold CV macro-F1 (train+val) | 0.920 ± 0.028 (TF-IDF baseline 0.728) |
| Held-out test macro-F1 | 0.885 (95% bootstrap CI 0.79–0.95), accuracy 0.896 |
| Calibration | ECE 0.33 → 0.05 after temperature scaling |
| Unknown-intent detection (held-out classes) | AUROC 0.79 (max-softmax) → 0.82–0.85 (Mahalanobis / energy) |
| CPU latency | ~25 ms per message (batch 1) |

## Reproduce

The dataset is **not** included. Place it at `data/raw/dataset.csv` (columns `id,text,label`), then:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # exact versions: requirements-lock.txt
hf auth login && wandb login               # only needed for pushing / online tracking
./scripts/run_all.sh                       # EDA -> split -> baseline -> bake-off -> Model A/B -> eval -> report
python -m pytest -q tests                  # unit checks (losses, splits, thresholds, AUROC orientation, augmentation)
```

`src/split.py` regenerates the pinned split and CV folds deterministically (seed 42; near-duplicate paraphrases are
kept in the same split). Apple-silicon MPS is used automatically; add `--set cpu=true` for CPU runs.

## Use the model

```python
from src.predict import IntentRouter
router = IntentRouter("Badalt/intent-classifier-mpnet")
router.predict("whats the eta on LD-55501 pls")    # -> (label, calibrated confidence); "unknown" below threshold
```

Serving stub: `MODEL_ID=Badalt/intent-classifier-mpnet uvicorn src.serve:app --port 8000` (or `docker build -t intent-router .`).

## Layout

| Path | What |
|---|---|
| `src/eda.py` | EDA: language ID, lengths, noise, near-duplicate / cross-lingual leakage check |
| `src/split.py` | Leakage-aware stratified 70/15/15 split + 5 group-aware CV folds |
| `src/baseline.py` | TF-IDF (char + word) + logistic regression baseline |
| `src/train.py` | Fine-tuning: `single` (train → val-selected → calibrate → test once) and `cv` modes; W&B logging; optional hierarchical loss, Outlier Exposure, augmentation |
| `src/select_winner.py` | Applies the pre-declared bake-off rule and generates the run configs |
| `src/openset.py` | Unknown-intent evaluation: MSP / temperature / energy / Mahalanobis / kNN scorers, AUROC, curves |
| `src/analysis.py`, `src/shortcut_probe.py` | Calibration + error analysis; entity-ID shortcut counterfactual probe |
| `src/predict.py`, `src/serve.py`, `Dockerfile` | Inference with the reject rule; FastAPI stub |
| `src/monitor.py` | Drift monitor (two-sample tests) + window-size power study |
| `src/augment.py`, `src/oe_data.py`, `src/active_learning.py` | Augmentation, Outlier-Exposure data screening, active-learning simulation |
| `src/llm_baseline.py` | Zero/few-shot LLM baseline (Gemini) on the same test rows |
| `scripts/` | `run_all.sh`, `core.sh`, `bakeoff.sh`, `stretch3.sh`, `push_to_hub.py`, `make_wandb_report.py`, `check_no_data_leak.py` |

## Data policy

This repo contains **no dataset content**: no messages, labels, row ids, per-row predictions or splits.
`scripts/check_no_data_leak.py` runs as a git pre-push hook and blocks any push that contains dataset text,
per-row data or task identifiers.
