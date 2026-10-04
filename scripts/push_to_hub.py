"""Push a trained single-mode run to the Hugging Face Hub with a generated model card, then verify that
the Hub copy loads with AutoModelForSequenceClassification in a FRESH process and reproduces the local
test predictions exactly.

Usage: python scripts/push_to_hub.py --run model_a --repo Badalt/intent-classifier-mpnet \
           [--trackb-run model_b] [--cv-name cv-mpnet-lr5e-5] [--wandb-url URL] [--private]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import REPORTS  # noqa: E402
from src.train import OUTPUTS  # noqa: E402

LICENSES = {"sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "apache-2.0",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "apache-2.0",
            "FacebookAI/xlm-roberta-base": "mit"}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def model_card(run: str, repo: str, trackb_run: str | None, cv_name: str | None, wandb_url: str | None) -> str:
    m = json.loads((OUTPUTS / run / "metrics.json").read_text())
    cfg, cal, te = m["config"], m["calibration"], m["test"]
    labels = list(te["per_class"])
    cv = json.loads((REPORTS / "cv" / f"{cv_name}.json").read_text())["best"] if cv_name else None
    tb = json.loads((REPORTS / "trackB" / trackb_run / "metrics.json").read_text()) if trackb_run else None

    per_class = "\n".join(f"| `{l}` | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {int(v['support'])} |"
                          for l, v in te["per_class"].items())
    lines = [
        "---",
        f"license: {LICENSES.get(cfg['model'], 'other')}",
        f"base_model: {cfg['model']}",
        "pipeline_tag: text-classification",
        "language: [en, es, fr, de, zh, nl, it, ar]",
        "tags: [intent-classification, logistics, multilingual, open-set, text-classification]",
        "model-index:",
        f"- name: {repo.split('/')[-1]}",
        "  results:",
        "  - task: {type: text-classification, name: Intent classification}",
        "    dataset: {name: synthetic logistics intents (500 rows, private), type: private}",
        "    metrics:",
        f"    - {{type: f1, name: macro-F1 (test), value: {te['macro_f1']:.4f}}}",
        f"    - {{type: accuracy, name: accuracy (test), value: {te['accuracy']:.4f}}}",
        "---",
        "",
        f"# {repo.split('/')[-1]}",
        "",
        "Routes a short user message from a logistics / supply-chain chat assistant to one of **12 intents**, and "
        "abstains with **`unknown`** when it is not confident (calibrated reject rule stored in `config.json`).",
        f"Fine-tuned from [`{cfg['model']}`](https://huggingface.co/{cfg['model']}) as a standard "
        "`AutoModelForSequenceClassification` (no custom code).",
        "",
        "## Usage",
        "",
        "```python",
        "import torch",
        "from transformers import AutoModelForSequenceClassification, AutoTokenizer",
        "",
        f'repo = "{repo}"',
        "tok = AutoTokenizer.from_pretrained(repo)",
        "model = AutoModelForSequenceClassification.from_pretrained(repo).eval()",
        "",
        'texts = ["whats the eta on LD-55501 pls", "¿Cuántos remolques hay en el patio?"]',
        "with torch.no_grad():",
        '    logits = model(**tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")).logits',
        "probs = torch.softmax(logits / model.config.calibration_temperature, dim=-1)",
        "conf, idx = probs.max(-1)",
        "for c, i in zip(conf, idx):",
        '    label = "unknown" if c < model.config.unknown_threshold else model.config.id2label[int(i)]',
        "    print(label, round(float(c), 3))",
        "```",
        "",
        f"Reject rule: `unknown` if `max softmax(logits / {cal['calibration_temperature']:.4f}) < "
        f"{cal['unknown_threshold']:.4f}`. The threshold keeps {pct(cal['retention_target'])} of known-intent "
        "validation messages; the temperature was fitted on the same validation set.",
        "",
        "## Labels",
        "",
        ", ".join(f"`{l}`" for l in labels) + " (+ `unknown` via the reject rule).",
        "",
        "## Training data",
        "",
        "500 labelled synthetic chat messages (12 intents, 30–55 per class; ~13% non-English: ES, FR, DE, ZH, NL, AR, IT; "
        "noisy, colloquial). The dataset is confidential and **not** distributed with this model. "
        "Split: stratified 70/15/15 (347 / 76 / 77) with near-duplicate paraphrases kept in the same split; seed 42.",
        "",
        "## Training procedure",
        "",
        f"lr {cfg['lr']}, batch {cfg['batch_size']}, ≤{cfg['epochs']} epochs with early stopping on validation macro-F1 "
        f"(best epoch {m['best_epoch']:.0f}), linear decay, warmup {cfg['warmup_ratio']}, weight decay {cfg['weight_decay']}, "
        f"class-weighted cross-entropy, max_length {cfg['max_length']}, seed {cfg['seed']}. "
        "Backbone and learning rate chosen by 5-fold cross-validation among 3 multilingual encoders × 2 learning rates.",
        "",
        "## Evaluation",
        "",
        f"**Held-out test set (n={te['n']}, never used for any decision):** macro-F1 **{te['macro_f1']:.3f}** "
        f"(95% bootstrap CI {te['macro_f1_ci'][0]:.3f}–{te['macro_f1_ci'][1]:.3f}), accuracy {te['accuracy']:.3f}; "
        f"English {te['slices']['english']['accuracy']:.3f} vs non-English {te['slices']['non_english']['accuracy']:.3f} "
        f"accuracy (n={te['slices']['non_english']['n']}).",
    ]
    if cv:
        lines += ["", f"**5-fold CV on train+val (n=423, out-of-fold):** macro-F1 {cv['pooled_macro_f1']:.3f} "
                      f"(fold mean {cv['fold_macro_f1_mean']:.3f} ± {cv['fold_macro_f1_std']:.3f}); non-English accuracy "
                      f"{cv['non_english_accuracy']:.3f}."]
    lines += ["", "| intent | precision | recall | F1 | n |", "|---|---|---|---|---|", per_class]
    if tb:
        msp = next(r for r in tb["scorers"] if r["scorer"] == "MSP (baseline)")
        best = max(tb["scorers"], key=lambda r: r["AUROC"])
        lines += ["", "**Open-set (unknown intents).** A twin model trained without "
                  f"{' and '.join('`' + c + '`' for c in tb['holdout_classes'])} was tested on known messages plus all "
                  f"messages of those unseen intents: max-softmax reject rule AUROC {msp['AUROC']:.3f}, rejection recall "
                  f"{pct(msp['rejection_recall_strict'])} at {pct(msp['test_retention'])} retention; best scorer "
                  f"({best['scorer']}) AUROC {best['AUROC']:.3f}."]
    lines += [
        "",
        "## Intended use & limitations",
        "",
        "- First-step router for a logistics assistant; low-confidence messages should go to a clarification / fallback path.",
        "- Trained on 500 synthetic rows: expect lower accuracy on real traffic; monitor confidence, unknown-rate and "
        "language mix for drift and re-calibrate the threshold on fresh labelled data.",
        "- The three `shipment_information.*` intents are semantically close and account for a large share of errors.",
        "- Non-English evaluation rests on few examples; languages not seen in training (e.g. Arabic, Italian) are untested beyond anecdotes.",
        "- `other` / `chitchat` are trained catch-alls; genuinely new intents are handled only by the confidence threshold.",
    ]
    if wandb_url:
        lines += ["", f"Training curves and metrics: {wandb_url}"]
    return "\n".join(lines) + "\n"


VERIFY = """
import json, sys, numpy as np, torch
sys.path.insert(0, {root!r})
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.common import load_with_split
repo = {repo!r}
tok = AutoTokenizer.from_pretrained(repo); model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
local = np.load({feats!r}, allow_pickle=True)
df = load_with_split().set_index("id").loc[local["ids"]]
with torch.no_grad():
    z = model(**tok(df.text.tolist(), padding=True, truncation=True, max_length=64, return_tensors="pt")).logits.numpy()
same = bool((z.argmax(1) == local["logits"].argmax(1)).all())
print(json.dumps({{"hub_argmax_identical": same, "max_abs_logit_diff": float(np.abs(z - local["logits"]).max()),
                  "n_labels": len(model.config.id2label), "has_reject_rule": hasattr(model.config, "unknown_threshold")}}))
sys.exit(0 if same else 1)
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--trackb-run")
    ap.add_argument("--cv-name")
    ap.add_argument("--wandb-url")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi

    model_dir = OUTPUTS / a.run / "model"
    (model_dir / "README.md").write_text(model_card(a.run, a.repo, a.trackb_run, a.cv_name, a.wandb_url))
    api = HfApi()
    api.create_repo(a.repo, private=a.private, exist_ok=True)
    api.upload_folder(repo_id=a.repo, folder_path=str(model_dir), commit_message=f"Upload {a.run}")
    print(f"[hub] pushed https://huggingface.co/{a.repo}")

    code = VERIFY.format(root=str(ROOT), repo=a.repo, feats=str(OUTPUTS / a.run / "features_test.npz"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print("[verify, fresh process]", r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-800:])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
