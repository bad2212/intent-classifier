"""Stretch S4 (active learning): simulated uncertainty sampling vs random sampling + learning curve.

For each CV fold k (pinned folds over train+val): pool = the other folds, eval = fold k (never queried).
Start from a stratified 10% of the pool; grow the labelled set to {20, 35, 50, 75, 100}% either by
  * random sampling, or
  * least-confidence sampling: query the pool rows where the current model's max-prob is lowest.
Each point is a fresh fine-tune (fixed recipe, fixed epochs, no early stopping) evaluated on fold k.
The 100% point also answers "would more labelled data help?" (is the curve still rising?).

Backbone: MiniLM-L12 (fastest; the question is about data efficiency, not the absolute ceiling).
Usage: python -m src.active_learning [--folds 0 1 2]
"""
from __future__ import annotations

import argparse
import dataclasses
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding

from src.common import FOLDS_CSV, REPORTS, SEED, label_maps, load_with_split
from src.metrics import core_metrics
from src.train import OUTPUTS, WeightedTrainer, class_weight_tensor, free, new_model, seed_all, to_dataset, training_args

CFG = {"model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "lr": 5e-5, "epochs": 12,
       "batch_size": 16, "warmup_ratio": 0.1, "weight_decay": 0.01, "max_length": 64, "seed": SEED,
       "name": "active-learning", "cpu": False}
FRACTIONS = [0.10, 0.20, 0.35, 0.50, 0.75, 1.00]


def fit_predict(labelled: pd.DataFrame, others: dict[str, pd.DataFrame], tok, label2id, id2label) -> dict:
    model = new_model(CFG, label2id, id2label)
    args = dataclasses.replace(training_args(CFG, OUTPUTS / "al_tmp", single=False, report_to=[]), eval_strategy="no")
    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=to_dataset(labelled, tok, CFG["max_length"], label2id), processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        class_weights=class_weight_tensor(labelled.label, label2id) if labelled.label.nunique() == len(label2id) else None)
    trainer.train()
    out = {k: trainer.predict(to_dataset(d, tok, CFG["max_length"], label2id)).predictions for k, d in others.items() if len(d)}
    del trainer, model
    free()
    return out


PARTIAL = REPORTS / "active_learning" / "partial.csv"


def run_fold(tv: pd.DataFrame, k: int, tok, label2id, id2label, done: set) -> list[dict]:
    """Resumable at (fold, strategy) granularity: finished curves are kept in partial.csv and skipped."""
    pool, ev = tv[tv.fold != k], tv[tv.fold == k]
    start, _ = train_test_split(pool, train_size=FRACTIONS[0], stratify=pool.label, random_state=SEED)
    rows = []
    for strategy in ("random", "least_confidence"):
        if (k, strategy) in done:
            print(f"[al] fold {k} {strategy}: already done, skipping")
            continue
        curve = []
        labelled = start.copy()
        rng = np.random.default_rng(SEED + k)
        for frac in FRACTIONS:
            target = int(round(frac * len(pool)))
            if len(labelled) < target:
                rest = pool[~pool.id.isin(labelled.id)]
                n = target - len(labelled)
                if strategy == "random" or len(rest) <= n:
                    pick = rest.sample(n=min(n, len(rest)), random_state=int(rng.integers(1e9)))
                else:  # score the remaining pool with the model trained on the current labelled set
                    z = fit_predict(labelled, {"rest": rest}, tok, label2id, id2label)["rest"]
                    pick = rest.iloc[np.argsort(softmax(z, axis=1).max(1))[:n]]
                labelled = pd.concat([labelled, pick])
            z = fit_predict(labelled, {"eval": ev}, tok, label2id, id2label)["eval"]
            m = core_metrics(ev.label.map(label2id).values, z.argmax(1))
            curve.append({"fold": k, "strategy": strategy, "fraction": frac, "n_labelled": len(labelled), **m})
            print(f"[al] fold {k} {strategy:16s} n={len(labelled):3d} macro-F1 {m['macro_f1']:.3f}")
        PARTIAL.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(curve).to_csv(PARTIAL, mode="a", header=not PARTIAL.exists(), index=False)
        rows += curve
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    seed_all(SEED)
    df = load_with_split()
    tv = df[df.split != "test"].merge(pd.read_csv(FOLDS_CSV), on="id", validate="one_to_one")
    label2id, id2label = label_maps(tv.label)
    tok = AutoTokenizer.from_pretrained(CFG["model"])
    prev = pd.read_csv(PARTIAL) if PARTIAL.exists() else pd.DataFrame(columns=["fold", "strategy"])
    done = set(zip(prev.fold, prev.strategy))
    for k in a.folds:
        run_fold(tv, k, tok, label2id, id2label, done)
    res = pd.read_csv(PARTIAL)
    res = res[res.fold.isin(a.folds)]
    out = REPORTS / "active_learning"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "curves.csv", index=False)
    agg = res.groupby(["strategy", "fraction"]).agg(n=("n_labelled", "mean"), f1=("macro_f1", "mean"),
                                                    f1_std=("macro_f1", "std")).reset_index()
    (out / "summary.json").write_text(json.dumps(agg.to_dict(orient="records"), indent=2, default=float))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for s, g in agg.groupby("strategy"):
        ax.errorbar(g.n, g.f1, yerr=g.f1_std, marker="o", capsize=3, label=s.replace("_", " "))
    ax.set_xlabel("labelled training rows"); ax.set_ylabel("macro-F1 on held-out fold")
    ax.set_title(f"Learning curve / active learning (MiniLM, folds {a.folds})"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "learning_curve.png", dpi=150); plt.close(fig)
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
