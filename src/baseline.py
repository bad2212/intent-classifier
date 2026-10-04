"""Non-fine-tuned floor: TF-IDF (char + word n-grams) + logistic regression.

Char n-grams are typo-robust and partly language-agnostic; this is the number the
transformer has to clearly beat. C is chosen on val (small fixed grid), test touched once.
Usage: python -m src.baseline          # pinned split: val-selected C, test once
       python -m src.baseline --cv     # same 5 pinned folds as the transformer bake-off (apples-to-apples)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from src.common import FOLDS_CSV, REPORTS, SEED, load_with_split, set_seed
from src.metrics import core_metrics, full_report, plot_confusion

C_GRID = [0.3, 1, 3, 10, 30]


def build(C: float):
    feats = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
    )
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)
    return make_pipeline(feats, clf)


def run_cv(C: float) -> None:
    """Pooled out-of-fold metrics on the pinned train+val folds, with the C chosen on val."""
    df = load_with_split()
    tv = df[df.split != "test"].merge(pd.read_csv(FOLDS_CSV), on="id", validate="one_to_one")
    preds = pd.Series(index=tv.index, dtype=object)
    fold_f1 = []
    for k in sorted(tv.fold.unique()):
        tr, ho = tv[tv.fold != k], tv[tv.fold == k]
        preds.loc[ho.index] = build(C).fit(tr.text, tr.label).predict(ho.text)
        fold_f1.append(core_metrics(ho.label, preds.loc[ho.index])["macro_f1"])
    rep = full_report(tv.label.values, preds.values, sorted(tv.label.unique()), tv.lang.values)
    best = {"epoch": None, "pooled_macro_f1": rep["macro_f1"], "pooled_accuracy": rep["accuracy"],
            "fold_macro_f1_mean": float(np.mean(fold_f1)), "fold_macro_f1_std": float(np.std(fold_f1)),
            "non_english_accuracy": rep["slices"]["non_english"]["accuracy"],
            "english_accuracy": rep["slices"]["english"]["accuracy"], "coarse_accuracy": rep["coarse"]["accuracy"]}
    out = REPORTS / "cv"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cv-tfidf-baseline.json").write_text(json.dumps(
        {"config": {"model": "tfidf_char2-5+word1-2_logreg", "C": C}, "best": best,
         "per_class_at_best": rep["per_class"]}, indent=2, default=float))
    pd.DataFrame({"id": tv.id, "label": tv.label, "fold": tv.fold, "lang": tv.lang, "pred": preds.values}
                 ).to_csv(out / "cv-tfidf-baseline_oof.csv", index=False)
    print(f"[cv] tfidf baseline (C={C}): pooled OOF macro-F1 {best['pooled_macro_f1']:.3f} "
          f"(fold mean {best['fold_macro_f1_mean']:.3f} ± {best['fold_macro_f1_std']:.3f}) · "
          f"non-EN acc {best['non_english_accuracy']:.3f} · EN acc {best['english_accuracy']:.3f}")


def main() -> None:
    set_seed()
    df = load_with_split()
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    labels = sorted(df.label.unique())

    val_scores = {}
    for C in C_GRID:
        model = build(C).fit(tr.text, tr.label)
        val_scores[C] = core_metrics(va.label, model.predict(va.text))["macro_f1"]
    best_C = max(val_scores, key=val_scores.get)

    model = build(best_C).fit(tr.text, tr.label)
    val = core_metrics(va.label, model.predict(va.text))
    pred = model.predict(te.text)
    test = full_report(te.label, pred, labels, te.lang)

    out = REPORTS / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    plot_confusion(te.label, pred, labels, out / "confusion_test.png", "TF-IDF + LR baseline — test")
    result = {"model": "tfidf_char2-5+word1-2_logreg", "val_grid": val_scores, "best_C": best_C,
              "val": val, "test": test}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float))

    print(f"val macro-F1 by C: { {k: round(v, 3) for k, v in val_scores.items()} } -> C={best_C}")
    print(f"VAL  macro-F1 {val['macro_f1']:.3f} acc {val['accuracy']:.3f}")
    print(f"TEST macro-F1 {test['macro_f1']:.3f} (95% CI {test['macro_f1_ci'][0]:.3f}–{test['macro_f1_ci'][1]:.3f}) "
          f"acc {test['accuracy']:.3f}")
    for k, v in test["slices"].items():
        print(f"  {k:12s} n={v['n']:3d} macro-F1 {v['macro_f1']:.3f} acc {v['accuracy']:.3f}")
    print(f"  coarse (siblings merged) acc {test['coarse']['accuracy']:.3f}")
    for l, m in test["per_class"].items():
        print(f"  {l:38s} P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f} (n={int(m['support'])})")


if __name__ == "__main__":
    if "--cv" in sys.argv:
        prev = json.loads((REPORTS / "baseline" / "metrics.json").read_text())
        run_cv(prev["best_C"])  # C chosen on val by the pinned-split run; no re-tuning on folds
    else:
        main()
