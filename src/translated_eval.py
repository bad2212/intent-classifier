"""Stretch S6: translated-test stress test.

The real non-English test slice is tiny (11 rows), so every ENGLISH test row is machine-translated (MarianMT,
src/augment.py) into es/fr/de/zh, keeping its label, and the trained model is evaluated per language on the SAME
underlying messages. Reported separately from the real non-English slice (decisions.md G30): translations are
cleaner than real multilingual chat and inherit English phrasing.
Usage: python -m src.translated_eval --run model_a
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.augment import TRANSLATIONS
from src.common import REPORTS, load_with_split
from src.metrics import core_metrics
from src.predict import IntentRouter
from src.train import OUTPUTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="model_a")
    a = ap.parse_args()
    df = load_with_split()
    test_en = df[(df.split == "test") & (df.lang == "en")]
    tr = pd.read_csv(TRANSLATIONS)
    tr = tr[tr.src_id.isin(test_en.id)].merge(test_en[["id", "label"]], left_on="src_id", right_on="id")
    router = IntentRouter(str(OUTPUTS / a.run / "model"))

    def score(texts, labels):
        preds = [r["top_intent"] for r in router.predict_batch(list(texts), reject=False)]
        rej = [r["label"] == "unknown" for r in router.predict_batch(list(texts))]
        return {**core_metrics(list(labels), preds), "reject_rate": sum(rej) / len(rej)}

    res = {"english_original": score(test_en.text, test_en.label)}
    for lang, g in tr.groupby("lang"):
        res[f"translated_{lang}"] = score(g.text, g.label)
    out = REPORTS / "trackA" / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "translated_test.json").write_text(json.dumps(res, indent=2, default=float))
    print(pd.DataFrame(res).T[["n", "macro_f1", "accuracy", "reject_rate"]].round(3).to_string())


if __name__ == "__main__":
    main()
