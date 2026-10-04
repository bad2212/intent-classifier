"""Error-analysis probe: does the model route on entity-ID PREFIXES (APT-/PO-/DOC-/TR-/LD-) instead of intent?

Motivation (OOF error review): support tickets that mention an ID are often routed to the class matching the ID type
("PO-71045 order lines are duplicated" -> orders, "DOC-9021 stuck in processing" -> document_processing).
Counterfactual test: for every message containing an ID, swap ONLY the prefix to each other ID type (digits and all
other words unchanged) and measure (a) how often the prediction changes and (b) how often it changes TO the class
associated with the new prefix. A model that understood intent should be ~invariant to the ID type.

Inference only (CPU by default so it never shares the MPS GPU with a training job, decisions.md G39).
Usage: python -m src.shortcut_probe --run model_a
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from src.common import ID_RE, PREFIX_CLASS, REPORTS, load_with_split
from src.predict import IntentRouter
from src.train import OUTPUTS



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="model_a")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    df = load_with_split()
    df = df[df.text.map(lambda t: ID_RE.search(t) is not None)].copy()  # same regex as the substitution
    router = IntentRouter(str(OUTPUTS / a.run / "model"), device=a.device)
    base = [r["top_intent"] for r in router.predict_batch(df.text.tolist(), reject=False)]
    df["pred"] = base

    rows = []
    for (idx, r), b in zip(df.iterrows(), base):
        orig = ID_RE.search(r.text).group(1)
        for new in PREFIX_CLASS:
            if new == orig:
                continue
            text = ID_RE.sub(lambda m: f"{new}-{m.group(2)}", r.text)
            rows.append({"id": r.id, "split": r.split, "label": r.label, "orig_prefix": orig, "new_prefix": new,
                         "base_pred": b, "text": text})
    cf = pd.DataFrame(rows)
    cf["cf_pred"] = [x["top_intent"] for x in router.predict_batch(cf.text.tolist(), reject=False)]
    cf["changed"] = cf.cf_pred != cf.base_pred
    cf["to_prefix_class"] = cf.cf_pred == cf.new_prefix.map(PREFIX_CLASS)
    cf["base_correct"] = cf.base_pred == cf.label

    by_prefix = cf.groupby("new_prefix").agg(flip_rate=("changed", "mean"), to_prefix_class=("to_prefix_class", "mean"),
                                             n=("changed", "size")).round(3)
    df["prefix_class"] = df.text.map(lambda t: PREFIX_CLASS[ID_RE.search(t).group(1)])
    err = df[df.pred != df.label]
    res = {
        "messages_with_id": int(len(df)), "share_of_dataset": round(len(df) / 500, 3),
        "label_equals_prefix_class": float((df.label == df.prefix_class).mean()),
        "accuracy_on_id_messages": float((df.pred == df.label).mean()),
        "errors_on_id_messages": int(len(err)),
        "errors_predicting_prefix_class": int((err.pred == err.prefix_class).sum()),
        "counterfactual_flip_rate": float(cf.changed.mean()),
        "counterfactual_flip_to_new_prefix_class": float(cf.to_prefix_class.mean()),
        "flip_rate_when_base_correct": float(cf[cf.base_correct].changed.mean()),
        "by_new_prefix": by_prefix.to_dict(orient="index"),
        "note": "model_a saw train rows; flip rates are reported for all splits and for test only",
        "test_only_flip_rate": float(cf[cf.split == "test"].changed.mean()) if (cf.split == "test").any() else None,
    }
    out = REPORTS / "trackA" / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "shortcut_probe.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "by_new_prefix"}, indent=1, default=float))
    print(by_prefix.to_string())


if __name__ == "__main__":
    main()
