"""Stretch S1b: screen the hand-written auxiliary outlier set used for Outlier Exposure (OE).

Why hand-written (decisions.md G32): in this dataset, generic off-domain text ("what is the capital of Peru",
"write a cover letter") IS the labelled `other` class, so public out-of-scope sets (e.g. CLINC150) would train
the opposite of `other`. The realistic unknown is a NEW LOGISTICS CAPABILITY (rate quotes, emissions, claims,
inventory, ...). data/aux/oe_outliers.csv holds 107 such requests in 15 topics, written to avoid all 12 intents.

Screening (all automatic, applied before any training):
  1. keyword screen: drop rows mentioning concepts of the Track B held-out classes (yard / documents) or of the
     near-OOD class (disruptions) -> OE must not leak information about the classes Track B tests on
  2. similarity screen: drop rows whose nearest dataset row (pretrained multilingual embedding cosine) is >= 0.70
     or whose nearest class centroid is any of the held-out / near-OOD classes
Output: data/aux/oe_outliers_screened.csv + reports/oe_screen.json
Usage: python -m src.oe_data
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.common import REPORTS, ROOT, load_raw

AUX = ROOT / "data" / "aux" / "oe_outliers.csv"
OUT = ROOT / "data" / "aux" / "oe_outliers_screened.csv"
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PROTECTED = ["yard_management", "document_processing", "shipment_information.disruptions"]
KEYWORDS = r"trailer|yard|dock|document|bill of lading|\bbol\b|\bpod\b|packing list|extract|scan|pdf|weather|strike|" \
           r"congestion|disrupt|delay|storm|remolque|patio|documento|remorque|cour\b|Anhänger|Dokument"
NN_MAX_SIM = 0.70


def main() -> None:
    aux = pd.read_csv(AUX)
    data = load_raw()
    aux["kw_hit"] = aux.text.str.contains(KEYWORDS, case=False, regex=True)

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(EMB_MODEL, device="cpu")
    e_aux = enc.encode(aux.text.tolist(), normalize_embeddings=True)
    e_dat = enc.encode(data.text.tolist(), normalize_embeddings=True)
    sims = e_aux @ e_dat.T
    aux["nn_sim"] = sims.max(1)
    aux["nn_label"] = data.label.values[sims.argmax(1)]
    labels = sorted(data.label.unique())
    cents = np.stack([e_dat[data.label.values == l].mean(0) for l in labels])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    aux["nearest_class"] = np.array(labels)[(e_aux @ cents.T).argmax(1)]

    aux["drop_reason"] = np.select(
        [aux.kw_hit, aux.nn_sim >= NN_MAX_SIM, aux.nearest_class.isin(PROTECTED)],
        ["keyword (protected class concept)", f"near-duplicate of a dataset row (cos>={NN_MAX_SIM})",
         "nearest class centroid is a protected class"], default="")
    kept = aux[aux.drop_reason == ""]
    kept[["text", "topic", "lang"]].to_csv(OUT, index=False)
    rep = {"n_written": len(aux), "n_kept": len(kept), "dropped": aux[aux.drop_reason != ""]
           [["text", "drop_reason", "nn_sim", "nn_label", "nearest_class"]].round(3).to_dict(orient="records"),
           "kept_nearest_class_counts": kept.nearest_class.value_counts().to_dict(),
           "kept_lang_counts": kept.lang.value_counts().to_dict(), "kept_topics": kept.topic.nunique()}
    (REPORTS / "oe_screen.json").write_text(json.dumps(rep, indent=2, default=float))
    print(f"written {len(aux)} -> kept {len(kept)} ({kept.topic.nunique()} topics, langs {rep['kept_lang_counts']})")
    print(pd.DataFrame(rep["dropped"]).to_string(index=False))
    print("kept rows' nearest class:", rep["kept_nearest_class_counts"])
    print("kept nn_sim: max %.3f median %.3f" % (kept.nn_sim.max(), kept.nn_sim.median()))


if __name__ == "__main__":
    main()
