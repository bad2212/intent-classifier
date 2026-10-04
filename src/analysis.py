"""Track A deep-dive for a trained single-mode run: calibration (S7) + error analysis.

  * Reliability diagrams + ECE / NLL before and after temperature scaling (val and test)
  * Confusion pairs and the shipment_information.* 3x3 sub-matrix, on test (official, n=77) AND on the
    CV out-of-fold predictions of the same recipe (n=423, ~5x more evidence for confusion patterns)
  * Per-language accuracy; confidence of correct vs wrong predictions
  * Misclassified examples (contain raw text -> written to a git-ignored CSV, only for the local report)

Usage: python -m src.analysis --run model_a --cv-name cv-mpnet-lr5e-5
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax

from src.common import REPORTS, load_with_split
from src.train import OUTPUTS

SIBLINGS = ["shipment_information.analytics", "shipment_information.disruptions", "shipment_information.realtime_query"]


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    return float(-log_softmax(logits / T, axis=1)[np.arange(len(y)), y].mean())


def reliability(ax, probs, y, title, n_bins: int = 10):
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            accs.append((pred[m] == y[m]).mean()); confs.append(conf[m].mean()); ns.append(m.sum())
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1)
    ax.scatter(confs, accs, s=[20 + 3 * n for n in ns], alpha=0.7)
    ax.plot(confs, accs, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE {ece(probs, y):.3f}", fontsize=9)


def confusion_pairs(y_true, y_pred, top: int = 10) -> list[dict]:
    wrong = pd.DataFrame({"true": y_true, "pred": y_pred}).query("true != pred")
    return wrong.value_counts().head(top).reset_index(name="n").to_dict(orient="records")


def sibling_matrix(y_true, y_pred) -> pd.DataFrame:
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df = df[df.true.isin(SIBLINGS)]
    df["pred"] = df.pred.where(df.pred.isin(SIBLINGS), "(non-shipment)")
    short = lambda s: s.replace("shipment_information.", "")
    return pd.crosstab(df.true.map(short), df.pred.map(short))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cv-name", required=True, help="reports/cv/<name>_oof.csv of the same recipe")
    a = ap.parse_args()

    run_dir = OUTPUTS / a.run
    meta = json.loads((run_dir / "metrics.json").read_text())
    T = meta["calibration"]["calibration_temperature"]
    from transformers import AutoConfig

    label2id = AutoConfig.from_pretrained(run_dir / "model").label2id
    feats = {k: dict(np.load(run_dir / f"features_{k}.npz", allow_pickle=True)) for k in ("val", "test")}
    out = REPORTS / "trackA" / a.run
    out.mkdir(parents=True, exist_ok=True)

    # --- calibration ---
    calib, fig_axes = {}, plt.subplots(1, 4, figsize=(15, 3.8))
    fig, axes = fig_axes
    for i, split in enumerate(("val", "test")):
        z, y = feats[split]["logits"], np.array([label2id[l] for l in feats[split]["labels"]])
        for j, (t, tag) in enumerate(((1.0, "raw"), (T, f"T={T:.2f}"))):
            p = softmax(z / t, axis=1)
            calib[f"{split}_{'raw' if t == 1.0 else 'scaled'}"] = {"ece": ece(p, y), "nll": nll(z, y, t)}
            reliability(axes[2 * i + j], p, y, f"{split} · {tag}")
    fig.tight_layout(); fig.savefig(out / "reliability.png", dpi=150); plt.close(fig)

    # --- errors on test (official) ---
    df = load_with_split().set_index("id")
    te = feats["test"]
    id2label = {v: k for k, v in label2id.items()}
    pred = np.array([id2label[i] for i in te["logits"].argmax(1)])
    conf = softmax(te["logits"] / T, axis=1).max(1)
    test = pd.DataFrame({"id": te["ids"], "true": te["labels"], "pred": pred, "conf": conf, "lang": te["langs"]})
    test["correct"] = test.true == test.pred
    test.join(df[["text"]], on="id").query("~correct").sort_values("conf", ascending=False) \
        .to_csv(out / "errors_test.csv", index=False)

    # --- errors on CV out-of-fold predictions (same recipe, 423 rows) ---
    oof = pd.read_csv(REPORTS / "cv" / f"{a.cv_name}_oof.csv")
    oof.join(df[["text"]], on="id").query("label != pred").to_csv(out / "errors_oof.csv", index=False)

    lang_acc = lambda d, t, p: d.assign(ok=d[t] == d[p]).groupby("lang").ok.agg(["mean", "size"]).round(3)
    result = {
        "calibration": calib,
        "test": {"confusion_pairs": confusion_pairs(test.true, test.pred),
                 "sibling_matrix": sibling_matrix(test.true, test.pred).to_dict(),
                 "per_language_accuracy": lang_acc(test, "true", "pred").to_dict(orient="index"),
                 "mean_conf_correct": float(test[test.correct].conf.mean()),
                 "mean_conf_wrong": float(test[~test.correct].conf.mean()) if (~test.correct).any() else None},
        "oof": {"n": len(oof), "confusion_pairs": confusion_pairs(oof.label, oof.pred, 15),
                "sibling_matrix": sibling_matrix(oof.label, oof.pred).to_dict(),
                "per_language_accuracy": lang_acc(oof, "label", "pred").to_dict(orient="index")},
    }
    (out / "analysis.json").write_text(json.dumps(result, indent=2, default=float))

    print(f"[calibration] {json.dumps({k: {m: round(v, 3) for m, v in d.items()} for k, d in calib.items()})}")
    print(f"[test] top confusions: {result['test']['confusion_pairs']}")
    print(f"[test] conf correct {result['test']['mean_conf_correct']:.3f} vs wrong {result['test']['mean_conf_wrong']}")
    print("[oof] sibling matrix:\n", sibling_matrix(oof.label, oof.pred).to_string())
    print(f"[oof] top confusions: {result['oof']['confusion_pairs'][:8]}")
    print("[oof] per-language accuracy:\n", lang_acc(oof, "label", "pred").to_string())


if __name__ == "__main__":
    main()
