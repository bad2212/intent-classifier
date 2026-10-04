"""Metric helpers shared by the baseline, Track A and Track B evaluation."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.common import SEED, parent_of


def core_metrics(y_true, y_pred) -> dict:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "n": len(y_true),
    }


def per_class(y_true, y_pred, labels) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return pd.DataFrame({l: rep[l] for l in labels}).T.rename(columns={"f1-score": "f1"})


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, alpha: float = 0.05, seed: int = SEED) -> dict:
    """Percentile CI of macro-F1 / accuracy by resampling test rows with replacement."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s, accs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
        accs.append((y_true[idx] == y_pred[idx]).mean())
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"macro_f1_ci": np.percentile(f1s, q).tolist(), "accuracy_ci": np.percentile(accs, q).tolist()}


def slice_metrics(y_true, y_pred, langs) -> dict:
    """English vs non-English slices ('und' = too short to tell, counted with English)."""
    y_true, y_pred, langs = map(np.asarray, (y_true, y_pred, langs))
    non_en = ~np.isin(langs, ["en", "und"])
    return {
        "english": core_metrics(y_true[~non_en], y_pred[~non_en]),
        "non_english": core_metrics(y_true[non_en], y_pred[non_en]),
    }


def coarse_metrics(y_true, y_pred) -> dict:
    """Metrics after merging shipment_information.* siblings: shows whether errors stay in-family."""
    return core_metrics([parent_of(y) for y in y_true], [parent_of(y) for y in y_pred])


def plot_confusion(y_true, y_pred, labels, path, title: str = "Confusion matrix") -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    short = [l.replace("shipment_information.", "ship.") for l in labels]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=short, yticklabels=short, ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return cm


def full_report(y_true, y_pred, labels, langs) -> dict:
    return {
        **core_metrics(y_true, y_pred),
        **bootstrap_ci(y_true, y_pred),
        "slices": slice_metrics(y_true, y_pred, langs),
        "coarse": coarse_metrics(y_true, y_pred),
        "per_class": per_class(y_true, y_pred, labels).round(4).to_dict(orient="index"),
    }
