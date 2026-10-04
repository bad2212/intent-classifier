"""Track B: open-set / abstention evaluation of a model trained with held-out classes (Model B).

Every scorer returns an "unknown-ness" score (HIGHER = more likely unknown). For each scorer:
  * threshold tau = the value that keeps `retention_target` of KNOWN-class **val** rows (leak-free: no unknowns used)
  * test set = known-class test rows + ALL held-out-class rows (never seen in training or validation)
  * metrics: AUROC / AUPR (unknown = positive), unknown-accept rate at 95% known retention (test-oracle,
    threshold-free comparison), and the operational numbers at the val-chosen tau: rejection recall
    (strict: predicted `unknown`; lenient: `unknown` or routed to other/chitchat), retention, per-held-out-class recall.

Scorers: MSP (required baseline), temperature-MSP, max-logit, energy (Liu et al. 2020), Mahalanobis with class means +
Ledoit-Wolf shared covariance (Lee et al. 2018; Podolskiy et al. 2021), kNN cosine distance (Sun et al. 2022).

Usage: python -m src.openset --run model_b [--wandb]
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common import REPORTS, SEED
from src.metrics import core_metrics
from src.train import OUTPUTS

CATCH_ALLS = {"other", "chitchat"}


# ----------------------------------------------------------------------------- scorers (higher = unknown)
def msp(z, **_):
    return -softmax(z, axis=1).max(1)


def msp_temp(z, T, **_):
    return -softmax(z / T, axis=1).max(1)


def max_logit(z, **_):
    return -z.max(1)


def energy(z, **_):
    return -logsumexp(z, axis=1)  # E(x) = -T·logsumexp(z/T) with T = 1


class Mahalanobis:
    """min over classes of (x-mu_c)^T S^-1 (x-mu_c); S = Ledoit-Wolf shrinkage of the pooled within-class
    covariance. Shrinkage is required: ~300 training points in 384-768 dims makes the sample covariance singular."""

    def __init__(self, feats: np.ndarray, labels: np.ndarray):
        self.classes = np.unique(labels)
        self.mu = np.stack([feats[labels == c].mean(0) for c in self.classes])
        centered = feats - self.mu[np.searchsorted(self.classes, labels)]
        self.prec = LedoitWolf().fit(centered).precision_

    def __call__(self, x: np.ndarray) -> np.ndarray:
        d = np.stack([np.einsum("ij,jk,ik->i", x - m, self.prec, x - m) for m in self.mu], 1)
        return d.min(1)


class KNN:
    """Negative cosine similarity to the k-th nearest training example (normalised features)."""

    def __init__(self, feats: np.ndarray, k: int):
        self.bank = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        self.k = k

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        sims = np.sort(x @ self.bank.T, axis=1)[:, ::-1]
        return -sims[:, self.k - 1]


# ----------------------------------------------------------------------------- metrics
def threshold_at_retention(known_scores: np.ndarray, retention: float) -> float:
    """Smallest observed tau such that >= `retention` of known rows have score <= tau (are kept)."""
    return float(np.quantile(known_scores, retention, method="higher"))


def unknown_accept_at_retention(known: np.ndarray, unknown: np.ndarray, retention: float = 0.95) -> float:
    """Test-oracle threshold-free comparison (a.k.a. FPR@95TPR with knowns as positives)."""
    tau = threshold_at_retention(known, retention)
    return float((unknown <= tau).mean())


def curve(known: np.ndarray, unknown: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    taus = np.unique(np.concatenate([known, unknown]))
    return np.array([(known <= t).mean() for t in taus]), np.array([(unknown > t).mean() for t in taus])


def load(run: str) -> tuple[dict, dict]:
    d = OUTPUTS / run
    feats = {k: dict(np.load(d / f"features_{k}.npz", allow_pickle=True)) for k in ("train", "val", "test", "unknown")}
    return feats, json.loads((d / "metrics.json").read_text())


def evaluate(run: str, use_wandb: bool = False) -> dict:
    feats, meta = load(run)
    cfg, calib = meta["config"], meta["calibration"]
    T, retention = calib["calibration_temperature"], cfg["retention_target"]
    from transformers import AutoConfig

    id2label = {int(k): v for k, v in AutoConfig.from_pretrained(OUTPUTS / run / "model").id2label.items()}
    tr, va, te, un = feats["train"], feats["val"], feats["test"], feats["unknown"]
    assert not (set(un["labels"]) & (set(tr["labels"]) | set(va["labels"]))), "held-out class leaked into train/val"

    scorers = {
        "MSP (baseline)": lambda f: msp(f["logits"]),
        "MSP + temperature": lambda f: msp_temp(f["logits"], T),
        "max-logit": lambda f: max_logit(f["logits"]),
        "energy": lambda f: energy(f["logits"]),
    }
    for feat in ("cls", "mean"):
        maha = Mahalanobis(tr[feat], tr["labels"])
        scorers[f"Mahalanobis [{feat}]"] = lambda f, m=maha, ft=feat: m(f[ft])
        for k in (1, 5, 10):
            knn = KNN(tr[feat], k)
            scorers[f"kNN k={k} [{feat}]"] = lambda f, m=knn, ft=feat: m(f[ft])

    # orientation unit check: random scores must give AUROC ~ 0.5
    rng = np.random.default_rng(SEED)
    y = np.r_[np.zeros(len(te["labels"])), np.ones(len(un["labels"]))]
    assert 0.35 < roc_auc_score(y, rng.random(len(y))) < 0.65

    known_pred = np.array([id2label[i] for i in te["logits"].argmax(1)])
    unk_pred = np.array([id2label[i] for i in un["logits"].argmax(1)])
    rows, curves = [], {}
    for name, fn in scorers.items():
        s_val, s_te, s_un = fn(va), fn(te), fn(un)
        tau = threshold_at_retention(s_val, retention)
        rejected_unk = s_un > tau
        lenient = rejected_unk | np.isin(unk_pred, list(CATCH_ALLS))
        row = {
            "scorer": name,
            "AUROC": roc_auc_score(y, np.r_[s_te, s_un]),
            "AUPR_unknown": average_precision_score(y, np.r_[s_te, s_un]),
            "unknown_accepted@95%retention(oracle)": unknown_accept_at_retention(s_te, s_un, 0.95),
            "tau(val)": tau,
            "val_retention": float((s_val <= tau).mean()),
            "test_retention": float((s_te <= tau).mean()),
            "rejection_recall_strict": float(rejected_unk.mean()),
            "rejection_recall_lenient": float(lenient.mean()),
        }
        for c in sorted(set(un["labels"])):
            row[f"rejection_recall[{c}]"] = float(rejected_unk[un["labels"] == c].mean())
        rows.append(row)
        curves[name] = curve(s_te, s_un)
    table = pd.DataFrame(rows)

    # --- known-class quality of Model B (sanity: it should still be a good 10-way classifier) ---
    known_metrics = core_metrics(te["labels"], known_pred)
    where_unknowns_go = pd.Series(unk_pred).value_counts().to_dict()

    out = REPORTS / "trackB" / run
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "scorers.csv", index=False)
    result = {"run": run, "holdout_classes": cfg["holdout_classes"], "retention_target": retention,
              "n_known_test": int(len(te["labels"])), "n_unknown": int(len(un["labels"])),
              "known_test_10way": known_metrics, "unknowns_argmax_distribution": where_unknowns_go,
              "scorers": rows}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float))

    # --- figures: retention vs rejection curves; MSP score histogram ---
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    keep = ["MSP (baseline)", "MSP + temperature", "energy", "Mahalanobis [mean]", "kNN k=5 [mean]", "kNN k=5 [cls]"]
    for name in keep:
        r, rej = curves[name]
        auroc = table.set_index("scorer").loc[name, "AUROC"]
        ax.plot(r, rej, label=f"{name} (AUROC {auroc:.3f})")
    ax.axvline(retention, ls="--", c="grey", lw=1)
    ax.set_xlabel("retention on known test inputs"); ax.set_ylabel("rejection recall on unknown inputs")
    ax.set_xlim(0.5, 1.0); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"Track B — held out: {', '.join(cfg['holdout_classes'])}")
    fig.tight_layout(); fig.savefig(out / "retention_vs_rejection.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    p_te, p_un = softmax(te["logits"], axis=1).max(1), softmax(un["logits"], axis=1).max(1)
    bins = np.linspace(0, 1, 26)
    ax.hist(p_te, bins=bins, alpha=0.6, label=f"known test (n={len(p_te)})", density=True)
    ax.hist(p_un, bins=bins, alpha=0.6, label=f"held-out / unknown (n={len(p_un)})", density=True)
    tau_msp = -table.set_index("scorer").loc["MSP (baseline)", "tau(val)"]
    ax.axvline(tau_msp, c="k", ls="--", lw=1, label=f"val threshold (max-prob {tau_msp:.2f})")
    ax.set_xlabel("max softmax probability"); ax.set_ylabel("density"); ax.legend(fontsize=8)
    ax.set_title("Confidence: known vs unknown inputs")
    fig.tight_layout(); fig.savefig(out / "msp_hist.png", dpi=150); plt.close(fig)

    if use_wandb:
        import wandb

        from src.train import init_wandb

        run_cfg = {**cfg, "name": f"{run}-trackB"}
        wb = init_wandb(run_cfg, job_type="trackB", group="model_b")
        wb.log({"trackB/scorers": wandb.Table(dataframe=table.round(4)),
                "trackB/retention_vs_rejection": wandb.Image(str(out / "retention_vs_rejection.png")),
                "trackB/msp_hist": wandb.Image(str(out / "msp_hist.png"))})
        msp_row = table.set_index("scorer").loc["MSP (baseline)"]
        wb.summary.update({"trackB/msp_auroc": msp_row["AUROC"], "trackB/msp_rejection_recall": msp_row["rejection_recall_strict"],
                           "trackB/msp_test_retention": msp_row["test_retention"],
                           "trackB/best_auroc": table.AUROC.max(), "trackB/best_scorer": table.loc[table.AUROC.idxmax(), "scorer"]})
        wb.finish()

    cols = ["scorer", "AUROC", "unknown_accepted@95%retention(oracle)", "test_retention", "rejection_recall_strict",
            "rejection_recall_lenient"]
    print(f"[trackB] {run} · held out {cfg['holdout_classes']} · known test {len(te['labels'])} · unknown {len(un['labels'])}")
    print(f"[trackB] Model B known-class test (10-way): macro-F1 {known_metrics['macro_f1']:.3f} acc {known_metrics['accuracy']:.3f}")
    print(f"[trackB] where unknowns go (argmax): {where_unknowns_go}")
    print(table[cols].round(3).to_string(index=False))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="outputs/<run> of a model trained with holdout_classes")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    evaluate(a.run, a.wandb)


if __name__ == "__main__":
    main()
