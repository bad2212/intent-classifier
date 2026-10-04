"""Fine-tune a multilingual encoder for intent classification.

Modes
  single  Train on train, select the best epoch by val macro-F1 (early stopping), fit temperature T and
          reject threshold tau on val, evaluate ONCE on --final-eval-split (test by default), save the
          model (with T, tau in config.json) and features for open-set analysis, log everything to one W&B run.
  cv      5-fold CV on train+val with pinned folds (data/splits/folds.csv), fixed epochs, per-epoch
          out-of-fold (OOF) evaluation. Used for the backbone/LR bake-off and as the uncertainty estimate.
          One W&B run per config with the pooled OOF macro-F1 curve. Saves metrics + OOF predictions only.

Usage
  python -m src.train --config configs/base.yaml --mode cv --set model=<hf-id> lr=2e-5 name=cv-minilm-2e-5
  python -m src.train --config configs/model_a.yaml --mode single
  python -m src.train --config configs/base.yaml --mode single --set name=smoke epochs=1 --final-eval-split val
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers import set_seed as hf_set_seed

from src.augment import idswap_copies, noise_copies, translation_copies
from src.common import FOLDS_CSV, REPORTS, ROOT, label_maps, load_with_split, parent_of, set_seed
from src.metrics import core_metrics, full_report, plot_confusion

OUTPUTS = ROOT / "outputs"


# ----------------------------------------------------------------------------- config / setup
def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for kv in overrides:
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise KeyError(f"unknown config key: {k}")
        cfg[k] = yaml.safe_load(v)
    cfg["lr"] = float(cfg["lr"])
    return cfg


def wandb_mode() -> str:
    """online if credentials exist, else offline (sync later with `wandb sync wandb/offline-run-*`)."""
    if os.environ.get("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    netrc = Path.home() / ".netrc"
    if netrc.exists() and "api.wandb.ai" in netrc.read_text():
        return "online"
    return "offline"


def init_wandb(cfg: dict, job_type: str, group: str):
    """Start a W&B run; on a network failure fall back to offline (sync later) instead of crashing a long job."""
    import wandb

    os.environ["WANDB_MODE"] = wandb_mode()
    os.environ.setdefault("WANDB_LOG_MODEL", "false")  # never upload weights to W&B
    kwargs = dict(project=cfg["wandb_project"], entity=cfg["wandb_entity"] or os.environ.get("WANDB_ENTITY"),
                  name=cfg["name"], group=group, job_type=job_type, config=cfg, reinit="finish_previous",
                  settings=wandb.Settings(init_timeout=300))
    try:
        run = wandb.init(**kwargs)
    except wandb.errors.CommError as e:
        print(f"[wandb] online init failed ({e.__class__.__name__}); falling back to offline")
        os.environ["WANDB_MODE"] = "offline"
        run = wandb.init(**kwargs)
    print(f"[wandb] mode={os.environ['WANDB_MODE']}")
    return run


def seed_all(seed: int) -> None:
    set_seed(seed)
    hf_set_seed(seed)


# ----------------------------------------------------------------------------- data / model
def known_and_holdout(df: pd.DataFrame, holdout: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    unknown = df.label.isin(holdout)
    return df[~unknown].copy(), df[unknown].copy()


def to_dataset(df: pd.DataFrame, tok, max_length: int, label2id: dict) -> Dataset:
    enc = tok(df.text.tolist(), truncation=True, max_length=max_length)
    return Dataset.from_dict({**enc, "labels": [label2id[l] for l in df.label]})


def train_dataset(tr: pd.DataFrame, cfg: dict, tok, label2id: dict) -> Dataset:
    """Training rows + optional augmentation + optional Outlier-Exposure rows (label -1).
    Augmented copies come only from rows in `tr` (keyed by source id, decisions.md D-23), so a paraphrase or
    translation of a val/test/held-out-fold row can never enter training."""
    parts = [tr[["text", "label"]]]
    if "noise" in cfg.get("augment", []):
        parts.append(noise_copies(tr, cfg.get("aug_copies", 1), cfg["seed"])[["text", "label"]])
    if "translate" in cfg.get("augment", []):
        parts.append(translation_copies(tr)[["text", "label"]])
    if "idswap" in cfg.get("augment", []):
        parts.append(idswap_copies(tr, cfg["seed"])[["text", "label"]])
    frame = pd.concat(parts, ignore_index=True)
    texts, labels = frame.text.tolist(), [label2id[l] for l in frame.label]
    if cfg.get("oe_lambda", 0) > 0:
        aux = pd.read_csv(ROOT / cfg["oe_file"])
        texts += aux.text.tolist()
        labels += [-1] * len(aux)
    enc = tok(texts, truncation=True, max_length=cfg["max_length"])
    return Dataset.from_dict({**enc, "labels": labels})


def parent_tensor(label2id: dict) -> torch.Tensor:
    """class id -> parent-group id (the 3 shipment_information.* siblings share one parent)."""
    parents = sorted({parent_of(l) for l in label2id})
    return torch.tensor([parents.index(parent_of(l)) for l in label2id])


def class_weight_tensor(labels: pd.Series, label2id: dict) -> torch.Tensor:
    counts = labels.value_counts()
    w = np.array([len(labels) / (len(label2id) * counts[l]) for l in label2id], dtype=np.float32)
    return torch.tensor(w)


class WeightedTrainer(Trainer):
    """Cross-entropy (+ optional class weights, label smoothing) with two optional stretch terms:
      * hierarchical parent loss (S3): + hier_lambda * CE(parent), parent log-prob = logsumexp of its children
      * Outlier Exposure (S1b, Hendrycks et al. 2019): rows labelled -1 are auxiliary outliers trained towards a
        uniform distribution: + oe_lambda * mean(logsumexp(z) - mean(z))"""

    def __init__(self, *a, class_weights: torch.Tensor | None = None, label_smoothing: float = 0.0,
                 parent_of_class: torch.Tensor | None = None, hier_lambda: float = 0.0, oe_lambda: float = 0.0, **kw):
        super().__init__(*a, **kw)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.hier_lambda, self.oe_lambda = hier_lambda, oe_lambda
        self.parent_of_class = parent_of_class
        if parent_of_class is not None:
            mask = torch.zeros(int(parent_of_class.max()) + 1, len(parent_of_class), dtype=torch.bool)
            mask[parent_of_class, torch.arange(len(parent_of_class))] = True
            self.parent_mask = mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        F = torch.nn.functional
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        z = outputs.logits
        known = labels >= 0
        loss = z.sum() * 0.0  # keeps the graph valid for an all-outlier batch
        if known.any():
            w = self.class_weights.to(z.device) if self.class_weights is not None else None
            loss = loss + F.cross_entropy(z[known], labels[known], weight=w, label_smoothing=self.label_smoothing)
            if self.hier_lambda > 0:
                logp = F.log_softmax(z[known], -1)
                mask = self.parent_mask.to(z.device)
                parent_logp = torch.logsumexp(logp.unsqueeze(1).masked_fill(~mask.unsqueeze(0), float("-inf")), -1)
                parent_y = self.parent_of_class.to(z.device)[labels[known]]
                loss = loss + self.hier_lambda * F.nll_loss(parent_logp, parent_y)
        if self.oe_lambda > 0 and (~known).any():
            zo = z[~known]
            loss = loss + self.oe_lambda * (torch.logsumexp(zo, -1) - zo.mean(-1)).mean()
        return (loss, outputs) if return_outputs else loss


class KeepBest(TrainerCallback):
    """One on-disk copy of the best weights by val macro-F1, overwritten in place (instead of HF checkpoint folders,
    which keep best + last = 2x model size). Same rule as HF: strictly greater -> the first best epoch wins."""

    def __init__(self, path: Path):
        self.path, self.best, self.best_epoch = path, -float("inf"), None

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        m = (metrics or {}).get("eval_macro_f1")
        if m is not None and m > self.best:
            self.best, self.best_epoch = m, state.epoch
            model.save_pretrained(self.path)


def new_model(cfg: dict, label2id: dict, id2label: dict):
    seed_all(cfg["seed"])  # identical classifier-head init across folds/configs
    return AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], num_labels=len(label2id), label2id=label2id, id2label=id2label)


def training_args(cfg: dict, out_dir: Path, single: bool, report_to: list[str]) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=cfg["lr"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=64,
        warmup_steps=cfg["warmup_ratio"],  # transformers 5: a float < 1 is a ratio of total steps
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="no",            # disk-bound dev machine: KeepBest keeps a single best copy instead
        load_best_model_at_end=False,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=5,
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        report_to=report_to,
        run_name=cfg["name"],
        use_cpu=cfg["cpu"],
        dataloader_pin_memory=False,
        disable_tqdm=True,
    )


def macro_f1_metrics(eval_pred) -> dict:
    preds = eval_pred.predictions.argmax(-1)
    m = core_metrics(eval_pred.label_ids, preds)
    return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}


def free() -> None:
    """Release accelerator memory between runs (callers `del` their own references first)."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ----------------------------------------------------------------------------- calibration / features
def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Single scalar T minimising NLL on held-out logits (Guo et al. 2017). Doesn't change argmax."""
    z, y = torch.tensor(logits, dtype=torch.float64), torch.tensor(labels)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().item())


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def retention_threshold(max_probs: np.ndarray, retention: float) -> float:
    """Largest tau that still keeps >= `retention` of known inputs (tau is an observed value)."""
    return float(np.quantile(max_probs, 1 - retention, method="lower"))


@torch.no_grad()
def extract_features(model, tok, texts: list[str], max_length: int, batch_size: int = 64) -> dict:
    """Logits + [CLS] + mean-pooled last hidden state, for calibration and open-set scorers."""
    model.eval()
    device = next(model.parameters()).device
    out = {"logits": [], "cls": [], "mean": []}
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt").to(device)
        o = model(**enc, output_hidden_states=True)
        h = o.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        out["logits"].append(o.logits.float().cpu())
        out["cls"].append(h[:, 0].float().cpu())
        out["mean"].append(((h * mask).sum(1) / mask.sum(1)).float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def save_features(path: Path, df: pd.DataFrame, feats: dict) -> None:
    np.savez_compressed(path, ids=df.id.values, labels=df.label.values, langs=df.lang.values, **feats)


# ----------------------------------------------------------------------------- single mode
def run_single(cfg: dict, final_split: str) -> dict:
    df = load_with_split()
    known, unknown = known_and_holdout(df, cfg["holdout_classes"])
    label2id, id2label = label_maps(known.label)
    labels = list(label2id)
    tr, va = known[known.split == "train"], known[known.split == "val"]
    fe = known[known.split == final_split]
    # unit checks: disjoint splits, held-out classes never seen in train/val
    assert not (set(tr.id) & set(va.id)) and not (set(tr.id) & set(fe.id))
    assert final_split == "val" or not (set(va.id) & set(fe.id))
    assert not (set(cfg["holdout_classes"]) & (set(tr.label) | set(va.label)))
    print(f"[data] train {len(tr)} · val {len(va)} · {final_split} {len(fe)} · held-out rows {len(unknown)} "
          f"· classes {len(labels)}")

    out = OUTPUTS / cfg["name"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    run = init_wandb(cfg, job_type="train", group="model_b" if cfg["holdout_classes"] else "model_a")

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = new_model(cfg, label2id, id2label)
    keep_best = KeepBest(out / "model")
    trainer = WeightedTrainer(
        model=model,
        args=training_args(cfg, out / "run", single=True, report_to=["wandb"]),
        train_dataset=train_dataset(tr, cfg, tok, label2id),
        eval_dataset=to_dataset(va, tok, cfg["max_length"], label2id),
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=macro_f1_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["patience"]), keep_best],
        class_weights=class_weight_tensor(tr.label, label2id) if cfg["class_weights"] else None,
        label_smoothing=cfg["label_smoothing"],
        parent_of_class=parent_tensor(label2id),
        hier_lambda=cfg.get("hier_lambda", 0.0),
        oe_lambda=cfg.get("oe_lambda", 0.0),
    )
    trainer.train()
    from safetensors.torch import load_file

    trainer.model.load_state_dict(load_file(out / "model" / "model.safetensors"))  # restore the best epoch
    best_epoch = keep_best.best_epoch
    assert abs(keep_best.best - trainer.state.best_metric) < 1e-12, "KeepBest disagrees with Trainer's best metric"

    # --- features for every set we will analyse (best checkpoint is loaded) ---
    sets = {"train": tr, "val": va, final_split: fe}
    if len(unknown):
        sets["unknown"] = unknown
    feats = {k: extract_features(trainer.model, tok, d.text.tolist(), cfg["max_length"]) for k, d in sets.items()}
    for k, d in sets.items():
        save_features(out / f"features_{k}.npz", d, feats[k])

    # restored-weights check: val macro-F1 recomputed from the restored model must equal the best epoch's score
    restored_f1 = core_metrics(va.label.map(label2id).values, feats["val"]["logits"].argmax(1))["macro_f1"]
    print(f"[check] restored-model val macro-F1 {restored_f1:.4f} vs best epoch {trainer.state.best_metric:.4f}")
    if abs(restored_f1 - trainer.state.best_metric) > 0.02:
        raise RuntimeError("restored weights do not reproduce the best epoch's val score")

    # --- calibration on val: temperature + reject threshold (same model that is deployed) ---
    y_val = va.label.map(label2id).values
    T = fit_temperature(feats["val"]["logits"], y_val)
    val_maxp = softmax(feats["val"]["logits"], T).max(1)
    tau = retention_threshold(val_maxp, cfg["retention_target"])
    # 'temperature' is a reserved generation key in transformers configs -> calibration_temperature
    calib = {"calibration_temperature": T, "unknown_threshold": tau, "retention_target": cfg["retention_target"],
             "val_retention_at_tau": float((val_maxp >= tau).mean())}

    # --- final evaluation (Track A headline = no reject rule) ---
    logits = feats[final_split]["logits"]
    pred = [id2label[i] for i in logits.argmax(1)]
    report = full_report(fe.label.values, pred, labels, fe.lang.values)
    maxp = softmax(logits, T).max(1)
    accepted = maxp >= tau
    report["with_reject_rule"] = {
        "retention": float(accepted.mean()),
        "accuracy_on_accepted": float((np.array(pred)[accepted] == fe.label.values[accepted]).mean()),
    }
    train_pred = [id2label[i] for i in feats["train"]["logits"].argmax(1)]
    report["train_fit"] = core_metrics(tr.label.values, train_pred)
    cm_path = out / f"confusion_{final_split}.png"
    plot_confusion(fe.label.values, pred, labels, cm_path, f"{cfg['name']} — {final_split} (n={len(fe)})")

    # --- save model with the reject rule in its config ---
    trainer.model.config.update({**calib, "reject_label": "unknown",
                                 "reject_rule": "predict 'unknown' if max softmax(logits / calibration_temperature) < unknown_threshold"})
    trainer.model.save_pretrained(out / "model")  # overwrites the KeepBest copy in place (no second copy on disk)
    tok.save_pretrained(out / "model")
    result = {"config": cfg, "best_epoch": best_epoch, "best_val_macro_f1": trainer.state.best_metric,
              "calibration": calib, final_split: report}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float))

    # --- W&B: final metrics, per-class table, confusion matrix (same run as the training curves) ---
    import wandb

    ns = final_split if final_split == "test" else f"{final_split}_final"
    pc = pd.DataFrame(report["per_class"]).T.reset_index(names="class")
    y_ids = [label2id[l] for l in fe.label]
    p_ids = [label2id[l] for l in pred]
    wandb.log({
        f"{ns}/per_class": wandb.Table(dataframe=pc[["class", "precision", "recall", "f1", "support"]]),
        f"{ns}/confusion_matrix": wandb.plot.confusion_matrix(y_true=y_ids, preds=p_ids, class_names=labels),
        f"{ns}/confusion_matrix_img": wandb.Image(str(cm_path)),
    })
    run.summary.update({
        f"{ns}/macro_f1": report["macro_f1"], f"{ns}/accuracy": report["accuracy"],
        f"{ns}/macro_f1_ci_low": report["macro_f1_ci"][0], f"{ns}/macro_f1_ci_high": report["macro_f1_ci"][1],
        f"{ns}/non_english_accuracy": report["slices"]["non_english"]["accuracy"],
        f"{ns}/coarse_accuracy": report["coarse"]["accuracy"],
        f"{ns}/retention_with_reject": report["with_reject_rule"]["retention"],
        "best_epoch": best_epoch, "train_fit/macro_f1": report["train_fit"]["macro_f1"], **calib,
        **{f"{ns}/f1/{c}": m["f1"] for c, m in report["per_class"].items()},
    })
    run.finish()

    print(f"[done] best epoch {best_epoch} · val macro-F1 {trainer.state.best_metric:.3f} · T={T:.2f} tau={tau:.3f}")
    print(f"[{final_split}] macro-F1 {report['macro_f1']:.3f} (CI {report['macro_f1_ci'][0]:.3f}–"
          f"{report['macro_f1_ci'][1]:.3f}) acc {report['accuracy']:.3f} · non-EN acc "
          f"{report['slices']['non_english']['accuracy']:.3f} · train-fit macro-F1 {report['train_fit']['macro_f1']:.3f}")
    del trainer, model
    free()
    return result


# ----------------------------------------------------------------------------- cv mode
def run_cv(cfg: dict) -> dict:
    df = load_with_split()
    tv = df[df.split != "test"].merge(pd.read_csv(FOLDS_CSV), on="id", validate="one_to_one")
    tv, _ = known_and_holdout(tv, cfg["holdout_classes"])
    label2id, id2label = label_maps(tv.label)
    folds = sorted(tv.fold.unique())[: cfg["cv_folds_limit"] or None]
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    run = init_wandb(cfg, job_type="cv", group="cv-bakeoff")

    epoch_logits: dict[int, list[np.ndarray]] = {}  # fold -> per-epoch OOF logits
    for k in folds:
        tr, ho = tv[tv.fold != k], tv[tv.fold == k]
        assert not set(tr.id) & set(ho.id)
        fold_logits: list[np.ndarray] = []

        def metrics(eval_pred, _store=fold_logits):
            _store.append(eval_pred.predictions)
            return macro_f1_metrics(eval_pred)

        model = new_model(cfg, label2id, id2label)
        trainer = WeightedTrainer(
            model=model,
            args=training_args(cfg, OUTPUTS / "cv_tmp", single=False, report_to=[]),
            train_dataset=train_dataset(tr, cfg, tok, label2id),
            eval_dataset=to_dataset(ho, tok, cfg["max_length"], label2id),
            processing_class=tok,
            data_collator=DataCollatorWithPadding(tok),
            compute_metrics=metrics,
            class_weights=class_weight_tensor(tr.label, label2id) if cfg["class_weights"] else None,
            label_smoothing=cfg["label_smoothing"],
            parent_of_class=parent_tensor(label2id),
            hier_lambda=cfg.get("hier_lambda", 0.0),
            oe_lambda=cfg.get("oe_lambda", 0.0),
        )
        trainer.train()
        epoch_logits[k] = fold_logits
        f1s = [core_metrics(ho.label.map(label2id), l.argmax(1))["macro_f1"] for l in fold_logits]
        print(f"[cv] fold {k}: macro-F1 by epoch {np.round(f1s, 3).tolist()}")
        del trainer, model
        free()
    shutil.rmtree(OUTPUTS / "cv_tmp", ignore_errors=True)

    # --- per-epoch pooled OOF metrics (every row predicted once by a model that never saw it) ---
    used = tv[tv.fold.isin(folds)]
    n_epochs = min(len(v) for v in epoch_logits.values())
    curve = []
    for e in range(n_epochs):
        preds = pd.Series(index=used.index, dtype=object)
        fold_f1 = []
        for k in folds:
            idx = used.index[used.fold == k]
            p = [id2label[i] for i in epoch_logits[k][e].argmax(1)]
            preds.loc[idx] = p
            fold_f1.append(core_metrics(used.loc[idx, "label"], p)["macro_f1"])
        pooled = full_report(used.label.values, preds.values, list(label2id), used.lang.values)
        curve.append({"epoch": e + 1, "pooled_macro_f1": pooled["macro_f1"], "pooled_accuracy": pooled["accuracy"],
                      "fold_macro_f1_mean": float(np.mean(fold_f1)), "fold_macro_f1_std": float(np.std(fold_f1)),
                      "non_english_accuracy": pooled["slices"]["non_english"]["accuracy"],
                      "english_accuracy": pooled["slices"]["english"]["accuracy"],
                      "coarse_accuracy": pooled["coarse"]["accuracy"], "_report": pooled, "_preds": preds.values})
        run.log({f"cv/{k}": v for k, v in curve[-1].items() if not k.startswith("_") and k != "epoch"}, step=e + 1)

    best = max(curve, key=lambda c: c["pooled_macro_f1"])
    out = REPORTS / "cv"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": used.id.values, "label": used.label.values, "fold": used.fold.values,
                  "lang": used.lang.values, "pred": best["_preds"]}).to_csv(out / f"{cfg['name']}_oof.csv", index=False)
    summary = {"config": cfg, "best_epoch": best["epoch"],
               "best": {k: v for k, v in best.items() if not k.startswith("_")},
               "per_class_at_best": best["_report"]["per_class"],
               "curve": [{k: v for k, v in c.items() if not k.startswith("_")} for c in curve]}
    (out / f"{cfg['name']}.json").write_text(json.dumps(summary, indent=2, default=float))
    run.summary.update({f"best/{k}": v for k, v in summary["best"].items()})
    run.finish()
    b = summary["best"]
    print(f"[cv] {cfg['name']}: best epoch {b['epoch']} · pooled OOF macro-F1 {b['pooled_macro_f1']:.3f} "
          f"(fold mean {b['fold_macro_f1_mean']:.3f} ± {b['fold_macro_f1_std']:.3f}) · non-EN acc "
          f"{b['non_english_accuracy']:.3f}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["single", "cv"], default="single")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--final-eval-split", choices=["test", "val"], default="test",
                    help="'val' only for smoke tests: exercises the final-eval path without touching test")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    seed_all(cfg["seed"])
    run_single(cfg, args.final_eval_split) if args.mode == "single" else run_cv(cfg)


if __name__ == "__main__":
    main()
