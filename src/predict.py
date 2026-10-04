"""Inference: predict(text) -> (label, confidence), with the calibrated reject rule stored in the model config.

    from src.predict import IntentRouter
    router = IntentRouter("Badalt/intent-classifier-mpnet")      # Hub id or local path
    router.predict("whats the eta on LD-55501 pls")              # -> ("shipment_information.realtime_query", 0.97)
    router.predict("how do I file my taxes in Spain?")           # -> ("unknown", 0.31) if below threshold

Confidence = max softmax(logits / calibration_temperature). If it is below `unknown_threshold`
the label is `unknown` (route to fallback / clarification), otherwise the argmax intent.

CLI: python -m src.predict --model outputs/model_a/model "text one" "text two"
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class IntentRouter:
    def __init__(self, model: str, device: str | None = None, max_length: int = 64):
        self.device = device or best_device()
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.device).eval()
        cfg = self.model.config
        self.id2label = {int(k): v for k, v in cfg.id2label.items()}
        self.temperature = float(getattr(cfg, "calibration_temperature", 1.0))
        self.threshold = float(getattr(cfg, "unknown_threshold", 0.0))
        self.reject_label = getattr(cfg, "reject_label", "unknown")
        self.max_length = max_length

    @torch.no_grad()
    def predict_batch(self, texts: list[str], reject: bool = True) -> list[dict]:
        enc = self.tok(texts, truncation=True, max_length=self.max_length, padding=True,
                       return_tensors="pt").to(self.device)
        probs = torch.softmax(self.model(**enc).logits.float() / self.temperature, dim=-1).cpu()
        conf, idx = probs.max(-1)
        out = []
        for c, i, p in zip(conf.tolist(), idx.tolist(), probs):
            intent = self.id2label[i]
            label = self.reject_label if reject and c < self.threshold else intent
            top3 = torch.topk(p, k=min(3, len(p)))
            out.append({"label": label, "confidence": c, "top_intent": intent,
                        "top3": [(self.id2label[j], float(v)) for v, j in zip(top3.values, top3.indices.tolist())]})
        return out

    def predict(self, text: str) -> tuple[str, float]:
        r = self.predict_batch([text])[0]
        return r["label"], r["confidence"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("texts", nargs="+")
    a = ap.parse_args()
    router = IntentRouter(a.model)
    print(f"temperature={router.temperature:.3f} threshold={router.threshold:.3f}")
    for t, r in zip(a.texts, router.predict_batch(a.texts)):
        print(f"{r['label']:38s} conf={r['confidence']:.3f}  top={r['top_intent']}  | {t}")


if __name__ == "__main__":
    main()
