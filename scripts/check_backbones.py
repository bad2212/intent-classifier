"""Gate for Step 4: every bake-off backbone must load via AutoModelForSequenceClassification (the Hub
requirement) and run a forward pass on multilingual input. Also reports tokenizer coverage (unknown tokens)
on the non-Latin scripts present in the data.

Usage: python scripts/check_backbones.py
"""
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BACKBONES = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "FacebookAI/xlm-roberta-base",
]
# made-up probes (no dataset rows): EN slang, ES, ZH, AR, DE + emoji
PROBES = ["whats the eta on LD-55501 pls", "¿Dónde está mi envío ahora mismo?",
          "这个货物什么时候到达？", "متى ستصل الشحنة؟", "Hallo zusammen! 👍"]

device = "mps" if torch.backends.mps.is_available() else "cpu"
for name in BACKBONES:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=12).to(device).eval()
    enc = tok(PROBES, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    unk = sum((ids == tok.unk_token_id).sum().item() for ids in enc["input_ids"])
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"OK  {name}\n    arch={type(model).__name__} params={n_params:.0f}M vocab={tok.vocab_size} "
          f"logits={tuple(logits.shape)} unk_tokens={unk} load+fwd={time.time() - t0:.1f}s")
    del model
