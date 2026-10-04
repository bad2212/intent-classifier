"""Serving latency of the fine-tuned router (S5): per-request (batch 1) and batched throughput, CPU by default
(the realistic serving target; also keeps the MPS GPU free for training, decisions.md G39).
Uses made-up messages of typical length (no dataset text). Usage: python scripts/bench_latency.py [--device cpu]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import IntentRouter  # noqa: E402

MSGS = ["whats the eta on LD-55501 pls", "can you show me late deliveries by carrier for last quarter",
        "¿Dónde está mi envío ahora mismo?", "why is the dashboard so slow to load this morning",
        "book a dock appointment for tomorrow 9am at the dallas dc", "thanks, that's all for now!",
        "这个货物什么时候到达？", "how many trailers are waiting in the yard right now"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "outputs" / "model_a" / "model"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    r = IntentRouter(a.model, device=a.device)
    for _ in range(5):
        r.predict_batch(MSGS[:1])  # warm-up
    single = []
    for i in range(200):
        t0 = time.perf_counter()
        r.predict_batch([MSGS[i % len(MSGS)]])
        single.append(1000 * (time.perf_counter() - t0))
    batch = (MSGS * 4)[:32]
    t0 = time.perf_counter()
    for _ in range(20):
        r.predict_batch(batch)
    per_msg_batched = 1000 * (time.perf_counter() - t0) / (20 * len(batch))
    res = {"device": a.device, "threads": a.threads, "params_M": round(sum(p.numel() for p in r.model.parameters()) / 1e6),
           "batch1_p50_ms": float(np.percentile(single, 50)), "batch1_p95_ms": float(np.percentile(single, 95)),
           "batch32_ms_per_msg": per_msg_batched, "throughput_msgs_per_s_batch32": 1000 / per_msg_batched}
    (ROOT / "reports" / "latency.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
