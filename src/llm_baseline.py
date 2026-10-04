"""Stretch S2: zero/few-shot LLM baseline (Google Gemini by default; OpenAI optional) vs the fine-tuned model,
on the SAME test rows.

Modes
  closed   12 labels, the model must pick one (Track A comparison: macro-F1, accuracy, non-EN slice)
  open     Track B protocol: only the 10 known labels of Model B + "unknown"; fed the known test rows of those
           10 classes + all held-out-class rows -> rejection recall / retention, comparable to Model B
Shots: 0 (label descriptions only) or k examples per class drawn from TRAIN only (seed 42) — never val/test (G29).
Structured output: enum-constrained answer (Gemini `text/x.enum`, OpenAI JSON schema) -> always a valid label.
Responses cached by (model, mode, shots, row id) in outputs/llm_cache.jsonl (ids only, no text), written as each
answer arrives -> reruns are free and a free-tier quota stop resumes where it left off.
Free tier: requests are paced (--rpm) and 429s are retried with the server's suggested delay; a DAILY quota stop
exits cleanly (re-run the next day to continue from the cache).

Key: GEMINI_API_KEY (or OPENAI_API_KEY) in the git-ignored .env file.
Usage: python -m src.llm_baseline --list-models
       python -m src.llm_baseline --model gemini-3.8-flash --mode closed --shots 0
Free tier (checked 2026-10-03): gemini-3.8-flash / 3.5-flash work but allow only 20 requests/day/model
(GenerateRequestsPerDayPerProjectPerModel-FreeTier) — too few for ~300 requests per model, so the runnable baseline is
gemini-3.5-flash-lite (+ flash on the cached subset); gemini-2.5-* is closed to new keys; pro has zero free quota.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.common import REPORTS, ROOT, SEED, load_with_split
from src.metrics import full_report, plot_confusion

CACHE = ROOT / "outputs" / "llm_cache.jsonl"
HOLDOUT = ["yard_management", "document_processing"]  # same as Model B (Track B)

# Written from the intent definitions and the TRAIN split only.
DESCRIPTIONS = {
    "shipment_information.realtime_query": "live status of specific shipments/loads right now: ETA, current location, "
        "last checkpoint, whether it departed/arrived, live map. Usually about one load (LD-xxxx) or a current set.",
    "shipment_information.analytics": "historical/aggregate shipment analytics: trends, counts, percentages, on-time "
        "rates, breakdowns by month/carrier/mode/region over a past period.",
    "shipment_information.disruptions": "external disruptions affecting shipments: weather, strikes, port congestion, "
        "border delays, infrastructure issues, disruption flags on lanes.",
    "appointment_manager": "dock/delivery appointments: booking, rescheduling, cancelling, calendar of appointments, "
        "slots at a facility or DC.",
    "orders": "purchase/sales orders (PO-xxxx): order status, order lines, fill rate, order cycle time, open orders, "
        "allocation of orders to loads.",
    "yard_management": "trailers and the yard: trailer location/spot, dwell time in the yard, drop trailers, yard moves, "
        "trailers waiting at a warehouse.",
    "customer_support": "problems with the platform needing help: bugs, errors, outages, login/password, permissions, "
        "app crashes, broken exports/notifications, opening a ticket.",
    "ai_agent_performance": "how the platform's AI agent performs: carrier response rates to the agent, escalations "
        "to humans, exceptions resolved, time saved, agent metrics.",
    "document_processing": "documents attached to shipments/orders: bills of lading, packing lists, PODs, invoices — "
        "extraction results, wrong extracted fields, missing or unreadable documents.",
    "knowledge_base": "how-to / product questions about using the platform: features, settings, file formats, "
        "integrations, definitions of terms, where to find something.",
    "chitchat": "social filler with no request: greetings, thanks, acknowledgements, small talk, goodbyes.",
    "other": "requests unrelated to logistics or the platform: general knowledge, personal tasks, coding help, "
        "weather, cooking, etc.",
}


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k and not k.startswith("#"):
                os.environ.setdefault(k, v)


def build_prompt(labels: list[str], open_set: bool, shots: pd.DataFrame | None) -> str:
    lines = ["You route user messages from a logistics / supply-chain platform's chat assistant to exactly one intent.",
             "Messages may be informal, contain typos, or be written in any language.", "", "Intents:"]
    lines += [f"- {l}: {DESCRIPTIONS[l]}" for l in labels]
    if open_set:
        lines += ["- unknown: the message is a request that fits NONE of the intents above (for example a capability "
                  "the platform does not offer). Use it whenever no listed intent is a good fit; do not force a guess."]
    if shots is not None and len(shots):
        lines += ["", "Examples:"]
        lines += [f'"{r.text}" -> {r.label}' for r in shots.itertuples()]
    lines += ["", "Answer with the single best intent."]
    return "\n".join(lines)


def schema(allowed: list[str]) -> dict:
    return {"type": "json_schema", "json_schema": {"name": "intent", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": ["intent"],
        "properties": {"intent": {"type": "string", "enum": allowed}}}}}


def read_cache() -> dict:
    if not CACHE.exists():
        return {}
    return {(r["model"], r["mode"], r["shots"], r["id"]): r for r in map(json.loads, CACHE.read_text().splitlines())}


class QuotaExhausted(RuntimeError):
    """Non-retryable API stop (daily/billing quota, invalid key, model unavailable): stop cleanly;
    everything answered so far is cached."""


class Pacer:
    """Spaces requests to stay under a requests-per-minute limit (shared across worker threads)."""

    def __init__(self, rpm: float):
        self.interval, self.next_t, self.lock = 60.0 / rpm, 0.0, threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        time.sleep(max(0.0, t - now))


def provider_of(model: str) -> str:
    return "gemini" if model.startswith("gemini") else "openai"


def make_client(provider: str):
    load_env()
    if provider == "gemini":
        from google import genai

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise SystemExit("GEMINI_API_KEY missing: add `GEMINI_API_KEY=...` to the git-ignored .env file")
        return genai.Client(api_key=key)
    from openai import OpenAI

    return OpenAI()


def _call_gemini(client, model: str, system: str, text: str, allowed: list[str]) -> dict:
    from google.genai import types

    cfg = dict(system_instruction=system, temperature=0.0, response_mime_type="text/x.enum",
               response_schema={"type": "STRING", "enum": allowed})
    if "flash" in model and "lite" not in model:  # no hidden reasoning for flash; lite rejects this setting
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    r = client.models.generate_content(model=model, contents=text, config=types.GenerateContentConfig(**cfg))
    u = r.usage_metadata
    return {"pred": (r.text or "").strip(), "in_tokens": u.prompt_token_count or 0,
            "out_tokens": (u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0)}


def _call_openai(client, model: str, system: str, text: str, allowed: list[str]) -> dict:
    kwargs = dict(model=model, response_format=schema(allowed),
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": text}])
    try:
        r = client.chat.completions.create(**kwargs, reasoning_effort="low")
    except Exception as e:  # models without reasoning controls
        if "reasoning" not in str(e).lower():
            raise
        r = client.chat.completions.create(**kwargs)
    return {"pred": json.loads(r.choices[0].message.content)["intent"], "in_tokens": r.usage.prompt_tokens,
            "out_tokens": r.usage.completion_tokens}


def classify(client, model: str, system: str, text: str, allowed: list[str], pacer: Pacer) -> dict:
    call = _call_gemini if provider_of(model) == "gemini" else _call_openai
    for attempt in range(8):
        pacer.wait()
        try:
            t0 = time.perf_counter()
            out = call(client, model, system, text, allowed)
            out["latency_s"] = time.perf_counter() - t0
            if out["pred"] not in allowed:
                raise ValueError(f"label outside the enum: {out['pred']!r}")
            return out
        except Exception as e:
            msg = str(e)
            if any(s in msg for s in ("insufficient_quota", "invalid_api_key", "API_KEY_INVALID", "PerDay",
                                      "per day", "model_not_found", "NOT_FOUND")):
                raise QuotaExhausted(msg[:300]) from e
            if attempt == 7:
                raise
            hint = re.search(r"retry(?:Delay)?['\"]?[:\s]+['\"]?(\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
            time.sleep(float(hint.group(1)) + 1 if hint else min(60, 2 ** attempt))


def list_models() -> None:
    client = make_client("gemini")
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []) and "gemini" in m.name:
            print(m.name.removeprefix("models/"))


def run(model: str, mode: str, shots_k: int, rpm: float = 8.0, workers: int = 2) -> dict:
    client = make_client(provider_of(model))
    df = load_with_split()
    train = df[df.split == "train"]
    labels = sorted(df.label.unique())
    if mode == "closed":
        allowed, eval_df = labels, df[df.split == "test"]
    else:
        allowed = [l for l in labels if l not in HOLDOUT]
        eval_df = pd.concat([df[(df.split == "test") & ~df.label.isin(HOLDOUT)], df[df.label.isin(HOLDOUT)]])
        allowed = allowed + ["unknown"]
    shot_pool = train[train.label.isin(allowed)]
    # groupby().sample keeps the label column (pandas 3 groupby().apply drops it -> few-shot crashed)
    shots = shot_pool.groupby("label").sample(n=shots_k, random_state=SEED) if shots_k else None
    system = build_prompt([l for l in allowed if l != "unknown"], mode == "open", shots)

    cache = read_cache()
    todo = [r for r in eval_df.itertuples() if (model, mode, shots_k, r.id) not in cache]
    pacer, lock = Pacer(rpm), threading.Lock()
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    def work(r):
        res = classify(client, model, system, r.text, allowed, pacer)
        rec = {"model": model, "mode": mode, "shots": shots_k, "id": r.id, **res}
        with lock:  # persist immediately: a quota stop loses nothing
            with CACHE.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            cache[(model, mode, shots_k, r.id)] = rec

    print(f"[llm] {model} {mode} {shots_k}-shot: {len(eval_df) - len(todo)} cached, {len(todo)} to query at <= {rpm} rpm")
    try:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, todo))
    except QuotaExhausted as e:
        done = sum((model, mode, shots_k, i) in cache for i in eval_df.id)
        raise SystemExit(f"[llm] stopped (quota, key or model availability) after {done}/{len(eval_df)} rows "
                         f"(cached; re-run to resume): {e}")
    recs = [cache[(model, mode, shots_k, i)] for i in eval_df.id]
    pred = np.array([r["pred"] for r in recs])
    lat = np.array([r["latency_s"] for r in recs])
    usage = {"mean_in_tokens": float(np.mean([r["in_tokens"] for r in recs])),
             "mean_out_tokens": float(np.mean([r["out_tokens"] for r in recs])),
             "latency_p50_s": float(np.percentile(lat, 50)), "latency_p95_s": float(np.percentile(lat, 95))}

    name = f"llm-{model}-{mode}-{shots_k}shot"
    out = REPORTS / "llm"
    out.mkdir(parents=True, exist_ok=True)
    if mode == "closed":
        result = {"name": name, **full_report(eval_df.label.values, pred, labels, eval_df.lang.values), "usage": usage}
        plot_confusion(eval_df.label.values, pred, labels, out / f"{name}_confusion.png", f"{model} {shots_k}-shot — test")
        print(f"[llm] {name}: test macro-F1 {result['macro_f1']:.3f} acc {result['accuracy']:.3f} · non-EN acc "
              f"{result['slices']['non_english']['accuracy']:.3f} · p50 {usage['latency_p50_s']:.2f}s · "
              f"tokens in/out {usage['mean_in_tokens']:.0f}/{usage['mean_out_tokens']:.0f}")
    else:
        is_unk = eval_df.label.isin(HOLDOUT).values
        known = ~is_unk
        result = {"name": name, "rejection_recall": float((pred[is_unk] == "unknown").mean()),
                  "rejection_recall_lenient": float(np.isin(pred[is_unk], ["unknown", "other", "chitchat"]).mean()),
                  "retention": float((pred[known] != "unknown").mean()),
                  "known_accuracy": float((pred[known] == eval_df.label.values[known]).mean()),
                  "per_holdout_class": {c: float((pred[eval_df.label.values == c] == "unknown").mean()) for c in HOLDOUT},
                  "unknowns_routed_to": pd.Series(pred[is_unk]).value_counts().to_dict(), "usage": usage}
        print(f"[llm] {name}: rejection recall {result['rejection_recall']:.3f} (lenient "
              f"{result['rejection_recall_lenient']:.3f}) · retention {result['retention']:.3f} · known acc "
              f"{result['known_accuracy']:.3f}")
    (out / f"{name}.json").write_text(json.dumps(result, indent=2, default=float))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--rpm", type=float, default=8.0, help="request pacing (free tier ~10 rpm for flash)")
    ap.add_argument("--mode", choices=["closed", "open"], default="closed")
    ap.add_argument("--shots", type=int, default=0)
    a = ap.parse_args()
    if a.list_models:
        list_models()
    else:
        run(a.model, a.mode, a.shots, a.rpm)


if __name__ == "__main__":
    main()
