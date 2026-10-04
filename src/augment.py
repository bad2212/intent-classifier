"""Stretch S4 / S6: data augmentation and the translated stress-test set.

noise_copies(text)   chat-style noise: adjacent-char swaps, drops, duplications, lowercasing, punctuation loss,
                     abbreviations (please->pls, what's->whats, hours->hrs, you->u, thanks->thx). Deterministic per seed.
translate (CLI)      offline MarianMT (Helsinki-NLP opus-mt) translations of every ENGLISH row into es/fr/de/zh.
                     Stored keyed by source id (data/aux/translations.csv, git-ignored: derived from confidential text).
                     Used two ways, never mixed:
                       * augmentation: a translation enters training only if its SOURCE row is in that training set
                         (decisions.md D-23) - never in val/test/held-out folds
                       * S6 stress test: translations of TEST rows, evaluated separately from the real non-English slice

Usage: python -m src.augment translate      (downloads one MarianMT model at a time and deletes it after use)
"""
from __future__ import annotations

import random
import re
import shutil
import sys

import pandas as pd

from src.common import ID_RE, PREFIX_CLASS, ROOT, load_with_split

TRANSLATIONS = ROOT / "data" / "aux" / "translations.csv"
MT_MODELS = {"es": "Helsinki-NLP/opus-mt-en-es", "fr": "Helsinki-NLP/opus-mt-en-fr",
             "de": "Helsinki-NLP/opus-mt-en-de", "zh": "Helsinki-NLP/opus-mt-en-zh"}
ABBREV = [(r"\bplease\b", "pls"), (r"\bwhat's\b", "whats"), (r"\bwhat is\b", "whats"), (r"\bhours\b", "hrs"),
          (r"\byou\b", "u"), (r"\bthanks\b", "thx"), (r"\bare\b", "r"), (r"\bbecause\b", "bc"), (r"\btomorrow\b", "tmrw")]


def noisy(text: str, rng: random.Random, p_char: float = 0.04) -> str:
    t = text
    for pat, rep in ABBREV:
        if rng.random() < 0.5:
            t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    if rng.random() < 0.5:
        t = t.lower()
    if rng.random() < 0.3:
        t = re.sub(r"[?.!,]", "", t)
    chars = list(t)
    out, i = [], 0
    while i < len(chars):
        c = chars[i]
        r = rng.random()
        if c.isalpha() and r < p_char / 3 and i + 1 < len(chars):      # swap with next
            out += [chars[i + 1], c]; i += 2; continue
        if c.isalpha() and r < 2 * p_char / 3:                          # drop
            i += 1; continue
        if c.isalpha() and r < p_char:                                  # duplicate
            out += [c, c]; i += 1; continue
        out.append(c); i += 1
    return "".join(out) or text


def noise_copies(df: pd.DataFrame, copies: int, seed: int) -> pd.DataFrame:
    """`copies` noisy variants per row, keyed by source id (`src_id`), same label."""
    rng = random.Random(seed)
    rows = [{"src_id": r.id, "text": noisy(r.text, rng), "label": r.label, "lang": r.lang}
            for r in df.itertuples() for _ in range(copies)]
    return pd.DataFrame(rows)


def idswap_copies(df: pd.DataFrame, seed: int, min_words: int = 6) -> pd.DataFrame:
    """Counterfactual augmentation against the ID-prefix shortcut: for messages with an entity ID and enough context
    (>= min_words words), add a copy whose ID prefix is swapped to a random OTHER type, keeping the label. Short
    messages where the ID is the only cue ("LD-123 status") are left alone (there the prefix is legitimately informative)."""
    rng = random.Random(seed)
    rows = []
    for r in df.itertuples():
        m = ID_RE.search(r.text)
        if not m or len(r.text.split()) < min_words:
            continue
        new = rng.choice([p for p in PREFIX_CLASS if p != m.group(1)])
        rows.append({"src_id": r.id, "text": ID_RE.sub(lambda mm: f"{new}-{mm.group(2)}", r.text),
                     "label": r.label, "lang": r.lang})
    return pd.DataFrame(rows, columns=["src_id", "text", "label", "lang"])


def translation_copies(df: pd.DataFrame) -> pd.DataFrame:
    """Translations whose SOURCE row is in df (the current training set) — the D-23 leakage rule."""
    if not TRANSLATIONS.exists():
        raise FileNotFoundError("run `python -m src.augment translate` first")
    tr = pd.read_csv(TRANSLATIONS)
    tr = tr[tr.src_id.isin(df.id)].merge(df[["id", "label"]], left_on="src_id", right_on="id").drop(columns="id")
    return tr[["src_id", "text", "label", "lang"]]


def translate_all() -> None:
    from transformers import MarianMTModel, MarianTokenizer
    from huggingface_hub import scan_cache_dir

    df = load_with_split()
    en = df[df.lang == "en"]
    done = pd.read_csv(TRANSLATIONS) if TRANSLATIONS.exists() else pd.DataFrame(columns=["src_id", "lang", "text", "split"])
    for lang, name in MT_MODELS.items():
        if lang in set(done.lang):
            print(f"[mt] {lang} already done"); continue
        tok, model = MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name).eval()
        out = []
        for i in range(0, len(en), 32):
            batch = en.iloc[i:i + 32]
            gen = model.generate(**tok(batch.text.tolist(), return_tensors="pt", padding=True, truncation=True),
                                 num_beams=4, max_new_tokens=96)
            out += tok.batch_decode(gen, skip_special_tokens=True)
        done = pd.concat([done, pd.DataFrame({"src_id": en.id.values, "lang": lang, "text": out,
                                              "split": en.split.values})])
        TRANSLATIONS.parent.mkdir(parents=True, exist_ok=True)
        done.to_csv(TRANSLATIONS, index=False)
        print(f"[mt] {lang}: {len(out)} rows · e.g. {en.text.iloc[0]!r} -> {out[0]!r}")
        del model
        # disk is tight: drop this MT model from the HF cache before downloading the next one
        for repo in scan_cache_dir().repos:
            if repo.repo_id == name:
                shutil.rmtree(repo.repo_path, ignore_errors=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["translate"]:
        translate_all()
    else:
        print(__doc__)
