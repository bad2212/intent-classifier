"""Exploratory data analysis: run BEFORE splitting/modeling.

Outputs:
  reports/eda/summary.md            human-readable findings
  reports/eda/*.png                 figures
  reports/eda/near_duplicates.csv   candidate leakage pairs (same- and cross-lingual)
  data/splits/meta.csv              id -> detected language (used for slice metrics)

Usage: python -m src.eda
"""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lingua import Language, LanguageDetectorBuilder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.common import META_CSV, REPORTS, load_raw, set_seed

OUT = REPORTS / "eda"
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOKENIZER = "FacebookAI/xlm-roberta-base"
# Deliberately broad: a first pass restricted to FR/ES/ZH found German, Italian,
# Dutch and Arabic rows being forced into the wrong language.
LANGS = {
    Language.ENGLISH: "en", Language.FRENCH: "fr", Language.SPANISH: "es", Language.CHINESE: "zh",
    Language.GERMAN: "de", Language.ITALIAN: "it", Language.DUTCH: "nl", Language.PORTUGUESE: "pt",
    Language.ARABIC: "ar", Language.JAPANESE: "ja", Language.KOREAN: "ko", Language.RUSSIAN: "ru",
    Language.HINDI: "hi",
}
MIN_ALPHA_WORDS = 2  # fewer alphabetic words (e.g. emoji, "thx!!") -> language undetermined
LOW_CONF_NON_EN = 0.35  # ID-heavy English ("LD-55501 eta") gets low-confidence foreign guesses -> English
# Manual audit of all rows with confidence < 0.7 (reports/eda/summary.md) left one residual error:
LANG_OVERRIDES = {"syn-0471": "nl"}  # a Dutch question that lingua labelled English
SLANG = r"\b(?:pls|plz|whats|aight|hrs|thx|u|ur|lol|haha|gonna|wanna|asap|btw|idk|rn)\b"


def detect_languages(texts: pd.Series) -> pd.DataFrame:
    """Top language + confidence per row. Very short / ID-like rows are 'und' (undetermined):
    lingua is unreliable there and these rows carry no language signal anyway.
    Note: lingua's multi-language segmentation was tried and flagged fluent English as mixed,
    so code-switching is not auto-detected."""
    detector = LanguageDetectorBuilder.from_languages(*LANGS).build()
    rows = []
    for t in texts:
        top = detector.compute_language_confidence_values(t)[0]
        n_alpha = len(re.findall(r"[^\W\d_]{2,}", t))
        non_latin = bool(re.search(r"[\u0600-\u06ff\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff]", t))
        lang = LANGS[top.language] if (n_alpha >= MIN_ALPHA_WORDS or non_latin) else "und"
        if lang not in ("en", "und") and not non_latin and top.value < LOW_CONF_NON_EN:
            lang = "en"
        rows.append({"lang": lang, "lang_conf": round(top.value, 3)})
    return pd.DataFrame(rows, index=texts.index)


def near_duplicates(df: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    """Candidate leakage pairs: char-ngram cosine (same language) + embedding cosine (cross-lingual)."""
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(df.text)
    char_sim = cosine_similarity(tfidf)
    emb_sim = emb @ emb.T
    iu = np.triu_indices(len(df), k=1)
    pairs = pd.DataFrame({
        "i": iu[0], "j": iu[1],
        "char_sim": char_sim[iu], "emb_sim": emb_sim[iu],
    })
    pairs = pairs[(pairs.char_sim >= 0.6) | (pairs.emb_sim >= 0.9)].copy()
    for side in ("i", "j"):
        pairs[f"id_{side}"] = df.id.values[pairs[side]]
        pairs[f"label_{side}"] = df.label.values[pairs[side]]
        pairs[f"lang_{side}"] = df.lang.values[pairs[side]]
        pairs[f"text_{side}"] = df.text.values[pairs[side]]
    pairs["cross_lingual"] = pairs.lang_i != pairs.lang_j
    pairs["same_label"] = pairs.label_i == pairs.label_j
    return pairs.drop(columns=["i", "j"]).sort_values("emb_sim", ascending=False)


def main() -> None:
    set_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    order = df.label.value_counts().index.tolist()

    # --- basic stats ---
    df["n_chars"] = df.text.str.len()
    df["n_words"] = df.text.str.split().str.len()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    df["n_tokens"] = [len(tok(t)["input_ids"]) for t in df.text]
    df["slang"] = df.text.str.contains(SLANG, case=False, regex=True).astype(bool)
    df["lower_start"] = df.text.str.match(r"^[a-z]")

    # --- language ---
    df = df.join(detect_languages(df.text))
    df.loc[df.id.isin(LANG_OVERRIDES), "lang"] = df.id.map(LANG_OVERRIDES)
    df[["id", "lang", "lang_conf"]].to_csv(META_CSV, index=False)

    # --- embeddings for cross-lingual near-dup check ---
    from sentence_transformers import SentenceTransformer

    emb = SentenceTransformer(EMB_MODEL, device="cpu").encode(
        df.text.tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    dups = near_duplicates(df, emb)
    dups.to_csv(OUT / "near_duplicates.csv", index=False)

    # --- sibling-class semantic overlap: mean cosine between class centroids ---
    labels = sorted(df.label.unique())
    cent = np.stack([emb[df.label.values == l].mean(0) for l in labels])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    cent_sim = pd.DataFrame(cent @ cent.T, index=labels, columns=labels)

    # --- figures ---
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.countplot(y=df.label, order=order, ax=ax, color="#3b6ea5")
    ax.set_title("Rows per class"); ax.set_xlabel("count"); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(OUT / "class_counts.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, y="label", x="n_chars", order=order, ax=ax, color="#9cc3e6")
    ax.set_title("Text length (characters) by class"); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(OUT / "length_by_class.png", dpi=150); plt.close(fig)

    lang_ct = pd.crosstab(df.label, df.lang).reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lang_ct.plot.barh(stacked=True, ax=ax, colormap="tab10")
    ax.invert_yaxis(); ax.set_title("Detected language by class"); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(OUT / "lang_by_class.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cent_sim, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=False, annot_kws={"size": 7})
    ax.set_title("Class-centroid cosine similarity (pretrained multilingual embeddings)")
    fig.tight_layout(); fig.savefig(OUT / "class_similarity.png", dpi=150); plt.close(fig)

    # --- summary ---
    non_en = df[~df.lang.isin(["en", "und"])]
    sib = [l for l in labels if l.startswith("shipment_information.")]
    off_diag = cent_sim.where(~np.eye(len(labels), dtype=bool))
    top_pairs = (off_diag.stack().sort_values(ascending=False)
                 .reset_index().iloc[::2].head(6))  # matrix is symmetric: keep one of each pair
    lines = [
        "# EDA summary", "",
        f"- Rows: {len(df)} · classes: {df.label.nunique()} · null text: {df.text.isna().sum()} · "
        f"exact duplicate texts: {df.text.duplicated().sum()} · case-insensitive duplicates: "
        f"{df.text.str.lower().duplicated().sum()}",
        f"- Imbalance: max/min class = {df.label.value_counts().max()}/{df.label.value_counts().min()} "
        f"= {df.label.value_counts().max() / df.label.value_counts().min():.2f}x",
        f"- Length: median {df.n_chars.median():.0f} chars / {df.n_words.median():.0f} words / "
        f"{df.n_tokens.median():.0f} XLM-R tokens; max {df.n_tokens.max()} tokens "
        f"(p99 {df.n_tokens.quantile(.99):.0f}) → max_length=64 covers all rows",
        f"- Language (lingua): {df.lang.value_counts().to_dict()} → non-English = "
        f"{len(non_en)} rows ({len(non_en) / len(df):.0%}); "
        f"low-confidence (<0.6) detections: {(df.lang_conf < 0.6).sum()}",
        f"- Noise: rows with slang/abbreviations: {df.slang.sum()} ({df.slang.mean():.0%}); "
        f"rows starting lowercase: {df.lower_start.sum()} ({df.lower_start.mean():.0%})",
        f"- Near-duplicate candidates: {len(dups)} pairs (char_sim≥0.6 or emb_sim≥0.9); "
        f"cross-lingual: {int(dups.cross_lingual.sum())}; with different labels: {int((~dups.same_label).sum())}",
        "", "## Language by class", "", lang_ct.to_markdown(),
        "", "## Most similar class pairs (centroid cosine)", "",
        top_pairs.rename(columns={"level_0": "class_a", "level_1": "class_b", 0: "cosine"}).to_markdown(index=False),
        "", "## Sibling-class centroid similarity", "", cent_sim.loc[sib, sib].round(3).to_markdown(),
        "", "## Shortest texts", "",
        df.nsmallest(8, "n_chars")[["id", "text", "label"]].to_markdown(index=False),
        "", "## Low-confidence language detections", "",
        df[df.lang_conf < 0.6].sort_values("lang_conf")[["id", "text", "lang", "lang_conf"]].head(15).to_markdown(index=False),
        "", "## Non-English examples", "",
        non_en.groupby("lang").head(3)[["id", "lang", "text", "label"]].to_markdown(index=False),
        "", "## Top near-duplicate candidates", "",
        dups.head(15)[["id_i", "id_j", "char_sim", "emb_sim", "label_i", "label_j", "text_i", "text_j"]]
        .round(3).to_markdown(index=False),
        "", "## Sibling examples", "",
    ]
    for l in sib:
        lines += [f"**{l}**", ""] + [f"- {t}" for t in df[df.label == l].text.head(5)] + [""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
