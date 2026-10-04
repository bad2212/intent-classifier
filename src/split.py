"""Pinned, stratified, leakage-aware 70/15/15 split.

Near-duplicate paraphrases (pretrained multilingual embedding cosine >= DUP_THRESHOLD, from
reports/eda/near_duplicates.csv) are grouped and kept in the same split, so a paraphrase of a
training row can never inflate the test score. Groups are split stratified by label (all members
of a group share a label — asserted below).

Writes data/splits/split.csv (id, split) and data/splits/folds.csv (id, fold) — 5 stratified,
group-aware CV folds over train+val used for the backbone/LR bake-off. No text: data is confidential.
Usage: python -m src.split   (run src.eda first)
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.common import FOLDS_CSV, META_CSV, REPORTS, SEED, SPLIT_CSV, load_raw

DUP_THRESHOLD = 0.90   # chosen by inspecting near_duplicates.csv: pairs >= 0.90 are true paraphrases,
                       # pairs in 0.79–0.83 are merely same-intent questions (not leakage)
VAL_FRAC, TEST_FRAC = 0.15, 0.15
N_FOLDS = 5


def duplicate_groups(ids: pd.Series) -> pd.Series:
    """Union-find over near-duplicate pairs -> group id per row (singletons keep their own id)."""
    pairs = pd.read_csv(REPORTS / "eda" / "near_duplicates.csv")
    pairs = pairs[pairs.emb_sim >= DUP_THRESHOLD]
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs.id_i, pairs.id_j):
        parent[find(a)] = find(b)
    return ids.map(find)


def main() -> None:
    df = load_raw()
    df["group"] = duplicate_groups(df.id)
    assert (df.groupby("group").label.nunique() == 1).all(), "a duplicate group spans labels"

    groups = df.groupby("group", as_index=False).label.first()
    g_trainval, g_test = train_test_split(
        groups, test_size=TEST_FRAC, stratify=groups.label, random_state=SEED)
    g_train, g_val = train_test_split(
        g_trainval, test_size=VAL_FRAC / (1 - TEST_FRAC), stratify=g_trainval.label, random_state=SEED)

    split_of = {**dict.fromkeys(g_train.group, "train"), **dict.fromkeys(g_val.group, "val"),
                **dict.fromkeys(g_test.group, "test")}
    df["split"] = df.group.map(split_of)
    assert df.split.notna().all()
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[["id", "split"]].to_csv(SPLIT_CSV, index=False)

    # --- pinned CV folds over train+val (test never enters CV) ---
    tv = df[df.split != "test"].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tv["fold"] = -1
    for k, (_, idx) in enumerate(sgkf.split(tv, tv.label, groups=tv.group)):
        tv.loc[idx, "fold"] = k
    assert (tv.groupby("group").fold.nunique() == 1).all()
    tv[["id", "fold"]].to_csv(FOLDS_CSV, index=False)
    print("fold sizes:", tv.fold.value_counts().sort_index().to_dict())
    print("min rows of any class in any fold:", pd.crosstab(tv.label, tv.fold).values.min())

    # --- sanity report ---
    df = df.merge(pd.read_csv(META_CSV), on="id")
    df["non_en"] = ~df.lang.isin(["en", "und"])
    n_groups = (df.group.value_counts() > 1).sum()
    print(f"duplicate groups kept together: {n_groups} ({(df.group.duplicated(keep=False)).sum()} rows)")
    print(df.split.value_counts().to_string(), "\n")
    print(pd.crosstab(df.label, df.split)[["train", "val", "test"]].to_string(), "\n")
    print("non-English rows per split:", df.groupby("split").non_en.sum().to_dict())
    print("languages per split:\n", pd.crosstab(df.lang, df.split)[["train", "val", "test"]].to_string())


if __name__ == "__main__":
    main()
