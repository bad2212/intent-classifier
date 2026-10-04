"""Shared paths, seeding, and data loading."""
from __future__ import annotations

import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "dataset.csv"
SPLIT_CSV = ROOT / "data" / "splits" / "split.csv"   # id -> train/val/test (no text: data is confidential)
META_CSV = ROOT / "data" / "splits" / "meta.csv"     # id -> detected language
FOLDS_CSV = ROOT / "data" / "splits" / "folds.csv"   # id -> CV fold (train+val only)
REPORTS = ROOT / "reports"
SEED = 42

SIBLING_PARENT = "shipment_information"

# Entity IDs in messages and the class each ID type co-occurs with (81% of ID messages carry that label ->
# a learnable shortcut, see src/shortcut_probe.py). Not "\b": in "订单PO-77123" Unicode \b sees no boundary.
ID_RE = re.compile(r"(?<![A-Za-z])(APT|PO|DOC|TR|LD)-(\d+)(?!\d)")
PREFIX_CLASS = {"APT": "appointment_manager", "PO": "orders", "DOC": "document_processing",
                "TR": "yard_management", "LD": "shipment_information.realtime_query"}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    df["text"] = df["text"].astype(str).str.strip()
    return df


def load_with_split() -> pd.DataFrame:
    """Raw rows joined with the pinned split and detected language."""
    df = load_raw().merge(pd.read_csv(SPLIT_CSV), on="id", how="inner", validate="one_to_one")
    if META_CSV.exists():
        df = df.merge(pd.read_csv(META_CSV), on="id", how="left", validate="one_to_one")
    return df


def label_maps(labels) -> tuple[dict[str, int], dict[int, str]]:
    """Deterministic label <-> id mapping (sorted)."""
    names = sorted(set(labels))
    label2id = {n: i for i, n in enumerate(names)}
    return label2id, {i: n for n, i in label2id.items()}


def parent_of(label: str) -> str:
    """Coarse label: collapses shipment_information.* siblings into one parent."""
    return label.split(".")[0]
