"""Unit checks for the logic most likely to fail silently (plan.md §4.5). Run: python -m pytest -q tests"""
import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.common import FOLDS_CSV, SPLIT_CSV, label_maps, load_with_split


def test_splits_disjoint_and_folds_exclude_test():
    s = pd.read_csv(SPLIT_CSV)
    assert s.id.is_unique and set(s.split) == {"train", "val", "test"}
    f = pd.read_csv(FOLDS_CSV)
    assert not set(f.id) & set(s[s.split == "test"].id)
    assert set(f.id) == set(s[s.split != "test"].id)


def test_near_duplicate_groups_never_cross_splits():
    from src.split import duplicate_groups

    df = load_with_split()
    df["group"] = duplicate_groups(df.id)
    assert (df.groupby("group").split.nunique() == 1).all()


def test_label_map_round_trip():
    l2i, i2l = label_maps(["b", "a", "c", "a"])
    assert l2i == {"a": 0, "b": 1, "c": 2} and all(i2l[l2i[k]] == k for k in l2i)


def _trainer(**kw):
    from src.train import WeightedTrainer

    t = WeightedTrainer.__new__(WeightedTrainer)  # loss logic only, no HF Trainer setup
    t.class_weights, t.label_smoothing = kw.get("w"), 0.0
    t.hier_lambda, t.oe_lambda = kw.get("hier", 0.0), kw.get("oe", 0.0)
    pc = kw.get("parent")
    t.parent_of_class = pc
    if pc is not None:
        m = torch.zeros(int(pc.max()) + 1, len(pc), dtype=torch.bool)
        m[pc, torch.arange(len(pc))] = True
        t.parent_mask = m
    return t


class _Model:
    def __init__(self, z):
        self.z = z

    def __call__(self, **_):
        return type("O", (), {"logits": self.z})()


def test_loss_equals_plain_ce_when_extras_off():
    z, y = torch.randn(6, 4), torch.tensor([0, 1, 2, 3, 1, 0])
    loss = _trainer().compute_loss(_Model(z), {"labels": y})
    assert torch.allclose(loss, torch.nn.functional.cross_entropy(z, y))


def test_parent_logprob_is_log_sum_of_children():
    z = torch.randn(5, 4)
    parent = torch.tensor([0, 0, 0, 1])  # classes 0-2 are siblings
    t = _trainer(hier=1.0, parent=parent)
    y = torch.tensor([0, 1, 2, 3, 0])
    loss = t.compute_loss(_Model(z), {"labels": y})
    p = torch.softmax(z, -1)
    parent_p = torch.stack([p[:, :3].sum(1), p[:, 3]], 1)
    expected = torch.nn.functional.cross_entropy(z, y) + torch.nn.functional.nll_loss(parent_p.log(), parent[y])
    assert torch.allclose(loss, expected, atol=1e-6)


def test_outlier_exposure_term_minimised_by_uniform():
    k = 5
    t = _trainer(oe=1.0)
    uniform = t.compute_loss(_Model(torch.zeros(3, k)), {"labels": torch.tensor([-1, -1, -1])})
    peaked = t.compute_loss(_Model(torch.tensor([[5.0, 0, 0, 0, 0]] * 3)), {"labels": torch.tensor([-1, -1, -1])})
    assert math.isclose(float(uniform), math.log(k), rel_tol=1e-6) and float(peaked) > float(uniform)


def test_augmentation_only_from_training_rows():
    from src.augment import noise_copies

    df = load_with_split()
    tr = df[df.split == "train"].head(20)
    aug = noise_copies(tr, 2, seed=0)
    assert set(aug.src_id) <= set(tr.id) and len(aug) == 40
    assert (aug.merge(tr[["id", "label"]], left_on="src_id", right_on="id").pipe(lambda d: d.label_x == d.label_y)).all()


def test_auroc_orientation():
    from sklearn.metrics import roc_auc_score

    from src.openset import msp

    known_logits = np.array([[8.0, 0, 0]] * 50)       # confident -> low unknown-score
    unknown_logits = np.array([[0.1, 0, 0.05]] * 50)  # flat -> high unknown-score
    s = np.r_[msp(known_logits), msp(unknown_logits)]
    y = np.r_[np.zeros(50), np.ones(50)]
    assert roc_auc_score(y, s) == pytest.approx(1.0)


def test_retention_threshold_keeps_at_least_target():
    from src.openset import threshold_at_retention
    from src.train import retention_threshold

    rng = np.random.default_rng(0)
    for n in (20, 76, 77):
        p = rng.random(n)
        assert (p >= retention_threshold(p, 0.95)).mean() >= 0.95
        s = rng.random(n)
        assert (s <= threshold_at_retention(s, 0.95)).mean() >= 0.95


def test_idswap_augmentation_only_swaps_prefix_on_training_rows():
    from src.augment import idswap_copies
    from src.common import ID_RE

    df = load_with_split()
    tr = df[df.split == "train"]
    aug = idswap_copies(tr, seed=0)
    src = tr.set_index("id").loc[aug.src_id]
    assert set(aug.src_id) <= set(tr.id) and (aug.label.values == src.label.values).all()
    for new, old in zip(aug.text, src.text):
        m_new, m_old = ID_RE.search(new), ID_RE.search(old)
        assert m_new.group(1) != m_old.group(1) and m_new.group(2) == m_old.group(2)   # prefix changed, digits kept
        assert ID_RE.sub("ID", new) == ID_RE.sub("ID", old)                            # nothing else changed
