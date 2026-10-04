"""Collect every stretch experiment into one comparable table (reports/stretch_summary.{json,md}).

CV variants are compared to the submitted recipe (configs/WINNER_CV) PAIRED on the folds the variant ran
(3-fold variants are compared on folds 0-2 of the baseline, not on its 5-fold mean). A variant "helps" only if its
mean paired fold difference is > 0 AND it wins on a majority of folds — with ~±0.03 seed/order noise (G39),
smaller effects are reported as "no detectable difference", not as wins.
Usage: python -m src.stretch_summary [--decide-aug-both]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.common import REPORTS, ROOT
from src.train import OUTPUTS

CV = REPORTS / "cv"


def fold_f1(name: str, folds=None) -> pd.Series:
    oof = pd.read_csv(CV / f"{name}_oof.csv")
    if folds is not None:
        oof = oof[oof.fold.isin(folds)]
    return oof.groupby("fold").apply(lambda g: f1_score(g.label, g.pred, average="macro"), include_groups=False)


def pooled(name: str, folds) -> dict:
    oof = pd.read_csv(CV / f"{name}_oof.csv")
    oof = oof[oof.fold.isin(folds)]
    non_en = ~oof.lang.isin(["en", "und"])
    return {"macro_f1": f1_score(oof.label, oof.pred, average="macro"),
            "non_en_acc": float((oof.pred == oof.label)[non_en].mean()),
            "other_f1": f1_score(oof.label, oof.pred, labels=["other"], average="macro")}


def compare_cv(base: str) -> pd.DataFrame:
    rows = []
    for f in sorted(CV.glob(f"{base}-*.json")):
        name = f.stem
        var_folds = sorted(pd.read_csv(CV / f"{name}_oof.csv").fold.unique())
        d = fold_f1(name) - fold_f1(base, var_folds)
        pv, pb = pooled(name, var_folds), pooled(base, var_folds)
        helps = d.mean() > 0 and (d > 0).sum() > len(d) / 2
        rows.append({"variant": name.removeprefix(base + "-"), "folds": len(var_folds),
                     "macro_f1": pv["macro_f1"], "baseline_same_folds": pb["macro_f1"],
                     "mean_paired_diff": float(d.mean()), "folds_better": f"{int((d > 0).sum())}/{len(d)}",
                     "non_en_acc": pv["non_en_acc"], "baseline_non_en_acc": pb["non_en_acc"],
                     "other_f1": pv["other_f1"], "baseline_other_f1": pb["other_f1"],
                     "verdict": "helps" if helps else ("hurts" if d.mean() < -0.01 and (d < 0).sum() > len(d) / 2
                                                       else "no detectable difference")})
    return pd.DataFrame(rows)


def trackb_compare() -> list[dict]:
    out = []
    for base, oe in (("model_b", "model_b_oe"), ("model_b_near", "model_b_near_oe")):
        if not (REPORTS / "trackB" / oe / "metrics.json").exists():
            continue
        for run in (base, oe):
            m = json.loads((REPORTS / "trackB" / run / "metrics.json").read_text())
            for sc in m["scorers"]:
                if sc["scorer"] in ("MSP (baseline)", "energy", "Mahalanobis [cls]", "kNN k=5 [cls]"):
                    out.append({"run": run, "scorer": sc["scorer"], "AUROC": sc["AUROC"],
                                "test_retention": sc["test_retention"], "rejection_recall": sc["rejection_recall_strict"],
                                "known_macro_f1": m["known_test_10way"]["macro_f1"]})
    return out


def seeds() -> dict:
    runs = {r: json.loads((OUTPUTS / r / "metrics.json").read_text())["test"]
            for r in ("model_a", "model_a_s43", "model_a_s44") if (OUTPUTS / r / "metrics.json").exists()}
    if len(runs) < 2:
        return {}
    f1 = [v["macro_f1"] for v in runs.values()]
    return {"runs": {k: {"macro_f1": v["macro_f1"], "accuracy": v["accuracy"]} for k, v in runs.items()},
            "macro_f1_mean": float(np.mean(f1)), "macro_f1_std": float(np.std(f1, ddof=1)),
            "note": "seed 42 is the declared submission; other seeds only quantify training variance (G10)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decide-aug-both", action="store_true")
    a = ap.parse_args()
    base = (ROOT / "configs" / "WINNER_CV").read_text().strip()
    cv = compare_cv(base)
    if a.decide_aug_both:
        aug = cv[cv.variant.isin(["aug-noise", "aug-translate-f3"])]
        print("RUN" if (aug.verdict == "helps").any() else "SKIP: neither noise nor translation helps on its own")
        return
    probe = {r: json.loads((REPORTS / "trackA" / r / "shortcut_probe.json").read_text())
             for r in ("model_a", "model_a_idswap") if (REPORTS / "trackA" / r / "shortcut_probe.json").exists()}
    al = REPORTS / "active_learning" / "summary.json"
    llm = {f.stem: json.loads(f.read_text()) for f in sorted((REPORTS / "llm").glob("*.json"))}
    summary = {"baseline_cv": base, "cv_variants": cv.to_dict(orient="records"), "trackB_oe": trackb_compare(),
               "seeds": seeds(), "shortcut_probe": {k: {kk: v[kk] for kk in ("counterfactual_flip_rate",
                                                     "counterfactual_flip_to_new_prefix_class", "test_only_flip_rate")}
                                                    for k, v in probe.items()},
               "active_learning": json.loads(al.read_text()) if al.exists() else None,
               "llm": {k: {kk: v.get(kk) for kk in ("macro_f1", "accuracy", "rejection_recall", "retention",
                                                    "known_accuracy", "usage")} for k, v in llm.items()}}
    (REPORTS / "stretch_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    md = ["## CV variants (paired vs submitted recipe)", "", cv.round(3).to_markdown(index=False) if len(cv) else "_none yet_"]
    if summary["trackB_oe"]:
        md += ["", "## Track B with / without Outlier Exposure", "", pd.DataFrame(summary["trackB_oe"]).round(3).to_markdown(index=False)]
    (REPORTS / "stretch_summary.md").write_text("\n".join(md))
    print("\n".join(md))
    for k in ("seeds", "shortcut_probe", "llm"):
        print(f"\n{k}: {json.dumps(summary[k], indent=1, default=float)[:1500]}")


if __name__ == "__main__":
    main()
