"""Re-log the CV bake-off into the PUBLIC W&B project from the saved results (reports/cv/*.json).

Why: the bake-off ran while logging to the team entity `badalthakur2212-iisc`, which is private-only
(privateOnly=true, cannot be made public). Deliverable runs go to the personal entity `badalthakur2212`
(public by default); this script mirrors the bake-off there so the public project is complete.
Each config becomes one run (group "cv-bakeoff") with its per-epoch pooled out-of-fold curve; the TF-IDF baseline
and the bake-off summary table are logged too. Idempotent: runs that already exist (same name) are skipped.
No message text is logged.

Usage: python scripts/relog_cv_to_wandb.py [--entity badalthakur2212] [--project intent-classifier]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "reports" / "cv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="badalthakur2212-iisc")
    ap.add_argument("--project", default="intent-classifier")
    a = ap.parse_args()
    try:
        existing = {r.name for r in wandb.Api().runs(f"{a.entity}/{a.project}")}
    except (ValueError, wandb.errors.CommError):  # project does not exist yet
        existing = set()

    for f in sorted(CV.glob("cv-*.json")):
        name = f"{f.stem} (bake-off)"
        if name in existing:
            print(f"[skip] {name}")
            continue
        s = json.loads(f.read_text())
        run = wandb.init(entity=a.entity, project=a.project, name=name, group="cv-bakeoff", job_type="cv",
                         config={**s["config"], "source": f"reports/cv/{f.name}", "relogged": True}, reinit="finish_previous")
        for c in s.get("curve", []):
            run.log({f"cv/{k}": v for k, v in c.items() if k != "epoch"}, step=int(c["epoch"]))
        run.summary.update({f"best/{k}": v for k, v in s["best"].items() if v is not None})
        pc = pd.DataFrame(s.get("per_class_at_best", {})).T.reset_index(names="class")
        if len(pc):
            run.log({"cv/per_class_at_best": wandb.Table(dataframe=pc.round(4))})
        run.finish()
        print(f"[logged] {name}")

    summ = CV / "bakeoff_summary.json"
    if summ.exists() and "bake-off summary" not in existing:
        s = json.loads(summ.read_text())
        run = wandb.init(entity=a.entity, project=a.project, name="bake-off summary", group="cv-bakeoff",
                         job_type="summary", config={"rule": s["rule"], "winner": s["winner"]}, reinit="finish_previous")
        df = pd.DataFrame(s["configs"])
        run.log({"bakeoff/table": wandb.Table(dataframe=df.round(4))})
        run.summary.update({"winner": s["winner"], "best_raw": s["best_raw"]})
        run.finish()
        print("[logged] bake-off summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
