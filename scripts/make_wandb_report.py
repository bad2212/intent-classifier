"""Build (or update in place) the W&B Report that reviewers open via a view-only link.

The team entity is private-only (privateOnly=true) and the personal entity cannot host projects on this account, so
the project itself cannot be public -> this report is
shared with "anyone with the link can view" (set in the W&B UI: Share -> view-only link) and reviewers are invited.
It contains: train/val loss, learning rate, val macro-F1 + accuracy per epoch,
test per-class F1/P/R table + confusion matrix, run config — plus the bake-off, Track B and stretch runs.

Idempotent: the report URL is kept in configs/WANDB_REPORT_URL; re-running updates that report (same link).
Usage: python scripts/make_wandb_report.py
"""
import sys
from pathlib import Path

import wandb_workspaces.reports.v2 as wr

ROOT = Path(__file__).resolve().parents[1]
ENTITY, PROJECT = "badalthakur2212-iisc", "intent-classifier"
URL_FILE = ROOT / "configs" / "WANDB_REPORT_URL"
LINKS = {"code": "https://github.com/bad2212/intent-classifier",
         "model": "https://huggingface.co/Badalt/intent-classifier-mpnet"}


def runs(*names: str, title: str = "Run set") -> wr.Runset:
    quoted = ", ".join(f"'{n}'" for n in names)
    return wr.Runset(entity=ENTITY, project=PROJECT, name=title, filters=f"Metric('Name') in [{quoted}]")


def grid(runset: wr.Runset, panels: list) -> wr.PanelGrid:
    return wr.PanelGrid(runsets=[runset], panels=panels)


def blocks() -> list:
    a = runs("model_a", title="Model A (submitted)")
    return [
        wr.TableOfContents(),
        wr.MarkdownBlock(text=(
            "Fine-tuned **paraphrase-multilingual-mpnet-base-v2** (lr 5e-5, seed 42) routing logistics-chat messages to "
            "12 intents, with a calibrated `unknown` reject rule. "
            f"Code: [{LINKS['code']}]({LINKS['code']}) · Model: [{LINKS['model']}]({LINKS['model']})")),
        wr.H1(text="Model A — training curves (Track A model)"),
        grid(a, [
            wr.LinePlot(title="train loss", x="train/global_step", y=["train/loss"]),
            wr.LinePlot(title="validation loss", x="train/epoch", y=["eval/loss"]),
            wr.LinePlot(title="learning rate", x="train/global_step", y=["train/learning_rate"]),
            wr.LinePlot(title="validation macro-F1 (selection metric)", x="train/epoch", y=["eval/macro_f1"]),
            wr.LinePlot(title="validation accuracy", x="train/epoch", y=["eval/accuracy"]),
        ]),
        wr.H1(text="Model A — held-out test set (n=77, evaluated once)"),
        grid(a, [
            wr.ScalarChart(title="test macro-F1", metric="test/macro_f1"),
            wr.ScalarChart(title="test accuracy", metric="test/accuracy"),
            wr.ScalarChart(title="test non-English accuracy", metric="test/non_english_accuracy"),
            wr.WeavePanelSummaryTable(table_name="test/per_class"),
            wr.MediaBrowser(media_keys=["test/confusion_matrix_img"], num_columns=1),
            wr.RunComparer(),
        ]),
        wr.H1(text="Backbone x LR bake-off (5-fold CV, out-of-fold)"),
        wr.P(text="6 configs; winner chosen by the pre-declared rule (best OOF macro-F1; if the top two are within 1 std, "
                  "take the smaller). Per-epoch pooled out-of-fold macro-F1 for every config:"),
        grid(wr.Runset(entity=ENTITY, project=PROJECT, name="bake-off", filters="Metric('Name') in "
                       "['cv-minilm-lr2e-5 (bake-off)', 'cv-minilm-lr5e-5 (bake-off)', 'cv-mpnet-lr2e-5 (bake-off)', "
                       "'cv-mpnet-lr5e-5 (bake-off)', 'cv-xlmr-lr2e-5 (bake-off)', 'cv-xlmr-lr5e-5 (bake-off)', "
                       "'bake-off summary']"),
             [wr.LinePlot(title="pooled OOF macro-F1 by epoch", x="Step", y=["cv/pooled_macro_f1"]),
              wr.LinePlot(title="OOF non-English accuracy by epoch", x="Step", y=["cv/non_english_accuracy"]),
              wr.WeavePanelSummaryTable(table_name="bakeoff/table")]),
        wr.H1(text="Track B — unknown intents (open-set)"),
        wr.P(text="Model B is trained without yard_management + document_processing (far-OOD); Model B-near without "
                  "shipment_information.disruptions (near-OOD). Threshold = 95% retention on known-class validation."),
        grid(runs("model_b", "model_b_near", "model_b_oe", "model_b_near_oe", title="Track B models"), [
            wr.LinePlot(title="validation macro-F1", x="train/epoch", y=["eval/macro_f1"]),
            wr.LinePlot(title="train loss", x="train/global_step", y=["train/loss"]),
        ]),
        grid(runs("model_b-trackB", "model_b_near-trackB", "model_b_oe-trackB", "model_b_near_oe-trackB",
                  title="Track B evaluation"), [
            wr.WeavePanelSummaryTable(table_name="trackB/scorers"),
            wr.MediaBrowser(media_keys=["trackB/retention_vs_rejection", "trackB/msp_hist"], num_columns=2),
        ]),
        wr.H1(text="Stretch experiments"),
        wr.P(text="Seeds (variance of the submitted recipe), ID-swap augmentation (shortcut fix). "),
        grid(runs("model_a", "model_a_s43", "model_a_s44", "model_a_idswap", title="Model A variants"), [
            wr.LinePlot(title="validation macro-F1", x="train/epoch", y=["eval/macro_f1"]),
            wr.RunComparer(diff_only=True),
        ]),
    ]


def main() -> int:
    if URL_FILE.exists():
        report = wr.Report.from_url(URL_FILE.read_text().strip())
        report.blocks = blocks()
    else:
        report = wr.Report(project=PROJECT, entity=ENTITY, title="Multilingual intent router — training & evaluation",
                           description="Track A (known intents) and Track B (unknown intents) results.",
                           width="fluid", blocks=blocks())
    report.save()
    URL_FILE.write_text(report.url + "\n")
    print(f"[wandb report] {report.url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
