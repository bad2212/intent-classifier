"""Stretch S9: production drift monitor for the intent router.

Given a reference (the run's own train/val features) and a batch of production messages, report one two-sample
test per signal and alert at p < ALPHA (0.01). PSI is kept as an effect-size number only: a fixed "PSI > 0.2" rule
fired on in-distribution test traffic at n=65 (decisions.md G41) because PSI is noisy on small windows.
  * confidence drift      KS test: calibrated max-prob of the batch vs validation
  * unknown rate          one-sided binomial test vs the expected 1 - retention_target (5%)
  * intent-mix drift      chi-square of predicted intents vs the train prediction mix (Monte-Carlo null, small counts)
  * language-mix drift    two-sided binomial test of the non-English share vs train
  * embedding drift       MMD (RBF kernel, mean-pooled features) vs train, permutation p-value

Demo (no labels needed in production): `python -m src.monitor --run model_b` compares
  (1) known-intent test traffic  -> expected: no alerts
  (2) traffic containing the held-out (never-seen) intents -> expected: alerts fire
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import binomtest, ks_2samp

from src.common import REPORTS, SEED
from src.train import OUTPUTS

ALPHA = 0.01   # family-wise: each of the N_TESTS signals alerts at ALPHA / N_TESTS (Bonferroni)
N_TESTS = 5


def psi(ref: np.ndarray, cur: np.ndarray, bins: np.ndarray) -> float:
    r = np.histogram(ref, bins)[0] / len(ref) + 1e-4
    c = np.histogram(cur, bins)[0] / len(cur) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref) | set(cur))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).values + 1e-4
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).values + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


def chi2_mc(ref: pd.Series, cur: pd.Series, n_sim: int = 5000, seed: int = SEED) -> tuple[float, float]:
    """Chi-square statistic of `cur` counts against `ref` proportions; p-value from multinomial simulation
    (valid for the small expected counts of a 65-message window)."""
    cats = sorted(set(ref) | set(cur))
    p = ref.value_counts(normalize=True).reindex(cats, fill_value=0).values + 1e-6
    p /= p.sum()
    obs = cur.value_counts().reindex(cats, fill_value=0).values
    exp = p * len(cur)
    stat = float(((obs - exp) ** 2 / exp).sum())
    sims = np.random.default_rng(seed).multinomial(len(cur), p, size=n_sim)
    null = ((sims - exp) ** 2 / exp).sum(1)
    return stat, float((np.sum(null >= stat) + 1) / (n_sim + 1))


def mmd_rbf(x: np.ndarray, y: np.ndarray, n_perm: int = 2000, seed: int = SEED) -> tuple[float, float]:
    """Unbiased MMD^2 with median-heuristic bandwidth + permutation p-value.
    n_perm must make the smallest attainable p, 1/(n_perm+1), fall below the alert level ALPHA/N_TESTS = 0.002;
    with 300 permutations (min p 0.0033) this test could never fire (decisions.md G41)."""
    z = np.vstack([x, y])
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    k = np.exp(-d2 / np.median(d2[d2 > 0]))
    n = len(x)

    def stat(idx):
        a, b = idx[:n], idx[n:]
        kxx, kyy, kxy = k[np.ix_(a, a)], k[np.ix_(b, b)], k[np.ix_(a, b)]
        return ((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (len(b) * (len(b) - 1))
                - 2 * kxy.mean())

    rng = np.random.default_rng(seed)
    obs = stat(np.arange(len(z)))
    perms = [stat(rng.permutation(len(z))) for _ in range(n_perm)]
    return float(obs), float((np.sum(np.array(perms) >= obs) + 1) / (n_perm + 1))


def drift_report(run: str, batch: dict, name: str) -> dict:
    meta = json.loads((OUTPUTS / run / "metrics.json").read_text())
    T, tau = meta["calibration"]["calibration_temperature"], meta["calibration"]["unknown_threshold"]
    target = meta["config"]["retention_target"]
    ref_tr = dict(np.load(OUTPUTS / run / "features_train.npz", allow_pickle=True))
    ref_va = dict(np.load(OUTPUTS / run / "features_val.npz", allow_pickle=True))
    from transformers import AutoConfig

    id2label = {int(k): v for k, v in AutoConfig.from_pretrained(OUTPUTS / run / "model").id2label.items()}

    conf_ref = softmax(ref_va["logits"] / T, axis=1).max(1)
    conf_cur = softmax(batch["logits"] / T, axis=1).max(1)
    pred_ref = pd.Series([id2label[i] for i in ref_tr["logits"].argmax(1)])
    pred_cur = pd.Series([id2label[i] for i in batch["logits"].argmax(1)])
    is_non_en = lambda langs: ~np.isin(langs, ["en", "und"])
    n = len(conf_cur)
    n_unknown = int((conf_cur < tau).sum())
    chi2, p_mix = chi2_mc(pred_ref, pred_cur)
    lang_ref = float(is_non_en(ref_tr["langs"]).mean())
    mmd, p_mmd = mmd_rbf(ref_tr["mean"], batch["mean"])
    r = {
        "batch": name, "n": int(n),
        "confidence_ks_p": float(ks_2samp(conf_cur, conf_ref).pvalue),
        "confidence_psi": psi(conf_ref, conf_cur, np.linspace(0, 1, 11)),
        "unknown_rate": n_unknown / n, "unknown_rate_expected": round(1 - target, 3),
        "unknown_rate_p": float(binomtest(n_unknown, n, 1 - target, alternative="greater").pvalue),
        "intent_mix_chi2": chi2, "intent_mix_p": p_mix, "intent_mix_psi": psi_categorical(pred_ref, pred_cur),
        "non_english_share": float(is_non_en(batch["langs"]).mean()), "non_english_share_train": lang_ref,
        "language_mix_p": float(binomtest(int(is_non_en(batch["langs"]).sum()), n, lang_ref).pvalue),
        "embedding_mmd2": mmd, "embedding_mmd_p": p_mmd,
    }
    r["alerts"] = [a for a, key in [("confidence drift", "confidence_ks_p"), ("unknown rate", "unknown_rate_p"),
                                    ("intent mix drift", "intent_mix_p"), ("language mix drift", "language_mix_p"),
                                    ("embedding drift", "embedding_mmd_p")] if r[key] < ALPHA / N_TESTS]
    return r


def power_study(run: str, te: dict, un: dict, sizes=(65, 130, 260, 520), share: float = 0.4,
                reps: int = 40) -> pd.DataFrame:
    """How large must a monitoring window be? For each window size, simulate `reps` windows of clean known-intent
    traffic (-> false-alarm rate) and of traffic with `share` never-seen intents (-> detection rate).
    Windows are bootstrap-resampled from the 65 known test rows / 80 held-out rows. Caveat: the small known pool
    genuinely differs a little from the training reference (e.g. non-English share, unknown rate), and large
    resampled windows detect that real-but-tiny difference -> clean false-alarm rates at n>=260 are pessimistic."""
    rng = np.random.default_rng(SEED)
    rows = []
    for n in sizes:
        for cond in ("clean", f"{int(share * 100)}% unknown"):
            fired = {k: 0 for k in ("confidence drift", "unknown rate", "intent mix drift", "language mix drift",
                                    "embedding drift", "any")}
            for _ in range(reps):
                n_unk = 0 if cond == "clean" else int(share * n)
                ki = rng.integers(0, len(te["ids"]), n - n_unk)
                ui = rng.integers(0, len(un["ids"]), n_unk)
                batch = {k: np.concatenate([te[k][ki], un[k][ui]]) for k in ("logits", "mean", "langs")}
                alerts = drift_report(run, batch, cond)["alerts"]
                for a_ in alerts:
                    fired[a_] += 1
                fired["any"] += bool(alerts)
            rows.append({"window": n, "traffic": cond, **{k: v / reps for k, v in fired.items()}})
            print(f"[power] n={n:3d} {cond:12s} any-alert rate {fired['any'] / reps:.2f}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="model_b")
    ap.add_argument("--power", action="store_true", help="also run the window-size power study")
    a = ap.parse_args()
    te = dict(np.load(OUTPUTS / a.run / "features_test.npz", allow_pickle=True))
    un = dict(np.load(OUTPUTS / a.run / "features_unknown.npz", allow_pickle=True))
    rng = np.random.default_rng(SEED)
    # batch 2: same size as the known batch, ~40% of it from never-seen intents (a new capability rolling out)
    n_unk = int(0.4 * len(te["ids"]))
    keep = rng.choice(len(te["ids"]), len(te["ids"]) - n_unk, replace=False)
    pick = rng.choice(len(un["ids"]), n_unk, replace=False)
    mixed = {k: np.concatenate([te[k][keep], un[k][pick]]) for k in ("logits", "mean", "langs")}
    reports = [drift_report(a.run, te, "known-intent traffic (test)"),
               drift_report(a.run, mixed, f"traffic with 40% never-seen intents")]
    out = REPORTS / "monitor"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.run}_drift_demo.json").write_text(json.dumps(reports, indent=2, default=float))
    print(pd.DataFrame(reports).set_index("batch").T.to_string())
    if a.power:
        pw = power_study(a.run, te, un)
        pw.to_csv(out / f"{a.run}_power.csv", index=False)
        print(pw.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
