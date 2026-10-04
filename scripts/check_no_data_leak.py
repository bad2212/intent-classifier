"""Pre-push guard: the dataset is confidential and the repo is public.
Fails (exit 1) if any file that git would commit contains
  * a dataset message — whole (normalised) or any verbatim 7-word window of one,
  * per-row dataset data — 5 or more distinct dataset row ids (split/fold/label/prediction tables),
  * any term listed (one regex per line) in the local, never-committed file `.leak_terms`.

Usage: python scripts/check_no_data_leak.py      (run before every push)
"""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BINARY = {".png", ".jpg", ".pdf", ".npz", ".safetensors", ".bin", ".pt", ".ico", ".wandb"}
WINDOW = 7
MAX_ROW_IDS = 4
TERMS_FILE = ROOT / ".leak_terms"   # local only (listed in .git/info/exclude); not part of the repo


def forbidden_terms() -> re.Pattern | None:
    if not TERMS_FILE.exists():
        print(f"warning: {TERMS_FILE.name} not found; checking dataset text and row ids only")
        return None
    terms = [ln.strip() for ln in TERMS_FILE.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return re.compile("|".join(terms), re.IGNORECASE) if terms else None


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def main() -> int:
    texts = pd.read_csv(ROOT / "data" / "raw" / "dataset.csv").text.astype(str).map(norm)
    full = [t for t in texts if len(t) >= 15]
    windows = {" ".join(w[i:i + WINDOW]) for t in texts for w in [t.split()] for i in range(len(w) - WINDOW + 1)}
    ids = set(pd.read_csv(ROOT / "data" / "raw" / "dataset.csv").id.astype(str))
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    forbidden = forbidden_terms()
    hits = []
    for f in files:
        p = ROOT / f
        if p.suffix.lower() in BINARY or not p.is_file():
            continue
        try:
            body = norm(p.read_text(errors="ignore"))
        except OSError:
            continue
        found = [t for t in full if t in body]
        n_ids = len({m for m in re.findall(r"[a-z]+-\d{3,}", body)} & {i.lower() for i in ids})
        if n_ids > MAX_ROW_IDS:
            found.append(f"<{n_ids} dataset row ids: per-row data>")
        if forbidden:
            found += [f"<listed term: {m}>" for m in set(forbidden.findall(body))]
        words = body.split()
        found += [w for w in {" ".join(words[i:i + WINDOW]) for i in range(len(words) - WINDOW + 1)} & windows]
        if found:
            hits.append((f, sorted(set(found))[:3], len(set(found))))
    for f, ex, n in hits:
        print(f"LEAK {f}: {n} match(es), e.g. {ex}")
    print(f"scanned {len(files)} files: {'FAIL' if hits else 'OK, no dataset text found'}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
