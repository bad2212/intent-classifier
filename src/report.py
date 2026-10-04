"""Build reports/report.pdf from reports/report.md: Markdown -> styled print HTML -> headless Chrome PDF.

pandoc/wkhtmltopdf are not installed on the dev machine; Chrome is (decisions.md D-19).
Images are referenced relative to reports/ and embedded as files next to the HTML.
Usage: python -m src.report [name]        (name defaults to "report")
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import markdown

from src.common import REPORTS

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "", shutil.which("chromium") or "",
]

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 10.2pt; line-height: 1.42;
       color: #1d2330; max-width: 100%; }
h1 { font-size: 20pt; margin: 0 0 4pt; color: #12223b; }
h2 { font-size: 13.5pt; margin: 16pt 0 5pt; padding-bottom: 2pt; border-bottom: 1.5px solid #2f6db3; color: #12223b;
     break-after: avoid; }
h3 { font-size: 11pt; margin: 11pt 0 3pt; color: #2f4a70; break-after: avoid; }
p, li { margin: 3pt 0; }
table { border-collapse: collapse; margin: 6pt 0 8pt; font-size: 8.8pt; break-inside: avoid; width: 100%; }
th, td { border: 1px solid #c9d2e0; padding: 2.5pt 5pt; text-align: left; vertical-align: top; }
th { background: #eef3fa; }
td code, li code, p code { font-size: 8.6pt; }
code { background: #f2f4f8; padding: 0 2px; border-radius: 2px; }
pre { background: #f5f7fa; padding: 6pt; font-size: 8.4pt; white-space: pre-wrap; break-inside: avoid; }
img { max-width: 70%; display: block; margin: 6pt auto; break-inside: avoid; }
.fig2 { display: flex; gap: 8pt; align-items: flex-start; break-inside: avoid; }
.fig2 > p { flex: 1 1 0; margin: 0; }
.fig2 img { max-width: 100%; width: 100%; }
blockquote { margin: 6pt 0; padding: 4pt 8pt; border-left: 3px solid #2f6db3; background: #f4f8fd; }
.small { font-size: 8.6pt; color: #4b5568; }
"""


def build(name: str = "report") -> Path:
    src = REPORTS / f"{name}.md"
    html_body = markdown.markdown(src.read_text(), extensions=["tables", "fenced_code", "attr_list", "md_in_html", "toc"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>Intent classifier report</title>" \
           f"<style>{CSS}</style></head><body>{html_body}</body></html>"
    html_path = REPORTS / f"{name}.html"
    html_path.write_text(html)
    chrome = next((c for c in CHROME_CANDIDATES if c and Path(c).exists()), None)
    if not chrome:
        raise SystemExit("Chrome/Chromium not found; open reports/report.html and print to PDF manually")
    pdf = REPORTS / f"{name}.pdf"
    pdf.unlink(missing_ok=True)
    # Isolated profile so we never collide with a running Chrome. Headless Chrome writes the PDF but may not exit
    # on its own, so wait until the file exists and its size is stable, then stop the process.
    with tempfile.TemporaryDirectory() as profile:
        proc = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-first-run",
                                 f"--user-data-dir={profile}", f"--print-to-pdf={pdf}", html_path.resolve().as_uri()],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        last, stable_since, deadline = -1, None, time.time() + 120
        while time.time() < deadline and proc.poll() is None:
            size = pdf.stat().st_size if pdf.exists() else -1
            if size > 0 and size == last:
                if stable_since and time.time() - stable_since > 2:
                    break
                stable_since = stable_since or time.time()
            else:
                stable_since = None
            last = size
            time.sleep(0.5)
        proc.terminate()
        proc.wait(timeout=10)
    if not pdf.exists() or pdf.stat().st_size == 0:
        raise SystemExit("Chrome did not produce a PDF; open reports/report.html and print to PDF manually")
    print(f"[report] {pdf} ({pdf.stat().st_size / 1e3:.0f} kB)")
    return pdf


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "report")
