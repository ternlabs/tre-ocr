"""Number truth for set A: the numbers each scanned Vietnamese page should read.

Step 4 of METHOD.md, first pass: page n's numbers, taken whole. The truth used
now is tools/align_pages.py's, which keeps only the part of the English text
the scan shows; this stays for comparison and for the hand-check renders.

The English twin page has a text layer; its numbers are
the truth for the scan's numbers (never for tone marks). A number is a digit
string of 3+ digits with every separator removed: Vietnamese writes
1.234.567,8 and English 1,234,567.8, and both read 12345678. The signature box
is removed first, as it adds numbers (tax code, time) that are not in the
document.

Writes <out>, one line per set entry: the Vietnamese page it is truth for, the
English page it came from, and the numbers in reading order. Not committed: an
English page can hold a person's ID or phone number, and the list is rebuilt
from the committed set and the PDFs.

With --check N, also renders N entries drawn at random (fixed seed), each
Vietnamese and English page as a PNG each, to <check-dir>, for the hand check: is
the English page the same content?

Usage:
  python -I tools/number_truth.py sets/disclosures-a.jsonl --out data/numbers-a.jsonl
  python -I tools/number_truth.py sets/disclosures-a.jsonl --out data/numbers-a.jsonl \\
      --check 10 --seed 8 --check-dir cache/check-a
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from label_pages import kept_text, read_pages, signature_boxes  # noqa: E402
from records import numbers  # noqa: E402


def cached(url: str, cache: str) -> str:
    """The sampler's cache name for a URL (sample_pairs.Client.download)."""
    return os.path.join(cache, hashlib.sha1(url.encode()).hexdigest()[:16] + ".pdf")


def page_numbers(path: str, page: int) -> list[str]:
    pg = read_pages(path)[page - 1]
    return numbers(kept_text(pg["blocks"], signature_boxes(pg["blocks"])))


def render_pair(vi: tuple[str, int], en: tuple[str, int], stem: str):
    """Both pages at 70 dpi: <stem>-vi.png and <stem>-en.png."""
    for lang, (path, page) in (("vi", vi), ("en", en)):
        subprocess.run(["pdftoppm", "-q", "-png", "-r", "70", "-f", str(page), "-l", str(page),
                        "-singlefile", path, f"{stem}-{lang}"], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="cache/pdfs")
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--check-dir", default="cache/check-a")
    a = ap.parse_args()
    with open(a.set, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    counts = []
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            nums = page_numbers(cached(r["en"]["pdf"], a.cache), r["en"]["page"])
            counts.append(len(nums))
            fh.write(json.dumps({"vi": {"sha256": r["vi"]["sha256"], "page": r["vi"]["page"]},
                                 "en": {"sha256": r["en"]["sha256"], "page": r["en"]["page"]},
                                 "numbers": nums}) + "\n")
    if a.check:
        os.makedirs(a.check_dir, exist_ok=True)
        for i in sorted(random.Random(a.seed).sample(range(len(rows)), a.check)):
            r = rows[i]
            out = os.path.join(a.check_dir, f"{i:03d}")
            render_pair((cached(r["vi"]["pdf"], a.cache), r["vi"]["page"]),
                         (cached(r["en"]["pdf"], a.cache), r["en"]["page"]), out)
            print(f"{out}  {r['symbol']} {r['kind']} vi p{r['vi']['page']} en p{r['en']['page']} "
                  f"by {r['paired_by']}", file=sys.stderr)
    counts.sort()
    print(json.dumps({"pages": len(rows), "numbers": sum(counts),
                      "per_page": {"min": counts[0], "median": counts[len(counts) // 2],
                                   "max": counts[-1]}} if counts else {"pages": 0}))


if __name__ == "__main__":
    main()
