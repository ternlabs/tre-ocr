"""Page images for a set, as the engines see them.

Step 8 of METHOD.md. Each page of the set rendered at 300 dpi with its signature
box (when the file carries a text layer with one) painted white, as for sets B
and C (tools/truth_b.py, tools/draft_c.py). Set A's rows hold the page under
"vi"; the English twin is the truth's source, never shown to an engine.

Writes <images>/<sha256 first 16>-p<page>.png (cache, gitignored). Pages done
before are skipped.

Usage:
  python -I tools/pages.py sets/disclosures-a.jsonl --images cache/set-a
"""
import argparse
import os
import sys
import tempfile

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from label_pages import read_pages, signature_boxes  # noqa: E402
from records import page_key, page_of, read_jsonl  # noqa: E402
from sample_pairs import Client, sha256_of  # noqa: E402
from truth_b import render  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set")
    ap.add_argument("--images", required=True)
    ap.add_argument("--cache", default="cache/pdfs")
    a = ap.parse_args()
    os.makedirs(a.images, exist_ok=True)
    entries = [page_of(r) for r in read_jsonl(a.set)]
    c, made = Client(), 0
    with tempfile.TemporaryDirectory() as tmp:
        for e in entries:
            png = os.path.join(a.images, page_key(e) + ".png")
            if os.path.exists(png):
                continue
            path, why = c.download(e["pdf"], a.cache)
            if not path:
                sys.exit(f"{e['pdf']}: {why}")
            if sha256_of(path) != e["sha256"]:
                sys.exit(f"{e['pdf']}: SHA-256 differs from the set's")
            pg = read_pages(path)[e["page"] - 1]
            Image.fromarray(render(path, e["page"], pg, signature_boxes(pg["blocks"]), tmp)).save(png)
            made += 1
    print(f"images: {made} made, {len(entries)} in {a.images}")


if __name__ == "__main__":
    main()
