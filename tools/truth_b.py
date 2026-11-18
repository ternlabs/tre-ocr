"""Truth and page images for set B.

Step 5 of METHOD.md. For each page of the set: the truth is the page's text
layer with the signature box removed (label_pages.signature_boxes), in NFC with
the lookalike Ð made Đ (syllables.normal); the image is the page rendered at
300 dpi with the same box painted white, so an engine is not asked to read
text the truth leaves out. The national title, motto and line order are left
as the text layer has them; the scoring code (step 7) deals with them.

The PDF is fetched again if it is not in the cache, and its SHA-256 must match
the set's.

Writes <out>, one line per page (sha256, page, text), not committed: the text
can hold names, and it is rebuilt from the committed set and the PDFs. Images
go to <images>/<sha256 first 16>-p<page>.png, under the gitignored cache.

Usage:
  python -I tools/truth_b.py sets/disclosures-b.jsonl --out data/truth-b.jsonl --images cache/set-b
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from label_pages import blank_out, kept_text, read_pages, read_ppm, signature_boxes  # noqa: E402
from sample_pairs import Client, sha256_of  # noqa: E402
from syllables import normal  # noqa: E402

DPI = 300
PAD = 2.0                # points around a signature box, for its frame


def render(path: str, page: int, pg: dict, boxes: list[tuple], tmp: str):
    """The page at DPI, its signature boxes (with PAD around) painted white."""
    prefix = os.path.join(tmp, "p")
    subprocess.run(["pdftoppm", "-r", str(DPI), "-f", str(page), "-l", str(page), "-singlefile",
                    path, prefix], check=True, capture_output=True)
    padded = [(x0 - PAD, y0 - PAD, x1 + PAD, y1 + PAD) for x0, y0, x1, y1 in boxes]
    return blank_out(read_ppm(prefix + ".ppm"), padded, pg["w"], pg["h"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--cache", default="cache/pdfs")
    a = ap.parse_args()
    os.makedirs(a.images, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.set, encoding="utf-8") as fh:
        entries = [json.loads(line) for line in fh]

    c = Client()
    done = 0
    with open(a.out + ".tmp", "w", encoding="utf-8") as out, tempfile.TemporaryDirectory() as tmp:
        for e in entries:
            path, why = c.download(e["pdf"], a.cache)
            if not path:
                sys.exit(f"{e['pdf']}: {why}")
            if sha256_of(path) != e["sha256"]:
                sys.exit(f"{e['pdf']}: SHA-256 differs from the set's")
            pg = read_pages(path)[e["page"] - 1]
            boxes = signature_boxes(pg["blocks"])
            out.write(json.dumps({"sha256": e["sha256"], "page": e["page"],
                                  "text": normal(kept_text(pg["blocks"], boxes))},
                                 ensure_ascii=False) + "\n")
            name = f"{e['sha256'][:16]}-p{e['page']}.png"
            Image.fromarray(render(path, e["page"], pg, boxes, tmp)).save(os.path.join(a.images, name))
            done += 1
    os.replace(a.out + ".tmp", a.out)
    print(json.dumps({"pages": done, "out": a.out, "images": a.images}))


if __name__ == "__main__":
    main()
