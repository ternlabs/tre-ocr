"""Page images and drafts for set C.

Step 6 of METHOD.md. For each page of the set: the page rendered at 300 dpi with
its signature box (when the scan carries a text layer with one) painted white,
as in set B; then one draft per engine, for the person who corrects the page
(tools/correct_c.py).

Two engines draft, so the person sees where they disagree:
  - easyocr (vi): not an engine under test (step 8), so the truth does not lean
    toward one; its boxes are joined into lines allowing for the scan's tilt,
    and two columns side by side are written one after the other, as the text
    rules ask (tilted_lines); the boxes are kept in <work>/drafts/easyocr-boxes/ so the joining
    can change without reading the page again;
  - tesseract (vie), when installed.
An engine that drafts the truth would still be helped by it: a person
correcting a draft leaves some of its errors. The second read checks every
number and heading field against the image, not the draft.

Writes <images>/<sha256 first 16>-p<page>.png (cache, gitignored) and
<work>/drafts/<engine>/<same>.txt, with the engine versions in
<work>/drafts/versions.json (gitignored: drafts hold the page's text, names
too). Pages done before are skipped.

Usage:
  python -I tools/draft_c.py sets/disclosures-c.jsonl --work data/set-c --images cache/set-c
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from label_pages import read_pages, signature_boxes  # noqa: E402
from sample_pairs import Client, sha256_of  # noqa: E402
from syllables import normal  # noqa: E402
from truth_b import render  # noqa: E402


def key(e: dict) -> str:
    return f"{e['sha256'][:16]}-p{e['page']}"


class EasyOcr:
    name = "easyocr"

    def __init__(self):
        import easyocr
        self.version = f"easyocr {easyocr.__version__}, vi"
        self.reader = easyocr.Reader(["vi"], gpu=False, verbose=False)

    def read(self, png: str) -> str:
        boxes = self.boxes(png)
        return "\n".join(tilted_lines(boxes))

    def boxes(self, png: str) -> list[list]:
        """Corner points and text per box, cached beside the drafts as JSON."""
        cache = os.path.join(self.box_dir, os.path.basename(png)[:-4] + ".json")
        if not os.path.exists(cache):
            boxes = [[[[float(x), float(y)] for x, y in bb], text]
                     for bb, text, _ in self.reader.readtext(png)]
            with open(cache, "w", encoding="utf-8") as fh:
                json.dump(boxes, fh, ensure_ascii=False)
        with open(cache, encoding="utf-8") as fh:
            return json.load(fh)


def tilted_lines(boxes: list[list]) -> list[str]:
    """Boxes (four corners from top-left clockwise, text) into lines, on a page
    scanned at a slight tilt. The tilt is the median slope of the boxes' top
    edges, and each box is moved up or down by it. A box then joins the box
    just to its left when the blank between them is under COLUMN_GAP heights
    and their heights mostly overlap: so a printed line stays whole even where
    the lines beside it are close. These pieces join the line their middle
    falls in, any line, not only the last one started; then columns side by
    side are split apart (columns)."""
    slopes = sorted((bb[1][1] - bb[0][1]) / (bb[1][0] - bb[0][0])
                    for bb, _ in boxes if bb[1][0] - bb[0][0] > 100)
    slope = slopes[len(slopes) // 2] if slopes else 0.0
    flat = []                                   # (top, left, bottom, right, text)
    for bb, text in boxes:
        xs, ys = [p[0] for p in bb], [p[1] for p in bb]
        shift = slope * (min(xs) + max(xs)) / 2
        flat.append((min(ys) - shift, min(xs), max(ys) - shift, max(xs), text))
    if not flat:
        return []
    heights = sorted(b[2] - b[0] for b in flat)
    h = heights[len(heights) // 2]
    gap = COLUMN_GAP * h

    def overlap(a, b):                          # share of the shorter height
        return (min(a[2], b[2]) - max(a[0], b[0])) / min(a[2] - a[0], b[2] - b[0])

    chains = []                                 # boxes, left to right
    for b in sorted(flat, key=lambda b: b[1]):
        near = [c for c in chains
                if -h <= b[1] - c[-1][3] <= gap and overlap(c[-1], b) >= 0.5]
        if near:
            max(near, key=lambda c: overlap(c[-1], b)).append(b)
        else:
            chains.append([b])

    rows = []                                   # [top, bottom, pieces]
    for c in sorted(chains, key=lambda c: min(b[0] for b in c)):
        top = sum(b[0] for b in c) / len(c)
        bottom = sum(b[2] for b in c) / len(c)
        piece = [min(b[1] for b in c), max(b[3] for b in c), " ".join(b[4] for b in c)]
        mid = (top + bottom) / 2
        hits = [r for r in rows if r[0] <= mid <= r[1]]
        if hits:
            min(hits, key=lambda r: abs((r[0] + r[1]) / 2 - mid))[2].append(piece)
        else:
            rows.append([top, bottom, [piece]])
    rows.sort(key=lambda r: (r[0] + r[1]) / 2)
    out = []
    for r in rows:
        merged = []
        for p in sorted(r[2], key=lambda p: p[0]):
            if merged and p[0] - merged[-1][1] <= gap:
                merged[-1] = [merged[-1][0], max(merged[-1][1], p[1]), merged[-1][2] + " " + p[2]]
            else:
                merged.append(p)
        out.append(merged)
    return columns(out)


COLUMN_GAP = 3      # box heights of blank between two columns
# Where a split between two columns may fall, as a share of the text's width:
# a heading's or signature block's two columns meet near the middle; a split
# far left or right is a table's column or a mark in the margin.
COLUMN_BAND = (0.35, 0.68)


def columns(rows: list[list]) -> list[str]:
    """Rows of pieces into lines. A row of exactly two pieces, split near the
    middle (COLUMN_BAND), opens a block of two columns, split at its second
    piece's left edge; the rows after it stay in the block while each piece lies on one side of the split. A block is
    written left column first, then right, as the text rules ask. A row of
    three or more pieces is a table row and stays one line, as does a row
    across the split; either ends the block. Rows of two pieces after a table
    row are the table's too (cells left empty), until a row of one piece."""
    out, left, right, split, table = [], [], [], None, False
    x0 = min((p[0] for row in rows for p in row), default=0.0)
    x1 = max((p[1] for row in rows for p in row), default=1.0)
    lo, hi = (x0 + f * (x1 - x0) for f in COLUMN_BAND)

    def close():
        out.extend(left + right)
        left.clear()
        right.clear()

    for row in rows:
        if split is not None and len(row) <= 2 and all(
                p[1] <= split or p[0] >= split for p in row):
            for p in row:
                (left if p[1] <= split else right).append(p[2])
            continue
        close()
        split = None
        table = len(row) >= 3 or (table and len(row) == 2)
        if len(row) == 2 and not table and lo <= row[1][0] <= hi:
            split = row[1][0]
            left.append(row[0][2])
            right.append(row[1][2])
        else:
            out.append(" ".join(p[2] for p in row))
    close()
    return out


class Tesseract:
    name = "tesseract"

    def __init__(self):
        if not shutil.which("tesseract"):
            raise RuntimeError("tesseract is not installed")
        out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout
        self.version = out.splitlines()[0] + ", vie"

    def read(self, png: str) -> str:
        return subprocess.run(["tesseract", png, "stdout", "-l", "vie", "--psm", "3"],
                              check=True, capture_output=True, text=True).stdout


ENGINES = {"easyocr": EasyOcr, "tesseract": Tesseract}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set")
    ap.add_argument("--work", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--engines", default="easyocr,tesseract")
    ap.add_argument("--cache", default="cache/pdfs")
    a = ap.parse_args()
    os.makedirs(a.images, exist_ok=True)
    with open(a.set, encoding="utf-8") as fh:
        entries = [json.loads(line) for line in fh]

    c = Client()
    with tempfile.TemporaryDirectory() as tmp:
        for e in entries:
            png = os.path.join(a.images, key(e) + ".png")
            if os.path.exists(png):
                continue
            path, why = c.download(e["pdf"], a.cache)
            if not path:
                sys.exit(f"{e['pdf']}: {why}")
            if sha256_of(path) != e["sha256"]:
                sys.exit(f"{e['pdf']}: SHA-256 differs from the set's")
            pg = read_pages(path)[e["page"] - 1]
            Image.fromarray(render(path, e["page"], pg, signature_boxes(pg["blocks"]), tmp)).save(png)
    print(f"images: {len(entries)} in {a.images}")

    vpath = os.path.join(a.work, "drafts", "versions.json")
    versions = json.load(open(vpath, encoding="utf-8")) if os.path.exists(vpath) else {}
    for name in a.engines.split(","):
        out = os.path.join(a.work, "drafts", name)
        todo = [e for e in entries if not os.path.exists(os.path.join(out, key(e) + ".txt"))]
        if not todo:
            continue
        try:
            eng = ENGINES[name]()
            eng.box_dir = os.path.join(a.work, "drafts", name + "-boxes")
            os.makedirs(eng.box_dir, exist_ok=True)
        except Exception as ex:
            print(f"{name}: skipped ({ex})")
            continue
        os.makedirs(out, exist_ok=True)
        versions[name] = eng.version
        with open(vpath, "w", encoding="utf-8") as fh:
            json.dump(versions, fh, indent=2)
        for n, e in enumerate(todo, 1):
            text = normal(eng.read(os.path.join(a.images, key(e) + ".png")))
            with open(os.path.join(out, key(e) + ".txt"), "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"{name}: {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
