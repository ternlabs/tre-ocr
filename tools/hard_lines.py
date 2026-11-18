"""Scanned lines two recognisers disagree on, cut for labelling by hand (round 2).

Typed pages made to look scanned teach no handwriting. Lines where the stock and the
fine-tuned reader read different digits are the ones worth a person's time, and so are
lines in blue pen, which come first. No page of
a test set takes part:
  - a file whose SHA-256 is in any set is left out;
  - a page whose text, as read, names a company of any set is left out;
  - so is a page holding an identity field or any identity label.
A scan has no text layer to check beforehand, so the last two run on the readers' output.

Three steps, the middle one tools/run_engine.py with both recognisers:

  python -I tools/hard_lines.py draw --work cache/train/hand --pages 500
  .venv-paddle/bin/python -I tools/run_engine.py cache/train/hand/pool.jsonl \
    --images cache/train/hand/pages --engine paddleocr --machine dev \
    --out cache/train/hand/stock.jsonl --boxes cache/train/hand/boxes-stock
  (the same with --rec-model, to tuned.jsonl and boxes-tuned)
  python -I tools/hard_lines.py cut --work cache/train/hand

Everything lands under <work>, gitignored: crops and their reads can hold names.
tools/label_lines.py is the page for labelling them.
"""
import argparse
import json
import os
import random
import sys
import tempfile

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from label_pages import read_pages, signature_boxes  # noqa: E402
from privacy import identity  # noqa: E402
from records import page_key, read_jsonl, write_jsonl  # noqa: E402
from sample_pairs import LISTING_URL, Client, sha256_of  # noqa: E402
from train_lines import LABELS, names_a_company, test_companies  # noqa: E402
from truth_b import render  # noqa: E402
from two_readers import digits, mostly_red  # noqa: E402

import numpy as np  # noqa: E402

PER_FILE = 3             # pages at most from a file, so long filings do not rule
PER_PAGE = 5             # crops at most from a page
PER_PAGE_AGREED = 3      # and lines in pen both readers agree on
PAD = 4                  # pixels kept around a box
MIN_DIGITS = 2           # a stray digit or a signature read as one teaches nothing
BLUE_OVER = 40           # how far blue must stand above red, for pen ink
BLUE_SHARE = 0.005


def draw(a):
    """Render a random draw of scanned pages outside the sets."""
    shas = test_companies(Client().get(LISTING_URL).json())[0]
    by_file: dict[str, dict[int, dict]] = {}
    for path in LABELS:
        for r in read_jsonl(path):
            if not r.get("text_layer") and not r.get("skip") and not r.get("error"):
                by_file.setdefault(r["file"], {})[r["page"]] = r
    rng = random.Random(a.seed)
    files = sorted(f for f in by_file if os.path.exists(os.path.join(a.cache, f)))
    rng.shuffle(files)
    pages_dir = os.path.join(a.work, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    pool, seen = [], set()
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            if len(pool) >= a.pages:
                break
            path = os.path.join(a.cache, f)
            sha = sha256_of(path)
            if sha in shas or sha in seen:       # the same PDF is sometimes posted twice
                continue
            seen.add(sha)
            numbers_ = sorted(by_file[f])
            rng.shuffle(numbers_)
            try:
                read = read_pages(path)
                for n in numbers_[:PER_FILE]:
                    e = {"sha256": sha, "page": n}
                    Image.fromarray(render(path, n, read[n - 1], signature_boxes(read[n - 1]["blocks"]), tmp)) \
                        .save(os.path.join(pages_dir, page_key(e) + ".png"))
                    pool.append(e)
            except Exception as ex:              # a broken PDF costs its pages, not the draw
                print(f"{f}: {type(ex).__name__}, left out", flush=True)
    write_jsonl(os.path.join(a.work, "pool.jsonl"), pool)
    print(f"{len(pool)} pages from {len(seen)} files", flush=True)


def worth_a_label(reads: list[str], box: list[int]) -> bool:
    """Not a stray digit, not a signature, not a line turned on its side."""
    return box[3] - box[1] <= box[2] - box[0] and max(map(len, map(digits, reads))) >= MIN_DIGITS


def pen_ink(page: np.ndarray, box: list[int]) -> bool:
    x0, y0, x1, y1 = box
    a = page[y0:y1, x0:x1]
    return bool(a.size and (a[..., 2] - a[..., 0] > BLUE_OVER).mean() >= BLUE_SHARE)


def cut(a):
    """Crop the lines whose digits differ, on pages that pass the checks."""
    _, symbols, names = test_companies(Client().get(LISTING_URL).json())
    stock = {(r["sha256"], r["page"]): r["text"].split("\n") for r in read_jsonl(os.path.join(a.work, "stock.jsonl"))}
    crops_dir = os.path.join(a.work, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    rng = random.Random(a.seed)
    rows, left_out = [], {"names a test company": 0, "identity": 0, "lines differ in count": 0}
    for r in read_jsonl(os.path.join(a.work, "tuned.jsonl")):
        k = (r["sha256"], r["page"])
        tuned, other = r["text"].split("\n"), stock.get(k, [])
        text = "\n".join(tuned + other)
        if names_a_company(text, symbols, names):
            left_out["names a test company"] += 1
            continue
        if identity(text):
            left_out["identity"] += 1
            continue
        stem = page_key(r)
        with open(os.path.join(a.work, "boxes-tuned", stem + ".json"), encoding="utf-8") as fh:
            boxes = json.load(fh)["boxes"]
        if not len(tuned) == len(other) == len(boxes):
            left_out["lines differ in count"] += 1
            continue
        image = Image.open(os.path.join(a.work, "pages", stem + ".png")).convert("RGB")
        page = np.array(image, dtype=int)
        good = [i for i in range(len(boxes)) if worth_a_label([tuned[i], other[i]], boxes[i])
                and not mostly_red(page, boxes[i])]
        hard = [i for i in good if digits(tuned[i]) != digits(other[i])]
        agreed = [i for i in good if i not in hard and pen_ink(page, boxes[i])]   # readers can both be wrong
        rng.shuffle(hard)
        rng.shuffle(agreed)
        for i in hard[:PER_PAGE] + agreed[:PER_PAGE_AGREED]:
            x0, y0, x1, y1 = boxes[i]
            name = f"{stem}-{i}"
            image.crop((max(0, x0 - PAD), max(0, y0 - PAD), min(image.width, x1 + PAD), min(image.height, y1 + PAD))).save(
                os.path.join(crops_dir, name + ".jpg"), quality=92)
            rows.append({"id": name, "sha256": k[0], "page": k[1], "line": i, "reads": [tuned[i], other[i]],
                         "pen": pen_ink(page, boxes[i])})
    rng.shuffle(rows)
    rows.sort(key=lambda x: not x["pen"])    # hand-filled digits first, the case round 2 is for
    write_jsonl(os.path.join(a.work, "candidates.jsonl"), rows)
    print(json.dumps({"crops": len(rows), "pen": sum(x["pen"] for x in rows), "pages": len({(x["sha256"], x["page"]) for x in rows}),
                      "left_out": left_out}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["draw", "cut"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--pages", type=int, default=500)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--cache", default="cache/pdfs")
    a = ap.parse_args()
    {"draw": draw, "cut": cut}[a.step](a)


if __name__ == "__main__":
    main()
