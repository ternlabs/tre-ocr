"""Line images and labels for fine-tuning a recogniser (v0.3).

Typed pages give the labels for free: the page is rendered, made to look scanned,
and cut along the text layer's word boxes. No page of a test set takes part:
  - a file whose SHA-256 is in any set is left out;
  - a page that names a company of any set, by its listed name or symbol, is
    left out, so that no test company's name can be learned;
  - a file goes wholly to training or wholly to validation, by its SHA-256.
The pages pass the same checks as set B's (grow_set_b.candidates): typed, read as
Vietnamese, sound text, no identity field, no court ruling.

Words are grouped into rows as a detector would box them. Across a wide gap a crop
sometimes runs on, labelled with one space: a detector's box can cover two cells of
a table, and the recogniser should then keep the two figures apart.

Writes, under the gitignored <out>: dict.txt, train_list.txt, val_list.txt (image
path, tab, label; a label can hold a name) and lines/<split>/*.jpg.

Usage:
  python -I tools/train_lines.py --out cache/train/data --pages 2000
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import random
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from grow_set_b import candidates  # noqa: E402
from label_pages import _inside, kept_text, read_pages, rotations, signature_boxes  # noqa: E402
from records import read_jsonl  # noqa: E402
from sample_pairs import LISTING_URL, Client, sha256_of  # noqa: E402
from syllables import normal  # noqa: E402
from truth_b import DPI, PAD, render  # noqa: E402

LABELS = ("data/pages.jsonl", "data/set-a/pages.jsonl", "data/set-b/pages.jsonl")
SETS = ("sets/disclosures-a.jsonl", "sets/disclosures-b.jsonl", "sets/disclosures-c.jsonl",
        "sets/disclosures-d.jsonl")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

PER_FILE = 20            # pages at most from a file, so long filings do not rule
VAL_SHARE = 20           # one file in this many goes to validation
MAX_CHARS = 40           # the training config's max_text_length
MAX_RATIO = 12           # crop width over height; a longer row is cut at a word
NEAR_GAP = 1.5           # gap in row heights a detector's box runs across
FAR_GAP = 6              # gap in row heights it never runs across
RUN_ON = 0.25            # how often a gap between the two runs on
RUN_ON_FIGURES = 0.6     # the same between two figures, the case that costs numbers
JPEG = 90

VIETNAMESE = ("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
PRINTABLE = "".join(chr(c) for c in range(33, 127))
OTHER = "–—“”‘’…•°§₫"
ALPHABET = PRINTABLE + VIETNAMESE + VIETNAMESE.upper() + OTHER


def fold(text: str) -> str:
    """Lower case, no marks, words split by one space: names compare across fonts."""
    t = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return " ".join(re.findall(r"[a-z0-9]+", t))


def core_name(name: str) -> str:
    """A company's name without the words every company has."""
    f = fold(name)
    for word in ("tong cong ty", "cong ty", "ctcp", "co phan", "tnhh"):
        f = f.replace(word, " ")
    return " ".join(f.split())


def test_companies(listing: list[dict]) -> tuple[set[str], set[str], set[str]]:
    """The sets' SHA-256s, and their companies' symbols and folded names."""
    shas, symbols = set(), set()
    for path in SETS:
        for r in read_jsonl(path):
            symbols.add(r["symbol"])
            if "sha256" in r:
                shas.add(r["sha256"])
            else:
                shas |= {r["vi"]["sha256"], r["en"]["sha256"]}
    names = {core_name(x["organName"]) for x in listing
             if x.get("symbol") in symbols and x.get("organName")}
    return shas, symbols, {n for n in names if n}


def names_a_company(text: str, symbols: set[str], names: set[str]) -> bool:
    f = f" {fold(text)} "
    return (any(f" {n} " in f for n in names) or
            bool(symbols & set(re.findall(r"\b[A-Z0-9]{3}\b", text))))


def rows_of(words: list[tuple]) -> list[list[tuple]]:
    """Words grouped by height on the page, each row read left to right."""
    rows = []
    for w in sorted(words, key=lambda w: (w[1] + w[3]) / 2):
        cy = (w[1] + w[3]) / 2
        if rows and abs(cy - rows[-1]["cy"]) < 0.5 * (w[3] - w[1]):
            rows[-1]["words"].append(w)
        else:
            rows.append({"cy": cy, "words": [w]})
    return [sorted(r["words"], key=lambda w: w[0]) for r in rows]


def crops_of(row: list[tuple], rng: random.Random) -> list[list[tuple]]:
    """A row cut into crops, as a detector might box it."""
    out, cur = [], [row[0]]
    for w in row[1:]:
        h = max(x[3] - x[1] for x in cur + [w])
        gap = (w[0] - cur[-1][2]) / max(h, 1)
        figures = re.search(r"\d{3}", cur[-1][4]) and re.search(r"\d{3}", w[4])
        runs_on = gap <= NEAR_GAP or (gap <= FAR_GAP and rng.random() < (RUN_ON_FIGURES if figures else RUN_ON))
        too_long = (w[2] - cur[0][0]) / max(h, 1) > MAX_RATIO
        chars = sum(len(x[4]) + 1 for x in cur) + len(w[4]) > MAX_CHARS
        if runs_on and not too_long and not chars:
            cur.append(w)
        else:
            out.append(cur)
            cur = [w]
    return out + [cur]


def stamp_over(img: Image.Image, rng: random.Random):
    """An invented red seal across the page: the text under it is still the label."""
    w, h = img.size
    r = rng.randint(int(0.08 * w), int(0.14 * w))
    layer = Image.new("RGBA", (2 * r, 2 * r), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    ink = (rng.randint(170, 225), rng.randint(10, 50), rng.randint(20, 60), rng.randint(110, 190))
    d.ellipse((2, 2, 2 * r - 2, 2 * r - 2), outline=ink, width=max(r // 22, 3))
    d.ellipse((r // 4, r // 4, 7 * r // 4, 7 * r // 4), outline=ink, width=max(r // 40, 2))
    letters = "".join(rng.choice("ABCDEGHIKLMNOPQRSTUVXY0123456789 ") for _ in range(10))
    d.text((r // 3, r - r // 8), letters, fill=ink)
    x, y = rng.randint(0, w - 2 * r), rng.randint(0, h - 2 * r)
    img.paste(layer, (x, y), layer)


def scribble_over(img: Image.Image, rng: random.Random):
    """An invented pen stroke, as a signature falls across a line."""
    w, h = img.size
    x, y = rng.randint(0, w), rng.randint(0, h)
    pts = [(x, y)]
    for _ in range(rng.randint(6, 14)):
        x += rng.randint(-w // 12, w // 12)
        y += rng.randint(-h // 40, h // 40)
        pts.append((x, y))
    ImageDraw.Draw(img).line(pts, fill=(rng.randint(10, 60), rng.randint(20, 70), rng.randint(90, 170)),
                             width=rng.randint(2, 5))


def scanned(page: np.ndarray, rng: random.Random) -> tuple[Image.Image, float]:
    """The page as a scan might show it, and the angle it was turned by."""
    img = Image.fromarray(page).convert("RGB")
    if rng.random() < 0.15:                  # some pages stay clean, as typed PDFs do
        return img, 0.0
    if rng.random() < 0.35:
        stamp_over(img, rng)
    if rng.random() < 0.2:
        scribble_over(img, rng)
    a = np.asarray(img, dtype=np.float32) / 255
    if rng.random() < 0.6:                   # a grey copy of a copy
        a = np.repeat(a.mean(axis=2, keepdims=True), 3, axis=2)
    a = a ** rng.uniform(0.75, 1.5)          # ink faded or heavy
    tint = np.array([rng.uniform(0.86, 1.0), rng.uniform(0.86, 1.0), rng.uniform(0.82, 1.0)])
    a = a * tint
    img = Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8))
    angle = rng.uniform(-1.2, 1.2)
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    size = img.size
    scale = rng.uniform(0.45, 1.0)           # scanned coarser than the 300 dpi rendered
    img = img.resize((int(size[0] * scale), int(size[1] * scale)), Image.BILINEAR).resize(size, Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.0, 1.0)))
    a = np.asarray(img, dtype=np.float32)
    noise = np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(1, 9), a.shape)
    return Image.fromarray((a + noise).clip(0, 255).astype(np.uint8)), angle


def turned(x: float, y: float, angle: float, w: int, h: int) -> tuple[float, float]:
    """Where a point lands when the page is turned about its centre."""
    t = np.deg2rad(angle)
    dx, dy = x - w / 2, y - h / 2
    return (w / 2 + dx * np.cos(t) + dy * np.sin(t), h / 2 - dx * np.sin(t) + dy * np.cos(t))


def kept_words(pg: dict, boxes: list[tuple]) -> list:
    """The words left on the page: none the white paint touches, as it reaches PAD past a
    signature box, and none in a block kept_text left out, as the page checks never saw it."""
    painted = [(x0 - PAD, y0 - PAD, x1 + PAD, y1 + PAD) for x0, y0, x1, y1 in boxes]
    gone = [bb for bb, _ in pg["blocks"] if _inside(bb, boxes)]
    return [w for w in pg["words"] if not _inside(w[:4], gone) and
            not any(w[0] < b[2] and b[0] < w[2] and w[1] < b[3] and b[1] < w[3] for b in painted)]


def cut_page(path, n, pg, boxes, sha, split, out, tmp, rng) -> list[tuple[str, str]]:
    """Render page n, make it scanned, and write its crops; (image, label) per crop."""
    img, angle = scanned(render(path, n, pg, boxes, tmp), rng)
    k = DPI / 72
    W, H = img.size
    made = []
    words = kept_words(pg, boxes)
    for row in rows_of(words):
        for crop in crops_of(row, rng):
            label = normal(" ".join(w[4] for w in crop)).strip()
            if len(label) < 2 or len(label) > MAX_CHARS or set(label) - set(ALPHABET) - {" "}:
                continue
            x0, y0 = min(w[0] for w in crop) * k, min(w[1] for w in crop) * k
            x1, y1 = max(w[2] for w in crop) * k, max(w[3] for w in crop) * k
            pts = [turned(x, y, angle, W, H) for x in (x0, x1) for y in (y0, y1)]
            pad = (y1 - y0) * rng.uniform(0.02, 0.2)     # detectors box tight and loose
            box = (int(max(min(p[0] for p in pts) - pad, 0)), int(max(min(p[1] for p in pts) - pad, 0)),
                   int(min(max(p[0] for p in pts) + pad, W)), int(min(max(p[1] for p in pts) + pad, H)))
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                continue
            name = f"lines/{split}/{sha[:16]}-p{n}-{len(made):03d}.jpg"
            img.crop(box).save(os.path.join(out, name), quality=JPEG if rng.random() < 0.5 else rng.randint(40, 85))
            made.append((name, label))
    return made


def one_file(job) -> dict:
    """What came of one file: its crops, or why its pages were left out."""
    path, rows, out, shas, symbols, names = job
    why = Counter()
    try:
        sha = sha256_of(path)
        if sha in shas:
            return {"why": {"file in a set": 1}}
        pages, left = candidates(path, rows)
        why.update(left)
        text = read_pages(path)
        turns = rotations(path, len(text))
        rng = random.Random(sha)
        rng.shuffle(pages)
        split = "val" if int(sha, 16) % VAL_SHARE == 0 else "train"
        made, used = [], 0
        with tempfile.TemporaryDirectory() as tmp:
            for p in pages:
                n = p["page"]
                pg = text[n - 1]
                boxes = signature_boxes(pg["blocks"])
                if used >= PER_FILE:
                    why["over the file's share"] += 1
                elif turns[n - 1]:
                    why["page turned"] += 1
                elif names_a_company(normal(kept_text(pg["blocks"], boxes)), symbols, names):
                    why["names a test company"] += 1
                else:
                    made += cut_page(path, n, pg, boxes, sha, split, out, tmp, rng)
                    used += 1
        return {"split": split, "made": made, "pages": used, "why": dict(why)}
    except Exception as e:                   # one bad PDF should not stop the build
        return {"why": {f"failed: {type(e).__name__}": 1}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", type=int, default=2000, help="stop once this many pages are cut")
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="cache/pdfs")
    a = ap.parse_args()

    shutil.rmtree(a.out, ignore_errors=True)
    for split in ("train", "val"):
        os.makedirs(os.path.join(a.out, "lines", split))
    with open(os.path.join(a.out, "dict.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(ALPHABET) + "\n")

    shas, symbols, names = test_companies(Client().get(LISTING_URL).json())
    by_file: dict[str, dict[int, dict]] = {}
    for path in LABELS:
        for r in read_jsonl(path):
            if r.get("text_layer") and r.get("paper") == "no" and not r.get("skip") and not r.get("error"):
                by_file.setdefault(r["file"], {})[r["page"]] = r
    files = sorted(f for f in by_file if os.path.exists(os.path.join(a.cache, f)))
    random.Random(a.seed).shuffle(files)
    copies = {}                              # the same PDF is sometimes posted twice
    for f in files:
        copies.setdefault(sha256_of(os.path.join(a.cache, f)), f)
    kept = set(copies.values())
    files = [f for f in files if f in kept]
    jobs = ((os.path.join(a.cache, f), list(by_file[f].values()), a.out, shas, symbols, names)
            for f in files)

    lists = {"train": [], "val": []}
    tally, pages, tried = Counter(), Counter(), 0
    with multiprocessing.Pool(a.workers) as pool:
        for r in pool.imap(one_file, jobs):
            tried += 1
            tally.update(r["why"])
            if r.get("pages"):
                pages[r["split"]] += r["pages"]
                lists[r["split"]] += r["made"]
            if sum(pages.values()) >= a.pages:
                pool.terminate()
                break
    for split, rows in lists.items():
        with open(os.path.join(a.out, f"{split}_list.txt"), "w", encoding="utf-8") as fh:
            fh.writelines(f"{name}\t{label}\n" for name, label in rows)
    all_labels = "".join(lb for rows in lists.values() for _, lb in rows)
    print(json.dumps({"files_tried": tried, "pages": pages, "lines": {k: len(v) for k, v in lists.items()},
                      "with_number": sum(bool(re.search(r"\d{3}", lb)) for rows in lists.values() for _, lb in rows),
                      "vietnamese_letters_seen": len(set(all_labels) & set(VIETNAMESE)),
                      "of": len(VIETNAMESE), "left_out": tally}, ensure_ascii=False))


if __name__ == "__main__":
    main()
