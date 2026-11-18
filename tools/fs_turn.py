"""Re-read the sideways pages of statement drafts, turned upright (profile 2, FS1).

A wide statement (B02 with four columns, often) is printed landscape on a portrait page.
The reader leaves page orientation off, as it was measured on sets A to G, so such a page
comes out as tall, thin boxes of nothing. Changing the reader for every page would change
a measured tool; instead, only for statements, a page is called sideways when most of its
lines' boxes are much taller than wide, and it is read again from its saved image turned
90 and 270 degrees. Only pages before the notes are turned: the notes are out of scope,
and many of their wide tables are sideways too. The turn with more sure lines is kept;
the page records "turned" and its new "size", so its boxes are on the turned image. A
line on its side is not counted as sure in any of the three readings: the page as it lies
can read its tall boxes surely as junk, and so outvote the turn that reads it.

Usage: .venv-paddle/bin/python -I tools/fs_turn.py [--device cpu] data/set-h/drafts/*.draft.json
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too

TALL = 1.5          # a box this many times taller than wide is a line on its side
SIDEWAYS = 0.5      # share of a page's lines on their side that makes it sideways
MIN_LINES = 10
NOTES = re.compile(r"thuyet minh bao cao tai chinh")
TOP = 700           # px at 300 dpi: where a portrait page's title sits


def sideways(page: dict) -> bool:
    boxes = [ln["box"] for ln in page.get("lines", []) if ln.get("box") and len(ln["text"].strip()) >= 3]
    tall = sum(b[3] - b[1] > TALL * (b[2] - b[0]) for b in boxes)
    return len(boxes) >= MIN_LINES and tall >= SIDEWAYS * len(boxes)


def before_notes(pages: list[dict]) -> list[dict]:
    """The pages up to the first whose top says the notes begin."""
    from sort_kinds import fold
    for k, p in enumerate(pages):
        if any(ln.get("box") and ln["box"][1] < TOP and NOTES.search(fold(ln["text"]))
               for ln in p.get("lines", [])):
            return pages[:k]
    return pages


def on_side(ln: dict) -> bool:
    b = ln.get("box")
    return bool(b) and b[3] - b[1] > TALL * (b[2] - b[0])


def sure_lines(lines: list[dict]) -> int:
    """Sure lines of 3+ characters, not on their side."""
    return sum(ln["sure"] and len(ln["text"].strip()) >= 3 and not on_side(ln) for ln in lines)


def turn(page: dict, images: str, readers, always: bool = False) -> dict:
    """The page read upright: the better of two turns, or the page as it was. always: the
    better turn even if it reads fewer sure lines (a person said the page is sideways;
    for building truth only)."""
    from PIL import Image
    from read_pdf import lines_of
    src = os.path.join(images, page["key"] + ".png")
    best = None
    for angle in (90, 270):
        png = os.path.join(images, f"{page['key']}-r{angle}.png")
        if not os.path.exists(png):
            Image.open(src).rotate(angle, expand=True).save(png)
        got = readers.page(png)
        lines = lines_of(got)
        if best is None or sure_lines(lines) > sure_lines(best[1]):
            w, h = Image.open(png).size
            best = ({**page, **got, "turned": angle, "size": [w, h]}, lines)
    if not always and sure_lines(best[1]) <= sure_lines(page["lines"]):
        return page
    out, lines = best
    out["lines"] = lines
    return out


def upright(d: dict, images: str, readers) -> list[int]:
    """Turn a draft's sideways pages before the notes, in place; the pages turned."""
    head = {p["page"] for p in before_notes(d["pages"])}
    done = []
    for i, p in enumerate(d["pages"]):
        if p["page"] in head and p.get("kind") == "read" and "turned" not in p and sideways(p):
            d["pages"][i] = turn(p, images, readers)
            if "turned" in d["pages"][i]:
                done.append(p["page"])
    return done


def main():
    import argparse
    from read_pdf import ROUND1, Readers
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("drafts", nargs="+")
    ap.add_argument("--device", default="gpu:0")
    a = ap.parse_args()
    readers = None
    for path in a.drafts:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        head = {p["page"] for p in before_notes(d["pages"])}
        if not any(p["page"] in head and p.get("kind") == "read" and "turned" not in p and sideways(p)
                   for p in d["pages"]):
            continue
        readers = readers or Readers(ROUND1, a.device)
        done = upright(d, os.path.join(os.path.dirname(path), "images"), readers)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=1)
        print(os.path.basename(path), "turned:", done)


if __name__ == "__main__":
    main()
