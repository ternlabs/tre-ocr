"""Join two recognisers' runs over one set of boxes into one run, marking what is unsure.

Both runs must come from tools/run_engine.py with the same detector, so that their
lines match box for box, within ALIGN_PIXELS; a page where they do not is written from the
first run alone, all unsure. For each page:

- **Seal text goes.** A line whose ink is mostly red is left out. A page with no colour
  at all (a black and white scan) cannot be tested, so its seal text may remain, and
  the record says so with "no_colour".
- **Digits that differ.** Where the two lines' digits differ, the second run's line is
  kept and the line is unsure.
- **Numbers that differ.** Two lines can hold the same digits split differently, into
  one figure or two. The first run's line stands, as it was trained to keep figures
  apart, and the line is unsure.
- **Sure** means the two lines' digits and numbers agree and both readers score at least --sure.
  Every other line's index is listed under "unsure". tools/score.py measures the rest.
- **An unsure line is kept, and marked.** This run is what is scored, so that recall can be
  told over all lines and over sure lines. Output for a reader leaves unsure values empty.
- **The heading's two columns stay together.** Detection sorts lines top to bottom, which
  weaves a heading's left and right blocks into each other and splits a name printed on
  two lines. The blank strip between the columns divides them (gutter), and the national
  title marks where the heading starts and cuts a box that holds both columns
  (split_title). See heading_order.

Usage:
  python -I tools/two_readers.py c \
    --first data/runs/paddleocr-tuned-c.jsonl --first-boxes cache/det-tuned-c \
    --second data/runs/paddleocr-c.jsonl --second-boxes cache/det-c \
    --images cache/set-c --out data/runs/two-readers-c.jsonl
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from records import numbers, read_jsonl, write_jsonl  # noqa: E402
from score import strip  # noqa: E402

RED_OVER = 40  # how far red must stand above green and blue, as in label_pages.red_mask
INK_UNDER = 160
MOSTLY_RED = 0.5
COLOUR_SHARE = 0.0005  # the share of a page's pixels with colour in them
HEADING_PART = 1 / 3  # of the page's text height
HEADING_LINES = 14  # more lines than this side by side is a table, not a heading
ALIGN_PIXELS = 4  # the same detector can differ by rounding between runs
GUTTER_LINES = 4  # how many lines' height of the heading to look for the blank strip in
GUTTER_FROM, GUTTER_TO = 0.3, 0.7  # of the page's width
GUTTER_INK = 0.01  # the share of a column's pixels that may be ink and still count as blank
NAME_BEFORE_TITLE = 4  # letters before the title on one line, to be a name and not a stray mark
TITLE_RE = re.compile(r"cong\W*hoa\W*xa\W*hoi\W*chu\W*nghia|socialist\W*republic", re.I)  # may wrap after this


def digits(line: str) -> str:
    return "".join(c for c in line if c.isdigit())


def mostly_red(page: np.ndarray, box: list[int]) -> bool:
    x0, y0, x1, y1 = box
    a = page[y0:y1, x0:x1]
    if not a.size:
        return False
    red = a[..., 0] - np.maximum(a[..., 1], a[..., 2]) > RED_OVER
    ink = (a.min(axis=2) < INK_UNDER) | red
    return bool(ink.any()) and red.sum() / ink.sum() >= MOSTLY_RED


def has_colour(page: np.ndarray) -> bool:
    return bool((page.max(axis=2) - page.min(axis=2) > RED_OVER).mean() >= COLOUR_SHARE)


def heading_limit(boxes: list[list[int]]) -> float:
    """The y where the heading part of the page ends."""
    top = min(b[1] for b in boxes)
    return top + (max(b[3] for b in boxes) - top) * HEADING_PART


def same_height(p: list[int], q: list[int]) -> bool:
    return min(p[3], q[3]) - max(p[1], q[1]) > 0.4 * min(p[3] - p[1], q[3] - q[1])


def widest_blank(blank: np.ndarray, first: int, last: int) -> float | None:
    """The x of the middle of the widest blank strip between two xs."""
    width, at, start = 0, None, None
    for x in range(first, last + 1):
        if x < last and blank[x]:
            start = x if start is None else start
        elif start is not None:
            if x - start > width:
                width, at = x - start, (start + x) / 2
            start = None
    return at


def gutter(page: np.ndarray, boxes: list[list[int]], titled: bool) -> float | None:
    """The x of the middle of the widest blank strip beside the first two boxes side by
    side in the heading, so that a short first line cannot place it. On a page with a title,
    a centred line below can fill the true strip and leave a wider one inside a column; so
    a strip inside either box of the pair gives way to the pair's own gap, when that gap is
    in the page's middle (beside a logo it is not). None without a pair."""
    if not boxes:
        return None
    limit = heading_limit(boxes)
    top = [b for b in boxes if b[1] <= limit]
    pair = next(((p, q) for i, p in enumerate(top) for q in top[i + 1:i + 3]
                 if same_height(p, q) and (q[0] > p[2] or p[0] > q[2])), None)
    if pair is None:
        return None
    p, q = pair
    high = max(p[3] - p[1], q[3] - q[1])
    y0 = int(min(p[1], q[1]) - high)
    band = page[max(0, y0):int(y0 + high * (GUTTER_LINES + 1))]
    blank = (band.min(axis=2) < INK_UNDER).mean(axis=0) <= GUTTER_INK
    first, last = int(page.shape[1] * GUTTER_FROM), int(page.shape[1] * GUTTER_TO)
    at = widest_blank(blank, first, last)
    gap = min(p[2], q[2]), max(p[0], q[0])
    if titled and at is not None and (p[0] < at < p[2] or q[0] < at < q[2]) and first <= gap[0] and gap[1] <= last:
        return widest_blank(blank, *gap) or (gap[0] + gap[1]) / 2
    return at


def title_at(line: str) -> int | None:
    """Where the national title starts in a line."""
    plain, where = strip(line)
    m = TITLE_RE.search(plain)
    return where[m.start()] if m else None


def find_title(lines: list[str], boxes: list[list[int]]) -> int | None:
    """The index of the first line in the heading part that holds the national title."""
    if not boxes:
        return None
    limit = heading_limit(boxes)
    return next((i for i in range(min(len(lines), len(boxes)))
                 if boxes[i][1] <= limit and title_at(lines[i]) is not None), None)


def split_title(first: list[str], second: list[str], one: dict, two: dict, at: float | None,
                i: int | None) -> tuple:
    """Cut the title's line, number i, if it also holds what is printed to the title's left.
    Returns both runs' lines, both runs' boxes and scores, and the title's index, or None
    when the heading has no title."""
    first, second = list(first), list(second)
    one = {**one, "boxes": list(one["boxes"]), "scores": list(one["scores"])}
    two = {**two, "boxes": list(two["boxes"]), "scores": list(two["scores"])}
    if i is None:
        return first, second, one, two, None
    k = title_at(first[i])
    if len(re.sub(r"\W", "", first[i][:k])) < NAME_BEFORE_TITLE:
        return first, second, one, two, i
    x0, y0, x1, y1 = one["boxes"][i]
    cut = int(at if at and x0 < at < x1 else x0 + (x1 - x0) * k / max(1, len(first[i])))
    if i < len(second):
        k2 = title_at(second[i])
        k2 = int(len(second[i]) * k / max(1, len(first[i]))) if k2 is None else k2
        second[i:i + 1] = [second[i][:k2].strip(), second[i][k2:].strip()]
    first[i:i + 1] = [first[i][:k].strip(), first[i][k:].strip()]
    for run in (one, two):
        if i < len(run["boxes"]):
            x0, y0, x1, y1 = run["boxes"][i]
            run["boxes"][i:i + 1] = [[x0, y0, cut, y1], [cut + 1, y0, x1, y1]]
            run["scores"][i:i + 1] = [run["scores"][i]] * 2
    return first, second, one, two, i + 1


def apart_below(boxes: list[list[int]], i: int, right: list[int]) -> bool:
    """Whether a left line stands apart below the right column, with a blank above it and
    nothing beside it: it is the document's next part, not more of the heading."""
    if not right:
        return False
    b = boxes[i]
    high = b[3] - b[1]
    beside = any(same_height(b, o) for o in boxes[max(0, i - 1):i] + boxes[i + 1:i + 2])
    return b[1] - boxes[right[-1]][3] > 2 * high and b[1] - boxes[i - 1][3] > 0.75 * high and not beside


def heading_order(boxes: list[list[int]], at: float | None, title: int | None) -> list[int]:
    """The boxes' indices in reading order: what comes above the heading, a two-column
    heading's left block, then its right, then all the rest as detected. The heading starts
    at the title's height, when there is a title, so a line above it stays out; it ends at
    the first line across the gutter, past the top part of the page, or apart below the
    right column. A block too large
    for a heading, or with nothing on one side, leaves the page as it was, since a table is
    read by rows."""
    order = list(range(len(boxes)))
    if at is None:
        return order
    start = 0
    if title is not None and title < len(boxes):
        above = boxes[title][1] - 0.5 * (boxes[title][3] - boxes[title][1])
        start = next(i for i, b in enumerate(boxes) if (b[1] + b[3]) / 2 > above or i == title)
    limit = heading_limit(boxes)
    left, right, end = [], [], len(boxes)
    for i in range(start, len(boxes)):
        b = boxes[i]
        if b[1] > limit or (b[0] < at < b[2]) or (title is not None and b[2] <= at and apart_below(boxes, i, right)):
            end = i
            break
        (left if b[2] <= at else right).append(i)
    if not left or not right or max(len(left), len(right)) < 2 or len(left) + len(right) > HEADING_LINES:
        return order
    return order[:start] + left + right + order[end:]


def same_boxes(one: list[list[int]], two: list[list[int]]) -> bool:
    return len(one) == len(two) and all(abs(x - y) <= ALIGN_PIXELS for p, q in zip(one, two) for x, y in zip(p, q))


def join_page(first: list[str], second: list[str], one: dict, two: dict, page: np.ndarray,
              sure: float) -> dict:
    """The page's text, which of its lines are unsure, and each line's box (for the tool's
    review page; runs leave it out)."""
    title = find_title(first, one["boxes"])
    at = gutter(page, one["boxes"], title is not None)
    first, second, one, two, title = split_title(first, second, one, two, at, title)
    aligned = len(first) == len(second) == len(one["boxes"]) and same_boxes(one["boxes"], two["boxes"])
    colour = has_colour(page)
    lines, unsure, kept = [], [], []
    for i in heading_order(one["boxes"], at, title):
        if i >= len(first):
            continue
        if colour and mostly_red(page, one["boxes"][i]):
            continue
        kept.append(one["boxes"][i])
        if not aligned:
            lines.append(first[i])
            unsure.append(len(lines) - 1)
            continue
        same = digits(first[i]) == digits(second[i])
        lines.append(first[i] if same else second[i])
        if not same or numbers(first[i]) != numbers(second[i]) or min(one["scores"][i], two["scores"][i]) < sure:
            unsure.append(len(lines) - 1)
    return {"text": "\n".join(lines), "unsure": unsure, "no_colour": not colour, "boxes": kept}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set")
    ap.add_argument("--first", required=True, help="the run whose line stands when digits agree")
    ap.add_argument("--first-boxes", required=True)
    ap.add_argument("--second", required=True, help="the run whose line stands when they differ")
    ap.add_argument("--second-boxes", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sure", type=float, default=0.95, help="the score both readers must reach")
    a = ap.parse_args()
    second = {(r["sha256"], r["page"]): r["text"] for r in read_jsonl(a.second)}
    rows = []
    for r in read_jsonl(a.first):
        k = (r["sha256"], r["page"])
        stem = f"{k[0][:16]}-p{k[1]}"
        with open(os.path.join(a.first_boxes, stem + ".json"), encoding="utf-8") as fh:
            one = json.load(fh)
        with open(os.path.join(a.second_boxes, stem + ".json"), encoding="utf-8") as fh:
            two = json.load(fh)
        page = np.array(Image.open(os.path.join(a.images, stem + ".png")).convert("RGB"), dtype=int)
        joined = join_page((r["text"] or "").split("\n"), (second.get(k) or "").split("\n"), one, two, page, a.sure)
        del joined["boxes"]
        rows.append({"sha256": k[0], "page": k[1], **joined})
    write_jsonl(a.out, rows)


if __name__ == "__main__":
    main()
