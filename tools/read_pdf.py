"""Read one disclosure PDF into a draft: the tool of v0.2 (METHOD.md, "The tool").

Each page is labelled as at step 1 (tools/label_pages.py): signature box out, text
layer or scan, document, skip. Then:

- **Skipped** pages keep their reason and no text.
- **Text-layer** pages are taken as they are, not read, unless step 5's syllable check
  finds the layer broken, or the layer came from OCR (text_from_paper): those are read
  as scans.
- **Scan** pages are rendered as tools/pages.py does (300 dpi, signature box painted
  white), read by stock PaddleOCR and the round 1 recogniser, and joined by
  tools/two_readers.py's join_page: the candidate that passed step 9 on F. Its text and
  unsure lines are as measured; it also returns each line's box, for the review page.

Then, as the reader gets it:

- **Unsure lines** keep their words, and every number on them is left empty ("[?]").
- **Heading fields** per document (tools/fields.py), from its first page that has any.
  Every field found is filled only to be confirmed by a person: the company passed on F
  at its limit, then was wrong 2 of 5 times in a spot check, so it is not left to stand.
- **English pages** are marked: a page of 30+ words where under 40% of the syllables carry
  a tone mark (Vietnamese pages on sets C to F and the first drafts: .63 and up; English:
  .27 and under). They are read all the same, but never measured.
- **The listing** (tools/listing.py), when the filing's symbol is given: each
  document's filled company is marked as matching the listing's name for it or not, and
  the draft names the filer as listed. A match on a page read by OCR stands, state
  "listing", and is not sent to the person: set G passed, provisionally (METHOD.md, "The
  company check"). A match on a text layer was not tested: it stays a label,
  to confirm. The listed name never fills a field.
- **Notes** say what was measured and what was not.

Writes <out>/<pdf name>.draft.json and <out>/images/ with each page not skipped, as
rendered; the review page (tools/review.py) works from these.

--look (statements, profile 2) gives each unsure line with a number a second look: a lone
dash in a narrow box is told by its ink, one short bar, and is zero; any other is cropped
from the page and read by both readers again, and stands only if it passes the same test.
Then each row code left empty ([?]) is read on its own crop by both readers; it stands
only if they agree on 2 or 3 digits that fall between the sure codes read above and below
it in its statement and that it has not read already (codes rise down a statement; the
scores are not used). Where the two differ and sure codes stand on both sides, the one
read that fits stands.
Then each statement cell the readers found nothing in is looked at: if its only ink is one
dash, a sure "-" line is added there (a dash too small for the detector to box at all).
A read page whose image is coarser than COARSE_PPI (a phone photo, a scan at low resolution)
is marked "coarse" and the draft says so: its figures are left empty, not guessed.
--again does this to drafts already made, from their images. It is off by default: the
disclosure tool stays the candidate F measured, and --check does not use it.

--check SET RUN reads the kept scan pages of a set through the same path and compares
them with a scored run, text and unsure lines, byte for byte: the tool must be the
measured candidate, not a new one. Each page must also have one box per line.

Usage:
  .venv-paddle/bin/python -I tools/read_pdf.py some.pdf --out cache/drafts [--symbol ABC]
  .venv-paddle/bin/python -I tools/read_pdf.py --check sets/disclosures-f.jsonl data/runs/two-readers-f.jsonl
  .venv-paddle/bin/python -I tools/read_pdf.py --again cache/drafts/*.draft.json
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone

from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from label_pages import image_list, kept_text, label_file, read_pages, signature_boxes  # noqa: E402
from records import page_key, read_jsonl  # noqa: E402
from sample_pairs import SCAN_CHARS_PER_PAGE, Client, sha256_of  # noqa: E402
from syllables import normal, page_check  # noqa: E402
from truth_b import render  # noqa: E402
from fields import extract  # noqa: E402
from listing import listing, matches  # noqa: E402
from records import NUMBER_RE, numbers  # noqa: E402
from two_readers import digits, join_page  # noqa: E402
from fs_rows import LEAD_CODE_RE, TILT_PAIRS, TRAIL_LABEL_RE, is_amount, statements_of  # noqa: E402
from fs_codes import form_of  # noqa: E402
from fs_sums import BLANK as NO_INK, SUMS, demands  # noqa: E402

ROUND1 = "cache/train/output/inference"    # tools/train_rec.sh's export of round 1
ROUND1_SHA256 = "bca2e21abe007fb6e1c79efc5427415b760ef3c8c2278fe340afbc7ce6354a4f"   # the weights F was read with
SURE = 0.95                                 # two_readers.py's --sure, as measured
BROKEN = 0.005                              # step 5: a page over this share of broken words is broken
BLANK = "[?]"
DASH_TEXT = ("-", "–")                      # read as zero (fs_rows.value_of)
LOOK_PAD = (40, 15)                         # px at 300 dpi around a line's box, for its second look
CODE_PAD = (12, 8)                          # px around a row code's box, for its look (code_look)
LOST = (80, 10, 3)                          # lost_codes: px from a figure line's right edge to its
                                            # column; px the code's band is widened by; coded rows a
                                            # page needs for its code column to be known
DASH = (80, 8, 6, 60, 6)                    # px: box at most this wide; bar at most this high, from
                                            # this to this wide; margin looked in around the box
COARSE_PPI = 90                             # a read page's image under this many pixels an inch is
                                            # coarse (coarse_pages): on sets H-J, the pages at 75 ppi
                                            # and under read 0-2% of lines sure, at 96 and up 75-84%
ENGLISH_MARKED = 0.40                       # under this share of marked syllables, a page is English
ENGLISH_WORDS = 30                          # ... if it has words enough to tell
NOTES = [
    "Machine read. Check what matters before relying on it.",
    "Numbers on sure lines: 1 wrong of 535 on one test set (F), at most about 1.1% at 95%; "
    "that one was the date in a digital signature box, printed but not the document's.",
    "A number can be missing with nothing to mark it: 98.7% of numbers were found on F, and a "
    "line the detector never boxes leaves no trace.",
    "Every number on an unsure line is left empty, [?], for a person.",
    "Weak spots: the heading's document number and dates. Heading fields are found by a rule "
    "and every one goes to a person: the company passed its test on F at the limit (1 wrong of "
    "27), then was wrong 2 of 5 times in a spot check of other files.",
    "Text-layer pages are taken as filed, not read, and were not measured as output.",
    "Tone marks and words were not checked by a person on any test set.",
    "Pages of kinds the test sets left out (lists of people, identity fields, English pages) "
    "are read but were never measured; a list of people on a scan is not found as one.",
]
LOOK_NOTE = ("Second look (--look): a number left empty is read once more, a lone dash by its ink and "
             "anything else on its own crop by both readers under the same test; an amount read with "
             "marks stuck to it, beside a red stamp, is read once more with the red ink taken out, and "
             "stands only as a whole amount. Measured on the statements of one set (H) only, the set "
             "it was made on; not on disclosures.")
OLD_LOOK_NOTES = (                          # a second look by an older rule: looked at again
    "Second look (--look): a number left empty is read once more, a lone dash by its ink and "
    "anything else on its own crop by both readers under the same test. Measured on the "
    "statements of one set (H) only, the set it was made on; not on disclosures.",
)
RED = (40, 200)                             # stamp ink: red this far over green and blue; a crop
                                            # with this many such pixels lies under a stamp
STUCK_RE = re.compile(r"\s*(\S{0,4}?)\s*(\(?-?\d{1,3}(?:[.,]\d{3})+\)?)\s*(\S{0,4})\s*")
CELL = (0.8, 30, 4)                         # a cell (cell_parts): this share of the gap to the column
                                            # before, px past the column's right edge, px around the code
CELL_EDGE = 3                               # px: a part the cell's edge cuts, this thin, is not the cell's
CELL_DUST = 4                               # px: ink with no side longer is dust
CELL_NOTE = ("Dash cells (--look): a statement cell the readers found nothing in, whose only ink is one "
             "short bar, on the code's line or on the line of a figure its row holds, followed across a "
             "page scanned askew, is a dash, zero; ruled lines (also where they meet at a cell's corners), "
             "dust, and figures other rows hold are not counted as its ink. Measured on the statements "
             "of sets H to J, the sets it was made on; never on a test set.")
OLD_CELL_NOTES = (                          # dash cells by an older rule or wording: looked at again
    "Dash cells (--look): a statement cell the readers found nothing in, whose only ink is one "
    "short bar, on the code's line or on the line of a figure its row holds, followed across a "
    "page scanned askew, is a dash, zero; ruled lines (also where they meet at a cell's corners), "
    "dust, and figures other rows hold are not counted as its ink. Measured on the statements "
    "of sets H to J only, the sets it was made on (docs/" "plan.md, dash cells in a ruled "
    "table).",                              # split: published code names no private file
    "Dash cells (--look): a statement cell the readers found nothing in, whose only ink is one "
    "short bar, on the code's line or on the line of a figure its row holds, followed across a "
    "page scanned askew, is a dash, zero; ruled lines (also where they meet at a cell's corners), "
    "dust, and figures other rows hold are not counted as its ink. Measured on the statements "
    "of sets H to J only, the sets it was made on.",
    "Dash cells (--look): a statement cell the readers found nothing in, whose only ink is one "
    "short bar, on the code's line or on the line of a figure its row holds, followed across a "
    "page scanned askew, is a dash, zero; ruled lines, dust, and figures other rows hold are "
    "not counted as its ink. Measured on the statements of one set (H) only, the set it was "
    "made on.",
    "Dash cells (--look): a statement cell the readers found nothing in, whose only ink is one "
    "short bar, is a dash, zero; ruled lines, dust, and figures other rows hold are not counted "
    "as its ink. Measured on the statements of one set (H) only, the set it was made on.",
    "Dash cells (--look): a statement cell the readers found nothing in, whose only ink is one "
    "short bar, is a dash, zero. Measured on the statements of one set (H) only, the set it was "
    "made on.",)
BLANK_INK = 200                             # blank_cells: a pixel darker than this is ink; lighter than
                                            # ink()'s 128, so that faint print is never called blank
RULE = (121, 101, 5, 7)                     # px: ruled_lines: a dark run this long across, or down, is a
                                            # ruled line; first closed: its steps on a page scanned askew,
                                            # and gaps this long along it (a rule printed faint, broken)
BLANK_RULE = (60, 2, 8)                     # px: a dark run this long across is a ruled line, taken out
                                            # with this much above and below it; a stroke down the band
                                            # and this far past both its ends is a ruled line too
BLANK_NOTE = ("Blank cells (--look): a statement cell the readers found nothing in is looked at from the "
              "row above down to the row below. With no ink there but ruled lines and dust, it is marked "
              "blank on its page (blank_cells). The cell stays empty: a blank is not a zero the filing "
              "printed, and what counts it as one says so.")
SUM_NOTE = ("Cells a sum leaves open (--look): where every cell of one of the form's sums is read but one, "
            "that cell's crop is read once more, and a reading is kept only when it is the figure the "
            "sum leaves it, digit for digit. Never filled from the sum alone. Made on set H; not yet "
            "measured on another.")
COARSE_NOTE = ("Coarse pages (--look): a read page whose image is under about %d pixels an inch, as a "
               "phone photo or a scan at low resolution often is, is marked coarse. The readers are "
               "unsure of almost all of such a page: its figures are left empty, not guessed, and a "
               "statement on it may not be found. Ask for a finer copy." % COARSE_PPI)
CODE_NOTE = ("Codes by order (--look): a row code left empty is read once more on its own crop and stands "
             "only if both readers agree on 2 or 3 digits between the sure codes above and below it in "
             "its statement, not read in it already; where they differ, between sure codes on both "
             "sides, the one read that fits. A figure line no row holds, with digits in the code "
             "column beside it that no line was found on, has them read so too. Measured on the "
             "statements of one set (H) only, the set it was made on.")
OLD_CODE_NOTES = ("Codes by order (--look): a row code left empty is read once more on its own crop and stands "
                  "only if both readers agree on 2 or 3 digits between the sure codes above and below it in "
                  "its statement, not read in it already; where they differ, between sure codes on both "
                  "sides, the one read that fits. Measured on the statements of one set (H) only, the set "
                  "it was made on.",
                  "Codes by order (--look): a row code left empty is read once more on its own crop and stands "
                  "only if both readers agree on 2 or 3 digits between the sure codes above and below it on "
                  "its page. Measured on the statements of one set (H) only, the set it was made on.",)


def layer_broken(text: str) -> bool:
    c = page_check(text)
    return c["broken"] > BROKEN or c["foreign"] > 0


def stuck_on(text: str) -> bool:
    """An amount with up to 4 marks stuck to it, on either side: "(1.234.567)Ư", "Y 1.234"."""
    m = STUCK_RE.fullmatch(text)
    return bool(m and (m.group(1) or m.group(3))) and not is_amount(text.strip())


def unstamped(piece: Image.Image) -> tuple[Image.Image, bool]:
    """The crop with its red ink made white, and whether it lies under a stamp; a crop with
    no stamp comes back as it was."""
    a = np.array(piece, dtype=int)
    red = a[..., 0] - np.maximum(a[..., 1], a[..., 2]) > RED[0]
    if red.sum() < RED[1]:
        return piece, False
    a[red] = 255
    return Image.fromarray(a.astype(np.uint8)), True


class Readers:
    """Stock PaddleOCR and round 1, as tools/run_engine.py runs them."""

    def __init__(self, rec_model: str, device: str):
        import hashlib
        with open(os.path.join(rec_model, "inference.pdiparams"), "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != ROUND1_SHA256:
                sys.exit(f"{rec_model}: not the round 1 weights F was read with")
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import PaddleOCR
        common = dict(lang="vi", device=device, use_doc_orientation_classify=False,
                      use_doc_unwarping=False, use_textline_orientation=False)
        self.tuned = PaddleOCR(text_recognition_model_name="PP-OCRv6_medium_rec",
                               text_recognition_model_dir=rec_model, **common)
        self.stock = PaddleOCR(**common)

    @staticmethod
    def _read(ocr, png: str) -> tuple[list[str], dict]:
        r = ocr.predict(png)[0].json["res"]
        boxes = json.loads(json.dumps({"boxes": r["rec_boxes"], "scores": r["rec_scores"]}))
        return normal("\n".join(r["rec_texts"])).split("\n"), boxes

    def page(self, png: str) -> dict:
        """The joined page: text, unsure line indexes, no_colour."""
        first, one = self._read(self.tuned, png)
        second, two = self._read(self.stock, png)
        page = np.array(Image.open(png).convert("RGB"), dtype=int)
        return join_page(first, second, one, two, page, SURE)

    def _line(self, ocr, png: str) -> tuple[str, float]:
        r = ocr.predict(png)[0].json["res"]
        return normal(" ".join(r["rec_texts"])), min(r["rec_scores"] or [0])

    def look_again(self, png: str, lines: list[dict], stamps_only: bool = False) -> list[dict]:
        """Each unsure line with a number left empty, or read as a dash, once more: a lone dash
        by its ink, else its own crop read by both readers under the page's test. A line that
        passes is sure, and says how ("look").

        An amount read with a few marks stuck to it ("(1.234.567)Ư"), sure or not, is looked
        at only beside a red stamp: its crop is read with the red ink taken out, and stands
        only when both readers make the same whole amount of it. A crop under a stamp is read
        so for any line looked at. stamps_only: only lines under a stamp, for a draft looked
        at by an older rule, whose other lines that look has had, and the code look after it
        may have cut (a code put on its own line leaves the label's rest unread)."""
        img = Image.open(png).convert("RGB")
        gray = np.array(img.convert("L"))
        out = []
        with tempfile.TemporaryDirectory() as tmp:
            crop = os.path.join(tmp, "line.png")
            for ln in lines:
                dash = ln["text"].strip() in DASH_TEXT        # no number in it to leave empty
                stuck = bool(ln["box"]) and stuck_on(ln["text"])
                if not stuck and (ln["sure"] or not (BLANK in ln["text"] or dash) or not ln["box"]):
                    out.append(ln)
                    continue
                if not stuck and not stamps_only and one_dash(gray, ln["box"]):
                    out.append({**ln, "text": "-", "sure": True, "look": "dash"})
                    continue
                if dash:                                       # a crop of a lone mark reads as nothing
                    out.append(ln)
                    continue
                x0, y0, x1, y1 = ln["box"]
                cut = (max(0, x0 - LOOK_PAD[0]), max(0, y0 - LOOK_PAD[1]),
                       min(img.width, x1 + LOOK_PAD[0]), min(img.height, y1 + LOOK_PAD[1]))
                if cut[2] <= cut[0] or cut[3] <= cut[1]:                # a box off the page: stays unsure
                    out.append(ln)
                    continue
                piece, stamped = unstamped(img.crop(cut))
                if (stuck or stamps_only) and not stamped:     # marks not from a stamp: left as read
                    out.append(ln)
                    continue
                piece.save(crop)
                (t1, s1), (t2, s2) = self._line(self.tuned, crop), self._line(self.stock, crop)
                if (digits(t1) and digits(t1) == digits(t2) and numbers(t1) == numbers(t2)
                        and min(s1, s2) >= SURE and (not stuck or is_amount(t1.strip()))):
                    out.append({**ln, "text": t1.strip() if stuck else t1, "sure": True,
                                "look": "stamp" if stamped else "crop"})
                else:
                    out.append(ln)
        return out

    def code(self, png: str, box: list) -> tuple[str, str] | None:
        """What each reader makes of a row code's crop (tuned, stock); None off the page."""
        img = Image.open(png).convert("RGB")
        x0, y0, x1, y1 = box
        cut = (max(0, int(x0 - CODE_PAD[0])), max(0, int(y0 - CODE_PAD[1])),
               min(img.width, int(x1 + CODE_PAD[0])), min(img.height, int(y1 + CODE_PAD[1])))
        if cut[2] <= cut[0] or cut[3] <= cut[1]:
            return None
        with tempfile.TemporaryDirectory() as tmp:
            crop = os.path.join(tmp, "code.png")
            img.crop(cut).save(crop)
            return self._line(self.tuned, crop)[0].strip(), self._line(self.stock, crop)[0].strip()


    def figure(self, png: str, box: list) -> tuple[str, str] | None:
        """What each reader makes of a figure's crop (tuned, stock), its red ink taken out
        where it lies under a stamp; None off the page."""
        img = Image.open(png).convert("RGB")
        x0, y0, x1, y1 = box
        cut = (max(0, int(x0 - LOOK_PAD[0])), max(0, int(y0 - LOOK_PAD[1])),
               min(img.width, int(x1 + LOOK_PAD[0])), min(img.height, int(y1 + LOOK_PAD[1])))
        if cut[2] <= cut[0] or cut[3] <= cut[1]:
            return None
        with tempfile.TemporaryDirectory() as tmp:
            crop = os.path.join(tmp, "figure.png")
            unstamped(img.crop(cut))[0].save(crop)
            return self._line(self.tuned, crop)[0].strip(), self._line(self.stock, crop)[0].strip()


NO_CODE = (-1, 1000)                        # the bounds past a statement's first and last sure codes


def code_between(rows: list[dict], k: int) -> tuple[int, int, bool] | None:
    """The sure codes nearest above and below row k in its statement, on any of its pages,
    with NO_CODE past either end, and whether both were read; None when they leave no room."""
    def near(step):
        j = k + step
        while 0 <= j < len(rows):
            c = rows[j]["code"]
            if rows[j].get("code_sure") and c and c.isdigit():
                return int(c)
            j += step
        return None
    a, b = near(-1), near(1)
    lo, hi = NO_CODE[0] if a is None else a, NO_CODE[1] if b is None else b
    return (lo, hi, a is not None and b is not None) if hi - lo >= 2 else None


def code_by_order(reads: tuple[str, str], room: tuple[int, int, bool], used: set[str]) -> str | None:
    """The code a row stands on, from the readers' reads of its crop: 2 or 3 digits strictly
    inside the room and not a code its statement already has. Both readers agreeing on it, or,
    where they differ and sure codes stand on both sides, the one read of the two that fits."""
    lo, hi, both = room
    fits = [t for t in dict.fromkeys(reads)
            if re.fullmatch(r"\d{2,3}", t) and lo < int(t) < hi and t not in used]
    if reads[0] == reads[1]:
        return fits[0] if fits else None
    return fits[0] if both and len(fits) == 1 else None


def put_code(lines: list[dict], box: list, code: str) -> bool:
    """The [?] at a row code's box read as code, in place: the line that is the box, or a
    line with its label that fs_rows split there (code first or last), split in two."""
    for i, ln in enumerate(lines):
        b, t = ln.get("box"), ln["text"].strip()
        if not b or BLANK not in t or not (b[0] <= box[0] + 1 and box[2] <= b[2] + 1
                                            and b[1] <= box[1] + 1 and box[3] <= b[3] + 1):
            continue
        code_line = {**ln, "text": code, "box": list(box), "sure": True, "look": "code by order"}
        if t == BLANK:
            lines[i] = code_line
            return True
        lead, trail = LEAD_CODE_RE.match(t), TRAIL_LABEL_RE.fullmatch(t)
        if lead and lead.group(1) == BLANK and abs(box[0] - b[0]) <= 1:
            lines[i:i + 1] = [code_line, {**ln, "text": lead.group(2), "box": [box[2], b[1], b[2], b[3]]}]
            return True
        if trail and trail.group(2) == BLANK and abs(box[2] - b[2]) <= 1:
            lines[i:i + 1] = [{**ln, "text": trail.group(1), "box": [b[0], b[1], box[0], b[3]]}, code_line]
            return True
    return False


def looked_at(pages: list[dict]) -> dict[int, dict]:
    """Page -> the read pages the looks go over: not a coarse one (coarse_pages), whose
    figures are left empty, never filled from a look."""
    return {p["page"]: p for p in pages if p.get("kind") == "read" and not p.get("coarse")}


def code_look(pages: list[dict], images: str, readers: Readers) -> int:
    """Row codes left empty, read by order (Readers.code, code_between, code_by_order), in
    place; returns how many stand."""
    by_page = looked_at(pages)
    found = 0
    for s in statements_of({"pages": pages}, images=images):
        rows = s["rows"]
        used = {r["code"] for r in rows if r.get("code_sure") and r["code"]}
        for k, r in enumerate(rows):
            p = by_page.get(r["page"])
            room = code_between(rows, k) if r["code"] is None and p is not None else None
            reads = readers.code(page_png(images, p), r["code_box"]) if room else None
            code = code_by_order(reads, room, used) if reads else None
            if code and put_code(p["lines"], r["code_box"], code):
                used.add(code)
                found += 1
        for pg in s["pages"]:
            p = by_page.get(pg)
            if p is None:
                continue
            gray = np.array(Image.open(page_png(images, p)).convert("L"))
            for k, box in lost_codes(s, pg, p["lines"], gray):
                room = code_between(rows[:k] + [{"code": None}] + rows[k:], k)
                reads = readers.code(page_png(images, p), box) if room else None
                code = code_by_order(reads, room, used) if reads else None
                if code:
                    p["lines"].append({"text": code, "box": box, "sure": True, "look": "lost code"})
                    used.add(code)
                    found += 1
    return found


def lost_codes(s: dict, page: int, lines: list[dict], gray: np.ndarray) -> list[tuple[int, list[int]]]:
    """Where a row's code was printed but no line was found on it, on one page of statement s:
    (the index in s's rows the row would take, the box of the code's ink).

    A figure line ending at one of the page's columns (LOST[0]) that no row holds, neither
    beside a row's code nor as one of its figures, with no line over the left half of the
    code column beside it, and in that column, left of any line starting in it, two or more
    parts of digit size (half to 1.3 times the page's code height, no wider than tall), none
    cut by the band's top or bottom edge (a ruled line). The code column is the span of the
    page's codes, so a page with fewer than LOST[2] codes has none."""
    rows = s["rows"]
    here = [r for r in rows if r["page"] == page]
    boxes = [r["code_box"] for r in here if r["code"]]
    cols = next((c["x"] for c in s["columns"] if c["page"] == page), [])
    if len(boxes) < LOST[2]:
        return []
    cx0, cx1 = min(b[0] for b in boxes), max(b[2] for b in boxes)
    h = float(np.median([b[3] - b[1] for b in boxes]))
    held = [v["box"] for r in here for v in r["values"] if v and v.get("box")]
    m = LOST[1]
    out = []
    for ln in sorted((ln for ln in lines if ln.get("box")), key=lambda ln: ln["box"][1]):
        x0, y0, x1, y1 = ln["box"]
        if (not is_amount(ln["text"].strip()) or not any(abs(x1 - c) <= LOST[0] for c in cols)
                or ln["box"] in held
                or any(min(y1, b[3]) - max(y0, b[1]) > 0.3 * (y1 - y0) for b in [r["code_box"] for r in here])
                or any(abs(y0 - b[1]) < h for _, b in out)):
            continue
        over = [o["box"][0] for o in lines if o.get("box") and o["box"][2] > cx0 and o["box"][0] < cx1
                and min(y1, o["box"][3]) - max(y0, o["box"][1]) > 0.3 * (y1 - y0)]
        if any(x < (cx0 + cx1) / 2 for x in over):
            continue
        right = min([cx1 + m] + [x - 4 for x in over])
        parts = [b for b in ink(gray, cx0 - 2 * m, y0 - m, right, y1 + m) or []
                 if 0.5 * h <= b[3] - b[1] <= 1.3 * h and b[2] - b[0] <= h and b[1] > y0 - m and b[3] < y1 + m]
        if len(parts) < 2:
            continue
        box = [min(b[0] for b in parts), min(b[1] for b in parts),
               max(b[2] for b in parts), max(b[3] for b in parts)]
        k = next((i for i, r in enumerate(rows) if r["page"] == page and r["code_box"][1] > box[1]),
                 max((i + 1 for i, r in enumerate(rows) if r["page"] <= page), default=0))
        out.append((k, box))
    return out


class Ruled(list):
    """A part's box [x0, y0, x1, y1], the part joined to a ruled line: never a dash."""


def ruled_lines(gray: np.ndarray) -> np.ndarray:
    """The page's ruled lines, as a mask: runs darker than BLANK_INK RULE[0] px across or
    RULE[1] px down, each found after closing gaps of RULE[2] px across its run (the steps of
    a line scanned askew) and RULE[3] px along it (a faint rule, broken), and widened by
    BLANK_RULE[1] px on its sides. Found on the whole page, not on a cell's
    crop: a line is long on the page, however short its piece inside a cell."""
    from scipy import ndimage
    a = (gray < BLANK_INK).astype(np.uint8)      # a rule printed faint is a rule all along
    q, out = BLANK_RULE[1], np.zeros(a.shape, bool)

    def close(m, n, axis):
        return ndimage.minimum_filter1d(ndimage.maximum_filter1d(m, n, axis=axis), n, axis=axis)
    for axis, n in ((1, RULE[0]), (0, RULE[1])):
        side = 1 - axis
        c = close(close(a, RULE[2], side), RULE[3], axis)
        run = ndimage.maximum_filter1d(ndimage.minimum_filter1d(c, n, axis=axis), n, axis=axis)
        out |= ndimage.maximum_filter1d(run, 2 * q + 1, axis=side).astype(bool)
    return out


def ink(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int,
        ruled: np.ndarray | None = None) -> list[list[int]] | None:
    """The boxes [x0, y0, x1, y1] of the ink parts in a region, specks out; None off the page.
    With ruled (the page's ruled_lines), a part that is ruled lines alone is left out: rules
    meeting at a table's corner join into one part the size of the cell. A part joined to a
    rule with more ink than dust besides (a tick, a stub past a corner, a digit touching it)
    is kept whole, as Ruled: what it is cannot be told apart from the rule."""
    from scipy import ndimage
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    a = gray[y0:int(y1), x0:int(x1)]
    if not a.size:
        return None
    lab, _ = ndimage.label(a < 128)
    boxes = []
    for n, s in enumerate(ndimage.find_objects(lab), 1):
        b = [x0 + s[1].start, y0 + s[0].start, x0 + s[1].stop, y0 + s[0].stop]
        if (b[2] - b[0]) * (b[3] - b[1]) < 6:
            continue
        if ruled is not None:
            on = ruled[b[1]:b[3], b[0]:b[2]]
            part = lab[s] == n
            if (part & on).any():
                rest, _ = ndimage.label(part & ~on)
                if not any(max(r[0].stop - r[0].start, r[1].stop - r[1].start) > CELL_DUST
                           for r in ndimage.find_objects(rest)):
                    continue                     # ruled lines alone
                b = Ruled(b)
        boxes.append(b)
    return boxes


def a_dash(parts: list[list[int]] | None) -> list[int] | None:
    """The one part, if the ink is one short horizontal bar: a dash, printed for zero."""
    if not parts or len(parts) != 1 or isinstance(parts[0], Ruled):
        return None
    w, h = parts[0][2] - parts[0][0], parts[0][3] - parts[0][1]
    return parts[0] if 2 <= h <= DASH[1] and DASH[2] <= w <= DASH[3] and w >= 1.8 * h else None


def one_dash(gray: np.ndarray, box: list[int]) -> bool:
    """The ink in a narrow box is one short horizontal bar: a dash, printed for zero."""
    x0, y0, x1, y1 = (int(v) for v in box)
    if x1 - x0 > DASH[0]:
        return False
    p = DASH[4]
    return a_dash(ink(gray, x0 - p, y0 - p, x1 + p, y1 + p)) is not None


def cell_parts(gray: np.ndarray, rows: list[dict], k: int, xs: list[float], i: int,
               reach: bool = True, band: tuple[float, float] | None = None,
               shift: float = 0.0, ruled: np.ndarray | None = None) -> list[list[int]] | None:
    """The ink in row k's cell in column i, rules out (None off the page).

    The cell is the code's box widened by CELL[2], and from CELL[1] past the column's right
    edge leftward over CELL[0] of the gap to the column before (the first column takes the
    gap to the second). Where the row before, on the same page, has its figure in this
    column, the cell reaches up to that row's code: a label wrapped over two lines may print
    its figure on the first (not when reach is False). With band (y0, y1), the cell is that
    band instead, with no reach: the line of a figure the row holds (row_lines). Left out,
    as not this cell's: a stroke as tall as the cell and narrow, a thin bar wider than twice a dash,
    or a stroke no wider than CELL_EDGE right of the column's edge, where no digit is printed
    (ruled lines); ink inside a figure another
    row of the page holds in this column; a stroke or sliver the cell's top or bottom edge
    cuts (CELL_EDGE: a broken ruled line, the foot of the line above); and dust, no side
    longer than CELL_DUST, which no figure is made of alone. A part that is ruled lines
    alone is left out, and one joined to a ruled line is never a dash (ink's ruled; the
    page's ruled_lines, given as ruled or found here). On a page scanned askew, the
    code's line lies shift px lower at the column (column_shifts); the cell follows it when
    that is more than the cell's margin (CELL[2])."""
    r = rows[k]
    gap = xs[i] - xs[i - 1] if i else (xs[1] - xs[0] if len(xs) > 1 else 400)
    dy = shift if not band and abs(shift) > CELL[2] else 0.0   # within the margin: the code's line
    y0, y1 = band or (r["code_box"][1] - CELL[2] + dy, r["code_box"][3] + CELL[2] + dy)
    before = rows[k - 1] if k else None
    if (reach and not band and before and before.get("code") and before["page"] == r["page"]
            and i < len(before["values"]) and before["values"][i]):
        y0 = min(y0, before["code_box"][3] + CELL[2] + dy)
    top, bottom = max(int(y0), 0), min(int(y1), gray.shape[0])   # the edges ink() cut at
    tall = 0.9 * (bottom - top)
    parts = ink(gray, xs[i] - CELL[0] * gap, y0, xs[i] + CELL[1], y1,
                ruled=ruled_lines(gray) if ruled is None else ruled)
    if parts is None:
        return None
    held = [v["box"] for o in rows if o is not r and o["page"] == r["page"]
            for v in o["values"][i:i + 1] if v and v.get("box")]
    p = CELL[2]

    def other(b):
        w, h = b[2] - b[0], b[3] - b[1]
        return ((w <= DASH[1] and h >= tall) or (h <= DASH[1] and w > 2 * DASH[3])
                or (w <= CELL_EDGE and b[0] >= xs[i] + CELL[2])
                or any(f[0] - p <= b[0] and b[2] <= f[2] + p and f[1] - p <= b[1] and b[3] <= f[3] + p
                       for f in held)
                or ((b[1] <= top or b[3] >= bottom) and (w <= CELL_EDGE or h <= CELL_EDGE))
                or max(w, h) <= CELL_DUST)
    return [b for b in parts if not other(b)]


def column_shifts(rows: list[dict], page: int, xs: list[float], tilt: float) -> list[float]:
    """Per column, how far below its code a row's figure lies on this page: the median over
    the amounts the page's rows hold there within half a code's height of where the page's
    tilt puts them (a wrapped label's figure, a line lower, left out), from TILT_PAIRS of
    them. Where most lie further off, the codes stand off the figures' lines (a code at the
    top of a tall ruled cell, its figure lower) and the median is of them all; else, too
    few, it is where the tilt puts them. Measured, it holds what the tilt alone misses (a
    page bowed or stretched in the scan); figures sit within a px or two of it."""
    here = [r for r in rows if r["page"] == page and r.get("code_box")]
    cx = sorted((r["code_box"][0] + r["code_box"][2]) / 2 for r in here)
    cx = cx[len(cx) // 2] if cx else 0.0
    hs = sorted(r["code_box"][3] - r["code_box"][1] for r in here)
    half = hs[len(hs) // 2] / 2 if hs else 0.0
    out = []
    for i, x in enumerate(xs):
        gap = x - xs[i - 1] if i else (xs[1] - x if len(xs) > 1 else 400)
        guess = tilt * (x - CELL[0] * gap / 2 - cx)
        every = sorted((v["box"][1] + v["box"][3] - r["code_box"][1] - r["code_box"][3]) / 2
                       for r in here for v in r["values"][i:i + 1]
                       if v and v.get("box") and v["box"][2] - v["box"][0] >= 60)
        near = [o for o in every if abs(o - guess) <= half]
        if len(near) >= TILT_PAIRS:
            out.append(near[len(near) // 2])
        elif len(every) >= TILT_PAIRS and len(near) < len(every) / 2:
            out.append(every[len(every) // 2])      # the codes stand off the figures' lines
        else:
            out.append(guess)
    return out


def row_lines(r: dict, x: float, tilt: float = 0.0) -> list[tuple[float, float]]:
    """Bands at x, as tall as the code's cell, on the lines of figures the row holds that lie
    off the code's own line: a wrapped label prints its figures on its last line, below the
    code. On a page that tilts, lines are followed across by the tilt."""
    c0, c1 = r["code_box"][1], r["code_box"][3]
    cx = (r["code_box"][0] + r["code_box"][2]) / 2
    half = (c1 - c0) / 2 + CELL[2]
    out = []
    for v in r["values"]:
        if v and v.get("box"):
            fx, m = (v["box"][0] + v["box"][2]) / 2, (v["box"][1] + v["box"][3]) / 2
            at = m + tilt * (x - fx)
            if not c0 <= m - tilt * (fx - cx) <= c1 and not any(a <= at <= b for a, b in out):
                out.append((at - half, at + half))
    return out


def image_ppi(path: str) -> dict[int, int]:
    """Page -> the resolution, in pixels an inch, of the finest image that covers most of
    that page as placed (pdfimages -list); pages with no image are left out. Finest, for a
    layered scan: a coarse colour background under a fine black-and-white mask with the text.
    Where pdfimages fails, the resolution is unknown: nothing is returned for the pages it
    did not list, and they are not marked coarse (the read goes on). Inline images are passed
    over: small marks drawn in the page's content, not a scan of it."""
    try:
        imgs = image_list(path, kinds=("image", "mask", "smask", "stencil"))
    except OSError:                         # pdfimages not there to run
        imgs = []
    placed: dict[int, list[tuple[float, int]]] = {}
    for im in imgs:
        if im["xppi"] and im["yppi"] and not im["inline"]:
            placed.setdefault(im["page"], []).append(((im["w"] / im["xppi"]) * (im["h"] / im["yppi"]),
                                                      min(im["xppi"], im["yppi"])))
    return {page: max(ppi for area, ppi in imgs if area >= 0.5 * max(a for a, _ in imgs))
            for page, imgs in placed.items()}


def coarse_pages(path: str, pages: list[dict]) -> int:
    """Marks the read pages whose image is coarser than COARSE_PPI, in place; how many. Run
    before the looks, which pass such a page over (looked_at). Its figures are left empty
    by tools/fs_rows.py, whatever a look by an older run put there."""
    ppi, n = image_ppi(path), 0
    for p in pages:
        if p["kind"] == "read" and p["page"] in ppi and ppi[p["page"]] < COARSE_PPI:
            p["coarse"] = f"its image is about {ppi[p['page']]} pixels an inch: what it holds is left empty, not guessed"
            n += 1
    return n


def dash_cells(pages: list[dict], images: str) -> int:
    """Statement cells the readers found nothing in, looked at (cell_parts): where the only
    ink is one dash, a sure "-" line is added there, which fs_rows reads as zero. In place;
    returns how many. Any other ink, a figure or a mark, leaves the cell empty; ink on the
    lines above, where the cell reaches up, does not when the code's own line holds the dash.
    Last, the line of a figure the row holds in another column (row_lines): that line is the
    row's, so a dash there in this column is its zero. Looked at again until no more are
    found: a dash found is a figure the row below may reach up to."""
    by_page = looked_at(pages)
    grays, added, more = {}, 0, True
    while more:
        more, shifts = 0, {}
        for s in statements_of({"pages": pages}, images=images):
            cols = {c["page"]: c["x"] for c in s["columns"]}
            tilt = {c["page"]: c.get("tilt", 0.0) for c in s["columns"]}
            rows = s["rows"]
            for k, r in enumerate(rows):
                xs, p = cols.get(r["page"]) or [], by_page.get(r["page"])
                if not r.get("code") or not xs or p is None:
                    continue
                if r["page"] not in grays:
                    g = np.array(Image.open(page_png(images, p)).convert("L"))
                    grays[r["page"]] = g, ruled_lines(g)
                (g, ruled), t = grays[r["page"]], tilt.get(r["page"], 0.0)
                if (id(s), r["page"]) not in shifts:
                    shifts[id(s), r["page"]] = column_shifts(rows, r["page"], xs, t)
                sh = shifts[id(s), r["page"]]
                for i in range(len(xs)):
                    if i < len(r["values"]) and r["values"][i]:
                        continue
                    # the code's own line alone holding one dash decides: a figure printed on
                    # a wrapped label's first line leaves that line empty, never a dash under it
                    bar = (a_dash(cell_parts(g, rows, k, xs, i, reach=False, shift=sh[i], ruled=ruled))
                           or a_dash(cell_parts(g, rows, k, xs, i, shift=sh[i], ruled=ruled))
                           or next((b for band in row_lines(r, xs[i] - 100, t)
                                    if (b := a_dash(cell_parts(g, rows, k, xs, i, band=band, ruled=ruled)))),
                                   None))
                    if bar and not any(ln.get("box") == bar for ln in p["lines"]):
                        p["lines"].append({"text": "-", "box": bar, "sure": True, "look": "dash cell"})
                        more += 1
        added += more
    return added


def no_ink(gray: np.ndarray, x0: float, y0: float, x1: float, y1: float,
           edge: tuple[float, float] | None = None) -> bool | None:
    """A region holds no ink but ruled lines and dust (None off the page, or too low to tell).

    Ink is any pixel under BLANK_INK. Taken out as ruled lines: a run BLANK_RULE[0] across,
    with BLANK_RULE[1] above and below it, and a column dark over nine tenths of the region
    stretched BLANK_RULE[2] past its top and bottom, which a digit as tall as the region is
    not. With edge (x, height), right of x, where no digit is printed: a stroke no wider
    than CELL_EDGE, and whatever stands in a column of pixels dark that height in one run
    (a rule that ends in the region: a table's last row, a frame). What is left is ink if
    any part has a side longer than CELL_DUST."""
    from scipy import ndimage
    x0, x1 = max(0, int(x0)), min(gray.shape[1], int(x1))
    y0, y1 = max(0, int(y0)), min(gray.shape[0], int(y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    a = gray[y0:y1, x0:x1] < BLANK_INK
    p = BLANK_RULE[2]
    down = (gray[max(0, y0 - p):y1 + p, x0:x1] < BLANK_INK).mean(axis=0) >= 0.9
    across = ndimage.binary_opening(a, np.ones((1, BLANK_RULE[0]), bool))
    a = a & ~ndimage.binary_dilation(across, np.ones((2 * BLANK_RULE[1] + 1, 1), bool))
    a[:, ndimage.binary_dilation(down, np.ones(2 * BLANK_RULE[1] + 1, bool))] = False
    at = a.shape[1] if edge is None else min(max(int(edge[0]) - x0, 0), a.shape[1])
    if edge is not None and at < a.shape[1]:
        run = ndimage.binary_opening(a[:, at:], np.ones((max(int(edge[1]), 1), 1), bool)).any(axis=0)
        a[:, at:][:, ndimage.binary_dilation(run, np.ones(2 * BLANK_RULE[1] + 1, bool))] = False
    lab, _ = ndimage.label(a)
    return not any(max(s[0].stop - s[0].start, s[1].stop - s[1].start) > CELL_DUST
                   and not (s[1].start >= at and s[1].stop - s[1].start <= CELL_EDGE)
                   for s in ndimage.find_objects(lab))


def blank_cells(pages: list[dict], images: str) -> int:
    """Statement cells left empty, looked at for ink (no_ink): those with none are listed on
    their page as "blank_cells", each the row's code box and the column's place. In place,
    afresh; returns how many. The cell looked at is wider than the one a dash is sought in:
    from the code of the row above to the code of the row below (a code's height past a
    row with none on its side), since a figure may be printed on any line of a wrapped
    label or anywhere in a tall cell; across, as cell_parts takes it, following the page's
    slant (column_shifts)."""
    by_page = looked_at(pages)
    for p in pages:
        p.pop("blank_cells", None)
    grays, found = {}, 0
    for s in statements_of({"pages": pages}, images=images):
        rows, shifts = s["rows"], {}
        for c in s["columns"]:
            xs, p = c["x"], by_page.get(c["page"])
            here = [r for r in rows if r["page"] == c["page"] and r.get("code_box")]
            if not xs or p is None or not here:
                continue
            if c["page"] not in grays:
                grays[c["page"]] = np.array(Image.open(page_png(images, p)).convert("L"))
            g = grays[c["page"]]
            sh = column_shifts(rows, c["page"], xs, c.get("tilt", 0.0))
            for r in here:
                if not r.get("code"):
                    continue
                b = r["code_box"]
                m, h = (b[1] + b[3]) / 2, b[3] - b[1]
                y0 = max((o["code_box"][3] for o in here if o is not r and o["code_box"][3] <= m), default=b[1] - h)
                y1 = min((o["code_box"][1] for o in here if o is not r and o["code_box"][1] >= m), default=b[3] + h)
                y0, y1 = min(y0, b[1]), max(y1, b[3])
                for i, x in enumerate(xs):
                    if i < len(r["values"]) and r["values"][i]:
                        continue
                    gap = x - xs[i - 1] if i else (xs[1] - x if len(xs) > 1 else 400)
                    dy = sh[i] if abs(sh[i]) > CELL[2] else 0.0
                    if no_ink(g, x - CELL[0] * gap, y0 + dy, x + CELL[1], y1 + dy, (x + CELL[2], h)):
                        p.setdefault("blank_cells", []).append({"code_box": b, "column": i})
                        found += 1
    return found


def open_cells(s: dict, form: str | None) -> list[tuple[int, int, set]]:
    """(row, column, the figures a sum leaves it) for each cell of a statement that one of
    the form's sums leaves open (fs_sums.demands). A code on two rows is no cell to ask
    about. With no form told, either form's sums may ask: the reading has the last word."""
    rows = s["rows"]
    at = {}
    for k, r in enumerate(rows):
        if r.get("code"):
            at.setdefault(r["code"], []).append(k)
    out = []
    for i in range(max((len(r["values"]) for r in rows), default=0)):
        values = {}
        for code, ks in at.items():
            r = rows[ks[0]]
            v = r["values"][i] if i < len(r["values"]) else None
            values[code] = ("twice" if len(ks) > 1 else v["value"] if v
                            else NO_INK if i in r.get("blank", ()) else None)
        want = {}
        for f in [form] if form else sorted(SUMS):
            for code, figures in demands(f, s["statement"], values, True).items():
                want.setdefault(code, set()).update(figures)
        out += [(at[code][0], i, figures) for code, figures in sorted(want.items())]
    return out


def figure_of(text: str) -> tuple[int, str] | None:
    """A reader's amount by its digits and sign alone, and the amount written out with its
    separators set right: "(1.234.5.67)" is -1234567, "(1.234.567)". None with anything but
    digits, separators and spaces between its ends, or with no digit other than zero."""
    t = text.strip()
    neg = t.startswith("(") and t.endswith(")") or t.startswith("-")
    core = t[1:-1] if t.startswith("(") and t.endswith(")") else t.lstrip("-")
    d = re.sub(r"[.,\s]", "", core)
    if not d.isdigit() or not int(d):
        return None
    sep = next((ch for ch in core if ch in ".,"), ".")
    d = str(int(d))
    out = sep.join([d[:len(d) % 3]] * bool(len(d) % 3) + [d[k:k + 3] for k in range(len(d) % 3, len(d), 3)])
    return (-int(d), f"({out})") if neg else (int(d), out)


def sum_look(pages: list[dict], images: str, readers: Readers) -> int:
    """Statement cells a sum leaves open (open_cells), read once more: the one line in the
    cell, sure or not, is read on its own crop, and a reader's amount that is a figure the
    sum leaves the cell (figure_of: by its digits and sign) stands, sure, its look "sum". In place; returns how many. Anything
    else leaves the line as it was. Gone over again until no more are kept: a cell kept may
    leave another sum with one cell open. The cell is the code's line at the column, as a
    dash is sought in (cell_parts), following the page's slant."""
    by_page = looked_at(pages)
    kept, tried, more = 0, set(), True
    while more:
        more = False
        every = statements_of({"pages": pages}, images=images)
        form = form_of(every)
        for s in every:
            rows = s["rows"]
            cols = {c["page"]: c for c in s["columns"]}
            for k, i, figures in open_cells(s, form):
                r, c = rows[k], cols.get(rows[k]["page"])
                p = by_page.get(r["page"])
                if c is None or p is None or i >= len(c["x"]):
                    continue
                xs = c["x"]
                gap = xs[i] - xs[i - 1] if i else (xs[1] - xs[0] if len(xs) > 1 else 400)
                sh = column_shifts(rows, r["page"], xs, c.get("tilt", 0.0))[i]
                dy = sh if abs(sh) > CELL[2] else 0.0
                x0, x1 = xs[i] - CELL[0] * gap, xs[i] + CELL[1]
                y0, y1 = r["code_box"][1] - CELL[2] + dy, r["code_box"][3] + CELL[2] + dy
                inside = [ln for ln in p["lines"] if ln.get("box")
                          and x0 <= (ln["box"][0] + ln["box"][2]) / 2 <= x1
                          and y0 <= (ln["box"][1] + ln["box"][3]) / 2 <= y1]
                if len(inside) != 1 or (r["page"], tuple(inside[0]["box"])) in tried:
                    continue
                ln = inside[0]
                tried.add((r["page"], tuple(ln["box"])))
                reads = readers.figure(page_png(images, p), ln["box"]) or ()
                text = next((f[1] for f in map(figure_of, reads) if f and f[0] in figures), None)
                if text is not None:
                    ln.update(text=text, sure=True, look="sum")
                    kept += 1
                    more = True
    return kept


def scan_png(path: str, n: int, pg: dict, images: str, stem: str, tmp: str) -> str:
    png = os.path.join(images, stem + ".png")
    if not os.path.exists(png):
        Image.fromarray(render(path, n, pg, signature_boxes(pg["blocks"]), tmp)).save(png)
    return png


def english(p: dict) -> bool:
    c = page_check(p.get("text") or "")
    return c["words"] >= ENGLISH_WORDS and c["marked"] < ENGLISH_MARKED


def blank(line: str) -> str:
    return NUMBER_RE.sub(BLANK, line)


def lines_of(p: dict) -> list[dict]:
    """A page's lines as the reader gets them: unsure ones with their numbers left empty."""
    texts = p["text"].split("\n") if p.get("text") else []
    bad, boxes = set(p.get("unsure", [])), p.get("boxes") or [None] * len(texts)
    return [{"text": blank(t) if i in bad else t, "sure": i not in bad, "box": b}
            for i, (t, b) in enumerate(zip(texts, boxes))]


def doc_fields(pages: list[dict]) -> dict:
    """The heading fields of one document, from its first page that shows any."""
    for p in pages:
        if p["kind"] == "skipped":
            continue
        got = extract(p["text"].split("\n"), p.get("unsure", []))
        if not any(g["seen"] for g in got.values()):
            continue
        out = {}
        for name, g in got.items():
            state = "empty" if g["value"] is None else "confirm"
            out[name] = {"value": g["value"], "state": state, "page": p["page"], "lines": g["lines"]}
        return out
    return {}


def draft(path: str, out: str, readers: Readers, filer: dict | None = None, show: bool = True,
          look: bool = False) -> dict:
    """filer: the listing's names for the filing's symbol, if given. show: render text-layer
    pages too, for the review page. look: unsure numbers get a second look (look_again)."""
    sha = sha256_of(path)
    images = os.path.join(out, "images")
    os.makedirs(images, exist_ok=True)
    pages = read_pages(path)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for r in label_file(path):
            n, pg = r["page"], pages[r["page"] - 1]
            row = {"page": n, "doc": r.get("doc"), "stamp": r.get("stamp"),
                   "key": page_key({"sha256": sha, "page": n})}
            if r.get("skip") or r.get("error"):
                rows.append({**row, "kind": "skipped", "why": r.get("skip") or r["error"]})
                continue
            if r["text_layer"]:
                text = kept_text(pg["blocks"], signature_boxes(pg["blocks"]))
                why = ("layer from OCR" if r.get("text_from_paper") else
                       "layer broken" if layer_broken(text) else None)
                if not why:
                    if show:                                            # shown, not read: for the review page
                        scan_png(path, n, pg, images, row["key"], tmp)
                    rows.append({**row, "kind": "text layer", "text": normal(text), "unsure": []})
                    continue
                row["why"] = why
            elif r.get("sig_box") and r.get("chars_raw", 0) >= SCAN_CHARS_PER_PAGE:
                row["why"] = "its text layer is only a signature box: read as a scan"
            t0 = time.time()
            png = scan_png(path, n, pg, images, row["key"], tmp)
            rows.append({**row, "kind": "read", **readers.page(png), "seconds": round(time.time() - t0, 1)})
    docs = {}
    for p in rows:
        docs.setdefault(p["doc"], []).append(p)
    fields = [{"doc": d, "pages": [p["page"] for p in ps], **doc_fields(ps)} for d, ps in docs.items()]
    kind = {p["page"]: p["kind"] for p in rows}
    for f in fields:
        if filer is not None and f.get("company"):
            g = f["company"]
            g["listing"] = matches(g["value"], filer)
            if g["listing"] and kind[g["page"]] == "read":
                g["state"] = "listing"          # set G: stands, not asked
    notes = list(NOTES)
    shown = [p for p in rows if p["kind"] != "skipped"]
    for p in shown:
        if english(p):
            p["english"] = "few tone marks: an English page, read but never measured"
    if shown and sum("english" in p for p in shown) * 2 > len(shown):
        notes.insert(1, "Most pages are in English, a kind never measured: use the Vietnamese copy of "
                        "this filing if there is one.")
    if filer is not None:
        notes.append("The company is compared with the listing's name for the filing's symbol. A match on a "
                     "page read by OCR stands without a person (tested on set G, provisionally: at most 1 of "
                     "60 wrong); on a text layer it is a label, not tested. A mismatch is no error: the author "
                     "may not be the filer (an exchange's notice, another body's letter).")
    for p in rows:
        if p.pop("no_colour", False):
            p["seal"] = "no colour on this page: a seal's text, if any, may remain in it"
        p["lines"] = lines_of(p)
        for k in ("text", "unsure", "boxes"):
            p.pop(k, None)
    if look:
        if coarse_pages(path, rows):
            notes.append(COARSE_NOTE)
        again(rows, images, readers)
        notes.append(LOOK_NOTE)
        code_look(rows, images, readers)
        notes.append(CODE_NOTE)
        dash_cells(rows, images)
        notes.append(CELL_NOTE)
        blank_cells(rows, images)
        notes.append(BLANK_NOTE)
        if sum_look(rows, images, readers):
            blank_cells(rows, images)
        notes.append(SUM_NOTE)
    return {"file": os.path.basename(path), "sha256": sha, "profile": "disclosures",
            "drafted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "filer": filer and filer.get("organName"), "notes": notes, "documents": fields, "pages": rows}


def page_png(images: str, p: dict) -> str:
    """The image a read page's boxes are on: for a page turned upright (tools/fs_turn.py),
    the turned image, made again if it is gone."""
    png = os.path.join(images, p["key"] + ".png")
    if "turned" not in p:
        return png
    turned = os.path.join(images, f"{p['key']}-r{p['turned']}.png")
    if not os.path.exists(turned):
        Image.open(png).rotate(p["turned"], expand=True).save(turned)
    return turned


def again(pages: list[dict], images: str, readers: Readers, stamps_only: bool = False) -> None:
    """The second look over a draft's read pages, in place."""
    for p in pages:
        if p["kind"] == "read" and not p.get("coarse"):      # as looked_at
            p["lines"] = readers.look_again(page_png(images, p), p["lines"], stamps_only)


def check(set_path: str, run_path: str, readers: Readers, cache: str, images: str) -> int:
    """Kept scan pages of a set, read by the tool, against the scored run."""
    run = {(r["sha256"], r["page"]): r for r in read_jsonl(run_path)}
    os.makedirs(images, exist_ok=True)
    c, same, differ, seconds, layouts = Client(), 0, [], [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for e in read_jsonl(set_path):
            k = (e["sha256"], e["page"])
            if e.get("fate") != "kept" or k not in run:
                continue
            path, why = c.download(e["pdf"], cache)
            if not path or sha256_of(path) != e["sha256"]:
                sys.exit(f"page {page_key(e)}: {why or 'SHA-256 differs from the set'}")
            t0 = time.time()
            if path not in layouts:
                layouts[path] = read_pages(path)
            png = scan_png(path, e["page"], layouts[path][e["page"] - 1], images, page_key(e), tmp)
            got = readers.page(png)
            seconds.append(time.time() - t0)
            want = run[k]
            if (got["text"] == want["text"] and got["unsure"] == want["unsure"]
                    and len(got["boxes"]) == len(got["text"].split("\n"))):
                same += 1
            else:
                differ.append(page_key(e))
    print(f"same: {same}, differ: {len(differ)}, seconds per page: {sum(seconds) / max(len(seconds), 1):.1f}")
    for k in differ:
        print("differs:", k)
    return 1 if differ else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pdfs", nargs="*")
    ap.add_argument("--out", default="cache/drafts")
    ap.add_argument("--rec-model", default=ROUND1)
    ap.add_argument("--device", default="gpu:0")
    ap.add_argument("--check", nargs=2, metavar=("SET", "RUN"))
    ap.add_argument("--cache", default="cache/pdfs")
    ap.add_argument("--symbol", help="the filing's symbol: its company is compared with the listing's name")
    ap.add_argument("--look", action="store_true", help="give unsure numbers a second look (statements)")
    ap.add_argument("--again", nargs="+", metavar="DRAFT",
                    help="give the unsure numbers of these drafts a second look, in place")
    a = ap.parse_args()
    readers = Readers(a.rec_model, a.device)
    if a.check:
        sys.exit(check(*a.check, readers, a.cache, os.path.join(a.out, "check-images")))
    if a.again:
        for path in a.again:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            images = os.path.join(os.path.dirname(path), "images")
            pdf = os.path.join(a.cache, d["file"])          # first: the looks pass coarse pages over
            if COARSE_NOTE not in d["notes"] and os.path.exists(pdf) and coarse_pages(pdf, d["pages"]):
                d["notes"].append(COARSE_NOTE)
            if LOOK_NOTE not in d["notes"]:
                old = [n for n in d["notes"] if n in OLD_LOOK_NOTES]   # a look by an older rule
                again(d["pages"], images, readers, stamps_only=bool(old))
                if old:
                    d["notes"][d["notes"].index(old[0])] = LOOK_NOTE
                    d["notes"] = [n for n in d["notes"] if n not in OLD_LOOK_NOTES]
                else:
                    d["notes"].append(LOOK_NOTE)
            new_codes = 0
            if CODE_NOTE not in d["notes"]:
                d["notes"] = [n for n in d["notes"] if n not in OLD_CODE_NOTES]   # a code look by an older rule
                new_codes = code_look(d["pages"], images, readers)
                d["notes"].insert(d["notes"].index(CELL_NOTE) if CELL_NOTE in d["notes"] else len(d["notes"]),
                                  CODE_NOTE)
            older = [n for n in d["notes"] if n in OLD_CELL_NOTES]
            if CELL_NOTE not in d["notes"] or new_codes:   # after the codes: it fills what they left empty
                if older:                                  # an older rule's dashes go; this one looks afresh
                    for p in d["pages"]:
                        if p.get("lines"):
                            p["lines"] = [ln for ln in p["lines"] if ln.get("look") != "dash cell"]
                dash_cells(d["pages"], images)
                if older:
                    d["notes"][d["notes"].index(older[0])] = CELL_NOTE
                    d["notes"] = [n for n in d["notes"] if n not in OLD_CELL_NOTES]
                elif CELL_NOTE not in d["notes"]:
                    d["notes"].append(CELL_NOTE)
            blank_cells(d["pages"], images)                # afresh: it follows whatever the looks left empty
            if BLANK_NOTE not in d["notes"]:
                d["notes"].append(BLANK_NOTE)
            if sum_look(d["pages"], images, readers):      # what it keeps may move a row's figures
                blank_cells(d["pages"], images)
            if SUM_NOTE not in d["notes"]:
                d["notes"].append(SUM_NOTE)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(d, fh, ensure_ascii=False, indent=1)
            print(os.path.basename(path), sum(ln.get("look") is not None for p in d["pages"] for ln in p.get("lines", [])))
        return
    filer = None
    if a.symbol:
        filer = listing().get(a.symbol.upper())
        if not filer:
            sys.exit(f"{a.symbol}: not in the listing")
    os.makedirs(a.out, exist_ok=True)
    for path in a.pdfs:
        d = draft(path, a.out, readers, filer, look=a.look)
        name = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(a.out, name + ".draft.json"), "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=1)
        kinds = {}
        for p in d["pages"]:
            kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
        print(name, kinds)


if __name__ == "__main__":
    main()
