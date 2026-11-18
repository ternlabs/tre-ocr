"""Label every page of a set of PDFs: text layer, stamp, document, skip.

Step 1 of METHOD.md. A file can join a scanned part and a typed part, so labels
are per page. Each page goes through, in order:

  1. Signature box. A digitally signed PDF carries a box of text (signer, time,
     "DN:" line, tax code). It is found by its markers and removed, with the
     text blocks it spans, before anything is counted.
  2. Text layer. Fewer than 200 characters left on the page means a scan.
  3. Stamp. A red ring of stamp size on the rendered page. "unknown" when the
     page's images were stored in grey: a grey scan can hide a stamp, and
     abstaining beats guessing. A colour scan with no red ring is "no". A stamp
     copied in grey is not searched for (see GREY_STAMPS).
  4. Documents. One file can join several paper documents. A document starts
     at the national header ("Cộng hòa xã hội chủ nghĩa Việt Nam" / "Socialist
     Republic of Vietnam", with the motto and a short rule under it) at the top
     of a page, and ends at a page with the company stamp: a whole ring, not
     cut by the page's edge (that is an edge seal) and not over the company
     name in the top-left corner. That stamp, on a letter or form with no
     national header, is what opens a document: it starts one. A closing
     page can be short, and a sideways table puts its stamp near the top, so
     the ring's height on the page is not used. A digitally signed document
     may have no stamp: its signature is a flat grey box pasted above the
     signer's name, which ends it too (see grey_sign).
     A cover or a contents page fronts the documents after it: a page that
     would start a document, nearly empty or with only a few lines of text
     (a title in large type, a picture), with no stamp and no national
     header, is a one-page document of its own (cover), and the next page
     starts another. The last pages of a file, after an end page, with no
     header and no stamp of any kind, are that document's appendix tables:
     they join it rather than open a document of their own. So does a page
     after an end page that opens with an appendix title ("Phụ lục số 01",
     "Appendix 1"), anywhere in the file: the body is signed, then its
     appendices follow. The title is read from the text layer, so on a scan
     this is not checked.
     When that end page is a full-page image, the document was paper, and so
     is every page of it: the pages between were stamped across their edges
     ("dấu giáp lai"), which is not searched for page by page. A text layer on
     a paper page is flagged (text_from_paper): it most likely came from OCR,
     not from the author, so it is not free ground truth.
  5. Skip. Blank pages, pages that hold only a signature box, and member lists
     (lists of people) are skipped: kept in the output with their reason and
     labels only (no text), never scored. Member lists are
     found from text, so on a scan page that check is not run (null).

Needs poppler-utils (pdftotext, pdftoppm, pdfimages) and numpy.

Writes <out>, one JSON line per page. Keys are the cache file name and the
page number; no text from the page is written.

Usage:
  python -I tools/label_pages.py cache/pdfs/*.pdf --out data/pages.jsonl
"""
import argparse
import functools
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from sample_pairs import SCAN_CHARS_PER_PAGE  # noqa: E402  (the sampler's rule; here after the box goes)

DPI = 40                        # enough to see ink, colour and a 40 mm stamp
HEAD_DPI = 100                  # the rule under the motto is a fraction of a mm
MM = DPI / 25.4                 # pixels per millimetre

# A block holding one of these is part of a signature box. Each is a label
# ("Ký bởi:") or a certificate field ("CN="), not a phrase: body text says
# "signed by the Chairman" too.
BOX_STRONG = re.compile(
    r"digitally signed|signed by\s*:|signature (not )?verified|signature valid|"
    r"ký bởi\s*:|người ký\s*:|thời gian ký\s*:|\bDN:\s*\w+=|\bCN=|\bUID=", re.I)
BOX_MAX_CHARS = 700             # a block longer than this is body text, whatever it says
# Lines that sit in a box but are too common to start one alone.
BOX_WEAK = re.compile(
    r"^(reason|location|date|ký ngày|ngày ký|email|mst|foxit|adobe)\b|^[+-]\d\d'?:?\d\d'?$|"
    r"^\d{4}[./-]\d\d[./-]\d\d|^\d\d?[./-]\d\d?[./-]\d{4}", re.I)
BOX_GAP = 12.0                  # points; marker blocks this close to the box join it
BOX_TOUCH = 2.0                 # points; a plain block must touch the box to join

# Member lists: a "list of" heading naming people, or many people on one page.
LIST_HEAD = re.compile(
    r"danh sách\s+(cổ đông|thành viên|người (có liên quan|nội bộ)|đại biểu|"
    r"người (sở hữu|tham dự))|list of\s+(shareholders|members|related|insiders|"
    r"attendees|delegates|persons)", re.I)
PERSON = re.compile(r"\b(?i:ông|bà|mr|mrs|ms)\.?\s+(\w)")   # then a capital letter
PERSONS_FOR_LIST = 8

INK_BLANK = 0.003               # share of ink pixels below which a page is empty

# The national header, compared with accents, case and spaces removed. An OCR
# text layer misreads some letters, so a few edits are allowed.
HEADER_TEXT = ("conghoaxahoichunghiavietnam", "socialistrepublicofvietnam")
HEADER_TOP = 0.3                # share of the page height the header sits in
SEAL_EDGE_MM = 2                # a ring this close to the page's edge is an edge seal
NAME_STAMP = (0.2, 0.4)         # a centre above and left of these shares is over the name
COVER_INK = 0.04                # a cover or contents page has less ink than this share
COVER_LINES = 18                # or this many lines of text at most (a title in large type, a footer),
COVER_ROWS = 0.6                # over less than this share of the page's rows: a picture adds ink, not lines
GREY_SIGN_PX = 300               # solid grey pixels a signature image holds at least
# Off: on pages outside the truth sets the grey ring test found 1 real stamp in 21;
# sideways tables, grey signature boxes and drawn circles pass it too (see grey_stamps).
GREY_STAMPS = False
GREY_DPI = 100                  # a stamp scanned in grey is told from text by its tone, which 40 dpi blurs
GMM = GREY_DPI / 25.4
STAMP_GREY = (110, 215)         # its ring at GREY_DPI: lighter than black text, darker than paper
RULED = 1.4                     # a straight line longer than this many radii through a ring: a table
PAGE_IMAGE = 0.85               # an image this share of the page's width and height


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, check=False, **kw)


# ---------- text, with the signature box removed ----------

def _box(el):
    return tuple(float(el.get(k)) for k in ("xMin", "yMin", "xMax", "yMax"))


def read_pages(path):
    """Per page: size and its text blocks as (bbox, [lines])."""
    out = run(["pdftotext", "-q", "-enc", "UTF-8", "-bbox-layout", path, "-"]).stdout
    text = out.decode("utf-8", "replace")
    # strip the XHTML namespace and doctype so ElementTree reads it plainly
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text)
    text = re.sub(r'\sxmlns="[^"]*"', "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", text)
    root = ET.fromstring(text)
    pages = []
    for pg in root.iter("page"):
        blocks = []
        for b in pg.iter("block"):
            lines = [" ".join(html.unescape(w.text or "") for w in ln.iter("word"))
                     for ln in b.iter("line")]
            blocks.append((_box(b), lines))
        pages.append({"w": float(pg.get("width")), "h": float(pg.get("height")),
                      "blocks": blocks,
                      "words": [(*_box(w), html.unescape(w.text or "")) for w in pg.iter("word")]})
    # pdftotext places the boxes as the page is shown but gives the size
    # unturned; a page turned a quarter is shown with its sides swapped
    for pg, rot in zip(pages, rotations(path, len(pages))):
        if rot in (90, 270):
            pg["w"], pg["h"] = pg["h"], pg["w"]
    return pages


def rotations(path, n):
    """Each page's /Rotate, in degrees clockwise."""
    out = run(["pdfinfo", "-f", "1", "-l", str(max(n, 1)), path]).stdout.decode("utf-8", "replace")
    rot = dict((int(a), int(b)) for a, b in re.findall(r"Page\s+(\d+) rot:\s+(\d+)", out))
    return [rot.get(i, 0) % 360 for i in range(1, n + 1)]


def _near(a, b, gap):
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or
                a[3] + gap < b[1] or b[3] + gap < a[1])


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def signature_boxes(blocks):
    """Regions covered by signature boxes, as a list of bboxes.

    Seeds are short blocks with a strong marker. A seed grows by nearby blocks
    that carry markers only, and by a block that touches it, lies inside its
    height and is not much wider: the signer's name, printed large beside the
    box. A heading next to the box is wider, so it stays."""
    def marked(lines):
        t = " ".join(lines)
        return len(t) <= BOX_MAX_CHARS and BOX_STRONG.search(t)
    seeds = [bb for bb, lines in blocks if marked(lines)]
    boxes = []
    for s in seeds:
        if any(_near(s, b, 0) for b in boxes):
            continue
        box, grown = s, True
        while grown:
            grown = False
            for bb, lines in blocks:
                if bb == box or (bb[0] >= box[0] and bb[2] <= box[2] and
                                 bb[1] >= box[1] and bb[3] <= box[3]):
                    continue
                if not _near(box, bb, BOX_GAP):
                    continue
                weak = all(BOX_WEAK.search(l.strip()) for l in lines if l.strip())
                beside = (_near(box, bb, BOX_TOUCH) and bb[1] >= box[1] - 4 and
                          bb[3] <= box[3] + 4 and
                          bb[2] - bb[0] <= 1.5 * (s[2] - s[0]))
                if weak or marked(lines) or beside:
                    box, grown = _union(box, bb), True
        for b in boxes:                 # a box this one touches joins it
            if _near(b, box, 0):
                box = _union(box, b)
        boxes = [b for b in boxes if not _near(b, box, 0)] + [box]
    return boxes


def _inside(bb, boxes):
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)


def kept_text(blocks, boxes):
    return "\n".join(l for bb, lines in blocks if not _inside(bb, boxes) for l in lines)


def nchars(s):
    return len(re.sub(r"\s", "", s))


# ---------- the rendered page ----------

def render(path, tmp):
    """One RGB array per page, at DPI."""
    prefix = os.path.join(tmp, "p")
    run(["pdftoppm", "-r", str(DPI), path, prefix])
    imgs = {}
    for name in os.listdir(tmp):
        m = re.fullmatch(r"p-0*(\d+)\.ppm", name)
        if m:
            imgs[int(m.group(1))] = read_ppm(os.path.join(tmp, name))
            os.remove(os.path.join(tmp, name))
    return imgs


def read_ppm(path):
    with open(path, "rb") as fh:
        data = fh.read()
    # header: P6 <w> <h> <max>, one whitespace, then the pixels
    m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(\d+)\s", data)
    w, h = int(m.group(1)), int(m.group(2))
    return np.frombuffer(data, np.uint8, w * h * 3, m.end()).reshape(h, w, 3)


def image_list(path, kinds=("image",)):
    """Every image pdfimages lists: page, size in pixels, colour, resolution. kinds: the
    types kept ("mask" or "stencil" too, for a layered scan's text). A failed run lists what it
    printed, often nothing."""
    out = run(["pdfimages", "-list", path]).stdout.decode("utf-8", "replace")
    imgs = []
    for line in out.splitlines()[2:]:
        f = line.split()
        # page num type width height color comp bpc enc interp object ID x-ppi y-ppi ...
        # (an inline image has one "[inline]" in place of object ID)
        if len(f) > 13 and f[0].isdigit() and f[2] in kinds and f[3].isdigit() and \
                f[4].isdigit() and f[6].isdigit():
            xp = int(f[12]) if f[12].isdigit() else 0
            yp = int(f[13]) if f[13].isdigit() else xp
            imgs.append({"page": int(f[0]), "w": int(f[3]), "h": int(f[4]),
                         "colour": f[5] == "index" or int(f[6]) >= 3, "xppi": xp, "yppi": yp,
                         "inline": f[10] == "[inline]"})
    return imgs


def image_modes(imgs):
    """Page -> "colour" or "grey", by how its largest image was stored. A small
    colour logo on a grey scan does not make the scan colour; a palette
    ("index") image is colour."""
    big = {}
    for im in imgs:
        if im["w"] * im["h"] > big.get(im["page"], (0, None))[0]:
            big[im["page"]] = (im["w"] * im["h"], im["colour"])
    return {n: "colour" if c else "grey" for n, (_, c) in big.items()}


def blank_out(img, boxes, pw, ph):
    """A copy of the page with the signature boxes painted white."""
    img = img.copy()
    sx, sy = img.shape[1] / pw, img.shape[0] / ph
    for x0, y0, x1, y1 in boxes:
        img[max(int(y0 * sy) - 1, 0):int(y1 * sy) + 2,
            max(int(x0 * sx) - 1, 0):int(x1 * sx) + 2] = 255
    return img


def ink_share(img):
    """Share of pixels clearly darker than the page's paper. Relative, so a
    pale scan still shows its text and tinted paper alone shows none."""
    lum = img.astype(np.int16).mean(axis=2)
    return float((lum < np.median(lum) - 40).mean())


def text_lines(img):
    """Lines of text on the page and the share of its rows they cover, margins left out.
    A row holds text when more than 3% of it is ink; a line is a run of such rows."""
    lum = img.astype(np.int16).mean(axis=2)
    ink = lum < np.median(lum) - 40
    h, w = ink.shape
    rows = ink[int(.06 * h):int(.94 * h), int(.08 * w):int(.92 * w)].mean(axis=1) > 0.03
    lines = int(rows[0]) + int((np.diff(rows.astype(np.int8)) == 1).sum())
    return lines, float(rows.mean())


def _solid_grey(lum, chroma, k=3):
    """Pixels whose whole (2k+1)-square is flat grey: lighter than text, darker than paper."""
    g = (lum > 100) & (lum < np.median(lum) - 25) & (chroma < 25)
    e = g.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            e &= np.roll(np.roll(g, dy, 0), dx, 1)
    e[:k] = e[-k:] = False
    e[:, :k] = e[:, -k:] = False
    return e


def grey_sign(img):
    """Is there a signature image on the page: a flat grey box with the handwriting in it,
    pasted where a hand signature goes? The box is wider than tall and less than half
    the page wide; it has no ruled frame (a shaded table cell has one), sits on white
    paper (a grey patch in a photo does not), and under it is at most one line
    (the signer's name), then empty space for several box heights (a table goes on)."""
    a = img.astype(np.int16)
    lum, chroma = a.mean(axis=2), a.max(axis=2) - a.min(axis=2)
    e = _solid_grey(lum, chroma)
    if e.sum() < GREY_SIGN_PX:
        return False
    h, w = e.shape
    paper = lum > np.median(lum) - 25
    ink = (lum < np.median(lum) - 40) & ~e
    dark = lum < 110
    blocks = []                         # the handwriting cuts a box into parts: join them
    for c in sorted(c for c in _components(e) if c[4] >= 20):
        for b in blocks:
            if c[1] <= b[3] and c[3] >= b[1] and c[0] <= b[2] + .03 * w:
                b[:] = [min(b[0], c[0]), min(b[1], c[1]), max(b[2], c[2]), max(b[3], c[3]),
                        b[4] + c[4], b[5] + 1]
                break
        else:
            blocks.append(list(c) + [1])
    for x0, y0, x1, y1, n, parts in blocks:
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if n < GREY_SIGN_PX or parts > 4 or bw > .45 * w or bw < 2.5 * bh:
            continue
        if x0 < .02 * w or y0 < .04 * h or y1 > .96 * h:
            continue                    # a banner or a band at the page's edge
        X0, Y0, X1, Y1 = x0 - 4, y0 - 4, x1 + 4, y1 + 4   # the true box, before the erosion
        edges = [dark[max(Y0 - 3, 0):max(Y0, 0), X0:X1 + 1].mean(axis=1),
                 dark[Y1 + 1:Y1 + 4, X0:X1 + 1].mean(axis=1),
                 dark[Y0:Y1 + 1, max(X0 - 3, 0):max(X0, 0)].mean(axis=0),
                 dark[Y0:Y1 + 1, X1 + 1:X1 + 4].mean(axis=0)]
        if max(v.max() if v.size else 0 for v in edges) >= 0.1:
            continue                    # framed: a table cell
        ring = np.zeros_like(paper)
        ring[max(Y0 - 8, 0):Y1 + 9, max(X0 - 8, 0):X1 + 9] = True
        ring[max(Y0 - 2, 0):Y1 + 3, max(X0 - 2, 0):X1 + 3] = False
        if paper[ring].mean() < 0.75:
            continue                    # inside a picture
        pad = int(.03 * w)
        below = ink[Y1 + 1:min(h, Y1 + 1 + 6 * bh), max(x0 - pad, 0):x1 + pad + 1].mean(axis=1) > 0.03
        lines = int(below[:1].sum()) + int((np.diff(below.astype(np.int8)) == 1).sum())
        run = gap = 0
        for v in below:
            run = 0 if v else run + 1
            gap = max(gap, run)
        gap += 6 * bh - below.size      # the page's end counts as empty
        if lines <= 1 and gap >= 3 * bh:
            return True
    return False


def red_mask(img):
    """Stamp ink: red to pink-magenta, even when faded. Gold and orange are not."""
    a = img.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    hi, lo = a.max(axis=2), a.min(axis=2)
    chroma = hi - lo
    # hue below 15 degrees, or above 320 (magenta side), with red the top channel
    reddish = (r == hi) & ((g - b) * 4 <= chroma) & ((b - g) * 2 <= chroma * 3)
    return reddish & (chroma >= 30) & (r >= 120)


def _components(mask):
    """Bounding boxes and pixel counts of 8-connected parts of mask, in the
    order of their first pixel, row by row.

    Works on runs of set pixels along each row, not on pixels: a run joins
    every run of the row above that it touches, corners included."""
    d = np.diff(np.pad(mask, ((0, 0), (1, 1))).view(np.int8), axis=1)
    ys, xs = np.nonzero(d == 1)
    ends = np.nonzero(d == -1)[1] - 1           # same row order as the starts
    ys, xs, ends = ys.tolist(), xs.tolist(), ends.tolist()
    parent = list(range(len(ys)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    above, row, cur = [], [], -1                # runs of the row above, and of this one
    for i, (y, a, b) in enumerate(zip(ys, xs, ends)):
        if y != cur:
            above, row, cur = (row if y == cur + 1 else []), [], y
        for j in above:
            if xs[j] <= b + 1 and ends[j] >= a - 1:
                ri, rj = root(i), root(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)   # the earlier run stays the root
        row.append(i)
    box = {}
    for i, (y, a, b) in enumerate(zip(ys, xs, ends)):
        r = root(i)
        if r in box:
            x0, y0, x1, y1, n = box[r]
            box[r] = (min(x0, a), y0, max(x1, b), y, n + b - a + 1)
        else:
            box[r] = (a, y, b, y, b - a + 1)
    return [box[r] for r in sorted(box)]


STAMP_MM = (24, 28, 32, 36, 40, 44, 48)  # diameters tried
_ANGLES = np.linspace(0, 2 * np.pi, 36, endpoint=False)


@functools.lru_cache(maxsize=None)
def _ring_offsets(r, ks):
    """Pixel offsets, per angle, of points at k * r from a centre."""
    dy = np.round(np.outer(np.sin(_ANGLES), ks) * r).astype(int)
    dx = np.round(np.outer(np.cos(_ANGLES), ks) * r).astype(int)
    return dy, dx


def _circle_at(m, d, light, cy, cx, r):
    """Is there a stamp-like ring of radius r centred here? The ring must be
    inked in most directions, the band just outside clean and mostly paper
    (a photo is not), and the disc not filled solid."""
    dy, dx = _ring_offsets(r, (0.82, 0.9, 0.98))
    by, bx = _ring_offsets(r, (1.15, 1.3))
    if d[cy + dy, cx + dx].any(axis=1).mean() < 0.75:
        return False
    if d[cy + by, cx + bx].any(axis=1).mean() > 0.2:
        return False
    if light[cy + by, cx + bx].mean() < 0.7:
        return False
    ri = int(r)
    return m[cy - ri:cy + ri + 1, cx - ri:cx + ri + 1].mean() <= 0.35


def _centres(lo, hi, r, step):
    first, last = int(lo + 0.3 * r), int(hi - 0.3 * r)
    return range(first, last + 1, step) if first <= last else [(lo + hi) // 2]


def _dilated(m, n=2):
    d = m.copy()
    for _ in range(n):
        e = d.copy()
        e[1:] |= d[:-1]; e[:-1] |= d[1:]; e[:, 1:] |= d[:, :-1]; e[:, :-1] |= d[:, 1:]
        d = e
    return d


def find_stamps(img):
    """Red rings of stamp size, as (y, x, ry, rx): the centre and the radius,
    each as a share of the page's height or width.

    A Vietnamese company stamp is a red ring with text, about 30-45 mm across,
    often crossed by a signature that can join it to other ink. So circles of
    stamp size are slid across each red part of the page, and one that is
    inked on its ring and clean just outside it is a stamp. The mask is dilated
    by about 1.5 mm so a broken ring joins up. A photo or a logo of the same
    colour fails the ring test."""
    return _rings(red_mask(img), img.astype(np.int16).mean(axis=2) > 180, MM)


def _rings(m, light, px_mm):
    """Rings of stamp size in mask m (see find_stamps); light is the paper,
    px_mm the pixels per millimetre."""
    if m.sum() < 20:
        return []
    H, W = m.shape
    d = _dilated(m)
    # pad so a circle near the edge can be tested: off the page is clean paper
    pad = int(STAMP_MM[-1] * px_mm)
    m, d = np.pad(m, pad), np.pad(d, pad)
    light = np.pad(light, pad, constant_values=True)
    step = max(int(2 * px_mm), 1)
    found = []
    for x0, y0, x1, y1, _ in _components(d):
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if min(w, h) < 20 * px_mm or max(w, h) > 120 * px_mm:
            continue
        for mm in STAMP_MM:
            r = mm * px_mm / 2
            if 2 * r > max(w, h) + 6 * px_mm:
                continue
            # part of a faded ring may not show, so a circle can stick out
            at = next(((cy, cx) for cy in _centres(y0, y1, r, step)
                       for cx in _centres(x0, x1, r, step)
                       if _circle_at(m, d, light, cy, cx, r)), None)
            if at:
                found.append(((at[0] - pad) / H, (at[1] - pad) / W, r / H, r / W))
                break                   # one stamp per inked part
    return found


def render_grey(path, n, tmp):
    """Page n in grey at GREY_DPI."""
    prefix = os.path.join(tmp, "g")
    run(["pdftoppm", "-gray", "-r", str(GREY_DPI), "-f", str(n), "-l", str(n), path, prefix])
    for name in os.listdir(tmp):
        if name.startswith("g-") and name.endswith(".pgm"):
            with open(os.path.join(tmp, name), "rb") as fh:
                data = fh.read()
            os.remove(os.path.join(tmp, name))
            h = re.match(rb"P5\s+(\d+)\s+(\d+)\s+(\d+)\s", data)
            w, hh = int(h.group(1)), int(h.group(2))
            return np.frombuffer(data, np.uint8, w * hh, h.end()).reshape(hh, w)
    return None


def grey_stamps(g):
    """Stamps scanned in grey, as find_stamps gives them; g is the page in grey
    at GREY_DPI.

    A red stamp copied or scanned in grey turns mid-grey, lighter than black
    text: so its ink is the mid-grey pixels not touching black (the soft edge of
    a black letter is mid-grey too), and the ring test runs on them. Grey table
    rules and cells can close a ring too, so a ring crossed by a long straight
    line is dropped: a stamp has none."""
    g = g.astype(np.int16)
    lo, hi = STAMP_GREY
    hi = min(hi, np.median(g) - 25)
    m = (g >= lo) & (g < hi) & ~_dilated(g < lo, 1)
    return [s for s in _rings(m, g > 180, GMM) if not _ruled(g, s)]


def _ruled(g, ring):
    """Does a straight line longer than RULED radii run through the ring's square?"""
    h, w = g.shape
    cy, cx, r = int(ring[0] * h), int(ring[1] * w), int(ring[2] * h)
    sq = g[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1] < 200
    sq = sq | np.roll(sq, 1, 0) | np.roll(sq, 1, 1)     # a line a pixel off level
    return any(_longest_run(np.ascontiguousarray(v)) > RULED * r for v in (*sq, *sq.T))


def end_stamp(ring, shape, px_mm=MM):
    """Is this ring (from find_stamps) the stamp that closes a document?"""
    y, x, ry, rx = ring
    H, W = shape
    my, mx = SEAL_EDGE_MM * px_mm / H, SEAL_EDGE_MM * px_mm / W
    if y - ry < my or y + ry > 1 - my or x - rx < mx or x + rx > 1 - mx:
        return False                    # cut by, or touching, the edge
    return not name_stamp(ring)


def name_stamp(ring):
    """Is this ring over the company name, top left, where a letter is opened?"""
    y, x = ring[:2]
    return y < NAME_STAMP[0] and x < NAME_STAMP[1]


def render_tops(path, tmp, ph):
    """Grey top part of every page, at HEAD_DPI; ph is the tallest page, in points."""
    prefix = os.path.join(tmp, "h")
    rows = int(HEADER_TOP * ph / 72 * HEAD_DPI) + 1
    run(["pdftoppm", "-gray", "-r", str(HEAD_DPI), "-x", "0", "-y", "0",
         "-W", "100000", "-H", str(rows), path, prefix])
    out = {}
    for name in os.listdir(tmp):
        m = re.fullmatch(r"h-0*(\d+)\.pgm", name)
        if m:
            with open(os.path.join(tmp, name), "rb") as fh:
                data = fh.read()
            os.remove(os.path.join(tmp, name))
            h = re.match(rb"P5\s+(\d+)\s+(\d+)\s+(\d+)\s", data)
            w, hh = int(h.group(1)), int(h.group(2))
            out[int(m.group(1))] = np.frombuffer(data, np.uint8, w * hh, h.end()).reshape(hh, w)
    return out


def page_images(imgs, pages):
    """Pages covered by one image: at least PAGE_IMAGE of their width and of
    their height (either way round, as a scan can be stored turned)."""
    full = set()
    for im in imgs:
        n = im["page"]
        if not (1 <= n <= len(pages) and im["xppi"] and im["yppi"]):
            continue
        pg = pages[n - 1]
        iw, ih = im["w"] / im["xppi"] * 72, im["h"] / im["yppi"] * 72
        if any(iw >= PAGE_IMAGE * a and ih >= PAGE_IMAGE * b
               for a, b in ((pg["w"], pg["h"]), (pg["h"], pg["w"]))):
            full.add(n)
    return full


# ---------- the national header ----------

def _flat(t):
    t = unicodedata.normalize("NFD", t.lower().replace("đ", "d"))
    return re.sub(r"[^a-z0-9]+", "", "".join(c for c in t if not unicodedata.combining(c)))


def _edits_in(k, t):
    """Fewest edits that turn k into some part of t."""
    prev = [0] * (len(t) + 1)
    for i, ch in enumerate(k, 1):
        cur = [i] + [0] * len(t)
        for j, tc in enumerate(t, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ch != tc))
        prev = cur
    return min(prev)


def header_in_text(pg):
    top = _flat(" ".join(" ".join(ls) for bb, ls in pg["blocks"]
                         if bb[1] < HEADER_TOP * pg["h"]))
    return any(_edits_in(k, top) <= len(k) // 5 for k in HEADER_TEXT)


APPENDIX = ("phuluc", "appendix", "annex")   # an appendix's title starts so (flattened)
APPENDIX_TOP = 0.25             # share of the page height the title sits in


def appendix_title(pg):
    """Does the page open with an appendix title ("Phụ lục số 01", "Appendix 1")?
    A short line in the top part, so a sentence that names an appendix does not count."""
    return any(len(ln) < 80 and _flat(ln).startswith(APPENDIX)
               for bb, ls in pg["blocks"] if bb[1] < APPENDIX_TOP * pg["h"] for ln in ls)


HMM = HEAD_DPI / 25.4


def _text_lines(win):
    """Text lines in a window, nearest the bottom first: (top, bottom, inked columns)."""
    rows = (win.sum(axis=1) > 2).astype(np.int8)
    d = np.diff(np.concatenate(([0], rows, [0])))
    out = []
    for t, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
        if b - t >= 1.2 * HMM:
            out.append((t, b, np.nonzero(win[t:b].any(axis=0))[0]))
    return out[::-1]


def _run_around(cols, c, gap):
    """Ends of the run of inked columns nearest c, gaps up to gap bridged."""
    if not len(cols):
        return None
    k = min(np.searchsorted(cols, c), len(cols) - 1)
    if k and abs(cols[k - 1] - c) < abs(cols[k] - c):
        k -= 1
    if abs(cols[k] - c) > gap:
        return None
    lo = hi = k
    while lo > 0 and cols[lo] - cols[lo - 1] <= gap:
        lo -= 1
    while hi < len(cols) - 1 and cols[hi + 1] - cols[hi] <= gap:
        hi += 1
    return cols[lo], cols[hi]


def _header_above(ink, y, c, L):
    """The motto just above the rule, the title above it, both centred on it."""
    x0 = max(c - int(70 * HMM), 0)
    win = ink[max(y - int(18 * HMM), 0):y - 2, x0:c + int(70 * HMM)]
    ls = _text_lines(win)
    if len(ls) < 2:
        return False
    (t1, b1, c1), (t2, b2, c2) = ls[0], ls[1]
    if win.shape[0] - b1 > 7.5 * HMM or t1 - b2 > 6 * HMM:
        return False
    motto = _run_around(c1, c - x0, int(5 * HMM))
    title = _run_around(c2, c - x0, int(5 * HMM))
    if not motto or not title:
        return False
    w1, w2 = motto[1] - motto[0], title[1] - title[0]
    # the rule is a third of the motto's width up to a little wider; the
    # Vietnamese title is wider than the motto, the English one about as wide
    if not (0.7 * L <= w1 <= 3.5 * L and 0.65 * w1 <= w2 <= 2.4 * w1):
        return False
    if w1 < 35 * HMM or w2 < 45 * HMM:        # table column headings are narrower
        return False
    # each line ends inside the window: a running page header spans the page
    edge = 3 * HMM
    if any(lo < edge or hi > win.shape[1] - edge for lo, hi in (motto, title)):
        return False
    return all(abs((a + b) / 2 - (c - x0)) < 5 * HMM for a, b in (motto, title))


def _closed(v, gap):
    """v with its gaps of up to gap pixels filled: a dashed line made whole."""
    d = np.diff(np.concatenate(([0], v.view(np.int8), [0])))
    st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    v = v.copy()
    for e, s in zip(en[:-1], st[1:]):
        if s - e <= gap:
            v[e:s] = True
    return v


def _longest_run(v):
    d = np.diff(np.concatenate(([0], v.view(np.int8), [0])))
    st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return int((en - st).max()) if len(st) else 0


def header_in_pixels(g, page_rows):
    """The short rule under the motto, right of the left column, in the top
    of the page, with the motto and title centred above it. g is the page's
    top at HEAD_DPI; page_rows its full height in pixels."""
    ink = g < min(160, int(np.median(g)) - 60)
    # the rule is often dashed, dotted or printed light: found on a lighter
    # threshold, its gaps of up to 1 mm bridged
    faint = g < int(np.median(g)) - 35
    gap = int(1 * HMM)
    W = ink.shape[1]
    x0 = int(0.35 * W)
    lo, hi = int(15 * HMM), int(85 * HMM)
    for y in range(int(5 * HMM), min(int(HEADER_TOP * page_rows), ink.shape[0] - int(3 * HMM))):
        row = faint[y, x0:]
        if row.sum() < 0.45 * lo:
            continue
        r = _closed(row, gap)
        # a tilted scan spreads a line over a few rows: measure it on five
        spread = faint[y - 2:y + 3, x0:].any(axis=0)
        band = _closed(spread, gap)
        d = np.diff(np.concatenate(([0], r.view(np.int8), [0])))
        bd = np.diff(np.concatenate(([0], band.view(np.int8), [0])))
        bst, ben = np.nonzero(bd == 1)[0], np.nonzero(bd == -1)[0]
        for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            if b - a < lo:
                continue
            k = np.searchsorted(bst, a, side="right") - 1
            a, b = bst[k], ben[k]               # the whole line this piece is part of
            if not lo <= b - a <= hi:
                continue
            if spread[a:b].mean() < 0.45:     # dots and dashes, not scattered specks
                continue
            r = band
            a, b = a + x0, b + x0
            # a rule is thin: the band just below it is nearly empty
            if ink[y + 3:y + int(2.5 * HMM), a:b].mean() > 0.08:
                continue
            # it stops: a longer line drifting across rows goes on past an end
            if any(e.size and e.any(axis=0).all() for e in (
                    ink[y - 3:y + 5, max(a - int(3 * HMM), 0):a],
                    ink[y - 3:y + 5, b:b + int(3 * HMM)])):
                continue
            # alone on its row: table rules come in rows
            s0 = max(a - x0 - int(30 * HMM), 0)
            near = r[s0:b - x0 + int(30 * HMM)].copy()
            near[a - x0 - s0:b - x0 - s0] = 0
            if _longest_run(near) >= 8 * HMM:
                continue
            if not _header_above(ink, y, (a + b) // 2, b - a):
                continue
            # free ends: not a table line
            if any(e.size and e.any(axis=1).mean() > 0.5 for e in (
                    ink[y - int(3 * HMM):y + int(3 * HMM), max(a - 4, 0):max(a - 1, 0)],
                    ink[y - int(3 * HMM):y + int(3 * HMM), b + 1:b + 4])):
                continue
            return True
    return False


def documents(rows):
    """Number the documents of a file and mark the pages of paper ones.

    A document starts at a header page or a name-stamp page, or after the
    previous one's end page or cover, and ends at an end page: a page with
    an end stamp (see end_stamp) or a signature image (see grey_sign). An
    appendix (see appendix_title) after an end page belongs to the document
    before it: the body is signed, then its appendices follow. A cover is a page that
    starts a document, is unstamped and without a header, and is nearly
    empty or holds only a few lines of text (see COVER_LINES).
    An unstamped, unheaded run at the file's end, after an end page, is an
    appendix and belongs to the document before it.
    Closed by an end stamp on a page that is a full-page image, a document was paper:
    every page of it is "yes", except
    a typed page (a text layer with no image under it) joined to a scan. A
    document with no end page, or a typed end page, is "unknown" for pages
    that are not themselves full-page images."""
    doc = 0
    for i, r in enumerate(rows):
        prev = rows[i - 1] if i else None
        ended = prev and (prev.get("end_stamp") or prev.get("grey_sign"))
        start = (not prev or r.get("header") or r.get("name_stamp") or prev.get("cover")
                 or ended and not r.get("appendix"))
        if prev and start:
            doc += 1
        r["doc"] = doc
        sparse = r.get("ink", 1) < COVER_INK or (
            r.get("lines", 99) <= COVER_LINES and r.get("line_rows", 1) < COVER_ROWS)
        r["cover"] = bool(start and not r.get("skip") and not r.get("header")
                          and r.get("stamp") == "no" and sparse)
    tail = [r for r in rows if r["doc"] == doc]
    i = len(rows) - len(tail)
    if i and (rows[i - 1].get("end_stamp") or rows[i - 1].get("grey_sign")) and not any(
            r.get("header") or r.get("cover") or r.get("stamp") == "yes" for r in tail):
        for r in tail:
            r["doc"] = doc - 1
    for d in sorted({r["doc"] for r in rows}):
        part = [r for r in rows if r["doc"] == d]
        last = next((r for r in reversed(part) if r.get("end_stamp")), part[-1])
        paper = last.get("end_stamp") and last.get("page_image")
        for r in part:
            if r["text_layer"] and not r.get("page_image"):
                r["paper"] = "no"               # typed text with no scan under it
            elif paper or r.get("page_image") and not r["text_layer"]:
                r["paper"] = "yes"
            elif not any(x.get("page_image") for x in part):
                r["paper"] = "no"
            else:
                r["paper"] = "unknown"
            r["text_from_paper"] = r["paper"] == "yes" if r["text_layer"] else None
    return rows


# ---------- labels ----------

def member_list(text):
    return bool(LIST_HEAD.search(text)) or sum(c.isupper() for c in PERSON.findall(text)) >= PERSONS_FOR_LIST


def label_file(path):
    pages = read_pages(path)
    listed = image_list(path)
    modes = image_modes(listed)
    full = page_images(listed, pages)
    with tempfile.TemporaryDirectory() as tmp:
        imgs = render(path, tmp)
        tops = render_tops(path, tmp, max((p["h"] for p in pages), default=842))
    rows = []
    for n, pg in enumerate(pages, 1):
        boxes = signature_boxes(pg["blocks"])
        raw = nchars("\n".join(l for _, ls in pg["blocks"] for l in ls))
        text = kept_text(pg["blocks"], boxes)
        chars = nchars(text)
        has_text = chars >= SCAN_CHARS_PER_PAGE
        img = imgs.get(n)
        row = {"file": os.path.basename(path), "page": n, "chars_raw": raw,
               "chars": chars, "sig_box": bool(boxes), "text_layer": has_text}
        if img is None or not (pg["w"] and pg["h"]):   # a broken page tree can give a page no size
            row.update(stamp="unknown", header=None, end_stamp=None, name_stamp=None, grey_sign=None, grey_stamp=None, appendix=None,
                       page_image=n in full,
                       skip=None, member_list=None, error="render")
            rows.append(row)
            continue
        grey = modes.get(n) == "grey"   # no colour: a red mark cannot show
        stamps = find_stamps(img)
        ends = [s for s in stamps if end_stamp(s, img.shape[:2])]
        greys = []
        if GREY_STAMPS and not stamps and n in modes:   # a stamp copied in grey?
            with tempfile.TemporaryDirectory() as tmp:
                g = render_grey(path, n, tmp)
            if g is not None:
                greys = [s for s in grey_stamps(g) if end_stamp(s, g.shape, GMM)]
        if stamps or greys:
            stamp = "yes"
        else:
            stamp = "unknown" if grey else "no"   # a colour image, or no image at all
        header = header_in_text(pg) if raw >= 20 else False
        appendix = appendix_title(pg) if raw >= 20 else None   # a scan's title cannot be read
        if not header and n in tops and pg["h"] > pg["w"]:   # headed pages are portrait
            header = header_in_pixels(tops[n], pg["h"] / 72 * HEAD_DPI)
        clean = blank_out(img, boxes, pg["w"], pg["h"])
        ink = ink_share(clean)
        lines, line_rows = text_lines(clean)
        lst = member_list(text) if has_text else None
        if ink < INK_BLANK and chars < 20:
            skip = "signature-only" if boxes else "blank"
        elif lst:
            skip = "member-list"
        else:
            skip = None
        row.update(images=modes.get(n), page_image=n in full, ink=round(ink, 4),
                   lines=lines, line_rows=round(line_rows, 3),
                   stamp=stamp, header=header, end_stamp=bool(ends or greys), grey_stamp=bool(greys),
                   name_stamp=any(name_stamp(s) for s in stamps), grey_sign=grey_sign(img),
                   appendix=appendix, member_list=lst, skip=skip)
        rows.append(row)
    return documents(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    counts = {}
    with open(a.out, "w", encoding="utf-8") as fh:
        for i, path in enumerate(a.pdfs, 1):
            try:
                rows = label_file(path)
            except Exception as e:
                print(f"[{i}/{len(a.pdfs)}] {os.path.basename(path)}: "
                      f"{type(e).__name__}", file=sys.stderr, flush=True)
                counts["files_failed"] = counts.get("files_failed", 0) + 1
                continue
            counts["files"] = counts.get("files", 0) + 1
            for r in rows:
                fh.write(json.dumps(r) + "\n")
                k = ("skip:" + r["skip"] if r.get("skip") else
                     ("text" if r["text_layer"] else "scan") + " stamp:" + r["stamp"])
                counts[k] = counts.get(k, 0) + 1
                if r.get("skip"):
                    continue            # the rest counts kept pages only, as the scores do
                for k in ("header", "end_stamp", "grey_stamp", "name_stamp", "grey_sign", "page_image"):
                    if r.get(k):
                        counts[k] = counts.get(k, 0) + 1
                k = "paper:" + str(r.get("paper")) + (" text" if r["text_layer"] else "")
                counts[k] = counts.get(k, 0) + 1
            if i % 25 == 0:
                print(f"[{i}/{len(a.pdfs)}]", file=sys.stderr, flush=True)
    print(json.dumps(dict(sorted(counts.items())), indent=2))


if __name__ == "__main__":
    main()
