"""Align each set-A scan page to the stretch of English text it shows.

Step 4 of METHOD.md. Pairing page n with page n fails when the two languages
lay out differently: a page break falls a few lines apart, so the English page
holds numbers the scan does not, or lacks some it has. This finds, in the
English pages around the paired one, the text the scan holds.

Anchors are what reads the same in both languages: heading numbers ("Điều 12"
and "Article 12", "II.", "3.") and numbers of 3+ digits. The scan's anchors
come from OCR (EasyOCR, a build tool here, not one of the engines scored); the
English anchors from the text layer, over the paired page and the pages either
side. A local alignment of the two streams gives the anchors both hold, in
order; it favours the paired page and charges for skipped tokens, so a running
header or a year repeated on a neighbouring page does not stretch it. The truth
is every English number from the first matched anchor to the last, those the
OCR missed included. Numbers before the first or after the last are left out:
where the page breaks there is not known. So the truth can lose numbers at a
page's edges (more where the OCR reads badly), but should hold none the scan
lacks.

A page is aligned when at least MIN_ANCHORS anchors match. Below that it is
"unaligned": in a hand check of all such pages, none was a usable pair (forms,
signature pages, another document of the filing). A scan whose OCR shows a
filled identity field (ID card, date of birth) is "identity": the English page
was checked for these, but a twin may leave them out. Only aligned pages stay
in the set; --prune rewrites the set file without the others.

Matches made only of years and list numbers are weak: two documents of one
filing (a decision and its cover letter, an agenda and a ballot) share them.
Such pages are worth a look by eye; see METHOD.md, step 4.

Writes <out>, one line per set entry, not committed (it holds numbers that can
be a person's): the Vietnamese page, status, anchors matched, the English span,
the truth numbers, and how many numbers of the paired English page fell outside
the span. OCR output is cached under <work>.

Usage:
  python -I tools/align_pages.py sets/disclosures-a.jsonl --out data/align-a.jsonl [--prune]
  python -I tools/align_pages.py sets/disclosures-a.jsonl --out /tmp/x.jsonl --rows 6,20
"""
import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from label_pages import kept_text, read_pages, signature_boxes  # noqa: E402
from number_truth import cached  # noqa: E402
from records import NUMBER_RE  # noqa: E402

MIN_ANCHORS = 3
NEIGHBOUR_MATCH = 0.6   # a match on the English page before or after the paired one
SKIP = 0.3              # a token passed over on either side
# A Vietnamese identity label, then a value, on one OCR line; tone marks optional
# as OCR may drop them.
VI_IDENTITY_RE = re.compile(r"(sinh ng[aà]y|ng[aà]y sinh|n[aă]m sinh|cccd|cmnd|c[aă]n c[uư][oớ]c"
                            r"|h[oộ] chi[eế]u)[^\d\n]{0,40}\d[\d./-]{5,}", re.I)
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9,
         "x": 10, "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15}
HEADINGS = {   # kind: (Vietnamese, English), each matched at a line's start
    "article": (r"[đd]i[^\s\d]{0,2}u\s*(\d{1,3})\b", r"article\s*(\d{1,3})\b"),
    "chapter": (r"ch[ưu][ơo]ng\s*([ivx]{1,5}|\d{1,2})\b", r"chapter\s*([ivx]{1,5}|\d{1,2})\b"),
    "roman": (r"([ivx]{1,5})\.\s", r"([ivx]{1,5})\.\s"),
    "item": (r"(\d{1,2})\.\s*[^\d\s]", r"(\d{1,2})\.\s*[^\d\s]"),
}


def tokens(lines: list[str], lang: int) -> list[tuple]:
    """(kind, key) per heading and number, in reading order; lang 0 is vi, 1 en."""
    out = []
    for line in lines:
        s = line.strip().lower()
        for kind, pats in HEADINGS.items():
            m = re.match(pats[lang], s)
            if m:
                k = m.group(1)
                if k in ROMAN and kind != "item":
                    k = str(ROMAN[k])
                elif not k.isdigit():
                    continue
                out.append(("h:" + kind, int(k)))
                break
        for m in NUMBER_RE.findall(line):
            d = re.sub(r"[.,]", "", m)
            if len(d) >= 3:
                out.append(("n", d))
    return out


def local_align(a: list, b: list, home: list[bool]) -> list[tuple[int, int]]:
    """Index pairs of the best local alignment of a (scan) and b (English):
    a match scores 1 on the paired page (home) and NEIGHBOUR_MATCH next to it,
    and each token skipped on either side costs SKIP. So a number repeated on
    a neighbouring page (a running header, a year) joins only when it extends
    the run, and a far-off stray match does not pay."""
    n, m = len(a), len(b)
    H = [[0.0] * (m + 1) for _ in range(n + 1)]
    best, at = 0.0, (0, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            h = max(0.0, H[i - 1][j] - SKIP, H[i][j - 1] - SKIP)
            if a[i - 1] == b[j - 1]:
                h = max(h, H[i - 1][j - 1] + (1.0 if home[j - 1] else NEIGHBOUR_MATCH))
            H[i][j] = h
            if h > best:
                best, at = h, (i, j)
    out, (i, j) = [], at
    while i > 0 and j > 0 and H[i][j] > 0:
        if a[i - 1] == b[j - 1] and H[i][j] == H[i - 1][j - 1] + (1.0 if home[j - 1] else NEIGHBOUR_MATCH):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif H[i][j] == H[i - 1][j] - SKIP:
            i -= 1
        else:
            j -= 1
    return out[::-1]


class Ocr:
    """EasyOCR boxes per scan page, cached as JSON under <work>/ocr."""

    def __init__(self, work):
        self.dir = os.path.join(work, "ocr")
        os.makedirs(self.dir, exist_ok=True)
        self.reader = None

    def lines(self, path: str, sha: str, page: int) -> list[str]:
        out = os.path.join(self.dir, f"{sha[:16]}-{page}.json")
        if not os.path.exists(out):
            if self.reader is None:
                import easyocr
                self.reader = easyocr.Reader(["vi"], gpu=False, verbose=False)
            stem = os.path.join(self.dir, "page")
            subprocess.run(["pdftoppm", "-q", "-png", "-gray", "-r", "200", "-f", str(page),
                            "-l", str(page), "-singlefile", path, stem], check=True)
            boxes = [[float(min(p[1] for p in bb)), float(min(p[0] for p in bb)),
                      float(max(p[1] for p in bb)), text]
                     for bb, text, _ in self.reader.readtext(stem + ".png")]
            os.remove(stem + ".png")
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(boxes, fh, ensure_ascii=False)
        with open(out, encoding="utf-8") as fh:
            return box_lines(json.load(fh))


def box_lines(boxes: list[list]) -> list[str]:
    """Boxes (top, left, bottom, text) into lines: a box joins the line whose
    middle it overlaps."""
    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    rows = []
    for b in boxes:
        mid = (b[0] + b[2]) / 2
        if rows and rows[-1][0] <= mid <= rows[-1][1]:
            rows[-1][2].append(b)
        else:
            rows.append([b[0], b[2], [b]])
    return [" ".join(b[3] for b in sorted(r[2], key=lambda b: b[1])) for r in rows]


def en_lines(path: str, pages: list[int]) -> list[tuple[int, str]]:
    text = read_pages(path)
    out = []
    for p in pages:
        if 1 <= p <= len(text):
            b = text[p - 1]["blocks"]
            out += [(p, l) for l in kept_text(b, signature_boxes(b)).split("\n")]
    return out


def align(r: dict, ocr: Ocr, cache: str) -> dict:
    scan = ocr.lines(cached(r["vi"]["pdf"], cache), r["vi"]["sha256"], r["vi"]["page"])
    vi = tokens(scan, 0)
    e = r["en"]["page"]
    lines = en_lines(cached(r["en"]["pdf"], cache), [e - 1, e, e + 1])
    en = [(p, t) for p, l in lines for t in tokens([l], 1)]
    pairs = local_align(vi, [t for _, t in en], [p == e for p, _ in en])
    out = {"vi": {"sha256": r["vi"]["sha256"], "page": r["vi"]["page"]},
           "anchors": len(pairs), "scan_anchors": len(vi)}
    if any(VI_IDENTITY_RE.search(l) for l in scan):
        return {**out, "status": "identity"}
    if len(pairs) < MIN_ANCHORS:
        return {**out, "status": "unaligned"}
    a, b = pairs[0][1], pairs[-1][1]
    span = en[a:b + 1]
    on_page = sum(1 for p, t in en if p == e and t[0] == "n")
    in_span = sum(1 for p, t in span if p == e and t[0] == "n")
    return {**out, "status": "aligned",
            "en": {"sha256": r["en"]["sha256"], "from": span[0][0], "to": span[-1][0]},
            "numbers": [t[1] for _, t in span if t[0] == "n"],
            "page_numbers_left_out": on_page - in_span,
            "numbers_from_next_pages": sum(1 for p, t in span if p != e and t[0] == "n")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="data/align-a")
    ap.add_argument("--cache", default="cache/pdfs")
    ap.add_argument("--rows", default="", help="comma-separated row numbers (0-based) only")
    ap.add_argument("--prune", action="store_true", help="rewrite the set with aligned pages only")
    a = ap.parse_args()
    if a.prune and a.rows:
        ap.error("--prune needs every row")
    with open(a.set, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh]
    todo = [int(x) for x in a.rows.split(",")] if a.rows else range(len(rows))
    ocr = Ocr(a.work)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    keep = []
    with open(a.out, "w", encoding="utf-8") as fh:
        for i in todo:
            res = {"row": i, **align(rows[i], ocr, a.cache)}
            if res["status"] == "aligned":
                keep.append(rows[i])
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(f"[{i}] {rows[i]['symbol']} {rows[i]['paired_by']}: {res['status']}, "
                  f"{res['anchors']}/{res['scan_anchors']} anchors"
                  + (f", en p{res['en']['from']}-{res['en']['to']}, {len(res['numbers'])} numbers, "
                     f"{res['page_numbers_left_out']} of the page left out, "
                     f"{res['numbers_from_next_pages']} from neighbours"
                     if res["status"] == "aligned" else ""), file=sys.stderr)
    if a.prune:
        with open(a.set, "w", encoding="utf-8") as fh:
            for r in keep:
                fh.write(json.dumps(r) + "\n")
        print(f"{a.set}: kept {len(keep)} of {len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
