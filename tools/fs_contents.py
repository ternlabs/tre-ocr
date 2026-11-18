"""The contents page as a second check on where the statements are (profile 2): a flag, never a move.

Many filings open with a contents page (MỤC LỤC) giving each statement's printed page.
Printed pages are off from the PDF's by a few (a cover, a report unnumbered), but by one
offset across a filing. The offset is the one most statements found by tools/fs_rows.py
agree on (on a tie, the smaller: a filing's unnumbered pages are few); then each statement the contents list should start at its printed page plus
it. Where it does not, or is listed and not found, or found and not listed, a flag says
so for a person, as does a statement found again outside its listed pages (running on to
the next statement listed: a listed page can be the first of two). The statements themselves are left as found: the contents page is read
by OCR too, and a page number on it can be misread or the filer's own slip.

Read: the contents page is one of the first LOOK_IN pages with "mục lục" or "nội dung" as a
line of its own; a statement's line there is a line with its title (not a note's, "thuyết minh") and
the page number is the last number on that row to its right, or ending the line itself
after dot leaders, "5" or a range "5-6". Only the first page of a range is checked: ranges are often loose.
A title whose number was not read (the detector often leaves a lone page number unboxed)
is listed with no page: nothing to check, and no flag. A page with no statement's number
read (a letter whose subject names the results) is not taken as contents.

Usage: .venv/bin/python -I tools/fs_contents.py some.draft.json [its.pdf]
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from fs_rows import layer_lines, statements_of  # noqa: E402
from sort_kinds import fold  # noqa: E402

LOOK_IN = 4                                  # the contents page is among the first pages
CONTENTS_RE = re.compile(r"\W*(?:muc luc|noi dung)\W*")          # a line of its own
LISTED = {"B01": re.compile(r"can doi ke toan|tinh hinh tai chinh"),
          "B02": re.compile(r"ket qua (?:hoat dong )?(?:san xuat )?kinh doanh"),
          "B03": re.compile(r"luu chuyen tien te")}
NOT_LISTED = re.compile(r"thuyet minh")      # "notes to the balance sheet" names no statement
PAGE_RE = re.compile(r"(\d{1,3})\s*(?:[-–]\s*(\d{1,3}))?")
ALONE_RE = re.compile(r"\s*" + PAGE_RE.pattern + r"\s*")
TRAIL_RE = re.compile(r"(?:\.{2,}|…)\s*" + PAGE_RE.pattern + r"\s*$")   # after dot leaders


def page_of(text: str, alone: bool) -> tuple[int, int] | None:
    m = (ALONE_RE.fullmatch if alone else TRAIL_RE.search)(text)
    if not m:
        return None
    a = int(m.group(1))
    return a, int(m.group(2) or a)


def listed(pages: list[dict], pdf: str | None = None) -> tuple[int | None, dict[str, tuple[int, int] | None]]:
    """The contents page's number and, per statement it lists, its printed (first, last) pages,
    None where its number was not read."""
    layer = None
    for p in pages[:LOOK_IN]:
        lines = p.get("lines") or []
        if p.get("kind") == "text layer" and pdf:
            layer = layer or layer_lines(pdf)
            lines = layer.get(p["page"], ([], None))[0]
        lines = [ln for ln in lines if ln.get("box")]
        if not any(CONTENTS_RE.fullmatch(fold(ln["text"])) for ln in lines):
            continue
        out: dict[str, tuple[int, int] | None] = {}
        for ln in lines:
            t = fold(ln["text"])
            kind = next((k for k, pat in LISTED.items() if pat.search(t)), None)
            if kind is None or out.get(kind) or NOT_LISTED.search(t):
                continue
            mid = (ln["box"][1] + ln["box"][3]) / 2
            right = sorted((m for m in lines if m is not ln and m["box"][1] < mid < m["box"][3]
                            and m["box"][0] > ln["box"][2]), key=lambda m: m["box"][0])
            got = next((g for m in reversed(right) for g in [page_of(m["text"], True)] if g), None)
            out[kind] = got or page_of(ln["text"], False)
        if any(out.values()):        # a page naming statements with no page numbers is no contents
            return p["page"], out
    return None, {}


def check(draft: dict, statements: list[dict], pdf: str | None = None) -> dict | None:
    """The contents page against the statements found: {"page", "listed", "offset", "flags"};
    None with no contents page. flags is empty when the two agree."""
    at, lists = listed(draft["pages"], pdf)
    if at is None:
        return None
    starts: dict[str, list[int]] = {}
    for s in statements:
        if s["pages"]:
            starts.setdefault(s["statement"], []).append(s["pages"][0])
    printed = {k: v[0] for k, v in lists.items() if v}
    # a statement listed by its first page alone runs on to the page before the next one
    # listed; the last one listed, to its own last page (the notes after it are not read)
    last = {k: max(v[1], min((b - 1 for b in printed.values() if b > v[0]), default=v[1]))
            for k, v in lists.items() if v}
    votes = Counter(o for k, a in printed.items() for o in {b - a for b in starts.get(k, [])})
    offset = min(votes, key=lambda o: (-votes[o], abs(o))) if votes else None
    flags = []
    for k, a in sorted(printed.items()):
        where = f"page {a + offset}" if offset is not None else "no offset to place it"
        if k not in starts:
            flags.append(f"{k}: listed at printed page {a} ({where}), not found")
        elif offset is not None and a + offset not in starts[k]:
            flags.append(f"{k}: listed at printed page {a} ({where}), found starting at page "
                         f"{', '.join(map(str, sorted(starts[k])))}")
        elif offset is not None:
            out = [b for b in starts[k] if not a + offset <= b <= last[k] + offset]
            if out:
                flags.append(f"{k}: listed at pages {a + offset}-{last[k] + offset}, found starting "
                             f"again at page {', '.join(map(str, out))}")
    for k in sorted(set(starts) - set(lists)):
        flags.append(f"{k}: found starting at page {', '.join(map(str, sorted(starts[k])))}, "
                     f"not read in the contents")
    return {"page": at, "listed": lists, "offset": offset, "flags": flags}


def main():
    with open(sys.argv[1], encoding="utf-8") as fh:
        d = json.load(fh)
    pdf = sys.argv[2] if len(sys.argv) > 2 else os.path.join("cache", "pdfs", d["file"])
    pdf = pdf if os.path.exists(pdf) else None
    got = check(d, statements_of(d, pdf, images=os.path.join(os.path.dirname(sys.argv[1]), "images")), pdf)
    if got is None:
        print("no contents page")
        return
    print(f"contents on page {got['page']}: {got['listed']}, offset {got['offset']}")
    for f in got["flags"] or ["agrees"]:
        print(" ", f)


if __name__ == "__main__":
    main()
