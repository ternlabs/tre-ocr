"""Pull a disclosure's heading fields out of the joined lines: company, document
number, place and date. Part of the tool (v0.2); METHOD.md, "The tool".

Step 9 measured whether a field's value is in the output, not whether a rule can pick
it out, so this rule has its own test: built on sets C, D and E, scored once on F.

The heading is the first HEAD_LINES lines, which tools/two_readers.py keeps in its two
columns: the company and "Số: …" on the left, the national title, motto and "place,
ngày … tháng … năm …" on the right.

- **Number:** the rest of the one line that opens with "Số", "No" or "Number".
- **Place and date:** the one line "place, ngày … tháng … năm yyyy", or, with none, its
  English form "place, Month d, yyyy".
- **Company:** the line that opens with a company word, before the number; a group's
  name above it ("TỔNG CÔNG TY", "TẬP ĐOÀN") gives way to the company under it. Lines
  after it join while they are capitals and Vietnamese (an English name under it does not
  join), up to the number.

A field is filled only when the rule finds exactly one value and every line it rests on
is sure; else it is empty, with what was found kept as "seen". A value is compared as
score.py's fields are: case and spaces dropped, and without marks for the pass mark.

Usage:
  python -I tools/fields.py c d e        # build: every page, value against truth
  python -I tools/fields.py f --once     # the test
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from score import SETS, base, load_truth, squeeze, strip  # noqa: E402
from syllables import normal  # noqa: E402

HEAD_LINES = 20
NUMBER_RE = re.compile(r"^\s*(?:s[ốoóồổôò6]?|no|number)\s*(?:/\s*no\b\.?)?\s*[.:]\s*(.*)$", re.I)
VI_DATE_RE = re.compile(r"^(.*?)\s*,\s*(ngày.*?năm\s*\d{4})", re.I)
MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
EN_DATE_RE = re.compile(rf"^(.*?)\s*,\s*((?:{MONTHS})[a-z]*\.?\s*\d{{1,2}}\s*(?:st|nd|rd|th)?\s*,?\s*\d{{4}})\s*$", re.I)
# Matched on the line without marks, lower case.
COMPANY_RE = re.compile(r"^(?:tong\s+cong\s+ty|cong\s+ty|ngan\s+hang|tap\s+doan|so\s+giao\s+dich|"
                        r"so\s+\w+|uy\s+ban)\b")
GROUP_RE = re.compile(r"^(?:tong\s+cong\s+ty|tap\s+doan)\b")
GENERIC_RE = re.compile(r"^(?:cong\s+ty\s+(?:co\s+phan|cp|tnhh)|cong\s+ty\s+co|cong\s+ty)\W*$")
TITLE_RE = re.compile(r"cong\W*hoa|doc\W*lap|tu\W*do|hanh\W*phuc|socialist|independence")
BRANCH_RE = re.compile(r"^(?:chi\s+nhanh|phong)\b")   # a branch or office of the one named
EN_NAME_RE = re.compile(r"\b(?:company|corporation|corp|jsc|bank|group)\b", re.I)
NOT_PLACE_RE = re.compile(r"^(?:hôm nay|today)\b", re.I)
FILL_RE = re.compile(r"\.{2,}|…|_+")
DAY_MONTH_RE = re.compile(r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm", re.I)


def plain(line: str) -> str:
    return " ".join(base(line).casefold().split())


def caps(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def vi(line: str) -> bool:
    return base(line) != line


def before_title(line: str) -> str:
    """A box over both columns: the part before the national title."""
    b, at = strip(line)
    m = TITLE_RE.search(b.casefold())
    return line[:at[m.start()]].strip() if m else line


def clean_number(v: str) -> str:
    return FILL_RE.sub("", v).strip(" .:")


def clean_date(v: str) -> str | None:
    v = FILL_RE.sub("", v).replace(".", "")
    m = DAY_MONTH_RE.search(v)
    if m and not (1 <= int(m.group(1)) <= 31 and 1 <= int(m.group(2)) <= 12):
        return None
    return v if m or EN_DATE_RE.match("x, " + v) else None


def company(head: list[str], stop: int) -> list[tuple[str, list[int]]]:
    starts = [i for i in range(stop) if COMPANY_RE.match(plain(head[i]))]
    while len(starts) > 1 and GROUP_RE.match(plain(head[starts[0]])) and not GROUP_RE.match(plain(head[starts[1]])):
        starts = starts[1:]
    if not starts:
        run = [i for i in range(stop) if caps(head[i]) and not vi(head[i]) and not TITLE_RE.search(plain(head[i]))]
        name = " ".join(head[i] for i in run)
        name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
        return [(name, run)] if run and EN_NAME_RE.search(name) and run == list(range(run[0], run[-1] + 1)) else []
    i = starts[0]
    parts, run = [before_title(head[i])], [i]
    for j in range(i + 1, stop):
        x, px = head[j], plain(head[j])
        generic = GENERIC_RE.match(plain(" ".join(parts)))
        if TITLE_RE.search(px) or (COMPANY_RE.match(px) and not generic):
            if j in starts:
                starts = [k for k in starts if k != j] + [j]
            continue
        if not caps(x) or not vi(x) and not generic or BRANCH_RE.match(px):
            break
        parts.append(before_title(x))
        run.append(j)
        if j in starts:
            starts.remove(j)
    found = [(re.sub(r"\s*\([^)]*\)\s*$", "", " ".join(parts)), run)]   # "(ONE CORP)": a short name after it
    return found + [(head[k], [k]) for k in starts[1:] if k not in run]


def extract(lines: list[str], unsure: list[int]) -> dict:
    """Each field: {"value": str or None, "lines": [...], "seen": [...]}"""
    head = [normal(x).strip() for x in lines[:HEAD_LINES]]
    out = {}

    nums = []
    for i, x in enumerate(head):
        m = NUMBER_RE.match(x)
        if not m:
            continue
        v, at = m.group(1), [i]
        if not v.strip() and i + 1 < len(head):
            v, at = head[i + 1], [i, i + 1]
        v = clean_number(v)
        if any(c.isdigit() for c in v):
            nums.append((v, at))
    out["number"] = nums

    dated = [(i, m) for i, x in enumerate(head) if (m := VI_DATE_RE.match(x))] or \
            [(i, m) for i, x in enumerate(head) if (m := EN_DATE_RE.match(x))]
    dated = [(i, m) for i, m in dated if not NOT_PLACE_RE.match(m.group(1))
             and not any(c.isdigit() for c in m.group(1)) and len(m.group(1).split()) <= 6]
    out["place"] = [(m.group(1).strip(), [i]) for i, m in dated]
    out["date"] = [(d, [i]) for i, m in dated if (d := clean_date(m.group(2).strip()))]

    out["company"] = company(head, nums[0][1][0] if nums else len(head))

    bad = set(unsure)
    res = {}
    for name, got in out.items():
        values = {squeeze(v) for v, _ in got}
        if len(values) == 1:
            v, at = got[0]
            ok = not bad & set(at)
            res[name] = {"value": v if ok else None, "lines": at, "seen": [v]}
        else:
            res[name] = {"value": None, "lines": [], "seen": [v for v, _ in got]}
    return res


def compare(value: str, truth: str) -> tuple[bool, bool]:
    """(right without marks, right with marks)"""
    v, t = squeeze(normal(value)), squeeze(normal(truth))
    return squeeze(base(v)) == squeeze(base(t)), v == t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--once", action="store_true", help="the test: counts only, no values shown")
    ap.add_argument("--show", action="store_true", help="every wrong or empty field, for building")
    a = ap.parse_args()
    for s in a.sets:
        truth, _ = load_truth(s, None)
        run = {(r["sha256"], r["page"]): r for r in
               map(json.loads, open(f"data/runs/two-readers-{s}.jsonl", encoding="utf-8"))}
        c = Counter()
        for k, t in truth.items():
            h = t.get("heading") or {}
            if not any(h.values()) or k not in run:
                continue
            got = extract(run[k]["text"].split("\n"), run[k]["unsure"])
            for name in ("company", "number", "place", "date"):
                tv = h.get(name) or ""
                g = got[name]
                if not tv:
                    c[f"{name} no truth"] += 1
                    c[f"{name} no truth, filled"] += g["value"] is not None
                    continue
                c[f"{name} truth"] += 1
                if g["value"] is None:
                    c[f"{name} empty"] += 1
                    seen_right = any(compare(v, tv)[0] for v in g["seen"])
                    c[f"{name} empty, seen right"] += seen_right
                    if a.show and not seen_right:
                        print(f"{s} {k[0][:8]}-p{k[1]} {name} EMPTY seen={g['seen']} truth={tv!r}")
                    continue
                c[f"{name} filled"] += 1
                nb, wm = compare(g["value"], tv)
                c[f"{name} filled right"] += nb
                c[f"{name} filled right with marks"] += wm
                if a.show and not nb:
                    print(f"{s} {k[0][:8]}-p{k[1]} {name} WRONG got={g['value']!r} truth={tv!r}")
        print(s, json.dumps(dict(sorted(c.items())), ensure_ascii=False))


if __name__ == "__main__":
    main()
