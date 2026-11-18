"""The statement forms' sums as data, and a check of them on a set's truth (profile 2, FS3).

Each sum is a total line and the lines it adds up, as the circulars give them: the 2014
form (TT200, with TT202's lines for groups) and the 2026 form (TT99/2025). A sum here is
data, so the same list later makes cells unsure (a sum over sure cells that fails) and,
if set H allows it, sure.

**Signs.** Filings differ on printing deductions: a cost, or a contra line such as a
provision or accumulated depreciation, is printed in parentheses by some and plain by
others, sometimes within one statement. A deduction therefore counts as minus what is
printed or as minus its digits, whichever holds; every other line counts as printed. A
filing that prints its costs in parentheses may print one plain to mean a credit (a
deferred tax benefit), so there the deductions may also count with their printed sign. The filing's own
sign is never changed in the truth: this only decides whether the sum holds.

**Ranges.** Where the form leaves room for lines a group's form or a later circular
adds (B03's sections, the 2026 form's equity), a sum takes every code read in a range
rather than a fixed list. Absent lines count as zero; a line read but unsure makes the
sum unknown. So does a line the filing prints whose figure the reader did not get, and
a line the pair's other filing prints that this reading lacks. A sum holding with such
lines taken as blank holds. A printed blank and a missed figure cannot be told apart
from the reading: a row empty in every column is as often a figure the reader lost (on
a wrapped label's second line, at the top of a tall cell) as an empty line, and taking
it as zero turned those misses into failures on H. A subtotal the filing leaves out is
replaced by its own lines.

**Blank cells.** A cell the page was looked at for and found without ink (read_pdf's
blank_cells, a row's "blank") is told apart from a missed figure. It counts as zero, as a
line or as the total, only where the sum then holds, and such sums are counted apart; a
blank total over lines with figures stays unknown, never a failure. The cell itself stays
empty in the reading.

**Lines a filing does not print.** A line the pair's other filing prints and this reading
lacks is unread, as above, unless the page shows it is not there (`no_room`): the lines
numbered just under and just over it were read as rows one after the other on one page,
closer than a row's height. Then the line is absent, as any line of the form the filing
leaves out.

**Groups.** A group's profit split (B02 60 = 61 + 62) is checked where the filing is
consolidated, or where one of the split's lines has a figure other than zero. A
company's own statements print the split empty, and that is not a sum to check; a
group's split the reader missed is unknown, as any other sum. A title that names
neither scope is taken as a company's own: a group's statements are titled as such.

**The check on truth** (`main`): each filing's side of each truth cell is taken as true
where the cross-period readings agree or a person typed it, and as read otherwise (a
line only one filing prints, or the sure side of an unsure cell). Lines the 2026 form
dropped have no truth cell, so for a 2014 form side they come from that filing's reading.
A sum that fails on true values alone is a fault of the sum, not of the reading, unless
one of its lines sits next to a row read without its code (the line may be that row),
or the page itself prints a total its lines do not make; the first is marked in the
output, the second is found by looking at the page.

**Sure.** A sum that fails makes its cells unsure. A sum that holds makes nothing sure:
set H could not test that, since none of its readings that reached the truth had a
wrong figure for a holding sum to hide.

**A cell a sum leaves open.** Where every cell of a sum is read but
one, the sum says what that one holds (`demands`). The figure is never filled from the
sum: read_pdf's sum_look reads the cell's crop once more and keeps a reading only when it
is that figure, digit for digit. The page says it and the sum agrees, which is more than
either alone; such a cell is marked (a line's "look" is "sum"), and a sum that holds
through one is counted apart here, since it had a say in its own cell.

Prints counts and line codes only, never a figure.

Usage: .venv/bin/python -I tools/fs_sums.py sets/statements-h.jsonl data/set-h/drafts data/truth-h
"""
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from fs_codes import OLD_TO_NEW, form_of  # noqa: E402


def span(lo: int, hi: int) -> str:
    return f"{lo}..{hi}"


def terms(*xs) -> list[tuple[int, str]]:
    """(+1 or -1, code or range); a code written "-22" is a deduction."""
    return [(-1, x[1:]) if x.startswith("-") else (+1, x) for x in xs]


# total -> alternatives, each a list of (sign, code or "lo..hi"). Codes as the form prints
# them, without the leading zero dropped: B02 and B03 codes are two digits.
SUMS = {
    "2014": {
        "B01": {
            "100": [terms("110", "120", "130", "140", "150")],
            "110": [terms("111", "112")],
            "120": [terms("121", "-122", "123")],
            "130": [terms("131", "132", "133", "134", "135", "136", "-137", "139")],
            "140": [terms("141", "-149")],
            "150": [terms(span(151, 155))],
            "200": [terms("210", "220", "230", "240", "250", "260")],
            "210": [terms("211", "212", "213", "214", "215", "216", "-219")],
            "220": [terms("221", "224", "227")],
            "221": [terms("222", "-223")],
            "224": [terms("225", "-226")],
            "227": [terms("228", "-229")],
            "230": [terms("231", "-232")],
            "240": [terms("241", "242")],
            "250": [terms("251", "252", "253", "-254", "255")],
            "260": [terms("261", "262", "263", "268", "269")],
            "270": [terms("100", "200")],
            "300": [terms("310", "330")],
            "310": [terms(span(311, 324))],
            "330": [terms(span(331, 343))],
            "400": [terms("410", "430")],
            "410": [terms("411", "412", "413", "414", "-415", "416", "417", "418", "419",
                          "420", "421", "422", "429")],
            "430": [terms("431", "432")],
            "440": [terms("300", "400"), terms("270")],
        },
        "B02": {
            "10": [terms("01", "-02")],
            "20": [terms("10", "-11")],
            "30": [terms("20", "21", "-22", "24", "-25", "-26")],
            "40": [terms("31", "-32")],
            "50": [terms("30", "40")],
            "60": [terms("50", "-51", "-52")],
        },
    },
    "2026": {
        "B01": {
            "100": [terms("110", "120", "130", "140", "150", "160")],
            "110": [terms("111", "112")],
            "120": [terms("121", "-122", "123", "-124", "125", "-126")],
            "130": [terms("131", "132", "133", "134", "135", "-136", "137")],
            "140": [terms("141", "-142")],
            "150": [terms("151", "152", "-153")],
            "160": [terms(span(161, 165))],
            "200": [terms("210", "220", "230", "240", "250", "260", "270")],
            "210": [terms("211", "212", "213", "214", "215", "-216")],
            "220": [terms("221", "224", "227")],
            "221": [terms("222", "-223")],
            "224": [terms("225", "-226")],
            "227": [terms("228", "-229")],
            "230": [terms("231", "236", "237", "-238")],
            "231": [terms("232", "233")],
            "233": [terms("234", "-235")],
            "240": [terms("241", "-242")],
            "250": [terms("251", "252")],
            "260": [terms("261", "262", "263", "-264", "265", "-266", span(267, 269))],
            "270": [terms(span(271, 279))],
            "280": [terms("100", "200")],
            "300": [terms("310", "330")],
            "310": [terms(span(311, 325))],
            "330": [terms(span(331, 344))],
            "400": [terms("411", "412", "413", "414", "-415", "416", "417", "418", "419",
                          "420", span(421, 429)),
                    terms("410", "430")],
            "440": [terms("300", "400"), terms("280")],
        },
        "B02": {
            "10": [terms("01", "-02")],
            "20": [terms("10", "-11")],
            "30": [terms("20", "21", "22", "-23", "27", "-25", "-26")],
            "40": [terms("31", "-32")],
            "50": [terms("30", "40")],
            "60": [terms("50", "-51", "-52")],
        },
    },
}
_B03 = {
    "08": [terms(span(1, 7))],                                  # indirect method only
    "20": [terms(span(8, 19)), terms(span(1, 7))],              # indirect, else direct
    "30": [terms(span(21, 29))],
    "40": [terms(span(31, 39))],
    "50": [terms("20", "30", "40")],
    "70": [terms("50", "60", "61")],
}
for _f in SUMS.values():
    _f["B03"] = _B03
GROUP_ONLY = {"B02": {"60": [terms("61", "62")]}}      # a group's profit split; checked too
MAX_FLEX = 6                                            # deductions tried both ways at most
BLANK = "blank"                                         # a cell's value: looked at, no ink


def expand(term: str, values: dict, width: int) -> list[str]:
    if ".." not in term:
        return [term]
    lo, hi = (int(x) for x in term.split(".."))
    return [c for c in (str(n).zfill(width) for n in range(lo, hi + 1)) if c in values]


def inline(alt: list, values: dict, rules: dict, depth: int = 2) -> list:
    """A subtotal the filing does not print is replaced by its own lines: 20 = 01 - 02 - 11
    where 10 is left out. Only a fixed list is inlined, and only its first alternative."""
    out = []
    for s, term in alt:
        if depth and ".." not in term and term not in values and term in rules:
            out += [(s * s2, t2) for s2, t2 in inline(rules[term][0], values, rules, depth - 1)]
        else:
            out.append((s, term))
    return out


def check(values: dict, total: str, alt: list, width: int):
    """(result, deductions counted as minus their digits) for one alternative.

    result: "holds", "fails", "unknown" (a line unsure), or None (the total not printed).
    values: code -> int, None when unsure or unread, or BLANK. Absent codes count as zero."""
    t = values.get(total, "absent")
    if t == "absent":
        return None, ()
    lines = [(s, c) for s, term in alt for c in expand(term, values, width)]
    if not any(c in values for _, c in lines):
        return None, ()                         # a way of adding up this filing does not print
    if t is None:
        return "unknown", ()
    blank = [c for _, c in lines if values.get(c, 0) is None]
    seen = [c for _, c in lines if values.get(c, 0) == BLANK] + [total] * (t == BLANK)
    if blank or seen:
        # unread lines as blanks: an exact hold says they are; anything else is unknown.
        # Cells seen blank likewise, the total among them, under their own name
        r = check({**values, **{c: 0 for c in blank + seen}}, total, alt, width)
        if r[0] != "holds":
            return "unknown", ()
        return "holds", r[1] + ("blanks",) * bool(blank) + ("blank cells",) * bool(seen)
    fixed = sum(s * values.get(c, 0) for s, c in lines if s > 0)
    flex = [c for s, c in lines if s < 0 and values.get(c, 0)]
    if len(flex) > MAX_FLEX:
        return ("holds", ()) if fixed - sum(values[c] for c in flex) == t else ("fails", ())
    for flip in itertools.product((False, True), repeat=len(flex)):
        got = fixed + sum(-abs(values[c]) if f else -values[c] for c, f in zip(flex, flip))
        if got == t:
            return "holds", tuple(c for c, f in zip(flex, flip) if f and values[c] < 0)
    if any(values[c] < 0 for c in flex) and fixed + sum(values[c] for c in flex) == t:
        # costs printed in parentheses, so a plain one is a credit (deferred tax, a reversal)
        return "holds", tuple(c for c in flex if values[c] < 0)
    return "fails", ()


def ways(form: str, statement: str, values: dict, group: bool = False):
    """(total, its ways of adding up with subtotals left out inlined, a group's rule) for
    every sum this filing's statement is checked on."""
    width = 3 if statement == "B01" else 2
    rules = dict(SUMS[form][statement])
    todo = [(t, a, False) for t, a in rules.items()]
    todo += [(t, a, True) for t, a in GROUP_ONLY.get(statement, {}).items()]
    for total, alts, is_group in todo:
        if is_group:
            lines = {c for alt in alts for _, term in alt for c in expand(term, values, width)}
            if not group and not any(isinstance(values.get(c), int) and values[c] for c in lines):
                continue                        # a company's own: its split absent or empty
        if statement == "B03" and total == "08" and "08" not in values:
            continue
        if statement == "B03" and total == "20":
            alts = alts[:1] if "08" in values else alts[1:]
        yield total, [inline(alt, values, rules) for alt in alts], is_group


def sums_of(form: str, statement: str, values: dict, group: bool = False) -> list[tuple]:
    """(name, total, result, deductions read as negative, the ways tried, inlined) for every
    sum whose total is printed. name is the total, or "60 split" for a group's rule.
    group: the filing is consolidated, so its split is checked even when unread."""
    width = 3 if statement == "B01" else 2
    out = []
    for total, tried, is_group in ways(form, statement, values, group):
        best = None
        for alt in tried:
            r = check(values, total, alt, width)
            if r[0] is None:
                continue
            # holds if one way holds; fails only if every way printed fails
            if best is None or r[0] == "holds" or (r[0] == "unknown" and best[0] == "fails"):
                best = r
            if r[0] == "holds":
                break
        if best is None and values.get(total, "absent") != "absent":
            # a total printed with none of its lines found: a heading left empty adds up to
            # zero, so a zero total holds; any other is unknown
            best = (("holds", ("none printed",) + ("blank cells",) * (values[total] == BLANK))
                    if values[total] in (0, BLANK) else ("unknown", ()))
        if best and best[0]:
            out.append((f"{total} split" if is_group else total, total, *best, tried))
    return out


def demands(form: str, statement: str, values: dict, group: bool = False) -> dict[str, set]:
    """code -> the figures a sum leaves it: for each sum with exactly one cell unread (None),
    its total or one of its lines, and every other read or seen blank, what that cell must
    hold for the sum to hold. More than one figure where the signs allow it: a deduction
    may be printed plain or in parentheses, here and on the sum's other lines. Never zero:
    a zero is a dash or a blank, and those are looked for by their ink."""
    width = 3 if statement == "B01" else 2
    val = lambda c: 0 if values.get(c, 0) == BLANK else values.get(c, 0)   # noqa: E731
    out = defaultdict(set)
    for total, tried, _ in ways(form, statement, values, group):
        if total not in values:
            continue
        for alt in tried:
            lines = [(s, c) for s, term in alt for c in expand(term, values, width)]
            cells = [total] + [c for _, c in lines]
            unread = [c for c in cells if val(c) is None]
            if len(unread) != 1 or not all(isinstance(val(c), int) for c in cells if c != unread[0]):
                continue
            x = unread[0]
            rest = [(s, c) for s, c in lines if c != x]
            fixed = sum(s * val(c) for s, c in rest if s > 0)
            flex = [c for s, c in rest if s < 0 and val(c)]
            made = set()
            if any(val(c) < 0 for c in flex):                      # costs in parentheses, a plain one a credit
                made.add(fixed + sum(val(c) for c in flex))
            for flip in itertools.product((False, True), repeat=min(len(flex), MAX_FLEX)):
                flip += (False,) * (len(flex) - len(flip))
                made.add(fixed + sum(-abs(val(c)) if f else -val(c) for c, f in zip(flex, flip)))
            if x == total:
                out[x] |= made
            else:
                sign = next(s for s, c in lines if c == x)
                for m in made:
                    out[x] |= {val(total) - m} if sign > 0 else {val(total) - m, m - val(total)}
    return {c: w - {0} for c, w in out.items() if w - {0}}


def gap_kind(values: dict, total: str, alts: list, width: int) -> str:
    """What a failed sum's gap looks like, without its figure."""
    for alt in alts:
        lines = [(s, c) for s, term in alt for c in expand(term, values, width)]
        if not all(isinstance(values.get(c, 0), int) for _, c in lines):
            continue
        got = sum(s * values.get(c, 0) for s, c in lines)
        gap = values[total] - got
        if abs(gap) <= 2:
            return "off by rounding"
        for s, c in lines:
            v = values.get(c, 0)
            if v and abs(gap) == abs(2 * v):
                return f"line {c} the other way"
            if v and abs(gap) == abs(v):
                return f"line {c} counted once too often or too few"
        if values[total] and gap % 10 ** 3 == 0 and abs(gap) < abs(values[total]):
            return "a line missing (gap in round thousands)"
        return "unexplained"
    return "unexplained"


# --- the check on a set's truth -------------------------------------------------------

def truth_sides(truth: str) -> dict:
    """(sha256, statement, column) -> code -> (value, "true" | "read" | "missing"), each
    filing's side; "missing" is a line only the other filing's reading has, its value None."""
    checked = {}
    path = os.path.join(truth, "checked.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                c = json.loads(line)
                checked[c["statement"], c["code"], c["base"]["sha256"], c["base"]["column"]] = c
    sides = defaultdict(dict)
    with open(os.path.join(truth, "cells.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            k = checked.get((c["statement"], c["code"], c["base"]["sha256"], c["base"]["column"]))
            for side in ("base", "later"):
                s = c[side]
                v, how = s["value"], "read"
                if c["status"] == "agree":
                    how = "true"
                elif k and "truth" in k[side]:
                    v, how = k[side]["truth"], "true"
                elif c["status"] == "ambiguous":
                    v = None
                if v == "missing":
                    # the other filing prints this line and this reading lacks it: unread, not zero
                    v, how = None, "missing"
                code = s["code"] if "code" in s else c["code"]
                if code is None:            # a line the base's older form does not have
                    continue
                sides[s["sha256"], c["statement"], s["column"]][code] = (v, how)
    return sides


def no_room(rows: list[dict], code: str) -> bool:
    """The page has no place for a row of this code: the rows of the nearest codes under and
    over it follow one another on one page, with less than a code's height between them."""
    n = int(code)
    at = [(int(r["code"]), k) for k, r in enumerate(rows) if (r.get("code") or "").isdigit()]
    under, over = [a for a in at if a[0] < n], [a for a in at if a[0] > n]
    if not under or not over or any(a[0] == n for a in at):
        return False
    k, j = max(under)[1], min(over)[1]
    a, b = rows[k], rows[j]
    return (j == k + 1 and a["page"] == b["page"]
            and 0 <= b["code_box"][1] - a["code_box"][3] < min(a["code_box"][3] - a["code_box"][1],
                                                               b["code_box"][3] - b["code_box"][1]))


def sides_of(set_path: str, drafts: str, truth: str):
    """(sha256, form, statement, column, code -> (value, "true" | "read"), codes next to a row
    read without its code, the filing is consolidated, codes whose cell a sum had a say in)
    per filing's side."""
    from fs_truth import cells_of, load, one   # the reading, for lines with no truth cell
    with open(set_path, encoding="utf-8") as fh:
        files = {f["sha256"]: f for f in map(json.loads, fh)}
    sides = truth_sides(truth)
    forms, read = {}, {}
    for sha in {k[0] for k in sides}:
        st = load(drafts, files[sha])
        forms[sha] = form_of(st) or ("2026" if files[sha]["year"] >= 2026 else "2014")
        read[sha] = st
    for (sha, kind, col), vals in sorted(sides.items()):
        form = forms[sha]
        best, hit, rows = None, -1, []          # the printing of this statement the truth came from
        for s in (s for s in read[sha] if s["statement"] == kind):
            cells = cells_of(s)
            n = sum(one(cells.get(c, {}).get(col, [])) == v for c, (v, _) in vals.items())
            if n > hit:
                best, hit, rows = cells, n, s["rows"]
        # lines next to a row read without its code: a line the sum needs may be that row
        near = {r.get("code") for i, r in enumerate(rows) for r in rows[max(i - 1, 0):i + 2]
                if not rows[i].get("code") and rows[i].get("values")} - {None}
        dropped = {o for o, n in OLD_TO_NEW.get(kind, {}).items() if n is None} if form == "2014" else set()
        for code in {r["code"] for r in rows if r.get("code")}:   # every line printed, figure or not
            if code in vals:
                continue
            if code in dropped and code in best:
                # a line the 2026 form dropped has no truth cell: take it as this filing read it
                v = one(best[code].get(col, []))
                if v not in ("missing", "ambiguous"):
                    vals = {**vals, code: (v, "read")}
                    continue
            # a line printed whose figure in this column was not read: unknown, not zero
            vals = {**vals, code: (None, "read")}
        # a line only the other filing's reading has, where this page has no room for it: absent
        vals = {c: v for c, v in vals.items() if not (v[1] == "missing" and no_room(rows, c))}
        for code in {r["code"] for r in rows if r.get("code")}:
            # ... unless every row of that code was looked at there and has no ink
            if vals[code][0] is None and all(col in r.get("blank", ()) for r in rows if r.get("code") == code):
                vals = {**vals, code: (BLANK, "read")}
        summed = {r["code"] for r in rows if r.get("code")
                  for v in r["values"][col:col + 1] if v and v.get("look") == "sum"}
        yield sha, form, kind, col, vals, near, files[sha].get("scope") == "consolidated", summed


def main():
    tally = defaultdict(Counter)
    fails = []
    for sha, form, kind, col, vals, near, group, summed in sides_of(*sys.argv[1:4]):
        values = {c: v for c, (v, _) in vals.items()}
        for name, total, result, negs, alts in sums_of(form, kind, values, group):
            width = 3 if kind == "B01" else 2
            used = {c for alt in alts for _, t in alt for c in expand(t, values, width)} | {total}
            src = "true" if all(vals.get(c, (0, "true"))[1] == "true" for c in used) else "some read"
            if result == "fails" and used & near:
                src += ", a row without its code next to it"
            t = tally[form, kind]
            t[f"{result}, {src}"] += 1
            if "blanks" in negs:
                t["  of which unread lines taken as blank"] += 1
            if "blank cells" in negs:
                t["  of which cells seen blank taken as zero"] += 1
            if result == "holds" and used & summed:
                t["  of which a cell kept because this sum or another asks for it"] += 1
            if "none printed" in negs:
                t["  of which a zero total with none of its lines printed"] += 1
            if [n for n in negs if n not in ("blanks", "blank cells", "none printed")]:
                t["  of which deductions printed negative"] += 1
            if result == "fails":
                fails.append((form, kind, name, sha[:8], col, src, gap_kind(values, total, alts, width)))
    for (form, kind), t in sorted(tally.items()):
        print(f"{form} {kind}: " + ", ".join(f"{k} {v}" for k, v in sorted(t.items())))
    for f in fails:
        print("fails:", *f)


if __name__ == "__main__":
    main()
