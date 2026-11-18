"""Cross-period truth for a statements set: a figure read in two filings (profile 2).

A flow pair's later filing repeats the base period's B02 and B03 in its prior-year
column(s); a balance pair's later filing repeats the base's closing B01 in its opening
column. The two are different scans of one number, so where both readings are sure and
agree, the cell is truth; everything else goes to a person.

**Which column repeats which** is found from the figures, not from the headings: most
filings print the current period first, but not all, and headings are often half read.
For each pair of columns (base i, later j), the agreement is the number of codes where
both values are sure, equal and not zero (zeros agree by chance). Each later column takes
the base column it agrees with most, if at least MIN_AGREE codes and AGREE_SHARE of the
codes compared. A wrong match agrees on next to nothing, so it cannot make truth. When a
filing prints a statement twice, the two printings that agree best are used; a cell
one leaves unread is taken from the other printing when that one matched the same
column with no code disagreeing (see reprinted).

Per cell (code, base column, later column):
- **agree:** both sure and equal: truth.
- **sign differs:** both sure, equal digits, opposite signs. Filings differ on printing
  costs in parentheses, so the digits are truth and the sign goes to a person.
- **differ:** both sure, unequal: a misread, or a restatement ("trình bày lại"); a person.
- **unsure:** either one left empty by the reader; a person.
- **one side:** the code read in only one filing; a person.
A code read twice in one statement with two values is **ambiguous**; a person.

**Across the form change:** 2025 filings use the 2014 form and most 2026 ones the 2026
form, which renumbers lines; the base's codes are then carried to the new form's
(tools/fs_codes.py) before anything is compared, and a line the new form dropped is
counted as "no place" and compared with nothing. A line only the new form has gets no
base code: its number may be a different line of the old form (2026's 421 is not 2014's).

**Doubted pages:** a matched statement is still asked of a person (check_i's where round)
when a page of either printing says, in its own print, that it is another statement: the
form label at its top names another ("Mẫu số B 02 - DN" on a page filed as B01), or
MIN_ODD or more of its codes are ones the statement does not use. The statement that page
names is asked too. The person's pages then decide; this only widens what is asked, so
it never makes truth agree with the reading. Its cells are not written here.

Writes the cells, with their values, to <out>/cells.jsonl. That holds figures, so it
lives in the gitignored data/ and is never committed. Prints counts only.

Usage: .venv/bin/python -I tools/fs_truth.py sets/statements-h.jsonl data/set-h/drafts data/truth-h
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from fs_codes import OLD_TO_NEW, form_of, old_of, to_new  # noqa: E402
from fs_rows import B02_ONLY, B03_ONLY, FORM_RE, TOP_SHARE, size_of, statements_of  # noqa: E402

REPEATED = {"flow": ("B02", "B03"), "balance": ("B01",)}
STATUSES = ("agree", "sign differs", "differ", "unsure", "one side", "ambiguous")
MIN_AGREE = 5
AGREE_SHARE = 0.3
MIN_ODD = 2         # codes a statement does not use, on one page, before the page is doubted


def cells_of(s: dict) -> dict[str, list]:
    """code -> per column the set of values read (None for unsure); codes once read."""
    by = defaultdict(lambda: defaultdict(list))
    for r in s["rows"]:
        if not r["code"]:
            continue
        for i, v in enumerate(r["values"]):
            if v:
                by[r["code"]][i].append(v["value"])
    return by


def carried(cells: dict, kind: str) -> tuple[dict, dict, int]:
    """A 2014 form's cells under the 2026 form's codes: (cells, new code -> old code,
    codes with no place in the new form)."""
    out, old, dropped = {}, {}, 0
    # a renumbered line first, so a stray code that is no line of the old form (read
    # where the new number lands) cannot take its place
    for code, cols in sorted(cells.items(), key=lambda kv: kv[0] not in OLD_TO_NEW[kind]):
        new = to_new(kind, code)
        if new is None or new in out:
            dropped += 1
            continue
        out[new], old[new] = cols, code
    return out, old, dropped


def one(vals: list):
    """A cell's value: the one read, None if unsure, "ambiguous" if two differ."""
    if not vals:
        return "missing"
    sure = {v for v in vals if v is not None}
    if len(sure) > 1:
        return "ambiguous"
    return next(iter(sure)) if sure else None


def match(b: dict, l: dict) -> list[tuple[int, int, int, int]]:
    """(base column, later column, codes agreed, codes compared) for each later column matched."""
    ib = {i for c in b.values() for i in c}
    il = {j for c in l.values() for j in c}
    score = {}
    for i in ib:
        for j in il:
            agree = seen = 0
            for code in b.keys() & l.keys():
                x, y = one(b[code].get(i, [])), one(l[code].get(j, []))
                if isinstance(x, int) and isinstance(y, int) and (x or y):
                    seen += 1
                    agree += x == y
            score[i, j] = (agree, seen)
    out, used = [], set()
    for (i, j), (agree, seen) in sorted(score.items(), key=lambda kv: -kv[1][0]):
        if i in used or any(j == o[1] for o in out):
            continue
        if agree >= MIN_AGREE and agree >= AGREE_SHARE * seen:
            out.append((i, j, agree, seen))
            used.add(i)
    return sorted(out, key=lambda o: o[1])


def best_of(sb: list[dict], sl: list[dict], kind: str, across: bool) -> tuple:
    """The two printings of a statement, one per filing, whose columns agree most:
    (matches, base cells, later cells, base printing, later printing, new -> old code,
    codes with no place), or None; and every pair of printings tried."""
    best, tried = None, []
    for x in (s for s in sb if s["statement"] == kind):
        for y in (s for s in sl if s["statement"] == kind):
            cb, cl, old, dropped = cells_of(x), cells_of(y), {}, 0
            if across:
                cb, old, dropped = carried(cb, kind)
            m = match(cb, cl)
            tried.append((m, cb, cl, x, y, old))
            if best is None or sum(o[2] for o in m) > sum(o[2] for o in best[0]):
                best = (m, cb, cl, x, y, old, dropped)
    return best, tried


def status(x, y) -> str:
    if x == "ambiguous" or y == "ambiguous":
        return "ambiguous"
    if x == "missing" or y == "missing":
        return "one side"
    if x is None or y is None:
        return "unsure"
    if x == y:
        return "agree"
    return "sign differs" if x == -y else "differ"


def load(drafts: str, f: dict, told: dict | None = None) -> list[dict]:
    """A file's statements; told (page -> statement) as in fs_rows.statements_of."""
    with open(os.path.join(drafts, f["sha256"][:16] + ".draft.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    pdf = os.path.join("cache", "pdfs", d["file"])
    return statements_of(d, pdf if os.path.exists(pdf) else None, told, os.path.join(drafts, "images"))


def forms_printed(drafts: str, f: dict) -> dict[int, str]:
    """page -> the statement its printed form label names, from the words at its top.
    Pages with no placed lines (a text layer) are not looked at; their codes still are."""
    with open(os.path.join(drafts, f["sha256"][:16] + ".draft.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    out = {}
    for p in d["pages"]:
        lines = [ln for ln in p.get("lines") or [] if ln.get("box")]
        if not lines:
            continue
        height = size_of(p, lines, os.path.join(drafts, "images"))[1]
        m = FORM_RE.search(" ".join(ln["text"] for ln in lines if ln["box"][1] < TOP_SHARE * height))
        if m:
            out[p["page"]] = "B0" + m.group(1)
    return out


def named_by(code: str) -> str | None:
    """The one statement a code belongs to, or None when B02 and B03 share it."""
    n = int(code)
    if n >= 100:
        return "B01"
    return "B02" if code.zfill(2) in B02_ONLY else "B03" if code.zfill(2) in B03_ONLY else None


def holds(kind: str, code: str) -> bool:
    """Whether a statement uses a code (of either form)."""
    if int(code) >= 100:
        return kind == "B01"
    return kind != "B01" and named_by(code) in (None, kind)


def doubts(s: dict, forms: dict[int, str]) -> dict[int, set]:
    """The pages of a printing whose own print names another statement: page -> those named."""
    out = {}
    for pg in s["pages"]:
        named = {forms[pg]} - {s["statement"]} if pg in forms else set()
        odd = [r["code"] for r in s["rows"] if r["page"] == pg and (r.get("code") or "").isdigit()
               and not holds(s["statement"], r["code"])]
        if len(odd) >= MIN_ODD:                 # under B01, codes B02 and B03 share name either
            named |= {named_by(c) for c in odd} - {None} or {"B02", "B03"}
        if named:
            out[pg] = named
    return out


def doubted(drafts: str, base: dict, later: dict, bests: dict, pair: str) -> dict[str, str]:
    """The statements of a pair to ask of a person though matched: kind -> why."""
    forms, out = {}, {}
    for kind, best in bests.items():
        if not best or not best[0]:
            continue
        for side, f, s in (("base", base, best[3]), ("later", later, best[4])):
            if f["sha256"] not in forms:
                forms[f["sha256"]] = forms_printed(drafts, f)
            for pg, named in sorted(doubts(s, forms[f["sha256"]]).items()):
                for k in sorted({kind} | named):
                    if k in REPEATED[pair]:
                        out.setdefault(k, f"{kind} {side} page {pg} reads as {'/'.join(sorted(named))}")
    return out


def x_at(s: dict, page: int | None, col: int) -> float | None:
    """The right edge of a statement's col-th column on a page, or None."""
    xs = next((c["x"] for c in s["columns"] if c["page"] == page), None)
    return xs[col] if xs and col < len(xs) else None


def reprinted(tried, fixed, col, code, side, v, s):
    """A cell the chosen printing left unread, from another printing of the statement in
    the same filing: one whose column matched the other filing's same column (fixed, col)
    with no code disagreeing. Every such printing must read the one same sure value;
    else the cell stays as it was. Returns (value, printing)."""
    got = []
    for m, cb, cl, x, y, _ in tried:
        own, other = (x, y) if side == 0 else (y, x)
        if other is not fixed or own is s:
            continue
        for i, j, agree, seen in m:
            if (j, i)[side] == col and agree == seen:
                got.append((one((cb if side == 0 else cl).get(code, {}).get((i, j)[side], [])), own))
    vals = {g for g, _ in got}
    if len(vals) == 1 and isinstance(next(iter(vals)), int):
        return got[0]
    return v, s


def main():
    set_path, drafts, out = sys.argv[1:4]
    with open(set_path, encoding="utf-8") as fh:
        files = [json.loads(line) for line in fh]
    pairs = defaultdict(dict)
    for f in files:
        pairs[f["symbol"]][f["role"]] = f
    os.makedirs(out, exist_ok=True)
    tally = defaultdict(Counter)
    unmatched = []
    forms = Counter()
    with open(os.path.join(out, "cells.jsonl"), "w", encoding="utf-8") as fh:
        for sym, p in sorted(pairs.items()):
            base, later = p["base"], p["later"]
            sb, sl = load(drafts, base), load(drafts, later)
            across = form_of(sb) == "2014" and form_of(sl) == "2026"
            forms[form_of(sb), form_of(sl)] += 1
            bests = {kind: best_of(sb, sl, kind, across) for kind in REPEATED[base["pair"]]}
            asked = doubted(drafts, base, later, {k: b for k, (b, _) in bests.items()}, base["pair"])
            for kind, (best, tried) in bests.items():
                t = tally[base["pair"], kind]
                t["statements"] += 1
                if kind in asked and best and best[0]:
                    t["doubted"] += 1
                    unmatched.append((base["sha256"][:8], later["sha256"][:8], kind, "doubted: " + asked[kind]))
                    continue
                if not best or not best[0]:
                    t["not matched"] += 1
                    unmatched.append((base["sha256"][:8], later["sha256"][:8], kind,
                                      "not found" if not best else "no column agrees"))
                    continue
                m, cb, cl, x, y, old, dropped = best
                t["matched"] += 1
                t["no place"] += dropped * len(m)
                t[f"{len(m)} column(s)"] += 1
                for i, j, _, _ in m:
                    for code in sorted(cb.keys() | cl.keys(), key=int):
                        bv, lv = one(cb.get(code, {}).get(i, [])), one(cl.get(code, {}).get(j, []))
                        bx, ly = x, y
                        if not isinstance(bv, int):     # left unread here, read in a reprint?
                            bv, bx = reprinted(tried, y, j, code, 0, bv, x)
                        if not isinstance(lv, int):
                            lv, ly = reprinted(tried, x, i, code, 1, lv, y)
                        st = status(bv, lv)
                        t[st] += 1
                        bcode = old.get(code, old_of(kind, code)) if across else code
                        pb = next((r["page"] for r in bx["rows"] if r["code"] == bcode), None)
                        pl = next((r["page"] for r in ly["rows"] if r["code"] == code), None)
                        fh.write(json.dumps({
                            "pair": base["pair"], "statement": kind, "code": code,
                            # a new-form line with no old one has no base code; its own
                            # number may be another line's in the old form. x: where the
                            # column stands on that page (score_fs looks the candidate up there)
                            "base": {"sha256": base["sha256"], "column": i, "code": bcode,
                                     "page": pb, "x": x_at(bx, pb, i), "value": bv},
                            "later": {"sha256": later["sha256"], "column": j,
                                      "page": pl, "x": x_at(ly, pl, j), "value": lv},
                            "status": st}, ensure_ascii=False) + "\n")
    for (pair, kind), t in sorted(tally.items()):
        cells = sum(t[k] for k in STATUSES)
        print(f"{pair} {kind}: {t['matched']} of {t['statements']} matched, {t['doubted']} doubted "
              f"({', '.join(f'{k} {v}' for k, v in sorted(t.items()) if 'column' in k)}); "
              f"cells {cells}: " + ", ".join(f"{k} {t[k]}" for k in
                                             STATUSES)
              + f"; no place in the new form {t['no place']}")
    print("forms (base, later):", {f"{a}/{b}": n for (a, b), n in forms.items()})
    for u in unmatched:
        print("not matched:", *u)


if __name__ == "__main__":
    main()
