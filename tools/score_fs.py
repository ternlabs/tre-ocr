"""Score a statements reading against a set's truth (profile 2).

The scored unit is a **side**: one figure as one filing prints it, at a statement's code
and the column holding the base period. A pair's two filings are two scans, so a cell
of truth gives up to two sides, each read by the candidate on its own.

**A side's truth**, in this order:
- typed by the person for that side (an unreadable answer gives none);
- for a sign asked about, the reading's digits with the sign the person saw printed;
- where both filings' readings agree, the agreed value. When the person checked the
  other side and typed something else, the two readings share a misread and this side
  has no truth.
Anything else (the side of an "unsure" or "one side" cell that was read but not typed)
has no truth and is counted as such: taking a reading as truth only where it matches
the person's typing of the other filing would drop the misread ones, and so flatter
"wrong when sure".

**The candidate's answer** for a side is what its reading holds at that filing,
statement, code and column: for a cell of a matched statement, looked up again in the
reading being scored, on the page and at the column's place in the row that fs_truth
recorded (the value fs_truth stored is the reading the truth was built from, which a
later candidate may change); for a statement the person had to point to, the
candidate's reading on that page, its column taken by place (the column's right edge
on the page, within a third of the gap between columns). It is "sure" when it is a
figure, and also when the reading holds two different figures for the code: one of them
is wrong and nothing marks it.

Scores, never blended with anything else (numbers only; this profile has no text):
- **cell recall:** sides whose truth is a figure, answered with that figure, sign
  included;
- **coverage:** of the same, the share answered sure;
- **wrong when sure:** of sure answers on any side with truth, the share not equal to
  it. A figure given where the person saw none printed is wrong;
- **statements found:** each filing's printing of each statement the pair repeats, not
  counting those the person said are not printed there. Found when the reading matched
  it across the pair, or when the reading's statement of that kind covers every page
  the person gave.
- **figures under the wrong statement:** of the sure figures the reading holds on pages
  the person pointed to, the share it files under another statement than the one the
  person gave for that page. Right figures under the wrong heading mislead a reader as
  a misread does, but no cell's lookup sees them; so they are counted apart, not in
  wrong when sure. A floor: only pointed-to pages are known.
- **figures under codes their statement does not use:** of all sure figures the reading
  holds, the share in rows whose code cannot be its statement's: under 100 in a balance
  sheet (B01), 100 or more in an income or cash flow statement (B02, B03). It needs no
  person, so it sees the pages the one above cannot: a statement swallowed whole by
  another, alike in both filings or never asked. A floor too: B02 and B03 share codes.
Each with a Wilson 95% interval on cells, and again with a percentile interval from
resampling whole statements (each filing's printing of one statement), since the cells
of one page are not independent. Prints counts and shares only, no figure.

**Corrections** (--corrections <file>): a person's where answer found wrong after the
score, e.g. pages given for a statement that run into the next one. Each line is
{"id": <where item>, "value": <pages, as answered>, "was": <the answer replaced>,
"why": ...}; the answers file is left as it was. A cell asked about on a page the
correction takes away is not scored: it was read from another statement. Every
correction is printed, so a corrected score is always reported beside the original.

Usage, for a set X (all four are required):
  .venv/bin/python -I tools/score_fs.py --set sets/statements-X.jsonl \
      --drafts data/set-X/drafts --truth data/truth-X --work data/check-X
"""
import argparse
import math
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from records import read_jsonl  # noqa: E402

MARKS = {"cell recall": (0.98, ">="), "wrong when sure": (0.005, "<="), "coverage": (0.85, ">="),
         "statements found": (0.95, ">="), "figures under the wrong statement": (0.005, "<="),
         "figures under codes their statement does not use": (0.005, "<=")}
Z = 1.96
RESAMPLE = 2000
SEED = 13
SIDES = ("base", "later")


# ---------------------------------------------------------------- truth

def signed(read, printed: str):
    """The digits read with the sign the person saw printed; None if not that."""
    if printed == "dash":
        return 0
    if printed == "nothing":
        return "missing"
    if not isinstance(read, int):
        return None
    return {"plain": abs(read), "parentheses": -abs(read), "minus": -abs(read)}.get(printed)


def truth_of(c: dict, side: str, rec: dict | None) -> tuple:
    """A side's truth and where it came from: (value or None, origin)."""
    mine = (rec or {}).get(side, {})
    other = (rec or {}).get(SIDES[1 - SIDES.index(side)], {})
    if "typed" in mine:
        return (mine["typed"], "typed") if mine["typed"] is not None else (None, "unreadable")
    if "printed" in mine:
        v = signed(c[side]["value"], mine["printed"])
        return (v, "sign") if v is not None else (None, "unreadable")
    if c["status"] == "agree":
        if "typed" in other and other["typed"] != c[side]["value"]:
            return None, "check differs"
        return c[side]["value"], "agreed"
    return None, "not typed"


# ---------------------------------------------------------------- the candidate

def read_at(statements: list[dict], kind: str, code: str, page: int, x: float):
    """The candidate's value at a statement's code, on a page, in the column at x:
    a figure, None if unsure, "ambiguous", or "missing"."""
    vals = []
    for s in statements:
        if s["statement"] != kind or page not in s["pages"]:
            continue
        xs = next((c["x"] for c in s["columns"] if c["page"] == page), None)
        if not xs:
            continue
        near = min((b - a for a, b in zip(xs, xs[1:])), default=900) / 3
        k = min(range(len(xs)), key=lambda k: abs(xs[k] - x))
        if abs(xs[k] - x) > near:
            continue
        for r in s["rows"]:
            if r["code"] == code and r["page"] == page and k < len(r["values"]) and r["values"][k]:
                vals.append(r["values"][k]["value"])
    if not vals:
        return "missing"
    sure = {v for v in vals if v is not None}
    return "ambiguous" if len(sure) > 1 else next(iter(sure)) if sure else None


def read_in_row(statements: list[dict], kind: str, code: str, page: int | None, col: int):
    """The candidate's value at a statement's code, at a column's place in the row (as
    fs_truth numbers them), on a page (any, when none was recorded): a figure, None if
    unsure, "ambiguous", or "missing"."""
    vals = [r["values"][col]["value"] for s in statements if s["statement"] == kind
            for r in s["rows"] if r["code"] == code and (page is None or r["page"] == page)
            and col < len(r["values"]) and r["values"][col]]
    if not vals:
        return "missing"
    sure = {v for v in vals if v is not None}
    return "ambiguous" if len(sure) > 1 else next(iter(sure)) if sure else None


def column_x(statements: list[dict], kind: str, code: str, page: int, col: int) -> float | None:
    """Where a matched cell's column stands on its page in the reading that made the truth
    (the right edge of the col-th column of the statement holding the row), or None."""
    for s in statements:
        if s["statement"] == kind and any(r["code"] == code and r["page"] == page for r in s["rows"]):
            xs = next((c["x"] for c in s["columns"] if c["page"] == page), None)
            return xs[col] if xs and col < len(xs) else None
    return None


def answer_of(c: dict, side: str, statements_of, truth_reading=None) -> object:
    """What the candidate answers for a side; statements_of(sha256) gives its reading.

    A matched cell is looked up at its column's place (x), as the person saw it cut out:
    a column's number is the reading's own, and the candidate's can differ on a page (a
    page that carries the columns before, a reprint with more). The place is the cell's
    own "x" (fs_truth writes it), else found in truth_reading(sha256), the reading that
    made the truth; only where neither has it is the column's number used."""
    s = c[side]
    if not c.get("told"):
        code = s.get("code", c["code"])
        x = s.get("x")
        if x is None and truth_reading and s.get("page") is not None:
            x = column_x(truth_reading(s["sha256"]), c["statement"], code, s["page"], s["column"])
        if x is not None:
            return read_at(statements_of(s["sha256"]), c["statement"], code, s["page"], x)
        return read_in_row(statements_of(s["sha256"]), c["statement"], code, s.get("page"), s["column"])
    if s.get("page") is None or not s.get("band"):
        return "missing"
    code = s.get("code", c["code"])
    return read_at(statements_of(s["sha256"]), c["statement"], code, s["page"], s["band"][1])


def judged(truth, got) -> dict:
    """One side: is it a figure of truth, answered sure, right, wrong when sure."""
    sure = isinstance(got, int) or got == "ambiguous"
    figure = isinstance(truth, int)
    return {"figure": figure, "sure": sure, "right": figure and got == truth,
            "wrong": sure and got != truth,
            "sign only": sure and figure and isinstance(got, int) and got == -truth != 0}


# ---------------------------------------------------------------- scores

def wilson(k: int, n: int) -> tuple:
    if not n:
        return None, None
    p = k / n
    d = 1 + Z * Z / n
    mid = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def shares(sides: list[dict]) -> dict:
    """name -> (k, n) for the three cell scores."""
    fig = [s for s in sides if s["figure"]]
    sure = [s for s in sides if s["sure"]]
    return {"cell recall": (sum(s["right"] for s in fig), len(fig)),
            "coverage": (sum(s["sure"] for s in fig), len(fig)),
            "wrong when sure": (sum(s["wrong"] for s in sure), len(sure))}


def by_statement(sides: list[dict], seed: int = SEED, n: int = RESAMPLE) -> dict:
    """name -> (low, high): percentile 95% intervals, whole statements resampled."""
    groups = defaultdict(list)
    for s in sides:
        groups[s["statement key"]].append(s)
    keys = sorted(groups)
    if not keys:
        return {}
    rnd, got = random.Random(seed), defaultdict(list)
    for _ in range(n):
        pick = [s for k in rnd.choices(keys, k=len(keys)) for s in groups[k]]
        for name, (k_, n_) in shares(pick).items():
            if n_:
                got[name].append(k_ / n_)
    out = {}
    for name, vals in got.items():
        vals.sort()
        out[name] = (vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))])
    return out


def resampled(groups: dict, seed: int = SEED, n: int = RESAMPLE) -> tuple:
    """Percentile 95% interval of a share, whole groups resampled: groups is key -> (k, n)."""
    keys = sorted(groups, key=str)
    rnd, vals = random.Random(seed), []
    for _ in range(n if keys else 0):
        pick = rnd.choices(keys, k=len(keys))
        k_, n_ = sum(groups[x][0] for x in pick), sum(groups[x][1] for x in pick)
        if n_:
            vals.append(k_ / n_)
    if not vals:
        return None, None
    vals.sort()
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def clustered(groups: dict) -> tuple:
    """A share's 95% interval when its figures come in groups (a statement printing's, filed
    or misfiled a page at a time): the wider of the groups resampled and Wilson on the
    figures, which alone takes each figure as a draw of its own; resampled alone, no error
    seen gives [0, 0]."""
    k, n = sum(g[0] for g in groups.values()), sum(g[1] for g in groups.values())
    w, b = wilson(k, n), resampled(groups)
    if w[0] is None or b[0] is None:
        return w
    return min(w[0], b[0]), max(w[1], b[1])


def verdict(name: str, k: int, n: int, ci: tuple) -> str:
    mark, way = MARKS[name]
    if not n:
        return "no cells"
    p = k / n
    meets = p >= mark if way == ">=" else p <= mark
    clear = ci[0] is not None and (ci[0] >= mark if way == ">=" else ci[1] <= mark)
    return ("meets the mark" if meets else "misses the mark") + (", interval clear of it" if clear else "")


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.2f}%"


# ---------------------------------------------------------------- statements found

def statements_found(files: list[dict], where_items: list[dict], where_answers: dict,
                     statements_of, repeated: dict) -> Counter:
    """Each filing's printing of each statement the pair repeats: found, partly, missed,
    or not printed (as the person said)."""
    from check_i import pages_of, pairs_of
    told = {(it["statement"], it["base"], it["side"]): it for it in where_items}
    out = Counter()
    for base, later in pairs_of(files):
        for kind in repeated[base["pair"]]:
            for side, f in zip(SIDES, (base, later)):
                it = told.get((kind, base["sha256"], side))
                if it is None:
                    out["found (matched)"] += 1
                    continue
                x = where_answers[it["id"]]
                if x["answer"] != "pages":
                    out["not printed"] += 1
                    continue
                want = set(pages_of(x["value"], it["pages"]))
                have = {p for s in statements_of(f["sha256"]) if s["statement"] == kind for p in s["pages"]}
                out["found" if want <= have else "partly" if want & have else "missed"] += 1
    return out


def misfiled(where_items: list[dict], where_answers: dict, statements_of) -> tuple[int, int, int, dict]:
    """Sure figures the reading files under another statement than the one a person said
    the page holds: (under the wrong statement, sure figures on pages pointed to, pages
    where any is, (sha256, statement pointed to) -> (wrong, sure) for clustered). Cells are looked up by statement, so these are scored nowhere else: the
    pointed-to statement reads "missing" and the figures under the wrong one meet no truth.
    A floor: a page both filings misfile alike is matched across the pair, never asked."""
    from check_i import pages_of
    kinds = defaultdict(lambda: defaultdict(set))       # sha256 -> page -> statements there
    for it in where_items:
        x = where_answers.get(it["id"])
        if x and x["answer"] == "pages":
            for p in pages_of(x["value"], it["pages"]) or []:
                kinds[it["sha256"]][p].add(it["statement"])
    wrong = sure = 0
    pages, groups = set(), defaultdict(lambda: [0, 0])
    for sha, at in kinds.items():
        for kind in {k for ks in at.values() for k in ks}:
            groups[sha, kind]                       # a printing with no sure figure on its pages
        for s in statements_of(sha):
            for r in s["rows"]:
                if r["page"] not in at:
                    continue
                n = sum(isinstance(v, dict) and isinstance(v.get("value"), int) for v in r["values"])
                sure += n
                bad = n if s["statement"] not in at[r["page"]] else 0
                g = groups[sha, min(at[r["page"]])]  # the printing pointed to there
                g[0] += bad
                g[1] += n
                if bad:
                    wrong += n
                    pages.add((sha, r["page"]))
    return wrong, sure, len(pages), {k: tuple(v) for k, v in groups.items()}


def out_of_range(files: list[dict], statements_of) -> tuple[int, int, int, dict]:
    """Sure figures in rows whose code cannot be their statement's (B01 codes run from
    100, B02's and B03's below it): (out of range, all sure figures, pages where any is,
    (sha256, statement) -> (out of range, sure) for clustered)."""
    import re
    wrong = sure = 0
    pages, groups = set(), defaultdict(lambda: [0, 0])
    for f in files:
        for s in statements_of(f["sha256"]):
            g = groups[f["sha256"], s["statement"]]
            for r in s["rows"]:
                n = sum(isinstance(v, dict) and isinstance(v.get("value"), int) for v in r["values"])
                sure += n
                g[1] += n
                m = re.match(r"\d+", str(r.get("code") or ""))
                if n and m and (int(m.group()) < 100) == (s["statement"] == "B01"):
                    wrong += n
                    g[0] += n
                    pages.add((f["sha256"], r["page"]))
    return wrong, sure, len(pages), {k: tuple(v) for k, v in groups.items()}


# ---------------------------------------------------------------- corrections

def corrected(where_items: list[dict], where_answers: dict, path: str) -> dict:
    """Applies the corrections to where_answers in place and prints each; returns
    (filing's sha256, statement) -> the pages it now holds."""
    from check_i import pages_of
    items = {it["id"]: it for it in where_items}
    kept = {}
    for fix in read_jsonl(path):
        it, x = items[fix["id"]], where_answers[fix["id"]]
        if x["value"] != fix["was"]:
            sys.exit(f"correction {fix['id']}: the answer is no longer what it replaces")
        where_answers[fix["id"]] = {**x, "value": fix["value"]}
        kept[it["sha256"], it["statement"]] = set(pages_of(fix["value"], it["pages"]))
        print(f"corrected where answer {fix['id']} ({it['statement']}, {it['side']} filing):"
              f" pages {fix['was']} -> {fix['value']}: {fix['why']}")
    return kept


# ---------------------------------------------------------------- main

def cells_with_truth(truth: str) -> tuple[list[dict], dict]:
    """The cells as check_i numbered them (matched, then told) and the checked records by
    that number."""
    cells = [{**c, "told": False} for c in read_jsonl(os.path.join(truth, "cells.jsonl"))]
    told = os.path.join(truth, "cells-told.jsonl")
    cells += read_jsonl(told) if os.path.exists(told) else []
    checked = os.path.join(truth, "checked.jsonl")
    if not os.path.exists(checked):
        sys.exit("no checked.jsonl: finish the type round first (tools/check_i.py --finish)")
    recs = {r["cell"]: r for r in read_jsonl(checked)}
    return cells, recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", required=True)
    ap.add_argument("--drafts", required=True, help="the candidate's own drafts")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--work", required=True, help="the check rounds' items and answers (never "
                    "defaulted: another set's would score every statement as found)")
    ap.add_argument("--truth-drafts", help="the drafts the truth was made from, where its cells' "
                    "columns stand (default: the work's drafts/)")
    ap.add_argument("--corrections", help="where answers corrected after the score (see the docstring)")
    a = ap.parse_args()
    from fs_truth import REPEATED, load
    files = read_jsonl(a.set)
    by = {f["sha256"]: f for f in files}
    cache = {}

    def statements_of(sha):
        if sha not in cache:
            cache[sha] = load(a.drafts, by[sha])
        return cache[sha]

    made = a.truth_drafts or os.path.join(a.work, "drafts")
    made_cache = {}

    def truth_reading(sha):                 # for cells written before fs_truth kept their x
        if sha not in made_cache:
            made_cache[sha] = load(made, by[sha]) if os.path.isdir(made) else []
        return made_cache[sha]

    where_items = read_jsonl(os.path.join(a.work, "items-where.jsonl"))
    where_answers = {x["id"]: x for x in read_jsonl(os.path.join(a.work, "answers-where.jsonl"))}
    kept = corrected(where_items, where_answers, a.corrections) if a.corrections else {}

    cells, recs = cells_with_truth(a.truth)
    sides, origin = [], Counter()
    for n, c in enumerate(cells):
        rec = recs.get(n)
        for side in SIDES:
            t, came = truth_of(c, side, rec)
            if c[side].get("sha256") is None:
                continue
            pages = kept.get((c[side]["sha256"], c["statement"]))
            if c["told"] and pages is not None and c[side].get("page") not in pages:
                origin["told, on a page a correction took away (not scored)"] += 1
                continue
            origin[("told " if c["told"] else "") + came] += 1
            if t is None:
                continue
            j = judged(t, answer_of(c, side, statements_of, truth_reading))
            j.update({"statement key": (c[side]["sha256"], c["statement"]), "told": c["told"],
                      "kind": c["statement"], "origin": came})
            sides.append(j)

    print("sides by where their truth came from:")
    for k, v in sorted(origin.items()):
        print(f"  {k}: {v}")
    checks = [(r, s) for r in recs.values() if r["status"] == "agree" for s in SIDES if "typed" in r[s]]
    same = sum(r[s]["typed"] == r[s]["value"] for r, s in checks)
    lo, hi = wilson(len(checks) - same, len(checks))
    print(f"agreed sides checked by the person: {same} of {len(checks)} as agreed "
          f"(shared misreads at most {pct(hi)})")

    statements = len({s["statement key"] for s in sides})
    print(f"\nscored: {len(sides)} sides with truth, in {statements} printings of statements")
    boot = by_statement(sides)
    for name, (k, n) in shares(sides).items():
        ci = wilson(k, n)
        b = boot.get(name, (None, None))
        print(f"  {name}: {k} of {n} = {pct(k / n if n else None)}  cells [{pct(ci[0])}, {pct(ci[1])}]"
              f"  statements [{pct(b[0])}, {pct(b[1])}]  {verdict(name, k, n, ci)}")
    wrong = [s for s in sides if s["wrong"]]
    print(f"  of the wrong when sure: sign only {sum(s['sign only'] for s in wrong)}, "
          f"figure where none printed {sum(not s['figure'] for s in wrong)}")
    for label, keep in (("matched statements", lambda s: not s["told"]),
                        ("statements pointed to", lambda s: s["told"]),
                        *((k, (lambda k: lambda s: s["kind"] == k)(k)) for k in ("B01", "B02", "B03"))):
        part = shares([s for s in sides if keep(s)])
        print(f"  {label}: " + ", ".join(f"{name} {k} of {n}" for name, (k, n) in part.items()))

    found = statements_found(files, where_items, where_answers, statements_of, REPEATED)
    k = found["found (matched)"] + found["found"]
    n = sum(v for key, v in found.items() if key != "not printed")
    ci = wilson(k, n)
    print(f"\nstatements found: {k} of {n} = {pct(k / n if n else None)}  [{pct(ci[0])}, {pct(ci[1])}]"
          f"  {verdict('statements found', k, n, ci)}")
    print("  " + ", ".join(f"{key} {v}" for key, v in sorted(found.items())))

    k, n, pages, groups = misfiled(where_items, where_answers, statements_of)
    ci = clustered(groups)
    print(f"\nfigures under the wrong statement: {k} of {n} sure figures on pages pointed to"
          f" = {pct(k / n if n else None)}  [{pct(ci[0])}, {pct(ci[1])}], on {pages} pages"
          f"  {verdict('figures under the wrong statement', k, n, ci)}"
          f"\n  a floor: pages matched across the pair are not asked; interval by statement printing")

    k, n, pages, groups = out_of_range(files, statements_of)
    ci = clustered(groups)
    print(f"\nfigures under codes their statement does not use: {k} of {n} sure figures"
          f" = {pct(k / n if n else None)}  [{pct(ci[0])}, {pct(ci[1])}], on {pages} pages"
          f"  {verdict('figures under codes their statement does not use', k, n, ci)}"
          f"\n  a floor: B02 and B03 share codes; interval by statement printing")


if __name__ == "__main__":
    main()
