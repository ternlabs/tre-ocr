"""A statements set's truth check: what the two filings' readings do not settle, typed blind (profile 2).
First made for set I.

Fixed before the draw: truth is the cross-period readings where they agree
(tools/fs_truth.py); a person types blind every disagreement and every cell the later
filing does not repeat, and checks a random CHECK agreed cells against the scan. A
statement whose two filings did not match is typed as well, so recall is over every
statement of the set, not only those the reading matched.

Three rounds, each served on 127.0.0.1 under a random key, as check_h's page; the person
never sees a reading.

  where    per statement not matched (or doubted, see fs_truth), per filing: on which
           pages of the PDF it is printed, or that it is not. Asked of every such filing, read or not, so the
           truth's pages are the person's, never the reader's.
  columns  per statement printed there: its first page with the figure columns
           numbered; which hold the base period (the quarter or half year alone, and
           the year to date).
  type     the cells:
           - sign: every "sign differs" cell, both sides;
           - type: both sides of every "differ" and "ambiguous" cell, the empty side of
             every "unsure" one, the read side of every "one side" one;
           - check: CHECK "agree" cells drawn with a fixed seed, one side each;
           - and, for the statements not matched, the same by the columns the person
             gave: where both filings print it, its cells compared as fs_truth does;
             where one does, every coded row of that one, typed.

Items are made once per round and kept, under random ids, in the --work folder
(gitignored); answers beside them. --finish (after the type round) writes checked.jsonl
in the --truth folder, each side's value as the person gave it, and prints counts only.

The set, its drafts, its truth and its work folder are always named, never defaulted: a
run that forgot one would write into another set's truth. For a set X:

  --set sets/statements-X.jsonl --drafts data/set-X/drafts --truth data/truth-X --work data/check-X

Usage:
  .venv/bin/python -I tools/check_i.py --round where --open
  .venv-paddle/bin/python -I tools/check_i.py --overlay
  .venv/bin/python -I tools/check_i.py --round columns --open
  .venv/bin/python -I tools/check_i.py --round type --open
  .venv/bin/python -I tools/check_i.py --finish
"""
import argparse
import json
import os
import random
import re
import secrets
import subprocess
import sys
import threading
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from check_h import ANSWERS, PAD, Pages, crop, given, printed_sign  # noqa: E402
from fs_codes import form_of, old_of  # noqa: E402
from fs_rows import value_of  # noqa: E402
from fs_truth import REPEATED, best_of, carried, doubted, load, one, status  # noqa: E402
from records import read_jsonl, write_jsonl  # noqa: E402

CHECK = 100
SEED = 12
WORK = None                                     # the --work folder, set in main()
NAMES = {"B01": "balance sheet", "B02": "income statement", "B03": "cash flow statement"}
FIELDS = ("period", "to date")      # the base period alone; the year to date up to its end


def pairs_of(files: list[dict]) -> list[tuple[dict, dict]]:
    by = defaultdict(dict)
    for f in files:
        by[f["symbol"]][f["role"]] = f
    return [(p["base"], p["later"]) for _, p in sorted(by.items())]


def unmatched(a, files) -> list[dict]:
    """The statements fs_truth could not match across a pair's two filings, or matched
    with a page whose own print names another statement (fs_truth.doubted)."""
    out = []
    for base, later in pairs_of(files):
        sb, sl = load(a.drafts, base), load(a.drafts, later)
        across = form_of(sb) == "2014" and form_of(sl) == "2026"
        bests = {kind: best_of(sb, sl, kind, across)[0] for kind in REPEATED[base["pair"]]}
        asked = doubted(a.drafts, base, later, bests, base["pair"])
        for kind, best in bests.items():
            if not best or not best[0] or kind in asked:
                out.append({"statement": kind, "base": base["sha256"], "later": later["sha256"]})
    return out


def period(f: dict) -> str:
    """The base period, as the person is asked about it."""
    if f["pair"] == "balance":                  # no date: a financial year need not end in December
        if f["period"] == "year":
            return f"the balance at the end of financial year {f['year']}"
        return f"the balance at the end of quarter {f['n']} of {f['year']}"
    if f["period"] == "half":
        return f"the first half of {f['year']}"
    return f"quarter {f['n']} of {f['year']}"


def pages_of(text: str, n: int) -> list[int] | None:
    """'7', '7-8' or '7, 9' as pages of a file of n pages; None if not that."""
    got = []
    for part in filter(None, re.split(r"\s*,\s*", text.strip())):
        m = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
        if not m:
            return None
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        if not 1 <= lo <= hi <= n:
            return None
        got += range(lo, hi + 1)
    return sorted(set(got)) or None


def columns_given(text: str, k: int) -> dict | None:
    """'2|4' (the base period's column, the year to date's; either may be empty) as
    {field: column index}; None if not that or both empty."""
    parts = text.split("|")
    if len(parts) != 2:
        return None
    out = {}
    for field, p in zip(FIELDS, parts):
        if p.strip():
            if not p.strip().isdigit() or not 1 <= int(p) <= k:
                return None
            out[field] = int(p) - 1
    return out or None


# ---------------------------------------------------------------- rounds

def path(a, round_, what):
    return os.path.join(WORK, f"{what}-{round_}.jsonl")


def answered(a, round_) -> dict:
    p = path(a, round_, "answers")
    return {x["id"]: x for x in (read_jsonl(p) if os.path.exists(p) else [])}


def all_answered(a, round_) -> tuple[list[dict], dict]:
    items, answers = read_jsonl(path(a, round_, "items")), answered(a, round_)
    left = sum(it["id"] not in answers for it in items)
    if left:
        sys.exit(f"round {round_}: {left} of {len(items)} items not answered")
    return items, answers


def build_where(a, files) -> list[dict]:
    by = {f["sha256"]: f for f in files}
    items = []
    for u in unmatched(a, files):
        for side in ("base", "later"):
            f = by[u[side]]
            items.append({"kind": "where", "statement": u["statement"], "side": side,
                          "sha256": f["sha256"], "base": u["base"], "later": u["later"],
                          "pages": f["pages"], "ask": period(by[u["base"]])})
    return items


def told_statement(a, f: dict, kind: str, pages: list[int]) -> dict | None:
    """The statement on the pages the person gave, read as told."""
    ss = load(a.drafts, f, {p: kind for p in pages})
    return next((s for s in ss if s["statement"] == kind and s["pages"][0] in pages), None)


def told_pages(a) -> dict:
    """sha256 -> {page: statement}, as the where round's answers give them."""
    p = path(a, "where", "items")
    if not os.path.exists(p):
        return {}
    answers, out = answered(a, "where"), defaultdict(dict)
    for it in read_jsonl(p):
        x = answers.get(it["id"])
        if x and x["answer"] == "pages":
            out[it["sha256"]].update({q: it["statement"] for q in pages_of(x["value"], it["pages"])})
    return dict(out)


class Told(Pages):
    """A filing's statements read with the pages the person gave."""

    def __init__(self, a, files: dict, told: dict):
        super().__init__(a, files)
        self.told = told

    def statements(self, sha: str) -> list[dict]:
        if sha not in self.ss:
            self.ss[sha] = load(self.a.drafts, self.files[sha], self.told.get(sha))
        return self.ss[sha]


def build_columns(a, files) -> list[dict]:
    by = {f["sha256"]: f for f in files}
    where, answers = all_answered(a, "where")
    items = []
    for it in where:
        x = answers[it["id"]]
        if x["answer"] != "pages":
            continue
        pages = pages_of(x["value"], it["pages"])
        s = told_statement(a, by[it["sha256"]], it["statement"], pages)
        if s is None:
            items.append({**it, "kind": "unread", "told": pages})
            continue
        items.append({**it, "kind": "columns", "told": pages, "page": s["pages"][0],
                      "k": len(s["columns"][0]["x"])})
    return items


def aligned(s: dict) -> tuple[dict, dict]:
    """A told statement's cells by the columns of its first page, each page's columns
    taken by place, not order (a later page may show fewer, as reader found them): (code
    -> column -> values read, (code, column) -> (page, (left, right) of the column there))."""
    first = s["columns"][0]["x"]
    near = min((b - a_ for a_, b in zip(first, first[1:])), default=900) / 3
    cols = {c["page"]: c["x"] for c in s["columns"]}
    by, where = defaultdict(lambda: defaultdict(list)), {}
    for r in s["rows"]:
        if not r["code"]:
            continue
        xs = cols[r["page"]]
        at = {}
        for k, x in enumerate(xs):
            i = min(range(len(first)), key=lambda i: abs(first[i] - x))
            if abs(first[i] - x) <= near:
                at[i] = k
                if k < len(r["values"]) and r["values"][k]:
                    by[r["code"]][i].append(r["values"][k]["value"])
        for i in range(len(first)):
            if i in at:
                k = at[i]
                right, left = xs[k], (xs[k - 1] + PAD if k else xs[k] - 420)
            else:                               # no column found here: the first page's place
                right, left = first[i], (first[i - 1] + PAD if i else first[i] - 420)
            where.setdefault((r["code"], i), (r["page"], [left, right]))
    return by, where


def told_cells(a, files) -> list[dict]:
    """The cells of the statements not matched, by the pages and columns the person gave."""
    by = {f["sha256"]: f for f in files}
    cols, answers = all_answered(a, "columns")
    given_ = defaultdict(dict)                  # (statement, base sha) -> side -> (statement, columns)
    for it in cols:
        if it["kind"] != "columns" or answers[it["id"]]["answer"] != "columns":
            continue
        s = told_statement(a, by[it["sha256"]], it["statement"], it["told"])
        given_[it["statement"], it["base"]][it["side"]] = (s, columns_given(answers[it["id"]]["value"], it["k"]),
                                                           it["later"])
    told = told_pages(a)
    out = []
    for (kind, base_sha), sides in sorted(given_.items()):
        later_sha = next(iter(sides.values()))[2]
        base, later = by[base_sha], by[later_sha]
        b, l_ = sides.get("base"), sides.get("later")
        cb, wb = aligned(b[0]) if b else ({}, {})
        cl, wl = aligned(l_[0]) if l_ else ({}, {})
        across = (bool(b and l_) and form_of(load(a.drafts, base, told.get(base_sha))) == "2014"
                  and form_of(load(a.drafts, later, told.get(later_sha))) == "2026")
        old = {}
        if across:
            cb, old, _ = carried(cb, kind)
        for field in FIELDS:
            i = b[1].get(field) if b else None
            j = l_[1].get(field) if l_ else None
            if i is None and j is None:
                continue
            for code in sorted(cb.keys() | cl.keys(), key=int):
                bv = one(cb.get(code, {}).get(i, [])) if i is not None else "missing"
                lv = one(cl.get(code, {}).get(j, [])) if j is not None else "missing"
                bcode = old.get(code, old_of(kind, code)) if across else code
                pb, band_b = wb.get((bcode, i), (None, None)) if i is not None else (None, None)
                pl, band_l = wl.get((code, j), (None, None)) if j is not None else (None, None)
                out.append({
                    "pair": base["pair"], "statement": kind, "code": code, "told": True,
                    "base": {"sha256": base_sha, "column": i, "code": bcode, "page": pb, "band": band_b,
                             "value": bv},
                    "later": {"sha256": later_sha, "column": j, "page": pl, "band": band_l, "value": lv},
                    "status": status(bv, lv)})
    return out


def build_type(a, files) -> list[dict]:
    cells = [{**c, "told": False} for c in read_jsonl(os.path.join(a.truth, "cells.jsonl"))]
    told = told_cells(a, files)
    write_jsonl(os.path.join(a.truth, "cells-told.jsonl"), told)
    cells += told
    items = []
    for n, c in enumerate(cells):
        c["cell"] = n
        st = c["status"]
        sides = {"sign differs": ("base", "later"), "differ": ("base", "later"),
                 "ambiguous": ("base", "later"),
                 "unsure": tuple(s for s in ("base", "later") if c[s]["value"] is None),
                 "one side": tuple(s for s in ("base", "later") if c[s]["value"] != "missing")}.get(st, ())
        if st == "one side" and not sides:      # a row read, no figure in either filing: asked once
            sides = tuple(s for s in ("base", "later") if c[s].get("column") is not None
                          and c[s]["page"] is not None)[:1]
        kind = "sign" if st == "sign differs" else "type"
        items += [{"kind": kind, "cell": n, "side": s, "c": c} for s in sides if c[s]["page"] is not None]
    agreed = [c for c in cells if c["status"] == "agree"]
    rnd = random.Random(SEED)
    for c in rnd.sample(agreed, min(CHECK, len(agreed))):
        items.append({"kind": "type", "check": True, "cell": c["cell"], "side": rnd.choice(("base", "later")), "c": c})
    rnd.shuffle(items)
    return items


def build(a, files) -> list[dict]:
    p = path(a, a.round, "items")
    if os.path.exists(p):
        items, known = read_jsonl(p), {f["sha256"] for f in files}
        if any((it["c"]["base"] if "c" in it else it)["sha256"] not in known for it in items):
            sys.exit(f"{p} was made for another set than {a.set}")
        return items
    items = {"where": build_where, "columns": build_columns, "type": build_type}[a.round](a, files)
    for it in items:
        it["id"] = secrets.token_hex(4)
    write_jsonl(p, items)
    print(f"round {a.round}: {len(items)} items, " + ", ".join(
        f"{k} {n}" for k, n in sorted(Counter(it["kind"] for it in items).items())))
    return items


# ---------------------------------------------------------------- truth's own drafts

SPACED = re.compile(r"\(?\d{1,3}(?: \d{3})+\)?")


def respaced(text: str) -> str:
    """A figure grouped by spaces ("1 234 567"), grouped by dots as the reader takes them."""
    t = text.strip()
    return t.replace(" ", ".") if SPACED.fullmatch(t) else text


def overlay(a, device: str) -> None:
    """The drafts truth is built from (WORK/drafts): the reading's own, but on the pages a
    person said hold a statement, a sideways page read turned upright and figures grouped
    by spaces taken as figures. The reading scored is never changed; it missed these."""
    from fs_turn import sideways, turn
    out, src = os.path.join(WORK, "drafts"), a.drafts
    images = os.path.join(out, "images")
    os.makedirs(images, exist_ok=True)
    for name in os.listdir(os.path.join(src, "images")):
        if not os.path.lexists(os.path.join(images, name)):
            os.symlink(os.path.abspath(os.path.join(src, "images", name)), os.path.join(images, name))
    told, readers, tally = told_pages(a), None, Counter()
    for f in read_jsonl(a.set):
        name = f["sha256"][:16] + ".draft.json"
        with open(os.path.join(src, name), encoding="utf-8") as fh:
            d = json.load(fh)
        pages = told.get(f["sha256"], {})
        for i, p in enumerate(d["pages"]):
            if p["page"] not in pages or p.get("kind") != "read":
                continue
            if "turned" not in p and sideways(p):
                if readers is None:
                    from read_pdf import ROUND1, Readers
                    readers = Readers(ROUND1, device)
                d["pages"][i] = p = turn(p, images, readers, always=True)
                tally["pages turned"] += 1
            for ln in p["lines"]:
                t = respaced(ln["text"])
                if t != ln["text"]:
                    ln["text"] = t
                    tally["figures grouped by spaces"] += 1
        with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=1)
    print("truth's drafts:", ", ".join(f"{k} {n}" for k, n in sorted(tally.items())) or "as read")


# ---------------------------------------------------------------- pictures

def numbered(pages: Pages, it: dict):
    """The statement's first page told, its figure columns banded and numbered."""
    f = pages.files[it["sha256"]]
    s = told_statement(pages.a, f, it["statement"], it["told"])
    xs = s["columns"][0]["x"]
    im = pages.image(it["sha256"], it["page"])
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 90)
    except OSError:
        font = ImageFont.load_default()
    for k, x in enumerate(xs):
        left = xs[k - 1] + 12 if k else x - 420
        draw.rectangle((left, 0, x + 12, im.height - 1), outline=(0, 102, 255), width=5)
        draw.text((left + 20, 20), str(k + 1), fill=(0, 102, 255), font=font)
    return im


def picture(pages: Pages, told: Pages, it: dict):
    if it["kind"] == "columns":
        return numbered(pages, it)
    c = it["c"]
    if c["told"]:
        return crop(told, c, it["side"], c["statement"], band=c[it["side"]]["band"])
    return crop(pages, c, it["side"], c["statement"])


def shown(it: dict) -> dict:
    """What the page is told of an item: what to ask, never a reading."""
    out = {"id": it["id"], "kind": it["kind"], "statement": it.get("statement") or it["c"]["statement"]}
    if it["kind"] in ("where", "columns"):
        out.update({"side": it["side"], "ask": it["ask"], "pages": it["pages"],
                    "balance": out["statement"] == "B01"})
    if it["kind"] == "columns":
        out["k"] = it["k"]
    if it["kind"] in ("type", "sign"):                # the page the cell is on, to see its headings
        out["page"] = it["c"][it["side"]].get("page")
    return out


def valid(it: dict, answer: str, value: str) -> bool:
    if it["kind"] == "where":
        return answer == "not printed" or (answer == "pages" and pages_of(value, it["pages"]) is not None)
    if it["kind"] == "columns":
        return answer == "not this" or (answer == "columns" and columns_given(value, it["k"]) is not None)
    if it["kind"] == "unread":
        return answer == "seen"
    if answer not in ANSWERS or (answer == "typed") != bool(value):
        return False
    return answer != "typed" or value_of(value) is not None


def serve(a, items, pages):
    by_id = {it["id"]: it for it in items}
    told = Told(a, pages.files, told_pages(a))
    html = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_i.html")
    crops = os.path.join(WORK, "crops-" + a.round)
    os.makedirs(crops, exist_ok=True)
    lock = threading.Lock()
    answers = answered(a, a.round)
    key = "/" + secrets.token_urlsafe(12)       # the link's key: no page or scan without it

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, ctype="application/json; charset=utf-8"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self.path.startswith(key + "/"):
                return self.send(404, {"error": "not found"})
            self.path = self.path[len(key):]
            if self.path == "/":
                with open(html, "rb") as fh:
                    return self.send(200, fh.read(), "text/html; charset=utf-8")
            if self.path == "/api/items":
                return self.send(200, {"round": a.round, "items": [shown(it) for it in items], "answers": {
                    i: {"answer": x["answer"], "value": x["value"]} for i, x in answers.items()}})
            m = re.fullmatch(r"/(img|pdf)/([0-9a-f]{8})\.(png|pdf)", self.path)
            it = by_id.get(m.group(2)) if m else None
            sha = (it.get("sha256") or it["c"][it["side"]]["sha256"]) if it and m.group(1) == "pdf" else None
            if sha:
                with open(os.path.join(a.drafts, sha[:16] + ".draft.json"), encoding="utf-8") as fh:
                    pdf = os.path.join("cache", "pdfs", json.load(fh)["file"])
                with open(pdf, "rb") as fh:
                    return self.send(200, fh.read(), "application/pdf")
            if it and m.group(1) == "img" and it["kind"] not in ("where", "unread"):
                p = os.path.join(crops, it["id"] + ".png")
                with lock:
                    if not os.path.exists(p):
                        picture(pages, told, it).save(p)
                with open(p, "rb") as fh:
                    return self.send(200, fh.read(), "image/png")
            self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != key + "/api/answer":
                return self.send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                it, answer = by_id[body["id"]], body["answer"]
                value = str(body.get("value") or "")
                if it["kind"] in ("sign", "type"):
                    value = re.sub(r"\s+", "", value)
            except (ValueError, TypeError, KeyError) as ex:
                return self.send(400, {"error": str(ex)})
            if answer is not None and not valid(it, answer, value):
                return self.send(400, {"error": "not an answer to this question"})
            with lock:
                if answer is None:
                    answers.pop(it["id"], None)
                else:
                    answers[it["id"]] = {"id": it["id"], "answer": answer, "value": value}
                write_jsonl(path(a, a.round, "answers"), answers.values())
            self.send(200, {"ok": True})

        def log_message(self, fmt, *args):
            pass                                # nothing about the pages in the terminal

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}{key}/"
    print(f"round {a.round}: {len(items)} items; open {url}  (Ctrl+C to stop)", flush=True)
    if a.open and os.environ.get("BROWSER"):    # VS Code's terminals set it to their forwarding helper
        subprocess.Popen([os.environ["BROWSER"], url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------- finish

def finish(a) -> int:
    items, answers = all_answered(a, "type")
    cells = {it["cell"]: it["c"] for it in items}
    tally, out = Counter(), {}
    for it in items:
        c, x = it["c"], answers[it["id"]]
        read = c[it["side"]]["value"]
        rec = out.setdefault(it["cell"], {k: (dict(v) if isinstance(v, dict) else v)   # "cell": score_fs joins on it
                                          for k, v in c.items()})
        side = rec[it["side"]]
        group = ("told " if c["told"] else "") + ("check" if it.get("check") else c["status"])
        if it["kind"] == "sign":
            side["printed"] = x["answer"]
            k = ("unreadable" if x["answer"] == "unreadable"
                 else "sign as read" if x["answer"] in printed_sign(read) else "sign misread")
            if k == "sign as read":
                side["truth"] = read
        else:
            v = given(x)
            side["typed"] = v
            if v is not None:
                side["truth"] = v
            k = ("unreadable" if v is None else "not read" if not isinstance(read, int)
                 else "as read" if v == read else "misread")
        tally[f"{group}: {k}"] += 1
    write_jsonl(os.path.join(a.truth, "checked.jsonl"), [out[k] for k in sorted(out)])
    for k, n in sorted(tally.items()):
        print(f"{k}: {n}")
    print(f"cells with a typed side: {len(out)} of {len(cells)} asked about")
    return 0


def main():
    global WORK
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", required=True)
    ap.add_argument("--drafts", required=True, help="the reading's drafts of the set")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--round", default="where", choices=("where", "columns", "type"))
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--build", action="store_true", help="make the round's items and stop")
    ap.add_argument("--finish", action="store_true")
    ap.add_argument("--overlay", action="store_true", help="after the where round: make truth's drafts "
                    "(under .venv-paddle when a page is to be turned)")
    ap.add_argument("--device", default="gpu:0")
    ap.add_argument("--work", required=True, help="the rounds' items, answers and truth's drafts")
    a = ap.parse_args()
    WORK = a.work
    os.makedirs(WORK, exist_ok=True)
    if a.overlay:
        return overlay(a, a.device)
    if a.round != "where" or a.finish:          # later rounds read truth's drafts
        a.drafts = os.path.join(WORK, "drafts")
        if not os.path.isdir(a.drafts):
            sys.exit("make truth's drafts first: --overlay")
    if a.finish:
        sys.exit(finish(a))
    files = read_jsonl(a.set)
    items = build(a, files)
    if a.build:
        return
    a.items = path(a, a.round, "items")         # Pages renders text-layer pages beside it
    serve(a, items, Pages(a, {f["sha256"]: f for f in files}))


if __name__ == "__main__":
    main()
