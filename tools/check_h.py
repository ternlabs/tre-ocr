"""Set H's truth check: the cells the two filings' readings do not settle, typed blind.

From <truth>/cells.jsonl (tools/fs_truth.py), each item is one side of one cell, its row
cut from the scan with the cell outlined. The person never sees a reading.

  sign    every "sign differs" cell, both sides: how is the figure printed?
  type    every "unsure" cell, the side left empty; and SAMPLE "differ" cells drawn
          with a fixed seed, both sides: what figure is printed?

Round 2 (--round 2): every disagreement is typed. Built from the truth as it is then:
both sides of every "differ" cell, the empty side of every "unsure" one, both sides of
every "sign differs" one. A side round 1 already answered (same file, page, column and
code) keeps its answer, except the items given to --drop (their crops pointed off the
figure).

The page shows the code outlined and the column's whole band: the person finds the line
the code is on and its figure in the band. Round 1 outlined a cell on the code's height;
where a label wraps, or a page is askew, the figure is printed off that height.

Items come shuffled under random ids; which cell and side each is stays in the item
file (gitignored, beside the answers), with the cell as it was when the items were made. Served
on 127.0.0.1, or with --lan on the local network for a phone, always under a random key in the
link: the scans are not ours.

--finish compares the answers with the readings and writes <truth>/checked.jsonl:
per cell, each side's value as the person gave it. It prints counts only. Exit 0: no
sampled disagreement was a misread (the rest are restatements, left out of truth);
1: not all answered; 2: a misread in the sample, so every disagreement is typed.

Usage:
  .venv/bin/python -I tools/check_h.py --open
  .venv/bin/python -I tools/check_h.py --finish
  .venv/bin/python -I tools/check_h.py --round 2 --drop <id,id,...> --open   (then --round 2 --finish)
  .venv/bin/python -I tools/check_h.py --round 2 --lan     (prints a link to open on a phone)
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
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from fs_rows import value_of  # noqa: E402
from fs_truth import load  # noqa: E402
from records import read_jsonl, write_jsonl  # noqa: E402

SAMPLE = 30
SEED = 11
SIGNS = ("plain", "parentheses", "minus", "dash", "nothing")   # how a figure is printed
ANSWERS = {"typed", "dash", "nothing", "unreadable"} | set(SIGNS)
CONTEXT = 3.5        # line heights of the page shown above and below the code (a label wraps)
PAD = 12             # px around the outlined code and column


def printed_sign(v: int | None) -> set[str]:
    """The ways a reading's sign can be printed."""
    if v is None:
        return set()
    return {"dash", "nothing"} if v == 0 else {"parentheses", "minus"} if v < 0 else {"plain"}


def build(a) -> list[dict]:
    """The items, made once and kept, so ids and order hold across runs."""
    if os.path.exists(a.items):
        return read_jsonl(a.items)
    cells = read_jsonl(os.path.join(a.truth, "cells.jsonl"))
    for i, c in enumerate(cells):
        c["cell"] = i
    differ = [c for c in cells if c["status"] == "differ"]
    picked = random.Random(SEED).sample(differ, min(SAMPLE, len(differ)))
    items = []
    for c in cells:
        if c["status"] == "sign differs":
            items += [{"kind": "sign", "cell": c["cell"], "side": s, "c": c} for s in ("base", "later")]
        elif c["status"] == "unsure":
            items += [{"kind": "type", "cell": c["cell"], "side": s, "c": c} for s in ("base", "later")
                      if c[s]["value"] is None]
    items += [{"kind": "type", "cell": c["cell"], "side": s, "sampled": True, "c": c}
              for c in picked for s in ("base", "later")]
    random.Random(SEED).shuffle(items)
    for it in items:
        it["id"] = secrets.token_hex(4)
    write_jsonl(a.items, items)
    return items


def place(it: dict) -> tuple:
    """Where an item's figure is printed: file, page, column, code on that side."""
    s = it["c"][it["side"]]
    return s["sha256"], s["page"], s["column"], s.get("code") or it["c"]["code"]


def build2(a, first: list[dict], first_answers: dict, drop: set[str]) -> list[dict]:
    """Round 2's items, made once; round 1's answers for the same places carried in."""
    if os.path.exists(a.items):
        return read_jsonl(a.items)
    cells = read_jsonl(os.path.join(a.truth, "cells.jsonl"))
    items = []
    for i, c in enumerate(cells):
        c["cell"] = i
        if c["status"] == "sign differs":
            items += [{"kind": "sign", "cell": i, "side": s, "c": c} for s in ("base", "later")]
        elif c["status"] == "unsure":
            items += [{"kind": "type", "cell": i, "side": s, "c": c} for s in ("base", "later")
                      if c[s]["value"] is None]
        elif c["status"] == "differ":
            items += [{"kind": "type", "cell": i, "side": s, "differ": True, "c": c} for s in ("base", "later")]
    random.Random(SEED).shuffle(items)
    old = {(it["kind"], place(it)): first_answers[it["id"]] for it in first
           if it["id"] in first_answers and it["id"] not in drop}
    carried = []
    for it in items:
        it["id"] = secrets.token_hex(4)
        x = old.get((it["kind"], place(it)))
        if x:
            carried.append({**x, "id": it["id"], "from": "round 1"})
    write_jsonl(a.items, items)
    write_jsonl(a.answers, carried)
    print(f"round 2: {len(items)} items, {len(carried)} answered in round 1")
    return items


class Pages:
    """Statements of a file, read once; the page image a row's boxes are on."""

    def __init__(self, a, files: dict):
        self.a, self.files, self.ss = a, files, {}

    def statements(self, sha: str) -> list[dict]:
        if sha not in self.ss:
            self.ss[sha] = load(self.a.drafts, self.files[sha])
        return self.ss[sha]

    def image(self, sha: str, page: int) -> Image.Image:
        with open(os.path.join(self.a.drafts, sha[:16] + ".draft.json"), encoding="utf-8") as fh:
            p = next(p for p in json.load(fh)["pages"] if p["page"] == page)
        if p.get("kind") == "text layer":        # placed from the PDF at 300 dpi (fs_rows.layer_lines)
            out = os.path.join(os.path.dirname(self.a.items), "pages", f"{sha[:16]}-p{page}")
            if not os.path.exists(out + ".png"):
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(os.path.join(self.a.drafts, sha[:16] + ".draft.json"), encoding="utf-8") as fh:
                    pdf = os.path.join("cache", "pdfs", json.load(fh)["file"])
                subprocess.run(["pdftoppm", "-q", "-png", "-r", "300", "-singlefile", "-f", str(page),
                                "-l", str(page), pdf, out], check=True)
            return Image.open(out + ".png").convert("RGB")
        name = p["key"] + (f"-r{p['turned']}" if "turned" in p else "")
        return Image.open(os.path.join(self.a.drafts, "images", name + ".png")).convert("RGB")


def crop(pages: Pages, cell: dict, side: str, statement: str, band=None) -> Image.Image:
    """The cell's row and a little around it, the cell outlined. band: the column's (left,
    right) edges, when they are not the page's own column s["column"]."""
    s = cell[side]
    code = s.get("code") or cell["code"]
    st = [x for x in pages.statements(s["sha256"]) if x["statement"] == statement]
    row = next(r for x in st for r in x["rows"] if r["code"] == code and r["page"] == s["page"])
    im = pages.image(s["sha256"], s["page"])
    x0, y0, x1, y1 = row["code_box"]
    h = max(y1 - y0, 30)
    top, bottom = max(0, int(y0 - CONTEXT * h)), min(im.height, int(y1 + CONTEXT * h))
    if band:
        left, right = band
    else:
        cols = next(c["x"] for x in st for c in x["columns"] if c["page"] == s["page"]
                    and any(r is row for r in x["rows"]))
        right = cols[s["column"]]
        left = cols[s["column"] - 1] + PAD if s["column"] else right - 420
    left = max(left, x1 + PAD)
    draw = ImageDraw.Draw(im)
    draw.rectangle((x0 - PAD, y0 - PAD, x1 + PAD, y1 + PAD), outline=(235, 120, 0), width=6)   # the code: orange
    draw.rectangle((left, top + 3, right + PAD, bottom - 3), outline=(0, 102, 255), width=3)   # the column: blue
    out = im.crop((0, top, im.width, bottom))
    dark = np.asarray(out.convert("L")) < 128
    inked = np.flatnonzero(dark.mean(axis=0) > 0.01)  # columns with print, not a speck at the edge
    lo = max(0, int(inked[0]) - 20) if inked.size else 0   # the page's margin left out: a phone shows more
    return out.crop((lo, 0, min(out.width, int(right + PAD + 60)), out.height))


def lan_address() -> str:
    """This machine's address on the local network (no packet is sent)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
        sk.connect(("10.255.255.255", 1))
        return sk.getsockname()[0]


def serve(a, items, pages):
    by_id = {it["id"]: it for it in items}
    html = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_h.html")
    crops = os.path.join(os.path.dirname(a.items), "crops")
    os.makedirs(crops, exist_ok=True)
    lock = threading.Lock()
    answers = {x["id"]: x for x in (read_jsonl(a.answers) if os.path.exists(a.answers) else [])}
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
                # no reading, cell or file is sent: only what the person is asked
                shown = [{"id": it["id"], "kind": it["kind"],
                          "statement": it["c"]["statement"]} for it in items]
                return self.send(200, {"items": shown, "answers": {
                    i: {"answer": x["answer"], "value": x["value"]} for i, x in answers.items()}})
            k = self.path[len("/img/"):-len(".png")] if self.path.startswith("/img/") else None
            if k in by_id:
                path = os.path.join(crops, k + ".png")
                with lock:
                    if not os.path.exists(path):
                        it = by_id[k]
                        c = it["c"]
                        crop(pages, c, it["side"], c["statement"]).save(path)
                with open(path, "rb") as fh:
                    return self.send(200, fh.read(), "image/png")
            self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != key + "/api/answer":
                return self.send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                it, answer = by_id[body["id"]], body["answer"]
                value = re.sub(r"\s+", "", str(body.get("value") or ""))
            except (ValueError, TypeError, KeyError) as ex:
                return self.send(400, {"error": str(ex)})
            if answer is not None and (answer not in ANSWERS or (answer == "typed") != bool(value)
                                       or (answer == "typed" and value_of(value) is None)):
                return self.send(400, {"error": "not a figure" if answer == "typed" else "unknown answer"})
            with lock:
                if answer is None:
                    answers.pop(it["id"], None)
                else:
                    answers[it["id"]] = {"id": it["id"], "answer": answer, "value": value}
                write_jsonl(a.answers, answers.values())
            self.send(200, {"ok": True})

        def log_message(self, fmt, *args):
            pass                                # nothing about the pages in the terminal

    host = lan_address() if a.lan else "127.0.0.1"
    srv = ThreadingHTTPServer((host, a.port), Handler)
    url = f"http://{host}:{a.port}{key}/"
    print(f"{len(items)} items; open {url}  (Ctrl+C to stop)", flush=True)
    if a.lan:
        print("on the local network: anyone with this link can see the scans while it runs", flush=True)
    if a.open and os.environ.get("BROWSER"):    # VS Code's terminals set it to their forwarding helper
        subprocess.Popen([os.environ["BROWSER"], url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif a.open:
        print("no $BROWSER here: forward the port and open the address yourself", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def given(x: dict):
    """A typed answer as a value: the figure, 0 for a dash, "missing", or None (unreadable)."""
    if x["answer"] == "typed":
        return value_of(x["value"])
    return {"dash": 0, "nothing": "missing"}.get(x["answer"])


def finish(a, items) -> int:
    answers = {x["id"]: x for x in (read_jsonl(a.answers) if os.path.exists(a.answers) else [])}
    left = [it for it in items if it["id"] not in answers]
    if left:
        print(f"{len(left)} of {len(items)} items not answered")
        return 1
    tally, out = Counter(), {}
    for it in items:
        c, x = it["c"], answers[it["id"]]
        read = c[it["side"]]["value"]
        rec = out.setdefault(it["cell"], {"statement": c["statement"], "code": c["code"],
                                          "status": c["status"], "base": dict(c["base"]),
                                          "later": dict(c["later"])})
        side = rec[it["side"]]
        if it["kind"] == "sign":
            side["printed"] = x["answer"]
            k = ("unreadable" if x["answer"] == "unreadable"
                 else "sign as read" if x["answer"] in printed_sign(read) else "sign misread")
            tally["sign: " + k] += 1
            if k == "sign as read":
                side["truth"] = read
        else:
            v = given(x)
            side["typed"] = v
            if v is not None:
                side["truth"] = v
            kind = ("sampled differ" if it.get("sampled") else "differ" if it.get("differ")
                    else "unsure side")
            if v is None:
                k = "unreadable"
            elif read is None:
                other = c["later" if it["side"] == "base" else "base"]["value"]
                k = "as the other filing read" if v == other else "not as the other filing read"
            else:
                k = "as read" if v == read else "misread"
            tally[f"{kind}: {k}"] += 1
    write_jsonl(os.path.join(a.truth, "checked.jsonl"), [out[k] for k in sorted(out)])
    for k, n in sorted(tally.items()):
        print(f"{k}: {n}")
    sampled = sum(n for k, n in tally.items() if k.startswith("sampled differ"))
    misread = tally["sampled differ: misread"]
    if a.round == 2:
        print(f"disagreements typed: {tally['differ: as read'] + tally['differ: misread']} sides; "
              f"misread {tally['differ: misread']}")
        return 0
    print(f"sampled disagreements: {sampled // 2} cells; sides misread {misread}")
    return 2 if misread else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", default="sets/statements-h.jsonl")
    ap.add_argument("--drafts", default="data/set-h/drafts")
    ap.add_argument("--truth", default="data/truth-h")
    ap.add_argument("--round", type=int, default=1, choices=(1, 2))
    ap.add_argument("--drop", default="", help="round 1 item ids whose answers are not carried")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--lan", action="store_true", help="listen on the local network, e.g. for a phone")
    ap.add_argument("--finish", action="store_true")
    a = ap.parse_args()
    a.items = "data/check-h/items.jsonl" if a.round == 1 else "data/check-h/items-2.jsonl"
    a.answers = "data/check-h/answers.jsonl" if a.round == 1 else "data/check-h/answers-2.jsonl"
    os.makedirs(os.path.dirname(a.items), exist_ok=True)
    files = {f["sha256"]: f for f in read_jsonl(a.set)}
    if a.round == 1:
        items = build(a)
    else:
        first = read_jsonl("data/check-h/items.jsonl")
        first_answers = {x["id"]: x for x in read_jsonl("data/check-h/answers.jsonl")}
        items = build2(a, first, first_answers, set(filter(None, a.drop.split(","))))
    if a.finish:
        sys.exit(finish(a, items))
    serve(a, items, Pages(a, files))


if __name__ == "__main__":
    main()
