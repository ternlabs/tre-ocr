"""The review page of the tool (v0.2): a person fills what the draft left empty.

Works from tools/read_pdf.py's <name>.draft.json and the images beside it. Serves
tools/review.html on 127.0.0.1 only. One item per:

  number   a number left empty, [?], on an unsure line: shown on its line's box;
           the person types what is printed, says nothing is, or leaves it unread
  field    a heading field to confirm, or left empty: the person
           agrees, types it, says it is not printed, or leaves it unread. A document
           where the rule found no heading at all is not asked about; its fields say so.
           Nor is a company that matches the listing on a page read by OCR (state
           "listing"): it stands, "how: matches the listing".

Every answer rewrites <name>.reviewed.json beside the draft: each page's text with the
typed numbers in, each document's fields with how each was got. A value left unread
stays [?] or empty; nothing is guessed for it.

Usage:
  .venv/bin/python -I tools/review.py cache/drafts/some.draft.json --open
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from read_pdf import BLANK  # noqa: E402
from syllables import normal  # noqa: E402

ANSWERS = {"ok", "typed", "absent", "unread"}
FIELDS = ("company", "number", "place", "date")
SLOT_RE = re.compile(re.escape(BLANK))


def fraction(boxes: list, size: tuple[int, int]) -> list[dict]:
    """The union of line boxes, as shares of the page."""
    boxes = [b for b in boxes if b]
    if not boxes:
        return []
    (w, h), pad = size, 6
    x1, y1 = min(b[0] for b in boxes) - pad, min(b[1] for b in boxes) - pad
    x2, y2 = max(b[2] for b in boxes) + pad, max(b[3] for b in boxes) + pad
    return [{"x": x1 / w, "y": y1 / h, "w": (x2 - x1) / w, "h": (y2 - y1) / h}]


def around(lines: list[dict], i: int, slot: int | None = None) -> dict:
    """A line with its neighbours; the nth slot, if given, marked."""
    line = lines[i]["text"]
    if slot is not None:
        m = list(SLOT_RE.finditer(line))[slot]
        line = line[:m.start()] + "\0" + line[m.end():]
    return {"before": lines[i - 1]["text"] if i else "", "line": line,
            "after": lines[i + 1]["text"] if i + 1 < len(lines) else ""}


def build(draft: dict, images: str) -> list[dict]:
    pages = {p["page"]: p for p in draft["pages"]}
    sizes = {}

    def size(p):
        if p["key"] not in sizes:
            with Image.open(os.path.join(images, p["key"] + ".png")) as im:
                sizes[p["key"]] = im.size
        return sizes[p["key"]]

    items = []
    for d in draft["documents"]:
        shown = [pages[n] for n in d["pages"] if pages[n]["kind"] != "skipped"]
        if not shown:
            continue
        for f in FIELDS if any(d.get(f) for f in FIELDS) else ():   # no heading found: nothing to ask
            g = d.get(f) or {"value": None, "state": "empty", "page": shown[0]["page"], "lines": []}
            if g["state"] == "listing":
                continue
            p = pages[g["page"]]
            at = g["lines"]
            items.append({"id": f"d{d['doc']}-{f}", "kind": "field", "field": f, "shown": g["value"],
                          "key": p["key"], "page": p["page"], "text_layer": p["kind"] == "text layer",
                          "filer": draft.get("filer") if f == "company" else None, "listing": g.get("listing"),
                          "context": around(p["lines"], at[0]) if at else None,
                          "boxes": fraction([p["lines"][i]["box"] for i in at], size(p))})
        for p in shown:
            for i, ln in enumerate(p["lines"]):
                for j, _ in enumerate(SLOT_RE.findall(ln["text"]) if not ln["sure"] else []):
                    items.append({"id": f"p{p['page']}-l{i}-s{j}", "kind": "number", "shown": None,
                                  "key": p["key"], "page": p["page"], "context": around(p["lines"], i, j),
                                  "boxes": fraction([ln["box"]], size(p))})
    return items


def reviewed(draft: dict, answers: dict, items: list[dict]) -> dict:
    """The draft with the answers in: what the person typed, and what stays unread."""
    pages = []
    for p in draft["pages"]:
        lines = []
        for i, ln in enumerate(p["lines"]):
            text, n = ln["text"], iter(range(10 ** 6))

            def put(m, page=p["page"], i=i):
                x = answers.get(f"p{page}-l{i}-s{next(n)}") or {}
                return x.get("value") if x.get("answer") == "typed" else "" if x.get("answer") == "absent" else BLANK
            lines.append(text if ln["sure"] else SLOT_RE.sub(put, text))
        pages.append({"page": p["page"], "kind": p["kind"], "why": p.get("why"), "seal": p.get("seal"),
                      "english": p.get("english"), "text": "\n".join(lines),
                      "unsure": [i for i, ln in enumerate(p["lines"]) if not ln["sure"]]})
    docs = []
    for d in draft["documents"]:
        out = {}
        for f in FIELDS:
            g = d.get(f) or {"value": None, "state": "empty"}
            x = answers.get(f"d{d['doc']}-{f}") or {}
            how = {"ok": "confirmed", "typed": "typed", "absent": "not printed", "unread": "unread"}.get(
                x.get("answer"), "no heading found" if not any(d.get(f) for f in FIELDS) else
                "matches the listing" if g["state"] == "listing" else "not reviewed")
            value = g["value"] if how in ("confirmed", "matches the listing") else x["value"] if how == "typed" else None
            out[f] = {"value": value, "how": how}
            if "listing" in g:
                out[f]["listing"] = g["listing"]
        docs.append({"doc": d["doc"], "pages": d["pages"], "fields": out})
    left = sum(r["id"] not in answers for r in items)
    unread = sum(x["answer"] == "unread" for x in answers.values())
    return {"file": draft["file"], "sha256": draft["sha256"], "drafted": draft.get("drafted"), "filer": draft.get("filer"),
            "profile": draft["profile"], "reviewed": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "items": len(items), "not_reviewed": left, "unread": unread,
            "notes": draft["notes"] + [
                f"Reviewed by a person: {len(items) - left} of {len(items)} items answered, {unread} left unread. "
                f"{BLANK} is a number nobody read; a field's \"how\" says where its value came from."],
            "documents": docs, "pages": pages}


def serve(a, draft, items):
    by_id = {r["id"]: r for r in items}
    keys = {r["key"] for r in items}
    images = os.path.join(os.path.dirname(a.draft), "images")
    out = re.sub(r"\.draft\.json$", "", a.draft) + ".reviewed.json"
    html = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review.html")
    lock = threading.Lock()
    answers = {}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            old = json.load(fh)
        if old.get("sha256") != draft["sha256"] or old.get("drafted") != draft.get("drafted"):
            sys.exit(f"{out} holds answers to another draft (another PDF, or this one drafted again): "
                     "move it away to start over")
        answers = {i: x for i, x in old.get("answers", {}).items() if i in by_id}

    def save():
        r = reviewed(draft, answers, items)
        r["answers"] = answers
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(r, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, out)

    class Handler(BaseHTTPRequestHandler):
        def local(self) -> bool:
            """Host and Origin name this machine: no other site, nor a rebound name, gets in."""
            def host(v):
                return re.sub(r":\d+$", "", re.sub(r"^https?://", "", v or "")).strip("[]")
            ok = {"127.0.0.1", "localhost", "::1"}
            return host(self.headers.get("Host")) in ok and (
                self.headers.get("Origin") is None or host(self.headers["Origin"]) in ok)

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
            if not self.local():
                return self.send(403, {"error": "not from this machine"})
            if self.path == "/":
                with open(html, "rb") as fh:
                    return self.send(200, fh.read(), "text/html; charset=utf-8")
            if self.path == "/api/items":
                with lock:
                    body = json.dumps({"file": draft["file"], "notes": draft["notes"], "out": out,
                                       "items": items, "answers": answers}, ensure_ascii=False).encode()
                return self.send(200, body)
            k = self.path[len("/img/"):-len(".png")] if self.path.startswith("/img/") else None
            if k in keys:
                with open(os.path.join(images, k + ".png"), "rb") as fh:
                    return self.send(200, fh.read(), "image/png")
            self.send(404, {"error": "not found"})

        def do_POST(self):
            # a JSON content type makes another site's request need a preflight, which is never answered
            if not self.local() or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send(403, {"error": "not from this page"})
            if self.path != "/api/answer":
                return self.send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                r, answer, value = by_id[body["id"]], body["answer"], normal(str(body.get("value") or "")).strip()
            except (ValueError, TypeError, KeyError) as ex:
                return self.send(400, {"error": str(ex)})
            if answer is not None and (answer not in ANSWERS or (answer == "typed") != bool(value)
                                       or (answer == "ok" and not r["shown"])):
                return self.send(400, {"error": "unknown answer"})
            with lock:
                if answer is None:
                    answers.pop(r["id"], None)
                else:
                    answers[r["id"]] = {"answer": answer, "value": value if answer == "typed" else ""}
                save()
            self.send(200, {"ok": True})

        def log_message(self, fmt, *args):
            pass                                # page text stays out of the terminal

    with lock:
        save()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"{len(items)} items; open {url}  (Ctrl+C to stop); answers to {out}", flush=True)
    if a.open and os.environ.get("BROWSER"):    # VS Code's terminals set it to their forwarding helper
        subprocess.Popen([os.environ["BROWSER"], url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif a.open:
        print("no $BROWSER here: forward the port and open the address yourself", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("draft", help="a <name>.draft.json from tools/read_pdf.py")
    ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--open", action="store_true", help="open the page in the browser; over VS Code SSH, the local one")
    a = ap.parse_args()
    with open(a.draft, encoding="utf-8") as fh:
        draft = json.load(fh)
    serve(a, draft, build(draft, os.path.join(os.path.dirname(a.draft), "images")))


if __name__ == "__main__":
    main()
