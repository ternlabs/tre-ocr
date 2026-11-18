"""Correct set C by hand: a local page that shows each scan beside its drafts.

Step 6 of METHOD.md. Serves tools/correct_c.html on 127.0.0.1 unless --host says
otherwise (the scans are not ours to publish). Any other address gets a random key
in the URL, since the network can reach it. Per page, the person:
  1. corrects the text, starting from a draft (tools/draft_c.py), and fills the
     heading fields (company, document number, place, date) when the page has a
     heading; saved as "corrected";
  2. on a second read, ticks every number (3+ digits) and heading field after
     checking it against the image, not the draft; saved as "checked";
  or drops the page, with a reason (identity field, list of people, unreadable).

Each page's record goes to <truth>/<sha256 first 16>-p<page>.json, gitignored:
the text holds names, and it is the only copy, so back the folder up.

--marks shows words to look at in yellow until they are gone: a JSON file, page key to
[{"word", "fix", "until"?}]. "until" names a line the marked one should come after.
It says what the page holds, so it stays gitignored.

--finish marks each row of the set with its fate: "kept" for the first --keep
pages in draw order that are checked, "dropped: <reason>", or "spare"; and
says whether the kept pages come from enough companies.

Usage:
  python -I tools/correct_c.py sets/disclosures-c.jsonl --work data/set-c --images cache/set-c --truth data/truth-c
  python -I tools/correct_c.py sets/disclosures-c.jsonl --work data/set-c --images cache/set-c --truth data/truth-c \
    --host <this machine's LAN address>
  python -I tools/correct_c.py sets/disclosures-c.jsonl --truth data/truth-c --finish
"""
import argparse
import json
import os
import re
import secrets
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from records import write_jsonl  # noqa: E402
from syllables import normal  # noqa: E402

KEY_RE = re.compile(r"^[0-9a-f]{16}-p\d+$")
STATES = {"corrected", "checked"}
DROPS = {"identity field", "list of people", "unreadable", "not a disclosure page"}
FIELDS = ("company", "number", "place", "date")
MIN_COMPANIES = 10


def key(e: dict) -> str:
    return f"{e['sha256'][:16]}-p{e['page']}"


def load(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def clean(rec: dict) -> dict:
    """The record as stored, or ValueError."""
    drop = rec.get("drop") or None
    if drop is not None and drop not in DROPS:
        raise ValueError(f"unknown drop reason: {drop}")
    state = rec.get("state")
    if drop is None and state not in STATES:
        raise ValueError(f"unknown state: {state}")
    heading = rec.get("heading")
    if heading is not None:
        heading = {f: normal(str(heading.get(f) or "")).strip() for f in FIELDS}
    return {"text": normal(str(rec.get("text") or "")), "heading": heading,
            "state": None if drop else state, "drop": drop,
            "checked_numbers": [str(x) for x in rec.get("checked_numbers") or []],
            "checked_fields": [f for f in rec.get("checked_fields") or [] if f in FIELDS],
            "note": str(rec.get("note") or ""),
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def serve(a, entries):
    by_key = {key(e): e for e in entries}
    html = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correct_c.html")
    drafts_dir = os.path.join(a.work, "drafts")
    os.makedirs(a.truth, exist_ok=True)
    key_part = "" if a.host == "127.0.0.1" else "/" + (a.key or secrets.token_urlsafe(16))
    marks = (load(a.marks) or {}) if a.marks else {}

    def drafts(k):
        out = {}
        if os.path.isdir(drafts_dir):
            for name in sorted(os.listdir(drafts_dir)):
                p = os.path.join(drafts_dir, name, k + ".txt")
                if os.path.exists(p):
                    with open(p, encoding="utf-8") as fh:
                        out[name] = fh.read()
        return out

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

        def route(self):
            """The path without the key, or None when the key is missing."""
            return self.path[len(key_part):] if self.path.startswith(key_part + "/") else None

        def page_key(self, path, prefix, suffix=""):
            k = path[len(prefix):len(path) - len(suffix)]
            return k if KEY_RE.match(k) and k in by_key else None

        def do_GET(self):
            path = self.route()
            if path is None:
                return self.send(404, {"error": "not found"})
            if path == "/":
                with open(html, "rb") as fh:
                    return self.send(200, fh.read(), "text/html; charset=utf-8")
            if path == "/api/pages":
                rows = []
                for e in entries:
                    r = load(os.path.join(a.truth, key(e) + ".json")) or {}
                    rows.append({"key": key(e), "order": e["order"], "symbol": e["symbol"],
                                 "kind": e["kind"], "page": e["page"], "stamp": e["stamp"],
                                 "state": r.get("state"), "drop": r.get("drop"),
                                 "marks": marks.get(key(e)), "text": r.get("text"),
                                 "heading": r.get("heading")})
                return self.send(200, rows)
            if path.startswith("/api/page/"):
                k = self.page_key(path, "/api/page/")
                if not k:
                    return self.send(404, {"error": "no such page"})
                return self.send(200, {"entry": by_key[k], "drafts": drafts(k), "marks": marks.get(k),
                                       "record": load(os.path.join(a.truth, k + ".json"))})
            if path.startswith("/img/") and path.endswith(".png"):
                k = self.page_key(path, "/img/", ".png")
                p = k and os.path.join(a.images, k + ".png")
                if not p or not os.path.exists(p):
                    return self.send(404, {"error": "no image; run tools/draft_c.py"})
                with open(p, "rb") as fh:
                    return self.send(200, fh.read(), "image/png")
            self.send(404, {"error": "not found"})

        def do_POST(self):
            path = self.route()
            k = self.page_key(path, "/api/page/") if path and path.startswith("/api/page/") else None
            if not k:
                return self.send(404, {"error": "no such page"})
            try:
                rec = clean(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            except (ValueError, TypeError, KeyError) as ex:
                return self.send(400, {"error": str(ex)})
            out = os.path.join(a.truth, k + ".json")
            with open(out + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=1)
            os.replace(out + ".tmp", out)
            self.send(200, rec)

        def log_message(self, fmt, *args):
            pass                                # page text stays out of the terminal

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"open http://{a.host}:{a.port}{key_part}/  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def finish(a, entries):
    kept, out = 0, []
    for e in sorted(entries, key=lambda e: e["order"]):
        r = load(os.path.join(a.truth, key(e) + ".json")) or {}
        e = {k: v for k, v in e.items() if k != "fate"}
        if r.get("drop"):
            e["fate"] = f"dropped: {r['drop']}"
        elif r.get("state") == "checked" and kept < a.keep:
            e["fate"], kept = "kept", kept + 1
        else:
            e["fate"] = "spare"
        out.append(e)
    write_jsonl(a.set, out)
    companies = len({e["symbol"] for e in out if e["fate"] == "kept"})
    print(json.dumps({"kept": kept, "companies": companies,
                      "dropped": sum(e["fate"].startswith("dropped") for e in out),
                      "spare": sum(e["fate"] == "spare" for e in out)}, indent=2))
    if kept < a.keep or companies < MIN_COMPANIES:
        sys.exit(f"not done: need {a.keep} checked pages from {MIN_COMPANIES}+ companies")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--work", default="data/set-c")
    ap.add_argument("--images", default="cache/set-c")
    ap.add_argument("--host", default="127.0.0.1", help="the address to serve on; any but 127.0.0.1 gets a key")
    ap.add_argument("--key", help="keep the key of an earlier start, so its link still works")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--marks", help="a JSON file of words to show in yellow until they are gone")
    ap.add_argument("--finish", action="store_true")
    ap.add_argument("--keep", type=int, default=20)
    a = ap.parse_args()
    with open(a.set, encoding="utf-8") as fh:
        entries = [json.loads(line) for line in fh]
    (finish if a.finish else serve)(a, entries)


if __name__ == "__main__":
    main()
