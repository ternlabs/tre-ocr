"""Check set F's truth by hand where it matters: numbers and the company field.

Step 9 again (METHOD.md, set F). The first reading of set F's pages is in --drafts
(fields in its fields.json); a person checks what a wrong reading would change in the
score, not the tone marks. Serves
tools/check_f.html on 127.0.0.1 only (the scans are not ours to publish).

  type     a number (3+ digits) where the draft and either reader differ:
           shown masked on its line; the person types what is printed
  extra    a number a reader read that the draft lacks: masked on the reader's
           line; the person types what is printed there, or says nothing is
  tick     100 numbers all three agree on, drawn at random: shown; right or retyped
  company  the heading's company field: shown; right or retyped

Answers go to --answers on every click (gitignored). "unsure" on a number in the
draft takes it out of the truth (written to --excuse), as on set A.

--finish writes <truth>/<key>.json for every page read (kept or set aside), the
draft with the typed numbers put in, and marks each row of the set with its fate.
A number printed but missing from the draft, or a company retyped that the draft
does not hold, is listed for a hand fix of the draft; finish stops until none is left.

Usage:
  python -I tools/check_f.py --open
  python -I tools/check_f.py --finish
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from check_missed import near_boxes, word_boxes  # noqa: E402
from records import NUMBER_RE, numbers, page_key, read_jsonl, write_jsonl  # noqa: E402
from syllables import normal  # noqa: E402

READERS = ("paddleocr", "paddleocr-tuned")
ANSWERS = {"ok", "typed", "absent", "unsure"}
TICKS = 100
SEED = 9
DROPS = {"identity field", "list of people", "unreadable", "not a disclosure page", "English page"}
FIELDS = ("company", "number", "place", "date")


def digits(s: str) -> str:
    return re.sub(r"[.,]", "", s)


def places(text: str) -> list[dict]:
    """Each number of 3+ digits in the text: its line, span and nth among equal digits."""
    out, seen = [], Counter()
    for i, line in enumerate(text.splitlines()):
        for m in NUMBER_RE.finditer(line):
            d = digits(m.group())
            if len(d) >= 3:
                out.append({"line": i, "start": m.start(), "end": m.end(), "read": m.group(),
                            "digits": d, "nth": seen[d]})
                seen[d] += 1
    return out


def around(lines: list[str], i: int, start: int | None = None, end: int | None = None) -> dict:
    """A line with its neighbours; the span, if given, masked."""
    line = lines[i]
    if start is not None:
        line = line[:start] + "\0" + line[end:]
    return {"before": lines[i - 1] if i else "", "line": line,
            "after": lines[i + 1] if i + 1 < len(lines) else ""}


def load(a):
    with open(os.path.join(a.drafts, "fields.json"), encoding="utf-8") as fh:
        fields = json.load(fh)
    drafts = {}
    for k, f in fields.items():
        if "drop" not in f:
            with open(os.path.join(a.drafts, k + ".txt"), encoding="utf-8") as fh:
                drafts[k] = normal(fh.read())
    runs = {n: {page_key(x): x["text"] for x in read_jsonl(os.path.join(a.runs, f"{n}-f.jsonl"))} for n in READERS}
    return fields, drafts, runs


def build(a) -> list[dict]:
    """Every item to check, in page order, with the boxes that say where to look."""
    fields, drafts, runs = load(a)
    items, agreed = [], []
    for k, text in drafts.items():
        lines = text.splitlines()
        reads = [Counter(numbers(runs[n].get(k, ""))) for n in READERS]
        both = reads[0] & reads[1]
        for p in places(text):
            r = {"key": k, "digits": p["digits"], "nth": p["nth"]}
            if both[p["digits"]] > p["nth"]:
                agreed.append(r | {"kind": "tick", "shown": p["read"], "context": around(lines, p["line"])})
            else:
                items.append(r | {"kind": "type", "context": around(lines, p["line"], p["start"], p["end"])})
        have = Counter(numbers(text))
        extra = (reads[0] | reads[1]) - have
        for n in READERS:
            rl = runs[n].get(k, "").splitlines()
            for p in places(runs[n].get(k, "")):
                if extra[p["digits"]] > 0:
                    extra[p["digits"]] -= 1
                    items.append({"key": k, "kind": "extra", "digits": p["digits"],
                                  "nth": have[p["digits"]] + extra[p["digits"]],
                                  "context": around(rl, p["line"], p["start"], p["end"]), "reader": n})
        if fields[k].get("company"):
            items.append({"key": k, "kind": "company", "digits": "", "nth": 0, "shown": fields[k]["company"]})
    items += random.Random(SEED).sample(agreed, min(TICKS, len(agreed)))
    for r in items:
        r["ref"] = f"{r['key']}/{r['kind']}/{r['digits']}/{r['nth']}"
        r["id"] = hashlib.sha1(r["ref"].encode()).hexdigest()[:12]   # opaque: the ref holds the digits

    os.makedirs(a.boxes, exist_ok=True)
    keys = sorted({r["key"] for r in items})

    def boxes_of(k):
        return k, word_boxes(os.path.join(a.images, k + ".png"), os.path.join(a.boxes, k + ".tsv"))

    print(f"finding digits on {len(keys)} pages ...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        boxes = dict(ex.map(boxes_of, keys))
    for r in items:
        r["boxes"] = near_boxes(*boxes[r["key"]], r["digits"]) if r["digits"] else []
    order = {k: i for i, k in enumerate(drafts)}
    kinds = {"company": 0, "type": 1, "extra": 2, "tick": 3}
    return sorted(items, key=lambda r: (order[r["key"]], kinds[r["kind"]], r["digits"], r["nth"]))


def serve(a, items):
    by_id = {r["id"]: r for r in items}
    keys = {r["key"] for r in items}
    html = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_f.html")
    lock = threading.Lock()
    answers = {x["id"]: x for x in (read_jsonl(a.answers) if os.path.exists(a.answers) else [])}

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
            if self.path == "/":
                with open(html, "rb") as fh:
                    return self.send(200, fh.read(), "text/html; charset=utf-8")
            if self.path == "/api/items":
                # a masked number's digits stay on the server: the check is blind
                shown = [{f: v for f, v in r.items() if f not in ("digits", "nth", "ref")} for r in items]
                return self.send(200, {"items": shown, "answers": {i: {"answer": x["answer"], "value": x["value"]}
                                                                   for i, x in answers.items()}})
            k = self.path[len("/img/"):-len(".png")] if self.path.startswith("/img/") else None
            if k in keys:
                with open(os.path.join(a.images, k + ".png"), "rb") as fh:
                    return self.send(200, fh.read(), "image/png")
            self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/answer":
                return self.send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                r, answer, value = by_id[body["id"]], body["answer"], normal(str(body.get("value") or "")).strip()
            except (ValueError, TypeError, KeyError) as ex:
                return self.send(400, {"error": str(ex)})
            if answer is not None and (answer not in ANSWERS or (answer == "typed") != bool(value)):
                return self.send(400, {"error": "unknown answer"})
            with lock:
                if answer is None:
                    answers.pop(r["id"], None)
                else:
                    answers[r["id"]] = {"id": r["id"], "answer": answer, "value": value}
                write_jsonl(a.answers, answers.values())
            self.send(200, {"ok": True})

        def log_message(self, fmt, *args):
            pass                                # page text stays out of the terminal

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"{len(items)} items; open {url}  (Ctrl+C to stop)", flush=True)
    if a.open and os.environ.get("BROWSER"):    # VS Code's terminals set it to their forwarding helper
        subprocess.Popen([os.environ["BROWSER"], url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif a.open:
        print("no $BROWSER here: forward the port and open the address yourself", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def squash(s: str) -> str:
    return "".join(s.split()).casefold()


def finish(a):
    fields, drafts, _ = load(a)
    built = build(a)
    by_id = {x["id"]: x for x in (read_jsonl(a.answers) if os.path.exists(a.answers) else [])}
    items = {r["ref"]: r for r in built}
    answers = {r["ref"]: by_id[r["id"]] for r in built if r["id"] in by_id}
    left = [i for i in items if i not in answers]
    if left:
        sys.exit(f"not done: {len(left)} of {len(items)} items unanswered")
    entries = read_jsonl(a.set)
    by_key = {page_key(e): e for e in entries}
    fixes, excuse, hand = 0, [], []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(a.truth, exist_ok=True)
    for k, text in drafts.items():
        lines = text.splitlines()
        swaps, checked = [], []
        for p in places(text):
            x = answers.get(f"{k}/type/{p['digits']}/{p['nth']}") or answers.get(f"{k}/tick/{p['digits']}/{p['nth']}")
            if not x:
                continue
            checked.append(f"{p['read']}#{p['nth'] + 1}")
            if x["answer"] == "typed" and digits(x["value"]) != p["digits"]:
                swaps.append((p, x["value"]))
            elif x["answer"] == "absent":
                swaps.append((p, ""))
            elif x["answer"] == "unsure":
                excuse.append({"sha256": by_key[k]["sha256"], "page": by_key[k]["page"], "number": p["digits"]})
        for p, value in sorted(swaps, key=lambda s: (-s[0]["line"], -s[0]["start"])):
            ln = lines[p["line"]]
            lines[p["line"]] = ln[:p["start"]] + value + ln[p["end"]:]
            fixes += 1
        text = "\n".join(lines) + "\n"
        have = Counter(numbers(text))
        for i, x in answers.items():
            r = items.get(i)
            if r and r["key"] == k and r["kind"] == "extra" and x["answer"] == "typed":
                d = digits(x["value"])
                # a piece of a longer number the reader split ("102, 448,257") is that number
                if len(d) >= 3 and have[d] <= 0 and not any(d in n for n in have if len(n) > len(d)):
                    hand.append(f"{k}: printed but not in the draft: {x['value']}")
                have[d] -= 1
        heading = None
        if fields[k].get("company"):
            heading = {f: fields[k].get(f, "") for f in FIELDS}
            x = answers.get(f"{k}/company//0")
            if x["answer"] == "typed":
                heading["company"] = x["value"]
                if squash(x["value"]) not in squash(text):
                    hand.append(f"{k}: company retyped, not in the draft's heading")
            elif x["answer"] == "unsure":
                heading["company"] = ""
        rec = {"text": text, "heading": heading, "state": "checked", "drop": None,
               "checked_numbers": checked, "checked_fields": ["company"] if heading and heading["company"] else [],
               "updated": now}
        with open(os.path.join(a.truth, k + ".json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=1)
    if hand:
        sys.exit("fix the drafts by hand, then finish again:\n  " + "\n  ".join(hand))
    for k, f in fields.items():
        if "drop" in f:
            if f["drop"] not in DROPS:
                sys.exit(f"{k}: unknown drop reason {f['drop']!r}")
            with open(os.path.join(a.truth, k + ".json"), "w", encoding="utf-8") as fh:
                json.dump({"text": "", "heading": None, "state": None, "drop": f["drop"], "checked_numbers": [],
                           "checked_fields": [], "note": "", "updated": now}, fh, ensure_ascii=False, indent=1)
    out = []
    for e in sorted(entries, key=lambda e: e["order"]):
        f = fields.get(page_key(e))
        e = {k: v for k, v in e.items() if k != "fate"}
        e["fate"] = "spare" if f is None else f"dropped: {f['drop']}" if "drop" in f else "kept"
        out.append(e)
    write_jsonl(a.set, out)
    write_jsonl(a.excuse, excuse)
    ticks = [answers[i]["answer"] for i, r in items.items() if r["kind"] == "tick"]
    print(json.dumps({"kept": sum(e["fate"] == "kept" for e in out),
                      "dropped": sum(e["fate"].startswith("dropped") for e in out),
                      "spare": sum(e["fate"] == "spare" for e in out),
                      "numbers_changed": fixes, "excused": len(excuse),
                      "ticks": dict(Counter(ticks))}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", default="sets/disclosures-f.jsonl")
    ap.add_argument("--drafts", default="data/set-f/drafts/claude")
    ap.add_argument("--runs", default="data/runs")
    ap.add_argument("--images", default="cache/set-f")
    ap.add_argument("--boxes", default="cache/check-f-boxes")
    ap.add_argument("--answers", default="data/check-f.jsonl")
    ap.add_argument("--excuse", default="data/excuse-f.jsonl")
    ap.add_argument("--truth", default="data/truth-f")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--open", action="store_true", help="open the page in the browser; over VS Code SSH, the local one")
    ap.add_argument("--finish", action="store_true")
    a = ap.parse_args()
    finish(a) if a.finish else serve(a, build(a))


if __name__ == "__main__":
    main()
