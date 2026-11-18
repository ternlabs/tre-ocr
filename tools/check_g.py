"""Set G's check: is each company that matched the listing what its heading prints?

METHOD.md, "The company check (set G)". The unit
is a company: its first document, in draw order, whose filled company matches the
listing on a page read by OCR (listing.read_matches). Each item is the heading's company
lines cut from the page image and a value; the person says whether the image prints the
same letters and marks, case and spaces aside.

Agreeing is easy, so of the first UNITS units, DECOYS drawn with a fixed seed are shown
only with one letter or mark changed; the other WANT are the test, shown as filled. Items
come in a shuffled order under random ids; which are decoys is kept in the round's item
file (gitignored, beside the answers), never sent to the page. Served on 127.0.0.1 only.

--finish counts: decoys passed as the same (more than 1 voids the round), and test units
found to differ or that the person cannot tell (both wrong; the mark: at most 1 of WANT).
Exit 0 passed, 1 not all answered, 2 failed, 3 void. After a void round, draw 10 more
units (tools/grow_set_g.py --units 80) and answer --round 2: those 10 are its decoys.

Usage:
  .venv/bin/python -I tools/check_g.py --open
  .venv/bin/python -I tools/check_g.py --finish
"""
import argparse
import io
import json
import os
import random
import re
import secrets
import subprocess
import sys
import threading
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from listing import key, read_matches  # noqa: E402
from records import read_jsonl  # noqa: E402

DECOYS = 10
SEED = 46
WANT = 60
UNITS = WANT + DECOYS
ANSWERS = {"same", "differs", "unsure"}
TONES = ["̀", "́", "̃", "̉", "̣"]
SWAP = {"a": "ă", "ă": "â", "â": "a", "e": "ê", "ê": "e", "o": "ô", "ô": "ơ", "ơ": "o",
        "u": "ư", "ư": "u", "đ": "d", "d": "đ", "n": "m", "m": "n", "h": "k", "t": "l",
        "i": "l", "c": "e", "g": "q", "r": "n", "s": "x", "x": "s", "y": "v", "v": "y"}


def change(value: str, rng: random.Random) -> str:
    """The value with one letter or one tone mark changed, so its key differs."""
    for _ in range(200):
        at = [i for i, c in enumerate(value) if c.isalpha()]
        i = rng.choice(at)
        ch = value[i]
        d = unicodedata.normalize("NFD", ch)
        base, marks = d[0], [m for m in d[1:] if m in TONES]
        rest = [m for m in d[1:] if m not in TONES]
        if marks and rng.random() < .5:
            new = base + "".join(rest) + rng.choice([t for t in TONES if t != marks[0]])
        else:
            low = unicodedata.normalize("NFC", base + "".join(rest)).lower()
            if low not in SWAP:
                continue
            sub = SWAP[low]
            sub = sub.upper() if ch.isupper() else sub
            new = unicodedata.normalize("NFD", sub) + "".join(marks)
        out = value[:i] + unicodedata.normalize("NFC", new) + value[i + 1:]
        if key(out) != key(value):
            return out
    raise ValueError("no change found")


def units_of(set_path: str, drafts: str) -> list[dict]:
    """Per company, in draw order, its first document matched on a read page."""
    out, seen = [], set()
    for f in sorted(read_jsonl(set_path), key=lambda f: f["order"]):
        if f["symbol"] in seen:
            continue
        with open(os.path.join(drafts, f["sha256"][:16] + ".draft.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        docs = read_matches(d)
        if not docs:
            continue
        seen.add(f["symbol"])
        g = docs[0]["company"]
        p = next(p for p in d["pages"] if p["page"] == g["page"])
        boxes = [p["lines"][i]["box"] for i in g["lines"] if p["lines"][i]["box"]]
        out.append({"unit": f"{f['sha256'][:16]}-d{docs[0]['doc']}", "value": g["value"],
                    "key": p["key"], "boxes": boxes, "english": "english" in p})
    return out


def build(units: list[dict], rnd: int) -> list[dict]:
    """The round's items, shuffled, under random ids: the test as filled, the decoys changed."""
    if len(units) < UNITS + DECOYS * (rnd - 1):
        sys.exit(f"{len(units)} units; round {rnd} needs {UNITS + DECOYS * (rnd - 1)}: draw on first")
    rng = random.Random(SEED)
    first = rng.sample(range(UNITS), DECOYS)
    test = [u for i, u in enumerate(units[:UNITS]) if i not in first]
    decoys = [units[i] for i in first] if rnd == 1 else units[UNITS + DECOYS * (rnd - 2):UNITS + DECOYS * (rnd - 1)]
    rng = random.Random(f"{SEED}:{rnd}")
    every = [{**u, "decoy": False} for u in test] + \
            [{**u, "decoy": True, "value": change(u["value"], rng)} for u in decoys]
    rng.shuffle(every)
    return [{**u, "id": secrets.token_hex(6)} for u in every]


def items(a, units) -> list[dict]:
    """The round's items, made once and kept, so ids and answers last across restarts."""
    if os.path.exists(a.items):
        return read_jsonl(a.items)
    every = build(units, a.round)
    os.makedirs(os.path.dirname(a.items) or ".", exist_ok=True)
    with open(a.items, "w", encoding="utf-8") as fh:
        for r in every:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return every


def crop(images: str, r: dict) -> bytes:
    with Image.open(os.path.join(images, r["key"] + ".png")) as im:
        w, h = im.size
        if r["boxes"]:
            pad = 30
            x1 = max(0, min(b[0] for b in r["boxes"]) - pad)
            y1 = max(0, min(b[1] for b in r["boxes"]) - pad)
            x2 = min(w, max(b[2] for b in r["boxes"]) + pad)
            y2 = min(h, max(b[3] for b in r["boxes"]) + pad)
            part = im.crop((x1, y1, x2, y2))
        else:
            part = im.crop((0, 0, w, int(h * .35)))
        buf = io.BytesIO()
        part.convert("RGB").save(buf, "PNG")
        return buf.getvalue()


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Set G check</title>
<style>
body{font:16px/1.4 system-ui,sans-serif;margin:16px;max-width:1100px}
img{max-width:100%;border:1px solid #888;background:#fff}
#value{font:600 24px/1.3 ui-monospace,monospace;margin:12px 0;overflow-wrap:anywhere}
button{font-size:16px;padding:8px 14px;margin-right:8px}
.muted{color:#777}
</style>
<p class="muted">Does the image print exactly the value below: the same letters and the same marks?
Case and spaces do not count. Some values were changed on purpose by one letter or one mark;
read letter by letter. Keys: <b>s</b> same, <b>d</b> differs, <b>u</b> cannot tell; <b>←</b> back, <b>→</b> next.
An answer can be changed: go back and answer again.</p>
<div id="pos"></div><img id="img" alt="The heading"><div id="value"></div>
<button data-a="same">Same as printed (s)</button><button data-a="differs">Differs (d)</button>
<button data-a="unsure">Cannot tell (u)</button>
<button id="back">← Back</button><button id="next">Next →</button>
<p id="state"></p>
<script>
let items=[],answers={},at=0;
const $=id=>document.getElementById(id);
function show(i){at=Math.max(0,Math.min(items.length-1,i));const r=items[at];
 $("pos").textContent=`${at+1} of ${items.length}; answered ${Object.keys(answers).length}`;
 $("img").src=`/img/${encodeURIComponent(r.id)}.png`;$("value").textContent=r.value;
 $("state").textContent=answers[r.id]?`Answered: ${answers[r.id]}`:"";}
async function answer(a){const r=items[at];
 const res=await fetch("/api/answer",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({id:r.id,answer:a})});
 if(!res.ok){$("state").textContent="Not saved: "+res.status;return;}
 answers[r.id]=a;show(at+1<items.length?at+1:at);}
document.querySelectorAll("button[data-a]").forEach(b=>b.onclick=()=>answer(b.dataset.a));
$("back").onclick=()=>show(at-1);$("next").onclick=()=>show(at+1);
document.addEventListener("keydown",e=>{if(!items.length)return;
 const k={s:"same",d:"differs",u:"unsure"}[e.key];if(k)answer(k);
 else if(e.key==="ArrowRight")show(at+1);else if(e.key==="ArrowLeft")show(at-1);});
fetch("/api/items").then(r=>r.json()).then(j=>{items=j.items;answers=j.answers;
 const f=items.findIndex(x=>!answers[x.id]);show(f<0?0:f);});
</script></html>"""


def serve(a, every):
    by_id = {r["id"]: r for r in every}
    lock = threading.Lock()
    answers = {}
    if os.path.exists(a.answers):
        for r in read_jsonl(a.answers):
            if r["id"] in by_id:
                answers[r["id"]] = r["answer"]
    images = os.path.join(a.drafts, "images")

    def save():
        tmp = a.answers + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for i, x in answers.items():
                fh.write(json.dumps({"id": i, "answer": x}) + "\n")
        os.replace(tmp, a.answers)

    class Handler(BaseHTTPRequestHandler):
        def local(self) -> bool:
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
                return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/api/items":
                with lock:
                    shown = [{"id": r["id"], "value": r["value"]} for r in every]   # no decoy flag
                    return self.send(200, {"items": shown, "answers": dict(answers)})
            m = re.fullmatch(r"/img/([0-9a-f]+)\.png", self.path)
            if m and m.group(1) in by_id:
                return self.send(200, crop(images, by_id[m.group(1)]), "image/png")
            self.send(404, {"error": "not found"})

        def do_POST(self):
            if not self.local() or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send(403, {"error": "not from this page"})
            if self.path != "/api/answer":
                return self.send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                r, answer = by_id[body["id"]], body["answer"]
            except (ValueError, TypeError, KeyError) as ex:
                return self.send(400, {"error": str(ex)})
            if answer not in ANSWERS:
                return self.send(400, {"error": "unknown answer"})
            with lock:
                answers[r["id"]] = answer
                save()
            self.send(200, {"ok": True})

        def log_message(self, fmt, *args):
            pass                                # values stay out of the terminal

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"{len(every)} items; open {url}  (Ctrl+C to stop); answers to {a.answers}", flush=True)
    if a.open and os.environ.get("BROWSER"):
        subprocess.Popen([os.environ["BROWSER"], url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def finish(a, every) -> int:
    answers = {r["id"]: r["answer"] for r in read_jsonl(a.answers)} if os.path.exists(a.answers) else {}
    test = [r for r in every if not r["decoy"]]
    decoys = [r for r in every if r["decoy"]]
    left = sum(r["id"] not in answers for r in every)
    passed = sum(answers.get(r["id"]) == "same" for r in decoys)
    wrong = [r for r in test if answers.get(r["id"]) in ("differs", "unsure")]     # unsure is wrong
    english = [r for r in test if r["english"]]
    out = {"round": a.round, "test": len(test), "decoys": len(decoys), "not answered": left,
           "decoys passed as same": passed,
           "test differs": sum(answers.get(r["id"]) == "differs" for r in test),
           "test cannot tell": sum(answers.get(r["id"]) == "unsure" for r in test),
           "test wrong": len(wrong),
           "on English pages": {"test": len(english), "wrong": sum(r["english"] for r in wrong)}}
    code = 1 if left else 3 if passed > 1 else 0 if len(test) >= WANT and len(wrong) <= 1 else 2
    out["result"] = {0: "passed", 1: "not all answered", 2: "failed", 3: "void: draw 10 more units, answer the next round"}[code]
    print(json.dumps(out, indent=1))
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", default="sets/disclosures-g.jsonl")
    ap.add_argument("--drafts", default="data/set-g/drafts")
    ap.add_argument("--round", type=int, default=1, help="2, 3 ...: after a void round")
    ap.add_argument("--port", type=int, default=8769)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--finish", action="store_true")
    a = ap.parse_args()
    a.items = f"data/check-g/round-{a.round}.items.jsonl"
    a.answers = f"data/check-g/round-{a.round}.answers.jsonl"
    if a.finish and not os.path.exists(a.items):
        sys.exit(f"no items for round {a.round}: nothing was shown")
    every = items(a, units_of(a.set, a.drafts))
    if a.finish:
        sys.exit(finish(a, every))
    serve(a, every)


if __name__ == "__main__":
    main()
