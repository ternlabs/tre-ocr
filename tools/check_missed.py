"""Check set A's missed numbers by hand: a local page, one number at a time.

Step 8 of METHOD.md. Set A's truth comes from the English twin, so a number no
engine read may simply not be on the Vietnamese scan. Serves
tools/check_missed.html on 127.0.0.1 only (the scans are not ours to publish).
For each number of --missed it shows the scan, the number's place in the twin,
the closest number each engine read, and boxes where Tesseract saw similar
digits. The boxes and reads say where to look; the answer comes from the image.

  on      the digits are on the scan: a real miss
  off     they are not: written to --excuse, taken out of the truth by score.py
  unsure  not excused, so it stays a miss

Answers go to --answers on every click (gitignored). Numbers answered before stay
listed when a later --missed leaves them out, as it does for excused ones.

Usage:
  python -I tools/score.py a data/runs/*-a.jsonl --missed data/missed-a.jsonl
  python -I tools/check_missed.py

Over SSH the server stays on 127.0.0.1 and the port is forwarded. In a VS Code
Remote-SSH terminal, --open forwards it and opens the page in the local browser
(or use the Ports panel: forward the port, then open it). In plain ssh:
  ssh -L 8766:127.0.0.1:8766 <host>
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from label_pages import kept_text, read_pages, signature_boxes  # noqa: E402
from number_truth import cached  # noqa: E402
from records import NUMBER_RE, page_key, read_jsonl, write_jsonl  # noqa: E402

ANSWERS = {"on", "off", "unsure"}
AROUND = 70                                     # characters of context each side


def edits(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        last, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            last, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, last + (x != y))
    return row[-1]


def digits_of(s: str) -> str:
    return re.sub(r"[.,]", "", s)


def twin_places(text: str, number: str) -> list[dict]:
    """Where the number stands in the twin's text, whitespace closed up."""
    text = " ".join(text.split())
    return [{"before": text[max(0, m.start() - AROUND):m.start()], "match": m.group(),
             "after": text[m.end():m.end() + AROUND]}
            for m in NUMBER_RE.finditer(text) if digits_of(m.group()) == number]


def closest_read(text: str, number: str) -> dict | None:
    """The engine's nearest number to this one, with the line it stands on."""
    best = None
    for line in text.splitlines():
        for m in NUMBER_RE.finditer(line):
            d = digits_of(m.group())
            if len(d) < 3:
                continue
            n = edits(d, number)
            if n <= max(1, len(number) // 3) and (best is None or n < best["differ"]):
                best = {"read": m.group(), "differ": n, "line": line.strip()[:200]}
    return best


def word_boxes(png: str, tsv: str) -> tuple[list[dict], int, int]:
    """Tesseract's words with digits in them, as boxes, and the page size."""
    if not os.path.exists(tsv):
        out = subprocess.run(["tesseract", png, "stdout", "-l", "vie", "--psm", "3", "tsv"],
                             capture_output=True, text=True, check=True,
                             env={**os.environ, "OMP_THREAD_LIMIT": "1"}).stdout
        with open(tsv + ".tmp", "w", encoding="utf-8") as fh:
            fh.write(out)
        os.replace(tsv + ".tmp", tsv)
    words, w, h = [], 1, 1
    with open(tsv, encoding="utf-8") as fh:
        for line in fh.read().splitlines()[1:]:
            f = line.split("\t")
            if len(f) < 12:
                continue
            if f[0] == "1":
                w, h = int(f[8]), int(f[9])
            elif f[0] == "5" and re.search(r"\d", f[11]):
                words.append({"digits": re.sub(r"\D", "", f[11]),
                              "box": [int(f[6]), int(f[7]), int(f[8]), int(f[9])]})
    return words, w, h


def near_boxes(words: list[dict], w: int, h: int, number: str) -> list[dict]:
    """Boxes of words like the number, or holding part of it, closest first."""
    out = []
    for x in words:
        d = x["digits"]
        n = edits(d, number)
        if n <= max(1, len(number) // 4):
            kind = n
        elif len(d) >= 3 and d in number:
            kind = len(number)                  # a piece: split by a stray space
        else:
            continue
        l, t, bw, bh = x["box"]
        out.append({"differ": kind, "x": l / w, "y": t / h, "w": bw / w, "h": bh / h})
    return sorted(out, key=lambda b: b["differ"])


def row_id(r: dict) -> str:
    return f"{page_key(r)}/{r['number']}/{r['nth']}"


def build(a) -> list[dict]:
    """The numbers to check, with all each one's aids."""
    answers = read_jsonl(a.answers) if os.path.exists(a.answers) else []
    missed, seen = [], Counter()
    for r in read_jsonl(a.missed):
        k = (r["sha256"], r["page"], str(r["number"]))
        missed.append({"sha256": k[0], "page": k[1], "number": k[2], "nth": seen[k]})
        seen[k] += 1
    known = {row_id(r) for r in missed}
    rows = missed + [{f: x[f] for f in ("sha256", "page", "number", "nth")}
                     for x in answers if row_id(x) not in known]

    pairs = {(r["vi"]["sha256"], r["vi"]["page"]): r for r in read_jsonl(a.set)}
    aligned = {(r["vi"]["sha256"], r["vi"]["page"]): r for r in read_jsonl(a.align)}
    runs = {os.path.basename(p).rsplit("-", 1)[0]:
            {(x["sha256"], x["page"]): x["text"] for x in read_jsonl(p)} for p in a.runs}

    os.makedirs(a.boxes, exist_ok=True)
    keys = sorted({(r["sha256"], r["page"]) for r in rows})

    def boxes_of(k):
        name = page_key({"sha256": k[0], "page": k[1]})
        return k, word_boxes(os.path.join(a.images, name + ".png"), os.path.join(a.boxes, name + ".tsv"))

    print(f"finding digits on {len(keys)} pages ...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        boxes = dict(ex.map(boxes_of, keys))

    twins = {}
    for k in keys:
        en = aligned[k]["en"]
        pages = read_pages(cached(pairs[k]["en"]["pdf"], a.cache))
        twins[k] = " ".join(kept_text(p["blocks"], signature_boxes(p["blocks"]))
                            for p in pages[en["from"] - 1:en["to"]])

    for r in rows:
        k = (r["sha256"], r["page"])
        r.update(id=row_id(r), key=page_key(r), stamp=pairs[k]["vi"]["stamp"],
                 twin=twin_places(twins[k], r["number"]),
                 reads={name: closest_read(texts.get(k, ""), r["number"]) for name, texts in runs.items()},
                 boxes=near_boxes(*boxes[k], r["number"]))
    return sorted(rows, key=lambda r: (r["key"], r["number"], r["nth"]))


def serve(a, rows):
    by_id = {r["id"]: r for r in rows}
    keys = {r["key"] for r in rows}
    html = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_missed.html")
    lock = threading.Lock()
    answers = {row_id(x): x for x in (read_jsonl(a.answers) if os.path.exists(a.answers) else [])}

    def save():
        write_jsonl(a.answers, answers.values())
        write_jsonl(a.excuse, [{"sha256": x["sha256"], "page": x["page"], "number": x["number"]}
                               for x in answers.values() if x["answer"] == "off"])

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
            if self.path == "/api/rows":
                return self.send(200, {"rows": rows, "answers": {i: x["answer"] for i, x in answers.items()},
                                       "excuse": a.excuse})
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
                r, answer = by_id[body["id"]], body["answer"]
            except (ValueError, TypeError, KeyError) as ex:
                return self.send(400, {"error": str(ex)})
            if answer not in ANSWERS and answer is not None:
                return self.send(400, {"error": "unknown answer"})
            with lock:
                if answer is None:
                    answers.pop(r["id"], None)
                else:
                    answers[r["id"]] = {f: r[f] for f in ("sha256", "page", "number", "nth")} | {"answer": answer}
                save()
            self.send(200, {"ok": True})

        def log_message(self, fmt, *args):
            pass                                # page text stays out of the terminal

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"{len(rows)} numbers; open {url}  (Ctrl+C to stop)", flush=True)
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
    ap.add_argument("--missed", default="data/missed-a.jsonl")
    ap.add_argument("--runs", nargs="*", default=sorted(glob.glob("data/runs/*-a.jsonl")))
    ap.add_argument("--set", default="sets/disclosures-a.jsonl")
    ap.add_argument("--align", default="data/align-a.jsonl")
    ap.add_argument("--images", default="cache/set-a")
    ap.add_argument("--cache", default="cache/pdfs")
    ap.add_argument("--boxes", default="cache/check-a-boxes")
    ap.add_argument("--answers", default="data/check-a.jsonl")
    ap.add_argument("--excuse", default="data/excuse-a.jsonl")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--open", action="store_true", help="open the page in the browser; over VS Code SSH, the local one")
    a = ap.parse_args()
    serve(a, build(a))


if __name__ == "__main__":
    main()
