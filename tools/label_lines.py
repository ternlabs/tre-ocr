"""A local page for labelling the crops of tools/hard_lines.py, one at a time.

Serves on 127.0.0.1 unless --host says otherwise (the scans are not ours to publish). Any
other address gets a random key in the URL, since the network can reach it. Each answer
is saved at once to <work>/labels.jsonl, gitignored, so stopping and coming back loses
nothing. Over SSH the port is forwarded; in a VS Code terminal, --open does it.

Usage:
  python -I tools/label_lines.py --work cache/train/hand --open
  python -I tools/label_lines.py --work cache/train/hand --host <this machine's LAN address>
"""
import argparse
import json
import os
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from records import read_jsonl, write_jsonl  # noqa: E402
from syllables import normal  # noqa: E402

PAGE = """<!doctype html><html lang="vi"><meta charset="utf-8"><title>Label lines</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:light dark;--line:#8886}
body{font:16px system-ui;margin:0 auto;max-width:1000px;padding:16px}
#crop{display:block;max-width:100%;min-height:48px;height:72px;object-fit:contain;object-position:left;
  background:#fff;border:1px solid var(--line);margin:12px 0}
#crop.tall{height:auto}
button{font:inherit;padding:8px 12px;margin:0 8px 8px 0;cursor:pointer}
.read{display:block;width:100%;text-align:left;font-size:20px}
textarea{display:block;font:20px/1.4 system-ui;width:100%;box-sizing:border-box;padding:8px;margin:8px 0;resize:none;overflow:hidden}
small{opacity:.7}
</style>
<p><b id="pos"></b> <small id="done"></small></p>
<img id="crop" alt="the line">
<button class="read" id="a"></button><button class="read" id="b"></button>
<textarea id="t" rows="1" autocomplete="off" spellcheck="false" aria-label="What the line says"></textarea>
<button id="save">Save (Enter)</button><button id="skip">Can't read (Esc)</button><button id="back">Back</button>
<p><small>Type exactly what the image shows, marks and punctuation too. A click on a reading
fills the box; Alt+↑ and Alt+↓ do the same. A dotted space to fill in: three dots, however long
it is, so it cannot pass for a decimal point. Words the writing has squeezed together keep their
normal space. Part of the line cut off or unclear: Can't read.</small></p>
<script>
let rows=[],at=0;const $=i=>document.getElementById(i);
function fit(){const t=$("t");t.style.height="auto";t.style.height=t.scrollHeight+"px"}
function put(v){const t=$("t");t.value=v;fit();t.focus();t.setSelectionRange(v.length,v.length)}
function show(){
  if(at>=rows.length){at=rows.length;$("pos").textContent="All done";$("crop").removeAttribute("src");return}
  const r=rows[at];$("pos").textContent=`${at+1} of ${rows.length}`;
  $("done").textContent=`${rows.filter(x=>x.answer!==undefined).length} answered`;
  $("crop").src="crop/"+encodeURIComponent(r.id)+".jpg";
  $("a").textContent=r.reads[0];$("b").textContent=r.reads[1];
  put(r.answer||r.reads[0]);
}
async function send(answer){
  const r=rows[at];
  const res=await fetch("api/answer",{method:"POST",body:JSON.stringify({id:r.id,answer})});
  if(!res.ok){alert("Not saved: "+res.status);return}
  r.answer=answer;at++;show();
}
$("a").onclick=()=>put(rows[at].reads[0]);
$("b").onclick=()=>put(rows[at].reads[1]);
const typed=()=>$("t").value.replace(/\\s*\\n\\s*/g," ");
$("save").onclick=()=>send(typed());
$("skip").onclick=()=>send(null);
$("back").onclick=()=>{if(at>0){at--;show()}};
$("t").addEventListener("keydown",e=>{
  if(e.key==="Enter"){e.preventDefault();send(typed())}
  if(e.key==="Escape")send(null);
  if(e.altKey&&e.key==="ArrowUp"){e.preventDefault();put(rows[at].reads[0])}
  if(e.altKey&&e.key==="ArrowDown"){e.preventDefault();put(rows[at].reads[1])}
});
$("t").addEventListener("input",fit);
window.addEventListener("resize",fit);
fetch("api/rows").then(r=>r.json()).then(d=>{rows=d;
  at=rows.findIndex(x=>x.answer===undefined);if(at<0)at=rows.length;show()});
</script></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--host", default="127.0.0.1", help="the address to serve on; any but 127.0.0.1 gets a key")
    ap.add_argument("--key", help="keep the key of an earlier start, so its link still works")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--open", action="store_true", help="open the page; over VS Code SSH this forwards the port")
    a = ap.parse_args()
    rows = read_jsonl(os.path.join(a.work, "candidates.jsonl"))
    by_id = {r["id"]: r for r in rows}
    path = os.path.join(a.work, "labels.jsonl")
    answers = {x["id"]: x for x in (read_jsonl(path) if os.path.exists(path) else [])}
    lock = threading.Lock()
    prefix = "" if a.host == "127.0.0.1" else "/" + (a.key or secrets.token_urlsafe(16))

    class Handler(BaseHTTPRequestHandler):
        def route(self):
            """The path without the key, or None when the key is missing."""
            return self.path[len(prefix):] if self.path.startswith(prefix + "/") else None

        def send(self, code, body, kind="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.route()
            if path == "/":
                return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/api/rows":
                return self.send(200, [{"id": r["id"], "reads": r["reads"],
                                        **({"answer": answers[r["id"]]["answer"]} if r["id"] in answers else {})}
                                       for r in rows])
            k = path[len("/crop/"):-len(".jpg")] if path and path.startswith("/crop/") else None
            if k in by_id:                       # only names from the list reach the disk
                with open(os.path.join(a.work, "crops", k + ".jpg"), "rb") as fh:
                    return self.send(200, fh.read(), "image/jpeg")
            self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.route() != "/api/answer":
                return self.send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                r, answer = by_id[body["id"]], body["answer"]
            except (ValueError, TypeError, KeyError) as ex:
                return self.send(400, {"error": str(ex)})
            with lock:
                answers[r["id"]] = {"id": r["id"], "answer": normal(answer).strip() if answer else None}
                write_jsonl(path, list(answers.values()))
            self.send(200, {"ok": True})

        def log_message(self, fmt, *args):
            pass                                 # page text stays out of the terminal

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}{prefix}/"
    print(f"{len(rows)} lines, {len(answers)} answered; open {url}  (Ctrl+C to stop)", flush=True)
    if a.open and os.environ.get("BROWSER"):     # VS Code's terminals set it to their forwarding helper
        subprocess.Popen([os.environ["BROWSER"], url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
