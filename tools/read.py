"""Read disclosure PDFs for a caller with no person to review them (METHOD.md, "The read entry point").

The tool of v0.2 (tools/read_pdf.py) minus the review page. Each value says how it was
got; a value that would have gone to a person is given as unchecked, or left empty, and
empty is final.

- **Refused before any page is read** when the meta's title is of a kind the sets did not
  measure, or there is no title: the output is measured: false, its reason, no text.
  --unmeasured reads it all the same, still marked measured: false.
- **Company:** only when it matches the listing's name for the meta's symbol; else empty.
- **Number, place:** as the rule found them, "rule, unchecked".
- **Date:** checked against the meta's published day when given: a date after it (one
  day's grace) disagrees and is left empty. The check was never measured.

Nothing kept, nothing fetched: page images go to a temporary folder; the meta and the
PDF's path are never written or logged; the listing is read from its cache, never fetched.

Meta: {"title": ..., "symbol": ... (optional), "published": "YYYY-MM-DD..." (optional)}.
A batch is a JSON-lines file of {"pdf": path, "meta": {...}}: the models load once.

Writes <out>/<first 16 hex of the SHA-256>.json per PDF, valid against
schemas/read.schema.json, and prints per PDF its batch line, output file and status. One
output per PDF: a later job for the same PDF replaces it.
Exit 0: all read; 2: some refused, none failed; 1: any failed.

Usage:
  .venv-paddle/bin/python -I tools/read.py some.pdf --meta meta.json --out folder
  .venv-paddle/bin/python -I tools/read.py --batch jobs.jsonl --out folder
"""
import argparse
import json
import os
import re
import sys
import tempfile
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)                                         # works under python -I too
from grow_set_b import wanted  # noqa: E402
from sort_kinds import kind_of  # noqa: E402

VERSION = "0.2.1"                   # the tool: v0.2, read without a person
SCHEMA = "urn:schema:pdf-read:1"    # schemas/read.schema.json's $id; its major number
FIELDS = ("company", "number", "place", "date")
GRACE = timedelta(days=1)           # a document dated the day after it was published: time zones
VI_DATE_RE = re.compile(r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})", re.I)
EN_DATE_RE = re.compile(r"([a-z]{3})[a-z]*\.?\s*(\d{1,2})\s*(?:st|nd|rd|th)?\s*,?\s*(\d{4})", re.I)
MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
NOTES_SWAP = {
    "Every number on an unsure line is left empty, [?], for a person.":
        "Every number on an unsure line is left empty, [?]. No person checks it: empty is final.",
}
WEAK = ("Weak spots: the heading's document number and dates, which no person checks here: "
        "\"how\" says how each value was got. The company is given only when it matches the listing.")


class Refused(Exception):
    pass


def day_of(text: str) -> date | None:
    """A heading date, Vietnamese or English, as a day; None when it is not one."""
    try:
        if m := VI_DATE_RE.search(text):
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if (m := EN_DATE_RE.search(text)) and m.group(1).lower() in MONTHS:
            return date(int(m.group(3)), MONTHS.index(m.group(1).lower()) + 1, int(m.group(2)))
    except ValueError:
        pass
    return None


def published_of(meta: dict) -> date | None:
    v = meta.get("published")
    if v is None:
        return None
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        raise ValueError("meta.published is not a date (YYYY-MM-DD)") from None


def refusal(meta: dict) -> str | None:
    """Why this filing is not read, from its title alone; None to read it."""
    title = meta.get("title")
    if not isinstance(title, str) or not title.strip():
        return "no title in the meta: its kind cannot be told"
    kind = kind_of(title)
    if not wanted(kind):
        return f"a kind not measured: {kind}"
    return None


def fields_of(d: dict, kinds: dict, filer_state: str, published: date | None) -> dict:
    """One document's heading fields, each {value, how}, nothing left for a person."""
    out = {}
    found = any(d.get(f) for f in FIELDS)
    for f in FIELDS:
        g = d.get(f) or {"value": None}
        v = g["value"]
        if not found:
            out[f] = {"value": None, "how": "no heading found"}
        elif v is None:
            out[f] = {"value": None, "how": "not found by the rule, or on an unsure line"}
        elif f == "company":
            if filer_state != "listed":
                out[f] = {"value": None, "how": filer_state}
            elif not g.get("listing"):
                out[f] = {"value": None, "how": "does not match the listing"}
            elif kinds[g["page"]] == "read":
                out[f] = {"value": v, "how": "matches the listing (provisional)"}
            else:
                out[f] = {"value": v, "how": "matches the listing, on a text layer (untested)"}
        elif f == "date" and published is not None:
            day = day_of(v)
            if day is None:
                out[f] = {"value": v, "how": "rule, unchecked"}
            elif day > published + GRACE:
                out[f] = {"value": None, "how": "disagrees with meta"}
            else:
                out[f] = {"value": v, "how": "consistent with meta (untested)"}
        else:
            out[f] = {"value": v, "how": "rule, unchecked"}
    return out


def notes_of(notes: list[str]) -> list[str]:
    out = []
    for n in notes:
        if n.startswith("Weak spots:"):
            out.append(WEAK)
        elif n.startswith("The company is compared with the listing"):
            continue                                # the review page's note; "how" says it here
        else:
            out.append(NOTES_SWAP.get(n, n))
    return out


def read_one(path: str, meta: dict, readers, symbols: dict | None, unmeasured: bool) -> dict:
    from read_pdf import draft
    from sample_pairs import sha256_of
    sha = sha256_of(path)
    head = {"schema": SCHEMA, "tool": VERSION, "profile": "disclosures", "sha256": sha}
    published = published_of(meta)
    why = refusal(meta)
    if why and not unmeasured:
        raise Refused(why)
    symbol = meta.get("symbol")
    filer, filer_state = None, "listed"
    if not symbol:
        filer_state = "no symbol in the meta"
    elif symbols is None:
        filer_state = "no cached listing here"
    elif not (filer := symbols.get(str(symbol).upper())):
        filer_state = "symbol not in the listing"
    with tempfile.TemporaryDirectory() as tmp:
        d = draft(path, tmp, readers, filer, show=False)
    kinds = {p["page"]: p["kind"] for p in d["pages"]}
    docs = [{"doc": x["doc"], "pages": x["pages"], "fields": fields_of(x, kinds, filer_state, published)}
            for x in d["documents"]]
    pages = []
    for p in d["pages"]:
        q = {"page": p["page"], "kind": p["kind"]}
        for k in ("why", "english", "seal"):
            if p.get(k):
                q[k] = p[k]
        q["lines"] = [{"text": ln["text"], "sure": ln["sure"]} for ln in p["lines"]]
        pages.append(q)
    out = {**head, "measured": why is None}
    if why:
        out["reason"] = why + "; read on request (--unmeasured)"
    return {**out, "notes": notes_of(d["notes"]), "documents": docs, "pages": pages}


def jobs_of(a) -> list[tuple[str, dict]]:
    if a.batch:
        jobs = []
        with open(a.batch, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    j = json.loads(line)
                    jobs.append((j.get("pdf"), j.get("meta")))
        return jobs
    if not a.pdf or not a.meta:
        sys.exit("give a PDF and --meta, or --batch")
    with open(a.meta, encoding="utf-8") as fh:
        return [(a.pdf, json.load(fh))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pdf", nargs="?")
    ap.add_argument("--meta", help="a JSON file: title, symbol, published")
    ap.add_argument("--batch", help="a JSON-lines file of {pdf, meta}")
    ap.add_argument("--out", required=True, help="the folder for the outputs")
    ap.add_argument("--unmeasured", action="store_true", help="read kinds never measured, by hand only")
    ap.add_argument("--device", default="gpu:0")
    a = ap.parse_args()
    jobs = [(os.path.abspath(p) if isinstance(p, str) else p, m) for p, m in jobs_of(a)]
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    os.chdir(ROOT)                          # the weights and the listing's cache are found from here
    from listing import CACHE
    symbols = None
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as fh:
            symbols = json.load(fh)["symbols"]
    readers, failed, refused = None, 0, 0
    for i, (path, meta) in enumerate(jobs, 1):
        name, status = None, "read"
        try:
            if not isinstance(path, str) or not isinstance(meta, dict):
                raise ValueError("a job needs a pdf path and a meta object")
            if not os.path.isfile(path):
                raise ValueError("no such file")
            if readers is None and (a.unmeasured or not refusal(meta)):
                from read_pdf import ROUND1, Readers
                readers = Readers(ROUND1, a.device)
            r = read_one(path, meta, readers, symbols, a.unmeasured)
        except Refused as e:
            from sample_pairs import sha256_of
            r = {"schema": SCHEMA, "tool": VERSION, "profile": "disclosures", "sha256": sha256_of(path),
                 "measured": False, "reason": str(e)}
            status, refused = "refused", refused + 1
        except Exception as e:                      # noqa: BLE001  one file's failure is its own
            r = {"schema": SCHEMA, "tool": VERSION, "profile": "disclosures", "measured": False,
                 "reason": "failed: " + (type(e).__name__ + ": " + str(e).replace(str(path), "<pdf>"))[:200]}
            status, failed = "failed", failed + 1
            try:
                from sample_pairs import sha256_of
                r["sha256"] = sha256_of(path)
            except Exception:                        # noqa: BLE001
                pass
        name = (r["sha256"][:16] if r.get("sha256") else f"failed-{i}") + ".json"
        tmp = os.path.join(out, name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(r, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(out, name))
        print(json.dumps({"job": i, "out": name, "status": status}), flush=True)
    sys.exit(1 if failed else 2 if refused else 0)


if __name__ == "__main__":
    main()
