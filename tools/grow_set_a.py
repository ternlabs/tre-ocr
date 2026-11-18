"""Grow set A: scanned Vietnamese pages whose English twin page has a text layer.

Step 3 of METHOD.md. Set A scores numbers: the English page's text says which
numbers its scanned Vietnamese page holds. Such pages are rare (most twins are
both scans), so the whole HOSE + HNX listing is walked, in an order drawn with
a fixed seed, until about 50 pages are found.

Per company, in that order:
  1. Survey. Its newest SURVEY_ARTICLES disclosures are opened; a company with
     no English attachment among them is passed over. A company that files in
     English does so for most filings, so a few articles show it.
  2. Pairs. All its disclosures in the feed are opened and paired VI/EN by
     article (sample_pairs.pair_items). Only the disclosures profile is kept,
     less insider trades and personnel filings: their titles and file names
     name people, and set A's file is committed. A filing of another kind
     whose file names hold an honorific (a resolution that elects "Mr ...")
     is left out too.
  3. Pages. Both files are downloaded and labelled (label_pages.label_file).
     Pages pair one to one: when the files have the same number of pages, page
     n with page n; otherwise inside the documents that agree in length, read
     from the start and from the end of both files. A pair is kept when the
     Vietnamese page is a scan and the English page has its own text layer
     (not OCR on paper), neither is skipped, and the English page, with its
     signature box removed, holds a number of 3+ digits and no filled identity
     field (ID, passport, date of birth): such numbers are a person's, and a
     filing may hide them on the page yet keep them in the text layer. The
     English page must read as English (not the Vietnamese scan's own OCR, not
     a list of names), and a filing that is a court ruling is left out whole.
     The pages found are then aligned and pruned (tools/align_pages.py).
  4. Choice. At most PER_COMPANY pages per company, spread over its filings:
     one page per filing in turn, each drawn at random. A file whose SHA-256 is
     already in the set (the same PDF under two filings) is not used again.

Writes, under <work> (gitignored): companies.jsonl (one line per company
walked, so a stopped run resumes), pairs.jsonl (each filing pair tried and
what came of it), pages.jsonl (labels of the files labelled). The set itself
goes to <out>, committed, rewritten after each company; a page walked before
but no longer in <out> (pruned by tools/align_pages.py) is not put back: per page pair the symbol, kind, both PDF URLs,
SHA-256s and page numbers; no text.

Usage:
  python -I tools/grow_set_a.py --seed 7 --work data/set-a --out sets/disclosures-a.jsonl
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from label_pages import kept_text, label_file, read_pages, signature_boxes  # noqa: E402
from privacy import IDENTITY_RE, court_ruling, named_file  # noqa: E402
from records import numbers, write_jsonl  # noqa: E402
from sample_pairs import (PROFILE, SCAN_CHARS_PER_PAGE, Client, article_items,  # noqa: E402
                          disclosures, log, pair_items, sha256_of)

SURVEY_ARTICLES = 8      # in the step 0 survey, every company with 3+ English
                         # filings showed one among its newest 5 articles
PER_COMPANY = 5
LEFT_OUT = {"insider_trade", "personnel"}
# The quick look counts the signature box (200-400 characters) as text, so only
# a page with more than any box holds is surely not a scan.
SURELY_TEXT = 700
# English prose is about a tenth or more of these words; a table in English
# still has some. An "English" text layer below ENGLISH_SHARE is Vietnamese
# without tone marks (OCR of the Vietnamese scan) or a list of names.
FUNCTION_WORDS = {"the", "of", "and", "to", "in", "for", "is", "on", "with", "by", "as", "at",
                  "be", "from", "this", "that", "are", "or", "an", "a"}
ENGLISH_SHARE, ENGLISH_MIN_WORDS = 0.05, 50


def english(text: str) -> bool:
    words = [w.lower() for w in re.findall(r"[^\W\d_]+", text)]
    return (len(words) < ENGLISH_MIN_WORDS
            or sum(w in FUNCTION_WORDS for w in words) / len(words) >= ENGLISH_SHARE)


def raw_chars(path: str) -> list[int]:
    """Characters per page, signature box left in: a quick look before labelling."""
    out = subprocess.run(["pdftotext", "-q", "-enc", "UTF-8", path, "-"],
                         capture_output=True, timeout=300).stdout.decode("utf-8", "replace")
    return [len(re.sub(r"\s", "", p)) for p in out.split("\f")[:-1]]


def aligned(vi: list[dict], en: list[dict]) -> tuple[list[tuple[dict, dict]], str]:
    """Page pairs, and how they were paired ("pages", "documents" or "none")."""
    if len(vi) == len(en):
        return list(zip(vi, en)), "pages"

    def docs(rows):
        out = []
        for r in rows:
            if out and out[-1][0]["doc"] == r["doc"]:
                out[-1].append(r)
            else:
                out.append([r])
        return out

    dv, de = docs(vi), docs(en)
    out = []
    i = 0
    while i < min(len(dv), len(de)) and len(dv[i]) == len(de[i]):
        out += zip(dv[i], de[i])
        i += 1
    j = 1
    while j <= min(len(dv), len(de)) - i and len(dv[-j]) == len(de[-j]):
        out += zip(dv[-j], de[-j])
        j += 1
    return out, "documents" if out else "none"


def usable(v: dict, e: dict) -> bool:
    return (not v["text_layer"] and not v.get("skip") and not v.get("error")
            and e["text_layer"] and e.get("paper") != "yes"
            and not e.get("skip") and not e.get("error"))


class Labels:
    """Page labels per cached file, kept in <work>/pages.jsonl across runs."""

    def __init__(self, path):
        self.path, self.rows = path, {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    self.rows.setdefault(r["file"], []).append(r)

    def of(self, path):
        name = os.path.basename(path)
        if name not in self.rows:
            rows = label_file(path)
            with open(self.path, "a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            self.rows[name] = rows
        return self.rows[name]


def try_pair(c, d, cache, labels, used) -> tuple[dict, list[dict]]:
    """What came of one filing pair, and its candidate page pairs."""
    if any(named_file(d[lang]["file"]) for lang in ("vi", "en")):
        return {"result": "file name names a person"}, []
    got = {}
    for lang in ("vi", "en"):
        path, why = c.download(d[lang]["pdf"], cache)
        if not path:
            return {"result": f"{lang} {why}"}, []
        got[lang] = (path, sha256_of(path))
    if got["vi"][1] in used or got["en"][1] in used:
        return {"result": "file already in the set"}, []
    if min(raw_chars(got["vi"][0]) or [0]) >= SURELY_TEXT:
        return {"result": "vi has no scan page"}, []
    if max(raw_chars(got["en"][0]) or [0]) < SCAN_CHARS_PER_PAGE:
        return {"result": "en has no text page"}, []
    vi, en = labels.of(got["vi"][0]), labels.of(got["en"][0])
    pairs, how = aligned(vi, en)
    keep = [(v, e) for v, e in pairs if usable(v, e)]
    if not keep:
        return {"result": f"no scan/text page pair (paired by {how})"}, []
    text = read_pages(got["en"][0])
    if court_ruling([kept_text(pg["blocks"], []) for pg in text]):
        return {"result": "court ruling"}, []
    out, named, foreign = [], 0, 0
    for v, e in keep:
        pg = text[e["page"] - 1]
        kept = kept_text(pg["blocks"], signature_boxes(pg["blocks"]))
        if IDENTITY_RE.search(kept):
            named += 1
            continue
        if not english(kept):
            foreign += 1
            continue
        n = len(numbers(kept))
        if n:
            out.append({"vi": {"pdf": d["vi"]["pdf"], "sha256": got["vi"][1], "page": v["page"],
                               "stamp": v["stamp"]},
                        "en": {"pdf": d["en"]["pdf"], "sha256": got["en"][1], "page": e["page"]},
                        "paired_by": how, "en_numbers": n})
    what = {"result": f"{len(out)} page pairs (paired by {how})"}
    if named:
        what["identity_pages"] = named
    if foreign:
        what["not_english_pages"] = foreign
    return what, out


def key(p: dict) -> tuple[str, int]:
    return p["vi"]["sha256"], p["vi"]["page"]


def write_set(path: str, chosen: list[dict]):
    """The set file, rewritten after each company so it is always current: a later
    run keeps only the walked pages still in it."""
    write_jsonl(path, ({k: p[k] for k in ("symbol", "kind", "vi", "en", "paired_by", "en_numbers")}
                       for p in sorted(chosen, key=lambda p: (p["symbol"], p["vi"]["pdf"], p["vi"]["page"]))))


def choose(by_filing: list[list[dict]], rng, k: int) -> list[dict]:
    """Up to k pages, one per filing in turn, each drawn at random."""
    pools = [rng.sample(p, len(p)) for p in by_filing]
    out = []
    while len(out) < k and any(pools):
        for p in pools:
            if p and len(out) < k:
                out.append(p.pop())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", type=int, default=50, help="stop once the set holds this many")
    ap.add_argument("--companies", type=int, default=0, help="walk at most this many (0: all)")
    ap.add_argument("--feed-pages", type=int, default=3, help="20 items per page")
    ap.add_argument("--cache", default="cache/pdfs")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    walked_path = os.path.join(a.work, "companies.jsonl")
    pairs_path = os.path.join(a.work, "pairs.jsonl")

    walked = {}
    if os.path.exists(walked_path):
        with open(walked_path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                walked[r["symbol"]] = r
    chosen = [p for r in walked.values() for p in r.get("chosen", [])]
    used = {p[s]["sha256"] for p in chosen for s in ("vi", "en")}
    if os.path.exists(a.out):               # pages pruned from the set stay out
        with open(a.out, encoding="utf-8") as fh:
            kept = {key(json.loads(line)) for line in fh}
        chosen = [p for p in chosen if key(p) in kept]
    labels = Labels(os.path.join(a.work, "pages.jsonl"))

    c = Client()
    universe = c.listing()
    order = random.Random(a.seed).sample(universe, len(universe))
    log(f"listing: {len(universe)} stocks; {len(walked)} walked before, {len(chosen)} pages")
    for n, sym in enumerate(order, 1):
        if len(chosen) >= a.pages or a.companies and n > a.companies:
            break
        if sym in walked:
            continue
        rec = {"symbol": sym, "order": n}
        try:
            disc = disclosures(c.feed(sym, a.feed_pages))
        except Exception as e:
            rec["result"] = f"feed failed ({type(e).__name__})"
            disc = None
        if disc is not None:
            items = []
            for i in disc[:SURVEY_ARTICLES]:
                items.extend(article_items(c, sym, i))
            rec["disclosures"] = len(disc)
            rec["english"] = any(it["lang"] == "en" for it in items)
            if not rec["english"]:
                rec["result"] = "files no English"
            else:
                for i in disc[SURVEY_ARTICLES:]:
                    items.extend(article_items(c, sym, i))
                rng = random.Random(f"{a.seed}:{sym}")
                todo = [d for _, _, d in pair_items(items)
                        if PROFILE[d["vi"]["kind"]] == "disclosures" and d["vi"]["kind"] not in LEFT_OUT]
                rec["pairs"] = len(todo)
                by_filing = []
                for d in rng.sample(todo, len(todo)):
                    if len(by_filing) >= PER_COMPANY:
                        break
                    try:
                        what, found = try_pair(c, d, a.cache, labels, used)
                    except Exception as e:
                        what, found = {"result": f"failed ({type(e).__name__})"}, []
                    with open(pairs_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"symbol": sym, "kind": d["vi"]["kind"],
                                             "vi": d["vi"]["pdf"], "en": d["en"]["pdf"], **what},
                                            ensure_ascii=False) + "\n")
                    if found:
                        for p in found:
                            p.update(symbol=sym, kind=d["vi"]["kind"])
                        by_filing.append(found)
                        used |= {found[0]["vi"]["sha256"], found[0]["en"]["sha256"]}
                rec["chosen"] = choose(by_filing, rng, PER_COMPANY)
                chosen += rec["chosen"]
                rec["result"] = f"{len(rec['chosen'])} pages from {len(todo)} pairs"
        log(f"[{n}/{len(order)}] {sym}: {rec['result']}; set holds {len(chosen)}")
        if disc is None:
            continue                            # not recorded: a later run tries again
        walked[sym] = rec
        with open(walked_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        write_set(a.out, chosen)

    write_set(a.out, chosen)
    rows = list(walked.values())
    print(json.dumps({"companies_walked": len(rows),
                      "files_english": sum(bool(r.get("english")) for r in rows),
                      "pages": len(chosen),
                      "companies_in_set": len({p["symbol"] for p in chosen}),
                      "by_kind": {k: sum(p["kind"] == k for p in chosen)
                                  for k in sorted({p["kind"] for p in chosen})}}, indent=2))


if __name__ == "__main__":
    main()
