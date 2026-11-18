"""Grow set B: Vietnamese disclosure pages with a sound text layer.

Step 5 of METHOD.md. Set B scores tone marks: the page is rendered to an image
(tools/truth_b.py) and its own text layer is the truth, exact to the mark.
Typed filings are scarce (most are scans), so many companies are walked: the
HOSE + HNX listing, in an order drawn with a fixed seed, until about 100 pages
are found.

Per company, in that order:
  1. Filings. Its disclosures in the feed, the disclosures profile only, less
     insider trades and personnel filings (their titles and file names name
     people), in an order drawn at random. Each filing's article is opened as
     its turn comes; its first attachment not marked English is the file tried,
     unless its name holds an honorific ("Mr", "Bà", ...), marks an insider's
     trade or ends in a full name (privacy.named_file).
  2. Pages. The file is downloaded. A file whose every page of much text
     lies on a full-page image (a scan with an OCR layer) is passed over
     before the slow labelling; the rest are labelled (label_pages.label_file). A
     page is a candidate when it has its own text layer and is typed (paper
     "no": not OCR on a scan), is not skipped (blank, member list) and its
     text, with the signature box removed:
       - holds MIN_WORDS words or more;
       - reads as Vietnamese: most words are syllables (all-capital
         abbreviations left out) and most syllables carry a mark, so an English
         page or a text layer without marks is left out;
       - is not broken: at most BROKEN_SHARE of its words carry a mark yet are
         no syllable, and none holds a letter outside the Vietnamese alphabet
         (a legacy font encoding, TCVN3 or VNI, read as Latin-1: "Thêi" for
         "Thời", "cæ phÇn" for "cổ phần");
       - holds no identity field (ID, passport, date of birth, residence),
         filled or not: a table of insiders keeps the values apart from the
         labels, where no pattern pairs them (privacy.identity).
     A filing whose pages read as a court ruling is left out whole. The text is
     checked in NFC, so a text layer that keeps its marks apart is read alike.
  3. Choice. One candidate page per filing, drawn at random; at most
     PER_COMPANY filings per company. A file whose SHA-256 is in another set
     (or already in this one) is not used.

Writes, under <work> (gitignored): companies.jsonl (one line per company
walked, so a stopped run resumes), filings.jsonl (each filing tried and what
came of it), pages.jsonl (labels of the files labelled). The set goes to
<out>, committed, rewritten after each company: per page the symbol, kind, PDF
URL, SHA-256, page number, stamp label and the text checks; no text.

Usage:
  python -I tools/grow_set_b.py --seed 10 --work data/set-b --out sets/disclosures-b.jsonl
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from grow_set_a import LEFT_OUT, SURELY_TEXT, Labels, raw_chars  # noqa: E402
from label_pages import image_list, kept_text, page_images, read_pages, signature_boxes  # noqa: E402
from privacy import court_ruling, identity, named_file  # noqa: E402
from records import write_jsonl  # noqa: E402
from sample_pairs import (PROFILE, Client, article_items,  # noqa: E402
                          disclosures, log, sha256_of)
from sort_kinds import kind_of  # noqa: E402
from syllables import normal, page_check  # noqa: E402

PER_COMPANY = 3          # 100 pages then come from 34 companies or more
TRIES_PER_COMPANY = 20   # filings opened at most per company; most are scans
MIN_WORDS = 50
# Measured on the typed pages labelled in steps 3 and 4 (about 5,800 read as
# Vietnamese): the shares below sit in the gaps between kinds of page.
SYLLABLE_SHARE = 0.5     # Vietnamese: 0.9 and up; half in English: 0.45 to 0.75 (lost
                         # below 0.5); English: under 0.2
MARKED_SHARE = 0.5       # Vietnamese: 0.65 and up; a layer without marks: under 0.2
BROKEN_SHARE = 0.005     # sound pages mostly read 0; 1.5% of them read more and are lost:
                         # units (Đvt, vnđ), words run together, highland names spelt
                         # without a tone (Đăk, Krông). A layer that cuts words ("Điệ u")
                         # or a wrong glyph map with no foreign letter ("PhiÕu"): 3% and up


def wanted(kind: str) -> bool:
    return PROFILE[kind] == "disclosures" and kind not in LEFT_OUT


def candidates(path: str, rows: list[dict]) -> tuple[list[dict], dict]:
    """The file's candidate pages, with their text checks, and the counts of
    pages left out by each check."""
    text = read_pages(path)
    if not text:
        return [], {"no page read": 1}
    kept = [normal(kept_text(pg["blocks"], signature_boxes(pg["blocks"]))) for pg in text]
    if court_ruling(kept):
        return [], {"court ruling": 1}
    out, why = [], {}
    for r in rows:
        if not r["text_layer"] or r.get("paper") != "no" or r.get("skip") or r.get("error"):
            continue
        t = kept[r["page"] - 1]
        c = page_check(t)
        reason = ("few words" if c["words"] < MIN_WORDS else
                  "not Vietnamese" if c["syllables"] < SYLLABLE_SHARE or c["marked"] < MARKED_SHARE else
                  "broken text" if c["broken"] > BROKEN_SHARE or c["foreign"] else
                  "identity field" if identity(t) else None)
        if reason:
            why[reason] = why.get(reason, 0) + 1
        else:
            out.append({"page": r["page"], "stamp": r["stamp"], **c})
    return out, why


def try_filing(c, sym, item, cache, labels, used) -> tuple[dict, dict | None]:
    """What came of one filing, and the page drawn from it."""
    files = [it for it in article_items(c, sym, item) if it["lang"] != "en"]
    if not files:
        return {"result": "no Vietnamese file"}, None
    f = files[0]
    if named_file(f["file"]):
        return {"result": "file name names a person"}, None
    path, why = c.download(f["pdf"], cache)
    if not path:
        return {"result": why}, None
    sha = sha256_of(path)
    if sha in used:
        return {"result": "file in a set"}, None
    raw = raw_chars(path)
    if max(raw or [0]) < SURELY_TEXT:                  # a scan's signature box can hold 400
        return {"result": "no text page"}, None
    full = page_images(image_list(path), read_pages(path))
    if all(n in full for n, k in enumerate(raw, 1) if k >= SURELY_TEXT):
        return {"result": "text only on scans"}, None  # OCR laid on a scan: no truth
    pages, left_out = candidates(path, labels.of(path))
    what = {"result": f"{len(pages)} candidate pages", **({"left_out": left_out} if left_out else {})}
    if not pages:
        return what, None
    p = random.Random(sha).choice(pages)
    return what, {"symbol": sym, "kind": f["kind"], "pdf": f["pdf"], "sha256": sha, **p}


def write_set(path: str, chosen: list[dict]):
    write_jsonl(path, sorted(chosen, key=lambda p: (p["symbol"], p["pdf"], p["page"])))


def other_sets(out: str) -> set[str]:
    """SHA-256 of every file in the other committed sets."""
    used = set()
    for path in glob.glob(os.path.join(os.path.dirname(out) or ".", "*.jsonl")):
        if os.path.abspath(path) == os.path.abspath(out):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                used |= {r[k]["sha256"] for k in ("vi", "en") if k in r} | ({r["sha256"]} if "sha256" in r else set())
    return used


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", type=int, default=100, help="stop once the set holds this many")
    ap.add_argument("--feed-pages", type=int, default=3, help="20 items per page")
    ap.add_argument("--cache", default="cache/pdfs")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    walked_path = os.path.join(a.work, "companies.jsonl")
    filings_path = os.path.join(a.work, "filings.jsonl")

    walked = {}
    if os.path.exists(walked_path):
        with open(walked_path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                walked[r["symbol"]] = r
    chosen = [p for r in walked.values() for p in r.get("chosen", [])]
    if os.path.exists(a.out):               # pages taken out of the set stay out
        with open(a.out, encoding="utf-8") as fh:
            kept = {(r["sha256"], r["page"]) for r in map(json.loads, fh)}
        chosen = [p for p in chosen if (p["sha256"], p["page"]) in kept]
    used = other_sets(a.out) | {p["sha256"] for r in walked.values() for p in r.get("chosen", [])}
    labels = Labels(os.path.join(a.work, "pages.jsonl"))

    c = Client()
    universe = c.listing()
    order = random.Random(a.seed).sample(universe, len(universe))
    log(f"listing: {len(universe)} stocks; {len(walked)} walked before, {len(chosen)} pages")
    for n, sym in enumerate(order, 1):
        if len(chosen) >= a.pages:
            break
        if sym in walked:
            continue
        rec = {"symbol": sym, "order": n}
        try:
            disc = disclosures(c.feed(sym, a.feed_pages))
        except Exception as e:
            log(f"[{n}/{len(order)}] {sym}: feed failed ({type(e).__name__})")
            continue                            # not recorded: a later run tries again
        rng = random.Random(f"{a.seed}:{sym}")
        todo = [i for i in disc if wanted(kind_of((i.get("Title") or "").strip()))]
        rec["filings"] = len(todo)
        rec["chosen"] = []
        for tries, item in enumerate(rng.sample(todo, len(todo)), 1):
            if len(rec["chosen"]) >= PER_COMPANY or tries > TRIES_PER_COMPANY:
                break
            try:
                what, page = try_filing(c, sym, item, a.cache, labels, used)
            except Exception as e:
                what, page = {"result": f"failed ({type(e).__name__})"}, None
            with open(filings_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"symbol": sym, "article": item.get("URL"), **what},
                                    ensure_ascii=False) + "\n")
            if page:
                rec["chosen"].append(page)
                used.add(page["sha256"])
        chosen += rec["chosen"]
        rec["result"] = f"{len(rec['chosen'])} pages from {len(todo)} filings"
        log(f"[{n}/{len(order)}] {sym}: {rec['result']}; set holds {len(chosen)}")
        walked[sym] = rec
        with open(walked_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        write_set(a.out, chosen)

    write_set(a.out, chosen)
    print(json.dumps({"companies_walked": len(walked), "pages": len(chosen),
                      "companies_in_set": len({p["symbol"] for p in chosen}),
                      "by_kind": {k: sum(p["kind"] == k for p in chosen)
                                  for k in sorted({p["kind"] for p in chosen})},
                      "stamped": sum(p["stamp"] == "yes" for p in chosen)}, indent=2))


if __name__ == "__main__":
    main()
