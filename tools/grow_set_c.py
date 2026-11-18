"""Grow set C: Vietnamese disclosure scan pages, to be corrected by hand.

Step 6 of METHOD.md. Set C is the only set of real scans with full truth: an
engine drafts each page, a person corrects it, a second read checks every number
and heading field. This tool only draws the pages; tools/draft_c.py drafts them
and tools/correct_c.py serves the page where they are corrected.

The HOSE + HNX listing is walked in an order drawn with a fixed seed. Per
company, in that order:
  1. Filings. Its disclosures, the disclosures profile only, less insider trades
     and personnel filings, in an order drawn at random, at most
     TRIES_PER_COMPANY opened; per filing its first file not marked English,
     unless the file name names a person (privacy.named_file).
  2. Pages. The file is labelled (label_pages.label_file). A page is a candidate
     when it is a scan (paper "yes"; an OCR layer laid on it is allowed), is not
     skipped (blank, member list) and holds MIN_LINES lines of text or more, so
     a cover or a signature page is not drawn.
  3. Choice. One candidate page per filing, drawn at random; at most PER_COMPANY
     filings per company. A file in another set (or already in this one) is not
     used.
  4. Training. With --training, a later draw also leaves out what the recogniser
     has seen: a file its lines came from, and a company its lines name, so the
     draw can judge it as set A and C could not.

More pages are drawn than the set keeps: a scan's text is unknown until it is
drafted, so a page with an identity field or a list of people is only found
then, and dropped (tools/correct_c.py). The next page in draw order takes its
place; "order" is the draw order.

Writes, under <work> (gitignored): companies.jsonl (one line per company walked,
so a stopped run resumes), filings.jsonl (each filing tried and what came of
it), pages.jsonl (labels). The set goes to <out>, committed: per page the
symbol, kind, PDF URL, SHA-256, page number, stamp label, draw order; no text.

Usage:
  python -I tools/grow_set_c.py --seed 13 --work data/set-c --out sets/disclosures-c.jsonl
  python -I tools/grow_set_c.py --seed 43 --work data/set-d --out sets/disclosures-d.jsonl \
    --training cache/train/data
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from grow_set_a import Labels  # noqa: E402
from grow_set_b import other_sets, wanted  # noqa: E402
from privacy import named_file  # noqa: E402
from records import write_jsonl  # noqa: E402
from sample_pairs import LISTING_URL, Client, article_items, disclosures, log, sha256_of  # noqa: E402
from sort_kinds import kind_of  # noqa: E402
from train_lines import core_name, fold  # noqa: E402

PER_COMPANY = 2          # 30 pages then come from 15 companies or more
TRIES_PER_COMPANY = 10   # most filings are scans: few need opening
MIN_LINES = 15           # lines of text at 40 dpi; a quarter of scan pages hold fewer


def seen_in_training(data: str) -> tuple[set[str], str]:
    """The files training lines came from, by the first 16 digits of their SHA-256,
    and the lines' text."""
    files, text = set(), []
    for name in ("train_list.txt", "val_list.txt", "hand_train.txt", "hand_val.txt"):
        for row in read_lines(os.path.join(data, name)):
            path, label = row.split("\t", 1)
            files.add(os.path.basename(path)[:16])
            text.append(label)
    return files, "\n".join(text)


def read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return fh.read().splitlines()


def candidates(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("paper") == "yes" and not r.get("skip") and not r.get("error")
            and (r.get("lines") or 0) >= MIN_LINES]


def try_filing(c, sym, item, cache, labels, used, seen_files) -> tuple[dict, dict | None]:
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
    if sha[:16] in seen_files:
        return {"result": "file in training"}, None
    pages = candidates(labels.of(path))
    what = {"result": f"{len(pages)} candidate pages"}
    if not pages:
        return what, None
    p = random.Random(sha).choice(pages)
    return what, {"symbol": sym, "kind": f["kind"], "pdf": f["pdf"], "sha256": sha,
                  "page": p["page"], "stamp": p["stamp"], "text_layer": p["text_layer"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", type=int, default=30, help="stop once this many are drawn")
    ap.add_argument("--feed-pages", type=int, default=3, help="20 items per page")
    ap.add_argument("--cache", default="cache/pdfs")
    ap.add_argument("--training", help="leave out what these training lines hold (tools/train_lines.py's folder)")
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
    chosen = [p for r in sorted(walked.values(), key=lambda r: r["order"]) for p in r["chosen"]]
    used = other_sets(a.out) | {p["sha256"] for p in chosen}
    seen_files, seen_text = seen_in_training(a.training) if a.training else (set(), "")
    seen_folded = f" {fold(seen_text)} "
    seen_codes = set(re.findall(r"\b[A-Z0-9]{3}\b", seen_text))
    labels = Labels(os.path.join(a.work, "pages.jsonl"))

    c = Client()
    universe = c.listing()
    names = {x["symbol"]: core_name(x["organName"]) for x in c.get(LISTING_URL).json()
             if x.get("symbol") and x.get("organName")}
    order = random.Random(a.seed).sample(universe, len(universe))
    log(f"listing: {len(universe)} stocks; {len(walked)} walked before, {len(chosen)} pages")
    for n, sym in enumerate(order, 1):
        if len(chosen) >= a.pages:
            break
        if sym in walked:
            continue
        rec = {"symbol": sym, "order": n}
        if sym in seen_codes or (names.get(sym) and f" {names[sym]} " in seen_folded):
            log(f"[{n}/{len(order)}] {sym}: named in training, left out")
            continue
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
                what, page = try_filing(c, sym, item, a.cache, labels, used, seen_files)
            except Exception as e:
                what, page = {"result": f"failed ({type(e).__name__})"}, None
            with open(filings_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"symbol": sym, "article": item.get("URL"), **what},
                                    ensure_ascii=False) + "\n")
            if page:
                rec["chosen"].append(page)
                used.add(page["sha256"])
        rec["chosen"] = rec["chosen"][:max(0, a.pages - len(chosen))]
        chosen += rec["chosen"]
        log(f"[{n}/{len(order)}] {sym}: {len(rec['chosen'])} pages from {len(todo)} filings; "
            f"drawn {len(chosen)}")
        walked[sym] = rec
        with open(walked_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        write_jsonl(a.out, [{**p, "order": i} for i, p in enumerate(chosen, 1)])

    write_jsonl(a.out, [{**p, "order": i} for i, p in enumerate(chosen, 1)])
    print(json.dumps({"companies_walked": len(walked), "pages": len(chosen),
                      "companies": len({p["symbol"] for p in chosen}),
                      "by_kind": {k: sum(p["kind"] == k for p in chosen)
                                  for k in sorted({p["kind"] for p in chosen})},
                      "stamped": sum(p["stamp"] == "yes" for p in chosen)}, indent=2))


if __name__ == "__main__":
    main()
