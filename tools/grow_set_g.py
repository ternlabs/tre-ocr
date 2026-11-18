"""Draw set G: whole disclosure files, read by the tool, for the listing's test.

METHOD.md, "The company check (set G)". The HOSE + HNX listing is walked in
an order drawn with a fixed seed. Per company, in that order, its disclosures (the
disclosures profile, less insider trades and personnel filings) in an order drawn at
random; per filing its first file not marked English, unless the file name names a
person (privacy.named_file), and not a file in another set. At most PER_COMPANY files
per company, TRIES_PER_COMPANY filings opened.

Each file is read by the tool (tools/read_pdf.py) with its symbol. A company gives a unit
when one of its documents has a filled company that matches the listing on a page read
by OCR (listing.read_matches). Drawing stops once UNITS companies have one, or at
MAX_FILES files.

Writes, under <work> (gitignored): companies.jsonl (one line per company walked, so a
stopped run resumes), filings.jsonl (each filing tried and what came of it), drafts/
(the tool's drafts and page images). The set goes to <out>, committed: per file the
symbol, kind, PDF URL, SHA-256, draw order, and how many documents matched; no text.

Usage:
  .venv-paddle/bin/python -I tools/grow_set_g.py --seed 46 --work data/set-g --out sets/disclosures-g.jsonl
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from grow_set_b import other_sets, wanted  # noqa: E402
from listing import listing, read_matches  # noqa: E402
from privacy import named_file  # noqa: E402
from read_pdf import ROUND1, Readers, draft  # noqa: E402
from records import write_jsonl  # noqa: E402
from sample_pairs import Client, article_items, disclosures, log, sha256_of  # noqa: E402
from sort_kinds import kind_of  # noqa: E402

PER_COMPANY = 2
TRIES_PER_COMPANY = 10
UNITS = 70                # companies with a match on a read page: 60 test, 10 decoys
MAX_FILES = 400           # files read at most, should units run short


def matched(d: dict) -> int:
    return sum(bool((f.get("company") or {}).get("listing")) for f in d["documents"])


def units(chosen: list[dict]) -> int:
    return len({f["symbol"] for f in chosen if f["matched_read"]})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feed-pages", type=int, default=3, help="20 items per page")
    ap.add_argument("--cache", default="cache/pdfs")
    ap.add_argument("--device", default="gpu:0")
    ap.add_argument("--units", type=int, default=UNITS, help="more after a void check: 10 per round")
    a = ap.parse_args()
    drafts = os.path.join(a.work, "drafts")
    os.makedirs(drafts, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    walked_path = os.path.join(a.work, "companies.jsonl")
    filings_path = os.path.join(a.work, "filings.jsonl")

    walked = {}
    if os.path.exists(walked_path):
        with open(walked_path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                walked[r["symbol"]] = r
    chosen = [f for r in sorted(walked.values(), key=lambda r: r["order"]) for f in r["chosen"]]
    for f in chosen:                            # drawn before the amendment: from its draft
        if "matched_read" not in f:
            with open(os.path.join(drafts, f["sha256"][:16] + ".draft.json"), encoding="utf-8") as fh:
                f["matched_read"] = len(read_matches(json.load(fh)))
    used = other_sets(a.out) | {f["sha256"] for f in chosen}
    names = listing()
    readers = Readers(ROUND1, a.device)

    c = Client()
    universe = c.listing()
    order = random.Random(a.seed).sample(universe, len(universe))
    log(f"listing: {len(universe)} stocks; {len(walked)} walked before, {len(chosen)} files")
    for n, sym in enumerate(order, 1):
        if units(chosen) >= a.units or len(chosen) >= MAX_FILES:
            break
        if sym in walked:
            continue
        if sym not in names:
            log(f"[{n}/{len(order)}] {sym}: not in the kept listing")
            continue
        try:
            disc = disclosures(c.feed(sym, a.feed_pages))
        except Exception as e:
            log(f"[{n}/{len(order)}] {sym}: feed failed ({type(e).__name__})")
            continue                            # not recorded: a later run tries again
        rng = random.Random(f"{a.seed}:{sym}")
        todo = [i for i in disc if wanted(kind_of((i.get("Title") or "").strip()))]
        rec = {"symbol": sym, "order": n, "filings": len(todo), "chosen": []}
        for tries, item in enumerate(rng.sample(todo, len(todo)), 1):
            if len(rec["chosen"]) >= PER_COMPANY or tries > TRIES_PER_COMPANY:
                break
            what, got = {}, None
            try:
                files = [it for it in article_items(c, sym, item) if it["lang"] != "en"]
                if not files:
                    what = {"result": "no Vietnamese file"}
                elif named_file(files[0]["file"]):
                    what = {"result": "file name names a person"}
                else:
                    f = files[0]
                    path, why = c.download(f["pdf"], a.cache)
                    sha = path and sha256_of(path)
                    if not path:
                        what = {"result": why}
                    elif sha in used:
                        what = {"result": "file in a set"}
                    else:
                        d = draft(path, drafts, readers, names[sym])
                        with open(os.path.join(drafts, sha[:16] + ".draft.json"), "w", encoding="utf-8") as fh:
                            json.dump(d, fh, ensure_ascii=False, indent=1)
                        got = {"symbol": sym, "kind": f["kind"], "pdf": f["pdf"], "sha256": sha,
                               "pages": len(d["pages"]), "documents": len(d["documents"]),
                               "matched": matched(d), "matched_read": len(read_matches(d))}
                        what = {"result": f"read: {got['matched']} of {got['documents']} documents matched"}
            except Exception as e:
                what = {"result": f"failed ({type(e).__name__})", "error": str(e)[:200]}
            with open(filings_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"symbol": sym, "article": item.get("URL"), **what},
                                    ensure_ascii=False) + "\n")
            if got:
                rec["chosen"].append(got)
                used.add(got["sha256"])
        chosen += rec["chosen"]
        log(f"[{n}/{len(order)}] {sym}: {len(rec['chosen'])} files; "
            f"{len(chosen)} files, {units(chosen)} units")
        walked[sym] = rec
        with open(walked_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        write_jsonl(a.out, [{**f, "order": i} for i, f in enumerate(chosen, 1)])

    write_jsonl(a.out, [{**f, "order": i} for i, f in enumerate(chosen, 1)])
    print(json.dumps({"companies_walked": len(walked), "files": len(chosen),
                      "documents": sum(f["documents"] for f in chosen),
                      "matched": sum(f["matched"] for f in chosen),
                      "matched_read": sum(f["matched_read"] for f in chosen), "units": units(chosen)}, indent=2))


if __name__ == "__main__":
    main()
