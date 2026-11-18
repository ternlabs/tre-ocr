"""Draw set H or I: financial statements paired with the later filing that repeats their figures.

The sets of profile 2 (METHOD.md, "Financial statements"): H for development, I the test. The HOSE + HNX listing
is walked in an order drawn with a fixed seed; banks, securities firms and insurers are
passed over by their listed name (their forms differ). Per company, its statement filings
of fiscal BASE_YEAR and BASE_YEAR + 1 are read as periods (tools/fs_periods.py) and
paired:

- **flow:** a quarter (1 to 3) or half year of BASE_YEAR, with the same period and scope
  a year on. The later one's prior-year column repeats the income statement (B02) and
  cash flow (B03).
- **balance:** a statement at the end of BASE_YEAR (the year, or quarter 4), with the
  first statement of the same scope in the next year. Its opening column repeats the
  balance sheet (B01).

One pair is drawn at random per company. Per filing, its longest file not marked English
is used, unless its name names a person (privacy.named_file), it is in another set, or
its pages read as mostly English; then the next longest.
Both files are read by the tool (tools/read_pdf.py), so their drafts keep each line's
box, and their sideways statement pages are read again turned (tools/fs_turn.py); the drafts are development data for the page finder and the row reader.

Writes, under <work> (gitignored): companies.jsonl (one line per company walked, so a
stopped run resumes), drafts/ (drafts and page images). The set goes to <out>, committed:
one line per file, with the pair, its role, the period, the PDF URL and SHA-256; no text.

With --apart, the companies of another set of statements are passed over: the test holds no
company the development set has.

Usage:
  .venv-paddle/bin/python -I tools/grow_set_h.py --seed 47 --work data/set-h --out sets/statements-h.jsonl
  .venv-paddle/bin/python -I tools/grow_set_h.py --seed 48 --work data/set-i --out sets/statements-i.jsonl \
      --apart sets/statements-h.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from fs_periods import key, period_of  # noqa: E402
from fs_turn import upright  # noqa: E402
from grow_set_b import other_sets  # noqa: E402
from listing import listing  # noqa: E402
from privacy import named_file  # noqa: E402
from records import write_jsonl  # noqa: E402
from sample_pairs import DATE_RE, Client, article_items, disclosures, log, pdf_verdict, sha256_of  # noqa: E402
from sort_kinds import fold, kind_of  # noqa: E402

COMPANIES = 30
BASE_YEAR = 2025
FINANCIAL = re.compile(r"\b(ngan hang|chung khoan|bao hiem|cong ty tai chinh)\b")
CODE_RE = re.compile(r"^\s*(\d{2,3})\s*$")


def financial(names: dict) -> bool:
    """A bank, securities firm, insurer or finance company, by its listed name."""
    return bool(FINANCIAL.search(fold(names.get("organName") or "")))


def pairs_of(items: list[dict]) -> list[dict]:
    """Every flow and balance pair among one company's statement filings."""
    by = {}
    for i in items:
        p = period_of((i.get("Title") or "").strip(), i["published"])
        if p and p["year"] in (BASE_YEAR, BASE_YEAR + 1):
            by.setdefault(key(p), []).append((p, i))
    out = []
    for (per, n, year, scope), got in by.items():
        if year != BASE_YEAR:
            continue
        base = got[0]
        if per in ("quarter", "half") and not (per == "quarter" and n == 4):
            later = by.get((per, n, year + 1, scope))
            if later:
                out.append({"kind": "flow", "base": base, "later": later[0]})
        if per == "year" or (per == "quarter" and n == 4):
            nxt = sorted((g for k, gs in by.items() for g in gs if k[2] == year + 1 and k[3] == scope),
                         key=lambda g: g[1]["published"])
            if nxt:
                out.append({"kind": "balance", "base": base, "later": nxt[0]})
    return sorted(out, key=lambda x: (x["kind"], x["base"][1]["published"], x["later"][1]["published"]))


def scope_of_file(name: str) -> str | None:
    """consolidated / separate from an attachment's file name, as fs_periods does titles."""
    t = fold(name.rsplit(".", 1)[0])
    return ("consolidated" if re.search(r"\b(hop ?nhat|hn|hopnhat)\b", t)
            else "separate" if re.search(r"\b(rieng|rieng ?le|me|ctme|cong ty me)\b", t) else None)


def statement_files(c, sym: str, item: dict, scope: str | None, cache: str) -> list[tuple[dict, str]]:
    """The filing's candidate statements, longest first: its files not marked English, of
    the wanted scope.

    A filing often attaches a cover letter, and sometimes both scopes; the statement
    itself is the longest file. A file of the other scope (by its name) is passed over."""
    files = [it for it in article_items(c, sym, item) if it["lang"] != "en"]
    if not files:
        raise LookupError("no Vietnamese file")
    if scope:
        files = [f for f in files if scope_of_file(f["file"]) in (scope, None)] or files
    got = []
    for f in files:
        if named_file(f["file"]):
            continue
        path, why = c.download(f["pdf"], cache)
        if path:
            got.append((pdf_verdict(path).get("pages", 0), f, path))
    if not got:
        raise LookupError("no file to read (named, or not downloaded)")
    return [(f, path) for _, f, path in sorted(got, key=lambda g: -g[0])]


def mostly_english(d: dict) -> bool:
    """Most pages read as English: a copy whose name did not say so ("_ta", "cksta"), or a
    text layer whose font maps to the wrong letters. Out of scope either way."""
    shown = [p for p in d["pages"] if p["kind"] != "skipped"]
    return bool(shown) and sum("english" in p for p in shown) * 2 > len(shown)


def codes_read(d: dict) -> list[str]:
    """Line codes read as lines of their own, for the survey: which form is in use."""
    return sorted({m.group(1) for p in d["pages"] for ln in p["lines"]
                   if (m := CODE_RE.match(ln["text"]))}, key=int)


def symbols_of(paths: list[str]) -> set[str]:
    """The companies of the sets at paths."""
    found = set()
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            found |= {json.loads(line)["symbol"] for line in fh if line.strip()}
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--companies", type=int, default=COMPANIES)
    ap.add_argument("--apart", action="append", default=[], help="a set whose companies are passed over")
    ap.add_argument("--feed-pages", type=int, default=15, help="20 items per page")
    ap.add_argument("--cache", default="cache/pdfs")
    ap.add_argument("--device", default="gpu:0")
    a = ap.parse_args()
    drafts = os.path.join(a.work, "drafts")
    os.makedirs(drafts, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    walked_path = os.path.join(a.work, "companies.jsonl")

    walked = {}
    if os.path.exists(walked_path):
        with open(walked_path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                walked[r["symbol"]] = r
    rows = [f for r in sorted(walked.values(), key=lambda r: r["order"]) for f in r.get("files", [])]
    kept = lambda: sum(1 for r in walked.values() if r.get("files"))  # noqa: E731
    used = other_sets(a.out) | {f["sha256"] for f in rows}
    names = listing()
    apart = symbols_of(a.apart)
    from read_pdf import ROUND1, Readers, draft
    readers = Readers(ROUND1, a.device)

    c = Client()
    universe = c.listing()
    order = random.Random(a.seed).sample(universe, len(universe))
    log(f"listing: {len(universe)} stocks; {len(walked)} walked before, {kept()} kept")
    for n, sym in enumerate(order, 1):
        if kept() >= a.companies:
            break
        if sym in walked:
            continue
        rec = {"symbol": sym, "order": n}
        if sym not in names:
            rec["result"] = "not in the kept listing"
        elif sym in apart:
            rec["result"] = "in a set kept apart"
        elif financial(names[sym]):
            rec["result"] = "a financial firm's forms"
        else:
            try:
                disc = disclosures(c.feed(sym, a.feed_pages))
            except Exception as e:
                log(f"[{n}] {sym}: feed failed ({type(e).__name__})")
                continue                            # not recorded: a later run tries again
            for i in disc:
                m = DATE_RE.search(i.get("PublishTime") or "")
                i["published"] = (datetime.fromtimestamp(int(m.group(1)) / 1000, tz=timezone.utc)
                                  .strftime("%Y-%m-%d") if m else "")
            fs = [i for i in disc if kind_of((i.get("Title") or "").strip()) == "financial_statement"
                  and i["published"]]
            pairs = pairs_of(fs)
            rec.update({"statements": len(fs), "pairs": len(pairs)})
            if not pairs:
                rec["result"] = "no pair"
            else:
                pick = random.Random(f"{a.seed}:{sym}").choice(pairs)
                got = []
                try:
                    for role in ("base", "later"):
                        p, item = pick[role]
                        want = p["scope"] or (got[0]["scope"] if got else None)
                        for f, path in statement_files(c, sym, item, want, a.cache):
                            sha = sha256_of(path)
                            if sha in used:
                                continue
                            d = draft(path, drafts, readers, names[sym])
                            if not mostly_english(d):
                                break
                        else:
                            raise LookupError("no Vietnamese statement to read")
                        p = {**p, "scope": p["scope"] or scope_of_file(f["file"])}
                        upright(d, os.path.join(drafts, "images"), readers)
                        with open(os.path.join(drafts, sha[:16] + ".draft.json"), "w", encoding="utf-8") as fh:
                            json.dump(d, fh, ensure_ascii=False, indent=1)
                        got.append({"symbol": sym, "pair": pick["kind"], "role": role, **p,
                                    "published": item["published"][:10], "pdf": f["pdf"],
                                    "sha256": sha, "pages": len(d["pages"]), "codes": codes_read(d)})
                    rec.update({"result": f"kept: {pick['kind']} pair", "files": got})
                    used |= {g["sha256"] for g in got}
                    rows += got
                except LookupError as e:
                    rec["result"] = str(e)
                except Exception as e:
                    rec["result"] = f"failed ({type(e).__name__})"
                    rec["error"] = str(e)[:200]
        log(f"[{n}] {sym}: {rec['result']}; {kept() + bool(rec.get('files'))} kept")
        walked[sym] = rec
        with open(walked_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        write_jsonl(a.out, [{k: v for k, v in f.items() if k != "codes"} for f in rows])

    write_jsonl(a.out, [{k: v for k, v in f.items() if k != "codes"} for f in rows])
    print(json.dumps({"walked": len(walked), "kept": kept(), "files": len(rows),
                      "pairs": {k: sum(1 for f in rows if f["pair"] == k and f["role"] == "base")
                                for k in ("flow", "balance")}}, indent=2))


if __name__ == "__main__":
    main()
