"""The listing's name for a filing's symbol: a second source for the company field.

The heading's company is read from the page; the listing (Vietcap's symbol list, the
one tools/sample_pairs.py draws from) names each listed company too. The two come from
different places, so a value the rule read that matches the listing is unlikely to be
misread. A mismatch is no error: the filer is not always the author (an exchange's
notice, a shareholder's report, a parent's resolution), names change, and the listing
holds today's name only.

Names are compared as keys: NFC, case, spaces and punctuation dropped, marks kept, and
the usual short forms written out ("CTCP" is "công ty cổ phần").

The listing is fetched once and kept in cache/listing.json (gitignored), with its date.

Usage:
  python -I tools/listing.py c d e f        # build: checked companies against the listing
  python -I tools/listing.py c d e f --show # the same, every page that does not match
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from syllables import normal  # noqa: E402

CACHE = "cache/listing.json"
NAMES = ("organName", "organShortName", "enOrganName", "enOrganShortName")
SHORT = [   # whole words, in this order, on the cased-folded name
    (r"\btổng\s+cty\b", "tổng công ty"), (r"\bcty\b", "công ty"),
    (r"\bctcp\b", "công ty cổ phần"), (r"\bcông\s+ty\s+cp\b", "công ty cổ phần"),
    (r"\btmcp\b", "thương mại cổ phần"), (r"\btnhh\b", "trách nhiệm hữu hạn"), (r"\bjsc\b", "joint stock company"),
    (r"\bcorp\b", "corporation"), (r"\bco\b", "company"),
]


def key(name: str) -> str:
    s = re.sub(r"\s*\([^)]*\)\s*$", "", normal(name))     # "(ONE CORP)": a short name after it
    s = " ".join(s.casefold().replace("&", " và ").split())
    s = s.replace("-", " ")
    for pat, full in SHORT:
        s = re.sub(pat, full, s)
    return "".join(c for c in s if c.isalnum())


def listing(refresh: bool = False) -> dict:
    """symbol -> {organName, ...}, from the cache or fetched."""
    if not refresh and os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as fh:
            return json.load(fh)["symbols"]
    from sample_pairs import LISTING_URL, Client
    rows = Client().get(LISTING_URL, headers={"Referer": "https://trading.vietcap.com.vn/"}).json()
    symbols = {r["symbol"]: {k: r.get(k) for k in NAMES} for r in rows if r.get("type") == "STOCK"}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as fh:
        json.dump({"fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "symbols": symbols}, fh, ensure_ascii=False)
    return symbols


def matches(value: str, names: dict | None) -> bool:
    """The value is one of the listing's names for the symbol, as keys."""
    if not value or not names:
        return False
    v = key(value)
    return any(n and key(n) == v for n in names.values())


def read_matches(draft: dict) -> list[dict]:
    """A draft's documents whose company matches the listing on a page read by OCR."""
    kind = {p["page"]: p for p in draft["pages"]}
    return [d for d in draft["documents"]
            if (d.get("company") or {}).get("listing") and kind[d["company"]["page"]]["kind"] == "read"]


def main():
    from fields import extract
    from records import read_jsonl
    from score import SETS, load_truth
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--show", action="store_true", help="each page that does not match, for building")
    ap.add_argument("--refresh", action="store_true", help="fetch the listing again")
    a = ap.parse_args()
    names = listing(a.refresh)
    for s in a.sets:
        symbol = {(r["sha256"], r["page"]): r["symbol"] for r in read_jsonl(SETS[s][0])}
        truth, _ = load_truth(s, None)
        run = {(r["sha256"], r["page"]): r for r in read_jsonl(f"data/runs/two-readers-{s}.jsonl")}
        c = Counter()
        for k, t in truth.items():
            tv = (t.get("heading") or {}).get("company") or ""
            if not tv or k not in run:
                continue
            n = names.get(symbol.get(k))
            c["pages"] += 1
            c["symbol not listed"] += n is None
            ok = matches(tv, n)
            c["truth matches listing"] += ok
            v = extract(run[k]["text"].split("\n"), run[k]["unsure"])["company"]["value"]
            if v:
                right = key(v) == key(tv)
                c["rule filled"] += 1
                c["rule right"] += right
                if matches(v, n):
                    c["rule matches listing"] += 1
                    c["rule matches listing, wrong"] += not right
            if a.show and not ok:
                print(f"{s} {k[0][:8]}-p{k[1]} truth={tv!r} listing={n and n['organName']!r}")
        print(s, json.dumps(dict(sorted(c.items())), ensure_ascii=False))


if __name__ == "__main__":
    main()
