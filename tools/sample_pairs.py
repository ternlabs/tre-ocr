"""Find Vietnamese/English disclosure pairs to use as OCR ground truth.

A listed company often files the same disclosure twice, once in Vietnamese and
once in English. When one copy is a scan and the other has a text layer, the
text copy says what the scan should read, at least for the numbers. This script
finds such pairs and says how often each kind occurs. It does not OCR anything.

Sampling: companies are drawn at random (fixed seed) from every stock listed on
HOSE and HNX, so the set leans toward no one's interests and anyone can rebuild it.

Sources, both public and read with plain HTTP:
  - the listing: Vietcap's symbol list
  - disclosures: Vietstock's per-symbol news feed, and each item's article page,
    which links the attached PDF

Writes:
  - <out>/items.jsonl   one line per disclosure seen (not committed: it holds
                        attachment file names, and some name insiders)
  - <out>/pairs.jsonl   one line per VI/EN pair, with each side's text/scan verdict
                        and the filing's kind and profile (tools/sort_kinds.py)
  - PDFs go to <cache>, which is gitignored

Usage:
  python -I tools/sample_pairs.py --symbols 30 --seed 1 --out data/sample-01
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from sort_kinds import PROFILE, kind_of  # noqa: E402

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
LISTING_URL = "https://trading.vietcap.com.vn/api/price/symbols/getAll"
FEED_BASE = "https://finance.vietstock.vn"
ARTICLE_BASE = "https://vietstock.vn"      # the finance.* host soft-404s articles
DISCLOSURE_SOURCES = {"HNX", "HOSE", "HSX", "UPCOM", "SSC", "VSD"}
BOARDS = ("HSX", "HNX")

TOKEN_RE = re.compile(
    r"name=[\"']?__RequestVerificationToken[\"']?[^>]*?value=[\"']?([A-Za-z0-9_-]+)")
PDF_RE = re.compile(r"href=[\"']?([^\"' >]+\.pdf[^\"' >]*)", re.I)
ATTACH_RE = re.compile(r"/vietstock/\d{4}/\d{1,2}/\d{1,2}/")
DATE_RE = re.compile(r"/Date\((\d+)\)/")
# Language marks seen in attachment file names: "___vi_", "___vn_", "_en_".
# "ta" (tiếng Anh) is left out: in file names it is more often a person's name.
# "song ngu" / "vie_eng" mark one document that holds both languages.
LANG_RE = {"vi": re.compile(r"(^|_)(vi|vn|tv|tiengviet)(_|$)"),
           "en": re.compile(r"(^|_)(en|eng|tienganh|english)(_|$)")}
BILINGUAL_RE = re.compile(r"song_?ngu|vie_eng|vi_en(_|$)|bilingual")

# A page with fewer extracted characters than this is a scan. Text-layer filings
# run 500-3000 per page; a scan yields only its signature block, 20-150.
SCAN_CHARS_PER_PAGE = 200
MAX_BYTES = 50_000_000         # every filing over the old 15 MB cap was under 40 MB
PAUSE_S = 0.4                  # between requests; this is a survey, not a crawl


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def lang_of(filename: str) -> str:
    stem = filename.lower().rsplit(".", 1)[0]
    if BILINGUAL_RE.search(stem):
        return "both"
    hits = [k for k, rx in LANG_RE.items() if rx.search(stem)]
    return hits[0] if len(hits) == 1 else "?"


def norm_stem(filename: str) -> str:
    """The file name without its language mark, signature suffix and dup counter,
    so the VI and EN copies of one disclosure compare equal when they are named
    alike."""
    s = filename.lower().rsplit(".", 1)[0]
    for rx in LANG_RE.values():
        s = rx.sub("_", s)
    s = re.sub(r"_(signed|ky_so|kyso)$", "", s)
    s = re.sub(r"_\d$", "", s)
    return re.sub(r"_+", "_", s).strip("_")


def sha256_of(path: str) -> str:
    """The file's SHA-256: the same PDF is sometimes posted under two filings."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pdf_verdict(path: str) -> dict:
    """text / scan / error for one downloaded file, by characters per page."""
    try:
        info = subprocess.run(["pdfinfo", path], capture_output=True, text=True,
                              errors="replace", timeout=20).stdout
        pages = int(next((l.split()[1] for l in info.splitlines()
                          if l.startswith("Pages:")), "0"))
        out = subprocess.run(["pdftotext", "-q", "-enc", "UTF-8", "-l", "40", path, "-"],
                             capture_output=True, timeout=60).stdout.decode("utf-8", "replace")
    except Exception as e:
        return {"verdict": "error", "detail": type(e).__name__}
    chars = len(re.sub(r"\s", "", out))
    per_page = chars / max(min(pages, 40), 1)
    return {"verdict": "scan" if per_page < SCAN_CHARS_PER_PAGE else "text",
            "pages": pages, "chars": chars}


def disclosures(feed: list[dict]) -> list[dict]:
    """The feed items that are filings to an exchange or regulator."""
    return [i for i in feed if (i.get("Source") or "").upper() in DISCLOSURE_SOURCES
            and (i.get("URL") or "").startswith("/")]


def article_items(c, sym: str, i: dict) -> list[dict]:
    """One item per PDF attached to the feed item's article page."""
    m = DATE_RE.search(i.get("PublishTime") or "")
    pub = (datetime.fromtimestamp(int(m.group(1)) / 1000, tz=timezone.utc)
           .strftime("%Y-%m-%d") if m else None)
    try:
        links = c.attachments(ARTICLE_BASE + i["URL"])
    except Exception:
        links = []
    title = (i.get("Title") or "").strip()
    out = []
    for u in links:
        fn = u.rsplit("/", 1)[-1]
        out.append({"symbol": sym, "title": title,
                    "published": pub, "article": ARTICLE_BASE + i["URL"],
                    "pdf": u, "file": fn, "lang": lang_of(fn),
                    "stem": norm_stem(fn), "kind": kind_of(title)})
    return out


def pair_items(items: list[dict]) -> list[tuple[str, str, dict]]:
    """(symbol, article, {"vi": item, "en": item}) per disclosure with both.

    The VI and EN files are named differently, so the page they hang off is
    the link. Where a page has several of a language (re-uploads,
    appendices), the first of each is taken."""
    by_key = {}
    for it in items:
        if it["lang"] in ("vi", "en"):
            by_key.setdefault((it["symbol"], it["article"]), {}).setdefault(it["lang"], it)
    return [(sym, article, d) for (sym, article), d in sorted(by_key.items())
            if "vi" in d and "en" in d]


class Client:
    def __init__(self):
        self.http = httpx.Client(headers={"User-Agent": UA}, follow_redirects=True,
                                 timeout=30)
        self.token = None

    def get(self, url, **kw):
        time.sleep(PAUSE_S)
        r = self.http.get(url, **kw)
        r.raise_for_status()
        return r

    def listing(self) -> list[str]:
        r = self.get(LISTING_URL, headers={"Referer": "https://trading.vietcap.com.vn/"})
        return sorted(x["symbol"] for x in r.json()
                      if x.get("type") == "STOCK" and x.get("board") in BOARDS)

    def feed(self, sym: str, pages: int) -> list[dict]:
        ref = f"{FEED_BASE}/{sym}/tin-tuc-su-kien.htm"
        if self.token is None:
            m = TOKEN_RE.search(self.get(ref).text)
            if not m:
                raise RuntimeError("no anti-forgery token on the feed page")
            self.token = m.group(1)
        out = []
        for p in range(1, pages + 1):
            time.sleep(PAUSE_S)
            r = self.http.post(f"{FEED_BASE}/data/getnewsbycode",
                               data={"code": sym, "types[]": ["-1", "3", "4", "5", "6", "7", "8"],
                                     "page": p, "pageSize": 20,
                                     "__RequestVerificationToken": self.token},
                               headers={"X-Requested-With": "XMLHttpRequest", "Referer": ref})
            r.raise_for_status()
            j = r.json()
            items = j[0] if j and isinstance(j[0], list) else j
            if not items:
                break
            out.extend(i for i in items if isinstance(i, dict))
            if len(items) < 20:
                break
        return out

    def attachments(self, url: str) -> list[str]:
        html = self.get(url).text
        return list(dict.fromkeys(u for u in PDF_RE.findall(html) if ATTACH_RE.search(u)))

    def download(self, url: str, cache: str) -> tuple[str | None, str]:
        """(path, "") once cached, else (None, why not)."""
        name = hashlib.sha1(url.encode()).hexdigest()[:16] + ".pdf"
        path = os.path.join(cache, name)
        if os.path.exists(path):
            return path, ""
        r = self.get(url, headers={"Referer": ARTICLE_BASE + "/"})
        data = r.content
        if not data.startswith(b"%PDF"):
            return None, "not a pdf"
        if len(data) > MAX_BYTES:
            return None, "too big"
        fd, tmp = tempfile.mkstemp(dir=cache, suffix=".part")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
        return path, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbols", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--feed-pages", type=int, default=3, help="20 items per page")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="cache/pdfs")
    ap.add_argument("--from-items", action="store_true",
                    help="re-read <out>/items.jsonl instead of fetching the feeds again")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)

    c = Client()
    items = []
    if a.from_items:
        with open(os.path.join(a.out, "items.jsonl"), encoding="utf-8") as fh:
            items = [json.loads(l) for l in fh]
        for it in items:                       # re-label with the current rules
            it["lang"], it["stem"] = lang_of(it["file"]), norm_stem(it["file"])
            it["kind"] = kind_of(it["title"])
        picks = sorted({it["symbol"] for it in items})
    else:
        universe = c.listing()
        picks = random.Random(a.seed).sample(universe, a.symbols)
        log(f"listing: {len(universe)} stocks on {'/'.join(BOARDS)}; sampled {len(picks)}")

    for n, sym in enumerate([] if a.from_items else picks, 1):
        try:
            feed = c.feed(sym, a.feed_pages)
        except Exception as e:
            log(f"[{n}/{len(picks)}] {sym}: feed failed ({type(e).__name__})")
            continue
        disc = disclosures(feed)
        for i in disc:
            items.extend(article_items(c, sym, i))
        log(f"[{n}/{len(picks)}] {sym}: {len(disc)} disclosures")

    with open(os.path.join(a.out, "items.jsonl"), "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    pairs = []
    for sym, article, d in pair_items(items):
        kind = d["vi"]["kind"]
        row = {"symbol": sym, "article": article, "kind": kind, "profile": PROFILE[kind]}
        for lang in ("vi", "en"):
            path, why = c.download(d[lang]["pdf"], a.cache)
            row[lang] = {"pdf": d[lang]["pdf"], "title": d[lang]["title"],
                         "published": d[lang]["published"],
                         **({**pdf_verdict(path), "sha256": sha256_of(path)} if path
                            else {"verdict": "unavailable", "detail": why})}
        pairs.append(row)
    with open(os.path.join(a.out, "pairs.jsonl"), "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Summary
    langs = {}
    for it in items:
        langs[it["lang"]] = langs.get(it["lang"], 0) + 1
    shapes = {}
    for p in pairs:
        k = f"vi={p['vi']['verdict']} en={p['en']['verdict']}"
        shapes[k] = shapes.get(k, 0) + 1
    print(json.dumps({"symbols": len(picks), "attachments": len(items),
                      "by_lang": langs, "pairs": len(pairs), "pair_shapes": shapes},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
