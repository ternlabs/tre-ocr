"""Sort filings by kind from their title words, and each kind into a profile.

A filing's title names what it is: "Nghị quyết HĐQT ...", "BCTC quý 2 ...",
"Báo cáo kết quả giao dịch cổ phiếu ...". The kind decides the profile:
resolutions, notices and explanations are disclosures (v0.1); financial
statements wait for their own profile, which needs tables; annual reports are
designed booklets, in no profile yet.

Matching: the title is lower-cased, its accents removed and every mark that is
not a letter or digit made a space ("05/NQ-HĐQT" reads "05 nq hdqt"). Each kind
has phrases; the phrase that starts earliest in the title wins, and of two at
the same place the longer. So "Nghị quyết HĐQT về việc chọn kiểm toán BCTC" is
a resolution, not a statement, and "Thông báo giao dịch cổ phiếu" is an insider
trade, not a plain notice. Words that only say "this is published"
("Công bố thông tin", "CBTT") are not phrases, so the kind after them decides.
A few notice words ("Thông báo", "Đính chính", ...) only say what is done to
the thing named next: when another phrase follows at once ("Đính chính BCTC
quý 2"), that one decides. A title no phrase matches is "unsorted" and goes in
no profile. Titles are mostly Vietnamese; a few English phrases cover feeds
that title in English.

The sampler stamps each item and pair with its kind and profile. This script
re-stamps a sampler folder after the phrases change, so there is one copy of
each filing's kind, and prints the counts.

Usage:
  python -I tools/sort_kinds.py data/sample-01
"""
import argparse
import functools
import json
import os
import re
import sys
import unicodedata

PROFILE = {
    "resolution": "disclosures",
    "notice": "disclosures",
    "personnel": "disclosures",
    "explanation": "disclosures",
    "insider_trade": "disclosures",
    "issue_report": "disclosures",
    "meeting_pack": "disclosures",
    "governance_report": "disclosures",
    "charter": "disclosures",
    "financial_statement": "financial_statements",
    "annual_report": None,
    "prospectus": None,
    "unsorted": None,
}

# Phrases, folded as by fold(). Word ends are checked, so "bctc" does not match
# inside a longer word.
PHRASES = {
    "resolution": ["nghi quyet", "quyet dinh", "qd", "bien ban", "nq hdqt",
                   "nq dhdcd", "hdqt thong qua",
                   "resolution", "decision", "minutes"],
    "notice": ["thong bao", "cong van", "giay xac nhan", "giay chung nhan",
               "giay phep", "dinh chinh", "dieu chinh", "ban an", "lai suat",
               "tai lieu", "hop dong", "to trinh", "ngay dang ky cuoi cung",
               "ngay dkcc", "ngay chot danh sach", "chi tra co tuc", "tra co tuc",
               "giay dang ky", "dang ky kinh doanh", "dang ky doanh nghiep",
               "con dau", "nguoi cong bo thong tin", "uy quyen cong bo thong tin",
               "nguoi uy quyen", "so huu nuoc ngoai", "dang ky giao dich",
               "dang ky niem yet", "ngay giao dich dau tien", "thoai von",
               "gop von", "kiem soat", "ky quy", "canh bao", "vi pham",
               "thu trao thau", "lua chon don vi kiem toan", "ke hoach",
               "to chuc dai hoi", "noi dung to chuc", "chuyen nhuong co phan",
               "quyen bieu quyet",
               "notice", "announcement", "official letter", "record date"],
    "personnel": ["thay doi nhan su", "thong bao thay doi nhan su", "thong bao bo nhiem",
                  "thong bao mien nhiem", "bo nhiem", "mien nhiem", "tu nhiem",
                  "thoi giu chuc", "thoi lam", "duoc bau", "bau bo sung",
                  "ket qua bau", "ly lich", "ung vien", "de cu",
                  "thanh vien hoi dong quan tri", "thuc hien cong tac",
                  "appointment", "dismissal", "resignation", "change of personnel"],
    "explanation": ["giai trinh", "bien phap va lo trinh", "explanation"],
    "insider_trade": ["ket qua giao dich", "thong bao giao dich", "bao cao giao dich",
                      "co dong lon", "thay doi so huu", "nguoi noi bo",
                      "chuyen nhuong co phieu", "thong bao thay doi nguoi co lien quan",
                      "insider", "major shareholder"],
    "meeting_pack": ["tai lieu hop", "tai lieu dai hoi", "tai lieu dhdcd",
                     "tai lieu lay y kien", "tai lieu xin y kien",
                     "cap nhat tai lieu", "bo sung tai lieu", "chuong trinh dai hoi",
                     "thu moi", "thong bao moi hop",
                     "meeting documents", "agm documents", "invitation"],
    "governance_report": ["bao cao quan tri", "bao cao tinh hinh quan tri",
                          "qtct", "governance report", "corporate governance"],
    "charter": ["dieu le", "quy che", "quy dinh", "charter", "regulation"],
    "issue_report": ["ket qua phat hanh", "ket qua dot phat hanh", "chao ban",
                     "su dung von", "su dung so tien", "thanh toan goc",
                     "goc lai", "trai phieu", "bond", "offering", "issuance"],
    "financial_statement": ["bctc", "bao cao tai chinh", "ty le an toan tai chinh",
                            "ty le attc", "financial statement", "financial statements",
                            "financial report", "capital adequacy"],
    "annual_report": ["bao cao thuong nien", "bctn", "annual report"],
    "prospectus": ["ban cao bach", "prospectus"],
}

# Notice words that only say what is done to the thing named next; then a
# phrase straight after them decides, past filler such as "về việc công bố".
LEAD_WORDS = {"thong bao", "cong van", "dinh chinh", "dieu chinh",
              "notice", "announcement"}
FILLER_RE = re.compile(r"(?: (?:ve viec|v v|ve|cua|cong bo thong tin|cong bo|cbtt|"
                       r"bo sung|on|of|the|information disclosure|disclosure)(?![a-z0-9]))*")


def fold(s: str) -> str:
    """Lower case, accents and đ removed, every other mark made a space."""
    s = unicodedata.normalize("NFD", s.lower()).replace("đ", "d").replace("ð", "d")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


_RULES = sorted(((k, p, re.compile(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])"))
                 for k, ps in PHRASES.items() for p in ps), key=lambda r: -len(r[1]))


def _matches(t: str) -> list[tuple[int, int, str, str]]:
    """Every phrase hit as (start, -length, kind, phrase), best first."""
    hits = [(m.start(), -len(p), k, p) for k, p, rx in _RULES for m in rx.finditer(t)]
    return sorted(hits)


@functools.lru_cache(maxsize=None)
def kind_of(title: str) -> str:
    t = fold(title or "")
    hits = _matches(t)
    if not hits:
        return "unsorted"
    start, neg_len, kind, phrase = hits[0]
    if phrase in LEAD_WORDS:
        nxt = FILLER_RE.match(t, start - neg_len).end() + 1   # the next word
        for s, _, k, p in hits:
            if s == nxt and p not in LEAD_WORDS:
                return k
    return kind


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", help="a sampler output folder holding items.jsonl")
    a = ap.parse_args()
    items_path = os.path.join(a.dir, "items.jsonl")
    pairs_path = os.path.join(a.dir, "pairs.jsonl")
    with open(items_path, encoding="utf-8") as fh:
        items = [json.loads(l) for l in fh]
    kinds = {}                                   # one filing = one article page
    for it in items:
        it["kind"] = kind_of(it["title"])
        kinds.setdefault(it["article"], it["kind"])
    with open(items_path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    if os.path.exists(pairs_path):
        with open(pairs_path, encoding="utf-8") as fh:
            pairs = [json.loads(l) for l in fh]
        for p in pairs:
            p["kind"] = kinds[p["article"]]
            p["profile"] = PROFILE[p["kind"]]
        with open(pairs_path, "w", encoding="utf-8") as fh:
            for p in pairs:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    by_kind, by_profile = {}, {}
    for k in kinds.values():
        by_kind[k] = by_kind.get(k, 0) + 1
        by_profile[str(PROFILE[k])] = by_profile.get(str(PROFILE[k]), 0) + 1
    print(json.dumps({"filings": len(kinds),
                      "by_kind": dict(sorted(by_kind.items(), key=lambda x: -x[1])),
                      "by_profile": by_profile}, indent=2))
    if by_kind.get("unsorted"):
        print(f"warning: {by_kind['unsorted']} filings unsorted; their titles need "
              "phrases", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
