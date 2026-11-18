"""A financial statement's period and scope, from its filing title (METHOD.md, profile 2).

Set H pairs a statement with the later one that repeats its figures (the "free truth from
the next filing"), so each title is read as (period, number, fiscal year, scope):

- period: "year", "half" (bán niên, 6 tháng đầu năm, giữa niên độ) or "quarter" (quý N,
  QN, quý I..IV);
- scope: "consolidated" (hợp nhất, HN), "separate" (riêng, riêng lẻ, công ty mẹ), or
  None when the title does not say (a company with no subsidiaries files one kind);
- year: the last four-digit year in the title; without one, from the day published: a
  fourth quarter or a year filed in January to April belongs to the year before.

Left out (None): summaries (tóm tắt), corrections (đính chính), capital-adequacy reports,
notices about the auditor, and titles whose period cannot be told.
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from sort_kinds import fold  # noqa: E402

SKIP = re.compile(r"\b(tom tat|dinh chinh|an toan tai chinh|attc|don vi kiem toan|"
                  r"giai trinh|chenh lech)\b")
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
QUARTER = re.compile(r"\b(?:quy|q)\s*(i{1,3}|iv|[1-4])\b")
HALF = re.compile(r"\b(ban nien|6 thang|giua nien do|30 06)\b")
YEAR = re.compile(r"\b(20\d\d)\b")
CONSOLIDATED = re.compile(r"\b(hop nhat|hn)\b")
SEPARATE = re.compile(r"\b(rieng|rieng le|cong ty me|me)\b")


def period_of(title: str, published: str) -> dict | None:
    """{"period", "n", "year", "scope"} for a statement's title, or None."""
    t = fold(re.sub(r"^[A-Z0-9]{3,4}:\s*", "", title))
    if SKIP.search(t):
        return None
    if m := QUARTER.search(t):
        period, n = "quarter", ROMAN.get(m.group(1)) or int(m.group(1))
    elif HALF.search(t):
        period, n = "half", 1
    elif re.search(r"\b(bao cao tai chinh|bctc)\b", t):
        period, n = "year", 1
    else:
        return None
    years = YEAR.findall(t)
    pub_year, pub_month = int(published[:4]), int(published[5:7])
    if years:
        year = int(years[-1])
    elif period == "half":
        year = pub_year
    elif period == "year" or (period == "quarter" and n == 4):
        year = pub_year - 1 if pub_month <= 4 else pub_year
    else:
        year = pub_year
    scope = ("consolidated" if CONSOLIDATED.search(t)
             else "separate" if SEPARATE.search(t) else None)
    return {"period": period, "n": n, "year": year, "scope": scope}


def partner_key(p: dict) -> tuple:
    """The key of the statement one year on, which repeats this one's figures."""
    return (p["period"], p["n"], p["year"] + 1, p["scope"])


def key(p: dict) -> tuple:
    return (p["period"], p["n"], p["year"], p["scope"])
