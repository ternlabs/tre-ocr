"""Line codes of the statement forms: the 2014 form (TT200, and TT202 for groups) and the
2026 form (TT99/2025), and how one's codes become the other's.

The 2026 form inserts lines (short-term investments and their provisions, biological
assets, dividends payable, long-term taxes payable) and renumbers what follows them, so a
figure printed under code 151 in a 2025 balance sheet is printed under 161 in 2026. To
compare two filings across the change, the older one's codes are carried to the newer.

OLD_TO_NEW holds only codes whose line kept its meaning. A line the 2026 form dropped or
split (short- and long-term loans receivable, goodwill, the restructuring fund, capital
construction funds, the funds and sources section) has no entry: its figure has no
single place in the new form, so it is compared with nothing. Codes not listed in a
statement's map keep their number. Layout only, from the forms; no figure.
"""
import re

from sort_kinds import fold

_B01_SHIFTS = [
    # (first old code, last old code, added to reach the new code)
    (136, 137, -1), (139, 139, -2), (149, 149, -7),
    (150, 155, +10),
    (216, 216, -1), (219, 219, -3),
    (230, 232, +10), (240, 242, +10), (250, 255, +10), (260, 263, +10), (268, 268, +6),
    (270, 270, +10),
    (313, 324, +1), (333, 343, +1),
    (420, 421, -1),
]
_B01_DROPPED = {135, 215, 269, 419, 422, 430, 431, 432}

OLD_TO_NEW = {
    "B01": {f"{c}": f"{c + d}" for lo, hi, d in _B01_SHIFTS for c in range(lo, hi + 1)}
           | {f"{c}": None for c in _B01_DROPPED},
    "B02": {"21": "22", "22": "23", "23": "24", "24": "27"},
    "B03": {},
}


def to_new(statement: str, code: str) -> str | None:
    """The 2026 form's code for a 2014 form's line; None when the line has no single place."""
    m = OLD_TO_NEW[statement]
    return m[code] if code in m else code


def old_of(statement: str, code: str) -> str | None:
    """The 2014 form's code for a 2026 form's line; None when the old form has no such line
    (its number may then be a different old line, so it is not taken as one)."""
    m = OLD_TO_NEW[statement]
    old = next((o for o, n in m.items() if n == code), None)
    return old if old is not None else None if code in m else code


def form_of(statements: list[dict]) -> str | None:
    """"2026" or "2014", from a filing's statements as tools/fs_rows.py reads them; None
    when nothing tells.

    The balance sheet says it best: the 2026 one is named "báo cáo tình hình tài chính",
    and totals its assets on code 280 (270 before). Failing that, the income statement:
    code 21 is gains on investment property in 2026, financial income before."""
    for s in statements:
        if s["statement"] != "B01":
            continue
        if s["form_2026"]:
            return "2026"
        for r in s["rows"]:
            if "tong cong tai san" in fold(r["label"]):
                if r["code"] in ("270", "280"):
                    return "2014" if r["code"] == "270" else "2026"
    for s in statements:
        if s["statement"] != "B02":
            continue
        for r in s["rows"]:
            if r["code"] == "21":
                t = fold(r["label"])
                if re.search(r"bat dong san", t):
                    return "2026"
                if re.search(r"hoat dong tai chinh", t):
                    return "2014"
    return None
