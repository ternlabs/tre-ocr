"""Small helpers shared by the tools, with no heavy imports: number strings,
JSON-lines files and the pages of a set."""
import json
import os
import re

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(text: str) -> list[str]:
    """Digit strings of 3+ digits, separators removed: 1.234,5 and 1,234.5 agree."""
    return [d for d in (re.sub(r"[.,]", "", m) for m in NUMBER_RE.findall(text)) if len(d) >= 3]


def short_numbers(text: str) -> list[str]:
    """The digits of each word holding 1 or 2 of them, cut at any mark but a letter, with
    I and l read as 1: "17B" gives "17", "ngày24" "24", "Q4/2026" "4". A
    print's 1 is sometimes typed as I in a truth, and a reader's digit is not wrong for it."""
    out = []
    for w in re.findall(r"[^\W_]+", text):
        if any(c.isdigit() for c in w):
            d = "".join(c for c in w.translate(FOLD) if c.isdigit())
            if 1 <= len(d) <= 2:
                out.append(d)
    return out


FOLD = str.maketrans("Il", "11")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str, rows):
    """One JSON line per row, written to a side file and moved into place, so
    a stopped run never leaves half a file."""
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)


def page_of(row: dict) -> dict:
    """The page an engine reads: set A keeps it under "vi"."""
    return row.get("vi", row)


def page_key(e: dict) -> str:
    """A page's image name: <sha256 first 16>-p<page>."""
    return f"{e['sha256'][:16]}-p{e['page']}"
