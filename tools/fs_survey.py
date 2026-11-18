"""FS0: a survey of a statements set's drafts, before anything is scored (profile 2).

For each primary statement found by tools/fs_rows.py: the form printed, the unit, the
pages it spans, its rows and columns, and the share of its values read as sure. For
each file: its pages, how they were read (scan, text layer, turned upright), and where
the statements and the notes begin. Counts only; no value is printed. Last, where a
file's contents page and the statements found disagree (tools/fs_contents.py), its flags.

Usage: .venv/bin/python -I tools/fs_survey.py sets/statements-h.jsonl data/set-h/drafts
"""
import json
import os
import re
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from fs_contents import check  # noqa: E402
from fs_rows import FORM_RE, statements_of  # noqa: E402
from fs_turn import before_notes  # noqa: E402

KINDS = ("B01", "B02", "B03")


def spread(xs: list[float]) -> str:
    if not xs:
        return "-"
    return f"median {statistics.median(xs):g}, {min(xs):g} to {max(xs):g}"


def form_of(s: dict) -> str:
    if not s["form"]:
        return "not found"
    m = FORM_RE.search(s["form"])
    if not m:
        return "printed, digits unread"
    return f"B0{m.group(1)}{(m.group(2) or '').lower()}-{re.sub(r'\s', '', m.group(3)).upper()}"


def unit_of(s: dict) -> str:
    u = (s["unit"] or "").lower().replace("vnđ", "vnd").replace("ngàn", "nghìn")
    return re.sub(r"\s+", " ", u) or "not printed"


def main():
    set_path, drafts = sys.argv[1], sys.argv[2]
    with open(set_path, encoding="utf-8") as fh:
        files = [json.loads(line) for line in fh]
    by = {k: [] for k in KINDS}
    pages = Counter()
    sizes, notes_at, first_at, missing, extra, kinds_found = [], [], [], Counter(), 0, 0
    contents, flagged = 0, []
    for f in files:
        with open(os.path.join(drafts, f["sha256"][:16] + ".draft.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        pdf = os.path.join("cache", "pdfs", d["file"])
        ss = statements_of(d, pdf if os.path.exists(pdf) else None, images=os.path.join(drafts, "images"))
        sizes.append(len(d["pages"]))
        got = check(d, ss, pdf if os.path.exists(pdf) else None)
        if got is not None:
            contents += 1
            flagged += [(f["sha256"][:16], x) for x in got["flags"]]
        head = before_notes(d["pages"])
        if len(head) < len(d["pages"]):
            notes_at.append(len(head) + 1)
        found = {s["statement"] for s in ss}
        missing.update(k for k in KINDS if k not in found)
        extra += len(ss) - len(found)
        kinds_found += len(found)
        if ss:
            first_at.append(min(s["pages"][0] for s in ss))
        for p in d["pages"]:
            pages["turned upright" if "turned" in p else p["kind"]] += 1
        for s in ss:
            vals = [v for r in s["rows"] for v in r["values"] if v]
            by[s["statement"]].append({
                "form": form_of(s), "new": s["form_2026"], "unit": unit_of(s),
                "pages": len(s["pages"]), "rows": len(s["rows"]),
                "cols": max(len(c["x"]) for c in s["columns"]),
                "codes": sum(r["code"] is not None for r in s["rows"]) / max(len(s["rows"]), 1),
                "sure": sum(v["value"] is not None for v in vals) / max(len(vals), 1),
                "on": Counter(next(p for p in d["pages"] if p["page"] == n)["kind"] for n in s["pages"]),
                "scope": f["scope"]})

    print(f"files {len(files)}; pages {spread(sizes)}; pages by how read: {dict(pages)}")
    print(f"statements begin on page {spread(first_at)}; notes begin on page {spread(notes_at)} "
          f"({len(notes_at)} files with notes found)")
    print(f"statements found {kinds_found} of {3 * len(files)}; "
          f"missing {dict(missing)}; a kind found twice in a file (printed again): {extra}")
    print(f"scopes: {dict(Counter(f['scope'] for f in files))}")
    for k in KINDS:
        ss = by[k]
        print(f"\n{k}: {len(ss)}")
        print("  form printed:", dict(Counter(s["form"] for s in ss).most_common()))
        if k == "B01":
            print("  2026 name (tình hình tài chính):", sum(bool(s["new"]) for s in ss))
        print("  unit:", dict(Counter(s["unit"] for s in ss).most_common()))
        print("  pages:", dict(sorted(Counter(s["pages"] for s in ss).items())))
        print("  rows:", spread([s["rows"] for s in ss]))
        print("  value columns:", dict(sorted(Counter(s["cols"] for s in ss).items())))
        print("  rows with a code read:", spread([round(100 * s["codes"]) for s in ss]), "%")
        print("  values sure:", spread([round(100 * s["sure"]) for s in ss]), "%")
        on = Counter()
        for s in ss:
            on.update(s["on"])
        print("  pages by how read:", dict(on))
    print(f"\ncontents pages: {contents} files; agreeing with the statements found: "
          f"{contents - len({k for k, _ in flagged})}")
    for k, x in flagged:
        print(f"  {k} {x}")


if __name__ == "__main__":
    main()
