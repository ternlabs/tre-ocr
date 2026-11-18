"""Publish a scored statements set's truth as one file, and unpack it to be scored (profile 2).

A set's truth is spread over its truth folder (cells.jsonl, cells-told.jsonl,
checked.jsonl) and its check work (the where round's items and answers, and any where
answer corrected after the score). **pack** joins them into sets/truth-statements-X.jsonl,
one record per line, each tagged with its "part". **unpack** writes them back as
tools/score_fs.py reads them, so anyone with the set's PDFs, the code and the weights can
score a reading against the same truth.

A matched cell is looked up at its column's place (x) in the reading that made the truth
(see score_fs.answer_of). That reading is not published, so pack writes each matched
side's x into the cell itself; a side whose place the reading does not give keeps none,
and is looked up by column number, as score_fs does when scoring from the folders.

The file holds figures read from public filings, the person's typed figures and page
answers, and nothing else. release.sh publishes it only once the set is in METHOD.md's
sets table, like the set itself.

Usage:
  .venv/bin/python -I tools/fs_truth_bundle.py pack X
  .venv/bin/python -I tools/fs_truth_bundle.py unpack sets/truth-statements-X.jsonl <out>
then
  .venv/bin/python -I tools/score_fs.py --set sets/statements-X.jsonl --drafts <drafts> \\
      --truth <out>/truth --work <out>/work [--corrections <out>/work/corrections-where.jsonl]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from records import read_jsonl  # noqa: E402

TRUTH = [("cell", "truth", "cells.jsonl"), ("told", "truth", "cells-told.jsonl"),
         ("checked", "truth", "checked.jsonl")]
WORK = [("where-item", "work", "items-where.jsonl"), ("where-answer", "work", "answers-where.jsonl"),
        ("where-correction", "work", "corrections-where.jsonl")]


def pack(letter: str) -> None:
    from fs_truth import load
    from score_fs import column_x
    x = letter.lower()
    truth, work = f"data/truth-{x}", f"data/check-{x}"
    made = os.path.join(work, "drafts")
    by = {f["sha256"]: f for f in read_jsonl(f"sets/statements-{x}.jsonl")}
    readings = {}

    def reading(sha):
        if sha not in readings:
            readings[sha] = load(made, by[sha])
        return readings[sha]

    out = f"sets/truth-statements-{x}.jsonl"
    n, placed, unplaced = {}, 0, 0
    with open(out, "w", encoding="utf-8") as fh:
        for part, where, name in TRUTH + WORK:
            p = os.path.join(truth if where == "truth" else work, name)
            if not os.path.exists(p):
                if part in ("cell", "checked", "where-item", "where-answer"):
                    sys.exit(f"missing {p}: only a set scored by score_fs can be packed")
                continue
            for r in read_jsonl(p):
                if part == "cell":
                    for side in ("base", "later"):
                        s = r.get(side) or {}
                        if s.get("sha256") and s.get("page") is not None and s.get("x") is None:
                            got = column_x(reading(s["sha256"]), r["statement"],
                                           s.get("code", r["code"]), s["page"], s["column"])
                            if got is None:
                                unplaced += 1
                            else:
                                s["x"] = got
                                placed += 1
                fh.write(json.dumps({"part": part, **r}, ensure_ascii=False) + "\n")
                n[part] = n.get(part, 0) + 1
    print(out, n, f"places written {placed}, none found {unplaced}")


def unpack(bundle: str, out: str) -> None:
    files = {}
    for r in read_jsonl(bundle):
        part = r.pop("part")
        where, name = next((w, f) for p, w, f in TRUTH + WORK if p == part)
        files.setdefault(os.path.join(out, where, name), []).append(r)
    for p, recs in files.items():
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(p, len(recs))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "pack":
        pack(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "unpack":
        unpack(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
