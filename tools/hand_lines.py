"""Turn the hand labels of tools/label_lines.py into training lines (round 2).

A label covers a whole detected line, often longer than training takes (MAX_CHARS,
MAX_RATIO). Such a crop is cut between words, the way tools/train_lines.py cuts a row,
but here no word boxes exist: the blank columns must be told apart from the label alone.
So a line is cut only when its widest gaps give each word a width that fits its length;
otherwise it is dropped, since a misplaced cut teaches a wrong reading.

A run of dots or an ellipsis becomes "...", however long the dotted space is in the
image. A character outside the dictionary, such as a tick, is taken out of the label.

Whole files go to the held-out part, so that no file both trains and judges.

Usage:
  python -I tools/hand_lines.py cut --work cache/train/hand --data cache/train/data
  .venv-paddle/bin/python -I tools/hand_lines.py check --data cache/train/data \
    --rec-model cache/train/output/inference
  .venv-paddle/bin/python -I tools/hand_lines.py measure --data cache/train/data \
    --rec-model cache/train/output2/inference
  python -I tools/hand_lines.py mix --data cache/train/data
Writes <data>/lines/hand/*.jpg, <data>/hand_train.txt and <data>/hand_val.txt, gitignored:
labels can hold names. "check" drops the cut pieces a reader cannot nearly read, which is
what a misplaced cut looks like, and then measures like "measure": on the held-out lines,
digits, letters without marks and letters with marks apart. It prints counts only.
"mix" writes <data>/round2_list.txt, for tools/train_round2.sh: a random part of the typed
lines, and the hand lines repeated, so a few hundred are not lost among the many.
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from records import read_jsonl  # noqa: E402
from two_readers import digits  # noqa: E402

MAX_CHARS = 40           # as in tools/train_lines.py; importing it needs the sampling tools
MAX_RATIO = 12

VAL_SHARE = 5            # one file in this many is held out
WIDTH_FIT = (0.25, 4)    # a word's width per character, against the line's median
INK_PIXELS = 2           # one dark pixel in a column is noise, not a letter
CUT_ERROR = 0.2          # a cut piece the present reader misreads this much is misplaced
LEADER_RE = re.compile(r"(?:[.…·]\s*){3,}|…")


def clean(label: str, alphabet: set[str]) -> str:
    text = LEADER_RE.sub(" ... ", label)
    text = "".join(c for c in text if c in alphabet or c == " ")
    return " ".join(text.split())


def gaps_of(image: Image.Image) -> tuple[list[tuple[int, int]], int, int]:
    """The blank column runs inside the text, and where the text starts and ends."""
    a = np.array(image.convert("L"), dtype=int)
    dark = a < (np.percentile(a, 5) + np.percentile(a, 95)) / 2
    ink = dark.sum(axis=0) >= INK_PIXELS
    xs = np.flatnonzero(ink)
    if not xs.size:
        return [], 0, a.shape[1]
    first, last = int(xs[0]), int(xs[-1]) + 1
    gaps, start = [], None
    for x in range(first, last):
        if not ink[x] and start is None:
            start = x
        elif ink[x] and start is not None:
            gaps.append((start, x))
            start = None
    return gaps, first, last


def word_spans(image: Image.Image, words: list[str]) -> list[tuple[int, int]] | None:
    """Each word's column span, or None when the gaps do not fit the words."""
    gaps, first, last = gaps_of(image)
    if len(gaps) < len(words) - 1:
        return None
    widest = sorted(sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[:len(words) - 1])
    edges = [first] + [x for g in widest for x in g] + [last]
    spans = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]
    per_char = [(e - s) / len(w) for (s, e), w in zip(spans, words) if len(w) >= 3 and w != "..."]
    if not per_char:
        return None
    mid = float(np.median(per_char))
    return spans if all(WIDTH_FIT[0] <= p / mid <= WIDTH_FIT[1] for p in per_char) else None


def pieces(image: Image.Image, label: str) -> list[tuple[Image.Image, str]] | None:
    """The crop whole if training can take it, else cut between words; None if it cannot."""
    w, h = image.size
    if len(label) <= MAX_CHARS and w / max(h, 1) <= MAX_RATIO:
        return [(image, label)]
    words = label.split(" ")
    spans = word_spans(image, words) if len(words) > 1 else None
    if spans is None:
        return None
    groups, cur = [], [0]
    for i in range(1, len(words)):
        chars = sum(len(words[j]) + 1 for j in cur) + len(words[i])
        if chars > MAX_CHARS or (spans[i][1] - spans[cur[0]][0]) / max(h, 1) > MAX_RATIO:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    out = []
    for g in groups:
        text = " ".join(words[j] for j in g)
        x0, x1 = spans[g[0]][0], spans[g[-1]][1]
        if len(text) > MAX_CHARS or (x1 - x0) / max(h, 1) > MAX_RATIO:   # one word too wide
            return None
        pad = max(h // 8, 2)
        out.append((image.crop((max(0, x0 - pad), 0, min(w, x1 + pad), h)), text))
    return out


def cut(a):
    with open(os.path.join(a.data, "dict.txt"), encoding="utf-8") as fh:
        alphabet = {line.rstrip("\n") for line in fh}
    answers = {r["id"]: r["answer"] for r in read_jsonl(os.path.join(a.work, "labels.jsonl"))}
    out_dir = os.path.join(a.data, "lines", "hand")
    os.makedirs(out_dir, exist_ok=True)
    lists = {"train": [], "val": []}
    tally = Counter()
    for r in read_jsonl(os.path.join(a.work, "candidates.jsonl")):
        answer = answers.get(r["id"])
        if not answer:
            tally["no label"] += 1
            continue
        label = clean(answer, alphabet)
        if len(label) < 2:
            tally["too short"] += 1
            continue
        got = pieces(Image.open(os.path.join(a.work, "crops", r["id"] + ".jpg")).convert("RGB"), label)
        if got is None:
            tally["gaps do not fit"] += 1
            continue
        split = "val" if int(r["sha256"], 16) % VAL_SHARE == 0 else "train"
        tally["lines kept"] += 1
        tally["cut" if len(got) > 1 else "whole"] += 1
        for n, (image, text) in enumerate(got):
            name = f"lines/hand/{r['id']}-{n}.jpg"
            image.save(os.path.join(a.data, name), quality=92)
            lists[split].append(f"{name}\t{text}")
    for split, rows in lists.items():
        with open(os.path.join(a.data, f"hand_{split}.txt"), "w", encoding="utf-8") as fh:
            fh.write("".join(x + "\n" for x in rows))
    print(json.dumps({**tally, "train pieces": len(lists["train"]), "val pieces": len(lists["val"])}), flush=True)


def read_all(a, rows: list[str]) -> list[str]:
    from paddleocr import TextRecognition
    model = TextRecognition(model_name="PP-OCRv6_medium_rec", model_dir=a.rec_model)
    paths = [os.path.join(a.data, x.split("\t")[0]) for x in rows]
    return [r["rec_text"] for r in model.predict(paths, batch_size=32)]


def measure(a):
    from rapidfuzz.distance import Levenshtein
    from score import base
    with open(os.path.join(a.data, "hand_val.txt"), encoding="utf-8") as fh:
        rows = fh.read().splitlines()
    c = Counter()
    for x, read in zip(rows, read_all(a, rows)):
        truth = x.split("\t", 1)[1]
        want, got = digits(truth), digits(read)
        c["pieces"] += 1
        if want or got:
            c["pieces with digits"] += 1
            c["digits right"] += want == got
        c["chars"] += len(truth)
        c["edits with marks"] += Levenshtein.distance(truth, read)
        c["edits without marks"] += Levenshtein.distance(base(truth), base(read))
    print(json.dumps(c), flush=True)


def check(a):
    from rapidfuzz.distance import Levenshtein
    from score import base
    dropped = Counter()
    for split in ("train", "val"):
        path = os.path.join(a.data, f"hand_{split}.txt")
        with open(path, encoding="utf-8") as fh:
            rows = fh.read().splitlines()
        cut_stems = Counter(x.split("\t")[0].rsplit("-", 1)[0] for x in rows)
        keep = []
        for x, read in zip(rows, read_all(a, rows)):
            name, truth = x.split("\t", 1)
            is_cut = cut_stems[name.rsplit("-", 1)[0]] > 1
            error = Levenshtein.distance(base(truth), base(read)) / max(len(truth), 1)
            if is_cut and error > CUT_ERROR:
                dropped[split] += 1
            else:
                keep.append(x)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("".join(x + "\n" for x in keep))
        print(json.dumps({"split": split, "kept": len(keep), "dropped": dropped[split]}), flush=True)
    measure(a)


def mix(a):
    with open(os.path.join(a.data, "train_list.txt"), encoding="utf-8") as fh:
        typed = fh.read().splitlines()
    with open(os.path.join(a.data, "hand_train.txt"), encoding="utf-8") as fh:
        hand = fh.read().splitlines()
    random.Random(a.seed).shuffle(typed)
    rows = typed[:a.typed] + hand * a.repeat
    random.Random(a.seed).shuffle(rows)
    with open(os.path.join(a.data, "round2_list.txt"), "w", encoding="utf-8") as fh:
        fh.write("".join(x + "\n" for x in rows))
    print(json.dumps({"typed": min(a.typed, len(typed)), "hand": len(hand), "repeat": a.repeat, "rows": len(rows)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["cut", "check", "measure", "mix"])
    ap.add_argument("--work")
    ap.add_argument("--data", required=True)
    ap.add_argument("--rec-model", help="default: the stock recogniser")
    ap.add_argument("--typed", type=int, default=30000)
    ap.add_argument("--repeat", type=int, default=40)
    ap.add_argument("--seed", type=int, default=37)
    a = ap.parse_args()
    {"cut": cut, "check": check, "measure": measure, "mix": mix}[a.step](a)


if __name__ == "__main__":
    main()
