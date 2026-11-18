"""Score an engine's output on set A, B or C of the disclosure profile.

Step 7 of METHOD.md: the four scores of METHOD.md, "Concepts", each reported on its own,
for all pages, stamped pages and pages without a stamp.

- **Numbers**: digit strings of 3+ digits, separators removed
  (records.numbers), counted as a multiset. Recall on every set, over the whole
  output. Precision on B and C only (set A's truth holds only the part of the
  English page the scan shows), over the numbers read inside matched lines;
  numbers read elsewhere (a stamp, handwriting) are counted apart.
- **Base-letter error rate**: edit distance inside matched lines, plus every
  truth line not found, over truth characters, with every mark removed (tone
  marks and the marks of ă â ê ô ơ ư, đ made d); case kept. Letters and digits
  of the output that no truth line claims are counted apart as "inserted": the
  truth leaves out stamps, handwriting and the signature box, which an engine
  may well read (the box is painted white on the renders it is given,
  cache/set-b and cache/set-c; the rest is not).
- **Marks**: of the words whose base letters are right, the share that are
  right with their marks too; also over the words whose truth carries a mark,
  since unmarked words (English, abbreviations) are easy.
- **Heading fields** (set C, which alone has them), each field on its own:
  right when its value appears whole in the output where the heading's lines
  were matched, case and spaces ignored (a company name is typed in the field
  as a name, printed in capitals); reported with marks and without, so a wrong
  mark is not counted as a wrong digit. A field the page's own truth text does
  not hold cannot be placed and is counted as "unplaced", not scored.
- **Kinds of number** (B and C, since A's truth has no text around its numbers): recall
  apart for amounts, codes (a document number), other numbers and short ones of 1 or 2
  digits, told apart by shape and the words beside them. Whole dates are compared as
  (day, month, year), including those written into a dotted space (dots between the
  digits, no space after "ngày"); the same mark must part day, month and year and both
  must be real, so a document number is no date. Their digits
  count under no other kind, so the kinds do not add up to Numbers. A short number can
  be found by chance, so its recall flatters. "split_over_lines" counts the places where
  a truth line ending in a digit and a separator is followed by one starting with a
  digit, a sign of a number wrapped in its cell; each part is scored as its own number.

- **Sure numbers** (a run that marks unsure lines): wrong-when-sure is the share of
  numbers on sure lines that the truth does not hold, and coverage the share of the
  run's numbers on sure lines. Not on set A, whose truth holds only what the English
  twin prints, so a right number the twin lacks would count as wrong. All sets also
  report the sure numbers within 2 edits of a truth number otherwise unread: on A that
  is a floor for wrong-when-sure, not the figure.
- **Short words** (not A): words of 1 or 2 digits with their letters ("17B", "05"),
  compared by their digits (records.short_numbers). Recall over the page; and on a run
  that marks unsure lines, wrong-when-sure, each sure line checked against the truth
  line most like it, or one holding it, so "1B" for "17B" is wrong even with a "1" elsewhere. A sure line
  like no truth line (a page number, a signature box's time) is counted as outside,
  not wrong. Reported apart from the numbers of 3+ digits, whose figures they do not change.

Text is scored on B and C only (A's truth is numbers). Before scoring, every
run of spaces and line breaks becomes one space, and truth lines holding no
letter or digit (rules, "---o0o---") are dropped. Lines are matched in any
order: each truth line, longest first, is found in the output's text not yet
claimed by another line and not cut out of a longer word. An exact copy is
taken nearest where the line would sit if the output kept the truth's order;
else rapidfuzz finds a window and an infix alignment the exact span. A fuzzy
match with more than MAX_WRONG of the line wrong does not count: the line is
deleted and the text left unclaimed. The national title and the motto, in
either language, are cut out of the truth lines and scored apart ("motto"),
so easy text does not lift the rate.

A run is one JSON line per page: {"sha256", "page", "text"}, the page as the
set names it (set A: its Vietnamese scan). A page the run lacks is scored as
an empty output and counted as missing.

Set A: the method asks that a truth number no engine reads be checked by hand
before it counts as missed. --missed writes those numbers, over all runs
given, for the check (tools/check_missed.py); numbers found not to be on the scan go to --excuse (one
line each: {"sha256", "page", "number"}), and are taken out of the truth. An
excuse that no longer fits the truth (the truth was rebuilt) is reported as
stale and ignored.

Set C: a seal's own text is not asked of an engine, so a number printed in a seal
goes to the same list. A number a seal is stamped over stays in the truth.

Usage:
  python -I tools/score.py c data/runs/tesseract-c.jsonl data/runs/paddle-c.jsonl
  python -I tools/score.py a data/runs/*-a.jsonl --missed data/missed-a.jsonl
  python -I tools/score.py b data/runs/tesseract-b.jsonl --pages data/pages-b.jsonl
"""
import argparse
import difflib
import json
import os
import re
import sys
import unicodedata
from collections import Counter

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from records import NUMBER_RE, numbers, read_jsonl, short_numbers, write_jsonl  # noqa: E402
from syllables import WORD_RE, normal  # noqa: E402

SETS = {"a": ("sets/disclosures-a.jsonl", "data/align-a.jsonl"),
        "b": ("sets/disclosures-b.jsonl", "data/truth-b.jsonl"),
        "c": ("sets/disclosures-c.jsonl", "data/truth-c"),
        "d": ("sets/disclosures-d.jsonl", "data/truth-d"),
        "e": ("sets/disclosures-e.jsonl", "data/truth-e"),
        "f": ("sets/disclosures-f.jsonl", "data/truth-f")}
MOTTO_RE = re.compile(r"cong\W*hoa\W*xa\W*hoi\W*chu\W*nghia\W*viet\W*nam"
                      r"|socialist\W*republic\W*of\W*viet\W*nam"
                      r"|doc\W*lap\W*tu\W*do\W*hanh\W*phuc"
                      r"|independence\W*freedom\W*happiness", re.I)
MIN_MATCH = 50           # rapidfuzz partial ratio below which a window is not tried
MAX_WRONG = 0.5          # share of a line's characters a fuzzy match may get wrong
SHORT_LINE = 80          # fuzz ratio for a sure line to be the truth line it is checked against
SHORT_PART = 8           # characters a sure line needs to be checked against part of a truth line
WIDEN = 10               # characters each side of rapidfuzz's window, for the infix alignment
FIELD_LINES = 4          # a heading field is printed over at most this many lines
FILL = r"[\s.…_]*"       # a dotted space a date is written into
DOTS = r"[.…_]*"          # the same dotted space, between a written digit and what follows
DAY = rf"(\d(?:{DOTS}\d)?)"  # 1 or 2 digits, maybe with the dotted space between them
DATE_RE = re.compile(rf"(?<!\d)(\d{{1,2}}){DOTS}\s?([/.\-])\s?{DOTS}(\d{{1,2}}){DOTS}\s?\2\s?{DOTS}(\d{{4}})(?!\d)"
                     rf"|\bngay{FILL}{DAY}{FILL}thang{FILL}{DAY}{FILL}nam{FILL}(\d{{4}})", re.I)
AMOUNT_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?")
CURRENCY_RE = re.compile(r"\s?(?:dong|vnd|usd)\b", re.I)
CODE_AFTER_RE = re.compile(r"\s?[/\-]\s?\w")
CODE_BEFORE_RE = re.compile(r"(?:\w\s?[/\-]\s?|\b(?:so|no)\b[\s:.]*)$", re.I)


def collapse(text: str) -> str:
    return " ".join(text.split())


def strip(text: str) -> tuple[str, list[int]]:
    """The text with every mark removed and đ made d, and where each of its
    characters sits in text (with len(text) at the end). A mark NFC could not
    join to its letter (a second tone mark) goes with the rest, so it counts
    against the marks, not the base letters."""
    out, at = [], []
    for k, ch in enumerate(text):
        b = "".join(x for x in unicodedata.normalize("NFD", ch) if not unicodedata.combining(x))
        if b:
            out.append({"đ": "d", "Đ": "D"}.get(ch, b if len(b) == 1 else ch))
            at.append(k)
    at.append(len(text))
    return "".join(out), at


def base(text: str) -> str:
    return strip(text)[0]


def squeeze(text: str) -> str:
    """For heading fields: case and spaces dropped ("NQ - HĐQT" is "nq-hđqt")."""
    return "".join(text.split()).casefold()


def truth_lines(text: str) -> list[tuple[str, str]]:
    """(line, "text" or "motto"): the title and motto cut out of the lines they
    sit in; what holds no letter or digit dropped."""
    out = []
    for line in normal(text).splitlines():
        line = collapse(line)
        lb, at = strip(line)
        k = 0
        for m in MOTTO_RE.finditer(lb):
            out += [(line[k:at[m.start()]].strip(), "text"), (line[at[m.start()]:at[m.end()]], "motto")]
            k = at[m.end()]
        out.append((line[k:].strip(), "text"))
    return [(t, g) for t, g in out if any(ch.isalnum() for ch in t)]


def infix(p: str, t: str) -> tuple[int, int, int]:
    """(edit distance, start, end) of the span of t closest to the whole of p."""
    n = len(p)
    col, st = list(range(n + 1)), [0] * (n + 1)
    best = (col[n], 0, 0)
    for j, c in enumerate(t, 1):
        new, nst = [0] * (n + 1), [j] * (n + 1)
        for i in range(1, n + 1):
            new[i], nst[i] = min((col[i - 1] + (p[i - 1] != c), st[i - 1]),
                                 (new[i - 1] + 1, nst[i - 1]),
                                 (col[i] + 1, st[i]))
        col, st = new, nst
        if col[n] < best[0]:
            best = (col[n], st[n], j)
    return best


def cuts_word(s: str, k: int) -> bool:
    return 0 < k < len(s) and s[k - 1].isalnum() and s[k].isalnum()


def match_lines(lines: list[str], ob: str) -> list[tuple[int, int, int] | None]:
    """Per truth line, (distance on base letters, start, end) in ob, the
    output's base letters, or None when the line is not found."""
    free = [(0, len(ob))]
    found: list[tuple[int, int, int] | None] = [None] * len(lines)
    total = sum(len(t) + 1 for t in lines) or 1
    expect, at = [], 0
    for t in lines:
        expect.append(at / total * len(ob))
        at += len(t) + 1

    def claim(i, k, d, a, b):
        found[i] = (d, a, b)
        s, e = free.pop(k)
        free[k:k] = [x for x in ((s, a), (b, e)) if x[1] > x[0]]

    for i in sorted(range(len(lines)), key=lambda i: -len(lines[i])):
        tb = base(lines[i])
        exact = []
        for k, (s, e) in enumerate(free):
            j = ob.find(tb, s, e)
            while j != -1:
                if not cuts_word(ob, j) and not cuts_word(ob, j + len(tb)):
                    exact.append((abs(j - expect[i]), k, j))
                j = ob.find(tb, j + 1, e)
        if exact:
            _, k, j = min(exact)
            claim(i, k, 0, j, j + len(tb))
            continue
        best = None
        for k, (s, e) in enumerate(free):
            if len(tb) >= e - s:
                score, ws, we = fuzz.ratio(tb, ob[s:e]), s, e
            else:
                r = fuzz.partial_ratio_alignment(tb, ob[s:e])
                score, ws, we = r.score, max(s, s + r.dest_start - WIDEN), min(e, s + r.dest_end + WIDEN)
            if score >= MIN_MATCH and (best is None or score > best[0]):
                best = (score, k, ws, we)
        if best is None:
            continue
        _, k, ws, we = best
        d, a, b = infix(tb, ob[ws:we])
        a, b = ws + a, ws + b
        s, e = free[k]
        while a > s and cuts_word(ob, a):
            a -= 1
        while b < e and cuts_word(ob, b):
            b += 1
        d = Levenshtein.distance(tb, ob[a:b])
        if b > a and d <= MAX_WRONG * len(tb):
            claim(i, k, d, a, b)
    return found


def marks(truth: str, ocr: str) -> Counter:
    """Words with right base letters, and of those the ones with right marks too."""
    tw, ow = WORD_RE.findall(truth), WORD_RE.findall(ocr)
    sm = difflib.SequenceMatcher(None, [base(w) for w in tw], [base(w) for w in ow], autojunk=False)
    c = Counter()
    for a, b, n in sm.get_matching_blocks():
        for t, o in zip(tw[a:a + n], ow[b:b + n]):
            c["words"] += 1
            c["right"] += t == o
            if base(t) != t:
                c["marked_words"] += 1
                c["marked_right"] += t == o
    return c


def place_field(value: str, lines: list[str]) -> list[int] | None:
    """The run of consecutive truth lines that holds the field, the first one."""
    sq = [squeeze(t) for t in lines]
    for i in range(len(lines)):
        for j in range(i, min(i + FIELD_LINES, len(lines))):
            if value in "".join(sq[i:j + 1]):
                return list(range(i, j + 1))
    return None


def date_spans(b: str) -> list[tuple[tuple[int, int], tuple[int, int, int]]]:
    """Where the whole dates in a base text are, and each as (day, month, year). The same
    mark must part day from month and month from year, and both must be real, so a
    document number ("01.10-2026", "2.0/2026") is no date."""
    out = []
    for m in DATE_RE.finditer(b):
        g = m.groups()
        d, mo, y = (int(re.sub(r"\D", "", x)) for x in ((g[0], g[2], g[3]) if g[0] else g[4:]))
        if 1 <= d <= 31 and 1 <= mo <= 12:
            out.append((m.span(), (d, mo, y)))
    return out


def dates(text: str) -> Counter:
    """The whole dates in a text, as (day, month, year), in either way of writing them."""
    return Counter(d for _, d in date_spans(base(text)))


def kinds(text: str) -> dict[str, list[str]]:
    """A text's numbers by what they look like, those inside a whole date left out:
    amount, code (a document number and the like), other, and short (1 or 2 digits)."""
    b = base(text)
    taken = [span for span, _ in date_spans(b)]
    out = {"amount": [], "code": [], "other": [], "short": []}
    for m in NUMBER_RE.finditer(b):
        if any(s <= m.start() < e for s, e in taken):
            continue
        d = re.sub(r"[.,]", "", m.group())
        if len(d) < 3:
            kind = "short"
        elif AMOUNT_RE.fullmatch(m.group()) or CURRENCY_RE.match(b, m.end()):
            kind = "amount"
        elif CODE_AFTER_RE.match(b, m.end()) or CODE_BEFORE_RE.search(b, 0, m.start()):
            kind = "code"
        else:
            kind = "other"
        out[kind].append(d)
    return out


def score_kinds(t: dict, ocr: str, read: Counter) -> Counter:
    """Recall by kind of number. Each truth number is counted once, an excused one not at all."""
    c = Counter()
    pool = read + Counter(x for x in kinds(ocr)["short"])
    left = Counter(t["numbers"])
    found = kinds(t["text"])
    for kind, values in found.items():
        for d in values:
            if len(d) >= 3:
                if left[d] <= 0:
                    continue
                left[d] -= 1
            c[f"kind_{kind}"] += 1
            if pool[d] > 0:
                pool[d] -= 1
                c[f"kind_{kind}_found"] += 1
    truth, got = dates(t["text"]), dates(ocr)
    c["dates"], c["dates_found"] = sum(truth.values()), sum((truth & got).values())
    c["split_over_lines"] = len(re.findall(r"\d[.,][ \t]*\n[ \t]*\d", t["text"]))
    return c


def score_page(t: dict, ocr: str | None, set_: str) -> tuple[Counter, Counter]:
    """The page's counts, and the numbers read anywhere on it."""
    c = Counter(pages=1, missing=ocr is None)
    ocr = collapse(normal(ocr or ""))
    truth, read = Counter(t["numbers"]), Counter(numbers(ocr))
    c["numbers"], c["numbers_found"] = sum(truth.values()), sum((truth & read).values())
    if set_ == "a":
        return c, read

    ob, at = strip(ocr)
    lines = truth_lines(t["text"])
    found = match_lines([x for x, _ in lines], ob)
    claimed = [False] * len(ob)
    in_lines = Counter()
    for (x, g), f in zip(lines, found):
        c[g + "_chars"] += len(base(x))
        if f is None:
            c[g + "_errors"] += len(base(x))
            continue
        d, s, e = f
        c[g + "_errors"] += d
        claimed[s:e] = [True] * (e - s)
        span = ocr[at[s]:at[e]]
        in_lines += Counter(numbers(span))
        if g == "text":
            c.update(marks(x, span))
    c["inserted"] = sum(1 for ch, k in zip(ob, claimed) if not k and ch.isalnum())
    excused = Counter(t.get("excused", []))
    c["numbers_read"] = sum((in_lines - excused).values())
    c["numbers_read_right"] = sum((truth & in_lines).values())
    c["numbers_read_outside"] = sum((read - excused).values()) - c["numbers_read"]
    c.update(score_kinds(t, ocr, read))
    small = Counter(short_numbers(t["text"]))
    c["short_words"], c["short_words_found"] = sum(small.values()), sum((small & Counter(short_numbers(ocr))).values())

    texts =[x for x, _ in lines]
    for name, v in (t.get("heading") or {}).items():
        v = squeeze(normal(v))
        if not v:
            continue
        run = place_field(v, texts)
        if run is None:
            c[f"field_{name}_unplaced"] += 1
            continue
        c[f"field_{name}"] += 1
        spans = [found[i] for i in run if found[i]]
        if spans:
            region = ocr[at[min(f[1] for f in spans)]:at[max(f[2] for f in spans)]]
            c[f"field_{name}_right"] += v in squeeze(region)
            c[f"field_{name}_base_right"] += squeeze(base(v)) in squeeze(base(region))
    return c, read


def score_sure(t: dict, ocr: str, unsure: list[int]) -> Counter:
    """How a run's sure numbers fare, for a run that marks unsure lines (tools/two_readers.py)."""
    per_line = [numbers(normal(x)) for x in ocr.split("\n")]
    excused = Counter(t.get("excused", []))      # reading one is neither right nor wrong
    sure = Counter(n for i, ns in enumerate(per_line) if i not in unsure for n in ns) - excused
    truth = Counter(t["numbers"])
    right = sure & truth
    left, near = truth - right, 0
    for n in (sure - right).elements():
        hit = next((x for x in left if left[x] > 0 and Levenshtein.distance(n, x) <= 2), None)
        if hit:
            left[hit] -= 1
            near += 1
    c = Counter(sure=sum(sure.values()), sure_of=sum((Counter(n for ns in per_line for n in ns) - excused).values()),
                sure_right=sum(right.values()), sure_not_in_truth=sum((sure - right).values()), sure_near_missed=near)
    if "text" in t:
        tl = [x for x in t["text"].split("\n") if x.strip()]
        for i, x in enumerate(normal(y) for y in ocr.split("\n")):
            ws = Counter(short_numbers(x))
            c["short_of"] += sum(ws.values())
            if i in unsure or not ws:
                continue
            c["short_sure"] += sum(ws.values())
            score, line = max((fuzz.ratio(x, y), y) for y in tl) if tl else (0, "")
            if score < SHORT_LINE and len(x.strip()) >= SHORT_PART:   # a cell or two of a table row
                score, line = max(((fuzz.partial_ratio(x, y), y) for y in tl if len(y) >= len(x)), default=(0, ""))
            if score < SHORT_LINE:
                c["short_sure_outside"] += sum(ws.values())
            else:
                c["short_sure_wrong"] += sum((ws - Counter(short_numbers(line))).values())
    return c


def share(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def report(c: Counter, set_: str) -> dict:
    r = {"pages": c["pages"], "missing": c["missing"],
         "numbers": {"truth": c["numbers"], "found": c["numbers_found"],
                     "recall": share(c["numbers_found"], c["numbers"])}}
    if c["sure_of"]:
        r["sure"] = {"numbers": c["sure"], "of": c["sure_of"], "coverage": share(c["sure"], c["sure_of"]),
                     "recall": share(c["sure_right"], c["numbers"]),
                     "not_in_truth": c["sure_not_in_truth"], "near_missed_truth": c["sure_near_missed"],
                     "near_missed_share": share(c["sure_near_missed"], c["sure"])}
        if set_ != "a":
            r["sure"]["wrong_when_sure"] = share(c["sure_not_in_truth"], c["sure"])
    if set_ == "a":
        return r
    r["numbers"].update(read=c["numbers_read"], precision=share(c["numbers_read_right"], c["numbers_read"]),
                        read_outside_lines=c["numbers_read_outside"])
    r["kinds"] = {k: {"truth": c[f"kind_{k}"], "found": c[f"kind_{k}_found"],
                      "recall": share(c[f"kind_{k}_found"], c[f"kind_{k}"])}
                  for k in ("amount", "code", "other", "short")}
    r["kinds"]["dates"] = {"truth": c["dates"], "found": c["dates_found"], "recall": share(c["dates_found"], c["dates"])}
    r["kinds"]["split_over_lines"] = c["split_over_lines"]
    r["short_words"] = {"truth": c["short_words"], "found": c["short_words_found"],
                        "recall": share(c["short_words_found"], c["short_words"])}
    if c["short_of"]:
        r["short_words"].update(sure=c["short_sure"], of=c["short_of"], coverage=share(c["short_sure"], c["short_of"]),
                                sure_outside=c["short_sure_outside"], sure_wrong=c["short_sure_wrong"],
                                wrong_when_sure=share(c["short_sure_wrong"], c["short_sure"] - c["short_sure_outside"]))
    r["base_letters"] = {"chars": c["text_chars"], "errors": c["text_errors"],
                         "error_rate": share(c["text_errors"], c["text_chars"]),
                         "inserted": c["inserted"], "inserted_rate": share(c["inserted"], c["text_chars"])}
    r["marks"] = {"words": c["words"], "accuracy": share(c["right"], c["words"]),
                  "marked_words": c["marked_words"], "marked_accuracy": share(c["marked_right"], c["marked_words"])}
    r["motto"] = {"chars": c["motto_chars"], "error_rate": share(c["motto_errors"], c["motto_chars"])}
    names = sorted({k.split("_")[1] for k in c if k.startswith("field_")})
    if names:
        r["fields"] = {n: {"total": c[f"field_{n}"], "unplaced": c[f"field_{n}_unplaced"],
                           "accuracy": share(c[f"field_{n}_right"], c[f"field_{n}"]),
                           "base_accuracy": share(c[f"field_{n}_base_right"], c[f"field_{n}"])}
                       for n in names}
    return r


def load_truth(set_: str, excuse: str | None) -> tuple[dict[tuple[str, int], dict], dict]:
    """(sha256, page) -> {stamp, numbers, text?, heading?}, and how the excuses fared."""
    set_path, truth_path = SETS[set_]
    pages, excused = {}, Counter()
    if set_ == "a":
        stamp = {(r["vi"]["sha256"], r["vi"]["page"]): r["vi"]["stamp"] for r in read_jsonl(set_path)}
        for r in read_jsonl(truth_path):
            k = (r["vi"]["sha256"], r["vi"]["page"])
            pages[k] = {"stamp": stamp[k], "numbers": list(r["numbers"])}
    elif set_ == "b":
        text = {(r["sha256"], r["page"]): r["text"] for r in read_jsonl(truth_path)}
        for r in read_jsonl(set_path):
            k = (r["sha256"], r["page"])
            pages[k] = {"stamp": r["stamp"], "text": text[k], "numbers": numbers(text[k])}
    else:
        for r in read_jsonl(set_path):
            if r["fate"] != "kept":
                continue
            with open(os.path.join(truth_path, f"{r['sha256'][:16]}-p{r['page']}.json"), encoding="utf-8") as fh:
                rec = json.load(fh)
            text = normal(rec["text"])
            pages[(r["sha256"], r["page"])] = {"stamp": r["stamp"], "text": text, "numbers": numbers(text),
                                               "heading": rec.get("heading")}
    for x in (read_jsonl(excuse) if excuse and os.path.exists(excuse) else []):
        nums = pages.get((x.get("sha256"), x.get("page")), {}).get("numbers", [])
        if str(x.get("number")) in nums:
            nums.remove(str(x["number"]))
            pages[(x["sha256"], x["page"])].setdefault("excused", []).append(str(x["number"]))
            excused["used"] += 1
        else:
            excused["stale"] += 1
            print(f"stale excuse, not in the truth: {json.dumps(x)}", file=sys.stderr)
    return pages, dict(excused)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set", choices=sorted(SETS))
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--pages", help="write each page's counts here, one line per run and page")
    ap.add_argument("--missed", help="set A: write the truth numbers no run found, for the hand check")
    ap.add_argument("--excuse", help="numbers taken out of the truth; default data/excuse-<set>.jsonl, if it exists")
    a = ap.parse_args()
    truth, excused = load_truth(a.set, a.excuse or f"data/excuse-{a.set}.jsonl")
    page_rows, unread = [], {k: Counter(t["numbers"]) for k, t in truth.items()}
    for run in a.runs:
        rows = read_jsonl(run)
        out = {(r["sha256"], r["page"]): r["text"] for r in rows}
        marked = {(r["sha256"], r["page"]): r["unsure"] for r in rows if "unsure" in r}
        groups = {"all": Counter(), "stamp": Counter(), "no_stamp": Counter()}
        for k, t in truth.items():
            c, read = score_page(t, out.get(k), a.set)
            if k in marked:
                c.update(score_sure(t, out[k], marked[k]))
            groups["all"].update(c)
            if t["stamp"] in ("yes", "no"):
                groups["stamp" if t["stamp"] == "yes" else "no_stamp"].update(c)
            page_rows.append({"run": run, "sha256": k[0], "page": k[1], "stamp": t["stamp"], **c})
            unread[k] -= read
        print(json.dumps({"run": run, "set": a.set, **({"excused": excused} if excused else {}),
                          **{g: report(c, a.set) for g, c in groups.items()}}, ensure_ascii=False))
    if a.pages:
        write_jsonl(a.pages, page_rows)
    if a.missed:
        write_jsonl(a.missed, [{"sha256": k[0], "page": k[1], "number": n}
                               for k, left in unread.items() for n in sorted(left.elements())])


if __name__ == "__main__":
    main()
