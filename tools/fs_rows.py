"""Rows of the three primary statements, rebuilt from a draft's line boxes (FS1 and FS2).

Profile 2 (METHOD.md, "Financial statements"). A draft (tools/read_pdf.py) keeps each
line's text, whether it is sure, and its box [x0, y0, x1, y1] on the 300 dpi page.

- **Which statement** (FS1): the topmost title line in a page's top fifth names it (a
  line in capitals holding the name, or one that starts with it; not the notes'):
  "BẢNG CÂN ĐỐI KẾ TOÁN" or, in the form used from 2026, "BÁO CÁO TÌNH HÌNH TÀI CHÍNH"
  starts B01; "KẾT QUẢ HOẠT ĐỘNG KINH DOANH" (or "… SẢN XUẤT KINH DOANH") B02; "LƯU
  CHUYỂN TIỀN TỆ" B03; a title printed over two lines is read joined. A title in capitals
  read a letter off ("KIẾ TOÁN", "TIỀN TIỆ") still names it: each of its words one letter
  off at most, and no more than one word in three. A page whose size the
  draft does not record is its image's; with no image, A4, landscape when its lines run
  past a portrait page's width. A page with
  the same title right after it ("(tiếp theo)"), or with no title but row codes,
  continues it; with fewer than 5 codes, only when they rise past the last code read
  before, one stray number aside (the last totals, alone over the signatures). A numbered note naming a
  statement (its items "ngoài bảng" / "ngoài báo cáo tình hình tài chính") is no title. A page with no title read but 8+ codes, nearly all 100 or more, is a
  balance sheet, if no statement came before it and it holds no prose (a letter
  explaining a restatement has codes too): only B01's codes run past 99, and B01
  comes first. Before the notes, any other page with no title read but a statement's form
  ("Mẫu số B 02 - DN") or codes that are not the statement's before it starts the
  one they name (untitled): 5+ codes nearly all 100 or more are B01's; nearly all under
  100, B02's or B03's by the codes only one of them prints. Codes under 100 after a
  balance sheet that name neither, or a form that the codes contradict, end the
  statement before: the page is left out rather than filed under a guess. The form ("Mẫu số B 01 - DN") is kept when printed, often it is not;
  printed with its digits unread ("Mẫu số B [?]-DN"), it is kept as read;
  the unit is one of VND / đồng, in nghìn, triệu or tỷ when so scaled, after "Đơn vị
  tính" or alone on a line over the columns.
- **Columns:** numbers are right-aligned, so value columns are the clusters of the right
  edges of the page's numbers, left to right. A page that goes on with a statement and
  finds fewer columns of its own, all in the page before's places, takes that page's
  (carried_columns), when nearly all its codes are the statement's and come after the
  last code it read. Their headings are the lines above the
  first row that fall in a column.
- **Rows:** a row is anchored on its code (a line of 2 or 3 digits in the leftmost such
  column, left of the values; notes form a column of their own to its right; a line read as [?] in the codes' column anchors a row whose code is None; a label line
  ending in the codes' column with a code is split there). Each figure goes to one row (owners): beside a label's continuing line,
  the code above; else the code nearest once its column's offset from the codes is taken
  off (a page scanned askew), within the code line's band. A figure
  no code reaches goes to the one code beside it with no figure, on its label's evidence
  (a label wrapped onto an abbreviation, "TNDN"; a tall cell with its code at the bottom).
  Words to its left are its label; a short line between the code and the first value is its note.
  On a page marked coarse (read_pdf.coarse_pages: an image too coarse to read figures
  from) every figure is kept unsure, its value empty: the readers' few sure ones there
  are not to be trusted, and are not guessed at.

- **Text-layer pages** hold no boxes in a draft; given the PDF, their words are placed
  again from it (layer_lines), and every line of them is sure.

A value is a whole number with its sign: "(1.234)" and "-1.234" are negative, "-" alone is
zero. A value on an unsure line is None ([?] in the draft): empty, never guessed.

Usage: .venv/bin/python -I tools/fs_rows.py some.draft.json [its.pdf]   (prints a summary;
       the PDF, for text-layer pages, defaults to cache/pdfs/<the draft's file>)
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python -I too
from sort_kinds import fold  # noqa: E402

TITLES = {"B01": r"bang can doi ke toan|bao cao tinh hinh tai chinh",
          "B02": r"ket \w{2,3} (?:hoat dong )?(?:san xuat )?kinh doanh|\w{3} \w{3} ket \w{2,3} hoat dong kinh d\w{1,3}nh",
          "B03": r"luu chuy\w*n ti\w*n te"}       # "CHUYẾỄN": a misread vowel still folds near
NEAR_TITLES = {"B01": ["bang can doi ke toan", "bao cao tinh hinh tai chinh"],   # TITLES word by word,
               "B02": [f"ket qua {a}{b}kinh doanh" for a in ("", "hoat dong ") for b in ("", "san xuat ")],
               "B03": ["luu chuyen tien te"]}                   # for a title read a letter off
NEW_FORM = r"bao cao tinh hinh tai chinh"   # the balance sheet's name in the form for 2026 on
FORM_RE = re.compile(r"\bB\s*0\s*([123])\s*([a-z])?\s*[-–/]\s*(DN(?:\s*/\s*HN)?|CTCK|TCTD|[A-Z/]+)", re.I)
FORM_SEEN = re.compile(r"\bm[aâẫ]u\s*(?:s\S\s*)?b\s*(?:\[\?\]|\d)\S*\s*[a-z]?\s*[-–/]\s*[A-Z/]+", re.I)  # digits unread
UNIT_RE = re.compile(r"đơn\s*vị\s*(?:tính)?\s*:?\s*((?:nghìn|ngàn|triệu|tỷ)?\s*(?:VND|VNĐ|đồng))", re.I)
UNIT_ALONE = re.compile(r"\(?\s*((?:nghìn|ngàn|triệu|tỷ)?\s*(?:VND|VNĐ|đồng))\s*\)?", re.I)  # over the columns
CODE_RE = re.compile(r"\d{2,3}")
LEAD_CODE_RE = re.compile(r"(\d{2,3}|\[\?\])\s+(\S.*)")
TRAIL_LABEL_RE = re.compile(r"(.*[^\W\d_].*?)\s+(\d{2,3}|\[\?\])")   # a label, then its code
TRAIL_CODE_RE = re.compile(r"(\d{2,3})\s*[.,]")     # a code with a stop read after it
NOTE_RE = re.compile(r"(?:[IVX]{1,4}\.?\s?)?\d{1,2}(?:\.\d{1,2})*[a-z]?(?:\s*[,;&]\s*\d{1,2}(?:\.\d{1,2})*)*")
NUM_RE = re.compile(r"\(?-?\s*\d{1,3}(?:[.,]\s?\d{3})+\s*\)?|\(?-?\s*\d{1,3}\s*\)?|-")
PORTRAIT = (2480, 3508)   # an A4 page at 300 dpi
TOP_SHARE = 0.2           # a statement's title is in the top part of its page
CENTRED = 0.15            # share of the page's width a centred title's middle is off its centre
COL_GAP = 120             # px at 300 dpi between two columns' right edges
MIN_COL = 3               # values a column needs on a page
SIZE_SLACK = 0.03         # pages this near in height are one size: scans of a paper differ by a few
                          # pixels; a turned page differs by a third, A4 and Letter by 6%
WIDE, NARROW = 150, 100   # px: a [?] box as wide as an amount; as narrow as a code
CODE_GAP = 40             # px between two code lines' right edges in one column
PAIR_OFF = 60             # px a merged pair's estimated edges may stand off their columns
TILT_PAIRS = 5             # pairs of figures a page's tilt is measured on, at least (tilt_of)
REPEAT = 0.5              # share of a titled page's codes already seen: the statement again
MIN_CODES = 5             # codes that tell a page's statement with no title read
B02_ONLY = {"51", "52", "62", "71"}                 # tax and earnings per share lines
B03_ONLY = {f"{c:02d}" for c in [*range(3, 10), *range(12, 18), *range(33, 38)]}   # adjustments, working capital, financing
NOTES = r"thuyet minh bao cao tai chinh"           # a page whose top says it: the notes begin
PROSE, MAX_PROSE = 60, 2  # chars of a sentence's line; more such lines: a letter, not a statement


def value_of(text: str) -> int | None:
    """A printed amount as a signed whole number; None when it is not one."""
    t = text.strip().replace(" ", "")
    if t in ("-", "–"):
        return 0
    if "[?]" in t or not t:
        return None
    neg = t.startswith("(") and t.endswith(")") or t.startswith("-")
    t = t.strip("()-")
    if not re.fullmatch(r"\d{1,3}(?:[.,]\d{3})*", t):
        return None
    v = int(re.sub(r"[.,]", "", t))
    return -v if neg else v


def is_amount(text: str) -> bool:
    t = text.strip()
    return bool(NUM_RE.fullmatch(t)) and (value_of(t) is not None) and not CODE_RE.fullmatch(t)


def in_cell(text: str) -> bool:
    """Text a value cell can hold: an amount, [?], or 2-3 bare digits, a small figure
    that only its place tells from a code (the caller checks it ends on a column)."""
    return is_amount(text) or text == "[?]" or bool(CODE_RE.fullmatch(text))


def mid(b) -> float:
    return (b[1] + b[3]) / 2


def capitals(text: str) -> bool:
    """A title is printed in capitals; a sentence that names a statement is not."""
    letters = [ch for ch in text if ch.isalpha()]
    return len(letters) >= 8 and sum(ch.isupper() for ch in letters) >= 0.8 * len(letters)


def title_of(lines: list[dict], height: float, width: float = PORTRAIT[0]) -> tuple[str, str] | None:
    """(statement, the title as folded) from the page's topmost title line.

    A line in ordinary case must also be centred: a table's row label can start with a
    statement's name ("Bảng cân đối kế toán hợp nhất", in a letter on a restatement).
    A title printed over two lines ("Báo cáo kết quả hoạt động" / "kinh doanh riêng") is
    read as the line joined to the one right under it."""
    top = sorted((ln for ln in lines if ln["box"][1] < TOP_SHARE * height), key=lambda ln: ln["box"][1])
    for i, ln in enumerate(top):
        for t, text, box in [(fold(ln["text"]), ln["text"], ln["box"]), *_joined(ln, top[i + 1:])]:
            if re.search(r"thuyet minh|ngoai (?:bang|bao cao)", t):   # the notes, and their off-balance items
                break
            for k, pat in TITLES.items():
                if (re.search(pat, t) if capitals(text) else
                        re.match(rf"(?:bao cao )?(?:{pat})", t) and abs((box[0] + box[2]) / 2 - width / 2) < CENTRED * width):
                    return k, t
            if capitals(text) and (k := near_title(t)):
                return k, t
    return None


def one_off(a: str, b: str) -> bool:
    """Whether two words differ by at most one letter: one added, dropped or changed."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = sorted((a, b), key=len)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def near_title(t: str) -> str | None:
    """The statement a folded line names with a letter misread ("BẢNG CÂN ĐỐI KIẾ TOÁN"):
    its words in a run, each one letter off at most, and no more than one word in three off."""
    words = t.split()
    for k, titles in NEAR_TITLES.items():
        for title in titles:
            want = title.split()
            for i in range(len(words) - len(want) + 1):
                got = words[i:i + len(want)]
                if all(map(one_off, got, want)) and sum(g != w for g, w in zip(got, want)) <= len(want) // 3:
                    return k
    return None


def _joined(ln: dict, below: list[dict]) -> list[tuple[str, str, list]]:
    """The line and the next one under it, as one title (folded, as printed, their box): the
    next starts within a line's height under it and overlaps it across the page."""
    h = ln["box"][3] - ln["box"][1]
    for nx in below:
        if nx["box"][1] > ln["box"][3] + h:
            break
        if nx["box"][1] > ln["box"][1] + h / 2 and nx["box"][0] < ln["box"][2] and nx["box"][2] > ln["box"][0]:
            text = ln["text"] + " " + nx["text"]
            box = [min(ln["box"][0], nx["box"][0]), ln["box"][1], max(ln["box"][2], nx["box"][2]), nx["box"][3]]
            return [(fold(text), text, box)]
    return []


def same_size(a: float | None, b: float) -> bool:
    """Two pages' heights are one page size (SIZE_SLACK), so a code column found on one
    stands where it was on the other."""
    return a is not None and abs(a - b) <= SIZE_SLACK * max(a, b)


def size_of(page: dict, lines: list[dict], images: str | None = None) -> tuple[float, float]:
    """A page's (width, height): as the draft records it (a turned page), else its image's,
    else A4, landscape when its lines run past a portrait page's width (a landscape page
    read as it lies). The guess is wrong when junk at a scan's edge reaches past the width."""
    if page.get("size"):
        return tuple(page["size"])
    png = os.path.join(images, page["key"] + ".png") if images and page.get("key") else None
    if png and os.path.exists(png):
        from PIL import Image
        with Image.open(png) as im:
            return im.size
    if any(ln["box"][2] > PORTRAIT[0] for ln in lines if ln.get("box")):
        return PORTRAIT[1], PORTRAIT[0]
    return PORTRAIT


def columns_of(lines: list[dict], unread: bool = False) -> list[float]:
    """Right edges of the value columns, left to right. On a page with no sure amount,
    the wide [?] boxes (amounts not read) place them, so its cells are found and left empty."""
    xs = sorted(ln["box"][2] for ln in lines if ln["box"] and (unread or is_amount(ln["text"])
                and len(re.sub(r"\D", "", ln["text"])) >= 4))
    groups, cur = [], []
    for x in xs:
        if cur and x - cur[-1] > COL_GAP:
            groups.append(cur)
            cur = []
        cur.append(x)
    if cur:
        groups.append(cur)
    cols = [sorted(g)[len(g) // 2] for g in groups if len(g) >= MIN_COL]
    if not cols and not unread:                     # a page read as nothing sure: place by the unread
        return columns_of([ln for ln in lines if ln["text"].strip() == "[?]"
                           and ln["box"][2] - ln["box"][0] >= WIDE], unread=True)
    return cols


def code_column(lines: list[dict], cols: list[float], known: tuple | None = None) -> list[dict]:
    """The row codes: the column of 2-3 digit lines, by right edge, that holds the most.

    Note references ("21", "V.1") are short numbers too, in a column of their own between
    the codes and the values; some statements also number their rows left of the label
    ("14 Tổng lợi nhuận..."). Codes run on every row, notes and two-digit row numbers on
    some, so the fullest column is the codes'; of two as full, the leftmost. known:
    the codes' (x0, x1) on the statement's page before; a page that continues it may hold
    only a few rows, so there any code in that place counts, unless the page's own fullest
    column holds more (a stray number on the page before can shift that place). That place
    is wide enough to take a notes column beside the codes; of what it takes, the fullest
    column is the codes'."""
    cand = sorted((ln for ln in lines if CODE_RE.fullmatch(ln["text"].strip())
                   and ln["box"][2] < min(cols) - 150), key=lambda ln: ln["box"][2])
    got = [ln for ln in cand if known[0] - 60 <= (ln["box"][0] + ln["box"][2]) / 2 <= known[1] + 60] if known else []
    if not cand:                                    # no code read: the narrow [?] boxes
        cand = sorted((ln for ln in lines if ln["text"].strip() == "[?]"
                       and ln["box"][2] - ln["box"][0] < NARROW and ln["box"][2] < min(cols) - 150),
                      key=lambda ln: ln["box"][2])
    groups = [g for g in by_column(cand) if len(g) >= MIN_COL]
    best = max(groups, key=len) if groups else []     # max keeps the first of equals
    got = max(by_column(got), key=len, default=[])  # the place before may reach a notes column
    return got if got and len(got) >= len(best) else best   # the place before, unless a stray widened it


def by_column(lines: list[dict]) -> list[list[dict]]:
    """Lines sorted by right edge, cut into columns where two edges are CODE_GAP apart."""
    groups, cur = [], []
    for ln in lines:
        if cur and ln["box"][2] - cur[-1]["box"][2] > CODE_GAP:
            groups.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        groups.append(cur)
    return groups


def split_codes(lines: list[dict]) -> list[dict]:
    """A code read on one line with its label ("110 I. Tiền", "[?] A. TÀI SẢN") as two, and
    a code read with a stop after it ("11.") as the code.

    The split point's x is estimated from the line's mean character width."""
    out = []
    for ln in lines:
        t = TRAIL_CODE_RE.fullmatch(ln["text"].strip())
        if t:                                       # "11." : a stop read after a code; the column decides
            ln = {**ln, "text": t.group(1), "trimmed": True}
        m = LEAD_CODE_RE.match(ln["text"].strip())
        if not m or not re.search(r"[^\W\d_]", m.group(2)):
            out.append(ln)
            continue
        x0, y0, x1, y1 = ln["box"]
        cw = (x1 - x0) / max(len(ln["text"].strip()), 1)
        cut = x0 + cw * len(m.group(1))
        out += [{**ln, "text": m.group(1), "box": [x0, y0, cut, y1], "split": True},
                {**ln, "text": m.group(2), "box": [cut + cw, y0, x1, y1], "split": True}]
    return out


def split_amounts(lines: list[dict], cols: list[float]) -> list[dict]:
    """Two amounts read as one line ("(1.234.567) 8.901.234") as two, one per column.

    Taken only when the text is exactly two well-formed amounts of 4 digits or more and
    each part's right edge, the first's estimated from the line's mean character width,
    lies within PAIR_OFF of two neighbouring columns; otherwise the line is left as read.
    A dash beside an amount is left: "- 1.234" can be a minus sign."""
    out = []
    for ln in lines:
        m = re.fullmatch(r"(\S+)\s+(\S+)", ln["text"].strip())
        if not (m and len(cols) >= 2 and all(
                is_amount(g) and len(re.sub(r"\D", "", g)) >= 4 for g in m.groups())):
            out.append(ln)
            continue
        x0, y0, x1, y1 = ln["box"]
        cw = (x1 - x0) / len(ln["text"].strip())
        cut = x0 + cw * len(m.group(1))
        i = min(range(len(cols)), key=lambda k: abs(cols[k] - cut))
        if not (i + 1 < len(cols) and abs(cols[i] - cut) <= PAIR_OFF
                and abs(cols[i + 1] - x1) <= PAIR_OFF):
            out.append(ln)
            continue
        out += [{**ln, "text": m.group(1), "box": [x0, y0, cut, y1], "split": True},
                {**ln, "text": m.group(2), "box": [cut + cw, y0, x1, y1], "split": True}]
    return out


SPACED_RE = re.compile(r"(\(?-?\s*)(\d{1,3}(?: \d{3})+)(\s*\)?)")
WIDE_CHARS = 1.5          # a line's mean character width over its page's usual: two figures read as one


def spaced_amounts(lines: list[dict]) -> list[dict]:
    """Amounts grouped by spaces ("1 234 567") as grouped by stops, where the page shows
    they are figures; other lines as read.

    The shape alone does not tell: codes read as one line ("100 110 120") and two figures
    read as one line ("1 234 567 890" for "1 234" and "567 890") look the same. So a
    line is taken only when its mean character width is not over WIDE_CHARS times its
    page's usual (two figures read as one span the gap between them), its right edge lies
    within PAIR_OFF of a value column that this makes, and that column lies right of the
    page's codes, the fullest column of 2-3 digit lines. A page with no such codes
    keeps its lines as read: there is nothing to tell a figure's place from a code's."""
    cand = [ln for ln in lines if ln.get("box") and SPACED_RE.fullmatch(ln["text"].strip())]
    codes = [g for g in by_column(sorted((ln for ln in lines if ln.get("box")
                                          and CODE_RE.fullmatch(ln["text"].strip())),
                                         key=lambda ln: ln["box"][2])) if len(g) >= MIN_COL]
    if not cand or not codes:
        return lines
    edge = max(ln["box"][2] for ln in max(codes, key=len)) + 150

    def cw(ln):
        return (ln["box"][2] - ln["box"][0]) / len(ln["text"].strip())
    def stops(text):                                # the digits' spaces as stops; a sign as read
        sign, digits, close = SPACED_RE.fullmatch(text.strip()).groups()
        return sign + digits.replace(" ", ".") + close
    usual = sorted(cw(ln) for ln in cand)[len(cand) // 2]
    as_read = {id(ln): {**ln, "text": stops(ln["text"])}
               for ln in cand if cw(ln) <= WIDE_CHARS * usual and ln["box"][2] > edge}
    if not as_read:
        return lines
    trial = [as_read.get(id(ln), ln) for ln in lines]
    cols = [c for c in columns_of(trial) if c > edge]
    return [{**as_read[id(ln)], "spaced": True}
            if id(ln) in as_read and cols and min(abs(c - ln["box"][2]) for c in cols) <= PAIR_OFF
            else ln for ln in lines]


def continues(text: str) -> bool:
    """A label's second line: it starts in lower case ("nộp", "cung cấp dịch vụ")."""
    m = re.search(r"[^\W\d_]", text)
    return bool(m) and m.start() == 0 and text[0].islower()


def tilt_of(lines: list[dict], cols: list[float]) -> float:
    """A page's tilt, px of height per px of width, positive where lines fall to the right.
    One printed line's amounts in two neighbouring columns are a column apart, close enough
    to pair without doubt (nearest in height, within half a line); the tilt is the median of
    their slopes, from TILT_PAIRS pairs, else 0. Dashes are left out: a bar sits at its own
    height, not the digits'. A page scanned askew puts a figure far right of its code a line
    higher or lower; owners() and read_pdf's dash cells follow the tilt."""
    if len(cols) < 2:
        return 0.0
    figs = [[] for _ in cols]
    for ln in lines:
        b, t = ln.get("box"), ln["text"].strip()
        if b and is_amount(t) and t not in ("-", "–") and b[2] - b[0] >= 60:
            k = min(range(len(cols)), key=lambda i: abs(cols[i] - b[2]))
            if abs(cols[k] - b[2]) <= COL_GAP:
                figs[k].append(b)
    hs = sorted(b[3] - b[1] for f in figs for b in f)
    if not hs:
        return 0.0
    h = hs[len(hs) // 2]
    slopes = []
    for k in range(len(cols) - 1):
        for b in figs[k]:
            near = min(figs[k + 1], key=lambda o: abs(mid(o) - mid(b)), default=None)
            if near is not None and abs(mid(near) - mid(b)) <= 0.5 * h:
                slopes.append((mid(near) - mid(b)) / ((near[0] + near[2]) / 2 - (b[0] + b[2]) / 2))
    slopes.sort()
    return round(slopes[len(slopes) // 2], 4) if len(slopes) >= TILT_PAIRS else 0.0


def owners(lines: list[dict], cols: list[float], codes: list[dict], tilt: float = 0.0) -> list[tuple]:
    """Each figure's row: (column, figure line, code line), one row per figure.

    A figure is printed beside its label's last line, so it may sit a line below its code
    when the label wraps; and on a page scanned askew, a column far from the codes sits
    higher or lower than they do. So, column by column from the codes outwards, the
    column's offset from the codes is the median of its figures' offsets from the code
    the previous column's offset points them to (a skew grows with x; the first column
    starts from none). The page's tilt (tilt_of) is where each
    column's search starts: the offset before it, plus the tilt times the way across, so a
    column half a line off its codes is not taken for the row below. A figure beside a label line that continues the one above belongs
    to the nearest code above it; any other, to the nearest code once the offset is taken
    off, within the band a code's own line allows. Of two figures one row gets in a
    column, the nearer is kept."""
    if not codes:
        return []
    x_code = max(c["box"][2] for c in codes)
    figs = [[] for _ in cols]
    labels = []
    for ln in lines:
        b, text = ln["box"], ln["text"].strip()
        if not b or any(ln is c for c in codes):
            continue
        if b[0] > min(c["box"][2] for c in codes) and in_cell(text):
            k = min(range(len(cols)), key=lambda i: abs(cols[i] - b[2]))
            if abs(cols[k] - b[2]) <= COL_GAP:
                figs[k].append(ln)
                continue
        if b[2] < min(cols) - 100:
            labels.append(ln)

    line_h = sorted(c["box"][3] - c["box"][1] for c in codes)[len(codes) // 2]

    def nearest(y):
        return min(codes, key=lambda c: abs(mid(c["box"]) - y))

    def wrapped(f):
        """Beside a label line that continues the one above (the leftmost line at its height).
        Measured on a text line's height at least: a dash's box is a few px tall."""
        h = max(f["box"][3] - f["box"][1], line_h)
        beside = [ln for ln in labels if abs(mid(ln["box"]) - mid(f["box"])) <= 0.35 * h]
        return bool(beside) and continues(min(beside, key=lambda ln: ln["box"][0])["text"].strip())

    out, shift, shifts = [], 0.0, {}
    x_was = sorted((c["box"][0] + c["box"][2]) / 2 for c in codes)[len(codes) // 2]
    for k in sorted(range(len(cols)), key=lambda i: abs(cols[i] - x_code)):
        xf = sorted((f["box"][0] + f["box"][2]) / 2 for f in figs[k])
        x_at = xf[len(xf) // 2] if xf else cols[k]
        shift += tilt * (x_at - x_was)
        x_was = x_at
        offs = sorted(mid(f["box"]) - mid(nearest(mid(f["box"]) - shift)["box"])
                      for f in figs[k] if not wrapped(f))
        if offs:
            shift = offs[len(offs) // 2]
        got = {}
        for f in figs[k]:
            y = mid(f["box"]) - shift
            c = nearest(y)
            h = c["box"][3] - c["box"][1]
            reach = 0.85 * h                          # a code's line and a third of one each side
            if abs(mid(c["box"]) - y) > 0.35 * h and wrapped(f):
                above = [c2 for c2 in codes if mid(c2["box"]) < y]
                if above:
                    c, reach = above[-1], 3.5 * h     # a label of up to four lines
            if abs(mid(c["box"]) - y) > reach:
                continue
            if f["box"][0] <= c["box"][2]:
                continue
            d = abs(mid(c["box"]) - y)
            if id(c) not in got or d < got[id(c)][0]:
                got[id(c)] = (d, f, c)
        out += [(k, f, c) for _, f, c in got.values()]
        shifts[k] = shift
    return out + orphans(figs, codes, labels, out, shifts)


ITEM_RE = re.compile(r"(?:[IVX]{1,4}|\d{1,2}(?:\.\d{1,2})?|[A-Za-z])\s?[.)]\s|[-–+*]\s")


def orphans(figs: list[list], codes: list[dict], labels: list[dict], out: list,
            shifts: dict) -> list[tuple]:
    """Figures no code reached, given to the one row they can belong to.

    A label that wraps puts its figures on its last line, which need not start in lower
    case ("TNDN"); a tall cell puts them at its top, with the code alone at its bottom.
    Either way the figures sit between two codes, or below a page's last, beyond a code
    line's reach. They go to the code above or below them that holds no figure in any
    column, when exactly one of the two is so, within a label of four lines, and only on
    the label's evidence: for the code above, the label line beside the figures continues
    a label (it starts in lower case or with a short abbreviation, never with an item's
    number or a dash) or starts under the code's label text, past its item's number; for
    the code below, one label line stands beside the figures, starting no item, and
    beside the code none (a note or a bare item number aside) or the label's last line,
    in lower case, at most a line and a half below.
    A row whose code was not read leaves figures between two codes too, and a neighbour
    whose dashes were not read looks free; its label starting a new item keeps it so.
    Given so, a row is free no longer, and the figures left are gone over again: of two
    wrapped rows one under the other, each with its figures above its code, the second's
    stand between two free codes until the first has its own."""
    owned = {id(f) for _, f, _ in out}
    full = {id(c) for _, _, c in out}

    def bare(ln):
        m = ITEM_RE.match(text(ln) + " ")
        return bool(m) and m.end() >= len(text(ln))

    xs = sorted(c["box"][0] for c in codes)
    labels_left = bool(xs) and 2 * sum(ln["box"][2] <= xs[len(xs) // 2] for ln in labels) > len(labels)

    def beside(y, h, code=None):
        """The leftmost label line level with y; level with a code, an item's bare number
        and, where labels stand left of the codes, a note right of it are passed over."""
        near = [ln for ln in labels if abs(mid(ln["box"]) - y) <= 0.5 * h
                and not (code and (bare(ln) or labels_left and ln["box"][0] >= code["box"][2]))]
        return min(near, key=lambda ln: ln["box"][0]) if near else None

    def text(ln):
        return ln["text"].strip() if ln else None

    def carries_on(t):
        if t is None or ITEM_RE.match(t):
            return False
        first = t.split()[0] if t.split() else ""
        return continues(t) or (first.isupper() and len(first) <= 5 and len(t.split()) <= 3)

    def under_text(ln, c, h):
        """The line starts under the text of the code's label, past its item's number: the
        label's next line, whatever its case ("Nhà nước")."""
        if ln is None or ITEM_RE.match(text(ln)):
            return False
        near = sorted((l for l in labels if abs(mid(l["box"]) - mid(c["box"])) <= 0.5 * h),
                      key=lambda l: l["box"][0])
        if not near:
            return False
        first, t = near[0], text(near[0])
        m = ITEM_RE.match(t + " ")
        if not m:
            return False
        if m.end() >= len(t):                       # the number alone, its text a line of its own
            if len(near) < 2:
                return False
            start = near[1]["box"][0]
        else:                                       # the number and text on one line
            start = first["box"][0] + (first["box"][2] - first["box"][0]) * m.end() / len(t)
        return abs(ln["box"][0] - start) <= 0.6 * h and ln["box"][0] > first["box"][0] + 0.5 * h

    def evidence(f, y, c):
        h = c["box"][3] - c["box"][1]
        at_figs, at_code = beside(y, h), beside(mid(c["box"]), h, c if mid(c["box"]) > y else None)
        if mid(c["box"]) < y:                        # the code above: a wrapped label
            return carries_on(text(at_figs)) or under_text(at_figs, c, h)
        if at_figs is None:
            return False
        # a tall cell: the code alone at its bottom, or on its label's last line, which then
        # starts in lower case ("hạn"; a heading in capitals beside a total is no such line)
        # with the label's first line, no item of its own, at most a line and a half above
        return at_code is None or (continues(text(at_code)) and not ITEM_RE.match(text(at_figs))
                                   and mid(c["box"]) - y <= 2.5 * h)

    given = []
    while True:
        got = {}
        for k, col in enumerate(figs):
            for f in col:
                if id(f) in owned:
                    continue
                y = mid(f["box"]) - shifts.get(k, 0.0)
                above = [c for c in codes if mid(c["box"]) < y][-1:]
                below = [c for c in codes if mid(c["box"]) >= y][:1]
                free = [c for c in above + below if id(c) not in full
                        and abs(mid(c["box"]) - y) <= 3.5 * (c["box"][3] - c["box"][1])
                        and f["box"][0] > c["box"][2]]
                free = [c for c in free if evidence(f, y, c)] if len(free) == 1 else []
                if len(free) == 1 and (len(above + below) == 2 or free == above):   # or a page's last
                    got.setdefault((k, id(free[0])), []).append((f, free[0]))
        new = [(k, *v[0]) for (k, _), v in got.items() if len(v) == 1]   # two for one cell: neither
        if not new:
            return given
        given += new
        owned |= {id(f) for _, f, _ in new}
        full |= {id(c) for _, _, c in new}


def trail_codes(lines: list[dict], codes: list[dict], x0: float, x1: float) -> list[dict]:
    """A label read on one line with the code printed right of it ("Doanh thu ... 10"): the
    line ends in the codes' column (x0, x1) with 2-3 digits, so the code is split off and
    joins the codes. Only where the codes' column is known, so a note number stays a label's."""
    out = []
    for ln in lines:
        m = TRAIL_LABEL_RE.fullmatch(ln["text"].strip())
        b = ln["box"]
        if ln in codes or not m or not b or not x0 <= b[2] <= x1 or b[0] >= x0:
            out.append(ln)
            continue
        cw = (b[2] - b[0]) / max(len(ln["text"].strip()), 1)
        cut = b[2] - cw * len(m.group(2))
        code = {**ln, "text": m.group(2), "box": [cut, b[1], b[2], b[3]], "split": True}
        out += [{**ln, "text": m.group(1), "box": [b[0], b[1], cut - cw, b[3]], "split": True}, code]
        codes.append(code)
    return out


def rows_of(lines: list[dict], cols: list[float], known: tuple | None = None,
            tilt: float = 0.0) -> list[dict]:
    """Rows anchored on codes. A row's label is its words left of the values (left of the
    code, or right of it when the code comes first); its note is a short reference."""
    if not cols:
        return []
    codes = code_column(lines, cols, known)
    if codes:                                       # a code read as [?] still anchors its row
        x0 = min(c["box"][0] for c in codes) - 30
        x1 = max(c["box"][2] for c in codes) + 30
        lines = trail_codes(lines, codes, x0, x1)
        codes += [ln for ln in lines if ln["text"].strip() == "[?]"
                  and x0 <= (ln["box"][0] + ln["box"][2]) / 2 <= x1]
    codes = sorted(codes, key=lambda ln: mid(ln["box"]))
    owner = owners(lines, cols, codes, tilt)
    rows = []
    for c in codes:
        y0, y1 = c["box"][1], c["box"][3]
        h = y1 - y0
        band = (y0 - 0.35 * h, y1 + 0.35 * h)
        values = [None] * len(cols)
        label, note = [], []
        for k, f, o in owner:
            if o is c:
                text = f["text"].strip()
                values[k] = {"text": text, "value": value_of(text) if f["sure"] else None,
                             "sure": f["sure"], "box": f["box"],
                             **({"look": f["look"]} if f.get("look") else {})}
        for ln in lines:
            b = ln["box"]
            if not b or ln is c or not band[0] <= mid(b) <= band[1]:
                continue
            text = ln["text"].strip()
            if b[0] > c["box"][2] and in_cell(text) \
                    and min(abs(x - b[2]) for x in cols) <= COL_GAP:
                continue                            # a figure: placed by owners()
            if b[2] >= min(cols) - 100:
                continue
            if b[0] >= c["box"][2] and NOTE_RE.fullmatch(text):
                note.append(text)
            else:
                label.append((b[0], text))
        code = c["text"].strip()
        rows.append({"code": code if CODE_RE.fullmatch(code) else None,
                     "code_sure": c["sure"] and bool(CODE_RE.fullmatch(code)),
                     "y": round(mid(c["box"])), "code_box": [round(v) for v in c["box"]],
                     "label": " ".join(t for _, t in sorted(label)), "note": " ".join(note),
                     "values": values})
    return rows


def headings_of(lines: list[dict], cols: list[float], first_row_y: float | None) -> list[str]:
    out = [[] for _ in cols]
    for ln in lines:
        b = ln["box"]
        if not b or first_row_y is None or mid(b) >= first_row_y - 20 or mid(b) < first_row_y - 500:
            continue
        for k, x in enumerate(cols):
            if b[0] <= x <= b[2] + 60 and b[2] >= x - 450:
                out[k].append((b[1], ln["text"].strip()))
                break
    return [" ".join(t for _, t in sorted(o)) for o in out]


def layer_lines(pdf: str) -> dict[int, tuple[list[dict], tuple[float, float]]]:
    """Per page of a PDF: its text layer as lines with boxes at 300 dpi, and the page's size.

    A draft keeps a text-layer page as text alone. Its words are placed again here
    (pdftotext, as the page is shown), joined into lines by their middle's height, and a
    line is cut where a gap between words is wider than its type is high: a label, a note
    and two amounts on one printed row become four lines, as OCR gives them. These lines
    are exact, so all are sure."""
    from label_pages import read_pages
    k = 300 / 72
    out = {}
    for n, pg in enumerate(read_pages(pdf), 1):
        words = sorted(pg["words"], key=lambda w: ((w[1] + w[3]) / 2, w[0]))
        rows: list[list] = []
        for w in words:
            m, h = (w[1] + w[3]) / 2, w[3] - w[1]
            if rows and abs(m - rows[-1][0]) < 0.4 * h:
                rows[-1][1].append(w)
            else:
                rows.append([m, [w]])
        lines = []
        for _, ws in rows:
            ws.sort(key=lambda w: w[0])
            run = [ws[0]]
            for w in ws[1:] + [None]:
                if w is not None and w[0] - run[-1][2] <= run[-1][3] - run[-1][1]:
                    run.append(w)
                    continue
                box = [min(r[0] for r in run) * k, min(r[1] for r in run) * k,
                       max(r[2] for r in run) * k, max(r[3] for r in run) * k]
                lines.append({"text": " ".join(r[4] for r in run), "sure": True,
                              "box": [round(v) for v in box]})
                run = [w]
        out[n] = (lines, (pg["w"] * k, pg["h"] * k))
    return out


def begin(kind: str, found: tuple, lines: list[dict], height: float) -> dict:
    """A statement starting on this page: its form and unit from the page's top."""
    text = " ".join(ln["text"] for ln in lines if ln["box"][1] < 0.3 * height)
    form = FORM_RE.search(text) or FORM_SEEN.search(text)
    unit = UNIT_RE.search(text) or next((m for ln in lines if ln["box"][1] < 0.3 * height
                                         for m in [UNIT_ALONE.fullmatch(ln["text"].strip())] if m), None)
    return {"statement": kind, "form": form.group(0) if form else None,
            "form_2026": bool(re.search(NEW_FORM, found[1])) if kind == "B01" else None,
            "title": found[1], "unit": unit.group(1).strip() if unit else None,
            "pages": [], "columns": [], "rows": []}


def restarts(before: list[dict], rows: list[dict]) -> bool:
    """A page that holds the statement's codes again: half or more already seen.

    A continued page's codes are new ones; some filings print a statement twice (the
    quarter's and the year to date's, portrait and landscape), or reprint a page. Half,
    not any: a stray number read as a code (in an address, say) repeats on every page."""
    seen = {r["code"] for r in before if r["code"]}
    codes = [r["code"] for r in rows if r["code"]]
    return bool(codes) and sum(c in seen for c in codes) >= REPEAT * len(codes)


def untitled(lines: list[dict], height: float, before: str | None) -> str | None:
    """The statement an untitled page starts, by the form printed at its top and by its
    codes (the fullest column of them, MIN_CODES or more): B01 when nearly all are 100 or
    more; under 100, B02 or B03 by which of the codes only one of them prints it holds more.
    "" when the page is not before's statement but nothing says which it is, or the form and
    the codes disagree: left out, never guessed. None when nothing tells (the caller goes
    on as before: a page of few codes, or of mixed ones, or with no column of figures).

    Codes overrule a title missing or misread: the codes 01 to 70 of an income or cash flow
    statement cannot be a balance sheet's, nor its codes 100 to 440 theirs."""
    top = " ".join(ln["text"] for ln in lines if ln["box"][1] < TOP_SHARE * height)
    m = FORM_RE.search(top)
    form = f"B0{m.group(1)}" if m else None
    cols = columns_of(spaced_amounts(lines))
    if not cols:                                    # no figures: a contents page names forms too
        return None
    codes = [ln["text"].strip() for ln in code_column(lines, cols)]
    codes = [c for c in codes if CODE_RE.fullmatch(c)]
    by = None
    if len(codes) >= MIN_CODES:
        big = sum(int(c) >= 100 for c in codes)
        if big >= 0.8 * len(codes):
            by = "B01"
        elif big <= 0.2 * len(codes):
            b2, b3 = len(set(codes) & B02_ONLY), len(set(codes) & B03_ONLY)
            by = "B02" if b2 > b3 else "B03" if b3 > b2 else "" if before == "B01" else None
    if form and by:
        return form if form == by else ""
    if form and by == "":
        return form if form != "B01" else ""
    return form or by


def carries_on(cur: dict, page: int, lines: list[dict]) -> bool:
    """An untitled page of few codes that ends the statement before it: the page right
    after, its codes (top to bottom) rising and past the last code read before, but for one
    stray number at most (a note's, read as a code).

    A statement's last totals often spill onto a page of their own (B03's 50 to 70 over
    the signatures). Few codes alone are not enough: a notes page holds stray numbers too."""
    if not cur["pages"] or cur["pages"][-1] != page - 1:
        return False
    seen = [int(r["code"]) for r in cur["rows"] if r["code"] and r["code"].isdigit()]
    if not seen:
        return False
    nums = [int(ln["text"]) for ln in sorted(lines, key=lambda ln: ln["box"][1])
            if CODE_RE.fullmatch(ln["text"].strip())]
    past = [n for n in nums if n > max(seen)]
    return (len(past) >= 2 and len(nums) - len(past) <= 1      # one stray number at most (a note's)
            and all(a < b for a, b in zip(past, past[1:])))


def carried_columns(cols: list[float], before: list[float], statement: str, lines: list[dict],
                    last: int | None = None) -> list[float]:
    """A page that goes on with a statement keeps the columns of its page before, when it
    finds fewer of its own and each of those is one of them: a last-totals page has too few
    figures a column (MIN_COL) to place them by itself. More columns, or another place, are
    its own. Only when nearly all its codes are the statement's (B01's from 100, the others'
    below) and come after the last code it read (last): an annex after the statements (a
    list of branches, numbered) is let on by its count of codes too, and its own columns,
    too few, were all that kept its rows out. Below 100 the range alone tells nothing (an
    annex numbered 11, 12, ... holds B03's codes); a statement's codes only rise. Codes are
    the numbers left of those columns, as code_column takes them (a small figure, a share's
    earnings, is a number too), so one stray among two codes or more is let pass, as in
    carries_on (a note's, the page's number)."""
    nums = [int(ln["text"]) for ln in lines if CODE_RE.fullmatch(ln["text"].strip())
            and ln["box"][2] < min(before) - 150]
    mine = sum((n >= 100) == (statement == "B01") and (last is None or n > last) for n in nums)
    strays = len(nums) - mine
    if (len(cols) < len(before) and mine and strays <= max(0.2 * len(nums), 1 if mine >= 2 else 0)
            and all(any(abs(c - b) <= COL_GAP for b in before) for c in cols)):
        return before
    return cols


def statements_of(draft: dict, pdf: str | None = None, told: dict | None = None,
                  images: str | None = None) -> list[dict]:
    """The primary statements found in a draft, each with its pages, form, unit and rows.
    images: the draft's page images, whose sizes are the pages' (size_of).

    With the draft's PDF, text-layer pages are placed from it (layer_lines); without, they
    have no boxes and are passed over. told (page -> statement), for building truth only:
    a person said the statement is printed there, so those pages are taken as it whatever
    their title reads; the reading itself never passes it."""
    out, cur = [], None
    layer, notes = None, False
    for p in draft["pages"]:
        src = p["lines"]
        width, height = size_of(p, src, images)
        if p.get("kind") == "text layer" and pdf:
            layer = layer or layer_lines(pdf)
            src, size = layer.get(p["page"], ([], None))
            width, height = size or (width, height)
        lines = split_codes([ln for ln in src if ln.get("box")])
        found = title_of(lines, height, width)
        if told and p["page"] in told:
            found = (told[p["page"]], found[1] if found and found[0] == told[p["page"]] else "")
        kind = found and found[0]
        nums = [int(ln["text"]) for ln in lines if CODE_RE.fullmatch(ln["text"].strip())]
        codes = len(nums)
        prose = sum(len(ln["text"]) >= PROSE for ln in lines)
        if (not kind and not out and codes >= 8 and sum(n >= 100 for n in nums) >= 0.8 * codes
                and prose <= MAX_PROSE):
            kind, found = "B01", ("B01", "")        # an untitled page of balance-sheet codes
        notes = notes or any(ln["box"][1] < TOP_SHARE * height and re.search(NOTES, fold(ln["text"]))
                             for ln in lines)
        if not kind and not notes and not told and prose <= MAX_PROSE:
            by = untitled(lines, height, cur["statement"] if cur else None)
            if by == "":                            # not the statement before; which, unknown
                cur = None
                continue
            if by and (cur is None or by != cur["statement"]):
                kind, found = by, (by, "")
        go_on = bool(kind and cur and cur["statement"] == kind and cur["pages"]
                     and cur["pages"][-1] == p["page"] - 1)
        fresh = bool(kind and not go_on)
        if fresh:
            cur = begin(kind, found, lines, height)
            out.append(cur)
        elif cur is None or (codes < 5 and kind != cur["statement"]
                             and not (kind is None and carries_on(cur, p["page"], lines))):
            cur = None
            continue
        lines = spaced_amounts(lines)
        cols = columns_of(lines)
        known = cur.get("code_x") if cur and same_size(cur.get("size"), height) else None
        if not fresh and known and cur["columns"]:
            seen = [int(r["code"]) for r in cur["rows"] if r["code"] and r["code"].isdigit()
                    and (int(r["code"]) >= 100) == (cur["statement"] == "B01")]
            cols = carried_columns(cols, cur["columns"][-1]["x"], cur["statement"], lines,
                                   max(seen, default=None))
        lines = split_amounts(lines, cols)
        tilt = tilt_of(lines, cols)
        rows = rows_of(lines, cols, known, tilt)
        if not fresh and restarts(cur["rows"], rows):    # the same statement printed again
            cur = begin(cur["statement"], found or (cur["statement"], cur["title"]), lines, height)
            out.append(cur)
            rows = rows_of(lines, cols, tilt=tilt)
        if not rows:
            if not kind:
                cur = None
            continue
        xs = [r["code_box"] for r in rows if r["code"]]
        if xs:
            cur["code_x"], cur["size"] = (min(b[0] for b in xs), max(b[2] for b in xs)), height
        cur["pages"].append(p["page"])
        cur["columns"].append({"page": p["page"], "x": cols, "tilt": tilt,
                               "headings": headings_of(lines, cols, rows[0]["y"] if rows else None)})
        blank = p.get("blank_cells") or []          # read_pdf.blank_cells: cells looked at, no ink
        coarse = bool(p.get("coarse"))              # read_pdf.coarse_pages: left empty, not guessed
        for r in rows:
            r["page"] = p["page"]
            if coarse:
                r["values"] = [v and {**v, "value": None, "sure": False} for v in r["values"]]
                continue
            at = sorted(b["column"] for b in blank if b["code_box"] == r["code_box"])
            if at:
                r["blank"] = at                     # the values there stay None: blank is not read
        cur["rows"] += rows
    for s in out:
        if s["statement"] == "B02":
            older_numbering(s["rows"])
    return [s for s in out if s["rows"]]


OLDER_B02 = [   # (code as printed, the 2014 form's code, words its label holds, the code that
                #  must be absent): lines an income statement numbers as the form before 2014 did
    ("03", "02", "giam tru", "02"),                     # deductions from revenue
    ("24", "25", "chi phi ban", "26"),                  # selling expenses
    ("25", "26", "quan ly", "26"),                      # general and administrative expenses
]


def older_numbering(rows: list[dict]) -> None:
    """An income statement numbered as before the 2014 form, renumbered to it in place; the
    code as printed is kept as "printed".

    Before 2014, deductions from revenue were 03, selling expenses 24 and administrative
    expenses 25; the 2014 form has 02, 25 and 26 (its 24 is the share of associates' profit).
    A code is renumbered only when its label says which line it is and the 2014 code it
    would take is not printed; selling and administrative expenses go together or not at
    all, since moving one alone would put two lines on 25."""
    by = {r["code"]: r for r in rows if r.get("code")}
    def says(code, words):
        return code in by and words in fold(by[code].get("label") or "")
    moves = [(old, new) for old, new, words, absent in OLDER_B02
             if says(old, words) and absent not in by]
    if len({o for o, _ in moves} & {"24", "25"}) == 1:
        moves = [m for m in moves if m[0] not in ("24", "25")]
    for old, _ in moves:
        by[old]["printed"] = old
    for old, new in moves:
        by[old]["code"] = new


def main():
    with open(sys.argv[1], encoding="utf-8") as fh:
        d = json.load(fh)
    pdf = sys.argv[2] if len(sys.argv) > 2 else os.path.join("cache", "pdfs", d["file"])
    pdf = pdf if os.path.exists(pdf) else None
    found = statements_of(d, pdf, images=os.path.join(os.path.dirname(sys.argv[1]), "images"))
    for s in found:
        vals = [v for r in s["rows"] for v in r["values"] if v]
        print(s["statement"], s["form"], s["unit"], "pages", s["pages"], "rows", len(s["rows"]),
              "values", len(vals), "sure", sum(v["value"] is not None for v in vals),
              "columns", [len(c["x"]) for c in s["columns"]])
    from fs_contents import check                   # here: it uses this module
    got = check(d, found, pdf)
    if got is not None:
        print("contents page", got["page"], "; ".join(got["flags"]) or "agrees")


if __name__ == "__main__":
    main()
