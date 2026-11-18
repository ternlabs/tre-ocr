"""read_pdf.sum_look, open_cells and figure_of: a statement cell one of the form's sums leaves
open is read once more, and the reading kept only when it is the figure the sum leaves it;
on invented pages, with readers that say what the test tells them to.
Run: .venv/bin/python -I -m unittest discover tests"""
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from fs_rows import statements_of  # noqa: E402
from read_pdf import Readers, figure_of, open_cells, sum_look  # noqa: E402

COLS = [600, 900]                       # the page's columns, by their right edges
TITLE = "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH"
ROWS = (("01", "1.000.000"), ("02", "200.000"), ("10", "800.000"), ("11", "300.000"), ("20", "500.000"))
Y = {code: 100 * (k + 1) for k, (code, _) in enumerate(ROWS)}      # 10 = 01 - 02, 20 = 10 - 11


def box(code, column=1):
    return [COLS[column] - 200, Y[code], COLS[column], Y[code] + 30]


def draft(tmp, **cells):
    """A read income-statement page whose sums hold in both columns. cells: code -> what the
    page's line in the second column says instead (None: no line there)."""
    lines = [{"text": TITLE, "box": [300, 40, 1000, 80], "sure": True}]
    for code, amount in ROWS:
        y = Y[code]
        lines += [{"text": code, "box": [100, y, 140, y + 30], "sure": True},
                  {"text": "Doanh thu", "box": [160, y, 380, y + 30], "sure": True},
                  {"text": amount, "box": box(code, 0), "sure": True}]
        said = cells.get("_" + code, amount)
        if said is not None:
            lines.append({"text": said, "box": box(code), "sure": "[?]" not in said})
    os.makedirs(os.path.join(tmp, "images"), exist_ok=True)
    Image.fromarray(np.full((1400, 1200, 3), 255, dtype=np.uint8)).save(os.path.join(tmp, "images", "p1.png"))
    return [{"page": 1, "kind": "read", "key": "p1", "lines": lines}]


def readers(reads, asked=None):
    """Readers whose two readings of the line in a box are reads[code]."""
    r = Readers.__new__(Readers)
    by_box = {tuple(box(code)): v for code, v in reads.items()}

    def figure(png, b):
        if asked is not None:
            asked.append(b)
        return by_box.get(tuple(b), ("", ""))
    r.figure = figure
    return r


class FigureOf(unittest.TestCase):
    def test_an_amount(self):
        self.assertEqual(figure_of("1.234.567"), (1234567, "1.234.567"))
        self.assertEqual(figure_of(" (1.234.567) "), (-1234567, "(1.234.567)"))
        self.assertEqual(figure_of("-1.234"), (-1234, "(1.234)"))
        self.assertEqual(figure_of("999"), (999, "999"))

    def test_separators_set_right_in_the_readers_own_kind(self):
        self.assertEqual(figure_of("1.234.5.67"), (1234567, "1.234.567"))
        self.assertEqual(figure_of("12,345 678"), (12345678, "12,345,678"))
        self.assertEqual(figure_of("(1234567)"), (-1234567, "(1.234.567)"))

    def test_not_an_amount(self):
        for t in ("", "-", "0", "(0)", "1.234x", "(1.234.567", "1.234.567)", "Y 1.234", "[?]"):
            self.assertIsNone(figure_of(t), t)


class OpenCells(unittest.TestCase):
    def of(self, **cells):
        with tempfile.TemporaryDirectory() as tmp:
            s = statements_of({"pages": draft(tmp, **cells)})[0]
        return [(s["rows"][k]["code"], i, want) for k, i, want in open_cells(s, None)]

    def test_all_read(self):
        self.assertEqual(self.of(), [])

    def test_a_cell_unsure_or_with_no_line(self):
        self.assertEqual(self.of(_10="[?]"), [("10", 1, {800000})])
        self.assertEqual(self.of(_10=None), [("10", 1, {800000})])

    def test_a_deduction_either_way(self):
        self.assertEqual(self.of(_11="[?]"), [("11", 1, {300000, -300000})])

    def test_two_open_in_one_sum(self):
        self.assertEqual(self.of(_01="[?]", _02="[?]"), [])

    def test_a_code_on_two_rows_is_not_asked_about_nor_reckoned_with(self):
        with tempfile.TemporaryDirectory() as tmp:
            pages = draft(tmp, _10="[?]")
            pages[0]["lines"] += [{"text": "11", "box": [100, 600, 140, 630], "sure": True},
                                  {"text": "300.000", "box": [400, 600, 600, 630], "sure": True},
                                  {"text": "300.000", "box": [700, 600, 900, 630], "sure": True}]
            s = statements_of({"pages": pages})[0]
        self.assertEqual([r["code"] for r in s["rows"]].count("11"), 2)
        self.assertEqual([(s["rows"][k]["code"], i, w) for k, i, w in open_cells(s, None)],
                         [("10", 1, {800000})])                 # by 10 = 01 - 02 alone

    def test_an_open_code_on_two_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            pages = draft(tmp, _10="[?]")
            pages[0]["lines"] += [{"text": "10", "box": [100, 600, 140, 630], "sure": True},
                                  {"text": "800.000", "box": [400, 600, 600, 630], "sure": True},
                                  {"text": "[?]", "box": [700, 600, 900, 630], "sure": False}]
            s = statements_of({"pages": pages})[0]
        self.assertEqual([r["code"] for r in s["rows"]].count("10"), 2)
        self.assertEqual(open_cells(s, None), [])

    def test_a_cell_seen_blank_counts_as_zero(self):
        """11 with no ink: 20 = 10 - 11 leaves 20 what 10 holds."""
        with tempfile.TemporaryDirectory() as tmp:
            pages = draft(tmp, _11=None, _20="[?]")
            self.assertEqual(open_cells(statements_of({"pages": pages})[0], None), [])
            pages[0]["blank_cells"] = [{"code_box": [100, Y["11"], 140, Y["11"] + 30], "column": 1}]
            s = statements_of({"pages": pages})[0]
        self.assertEqual([(s["rows"][k]["code"], i, w) for k, i, w in open_cells(s, None)],
                         [("20", 1, {800000})])


class SumLook(unittest.TestCase):
    def look(self, reads, **cells):
        with tempfile.TemporaryDirectory() as tmp:
            pages = draft(tmp, **cells)
            asked = []
            n = sum_look(pages, os.path.join(tmp, "images"), readers(reads, asked))
            rows = {r["code"]: r["values"][1] for r in statements_of({"pages": pages})[0]["rows"]}
        return n, pages[0]["lines"], rows, asked

    def line(self, lines, code):
        return next(ln for ln in lines if ln["box"] == box(code))

    def test_a_reading_that_is_the_figure(self):
        n, lines, rows, _ = self.look({"10": ("800.000", "8OO.000")}, _10="[?]")
        self.assertEqual(n, 1)
        self.assertEqual(self.line(lines, "10"), {"text": "800.000", "box": box("10"), "sure": True, "look": "sum"})
        self.assertEqual((rows["10"]["value"], rows["10"]["look"]), (800000, "sum"))

    def test_either_reader(self):
        self.assertEqual(self.look({"10": ("B00.000", "800.000")}, _10="[?]")[0], 1)

    def test_another_figure_is_left(self):
        n, lines, rows, _ = self.look({"10": ("300.000", "860.000")}, _10="[?]")
        self.assertEqual(n, 0)
        self.assertEqual(self.line(lines, "10"), {"text": "[?]", "box": box("10"), "sure": False})
        self.assertIsNone(rows["10"]["value"])

    def test_the_sign_is_read_too(self):
        self.assertEqual(self.look({"10": ("(800.000)", "(800.000)")}, _10="[?]")[0], 0)
        self.assertEqual(self.look({"11": ("(300.000)", "")}, _11="[?]")[0], 1)     # a deduction: either way

    def test_a_sure_line_that_is_no_amount(self):
        """A stray separator: the row holds nothing there, and the line is in its cell."""
        n, lines, _, _ = self.look({"10": ("80.0.000", "80.0.000")}, _10="80.0.000")
        self.assertEqual((n, self.line(lines, "10")["text"]), (1, "800.000"))

    def test_nothing_asked_where_all_is_read_or_no_line_is_there(self):
        self.assertEqual(self.look({})[3], [])
        self.assertEqual(self.look({}, _10=None)[3], [])

    def test_two_lines_in_the_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            pages = draft(tmp, _10="[?]")
            pages[0]["lines"].append({"text": "[?]", "box": [650, Y["10"] + 2, 880, Y["10"] + 28], "sure": False})
            asked = []
            sum_look(pages, os.path.join(tmp, "images"), readers({"10": ("800.000",) * 2}, asked))
        self.assertEqual(asked, [])

    def test_a_cell_kept_opens_the_next(self):
        """02 and 10 unread: 20 = 10 - 11 leaves 10, and with it 10 = 01 - 02 leaves 02."""
        reads = {"10": ("800.000", "800.000"), "02": ("200.000", "200.000")}
        n, lines, rows, asked = self.look(reads, _10="[?]", _02="[?]")
        self.assertEqual(n, 2)
        self.assertEqual(asked, [box("10"), box("02")])
        self.assertEqual((rows["02"]["value"], rows["10"]["value"]), (200000, 800000))


if __name__ == "__main__":
    unittest.main()
