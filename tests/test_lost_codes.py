"""read_pdf.lost_codes and code_look: a row code printed, but with no line found on it, found by
the figure line beside it and read from its ink, on invented pages.
Run: .venv/bin/python -I -m unittest discover tests"""
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from fs_rows import statements_of  # noqa: E402
from read_pdf import Readers, code_look, lost_codes  # noqa: E402

COLS = [600, 900]                       # the page's columns, by their right edges
CODE = [(105, 118), (124, 137)]         # x spans of a two-digit code's ink


def row(y, code="10", page=1):
    """A coded row whose code's box starts at y, holding a figure in each column."""
    return {"code": code, "code_box": [100, y, 140, y + 30], "page": page,
            "values": [{"text": "1.000", "value": 1000, "sure": True, "box": [x - 200, y, x, y + 30]}
                       for x in COLS]}


def statement(*rows):
    return {"rows": list(rows), "columns": [{"page": 1, "x": COLS}], "pages": [1]}


S = statement(row(100, "10"), row(200, "20"), row(400, "40"))


def figure(y, x1=900, text="1.234.567"):
    return {"text": text, "box": [x1 - 200, y, x1, y + 30], "sure": True}


def gray(*bars):
    """A white page with black rectangles (x0, y0, x1, y1)."""
    g = np.full((800, 1200), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in bars:
        g[y0:y1, x0:x1] = 0
    return g


def code_ink(y):
    return [(x0, y + 2, x1, y + 28) for x0, x1 in CODE]


class Lost(unittest.TestCase):
    def test_a_code_with_no_line(self):
        out = lost_codes(S, 1, [figure(300)], gray(*code_ink(300)))
        self.assertEqual(out, [(2, [105, 302, 137, 328])])

    def test_beside_a_label_starting_right_of_the_column(self):
        lines = [figure(300), {"text": "9. Lợi nhuận", "box": [145, 298, 500, 332], "sure": True}]
        self.assertEqual(lost_codes(S, 1, lines, gray(*code_ink(300))), [(2, [105, 302, 137, 328])])

    def test_a_line_over_the_code_column_left(self):
        """A line over the column's left half holds the code (put_code's), even with ink left of it."""
        lines = [figure(300), {"text": "[?] Lợi nhuận", "box": [119, 298, 500, 332], "sure": False}]
        g = gray((84, 302, 93, 328), (96, 302, 105, 328), (108, 302, 117, 328))
        self.assertEqual(lost_codes(S, 1, lines, g), [])
        lines[1]["box"][0] = 121
        self.assertEqual(lost_codes(S, 1, lines, g), [(2, [84, 302, 117, 328])])

    def test_ink_of_a_label_starting_in_the_column_not_a_code(self):
        lines = [figure(300), {"text": "Lợi nhuận", "box": [125, 298, 500, 332], "sure": True}]
        g = gray((127, 302, 138, 328), (140, 302, 150, 328))
        self.assertEqual(lost_codes(S, 1, lines, g), [])

    def test_a_figure_a_row_holds(self):
        s = statement(row(100, "10"), row(200, "20"), row(400, "40"))
        s["rows"][1]["values"][1]["box"] = [700, 300, 900, 330]     # a wrapped label's figure
        self.assertEqual(lost_codes(s, 1, [figure(300)], gray(*code_ink(300))), [])

    def test_a_figure_beside_a_rows_code(self):
        self.assertEqual(lost_codes(S, 1, [figure(205)], gray(*code_ink(205))), [])

    def test_one_part_and_a_ruled_line(self):
        g = gray(code_ink(300)[0], (98, 250, 101, 380))
        self.assertEqual(lost_codes(S, 1, [figure(300)], g), [])

    def test_parts_not_of_digit_size(self):
        self.assertEqual(lost_codes(S, 1, [figure(300)], gray((105, 310, 118, 318), (124, 310, 137, 318))), [])
        self.assertEqual(lost_codes(S, 1, [figure(300)], gray((105, 302, 140, 328), (144, 302, 180, 328))), [])

    def test_parts_too_tall(self):
        self.assertEqual(lost_codes(S, 1, [figure(300)], gray((105, 293, 118, 338), (124, 293, 137, 338))), [])

    def test_a_part_cut_by_the_band(self):
        """The foot of the line above reaches into the band: not a digit."""
        g = gray(code_ink(300)[0], (124, 285, 137, 312))
        self.assertEqual(lost_codes(S, 1, [figure(300)], g), [])

    def test_no_ink(self):
        self.assertEqual(lost_codes(S, 1, [figure(300)], gray()), [])

    def test_not_a_figure(self):
        self.assertEqual(lost_codes(S, 1, [figure(300, text="Lợi nhuận")], gray(*code_ink(300))), [])

    def test_not_at_a_column(self):
        self.assertEqual(lost_codes(S, 1, [figure(300, x1=1100)], gray(*code_ink(300))), [])

    def test_two_figures_one_code(self):
        out = lost_codes(S, 1, [figure(300, 600), figure(302)], gray(*code_ink(300)))
        self.assertEqual(out, [(2, [105, 302, 137, 328])])

    def test_too_few_codes_on_the_page(self):
        s = statement(row(100, "10"), row(400, "40"))
        self.assertEqual(lost_codes(s, 1, [figure(300)], gray(*code_ink(300))), [])

    def test_no_columns_on_the_page(self):
        s = {**S, "columns": [{"page": 2, "x": COLS}]}
        self.assertEqual(lost_codes(s, 1, [figure(300)], gray(*code_ink(300))), [])

    def test_below_the_last_row(self):
        self.assertEqual(lost_codes(S, 1, [figure(500)], gray(*code_ink(500))), [(3, [105, 502, 137, 528])])

    def test_below_the_last_row_with_a_page_after(self):
        s = statement(row(100, "10"), row(200, "20"), row(400, "40"), row(100, "50", page=2))
        self.assertEqual(lost_codes(s, 1, [figure(500)], gray(*code_ink(500)))[0][0], 3)

    def test_above_the_first_row(self):
        s = statement(row(300, "10"), row(400, "20"), row(500, "40"))
        self.assertEqual(lost_codes(s, 1, [figure(150)], gray(*code_ink(150)))[0][0], 0)


TITLE = "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH"


def draft_page(tmp, lost_y=None):
    """A read income-statement page: codes 10, 20, 40, 50 with their labels and two figures,
    and, at lost_y, a figure line whose code's ink is on the page but in no line."""
    lines = [{"text": TITLE, "box": [300, 60, 1000, 100], "sure": True}]
    a = np.full((1400, 1200, 3), 255, dtype=np.uint8)
    for y, code in ((200, "10"), (300, "20"), (500, "40"), (600, "50")):
        lines += [{"text": code, "box": [100, y, 140, y + 30], "sure": True},
                  {"text": "Doanh thu", "box": [160, y, 400, y + 30], "sure": True}]
        lines += [figure(y, x) for x in COLS]
    if lost_y:
        lines += [{"text": "Lợi nhuận", "box": [160, lost_y, 400, lost_y + 30], "sure": True}]
        lines += [figure(lost_y, x) for x in COLS]
        for x0, y0, x1, y1 in code_ink(lost_y):
            a[y0:y1, x0:x1] = 0
    os.makedirs(os.path.join(tmp, "images"), exist_ok=True)
    Image.fromarray(a).save(os.path.join(tmp, "images", "p1.png"))
    return [{"page": 1, "kind": "read", "key": "p1", "lines": lines}]


def readers(read):
    r = Readers.__new__(Readers)
    r.code = lambda png, box: read
    return r


class CodeLook(unittest.TestCase):
    def look(self, read, lost_y=400):
        with tempfile.TemporaryDirectory() as tmp:
            pages = draft_page(tmp, lost_y)
            found = code_look(pages, os.path.join(tmp, "images"), readers(read))
        return found, pages

    def test_the_page_has_its_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = statements_of({"pages": draft_page(tmp)})
        self.assertEqual([r["code"] for r in s[0]["rows"]], ["10", "20", "40", "50"])

    def test_a_lost_code_read_in_its_place(self):
        found, pages = self.look(("30", "30"))
        self.assertEqual(found, 1)
        self.assertIn({"text": "30", "box": [105, 402, 137, 428], "sure": True, "look": "lost code"},
                      pages[0]["lines"])
        s = statements_of({"pages": pages})
        self.assertEqual([r["code"] for r in s[0]["rows"]], ["10", "20", "30", "40", "50"])

    def test_a_read_out_of_order_left(self):
        found, pages = self.look(("45", "45"))
        self.assertEqual(found, 0)
        self.assertFalse(any(ln.get("look") == "lost code" for ln in pages[0]["lines"]))

    def test_a_code_already_read_left(self):
        self.assertEqual(self.look(("20", "20"))[0], 0)

    def test_readers_that_differ_left(self):
        self.assertEqual(self.look(("30", "38"))[0], 0)


if __name__ == "__main__":
    unittest.main()
