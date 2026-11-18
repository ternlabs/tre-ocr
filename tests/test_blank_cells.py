"""read_pdf.no_ink and blank_cells: a statement cell left empty, looked at for ink from the row
above to the row below, on invented pages; and what fs_rows and fs_sums make of the mark.
Run: .venv/bin/python -I -m unittest discover tests"""
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from fs_rows import statements_of  # noqa: E402
from fs_sums import BLANK, sums_of  # noqa: E402
from read_pdf import blank_cells, no_ink  # noqa: E402

COLS = [600, 900]                       # the page's columns, by their right edges
TITLE = "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH"
BOX = [100, 400, 140, 430]              # the code box of the row left empty


def page(*bars, shade=0):
    g = np.full((400, 600), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in bars:
        g[y0:y1, x0:x1] = shade
    return g


class NoInk(unittest.TestCase):
    def test_white(self):
        self.assertTrue(no_ink(page(), 100, 100, 400, 200))

    def test_a_figure(self):
        self.assertFalse(no_ink(page((300, 140, 320, 170)), 100, 100, 400, 200))

    def test_faint_print_is_ink(self):
        self.assertFalse(no_ink(page((300, 140, 320, 170), shade=190), 100, 100, 400, 200))
        self.assertTrue(no_ink(page((300, 140, 320, 170), shade=210), 100, 100, 400, 200))

    def test_dust(self):
        self.assertTrue(no_ink(page((300, 140, 304, 144)), 100, 100, 400, 200))
        self.assertFalse(no_ink(page((300, 140, 305, 144)), 100, 100, 400, 200))
        self.assertFalse(no_ink(page((300, 140, 304, 145)), 100, 100, 400, 200))

    def test_ruled_lines_are_not_ink(self):
        across, down = (0, 150, 600, 153), (250, 0, 253, 400)
        self.assertTrue(no_ink(page(across, down), 100, 100, 400, 200))

    def test_a_short_bar_is_ink(self):
        """A dash, or a line under a figure, shorter than a ruled line."""
        self.assertFalse(no_ink(page((300, 150, 359, 153)), 100, 100, 400, 200))
        self.assertTrue(no_ink(page((300, 150, 360, 153)), 100, 100, 400, 200))

    def test_a_figure_resting_on_a_rule(self):
        self.assertFalse(no_ink(page((0, 170, 600, 173), (300, 140, 320, 170)), 100, 100, 400, 200))

    def test_a_stroke_as_tall_as_the_region_is_ink(self):
        """A digit 1 fills the height of a region no taller than its row; a ruled line goes past it."""
        self.assertFalse(no_ink(page((300, 100, 304, 130)), 100, 100, 400, 130))
        self.assertTrue(no_ink(page((300, 90, 304, 140)), 100, 100, 400, 130))

    def test_right_of_the_edge_a_thin_stroke_is_a_rule(self):
        """A rule that ends in the region, so not dark all the way down it."""
        stroke = (350, 100, 353, 160)
        self.assertFalse(no_ink(page(stroke), 100, 100, 400, 200))
        self.assertTrue(no_ink(page(stroke), 100, 100, 400, 200, (350, 30)))
        self.assertFalse(no_ink(page(stroke), 100, 100, 400, 200, (351, 30)))
        self.assertFalse(no_ink(page((350, 100, 354, 120)), 100, 100, 400, 200, (350, 30)))

    def test_right_of_the_edge_a_frame_as_tall_as_a_row(self):
        frame, foot = (350, 100, 361, 160), (350, 163, 361, 170)
        self.assertTrue(no_ink(page(frame, foot), 100, 100, 400, 200, (350, 60)))
        self.assertFalse(no_ink(page(frame, foot), 100, 100, 400, 200, (350, 61)))
        self.assertFalse(no_ink(page(frame, foot), 100, 100, 400, 200))
        self.assertFalse(no_ink(page(frame, (330, 120, 345, 150)), 100, 100, 400, 200, (350, 60)))

    def test_too_small_to_tell(self):
        self.assertIsNone(no_ink(page(), 100, 100, 400, 107))
        self.assertIsNone(no_ink(page(), 595, 100, 700, 200))      # the page ends at 600


def draft(tmp, *bars, shade=0, held=(), last=False, first=False):
    """A read income-statement page: codes 10 to 70 a hundred px apart, each with a figure in
    both columns but 40 (BOX), which holds one only in the columns of held. With last, the
    page ends at 40; with first, it begins there."""
    lines = [{"text": TITLE, "box": [300, 60, 1000, 100], "sure": True}]
    for y, code in ((100, "10"), (200, "20"), (300, "30"), (400, "40"), (500, "50"), (600, "60"), (700, "70")):
        if (last and code > "40") or (first and code < "40"):
            continue
        lines += [{"text": code, "box": [100, y, 140, y + 30], "sure": True},
                  {"text": "Doanh thu", "box": [160, y, 400, y + 30], "sure": True}]
        lines += [{"text": "1.234.567", "box": [x - 200, y, x, y + 30], "sure": True}
                  for i, x in enumerate(COLS) if code != "40" or i in held]
    a = np.full((1400, 1200), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in bars:
        a[y0:y1, x0:x1] = shade
    os.makedirs(os.path.join(tmp, "images"), exist_ok=True)
    Image.fromarray(a).convert("RGB").save(os.path.join(tmp, "images", "p1.png"))
    return [{"page": 1, "kind": "read", "key": "p1", "lines": lines}]


class BlankCells(unittest.TestCase):
    def look(self, *bars, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            pages = draft(tmp, *bars, **kw)
            n = blank_cells(pages, os.path.join(tmp, "images"))
        got = pages[0].get("blank_cells", [])
        self.assertEqual(n, len(got))
        self.assertTrue(all(b["code_box"] == BOX for b in got), got)
        return [b["column"] for b in got], pages

    def test_an_empty_row(self):
        got, pages = self.look()
        self.assertEqual(got, [0, 1])
        rows = statements_of({"pages": pages})[0]["rows"]
        self.assertEqual([r.get("blank") for r in rows], [None, None, None, [0, 1], None, None, None])
        self.assertEqual(rows[3]["values"], [None, None])           # blank is not a value

    def test_a_figure_on_the_code_line(self):
        self.assertEqual(self.look((800, 405, 820, 425))[0], [0])
        self.assertEqual(self.look((500, 405, 520, 425))[0], [1])

    def test_a_figure_on_a_line_above_or_below_the_code(self):
        """A wrapped label: the figure on its first line, or its last."""
        self.assertEqual(self.look((800, 340, 820, 360))[0], [0])
        self.assertEqual(self.look((800, 465, 820, 485))[0], [0])

    def test_the_rows_beside_are_not_this_cell(self):
        self.assertEqual(self.look((800, 305, 820, 325), (800, 505, 820, 525))[0], [0, 1])

    def test_ink_left_of_the_cell(self):
        """The cell takes CELL[0] of the gap to the column before: the rest is the other column's."""
        self.assertEqual(self.look((640, 405, 655, 425))[0], [0, 1])
        self.assertEqual(self.look((650, 405, 665, 425))[0], [0])

    def test_a_ruled_grid(self):
        grid = [(50, y, 1000, y + 3) for y in (392, 436)] + [(x, 50, x + 3, 700) for x in (610, 910)]
        self.assertEqual(self.look(*grid)[0], [0, 1])

    def test_a_held_cell_is_not_listed(self):
        self.assertEqual(self.look(held=(0,))[0], [1])

    def test_the_last_row_reaches_a_code_height_down(self):
        self.assertEqual(self.look((800, 440, 820, 458), last=True)[0], [0])
        self.assertEqual(self.look((800, 462, 820, 480), last=True)[0], [0, 1])

    def test_the_first_row_reaches_a_code_height_up(self):
        self.assertEqual(self.look((800, 375, 820, 393), first=True)[0], [0])
        self.assertEqual(self.look((800, 348, 820, 366), first=True)[0], [0, 1])

    def test_a_table_that_ends_at_the_last_row(self):
        """Its rules end with it, a code's height above the foot of the cell looked at."""
        rules = [(x, 50, x + 3, 436) for x in (610, 910)] + [(50, 436, 1000, 440)]
        self.assertEqual(self.look(*rules, last=True)[0], [0, 1])

    def test_a_thin_stroke_is_a_rule_only_right_of_the_column(self):
        self.assertEqual(self.look((900, 405, 903, 428))[0], [0])
        self.assertEqual(self.look((904, 405, 907, 428))[0], [0, 1])

    def test_looked_at_afresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            pages = draft(tmp, held=(0, 1))
            pages[0]["blank_cells"] = [{"code_box": BOX, "column": 0}]
            self.assertEqual(blank_cells(pages, os.path.join(tmp, "images")), 0)
        self.assertNotIn("blank_cells", pages[0])


def b01(**kw):
    v = {"220": 500, "221": 500, "222": 800, "223": 300, "224": 0, "225": 0, "226": 0, "227": 0}
    v.update({k[1:]: x for k, x in kw.items()})
    return v


class Sums(unittest.TestCase):
    def of(self, **kw):
        return {r[0]: r for r in sums_of("2026", "B01", b01(**kw))}

    def test_a_blank_total_over_blank_lines_holds_counted_apart(self):
        got = self.of(_224=BLANK, _225=BLANK, _226=BLANK)
        self.assertEqual(got["224"][2:4], ("holds", ("blank cells",)))
        self.assertEqual(got["220"][2:4], ("holds", ("blank cells",)))

    def test_a_blank_total_over_figures_is_unknown_never_a_failure(self):
        self.assertEqual(self.of(_224=BLANK, _225=70, _226=0)["224"][2], "unknown")

    def test_a_blank_line_that_does_not_make_the_sum_is_unknown(self):
        self.assertEqual(self.of(_224=70, _225=BLANK, _226=0)["224"][2], "unknown")

    def test_blank_and_unread_lines_are_named_apart(self):
        got = self.of(_225=BLANK, _226=None)
        self.assertEqual(got["224"][2:4], ("holds", ("blanks", "blank cells")))
        self.assertEqual(self.of(_226=None)["224"][3], ("blanks",))

    def test_an_unread_total_stays_unknown(self):
        self.assertEqual(self.of(_224=None, _225=BLANK, _226=BLANK)["224"][2], "unknown")

    def test_a_blank_heading_with_no_lines(self):
        got = {r[0]: r for r in sums_of("2026", "B01", {"150": BLANK})}
        self.assertEqual(got["150"][2:4], ("holds", ("none printed", "blank cells")))


if __name__ == "__main__":
    unittest.main()
