"""read_pdf.one_dash and cell_parts: a dash printed for zero, told by its ink, on drawn images.
Run: .venv/bin/python -I -m unittest discover tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from read_pdf import a_dash, cell_parts, column_shifts, one_dash  # noqa: E402


def page(*bars):
    """A white page with black rectangles (x0, y0, x1, y1)."""
    g = np.full((200, 700), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in bars:
        g[y0:y1, x0:x1] = 0
    return g


BOX = [100, 90, 125, 112]                 # a narrow box, as the detector draws round a dash


class OneDash(unittest.TestCase):
    def test_a_bar(self):
        self.assertTrue(one_dash(page((105, 99, 120, 103)), BOX))

    def test_a_short_bar(self):
        self.assertTrue(one_dash(page((108, 99, 116, 103)), BOX))

    def test_a_stroke_upright_is_a_one(self):
        self.assertFalse(one_dash(page((110, 92, 114, 110)), BOX))

    def test_two_bars_are_not_one(self):
        self.assertFalse(one_dash(page((105, 96, 120, 99), (105, 104, 120, 107)), BOX))

    def test_a_dot_is_not_a_dash(self):
        self.assertFalse(one_dash(page((110, 100, 114, 104)), BOX))

    def test_a_bar_beside_a_digit(self):
        self.assertFalse(one_dash(page((105, 99, 112, 103), (116, 92, 119, 110)), BOX))

    def test_a_wide_box_is_an_amount(self):
        self.assertFalse(one_dash(page((105, 99, 120, 103)), [100, 90, 300, 112]))

    def test_an_empty_box(self):
        self.assertFalse(one_dash(page(), BOX))

    def test_a_box_off_the_page(self):
        self.assertFalse(one_dash(page(), [720, 90, 740, 112]))


    def test_a_speck_is_not_a_dash(self):
        self.assertFalse(one_dash(page((105, 100, 115, 101)), BOX))


XS = [300, 600]                             # two columns' right edges
ROWS = [{"code": "10", "page": 1, "code_box": [20, 40, 50, 60], "values": [{"value": 5}, {"value": 5}]},
        {"code": "11", "page": 1, "code_box": [20, 100, 50, 120], "values": [None, None]}]


def cell(g, i=0, rows=ROWS):
    return a_dash(cell_parts(g, rows, 1, XS, i))


class Cells(unittest.TestCase):
    def test_a_dash_in_the_cell(self):
        self.assertIsNotNone(cell(page((280, 108, 292, 112))))

    def test_a_dash_by_a_ruled_line(self):
        self.assertIsNotNone(cell(page((280, 108, 292, 112), (305, 0, 308, 200))))

    def test_under_a_long_rule(self):
        self.assertIsNotNone(cell(page((280, 108, 292, 112), (100, 98, 330, 100))))

    # rules all four inside the cell (x 60-330, y 64-124 reaching up): they meet at its corners
    RULES = ((0, 70, 700, 72), (0, 118, 700, 120), (70, 0, 73, 200), (310, 0, 313, 200))

    def test_a_dash_in_a_cell_ruled_on_every_side(self):
        self.assertIsNotNone(cell(page((280, 108, 292, 112), *self.RULES)))
        self.assertIsNotNone(a_dash(cell_parts(page((280, 108, 292, 112), *self.RULES), ROWS, 1, XS, 0,
                                               reach=False)))

    def test_a_blank_ruled_cell_stays_empty(self):
        self.assertIsNone(cell(page(*self.RULES)))

    def test_a_figure_in_a_ruled_cell_is_not_a_dash(self):
        self.assertIsNone(cell(page((280, 108, 292, 112), (260, 103, 263, 116), *self.RULES)))

    def test_a_narrow_ruled_column(self):
        g = page((280, 108, 292, 112), (0, 117, 700, 120), (0, 61, 700, 64), (315, 0, 318, 200),
                 (215, 0, 218, 200))
        self.assertIsNotNone(a_dash(cell_parts(g, ROWS, 1, [300, 400], 0)))

    def test_a_rule_scanned_askew_in_a_blank_cell(self):
        for tilt in (0.0175, 0.02, 0.025):
            for t in (2, 3, 4):
                g = page((310, 0, 313, 200))
                for x in range(700):
                    y = round(78 + tilt * x)
                    g[y:y + t, x] = 0
                self.assertIsNone(cell(g), (tilt, t))

    def test_ink_joined_to_a_rule_is_not_a_dash(self):
        self.assertIsNone(cell(page((310, 0, 313, 200), (313, 108, 328, 110))))      # a tick past it
        self.assertIsNone(cell(page((310, 0, 313, 200), (262, 80, 310, 82))))       # a short underline
        self.assertIsNone(cell(page((286, 96, 290, 122), (270, 110, 286, 114))))    # a digit's tall stem

    @staticmethod
    def faint_rule(*bars):
        """A rule printed faint (gray 170) and broken every 40 px, one piece of it dark."""
        g = page(*bars)
        for x in range(0, 700, 40):
            g[116:118, x:x + 36] = np.minimum(g[116:118, x:x + 36], 170)
        g[116:118, 244:262] = 40
        return g

    def test_a_dark_piece_of_a_faint_rule_is_not_a_dash(self):
        self.assertIsNone(cell(self.faint_rule()))

    def test_a_dash_above_a_faint_rule(self):
        self.assertIsNotNone(cell(self.faint_rule((280, 106, 292, 110))))

    def test_a_figure_is_not_a_dash(self):
        self.assertIsNone(cell(page((280, 108, 292, 112), (260, 103, 263, 118))))

    def test_nothing_in_the_cell(self):
        self.assertIsNone(cell(page()))

    def test_the_other_column(self):
        self.assertIsNone(cell(page((280, 108, 292, 112)), i=1))

    def test_a_wrapped_label_dash_above_the_code(self):
        # the row before holds its figure, so the cell reaches up to it
        self.assertIsNotNone(cell(page((280, 78, 292, 82))))

    def test_not_above_when_the_row_before_is_empty(self):
        rows = [dict(ROWS[0], values=[None, None]), ROWS[1]]
        self.assertIsNone(cell(page((280, 78, 292, 82)), rows=rows))

    def test_the_figure_above_held_by_its_row(self):
        # the cell reaches up over the row before's figure, which that row holds
        rows = [dict(ROWS[0], values=[{"value": 5, "box": [200, 62, 300, 90]}, None]), ROWS[1]]
        self.assertIsNotNone(cell(page((280, 108, 292, 112), (220, 66, 223, 86), (260, 66, 263, 86)),
                                  rows=rows))

    def test_a_figure_above_held_by_no_row(self):
        self.assertIsNone(cell(page((280, 108, 292, 112), (220, 66, 223, 86), (260, 66, 263, 86))))

    def test_the_code_line_alone(self):
        # what dash_cells looks at first: the code's own line, not the line above
        g = page((280, 108, 292, 112), (220, 66, 223, 86), (260, 66, 263, 86))
        self.assertIsNotNone(a_dash(cell_parts(g, ROWS, 1, XS, 0, reach=False)))

    def test_a_broken_ruled_line(self):
        # an upright rule in pieces, each cut by the cell's top or bottom edge
        self.assertIsNotNone(cell(page((280, 108, 292, 112), (310, 50, 312, 80), (310, 84, 312, 130))))

    def test_dust_beside_the_dash(self):
        self.assertIsNotNone(cell(page((280, 108, 292, 112), (240, 100, 243, 103))))

    def test_a_digit_cut_by_the_edge_is_not_dust(self):
        # a stroke wider than a rule, cut by the bottom edge: it may be this cell's
        self.assertIsNone(cell(page((280, 108, 292, 112), (250, 115, 256, 140))))

    def test_a_stroke_right_of_the_column_is_a_rule(self):
        # a rule-thin stroke past the column's right edge, short: a piece of a ruled line
        self.assertIsNotNone(cell(page((280, 108, 292, 112), (306, 104, 308, 116))))

    def test_a_sliver_cut_where_the_crop_ends(self):
        # the code's box ends between pixels; the foot of the line below, cut where ink() cut it
        rows = [ROWS[0], dict(ROWS[1], code_box=[20, 100, 50, 120.6])]
        g = page((280, 108, 292, 112), (270, 122, 284, 130))
        self.assertIsNotNone(a_dash(cell_parts(g, rows, 1, XS, 0, reach=False)))


class Shifted(unittest.TestCase):
    """A page scanned askew: the code's line lies lower at the column."""
    G = page((280, 134, 292, 138))                  # a dash 26 px below the code's line

    def test_followed_by_the_shift(self):
        self.assertIsNotNone(a_dash(cell_parts(self.G, ROWS, 1, XS, 0, reach=False, shift=26)))

    def test_not_on_the_code_line(self):
        self.assertIsNone(a_dash(cell_parts(self.G, ROWS, 1, XS, 0, reach=False)))

    def test_within_the_margin_the_code_line(self):
        # a shift no more than the cell's margin is not followed: the dash just below stays out
        g = page((280, 124, 292, 127))
        self.assertIsNone(a_dash(cell_parts(g, ROWS, 1, XS, 0, reach=False, shift=4)))


def held(n, off, y0=0, w=200, col=0):
    """n rows, codes 20 px tall and 100 apart, each holding a figure off px below its code."""
    rows = []
    for j in range(n):
        y = y0 + 100 * j
        v = [None, None]
        v[col] = {"value": 5, "box": [XS[col] - w, y + off, XS[col], y + 20 + off]}
        rows.append({"code": str(10 + j), "page": 1, "code_box": [20, y, 50, y + 20], "values": v})
    return rows


class ColumnShifts(unittest.TestCase):
    def test_measured_on_the_figures(self):
        self.assertEqual(column_shifts(held(5, 3), 1, XS, 0.0)[0], 3)

    def test_a_wrapped_label_line_left_out(self):
        rows = held(5, 3) + held(6, 40, y0=1000)
        self.assertEqual(column_shifts(rows, 1, XS, 0.0)[0], 3)

    def test_codes_off_the_figures_lines(self):
        # every figure lies far below its code: the codes stand at the top of tall cells
        self.assertEqual(column_shifts(held(5, 30), 1, XS, 0.0)[0], 30)

    def test_too_few_the_tilt(self):
        # the column's middle is 300 - 0.8 * 300 / 2 = 180 across, the codes' 35
        self.assertAlmostEqual(column_shifts(held(4, 3), 1, XS, 0.01)[0], 1.45)

    def test_most_near_but_too_few_the_tilt(self):
        rows = held(4, 3) + held(2, 40, y0=1000)
        self.assertAlmostEqual(column_shifts(rows, 1, XS, 0.01)[0], 1.45)

    def test_dashes_and_narrow_figures_left_out(self):
        self.assertAlmostEqual(column_shifts(held(5, 3, w=40), 1, XS, 0.01)[0], 1.45)

    def test_each_column_its_own(self):
        sh = column_shifts(held(5, 3) + held(5, 7, y0=1000, col=1), 1, XS, 0.0)
        self.assertEqual(sh, [3, 7])


if __name__ == "__main__":
    unittest.main()
