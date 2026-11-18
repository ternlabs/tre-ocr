"""A page scanned askew, and a continuation page's codes, on invented page lines.
Run: .venv/bin/python -I -m unittest discover tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from fs_rows import code_column, owners, tilt_of  # noqa: E402

H = 50                       # a line's height at 300 dpi
COLS = [1700.0, 2100.0]      # the value columns' right edges; their middles are 400 apart


def ln(text, x0, x1, y, h=H):
    return {"text": text, "box": [x0, y - h // 2, x1, y + h // 2], "sure": True}


def pair(y, drop):
    """One printed line's two figures, the second drop px lower."""
    return [ln("1.234.567", 1450, 1700, y), ln("7.654.321", 1850, 2100, y + drop)]


class Tilt(unittest.TestCase):
    def test_measured_on_pairs(self):
        lines = [f for j in range(5) for f in pair(500 + 100 * j, 8)]
        self.assertEqual(tilt_of(lines, COLS), 0.02)

    def test_too_few_pairs(self):
        lines = [f for j in range(4) for f in pair(500 + 100 * j, 8)]
        self.assertEqual(tilt_of(lines, COLS), 0.0)

    def test_dashes_left_out(self):
        # a dash beside each figure, at its own height: not a pair
        lines = [f for j in range(5) for f in pair(500 + 100 * j, 8)]
        lines += [ln("-", 2030, 2100, 500 + 100 * j, h=6) for j in range(5)]
        self.assertEqual(tilt_of(lines, COLS), 0.02)

    def test_one_column(self):
        self.assertEqual(tilt_of(pair(500, 8), [1700.0]), 0.0)

    def test_no_columns(self):
        self.assertEqual(tilt_of(pair(500, 8), []), 0.0)


def code(c, y):
    return ln(c, 370, 420, y)


class Owners(unittest.TestCase):
    def test_a_column_off_its_codes_followed(self):
        # rows a line apart; the page falls 0.025 px a px, so even the first column lies
        # past half a line below its codes: by the tilt, still its own row's
        t = 0.025
        codes = [code(str(10 + j), 500 + H * j) for j in range(6)]
        lines = list(codes)
        for j in range(6):
            y = 500 + H * j
            lines += [ln("1.234.567", 1450, 1700, y + t * (1575 - 395)),
                      ln("7.654.321", 1850, 2100, y + t * (1975 - 395))]
        got = {(k, c["text"]) for k, _, c in owners(lines, COLS, codes, t)}
        self.assertEqual(got, {(k, str(10 + j)) for k in (0, 1) for j in range(6)})

    def test_a_dash_beside_a_wrapped_label(self):
        # a label over three lines prints its figures on the last, a dash among them: the
        # dash's box is a few px tall, a little off the label line's middle, and still beside it
        codes = [code(c, y) for c, y in (("01", 300), ("02", 400), ("10", 500), ("11", 700))]
        lines = list(codes) + [ln("Doanh thu", 450, 700, 500), ln("bán hàng và", 450, 700, 550),
                               ln("cung cấp", 450, 700, 600), ln("-", 1680, 1700, 604, h=6),
                               ln("7.654.321", 1850, 2100, 600)]
        for y in (300, 400, 700):
            lines += [ln("1.234.567", 1450, 1700, y), ln("7.654.321", 1850, 2100, y)]
        got = {(k, f["text"], c["text"]) for k, f, c in owners(lines, COLS, codes)}
        self.assertIn((0, "-", "10"), got)


class Continued(unittest.TestCase):
    def test_a_notes_column_beside_the_codes(self):
        # the codes' place on the page before reaches the notes beside them: codes only
        lines = [code(c, 500 + 100 * j) for j, c in enumerate(["10", "11", "12", "13"])]
        lines += [ln(n, 440, 470, 500 + 100 * j) for j, n in ((0, "15"), (2, "16"))]
        got = code_column(lines, COLS, known=(370, 420))
        self.assertEqual(sorted(c["text"] for c in got), ["10", "11", "12", "13"])


if __name__ == "__main__":
    unittest.main()
