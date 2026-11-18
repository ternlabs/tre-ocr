"""read_pdf stuck_on, unstamped and look_again: an amount read with a stamp's marks stuck to it,
read once more with the red ink taken out, on invented images and readers.
Run: .venv/bin/python -I -m unittest discover tests"""
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from read_pdf import RED, Readers, again, stuck_on, unstamped  # noqa: E402

BOX = [200, 100, 500, 150]          # a figure's line on the page


def page_png(tmp, stamp=True):
    """A white page with a black figure-like bar in BOX and, if asked, a red mark over its end."""
    a = np.full((300, 800, 3), 255, dtype=np.uint8)
    a[115:135, 210:480] = 20                        # the figure's ink
    if stamp:
        a[105:145, 470:540] = (200, 30, 40)         # the stamp's ink
        a[115:135, 470:480] = (60, 20, 40)          # the figure's ink printed over it
    png = os.path.join(tmp, "page.png")
    Image.fromarray(a).save(png)
    return png


def readers(tuned, stock, seen):
    """Readers that answer as told and keep whether each crop held red ink."""
    r = Readers.__new__(Readers)
    r.tuned, r.stock = "tuned", "stock"

    def line(ocr, png):
        a = np.array(Image.open(png).convert("RGB"), dtype=int)
        seen.append(bool((a[..., 0] - np.maximum(a[..., 1], a[..., 2]) > RED[0]).any()))
        return (tuned, 0.99) if ocr == "tuned" else (stock, 0.99)
    r._line = line
    return r


def line(text, sure=True):
    return {"text": text, "box": list(BOX), "sure": sure}


class Stuck(unittest.TestCase):
    def test_marks_after_an_amount(self):
        self.assertTrue(stuck_on("(1.234.567)Ư"))
        self.assertTrue(stuck_on("(1.234.567)123"))

    def test_marks_before_an_amount(self):
        self.assertTrue(stuck_on("Y 1.234.567"))

    def test_a_whole_amount_is_not(self):
        self.assertFalse(stuck_on("(1.234.567)"))
        self.assertFalse(stuck_on("1.234.567"))
        self.assertFalse(stuck_on("( 1.234.567 )"))         # spaces inside the brackets

    def test_a_label_with_a_figure_is_not(self):
        self.assertFalse(stuck_on("Doanh thu 1.234.567"))

    def test_marks_alone_are_not(self):
        self.assertFalse(stuck_on("ƯY"))
        self.assertFalse(stuck_on("[?]"))


class Unstamped(unittest.TestCase):
    def test_red_ink_made_white(self):
        with tempfile.TemporaryDirectory() as tmp:
            piece, stamped = unstamped(Image.open(page_png(tmp)).convert("RGB"))
        a = np.array(piece)
        self.assertTrue(stamped)
        self.assertEqual(tuple(a[110, 520]), (255, 255, 255))
        self.assertEqual(tuple(a[125, 300]), (20, 20, 20))

    def test_dark_ink_under_the_stamp_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            piece, _ = unstamped(Image.open(page_png(tmp)).convert("RGB"))
        self.assertEqual(tuple(np.array(piece)[125, 475]), (60, 20, 40))

    def test_a_few_red_pixels_are_no_stamp(self):
        a = np.full((50, 50, 3), 255, dtype=np.uint8)
        a[0:5, 0:5] = (200, 30, 40)
        piece, stamped = unstamped(Image.fromarray(a))
        self.assertFalse(stamped)
        self.assertEqual(tuple(np.array(piece)[0, 0]), (200, 30, 40))


class Look(unittest.TestCase):
    def look(self, ln, tuned, stock, stamp=True, stamps_only=False):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            out = readers(tuned, stock, seen).look_again(page_png(tmp, stamp), [ln], stamps_only)
        return out[0], seen

    def test_a_stuck_amount_under_a_stamp(self):
        out, seen = self.look(line("(1.234.567)Ư"), "(1.234.567)", "(1.234.567)")
        self.assertEqual((out["text"], out["sure"], out["look"]), ("(1.234.567)", True, "stamp"))
        self.assertEqual(seen, [False, False])          # read with the red taken out

    def test_spaces_round_the_read_trimmed(self):
        out, _ = self.look(line("(1.234.567)Ư"), "(1.234.567) ", "(1.234.567)")
        self.assertEqual(out["text"], "(1.234.567)")

    def test_stuck_marks_with_no_stamp_left(self):
        ln = line("(1.234.567)Ư")
        out, seen = self.look(ln, "(1.234.567)", "(1.234.567)", stamp=False)
        self.assertEqual(out, ln)
        self.assertEqual(seen, [])

    def test_a_read_not_a_whole_amount_left(self):
        ln = line("(1.234.567)Ư")
        out, _ = self.look(ln, "(1.234.567) Ư", "(1.234.567) Ư")
        self.assertEqual(out, ln)

    def test_a_stuck_line_with_no_box_left(self):
        ln = {**line("(1.234.567)Ư"), "box": []}
        out, seen = self.look(ln, "(1.234.567)", "(1.234.567)")
        self.assertEqual((out, seen), (ln, []))

    def test_readers_that_differ_left(self):
        ln = line("(1.234.567)Ư")
        out, _ = self.look(ln, "(1.234.567)", "(1.284.567)")
        self.assertEqual(out, ln)

    def test_an_unread_amount_under_a_stamp(self):
        out, seen = self.look(line("([?]).Y", sure=False), "(1.234.567)", "(1.234.567)")
        self.assertEqual((out["text"], out["look"]), ("(1.234.567)", "stamp"))
        self.assertEqual(seen, [False, False])

    def test_an_unread_amount_with_no_stamp(self):
        out, _ = self.look(line("[?]", sure=False), "1.234.567", "1.234.567", stamp=False)
        self.assertEqual((out["text"], out["look"]), ("1.234.567", "crop"))

    def test_a_sure_whole_amount_not_looked_at(self):
        ln = line("1.234.567")
        out, seen = self.look(ln, "7.654.321", "7.654.321")
        self.assertEqual((out, seen), (ln, []))


def dash_png(tmp):
    """A white page with one short bar, a dash printed for zero, in DASH_BOX."""
    a = np.full((200, 700, 3), 255, dtype=np.uint8)
    a[99:103, 105:120] = 0
    png = os.path.join(tmp, "dash.png")
    Image.fromarray(a).save(png)
    return png


DASH_BOX = [100, 90, 125, 112]


class StampsOnly(unittest.TestCase):
    """A draft looked at by an older rule: only lines under a stamp are looked at again."""
    look = Look.look

    def dash(self, stamps_only):
        ln = {"text": "[?]", "box": list(DASH_BOX), "sure": False}
        with tempfile.TemporaryDirectory() as tmp:
            return ln, readers("", "", []).look_again(dash_png(tmp), [ln], stamps_only)[0]

    def test_a_dash_told_by_its_ink(self):
        _, out = self.dash(False)
        self.assertEqual((out["text"], out["look"]), ("-", "dash"))

    def test_no_dash_look_for_stamps_only(self):
        ln, out = self.dash(True)
        self.assertEqual(out, ln)

    def test_again_passes_it_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.rename(dash_png(tmp), os.path.join(tmp, "p1.png"))
            ln = {"text": "[?]", "box": list(DASH_BOX), "sure": False}
            pages = [{"kind": "read", "key": "p1", "lines": [ln]}, {"kind": "text", "lines": []}]
            again(pages, tmp, readers("", "", []), stamps_only=True)
            self.assertEqual(pages[0]["lines"], [ln])
            again(pages, tmp, readers("", "", []))
            self.assertEqual(pages[0]["lines"][0]["text"], "-")

    def test_an_unread_line_with_no_stamp_left(self):
        ln = line("[?] 13. Tổng lợi nhuận", sure=False)
        out, seen = self.look(ln, "50 13. Tổng lợi nhuận", "50 13. Tổng lợi nhuận",
                              stamp=False, stamps_only=True)
        self.assertEqual((out, seen), (ln, []))

    def test_an_unread_amount_under_a_stamp(self):
        out, _ = self.look(line("([?]).Y", sure=False), "(1.234.567)", "(1.234.567)", stamps_only=True)
        self.assertEqual((out["text"], out["look"]), ("(1.234.567)", "stamp"))

    def test_a_stuck_amount_under_a_stamp(self):
        out, _ = self.look(line("(1.234.567)Ư"), "(1.234.567)", "(1.234.567)", stamps_only=True)
        self.assertEqual((out["text"], out["look"]), ("(1.234.567)", "stamp"))


if __name__ == "__main__":
    unittest.main()
