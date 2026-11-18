"""read_pdf.code_between, code_by_order and put_code: a row code left empty, read by order, on invented rows
and lines. Run: .venv/bin/python -I -m unittest discover tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from read_pdf import NO_CODE, code_between, code_by_order, put_code  # noqa: E402


def row(code, page=1, sure=True):
    return {"code": code, "page": page, "code_sure": sure}


class Between(unittest.TestCase):
    def test_sure_codes_either_side(self):
        self.assertEqual(code_between([row("10"), row(None), row("20")], 1), (10, 20, True))

    def test_an_unsure_neighbour_is_passed_over(self):
        rows = [row("10"), row("11", sure=False), row(None), row("20")]
        self.assertEqual(code_between(rows, 2), (10, 20, True))

    def test_the_code_below_on_the_next_page(self):
        self.assertEqual(code_between([row("10"), row(None), row("20", page=2)], 1), (10, 20, True))

    def test_the_first_row(self):
        self.assertEqual(code_between([row(None), row("20")], 0), (NO_CODE[0], 20, False))

    def test_the_last_row(self):
        self.assertEqual(code_between([row("60"), row(None)], 1), (60, NO_CODE[1], False))

    def test_no_room_between(self):
        self.assertIsNone(code_between([row("10"), row(None), row("11")], 1))


class ByOrder(unittest.TestCase):
    def test_both_agree_inside(self):
        self.assertEqual(code_by_order(("11", "11"), (10, 20, True), set()), "11")

    def test_both_agree_outside(self):
        self.assertIsNone(code_by_order(("21", "21"), (10, 20, True), set()))

    def test_a_code_read_already(self):
        self.assertIsNone(code_by_order(("70", "70"), (60, NO_CODE[1], False), {"70"}))

    def test_one_read_fits(self):
        self.assertEqual(code_by_order(("149", "140"), (139, 141, True), set()), "140")

    def test_both_reads_fit(self):
        self.assertIsNone(code_by_order(("14", "11"), (10, 20, True), set()))

    def test_one_read_fits_with_one_side_open(self):
        self.assertIsNone(code_by_order(("30", "60"), (50, NO_CODE[1], False), set()))

    def test_not_digits(self):
        self.assertIsNone(code_by_order(("1a", "1a"), (10, 20, True), set()))


def line(text, box):
    return {"text": text, "box": box, "sure": False}


class Put(unittest.TestCase):
    def test_a_code_alone(self):
        lines = [line("[?]", [100, 10, 150, 40])]
        self.assertTrue(put_code(lines, [100, 10, 150, 40], "20"))
        self.assertEqual((lines[0]["text"], lines[0]["sure"], lines[0]["look"]), ("20", True, "code by order"))

    def test_a_code_before_its_label(self):
        lines = [line("[?] A. TÀI SẢN", [100, 10, 400, 40])]
        self.assertTrue(put_code(lines, [100, 10, 160, 40], "100"))
        self.assertEqual([ln["text"] for ln in lines], ["100", "A. TÀI SẢN"])
        self.assertFalse(lines[1]["sure"])

    def test_a_code_after_its_label(self):
        lines = [line("Giá vốn [?]", [100, 10, 400, 40])]
        self.assertTrue(put_code(lines, [340, 10, 400, 40], "11"))
        self.assertEqual([ln["text"] for ln in lines], ["Giá vốn", "11"])

    def test_a_line_elsewhere_is_left(self):
        lines = [line("[?]", [100, 100, 150, 130])]
        self.assertFalse(put_code(lines, [100, 10, 150, 40], "20"))
        self.assertEqual(lines[0]["text"], "[?]")


if __name__ == "__main__":
    unittest.main()
