"""check_i: the answers its where and columns rounds take, and what each round asks.
Run: .venv/bin/python -I -m unittest discover tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from check_i import columns_given, pages_of, period, shown, valid  # noqa: E402


class Respaced(unittest.TestCase):
    def test_spaced_figures(self):
        from check_i import respaced
        self.assertEqual(respaced("12 345 678"), "12.345.678")
        self.assertEqual(respaced(" (1 234) "), "(1.234)")
        for t in ("12 34", "1234 567", "Quý 3", "12/34/5678", "1.234.567", "10"):
            self.assertEqual(respaced(t), t, t)


class Aligned(unittest.TestCase):
    def test_a_page_with_a_column_fewer(self):
        """Page 2 shows columns 1, 2 and 4 of page 1's four, a little left: taken by place."""
        from check_i import aligned
        v = lambda n: {"value": n}                              # noqa: E731
        s = {"columns": [{"page": 1, "x": [1000, 1300, 1600, 1900]}, {"page": 2, "x": [985, 1285, 1885]}],
             "rows": [{"code": "01", "page": 1, "values": [v(1), v(2), v(3), v(4)]},
                      {"code": "50", "page": 2, "values": [v(5), v(6), v(8)]},
                      {"code": "", "page": 2, "values": [v(9), None, None]}]}
        by, where = aligned(s)
        self.assertEqual({i: x for i, x in by["50"].items()}, {0: [5], 1: [6], 3: [8]})
        self.assertEqual(by["01"][2], [3])
        self.assertEqual(where["50", 3], (2, [1285 + 12, 1885]))
        self.assertEqual(where["50", 2], (2, [1300 + 12, 1600]))   # not found there: page 1's place
        self.assertEqual(where["50", 0], (2, [985 - 420, 985]))


class PagesOf(unittest.TestCase):
    def test_pages(self):
        self.assertEqual(pages_of("7", 20), [7])
        self.assertEqual(pages_of(" 7 - 9 ", 20), [7, 8, 9])
        self.assertEqual(pages_of("7, 9", 20), [7, 9])
        self.assertEqual(pages_of("9,7-8,8", 20), [7, 8, 9])

    def test_not_pages(self):
        for t in ("", "0", "21", "9-7", "7-21", "p7", "7;8", ","):
            self.assertIsNone(pages_of(t, 20), t)


class ColumnsGiven(unittest.TestCase):
    def test_columns(self):
        self.assertEqual(columns_given("2|4", 4), {"period": 1, "to date": 3})
        self.assertEqual(columns_given("2|", 4), {"period": 1})
        self.assertEqual(columns_given("|3", 4), {"to date": 2})

    def test_not_columns(self):
        for t in ("|", "", "2", "0|1", "5|1", "a|1", "1|2|3"):
            self.assertIsNone(columns_given(t, 4), t)


class Asked(unittest.TestCase):
    def test_period(self):
        self.assertEqual(period({"pair": "balance", "period": "year", "year": 2099}),
                         "the balance at the end of financial year 2099")
        self.assertEqual(period({"pair": "balance", "period": "quarter", "n": 4, "year": 2099}),
                         "the balance at the end of quarter 4 of 2099")
        self.assertEqual(period({"pair": "flow", "period": "half", "year": 2099}), "the first half of 2099")
        self.assertEqual(period({"pair": "flow", "period": "quarter", "n": 3, "year": 2099}), "quarter 3 of 2099")

    def test_no_reading_is_shown(self):
        cell = {"statement": "B02", "status": "differ", "base": {"value": 123, "page": 4},
                "later": {"value": 456, "page": 5}}
        out = shown({"id": "ab12cd34", "kind": "type", "c": cell, "side": "base"})
        self.assertEqual(out, {"id": "ab12cd34", "kind": "type", "statement": "B02", "page": 4})  # the page, to see its headings
        where = shown({"id": "ab12cd34", "kind": "where", "statement": "B03", "side": "later",
                       "sha256": "f" * 64, "pages": 20, "ask": "quarter 3 of 2099"})
        self.assertNotIn("sha256", where)

    def test_valid(self):
        where = {"kind": "where", "pages": 20}
        self.assertTrue(valid(where, "pages", "7-8"))
        self.assertTrue(valid(where, "not printed", ""))
        self.assertFalse(valid(where, "pages", "30"))
        cols = {"kind": "columns", "k": 4}
        self.assertTrue(valid(cols, "columns", "1|3"))
        self.assertTrue(valid(cols, "not this", ""))
        self.assertFalse(valid(cols, "columns", "|"))
        typ = {"kind": "type"}
        self.assertTrue(valid(typ, "typed", "1.234"))
        self.assertTrue(valid(typ, "dash", ""))
        self.assertFalse(valid(typ, "typed", "abc"))
        self.assertFalse(valid(typ, "pages", "7"))


if __name__ == "__main__":
    unittest.main()
