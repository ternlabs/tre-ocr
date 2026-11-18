"""score_fs: a side's truth, the candidate's answer by place, and the scores. Invented values.
Run: .venv/bin/python -I -m unittest discover tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from score_fs import answer_of, by_statement, judged, read_at, shares, signed, truth_of, verdict, wilson  # noqa: E402


def cell(status, b, l, **kw):
    return {"statement": "B02", "code": "10", "status": status, "told": False,
            "base": {"sha256": "a" * 64, "value": b}, "later": {"sha256": "b" * 64, "value": l}, **kw}


class Truth(unittest.TestCase):
    def test_agreed(self):
        self.assertEqual(truth_of(cell("agree", 700, 700), "base", None), (700, "agreed"))

    def test_typed_wins(self):
        rec = {"base": {"typed": 701}, "later": {}}
        self.assertEqual(truth_of(cell("differ", 700, 701), "base", rec), (701, "typed"))
        self.assertEqual(truth_of(cell("differ", 700, 701), "later", rec), (None, "not typed"))

    def test_check_that_differs_drops_the_other_side(self):
        rec = {"base": {"typed": 709}, "later": {}}
        self.assertEqual(truth_of(cell("agree", 700, 700), "later", rec), (None, "check differs"))
        rec = {"base": {"typed": 700}, "later": {}}
        self.assertEqual(truth_of(cell("agree", 700, 700), "later", rec), (700, "agreed"))

    def test_unreadable_and_blank(self):
        rec = {"base": {"typed": None}, "later": {"typed": "missing"}}
        self.assertEqual(truth_of(cell("differ", 1, 2), "base", rec), (None, "unreadable"))
        self.assertEqual(truth_of(cell("differ", 1, 2), "later", rec), ("missing", "typed"))

    def test_sign(self):
        self.assertEqual(signed(-55, "plain"), 55)
        self.assertEqual(signed(55, "parentheses"), -55)
        self.assertEqual(signed(55, "dash"), 0)
        self.assertEqual(signed(55, "nothing"), "missing")
        self.assertIsNone(signed(55, "unreadable"))
        rec = {"base": {"printed": "minus"}, "later": {"printed": "unreadable"}}
        self.assertEqual(truth_of(cell("sign differs", 55, -55), "base", rec), (-55, "sign"))
        self.assertEqual(truth_of(cell("sign differs", 55, -55), "later", rec), (None, "unreadable"))


class Candidate(unittest.TestCase):
    s = [{"statement": "B02", "pages": [8, 9],
          "columns": [{"page": 8, "x": [1000, 1300]}, {"page": 9, "x": [1010, 1290]}],
          "rows": [{"code": "10", "page": 9, "values": [{"value": 4}, {"value": 5}]},
                   {"code": "20", "page": 9, "values": [{"value": None}, None]},
                   {"code": "30", "page": 9, "values": [{"value": 6}, None]},
                   {"code": "30", "page": 9, "values": [{"value": 7}, None]}]}]

    def test_in_row(self):
        from score_fs import read_in_row
        self.assertEqual(read_in_row(self.s, "B02", "10", 9, 0), 4)
        self.assertEqual(read_in_row(self.s, "B02", "10", None, 1), 5)          # no page recorded: any
        self.assertEqual(read_in_row(self.s, "B02", "10", 8, 1), "missing")     # not on that page
        self.assertEqual(read_in_row(self.s, "B03", "10", 9, 1), "missing")     # under another statement
        self.assertIsNone(read_in_row(self.s, "B02", "20", 9, 0))
        self.assertEqual(read_in_row(self.s, "B02", "20", 9, 1), "missing")
        self.assertEqual(read_in_row(self.s, "B02", "30", 9, 0), "ambiguous")

    def test_by_place(self):
        self.assertEqual(read_at(self.s, "B02", "10", 9, 1300), 5)
        self.assertEqual(read_at(self.s, "B02", "10", 9, 1000), 4)
        self.assertEqual(read_at(self.s, "B02", "10", 9, 1600), "missing")    # no column there
        self.assertEqual(read_at(self.s, "B02", "10", 8, 1300), "missing")    # not on that page
        self.assertEqual(read_at(self.s, "B03", "10", 9, 1300), "missing")
        self.assertIsNone(read_at(self.s, "B02", "20", 9, 1000))
        self.assertEqual(read_at(self.s, "B02", "20", 9, 1300), "missing")
        self.assertEqual(read_at(self.s, "B02", "30", 9, 1000), "ambiguous")

    def test_answer(self):
        c = cell("differ", 1, 2)
        c["later"].update({"page": 9, "column": 1})
        # matched: looked up in the reading scored, not the value kept in the cell
        self.assertEqual(answer_of(c, "later", lambda sha: self.s), 5)
        t = {**c, "told": True, "later": {"sha256": "b" * 64, "value": 99, "page": 9, "band": [1022, 1290]}}
        self.assertEqual(answer_of(t, "later", lambda sha: self.s), 5)
        t["later"]["page"] = None
        self.assertEqual(answer_of(t, "later", lambda sha: self.s), "missing")

    def test_a_matched_cell_by_its_place(self):
        # the truth's reading had one column of its own on page 9 (at 1300), so its column 0;
        # the candidate carries two there: its column 0 is the other one
        c = cell("agree", 5, 5)
        c["later"].update({"page": 9, "column": 0, "x": 1300})
        self.assertEqual(answer_of(c, "later", lambda sha: self.s), 5)
        made = [{"statement": "B02", "pages": [9], "columns": [{"page": 9, "x": [1295]}],
                 "rows": [{"code": "10", "page": 9, "values": [{"value": 5}]}]}]
        del c["later"]["x"]
        self.assertEqual(answer_of(c, "later", lambda sha: self.s, lambda sha: made), 5)
        self.assertEqual(answer_of(c, "later", lambda sha: self.s), 4)        # no place: its number
        c["later"]["page"] = 8                                                  # no such row where it was made
        self.assertEqual(answer_of(c, "later", lambda sha: self.s, lambda sha: made), "missing")

    def test_the_column_where_the_truth_was_made(self):
        from score_fs import column_x
        self.assertEqual(column_x(self.s, "B02", "10", 9, 1), 1290)
        self.assertIsNone(column_x(self.s, "B02", "10", 9, 2))
        self.assertIsNone(column_x(self.s, "B02", "10", 8, 0))
        self.assertIsNone(column_x(self.s, "B03", "10", 9, 0))


class Scores(unittest.TestCase):
    def test_judged(self):
        self.assertEqual(judged(5, 5)["right"], True)
        self.assertTrue(judged(5, -5)["sign only"] and judged(5, -5)["wrong"])
        self.assertEqual((judged(5, None)["sure"], judged(5, None)["wrong"]), (False, False))
        self.assertTrue(judged(5, "ambiguous")["wrong"])
        self.assertTrue(judged("missing", 3)["wrong"] and not judged("missing", 3)["figure"])
        self.assertFalse(judged("missing", "missing")["wrong"])
        self.assertFalse(judged(5, "missing")["sure"])

    def test_shares(self):
        sides = [judged(1, 1), judged(2, 3), judged(4, None), judged("missing", 9)]
        self.assertEqual(shares(sides), {"cell recall": (1, 3), "coverage": (2, 3), "wrong when sure": (2, 3)})

    def test_wilson(self):
        lo, hi = wilson(0, 100)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.037, places=3)
        lo, hi = wilson(50, 100)
        self.assertAlmostEqual(lo, 0.404, places=3)
        self.assertEqual(wilson(0, 0), (None, None))

    def test_by_statement_is_fixed_and_wider_for_clustered_errors(self):
        sides = []
        for k in range(20):
            for _ in range(10):
                sides.append({**judged(1, 1 if k else 2), "statement key": k})
        a, b = by_statement(sides), by_statement(sides)
        self.assertEqual(a, b)
        lo, hi = a["cell recall"]
        self.assertLessEqual(lo, 0.9)
        self.assertEqual(hi, 1.0)

    def test_clustered_figures_widen_the_interval(self):
        from score_fs import clustered, resampled
        # 40 printings of 50 figures, one wholly misfiled: 50 of 2000 wrong, all in one
        groups = {k: (50 if k == 0 else 0, 50) for k in range(40)}
        lo, hi = clustered(groups)
        w = wilson(50, 2000)
        self.assertEqual(resampled(groups), resampled(groups))         # fixed seed
        self.assertLess(lo, w[0])
        self.assertGreater(hi, w[1])
        # none wrong: resampling gives [0, 0], Wilson's upper bound stays
        none = {k: (0, 50) for k in range(40)}
        self.assertEqual(resampled(none), (0.0, 0.0))
        self.assertEqual(clustered(none), wilson(0, 2000))
        self.assertEqual(clustered({}), (None, None))

    def test_verdict(self):
        self.assertEqual(verdict("cell recall", 99, 100, wilson(99, 100)), "meets the mark")
        self.assertEqual(verdict("wrong when sure", 0, 2000, wilson(0, 2000)),
                         "meets the mark, interval clear of it")
        self.assertEqual(verdict("coverage", 1, 2, wilson(1, 2)), "misses the mark")


if __name__ == "__main__":
    unittest.main()


class Misfiled(unittest.TestCase):
    def test_counts_sure_figures_under_another_statement(self):
        from score_fs import misfiled
        items = [{"id": "w1", "sha256": "a" * 64, "statement": "B02", "pages": 9},
                 {"id": "w2", "sha256": "a" * 64, "statement": "B03", "pages": 9}]
        answers = {"w1": {"answer": "pages", "value": "6"}, "w2": {"answer": "pages", "value": "7"}}
        row = lambda page, *vals: {"page": page, "values": [{"value": v} if v != "-" else None for v in vals]}
        reading = [{"statement": "B01", "rows": [row(5, 10, 20), row(6, 30, None), row(7, 40, "-")]},
                   {"statement": "B03", "rows": [row(7, 50, 60)]}]
        # page 5 not pointed to; page 6 (B02) under B01: 1 sure; page 7 (B03): 1 under B01, 2 right
        got = misfiled(items, answers, lambda sha: reading)
        self.assertEqual(got[:3], (2, 4, 2))
        self.assertEqual(got[3], {("a" * 64, "B02"): (1, 1), ("a" * 64, "B03"): (1, 3)})

    def test_not_printed_points_to_nothing(self):
        from score_fs import misfiled
        items = [{"id": "w1", "sha256": "a" * 64, "statement": "B02", "pages": 9}]
        reading = [{"statement": "B01", "rows": [{"page": 6, "values": [{"value": 1}]}]}]
        self.assertEqual(misfiled(items, {"w1": {"answer": "not printed", "value": ""}},
                                  lambda sha: reading)[:3], (0, 0, 0))


class OutOfRange(unittest.TestCase):
    def test_counts_sure_figures_under_another_statements_codes(self):
        from score_fs import out_of_range
        row = lambda page, code, *vals: {"page": page, "code": code,
                                         "values": [{"value": v} if v is not None else None for v in vals]}
        reading = [{"statement": "B01", "rows": [row(4, "110", 1, 2), row(5, "20", 3, None), row(5, None, 4, 5)]},
                   {"statement": "B02", "rows": [row(6, "01", 6, 7), row(6, "152", 8, 9), row(6, "300", None, None)]}]
        # B01 code 20: 1 sure; B02 code 152: 2 sure; code 300 holds none; 9 sure in all
        self.assertEqual(out_of_range([{"sha256": "a" * 64}], lambda sha: reading)[:3], (3, 9, 2))
