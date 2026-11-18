"""Cross-period truth on invented statements. Run: .venv/bin/python -I -m unittest discover tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from fs_truth import cells_of, doubts, holds, match, reprinted  # noqa: E402

CODES = ["01", "02", "10", "11", "20", "21", "22", "30"]


def printing(page, cols):
    """A statement: per code one value per column (None: unread)."""
    rows = [{"code": c, "page": page,
             "values": [None if v is None else {"value": v} for v in vals]}
            for c, vals in zip(CODES, zip(*cols))]
    return {"rows": rows}


class Reprinted(unittest.TestCase):
    def setUp(self):
        prior = [101, 2, 99, 40, 59, 7, 3, 63]
        self.later = printing(12, [[x + 5 for x in prior], prior])
        self.full = printing(10, [prior])
        self.gaps = printing(11, [[0] + prior[1:6] + [None, None], [1] * 8])   # a quarter column too
        self.tried = []
        for x in (self.full, self.gaps):
            cb, cl = cells_of(x), cells_of(self.later)
            self.tried.append((match(cb, cl), cb, cl, x, self.later, {}))

    def test_unread_cell_taken_from_the_reprint_that_matched_the_same_column(self):
        self.assertEqual(reprinted(self.tried, self.later, 1, "22", 0, None, self.gaps), (3, self.full))

    def add(self, page, col, later=None):
        later = later or self.later
        x = printing(page, [col])
        cb, cl = cells_of(x), cells_of(later)
        self.tried.append((match(cb, cl), cb, cl, x, later, {}))

    def test_reprint_disagreeing_with_the_other_filing_is_not_used(self):
        self.tried = self.tried[1:]                       # only the gaps printing, and:
        self.add(9, [101, 2, 99, 40, 59, 7, 4, 64])       # one that disagrees somewhere
        self.assertEqual(reprinted(self.tried, self.later, 1, "22", 0, None, self.gaps)[0], None)

    def test_reprints_reading_two_values_give_none(self):
        # 22 unread in the other filing: two printings agree on all else, not on 22
        later = printing(12, [[1] * 8, [101, 2, 99, 40, 59, 7, None, 63]])
        self.tried = []
        for page, col in ((10, [101, 2, 99, 40, 59, 7, 3, 63]), (9, [101, 2, 99, 40, 59, 7, 4, 63]),
                          (11, [101, 2, 99, 40, 59, 7, None, 63])):
            self.add(page, col, later)
        gaps = self.tried[2][3]
        self.assertEqual(reprinted(self.tried, later, 1, "22", 0, None, gaps), (None, gaps))

if __name__ == "__main__":
    unittest.main()


def filed(kind, rows):
    """A printing filed as kind: rows as (page, code)."""
    return {"statement": kind, "pages": sorted({p for p, _ in rows}),
            "rows": [{"page": p, "code": c, "values": []} for p, c in rows]}


class Doubts(unittest.TestCase):
    def test_codes_tell_the_statements_apart(self):
        self.assertTrue(holds("B01", "270"))
        self.assertFalse(holds("B01", "20"))
        self.assertFalse(holds("B02", "08"))        # a cash flow statement's only
        self.assertTrue(holds("B03", "20"))         # shared by B02 and B03
        self.assertFalse(holds("B03", "51"))

    def test_page_whose_form_label_names_another_statement(self):
        s = filed("B01", [(3, "110"), (3, "120"), (4, "300"), (4, "400")])
        self.assertEqual(doubts(s, {3: "B01", 4: "B02"}), {4: {"B02"}})

    def test_codes_under_100_under_a_balance_sheet_name_both_flows(self):
        s = filed("B01", [(3, "110"), (3, "270"), (4, "01"), (4, "10"), (4, "20")])
        self.assertEqual(doubts(s, {}), {4: {"B02", "B03"}})

    def test_one_only_statement_named_when_its_codes_say(self):
        s = filed("B01", [(3, "110"), (4, "01"), (4, "51"), (4, "52")])
        self.assertEqual(doubts(s, {}), {4: {"B02"}})

    def test_cash_flow_codes_under_an_income_statement(self):
        s = filed("B02", [(5, "01"), (5, "02"), (6, "03"), (6, "05"), (6, "20")])
        self.assertEqual(doubts(s, {}), {6: {"B03"}})

    def test_one_stray_code_is_not_enough(self):
        s = filed("B02", [(5, "01"), (5, "08"), (5, "20")])
        self.assertEqual(doubts(s, {}), {})

    def test_label_and_codes_that_agree_raise_nothing(self):
        s = filed("B03", [(7, "01"), (7, "08"), (7, "20")])
        self.assertEqual(doubts(s, {7: "B03"}), {})
