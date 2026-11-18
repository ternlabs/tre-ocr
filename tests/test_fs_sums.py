"""The sums' behaviour on invented statements. Run: .venv/bin/python -I -m unittest discover tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from fs_sums import BLANK, demands, gap_kind, no_room, sums_of  # noqa: E402


def b02(**kw):
    """A 2026 B02 whose sums all hold: 10 = 01 - 02 ... 60 = 50 - 51 - 52."""
    v = {"01": 1000, "02": 100, "10": 900, "11": 600, "20": 300, "21": 50, "22": 40,
         "23": 10, "25": 20, "26": 30, "27": 0, "30": 300, "31": 5, "32": 5, "40": 0,
         "50": 300, "51": 60, "52": 0, "60": 240}
    v.update({k[1:]: x for k, x in kw.items()})
    return {k: x for k, x in v.items() if x != "absent"}


def by_name(rows):
    return {r[0]: r for r in rows}


class GroupSplit(unittest.TestCase):
    def test_company_with_split_absent_or_empty_is_not_checked(self):
        for extra in ({}, {"_61": None, "_62": None}, {"_61": 0, "_62": 0}):
            got = by_name(sums_of("2026", "B02", b02(**extra)))
            self.assertNotIn("60 split", got, extra)
            self.assertEqual(got["60"][2], "holds")

    def test_group_split_unread_is_unknown(self):
        got = by_name(sums_of("2026", "B02", b02(_61=None, _62=None), group=True))
        self.assertEqual(got["60 split"][2], "unknown")

    def test_split_with_one_figure_holds_through_blank(self):
        got = by_name(sums_of("2026", "B02", b02(_61=240, _62=None)))
        self.assertEqual(got["60 split"][2], "holds")
        self.assertIn("blanks", got["60 split"][3])

    def test_split_that_fails_reports_its_own_gap(self):
        got = by_name(sums_of("2026", "B02", b02(_61=200, _62=30), group=True))
        name, total, result, _, alts = got["60 split"]
        self.assertEqual((total, result), ("60", "fails"))
        self.assertEqual(got["60"][2], "holds")
        self.assertEqual(gap_kind(b02(_61=200, _62=30), total, alts, 2), "unexplained")

    def test_each_sum_carries_only_its_own_lines(self):
        got = by_name(sums_of("2026", "B02", b02(_61=240, _62=0), group=True))
        lines = lambda r: {c for alt in r[4] for _, c in alt}
        self.assertEqual(lines(got["60"]), {"50", "51", "52"})
        self.assertEqual(lines(got["60 split"]), {"61", "62"})


class Lines(unittest.TestCase):
    def test_unread_line_is_unknown_not_zero(self):
        got = by_name(sums_of("2026", "B02", b02(_02=None)))
        self.assertEqual(got["10"][2], "unknown")

    def test_absent_subtotal_is_replaced_by_its_lines(self):
        got = by_name(sums_of("2026", "B02", b02(_10="absent")))
        self.assertEqual(got["20"][2], "holds")
        self.assertNotIn("10", got)

    def test_deduction_printed_negative_holds(self):
        got = by_name(sums_of("2026", "B02", b02(_02=-100)))
        self.assertEqual(got["10"][2:4], ("holds", ("02",)))

    def test_wrong_total_fails(self):
        got = by_name(sums_of("2026", "B02", b02(_10=901)))
        self.assertEqual(got["10"][2], "fails")


class NoLines(unittest.TestCase):
    def test_a_zero_heading_with_no_lines_holds(self):
        got = by_name(sums_of("2026", "B01", {"150": 0}))
        self.assertEqual(got["150"][2], "holds")
        self.assertIn("none printed", got["150"][3])

    def test_a_figure_with_no_lines_is_unknown(self):
        got = by_name(sums_of("2026", "B01", {"150": 7}))
        self.assertEqual(got["150"][2], "unknown")


def rows(*at, page=1):
    """Rows by (code, y of the code's box, 40 high)."""
    return [{"code": c, "page": page, "code_box": [100, y, 160, y + 40]} for c, y in at]


class NoRoom(unittest.TestCase):
    def test_neighbours_one_after_the_other(self):
        self.assertTrue(no_room(rows(("229", 100), ("240", 150)), "230"))

    def test_room_for_a_row_between(self):
        self.assertFalse(no_room(rows(("229", 100), ("240", 180)), "230"))
        self.assertTrue(no_room(rows(("229", 100), ("240", 179)), "230"))

    def test_a_row_between_read_without_its_code(self):
        self.assertFalse(no_room(rows(("229", 100), (None, 145), ("240", 150)), "230"))

    def test_the_line_itself_is_printed(self):
        self.assertFalse(no_room(rows(("229", 100), ("240", 150), ("230", 300)), "230"))

    def test_over_a_page_break(self):
        r = rows(("229", 100)) + rows(("240", 150), page=2)
        self.assertFalse(no_room(r, "230"))

    def test_no_neighbour_on_one_side(self):
        self.assertFalse(no_room(rows(("229", 100), ("228", 150)), "230"))
        self.assertFalse(no_room(rows(("240", 100), ("250", 150)), "230"))

    def test_the_nearest_codes_decide(self):
        """210 and 240 follow one another nowhere: 229 stands between, further down the page."""
        self.assertFalse(no_room(rows(("210", 100), ("240", 150), ("229", 200)), "230"))

    def test_rows_that_overlap_are_not_a_page_to_trust(self):
        self.assertFalse(no_room(rows(("229", 100), ("240", 130)), "230"))


class Demands(unittest.TestCase):
    def of(self, **kw):
        """Of a B02 whose every sum holds, but for what kw changes."""
        return demands("2026", "B02", b02(**{"_30": 330, "_50": 330, "_60": 270, **kw}))

    def test_nothing_unread_nothing_asked(self):
        self.assertEqual(self.of(), {})

    def test_a_line(self):
        self.assertEqual(self.of(_21=None), {"21": {50}})

    def test_a_total_by_its_own_sum_or_the_one_it_is_a_line_of(self):
        self.assertEqual(self.of(_50=None), {"50": {330}})
        self.assertEqual(self.of(_50=None, _30=None), {"30": {330}, "50": {330}})
        self.assertEqual(self.of(_50=None, _30=None, _51=None), {"30": {330}})

    def test_a_deduction_plain_or_in_parentheses(self):
        self.assertEqual(self.of(_11=None), {"11": {600, -600}})

    def test_the_other_deductions_either_way(self):
        """23 in parentheses: minus its digits, minus what is printed, or (costs in
        parentheses, a plain one a credit) every deduction as printed."""
        self.assertEqual(self.of(_23=-10, _21=None), {"21": {50, 30, -50}})
        self.assertEqual(self.of(_23=10, _21=None), {"21": {50}})

    def test_two_unread_in_one_sum(self):
        self.assertEqual(self.of(_21=None, _22=None), {})

    def test_a_cell_not_a_figure_elsewhere_in_the_sum(self):
        self.assertEqual(self.of(_21=None, _20="twice"), {})

    def test_a_blank_cell_counts_as_zero(self):
        self.assertEqual(self.of(_27=BLANK, _21=None), {"21": {50}})

    def test_never_zero(self):
        self.assertEqual(self.of(_27=None), {})
        self.assertEqual(self.of(_40=None), {})

    def test_a_line_of_a_total_not_printed(self):
        self.assertEqual(self.of(_60="absent", _51=None), {})
        self.assertEqual(self.of(_40="absent", _31=None), {"31": {5}})      # 50 = 30 + 31 - 32


if __name__ == "__main__":
    unittest.main()
