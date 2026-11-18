"""The code crosswalk between the 2014 and 2026 forms, and a truth side keyed by it.
Run: .venv/bin/python -I -m unittest discover tests"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from fs_codes import OLD_TO_NEW, old_of, to_new  # noqa: E402
from fs_sums import truth_sides  # noqa: E402


class OldOf(unittest.TestCase):
    def test_a_renumbered_line(self):
        self.assertEqual(old_of("B02", "27"), "24")
        self.assertEqual(old_of("B01", "420"), "421")

    def test_a_line_kept_its_number(self):
        self.assertEqual(old_of("B02", "26"), "26")
        self.assertEqual(old_of("B03", "33"), "33")

    def test_a_new_line_whose_number_is_another_old_line(self):
        # 2026's 421 and B02 21 are new lines; 2014's 421 and 21 are other lines
        self.assertIsNone(old_of("B01", "421"))
        self.assertIsNone(old_of("B02", "21"))

    def test_every_renumbered_line_comes_back(self):
        for st in ("B01", "B02", "B03"):
            for old, new in OLD_TO_NEW[st].items():
                if new is not None:
                    self.assertEqual(old_of(st, new), old, (st, old))
                    self.assertEqual(to_new(st, old), new)


def cell(code, base_code, base_value, later_value, status):
    return {"pair": "balance", "statement": "B01", "code": code,
            "base": {"sha256": "a" * 64, "column": 0, "code": base_code, "page": 1, "value": base_value},
            "later": {"sha256": "b" * 64, "column": 1, "page": 1, "value": later_value},
            "status": status}


class TruthSides(unittest.TestCase):
    def sides(self, cells):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "cells.jsonl"), "w", encoding="utf-8") as fh:
                fh.writelines(json.dumps(c) + "\n" for c in cells)
            return truth_sides(d)

    def test_a_new_only_line_does_not_blank_the_old_line_of_its_number(self):
        # the 2014 base's 421 is the 2026 form's 420; the 2026 form's own 421 has no base line
        got = self.sides([cell("420", "421", 700, 700, "agree"),
                          cell("421", None, "missing", 300, "one side")])
        self.assertEqual(got["a" * 64, "B01", 0]["421"], (700, "true"))
        self.assertEqual(got["b" * 64, "B01", 1]["421"], (300, "read"))


if __name__ == "__main__":
    unittest.main()
