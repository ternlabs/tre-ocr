"""fs_contents: the contents page as a flag on where the statements are found. Invented pages.
Run: .venv/bin/python -I -m unittest discover tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from fs_contents import check, listed  # noqa: E402


def line(text, y, x0=300, x1=1500):
    return {"text": text, "sure": True, "box": [x0, y, x1, y + 50]}


def contents(*rows, heading="MỤC LỤC"):
    """A contents page: (title, page number or None) per row, the number at the right."""
    lines = [line(heading, 200, 1000, 1300)]
    for i, (title, num) in enumerate(rows):
        y = 400 + 100 * i
        lines.append(line(title, y))
        if num is not None:
            lines.append(line(num, y + 5, 2100, 2200))
    return lines


ROWS = (("Báo cáo của Ban Giám đốc", "1 - 2"), ("Bảng cân đối kế toán", "3 - 4"),
        ("Báo cáo kết quả hoạt động kinh doanh", "5"), ("Báo cáo lưu chuyển tiền tệ", "6"),
        ("Thuyết minh báo cáo tài chính", "7 - 20"))


def draft(lines, n=12):
    pages = [{"page": 1, "kind": "read", "lines": [line("BÁO CÁO TÀI CHÍNH", 500)]},
             {"page": 2, "kind": "read", "lines": lines}]
    return {"pages": pages + [{"page": k, "kind": "read", "lines": []} for k in range(3, n + 1)]}


def found(**starts):
    """Statements found, as fs_rows gives them: kind -> its pages, or a list of them."""
    out = []
    for k, pages in starts.items():
        for ps in pages if isinstance(pages[0], list) else [pages]:
            out.append({"statement": k, "pages": ps})
    return out


class Listed(unittest.TestCase):
    def test_the_contents_give_printed_pages(self):
        self.assertEqual(listed(draft(contents(*ROWS))["pages"]),
                         (2, {"B01": (3, 4), "B02": (5, 5), "B03": (6, 6)}))

    def test_a_number_after_dot_leaders(self):
        page = contents(("Bảng cân đối kế toán ........ 3", None))
        self.assertEqual(listed(draft(page)["pages"])[1], {"B01": (3, 3)})

    def test_a_title_with_no_number_read_is_listed_with_none(self):
        page = contents(("Bảng cân đối kế toán", "3"), ("Báo cáo lưu chuyển tiền tệ", None))
        self.assertEqual(listed(draft(page)["pages"])[1], {"B01": (3, 3), "B03": None})

    def test_a_letter_naming_the_statements_is_no_contents(self):
        letter = [line("Về nội dung báo cáo, tình hình lưu chuyển tiền tệ cho 6", 400),
                  line("tháng đầu năm phản ánh trung thực.", 460)]
        self.assertEqual(listed(draft(letter)["pages"]), (None, {}))
        self.assertIsNone(check(draft(letter), found(B03=[5])))

    def test_a_page_with_no_number_read_is_no_contents(self):
        page = contents(("Báo cáo kết quả hoạt động kinh doanh", None), heading="NỘI DUNG")
        self.assertEqual(listed(draft(page)["pages"]), (None, {}))


class Flags(unittest.TestCase):
    def test_statements_where_listed_agree(self):
        got = check(draft(contents(*ROWS)), found(B01=[5, 6], B02=[7], B03=[8]))
        self.assertEqual((got["offset"], got["flags"]), (2, []))

    def test_listed_not_found(self):
        got = check(draft(contents(*ROWS)), found(B01=[5, 6], B03=[8]))
        self.assertEqual(got["flags"], ["B02: listed at printed page 5 (page 7), not found"])

    def test_found_away_from_its_listed_page(self):
        got = check(draft(contents(*ROWS)), found(B01=[5, 6], B02=[7], B03=[10]))
        self.assertEqual(got["flags"], ["B03: listed at printed page 6 (page 8), found starting at page 10"])

    def test_found_again_outside_its_listed_pages(self):
        got = check(draft(contents(*ROWS)), found(B01=[[5, 6], [11]], B02=[7], B03=[8]))
        self.assertEqual(got["flags"], ["B01: listed at pages 5-6, found starting again at page 11"])

    def test_a_single_listed_page_runs_to_the_next_listed(self):
        # B01 listed at printed page 3 alone, B02 at 5: B01 found again at page 6 (printed 4)
        # is its own second page started afresh, not a printing elsewhere
        page = contents(("Bảng cân đối kế toán", "3"), ("Báo cáo kết quả hoạt động kinh doanh", "5"),
                        ("Báo cáo lưu chuyển tiền tệ", "6"))
        got = check(draft(page), found(B01=[[5], [6]], B02=[7], B03=[8]))
        self.assertEqual(got["flags"], [])

    def test_found_again_past_the_next_listed(self):
        page = contents(("Bảng cân đối kế toán", "3"), ("Báo cáo kết quả hoạt động kinh doanh", "5"),
                        ("Báo cáo lưu chuyển tiền tệ", "6"))
        got = check(draft(page), found(B01=[[5], [7]], B02=[7], B03=[8]))
        self.assertEqual(got["flags"], ["B01: listed at pages 5-6, found starting again at page 7"])
        got = check(draft(page), found(B01=[5], B02=[7], B03=[[8], [9]]))     # the last listed: as printed
        self.assertEqual(got["flags"], ["B03: listed at pages 8-8, found starting again at page 9"])

    def test_found_not_read_in_the_contents(self):
        page = contents(("Bảng cân đối kế toán", "3"), ("Báo cáo kết quả hoạt động kinh doanh", "5"))
        got = check(draft(page), found(B01=[5], B02=[7], B03=[8]))
        self.assertEqual(got["flags"], ["B03: found starting at page 8, not read in the contents"])

    def test_a_title_with_no_number_is_not_flagged(self):
        page = contents(("Bảng cân đối kế toán", "3"), ("Báo cáo lưu chuyển tiền tệ", None))
        self.assertEqual(check(draft(page), found(B01=[5], B03=[9]))["flags"], [])

    def test_a_tie_takes_the_smaller_offset(self):
        page = contents(("Bảng cân đối kế toán", "3"))
        got = check(draft(page), found(B01=[[5], [15]]))
        self.assertEqual(got["offset"], 2)


if __name__ == "__main__":
    unittest.main()
