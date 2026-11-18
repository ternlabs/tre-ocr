"""Figures placed in rows, on invented page lines. Run: .venv/bin/python -I -m unittest discover tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from fs_rows import (older_numbering, owners, rows_of, spaced_amounts, split_amounts, split_codes,
                     statements_of, title_of)  # noqa: E402

H = 50                       # a line's height at 300 dpi
COLS = [1700.0, 2100.0]      # the value columns' right edges


def ln(text, x0, x1, y):
    return {"text": text, "box": [x0, y - H // 2, x1, y + H // 2], "sure": True}


def code(c, y):
    return ln(c, 370, 420, y)


def label(t, y):
    return ln(t, 450, 450 + 20 * len(t), y)


def figs(y, a="1.234.567", b="7.654.321"):
    return [ln(a, 1450, 1700, y), ln(b, 1850, 2100, y)]


def placed(lines, codes):
    return {(k, c["text"]) for k, _, c in owners(lines, COLS, codes)}


class Orphans(unittest.TestCase):
    def test_label_wrapped_onto_an_abbreviation(self):
        # 60's label runs onto a second line in capitals; its figures sit beside that line
        codes = [code("52", 1000), code("60", 1100), code("61", 1300)]
        lines = codes + [label("Chi phí thuế hoãn lại", 1000), *figs(1000),
                         label("Lợi nhuận sau thuế", 1100), label("TNDN", 1160), *figs(1160),
                         label("Lợi nhuận của cổ đông", 1300), *figs(1300)]
        self.assertEqual(placed(lines, codes),
                         {(0, "52"), (1, "52"), (0, "60"), (1, "60"), (0, "61"), (1, "61")})

    def test_tall_cell_with_its_code_at_the_bottom(self):
        codes = [code("02", 1000), code("10", 1200), code("11", 1260)]
        lines = codes + [label("Các khoản giảm trừ", 1000), *figs(1000),
                         label("Doanh thu thuần (10", 1120), label("= 01 - 02)", 1160),
                         *figs(1120), label("Giá vốn hàng bán", 1260), *figs(1260)]
        self.assertIn((0, "10"), placed(lines, codes))
        self.assertIn((1, "10"), placed(lines, codes))

    def test_unread_code_does_not_hand_its_figures_to_a_neighbour(self):
        # 10's code was not read and 02's dashes were not either: 02 looks free, but the
        # figures sit beside a label starting a new item, so they go nowhere
        codes = [code("01", 1000), code("02", 1090), code("11", 1270)]
        lines = codes + [label("1. Doanh thu bán hàng", 1000), *figs(1000),
                         label("2. Các khoản giảm trừ", 1090),
                         label("3. Doanh thu thuần", 1170), *figs(1170),
                         label("4. Giá vốn hàng bán", 1270), *figs(1270)]
        self.assertFalse({c for _, c in placed(lines, codes)} & {"02"})

    def test_two_free_codes_get_nothing(self):
        # rows with figures of their own first, so the columns' skew is measured as none
        codes = [code("19", 800), code("20", 900), code("21", 1000), code("22", 1200)]
        lines = codes + [label("Tiền thu khác", 800), *figs(800), label("Tiền chi khác", 900),
                         *figs(900), label("Tiền chi mua sắm", 1000), label("TSCĐ", 1100),
                         *figs(1100), label("Tiền thu thanh lý", 1200)]
        self.assertEqual({c for _, c in placed(lines, codes)}, {"19", "20"})


    def test_continuation_in_capitals_under_the_label_text(self):
        # the page's last code: its label's second line starts in capitals, under the
        # label's text past the item's number, and the figures sit beside that line
        codes = [code("161", 1000), code("162", 1100), code("163", 1200)]
        lines = codes + [label("1. Chi phí trả trước", 1000), *figs(1000),
                         label("2. Thuế được khấu trừ", 1100), *figs(1100),
                         ln("3.", 450, 490, 1200), ln("Thuế và các khoản phải thu", 520, 1000, 1200),
                         ln("Nhà nước", 522, 700, 1250), *figs(1250)]
        self.assertEqual({k for k, c in placed(lines, codes) if c == "163"}, {0, 1})

    def test_code_on_the_label_last_line_figures_on_its_first(self):
        # the item's number and the code stand by the label's last line ("hạn"), the
        # figures by its first
        codes = [code("253", 1000), code("254", 1160), code("255", 1260)]
        lines = codes + [label("3. Đầu tư khác", 1000), *figs(1000),
                         label("Dự phòng đầu tư tài chính dài", 1100), *figs(1100),
                         ln("4.", 450, 490, 1160), ln("hạn", 520, 600, 1160),
                         label("5. Đầu tư nắm giữ", 1260), *figs(1260)]
        self.assertEqual({k for k, c in placed(lines, codes) if c == "254"}, {0, 1})

    def test_two_such_rows_one_under_the_other(self):
        # the second's figures stand between two free codes until the first has its own
        codes = [code("253", 1000), code("254", 1160), code("255", 1320), code("256", 1420)]
        lines = codes + [label("3. Đầu tư khác", 1000), *figs(1000),
                         label("Dự phòng đầu tư tài chính dài", 1100), *figs(1100),
                         ln("4.", 450, 490, 1160), ln("hạn", 520, 600, 1160),
                         label("Dự phòng đầu tư khác dài", 1260), *figs(1260),
                         ln("5.", 450, 490, 1320), ln("hạn", 520, 600, 1320),
                         label("6. Đầu tư nắm giữ", 1420), *figs(1420)]
        got = placed(lines, codes)
        self.assertEqual({k for k, c in got if c == "254"}, {0, 1})
        self.assertEqual({k for k, c in got if c == "255"}, {0, 1})
        by = {(k, c["text"]): f["box"][1] for k, f, c in owners(lines, COLS, codes)}
        self.assertEqual((by[0, "254"], by[0, "255"]), (1100 - H // 2, 1260 - H // 2))

    def test_total_in_capitals_below_a_continuation_gets_nothing(self):
        # figures beside a sub-row's lower-case line do not belong to the total below,
        # whose own figures were not read
        codes = [code("420", 900), code("421", 1000), code("440", 1160)]
        lines = codes + [label("Quỹ khác", 900), *figs(900),
                         label("- Lợi nhuận chưa phân phối", 1000), *figs(1000),
                         label("lũy kế đến cuối kỳ trước", 1090), *figs(1090),
                         label("TỔNG NGUỒN VỐN", 1160)]
        self.assertFalse({c for _, c in placed(lines, codes)} & {"440"})

    def test_short_figure_on_a_column_short_note_beside_the_code(self):
        # earnings per share print in 2-3 digits: on a value column they are figures;
        # the note number between the code and the values is not
        codes = [code("50", 900), code("60", 1000), code("70", 1100)]
        lines = codes + [label("Lợi nhuận trước thuế", 900), *figs(900),
                         label("Lợi nhuận sau thuế", 1000), *figs(1000),
                         label("Lãi cơ bản trên cổ phiếu", 1100), ln("34", 1100, 1140, 1100),
                         ln("275", 1640, 1700, 1100), ln("(48)", 2020, 2100, 1100)]
        self.assertEqual({k for k, c in placed(lines, codes) if c == "70"}, {0, 1})

class Codes(unittest.TestCase):
    def test_a_stop_after_a_code_is_trimmed(self):
        out = split_codes([code("11.", 500)])
        self.assertEqual([(o["text"], o.get("trimmed")) for o in out], [("11", True)])

    def test_a_figure_with_a_stop_is_not_a_code(self):
        out = split_codes([ln("1.234", 1450, 1700, 500)])
        self.assertEqual([o["text"] for o in out], ["1.234"])

    def test_a_code_with_its_label_is_split(self):
        out = split_codes([ln("110 I. Tiền", 370, 600, 500)])
        self.assertEqual([o["text"] for o in out], ["110", "I. Tiền"])


def pair(text, x0, x1, y=500):
    return [o["text"] for o in split_amounts([ln(text, x0, x1, y)], COLS)]


class TwoAmounts(unittest.TestCase):
    """Two amounts read as one line, split when each part ends on a neighbouring column."""
    def test_split_on_two_columns(self):
        # 19 chars over 760 px: 40 px each; the first part (9 chars) ends at 1710
        out = split_amounts([ln("1.234.567 7.654.321", 1350, 2110, 500)], COLS)
        self.assertEqual([o["text"] for o in out], ["1.234.567", "7.654.321"])
        self.assertTrue(all(o["split"] for o in out))
        self.assertEqual(out[0]["box"][2], 1710)
        self.assertEqual(out[1]["box"][2], 2110)

    def test_brackets_kept(self):
        self.assertEqual(pair("(1.234.567) (7.654.321)", 1260, 2110), ["(1.234.567)", "(7.654.321)"])

    def test_left_when_the_first_part_is_off_its_column(self):
        # first part ends at 1550: 150 px short of the column
        self.assertEqual(pair("1.234.567 7.654.321", 1190, 2100), ["1.234.567 7.654.321"])

    def test_left_when_the_second_part_is_off_its_column(self):
        # 60 px a char: the first part ends at 1700, the second 200 px past its column
        self.assertEqual(pair("1.234.567 7.654.321", 1160, 2300), ["1.234.567 7.654.321"])

    def test_left_when_the_first_ends_on_the_last_column(self):
        self.assertEqual(pair("1.234.567 7.654.321", 1740, 2480), ["1.234.567 7.654.321"])

    def test_a_dash_beside_an_amount_is_left(self):
        # "- 7.654.321" may be a minus sign
        self.assertEqual(pair("- 7.654.321", 1680, 2100), ["- 7.654.321"])

    def test_short_numbers_are_left(self):
        self.assertEqual(pair("12 345", 1580, 2100), ["12 345"])

    def test_three_parts_are_left(self):
        self.assertEqual(pair("1.234 5.678 9.012", 1400, 2100), ["1.234 5.678 9.012"])

    def test_one_column_is_left(self):
        for cols in ([1700.0], []):
            out = split_amounts([ln("1.234.567 7.654.321", 1350, 2110, 500)], cols)
            self.assertEqual(len(out), 1)

    def test_a_split_pair_fills_its_row(self):
        lines = [code("01", 400), label("Doanh thu", 400), *figs(400),
                 code("10", 500), label("Doanh thu thuần", 500),
                 ln("1.234.567 7.654.321", 1350, 2110, 500),
                 code("11", 600), label("Giá vốn", 600), *figs(600)]
        rows = rows_of(split_amounts(lines, COLS), COLS)
        self.assertEqual([v and v["text"] for v in rows[1]["values"]], ["1.234.567", "7.654.321"])

    def test_statements_split_the_pair(self):
        p = page(3, ["01", "02", "10", "11", "20"], "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH")
        p["lines"] = [x for x in p["lines"] if not (x["box"][1] < 960 < x["box"][3] and x["box"][0] > 1400)]
        p["lines"].append(ln("2.345.678 8.765.432", 1350, 2110, 960))
        rows = statements_of({"pages": [p]})[0]["rows"]
        self.assertEqual(rows[2]["code"], "10")
        self.assertEqual([v and v["text"] for v in rows[2]["values"]], ["2.345.678", "8.765.432"])


def right_code(c, y):
    return ln(c, 1300, 1350, y)


class CodesRightOfLabels(unittest.TestCase):
    """Codes printed right of their labels; one label read on one line with its code."""
    def lines(self, last):
        out = []
        for i, c in enumerate(["01", "02", "11", "20"]):
            y = 300 + 80 * i
            out += figs(y)
            out += [right_code(c, y), label("Doanh thu", y)] if c != "11" else [last(y)]
        return out

    def test_the_code_is_split_off(self):
        rows = rows_of(self.lines(lambda y: ln("Giá vốn hàng bán 11", 450, 1350, y)), COLS)
        self.assertEqual([r["code"] for r in rows], ["01", "02", "11", "20"])
        self.assertEqual(rows[2]["label"], "Giá vốn hàng bán")
        self.assertIsNotNone(rows[2]["values"][0])

    def test_a_line_ending_short_of_the_codes_keeps_its_number(self):
        rows = rows_of(self.lines(lambda y: ln("Giá vốn hàng bán 11", 450, 1000, y)), COLS)
        self.assertEqual([r["code"] for r in rows], ["01", "02", "20"])


def page(n, codes, title=None):
    """A page of rows, each a code, a label and two figures, under an optional title."""
    lines = [ln(title, 700, 1800, 300)] if title else []
    for i, c in enumerate(codes):
        y = 800 + 80 * i
        lines += [code(c, y), label("Tiền thuần", y)] + figs(y)
    return {"page": n, "lines": lines}


B03 = "BÁO CÁO LƯU CHUYỂN TIỀN TỆ"


class LastTotalsOverleaf(unittest.TestCase):
    """A statement's last totals on a page of their own, with no title."""
    def found(self, overleaf, n=2):
        draft = {"pages": [page(1, ["01", "02", "20", "30", "40"], B03), page(n, overleaf)]}
        return [(s["statement"], s["pages"]) for s in statements_of(draft)]

    def test_codes_carrying_on_join_the_statement(self):
        self.assertEqual(self.found(["50", "60", "70"]), [("B03", [1, 2])])

    def test_codes_starting_over_do_not(self):
        self.assertEqual(self.found(["21", "22", "23"]), [("B03", [1])])

    def test_codes_out_of_order_do_not(self):
        self.assertEqual(self.found(["60", "50", "70"]), [("B03", [1])])

    def test_one_code_does_not(self):
        self.assertEqual(self.found(["50"]), [("B03", [1])])

    def test_a_page_further_on_does_not(self):
        self.assertEqual(self.found(["50", "60", "70"], n=3), [("B03", [1])])

    def test_a_coarse_page_keeps_its_rows_with_no_figure(self):
        # read_pdf.coarse_pages marked page 2: its figures, sure as read, are left empty
        coarse = dict(page(2, ["50", "60", "70"]), coarse="its image is about 70 pixels an inch")
        s, = statements_of({"pages": [page(1, ["01", "02", "20", "30", "40"], B03), coarse]})
        self.assertEqual(s["pages"], [1, 2])
        on = {r["page"]: [v["value"] for v in r["values"]] for r in s["rows"]}
        self.assertEqual(on[1], [1234567, 7654321])
        self.assertEqual(on[2], [None, None])
        self.assertFalse(any(v["sure"] for r in s["rows"] if r["page"] == 2 for v in r["values"]))

    def test_one_stray_number_does_not_stop_them(self):
        # a note's number, read as a code, among the last totals
        self.assertEqual(self.found(["50", "60", "28", "70"]), [("B03", [1, 2])])

    def test_two_stray_numbers_do(self):
        self.assertEqual(self.found(["50", "28", "60", "29"]), [("B03", [1])])

    def test_too_few_figures_a_column_take_the_columns_before(self):
        # two amounts a column and a line of small figures: under MIN_COL of its own
        overleaf = [x for i, c in enumerate(["50", "60"]) for x in [code(c, 800 + 80 * i), label("Lợi nhuận", 800 + 80 * i)]
                    + figs(800 + 80 * i)] + [code("70", 960), label("Lãi cơ bản trên cổ phiếu", 960),
                                             ln("125", 1640, 1700, 960), ln("98", 2055, 2100, 960)]
        draft = {"pages": [page(1, ["01", "02", "20", "30", "40"], B03), {"page": 2, "lines": overleaf}]}
        s = statements_of(draft)[0]
        self.assertEqual(s["pages"], [1, 2])
        self.assertEqual([(r["code"], [v and v["value"] for v in r["values"]]) for r in s["rows"] if r["page"] == 2],
                         [("50", [1234567, 7654321]), ("60", [1234567, 7654321]), ("70", [125, 98])])

    def overleaf_sized(self, size):
        overleaf = [x for i, c in enumerate(["50", "60"]) for x in [code(c, 800 + 80 * i), label("Lợi nhuận", 800 + 80 * i)]
                    + figs(800 + 80 * i)] + [code("70", 960), label("Lãi cơ bản trên cổ phiếu", 960),
                                             ln("125", 1640, 1700, 960), ln("98", 2055, 2100, 960)]
        draft = {"pages": [{**page(1, ["01", "02", "20", "30", "40"], B03), "size": [2481, 3508]},
                           {"page": 2, "lines": overleaf, "size": size}]}
        s = statements_of(draft)[0]
        return [(r["code"], [v and v["value"] for v in r["values"]]) for r in s["rows"] if r["page"] == 2]

    def test_a_page_scanned_a_few_pixels_shorter_goes_on(self):
        # scans of one paper differ by a few pixels: the code column and columns carry over
        self.assertEqual(self.overleaf_sized([2479, 3502]),
                         [("50", [1234567, 7654321]), ("60", [1234567, 7654321]), ("70", [125, 98])])

    def test_a_page_of_another_size_does_not_take_the_code_column(self):
        from fs_rows import same_size
        self.assertTrue(same_size(3508, 3502))
        self.assertFalse(same_size(3508, 3300))         # A4 beside Letter
        self.assertFalse(same_size(3508, 2481))         # a turned page
        self.assertFalse(same_size(None, 3508))

    def test_columns_of_its_own_elsewhere_are_kept(self):
        from fs_rows import carried_columns
        codes = lambda cs: [code(str(c), 800 + 80 * i) for i, c in enumerate(cs)]
        last = [50, 60, 70]
        self.assertEqual(carried_columns([], [1700, 2100], "B03", codes(last)), [1700, 2100])
        self.assertEqual(carried_columns([2110], [1700, 2100], "B03", codes(last)), [1700, 2100])
        self.assertEqual(carried_columns([2500], [1700, 2100], "B03", codes(last)), [2500])            # another place
        self.assertEqual(carried_columns([1300, 1700, 2100], [1700, 2100], "B03", codes(last)), [1300, 1700, 2100])
        self.assertEqual(carried_columns([], [1700, 2100], "B01", codes([430, 431, 440])), [1700, 2100])

    def test_an_annex_of_numbered_branches_keeps_its_own(self):
        from fs_rows import carried_columns
        codes = lambda cs: [code(str(c), 800 + 80 * i) for i, c in enumerate(cs)]
        self.assertEqual(carried_columns([], [1700, 2100], "B03", codes([100, 100, 100, 87, 30])), [])
        self.assertEqual(carried_columns([], [1700, 2100], "B01", codes([262, 135, 52, 247, 18, 26])), [])
        self.assertEqual(carried_columns([], [1700, 2100], "B03", codes([])), [])

    def test_codes_not_after_the_last_read_keep_their_own(self):
        from fs_rows import carried_columns
        codes = lambda cs: [code(str(c), 800 + 80 * i) for i, c in enumerate(cs)]
        self.assertEqual(carried_columns([], [1700, 2100], "B03", codes([50, 60, 70]), 40), [1700, 2100])
        self.assertEqual(carried_columns([], [1700, 2100], "B03", codes([11, 12, 13, 14]), 70), [])
        self.assertEqual(carried_columns([], [1700, 2100], "B02", codes([1, 2, 3, 4]), 60), [])
        self.assertEqual(carried_columns([], [1700, 2100], "B01", codes([430, 431, 440]), 421), [1700, 2100])
        self.assertEqual(carried_columns([], [1700, 2100], "B01", codes([110, 120, 130]), 440), [])

    def test_one_stray_number_among_the_last_codes(self):
        # a note's number beside a code, or the page's number at the foot
        from fs_rows import carried_columns
        codes = lambda cs: [code(str(c), 800 + 80 * i) for i, c in enumerate(cs)]
        self.assertEqual(carried_columns([], [1700, 2100], "B02", codes([60, 70, 28, 71]), 52), [1700, 2100])
        self.assertEqual(carried_columns([], [1700, 2100], "B03", codes([60, 70, 10]), 50), [1700, 2100])
        self.assertEqual(carried_columns([], [1700, 2100], "B03", codes([70, 10]), 50), [])
        self.assertEqual(carried_columns([], [1700, 2100], "B03", codes([60, 70, 10, 12]), 50), [])

    def test_an_annex_numbered_in_b03s_codes_is_not_carried(self):
        # an untitled annex after B03, its items numbered 11 to 16 (B03's codes), two with a
        # figure in B03's second column: too few for a column of its own (MIN_COL)
        lines = [x for i, c in enumerate(["11", "12", "13", "14", "15", "16"])
                 for x in (code(c, 800 + 80 * i), label("Chi nhánh", 800 + 80 * i))]
        lines += [ln("1.234.567", 1850, 2100, 800), ln("7.654.321", 1850, 2100, 880)]
        draft = {"pages": [page(1, ["01", "02", "20", "30", "40", "50", "60", "70"], B03),
                           {"page": 2, "lines": lines}]}
        s = statements_of(draft)[0]
        self.assertEqual([(r["code"], [v and v["value"] for v in r["values"]]) for r in s["rows"] if r["page"] == 2],
                         [])

    def test_small_figures_in_the_columns_are_not_codes(self):
        from fs_rows import carried_columns
        lines = [code("60", 800), code("70", 880), ln("152", 1640, 1700, 880), ln("294", 2040, 2100, 880)]
        self.assertEqual(carried_columns([], [1700, 2100], "B02", lines), [1700, 2100])


def spaced_page(rows, title=None):
    """A page whose figures are grouped by spaces: rows of (code, first figure, second)."""
    lines = [ln(title, 700, 1800, 300)] if title else []
    for i, (c, a, b) in enumerate(rows):
        y = 800 + 80 * i
        lines += [code(c, y), label("Tiền thuần", y),
                  ln(a, 1700 - 28 * len(a), 1700, y), ln(b, 2100 - 28 * len(b), 2100, y)]
    return lines


SPACED = [("110", "1 234 567", "7 654 321"), ("120", "(2 345 678)", "8 765 432"),
          ("130", "3 456 789", "- 9 876 543"), ("140", "45 678", "123 456 789")]


class SpacedAmounts(unittest.TestCase):
    """Figures grouped by spaces, told from codes and from two figures read as one."""
    def texts(self, lines):
        return [x["text"] for x in spaced_amounts(lines) if x.get("spaced")]

    def test_figures_in_columns_right_of_the_codes(self):
        self.assertEqual(self.texts(spaced_page(SPACED)),
                         ["1.234.567", "7.654.321", "(2.345.678)", "8.765.432",
                          "3.456.789", "- 9.876.543", "45.678", "123.456.789"])

    def test_a_statement_of_them_is_read(self):
        draft = {"pages": [{"page": 1, "lines": spaced_page(SPACED, "BẢNG CÂN ĐỐI KẾ TOÁN")}]}
        (s,) = statements_of(draft)
        self.assertEqual([[v and v["value"] for v in r["values"]] for r in s["rows"]],
                         [[1234567, 7654321], [-2345678, 8765432], [3456789, -9876543],
                          [45678, 123456789]])

    def test_codes_read_as_one_line_are_no_figure(self):
        lines = spaced_page(SPACED) + [ln("150 160 170", 300, 420, 1200)]
        self.assertNotIn("150.160.170", self.texts(lines))

    def test_two_figures_read_as_one_are_no_figure(self):
        lines = spaced_page(SPACED) + [code("150", 1200), ln("1 234 567 890", 1450, 2100, 1200)]
        self.assertNotIn("1.234.567.890", self.texts(lines))

    def test_a_figure_off_the_columns_is_left(self):
        lines = spaced_page(SPACED) + [code("150", 1200), ln("1 234 567", 1000, 1250, 1200)]
        self.assertEqual(len(self.texts(lines)), 8)

    def test_a_page_with_no_codes_is_left_as_read(self):
        lines = [x for x in spaced_page(SPACED) if x["text"] not in {"110", "120", "130", "140"}]
        self.assertEqual(self.texts(lines), [])


class Titles(unittest.TestCase):
    def test_a_note_on_items_off_the_balance_sheet_is_not_one(self):
        # in the 2026 form the notes name them "outside the statement of financial position"
        lines = [ln("21. CÁC KHOẢN MỤC NGOÀI BÁO CÁO TÌNH HÌNH TÀI CHÍNH", 300, 2000, 500)]
        self.assertIsNone(title_of(lines, 3508))

    def test_the_balance_sheet_in_the_2026_form(self):
        lines = [ln("BÁO CÁO TÌNH HÌNH TÀI CHÍNH", 700, 1800, 300)]
        self.assertEqual(title_of(lines, 3508)[0], "B01")

    def test_income_statement_of_production_and_business(self):
        for t in ("BÁO CÁO KẾT QUẢ SẢN XUẤT KINH DOANH", "KẾT QUẢ HOẠT ĐỘNG SẢN XUẤT KINH DOANH"):
            self.assertEqual(title_of([ln(t, 600, 1900, 300)], 3508)[0], "B02", t)

    def test_a_title_read_a_letter_off(self):
        for t, k in (("DN - BẢNG CÂN ĐỐI KIẾ TOÁN", "B01"), ("BÁO CÁO LƯU CHUYỂN TIỀN TIỆ", "B03"),
                     ("BÁO CÁO KIẾT QUẢ HOẠT ĐỘNG KINIH DOANH", "B02")):
            self.assertEqual(title_of([ln(t, 600, 1900, 300)], 3508)[0], k, t)

    def test_too_far_off_is_no_title(self):
        for t in ("BẢNG CÂN ĐỐI KẾ TOOÁNN",         # two letters off in one word
                  "BẢNG CẦM ĐỐI KIẾ TOÁN",          # two words off in five
                  "Lưu chuyển tiền tiệ"):          # sentence case is not tried
            self.assertIsNone(title_of([ln(t, 600, 1900, 300)], 3508), t)

    def test_a_note_read_a_letter_off_is_no_title(self):
        lines = [ln("THUYẾT MINH BÁO CÁO LƯU CHUYỂN TIỀN TIỆ", 300, 2000, 300)]
        self.assertIsNone(title_of(lines, 3508))

    def test_a_page_is_its_images_size(self):
        import tempfile
        from PIL import Image
        from fs_rows import size_of
        images = tempfile.mkdtemp()
        Image.new("L", (2481, 3510), 255).save(os.path.join(images, "x-p1.png"))
        junk = [ln("Doanh thu", 300, 900, 900), ln("|", 2470, 2481, 1500)]     # a scan's edge
        self.assertEqual(size_of({"key": "x-p1"}, junk), (3508, 2480))            # the guess: landscape
        self.assertEqual(size_of({"key": "x-p1"}, junk, images), (2481, 3510))
        self.assertEqual(size_of({"key": "x-p2"}, junk, images), (3508, 2480))    # no image: the guess
        self.assertEqual(size_of({"key": "x-p1", "size": [3510, 2481]}, junk, images), (3510, 2481))

    def test_production_costs_in_a_note_are_no_title(self):
        lines = [ln("8. CHI PHÍ SẢN XUẤT KINH DOANH DỞ DANG DÀI HẠN", 300, 2000, 300)]
        self.assertIsNone(title_of(lines, 3508))

    def test_a_title_over_two_lines(self):
        lines = [ln("Báo cáo kết quả hoạt động", 900, 1600, 400),
                 ln("kinh doanh riêng giữa niên độ", 850, 1650, 455)]
        self.assertEqual(title_of(lines, 3508)[0], "B02")

    def test_two_lines_far_apart_are_not_joined(self):
        lines = [ln("Báo cáo kết quả hoạt động", 900, 1600, 300),
                 ln("kinh doanh riêng giữa niên độ", 850, 1650, 600)]
        self.assertIsNone(title_of(lines, 3508))

    def test_a_centred_title_on_a_landscape_page_read_as_it_lies(self):
        # no size in the draft; its lines run past a portrait page's width
        title = "Báo cáo kết quả hoạt động kinh doanh"
        draft = {"pages": [{"page": 1, "lines": [ln(title, 1200, 2300, 300)] +
                            [x for i, c in enumerate(["01", "02", "10", "11", "20"])
                             for x in (code(c, 800 + 80 * i), label("Doanh thu", 800 + 80 * i),
                                       ln("1.234.567", 2700, 2950, 800 + 80 * i),
                                       ln("7.654.321", 3100, 3350, 800 + 80 * i))]}]}
        self.assertEqual([s["statement"] for s in statements_of(draft)], ["B02"])


def b02(*lines):
    return [{"code": c, "label": t} for c, t in lines]


class OlderNumbering(unittest.TestCase):
    """An income statement numbered as before the 2014 form."""
    def codes(self, rows):
        older_numbering(rows)
        return [(r["code"], r.get("printed")) for r in rows]

    def test_selling_and_administrative_expenses(self):
        rows = b02(("23", "Trong đó: Chi phí lãi vay"), ("24", "8. Chi phí bán hàng"),
                   ("25", "9. Chi phí quản lý doanh nghiệp"), ("30", "10. Lợi nhuận thuần"))
        self.assertEqual(self.codes(rows), [("23", None), ("25", "24"), ("26", "25"), ("30", None)])

    def test_deductions(self):
        rows = b02(("01", "1. Doanh thu"), ("03", "2. Các khoản giảm trừ"), ("10", "3. Doanh thu thuần"))
        self.assertEqual(self.codes(rows), [("01", None), ("02", "03"), ("10", None)])

    def test_the_2014_form_is_left(self):
        rows = b02(("24", "8. Phần lãi hoặc lỗ trong công ty liên kết"), ("25", "9. Chi phí bán hàng"),
                   ("26", "10. Chi phí quản lý doanh nghiệp"))
        self.assertEqual(self.codes(rows), [("24", None), ("25", None), ("26", None)])

    def test_one_of_the_two_alone_is_left(self):
        # an unreadable label on 25: moving 24 alone would put two lines on 25
        rows = b02(("24", "8. Chi phí bán hàng"), ("25", "[?]"))
        self.assertEqual(self.codes(rows), [("24", None), ("25", None)])


if __name__ == "__main__":
    unittest.main()


B01_TITLE, B02_TITLE = "BẢNG CÂN ĐỐI KẾ TOÁN", "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH"
B02_CODES = ["01", "02", "10", "11", "20", "21", "22", "50", "51", "52", "60"]
B03_CODES = ["01", "02", "03", "04", "05", "08", "09", "10", "11", "12"]
SHARED = ["01", "02", "10", "11", "20", "21", "22"]       # printed by both B02 and B03


def topped(pg, text):
    """The page with a line at its top: a form, or the notes' heading."""
    pg["lines"].insert(0, ln(text, 1700, 2300, 150))
    return pg


class Untitled(unittest.TestCase):
    """A page whose title is not read: its form or its codes say which statement it starts."""
    def found(self, *pages):
        return [(s["statement"], s["pages"]) for s in statements_of({"pages": list(pages)})]

    def b01(self):
        return page(1, ["100", "110", "111", "120", "130", "131"], B01_TITLE)

    def test_income_statement_codes_after_a_balance_sheet(self):
        self.assertEqual(self.found(self.b01(), page(2, B02_CODES)), [("B01", [1]), ("B02", [2])])

    def test_cash_flow_codes_after_an_income_statement(self):
        draft = (page(1, B02_CODES, B02_TITLE), page(2, B03_CODES))
        self.assertEqual(self.found(*draft), [("B02", [1]), ("B03", [2])])

    def test_codes_that_name_neither_are_left_out(self):
        self.assertEqual(self.found(self.b01(), page(2, SHARED)), [("B01", [1])])

    def test_the_form_names_them(self):
        self.assertEqual(self.found(self.b01(), topped(page(2, SHARED), "Mẫu số B 03 - DN")),
                         [("B01", [1]), ("B03", [2])])

    def test_a_form_the_codes_contradict_is_left_out(self):
        self.assertEqual(self.found(self.b01(), topped(page(2, B02_CODES), "Mẫu số B 03 - DN")),
                         [("B01", [1])])

    def test_balance_sheet_codes_after_an_income_statement(self):
        draft = (page(1, B02_CODES, B02_TITLE), page(2, ["100", "110", "111", "120", "130"]))
        self.assertEqual(self.found(*draft), [("B02", [1]), ("B01", [2])])

    def test_its_own_statement_goes_on(self):
        draft = (self.b01(), page(2, ["140", "141", "150", "151", "200"]))
        self.assertEqual(self.found(*draft), [("B01", [1, 2])])

    def test_not_in_the_notes(self):
        notes = topped(page(2, B02_CODES), "THUYẾT MINH BÁO CÁO TÀI CHÍNH")
        # no statement starts there (an untitled page of codes goes on as it always did)
        self.assertEqual([k for k, _ in self.found(self.b01(), notes, page(3, B03_CODES))], ["B01"])
