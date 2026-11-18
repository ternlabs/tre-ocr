"""fs_turn: which reading of a sideways page is kept. Invented pages, a fake reader.
Run: .venv/bin/python -I -m unittest discover tests"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from fs_turn import sure_lines, turn  # noqa: E402

WIDE, TALL = [100, 100, 900, 140], [100, 100, 140, 900]


def line(box, text="Doanh thu bán hàng", sure=True):
    return {"text": text, "sure": sure, "box": box}


class FakeReader:
    """Reads a turn as `got` says: {angle: (wide sure lines)}."""
    def __init__(self, got):
        self.got = got

    def page(self, png):
        angle = int(png.rsplit("-r", 1)[1].split(".")[0])
        n = self.got[angle]
        return {"text": "\n".join(["Doanh thu bán hàng"] * n), "boxes": [WIDE] * n, "unsure": []}


class Vote(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.dir = tempfile.mkdtemp()
        Image.new("L", (40, 60), 255).save(os.path.join(self.dir, "x-p1.png"))

    def test_lines_on_their_side_are_not_counted(self):
        self.assertEqual(sure_lines([line(WIDE), line(TALL), line(WIDE, sure=False), line(WIDE, "ab")]), 1)

    def test_junk_read_surely_on_its_side_does_not_keep_the_page(self):
        page = {"page": 1, "key": "x-p1", "lines": [line(TALL)] * 80}
        out = turn(page, self.dir, FakeReader({90: 30, 270: 5}))
        self.assertEqual((out["turned"], len(out["lines"])), (90, 30))

    def test_a_page_reading_better_as_it_lies_is_kept(self):
        page = {"page": 1, "key": "x-p1", "lines": [line(WIDE)] * 40 + [line(TALL)] * 60}
        self.assertNotIn("turned", turn(page, self.dir, FakeReader({90: 30, 270: 5})))


if __name__ == "__main__":
    unittest.main()
