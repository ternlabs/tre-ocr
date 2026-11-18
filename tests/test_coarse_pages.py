"""read_pdf.image_ppi and coarse_pages: a read page whose image is coarse is marked so, and
nothing else is. Invented PDFs, blank images placed at a set resolution.
Run: .venv/bin/python -I -m unittest discover tests"""
import os
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from read_pdf import COARSE_PPI, coarse_pages, image_ppi, looked_at  # noqa: E402

A4 = (8.27, 11.69)                          # inches


def pdf(tmp, *ppis):
    """A PDF with one A4 page per resolution given, each a blank image placed at it."""
    pages = [Image.new("RGB", (round(A4[0] * r), round(A4[1] * r)), "white") for r in ppis]
    path = os.path.join(tmp, "x.pdf")
    pages[0].save(path, save_all=True, append_images=pages[1:], resolution=ppis[0])
    return path


class Coarse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_the_image_says_its_resolution(self):
        self.assertEqual(image_ppi(pdf(self.tmp, 70)), {1: 70})

    def test_a_coarse_read_page_is_marked(self):
        pages = [{"page": 1, "kind": "read", "lines": []}]
        self.assertEqual(coarse_pages(pdf(self.tmp, COARSE_PPI - 20), pages), 1)
        self.assertIn("not guessed", pages[0]["coarse"])

    def test_a_layered_scan_is_as_fine_as_its_mask(self):
        listing = ("page num type width height color comp bpc enc interp object ID x-ppi y-ppi size ratio\n"
                   "---\n"
                   "1 0 image 620 877 rgb 3 8 jpeg yes 10 0 75 75 10K 1%\n"
                   "1 1 image 306 433 rgb 3 8 jpeg yes 7 0 37 37 3K 1%\n"
                   "1 2 mask 1241 1754 - 1 1 ccitt yes 7 0 150 150 3K 1%\n"
                   "1 3 image 50 50 rgb 3 8 jpeg no 8 0 600 600 1K 1%\n")
        with mock.patch("label_pages.subprocess.run", return_value=mock.Mock(stdout=listing.encode())):
            self.assertEqual(image_ppi("x.pdf"), {1: 150})     # not the small logo's 600

    def test_a_stencil_counts_and_inline_marks_do_not(self):
        listing = ("page num type width height color comp bpc enc interp object ID x-ppi y-ppi size ratio\n"
                   "---\n"
                   "1 0 image 620 877 rgb 3 8 jpeg yes 10 0 75 75 10K 1%\n"
                   "1 1 stencil 2480 3508 - 1 1 ccitt no 11 0 300 300 9K 1%\n"
                   "2 2 image 8 1 rgb 3 8 image no [inline] 60 60 0B 0.0%\n")
        with mock.patch("label_pages.subprocess.run", return_value=mock.Mock(stdout=listing.encode())):
            self.assertEqual(image_ppi("x.pdf"), {1: 300})     # page 2 has no scan to judge

    def test_a_failed_listing_leaves_the_resolution_unknown(self):
        pages = [{"page": 1, "kind": "read", "lines": []}]
        failed = mock.Mock(returncode=1, stdout=b"", stderr=b"Syntax Error: broken xref")
        with mock.patch("label_pages.subprocess.run", return_value=failed):
            self.assertEqual(image_ppi("x.pdf"), {})
            self.assertEqual(coarse_pages("x.pdf", pages), 0)
        with mock.patch("label_pages.subprocess.run", side_effect=FileNotFoundError("pdfimages")):
            self.assertEqual(image_ppi("x.pdf"), {})
        self.assertNotIn("coarse", pages[0])

    def test_a_fine_page_is_not_marked(self):
        pages = [{"page": 1, "kind": "read", "lines": []}]
        self.assertEqual(coarse_pages(pdf(self.tmp, 200), pages), 0)
        self.assertNotIn("coarse", pages[0])

    def test_the_looks_pass_a_coarse_page_over(self):
        pages = [{"page": 1, "kind": "read", "lines": []}, {"page": 2, "kind": "read", "lines": []},
                 {"page": 3, "kind": "text layer", "lines": []}]
        with mock.patch("read_pdf.image_ppi", return_value={1: 200, 2: COARSE_PPI - 20, 3: 70}):
            coarse_pages("x.pdf", pages)
        self.assertEqual(list(looked_at(pages)), [1])

    def test_a_page_not_read_is_not_marked(self):
        pages = [{"page": 1, "kind": "text layer", "lines": []}]
        self.assertEqual(coarse_pages(pdf(self.tmp, 60), pages), 0)


if __name__ == "__main__":
    unittest.main()
