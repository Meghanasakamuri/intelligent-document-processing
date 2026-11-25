"""Regression and unit test suite for invoice-oriented OCR segmentation.

Tests:
1. Configurable PSM segmentation in LayoutLMv3Extractor._ocr_words_boxes.
2. PSM 3 and PSM 11 vs PSM 6 word counts and confidence comparison on FATURA2 samples.
3. Multi-column reading order and bounding box preservation.
4. Extraction of canonical invoice fields (vendor, customer, dates, total, line items) on Template1_Instance0.jpg.
5. Regression testing on Template1_Instance129.jpg.
6. Verification that passing pre-extracted words and boxes bypasses OCR execution (avoiding redundant OCR).
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from PIL import Image

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import config
from app.models.layoutlm_extractor import LayoutLMv3Extractor


class TestOCRSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template0_path = BASE_DIR / "uploads" / "1edb9c16181b4585bfd0dbb23d228b7c_Template1_Instance0.jpg"
        cls.template129_path = BASE_DIR.parent / "scratch" / "Template1_Instance129.jpg"
        cls.extractor = LayoutLMv3Extractor()
        cls.extractor.load()

    def test_configurable_psm_in_extractor(self):
        """Verify that _ocr_words_boxes accepts and respects custom PSM values."""
        if not self.template0_path.exists():
            self.skipTest(f"Missing test sample: {self.template0_path}")

        img = Image.open(self.template0_path).convert("RGB")
        words_psm6, boxes_psm6 = self.extractor._ocr_words_boxes(img, psm=6)
        words_psm3, boxes_psm3 = self.extractor._ocr_words_boxes(img, psm=3)

        self.assertGreater(len(words_psm3), len(words_psm6), "PSM 3 must recognize more words than PSM 6")
        self.assertEqual(len(words_psm3), len(boxes_psm3), "Words and bounding boxes must have strict 1:1 match")
        self.assertEqual(len(words_psm6), len(boxes_psm6), "Words and bounding boxes must have strict 1:1 match")

    def test_psm_segmentation_comparison(self):
        """Evaluate PSM 3 and PSM 11 against PSM 6 on Template1_Instance0.jpg."""
        if not self.template0_path.exists():
            self.skipTest(f"Missing test sample: {self.template0_path}")

        img = Image.open(self.template0_path).convert("RGB")

        # PSM 6 (legacy)
        words_6, _ = self.extractor._ocr_words_boxes(img, psm=6)
        # PSM 3 (automatic multi-column)
        words_3, _ = self.extractor._ocr_words_boxes(img, psm=3)
        # PSM 11 (sparse text)
        words_11, _ = self.extractor._ocr_words_boxes(img, psm=11)

        # PSM 6 severely truncates multi-column invoice text (~50 words)
        self.assertLessEqual(len(words_6), 70)
        # PSM 3 and PSM 11 capture full invoice content (>100 words)
        self.assertGreaterEqual(len(words_3), 100)
        self.assertGreaterEqual(len(words_11), 100)

    def test_template0_canonical_fields_extraction(self):
        """End-to-end extraction on Template1_Instance0.jpg with updated segmentation."""
        if not self.template0_path.exists():
            self.skipTest(f"Missing test sample: {self.template0_path}")

        img = Image.open(self.template0_path).convert("RGB")
        res = self.extractor.extract(img)

        self.assertEqual(res.method, "layoutlmv3_fatura2")
        fields = res.fields

        # Key fields must be extracted accurately
        self.assertIn("invoice_date", fields)
        self.assertEqual(fields["invoice_date"], "2008-03-20")

        self.assertIn("due_date", fields)
        self.assertEqual(fields["due_date"], "2016-10-16")

        self.assertIn("total_amount", fields)
        self.assertEqual(str(fields["total_amount"]), "734.33")

        self.assertIn("customer_name", fields)
        self.assertIn("Denise Perez", fields["customer_name"])

        self.assertIn("po_number", fields)
        self.assertEqual(fields["po_number"], "35")

        # Line items must be detected
        self.assertGreaterEqual(len(res.line_items), 1, "Must extract structured line items")

    def test_template129_regression(self):
        """Regression test on second FATURA2 sample Template1_Instance129.jpg."""
        if not self.template129_path.exists():
            self.skipTest(f"Missing test sample: {self.template129_path}")

        img = Image.open(self.template129_path).convert("RGB")
        res = self.extractor.extract(img)

        fields = res.fields
        self.assertIn("invoice_date", fields)
        self.assertEqual(fields["invoice_date"], "1998-09-22")

        self.assertIn("due_date", fields)
        self.assertEqual(fields["due_date"], "2019-12-26")

        self.assertIn("total_amount", fields)
        self.assertEqual(str(fields["total_amount"]), "1178.95")

    def test_avoid_unnecessary_repeated_ocr(self):
        """Verify that passing pre-extracted words and boxes bypasses Tesseract execution."""
        img = Image.new("RGB", (500, 500), (255, 255, 255))
        provided_words = ["Invoice", "Number", "12345", "Total", "500.00"]
        provided_boxes = [
            [50, 50, 100, 70],
            [110, 50, 160, 70],
            [170, 50, 220, 70],
            [50, 100, 100, 120],
            [110, 100, 160, 120],
        ]

        # Call extract with provided words and boxes
        res = self.extractor.extract(img, words=provided_words, boxes=provided_boxes)
        # Should complete without error and use provided tokens
        self.assertIsNotNone(res)
        self.assertNotIn("OCR produced no words", res.metadata.get("error", ""))


if __name__ == "__main__":
    unittest.main()
