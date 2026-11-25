"""Unit and regression tests for LayoutLMv3 entity extraction, field mapping,
and prediction accuracy improvements (Task 2).
"""

import sys
import unittest
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.models.layoutlm_extractor import LayoutLMv3Extractor


class TestEntityExtraction(unittest.TestCase):
    """Verify entity grouping, field disambiguation, financial mapping, and regression samples."""

    @classmethod
    def setUpClass(cls):
        cls.extractor = LayoutLMv3Extractor()
        cls.extractor.load()

    def test_bio_grouping_logic(self):
        """Ensure B- labels start new entities and I- labels properly extend."""
        valid_words = ["Invoice", "Number", "INV-101", "PO", "Number", "PO-99"]
        norm_boxes = [
            [10, 10, 50, 20],
            [55, 10, 100, 20],
            [105, 10, 160, 20],
            [10, 30, 50, 40],
            [55, 30, 100, 40],
            [105, 30, 160, 40],
        ]
        # label IDs
        id2label = {
            0: "O",
            7: "B-NUMBER",
            8: "I-NUMBER",
            13: "B-PO_NUMBER",
            14: "I-PO_NUMBER",
        }
        word_preds = {0: 7, 1: 8, 2: 8, 3: 13, 4: 14, 5: 14}
        word_confs = {i: 0.95 for i in range(6)}

        res = self.extractor._extract_invoice(
            valid_words=valid_words,
            word_preds=word_preds,
            word_confs=word_confs,
            norm_boxes=norm_boxes,
            id2label=id2label,
            ocr_text=" ".join(valid_words),
            document_type="Invoice",
            start=0.0,
        )

        self.assertEqual(res.fields.get("invoice_number"), "101")
        self.assertEqual(res.fields.get("po_number"), "99")
        self.assertIn("field_sources", res.metadata)
        self.assertIn("field_bboxes", res.metadata)

    def test_currency_priority_on_total_line(self):
        """Ensure currency code on TOTAL line overrides line item $ symbol."""
        valid_words = ["TOTAL", ":", "734.33", "EUR", "Item", "$10.00"]
        norm_boxes = [
            [388, 739, 427, 748],
            [428, 739, 433, 748],
            [439, 739, 474, 748],
            [479, 739, 503, 748],
            [100, 500, 150, 515],
            [200, 500, 250, 515],
        ]
        id2label = {0: "O", 35: "B-TOTAL", 36: "I-TOTAL"}
        word_preds = {0: 35, 1: 36, 2: 36, 3: 36, 4: 0, 5: 0}
        word_confs = {i: 0.99 for i in range(6)}

        res = self.extractor._extract_invoice(
            valid_words=valid_words,
            word_preds=word_preds,
            word_confs=word_confs,
            norm_boxes=norm_boxes,
            id2label=id2label,
            ocr_text=" ".join(valid_words),
            document_type="Invoice",
            start=0.0,
        )

        self.assertEqual(res.fields.get("currency"), "EUR")
        self.assertEqual(res.fields.get("total_amount"), "734.33")

    def test_regression_template1_instance0(self):
        """Test complete extraction on Template1_Instance0 ground truth sample."""
        img_path = Path("scratch/fatura_Template1_Instance0.jpg")
        if not img_path.exists():
            img_path = Path("backend/uploads/1edb9c16181b4585bfd0dbb23d228b7c_Template1_Instance0.jpg")
        self.assertTrue(img_path.exists(), f"Sample image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        res = self.extractor.extract(img, psm=3)

        self.assertEqual(res.fields.get("vendor_name"), "Thompson and Sons")
        self.assertEqual(res.fields.get("customer_name"), "Denise Perez")
        self.assertEqual(res.fields.get("po_number"), "35")
        self.assertEqual(res.fields.get("invoice_date"), "2008-03-20")
        self.assertEqual(res.fields.get("due_date"), "2016-10-16")
        self.assertEqual(res.fields.get("currency"), "EUR")
        self.assertEqual(res.fields.get("total_amount"), "734.33")
        self.assertEqual(res.fields.get("tax_amount"), "28.18")
        self.assertEqual(res.fields.get("discount"), "13.42")
        self.assertEqual(res.fields.get("subtotal"), "719.57")
        self.assertEqual(len(res.line_items), 5)
        self.assertIn("field_sources", res.metadata)

    def test_regression_template1_instance129(self):
        """Test complete extraction on Template1_Instance129 ground truth sample."""
        img_path = Path("scratch/Template1_Instance129.jpg")
        self.assertTrue(img_path.exists(), f"Sample image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        res = self.extractor.extract(img, psm=3)

        self.assertEqual(res.fields.get("vendor_name"), "Thompson and Sons")
        self.assertEqual(res.fields.get("customer_name"), "John Marks")
        self.assertEqual(res.fields.get("po_number"), "04")
        self.assertEqual(res.fields.get("invoice_date"), "1998-09-22")
        self.assertEqual(res.fields.get("due_date"), "2019-12-26")
        self.assertEqual(res.fields.get("currency"), "USD")
        self.assertEqual(res.fields.get("total_amount"), "1178.95")
        self.assertEqual(res.fields.get("subtotal"), "1181.96")
        self.assertEqual(res.fields.get("discount"), "44.8")
        self.assertEqual(res.fields.get("tax_amount"), "52.07")
        self.assertEqual(len(res.line_items), 5)
        self.assertIn("field_bboxes", res.metadata)

    def test_regression_unseen_template45_instance103(self):
        """Test extraction on held-out test Template45 (single-block layout with customer and total)."""
        img_path = Path("evaluation/fatura2_test50/images/Template45_Instance103.jpg")
        if not img_path.exists():
            self.skipTest("evaluation/fatura2_test50 dataset not found")

        img = Image.open(img_path).convert("RGB")
        res = self.extractor.extract(img)

        self.assertEqual(res.fields.get("customer_name"), "Barbara Phelps")
        self.assertEqual(res.fields.get("invoice_number"), "INV/35-61/437")
        self.assertEqual(res.fields.get("total_amount"), "300.89")
        self.assertEqual(res.fields.get("vendor_name"), "Patton, Ortiz and Moore")
        self.assertEqual(res.fields.get("currency"), "USD")

    def test_regression_unseen_template50_instance123(self):
        """Test extraction on held-out test Template50 (multi-column summary with tax, discount, subtotal)."""
        img_path = Path("evaluation/fatura2_test50/images/Template50_Instance123.jpg")
        if not img_path.exists():
            self.skipTest("evaluation/fatura2_test50 dataset not found")

        img = Image.open(img_path).convert("RGB")
        res = self.extractor.extract(img)

        self.assertEqual(res.fields.get("customer_name"), "Zachary Moody")
        self.assertEqual(res.fields.get("total_amount"), "370.50")
        self.assertEqual(res.fields.get("subtotal"), "361.24")
        self.assertEqual(res.fields.get("tax_amount"), "17.64")
        self.assertEqual(res.fields.get("currency"), "USD")


if __name__ == "__main__":
    unittest.main()
