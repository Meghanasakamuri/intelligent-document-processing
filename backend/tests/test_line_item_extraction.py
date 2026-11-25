"""Unit and regression tests for invoice line-item extraction in LayoutLMv3Extractor.

Covers:
- Multi-line wrapped description continuation
- Swapped column order (Unit Price before Quantity)
- Item ID separation from Quantity
- Tax column presence (Quantity, Price, Tax, Total)
- Missing decimal point recovery (OCR dot dropping)
- Arithmetic validation and suspicious flag assignment
- Table boundary detection and footer stopping
"""

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.models.layoutlm_extractor import LayoutLMv3Extractor


class TestLineItemExtraction(unittest.TestCase):
    def setUp(self):
        self.extractor = LayoutLMv3Extractor()

    def test_01_multi_line_wrapped_description(self):
        """Verify that wrapped descriptions without numbers merge into previous row."""
        # Row 1 line 1: desc + numbers
        words = ["Professional", "Consulting", "Services", "2", "150.00", "300.00",
                 # Row 1 continuation:
                 "including", "architecture", "review",
                 # Row 2:
                 "Hardware", "Setup", "1", "50.00", "50.00"]
        boxes = [
            [50, 250, 150, 265], [155, 250, 230, 265], [235, 250, 300, 265],
            [350, 250, 370, 265], [450, 250, 500, 265], [550, 250, 600, 265],
            # Continuation line (dy = 260 - 265 = -5 to top, ymid ~ 275, gap <= 22)
            [50, 270, 120, 282], [125, 270, 210, 282], [215, 270, 260, 282],
            # Row 2 (ymid ~ 305)
            [50, 300, 120, 315], [125, 300, 170, 315],
            [350, 300, 370, 315], [450, 300, 500, 315], [550, 300, 600, 315],
        ]

        items = self.extractor._extract_table_line_items([], valid_words=words, norm_boxes=boxes)
        self.assertEqual(len(items), 2)
        self.assertIn("including architecture review", items[0]["description"])
        self.assertEqual(items[0]["quantity"], 2.0)
        self.assertEqual(items[0]["unit_price"], 150.0)
        self.assertEqual(items[0]["item_total"], 300.0)
        self.assertTrue(items[0]["is_valid"])

        self.assertEqual(items[1]["description"], "Hardware Setup")
        self.assertEqual(items[1]["quantity"], 1.0)
        self.assertEqual(items[1]["unit_price"], 50.0)
        self.assertEqual(items[1]["item_total"], 50.0)
        self.assertTrue(items[1]["is_valid"])

    def test_02_swapped_column_order_price_before_quantity(self):
        """Verify templates where Price comes before Quantity (e.g. Templates 49 and 50)."""
        words = [
            "Description", "Price", "Qty", "Amount",
            "Widget", "Alpha", "45.00", "3", "135.00",
        ]
        boxes = [
            [50, 200, 150, 215], [300, 200, 350, 215], [400, 200, 440, 215], [550, 200, 600, 215],
            [50, 230, 100, 245], [105, 230, 150, 245], [300, 230, 350, 245], [410, 230, 430, 245], [550, 230, 600, 245],
        ]

        items = self.extractor._extract_table_line_items([], valid_words=words, norm_boxes=boxes)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["description"], "Widget Alpha")
        self.assertEqual(items[0]["unit_price"], 45.0)
        self.assertEqual(items[0]["quantity"], 3.0)
        self.assertEqual(items[0]["item_total"], 135.0)
        self.assertTrue(items[0]["is_valid"])

    def test_03_item_id_separation(self):
        """Verify that 5-8 digit item part numbers on the left are not treated as quantity."""
        words = [
            "Part", "No", "Description", "Qty", "Rate", "Total",
            "483920", "Industrial", "Sensor", "4", "25.00", "100.00",
        ]
        boxes = [
            [30, 200, 60, 215], [65, 200, 85, 215], [120, 200, 200, 215], [300, 200, 330, 215], [400, 200, 440, 215], [550, 200, 600, 215],
            [30, 230, 80, 245], [120, 230, 190, 245], [195, 230, 250, 245], [310, 230, 325, 245], [400, 230, 440, 245], [550, 230, 600, 245],
        ]

        items = self.extractor._extract_table_line_items([], valid_words=words, norm_boxes=boxes)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["description"], "Industrial Sensor")
        self.assertEqual(items[0]["quantity"], 4.0)
        self.assertEqual(items[0]["unit_price"], 25.0)
        self.assertEqual(items[0]["item_total"], 100.0)
        self.assertTrue(items[0]["is_valid"])

    def test_04_tax_column_handling(self):
        """Verify 4-column numeric tables with Tax (e.g. Quantity, Price, Tax, Total)."""
        words = [
            "Description", "Quantity", "Price", "Tax", "Total",
            "Server", "Hosting", "5", "1785.00", "0.00", "8925.00",
        ]
        boxes = [
            [50, 200, 150, 215], [250, 200, 310, 215], [380, 200, 420, 215], [480, 200, 510, 215], [570, 200, 610, 215],
            [50, 230, 100, 245], [105, 230, 160, 245], [270, 230, 290, 245], [370, 230, 430, 245], [480, 230, 510, 245], [560, 230, 620, 245],
        ]

        items = self.extractor._extract_table_line_items([], valid_words=words, norm_boxes=boxes)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["description"], "Server Hosting")
        self.assertEqual(items[0]["quantity"], 5.0)
        self.assertEqual(items[0]["unit_price"], 1785.0)
        self.assertEqual(items[0]["tax_amount"], 0.0)
        self.assertEqual(items[0]["item_total"], 8925.0)
        self.assertTrue(items[0]["is_valid"])

    def test_05_missing_decimal_point_reconstruction(self):
        """Verify arithmetic cross-validation restores dropped decimal points (e.g. 300 for 3.00, 3750 for 37.50)."""
        words = [
            "Description", "Qty", "Price", "Total",
            "Storage", "Cable", "300", "1250", "3750",  # OCR read 3.00 as 300, 12.50 as 1250, 37.50 as 3750
        ]
        boxes = [
            [50, 200, 150, 215], [280, 200, 310, 215], [400, 200, 440, 215], [550, 200, 600, 215],
            [50, 230, 110, 245], [115, 230, 160, 245], [280, 230, 310, 245], [400, 230, 440, 245], [550, 230, 600, 245],
        ]

        items = self.extractor._extract_table_line_items([], valid_words=words, norm_boxes=boxes)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["description"], "Storage Cable")
        self.assertEqual(items[0]["quantity"], 3.0)
        self.assertEqual(items[0]["unit_price"], 12.50)
        self.assertEqual(items[0]["item_total"], 37.50)
        self.assertTrue(items[0]["is_valid"])

    def test_06_table_termination_on_footer(self):
        """Verify table parser terminates at summary or banking footer."""
        words = [
            "Description", "Qty", "Price", "Amount",
            "Product", "One", "2", "50.00", "100.00",
            "Subtotal", "100.00",
            "BankName", "Standard", "Chartered",
            "Total", "100.00",
        ]
        boxes = [
            [50, 200, 150, 215], [280, 200, 310, 215], [400, 200, 440, 215], [550, 200, 600, 215],
            [50, 230, 110, 245], [115, 230, 150, 245], [290, 230, 300, 245], [400, 230, 440, 245], [550, 230, 600, 245],
            [400, 270, 460, 285], [550, 270, 600, 285],
            [50, 300, 120, 315], [130, 300, 190, 315], [200, 300, 260, 315],
            [400, 330, 440, 345], [550, 330, 600, 345],
        ]

        items = self.extractor._extract_table_line_items([], valid_words=words, norm_boxes=boxes)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["description"], "Product One")
        # Subtotal, BankName, and Total should not become line items


if __name__ == "__main__":
    unittest.main()
