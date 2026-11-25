"""Unit and integration tests for Objective 03: Hybrid Document Question Answering."""

import unittest
from unittest.mock import MagicMock
from PIL import Image

from app.models.base import QAResult
from app.models.document_qa import DocumentQA, classify_question_intent


class TestHybridDocumentQA(unittest.TestCase):
    def setUp(self):
        self.qa = DocumentQA(local_files_only=True)
        self.sample_document = {
            "id": "test_invoice_doc",
            "filename": "invoice_32.pdf",
            "extracted_fields": {
                "vendor_name": "SIVA SAI ENTERPRISES",
                "customer_name": "PRASANTHI PHARMACY",
                "invoice_number": "GSTSAL/379",
                "invoice_date": "2026-06-02",
                "raw_invoice_date": "02/Jun/2026",
                "total_amount": "17591.00",
                "subtotal": "16753.50",
                "tax_amount": "837.68",
                "cgst_amount": "418.84",
                "sgst_amount": "418.84",
                "vendor_gstin": "37COAPS3244E1Z1",
                "customer_gstin": "37BAPPK1430C1Z0",
                "currency": "INR",
            },
            "invoice_fields": {
                "vendor_name": "SIVA SAI ENTERPRISES",
                "customer_name": "PRASANTHI PHARMACY",
                "invoice_number": "GSTSAL/379",
                "invoice_date": "2026-06-02",
                "total_amount": "17591.00",
            },
            "line_items": [
                {
                    "row_index": 1,
                    "description": "ADVACON 0.5MG",
                    "quantity": 4.0,
                    "unit_price": 660.8,
                    "item_total": 2643.2,
                },
                {
                    "row_index": 2,
                    "description": "LIBRIUM 25MG TABLETS",
                    "quantity": 3.0,
                    "unit_price": 190.33,
                    "item_total": 571.0,
                },
            ],
            "extracted_text": (
                "SIVA SAI ENTERPRISES\n"
                "TAX INVOICE\n"
                "GSTIN NO : 37COAPS3244E1Z1\n"
                "To PRASANTHI PHARMACY\n"
                "INVOICE NO: GSTSAL/379\n"
                "DATE: 02/Jun/2026\n"
                "Total Amount: 17591.00\n"
                "BANK DETAILS: SBI ,AC NO:44026748722 IFSC CODE :SBIN0017766\n"
                "20B : WLF20B2025AP000836 21B : WLF21B2025AP000835\n"
            ),
            "page_count": 1,
        }

    def test_intent_classification(self):
        """Verify question understanding maps various natural language questions to canonical intents."""
        test_cases = [
            ("Who is the vendor?", "vendor_name"),
            ("Who supplied this invoice?", "vendor_name"),
            ("Who is the seller?", "vendor_name"),
            ("What is the vendor name?", "vendor_name"),
            ("Who is the customer?", "customer_name"),
            ("Who was billed?", "customer_name"),
            ("What is the invoice number?", "invoice_number"),
            ("Invoice #", "invoice_number"),
            ("What is the invoice date?", "invoice_date"),
            ("When was this invoice issued?", "invoice_date"),
            ("What is the total invoice amount?", "total_amount"),
            ("How much is the total?", "total_amount"),
            ("What is the subtotal?", "subtotal"),
            ("What are the CGST and SGST amounts?", "cgst_and_sgst"),
            ("What is the vendor GSTIN?", "vendor_gstin"),
            ("What is the customer GSTIN?", "customer_gstin"),
            ("What products are listed in this invoice?", "line_items"),
            ("What is the IFSC code?", "ifsc_code"),
            ("What is the bank account number?", "account_number"),
            ("What are the bank details?", "bank_details"),
            ("What is the drug licence number?", "drug_licence"),
        ]
        for q, expected in test_cases:
            self.assertEqual(classify_question_intent(q), expected, f"Failed for question: {q}")

    def test_structured_field_retrieval(self):
        """Verify structured invoice fields are retrieved with high confidence without invoking DocQA."""
        dummy_img = Image.new("RGB", (100, 100), color="white")
        page_entries = [(1, dummy_img)]

        # Vendor
        res_vendor = self.qa.answer_document(self.sample_document, question="Who is the vendor?", page_entries=page_entries)
        self.assertEqual(res_vendor.answer, "SIVA SAI ENTERPRISES")
        self.assertEqual(res_vendor.source_type, "structured_field")
        self.assertGreaterEqual(res_vendor.confidence, 0.95)

        # Invoice Number
        res_num = self.qa.answer_document(self.sample_document, question="What is the invoice number?", page_entries=page_entries)
        self.assertEqual(res_num.answer, "GSTSAL/379")
        self.assertEqual(res_num.source_type, "structured_field")
        self.assertGreaterEqual(res_num.confidence, 0.95)

        # Invoice Date
        res_date = self.qa.answer_document(self.sample_document, question="What is the invoice date?", page_entries=page_entries)
        self.assertEqual(res_date.answer, "02/Jun/2026")
        self.assertEqual(res_date.source_type, "structured_field")
        self.assertGreaterEqual(res_date.confidence, 0.95)

        # Total Amount
        res_tot = self.qa.answer_document(self.sample_document, question="What is the total invoice amount?", page_entries=page_entries)
        self.assertEqual(res_tot.answer, "₹17591.00")
        self.assertEqual(res_tot.source_type, "structured_field")
        self.assertGreaterEqual(res_tot.confidence, 0.95)

        # Vendor GSTIN
        res_gst = self.qa.answer_document(self.sample_document, question="What is the vendor GSTIN?", page_entries=page_entries)
        self.assertEqual(res_gst.answer, "37COAPS3244E1Z1")
        self.assertEqual(res_gst.source_type, "structured_field")
        self.assertGreaterEqual(res_gst.confidence, 0.95)

    def test_line_item_retrieval(self):
        """Verify table line items and specific product queries are extracted accurately."""
        dummy_img = Image.new("RGB", (100, 100), color="white")
        page_entries = [(1, dummy_img)]

        # All products
        res_items = self.qa.answer_document(self.sample_document, question="What products are listed in this invoice?", page_entries=page_entries)
        self.assertIn("2 line items found", res_items.answer)
        self.assertIn("ADVACON 0.5MG", res_items.answer)
        self.assertEqual(res_items.source_type, "line_items")
        self.assertGreaterEqual(res_items.confidence, 0.95)

        # Specific item quantity
        res_qty = self.qa.answer_document(self.sample_document, question="What is the quantity of LIBRIUM?", page_entries=page_entries)
        self.assertIn("3.0", res_qty.answer)
        self.assertEqual(res_qty.source_type, "line_items")

    def test_contextual_text_extraction(self):
        """Verify bank details and IFSC code are extracted from document text."""
        dummy_img = Image.new("RGB", (100, 100), color="white")
        page_entries = [(1, dummy_img)]

        res_ifsc = self.qa.answer_document(self.sample_document, question="What is the IFSC code?", page_entries=page_entries)
        self.assertEqual(res_ifsc.answer, "SBIN0017766")
        self.assertEqual(res_ifsc.source_type, "document_text")

        res_acc = self.qa.answer_document(self.sample_document, question="What is the bank account number?", page_entries=page_entries)
        self.assertEqual(res_acc.answer, "44026748722")
        self.assertEqual(res_acc.source_type, "document_text")

    def test_unanswerable_guard(self):
        """Verify unanswerable queries return answer=None with 0.0 confidence without fabricating."""
        dummy_img = Image.new("RGB", (100, 100), color="white")
        page_entries = [(1, dummy_img)]

        # Mock answer_pages to return low-confidence result
        orig_answer_pages = self.qa.answer_pages
        try:
            self.qa.answer_pages = MagicMock(return_value=(1, QAResult(question="random", answer=None, confidence=0.0)))
            res = self.qa.answer_document(self.sample_document, question="Who is the astronaut on this invoice?", page_entries=page_entries)
            self.assertIsNone(res.answer)
            self.assertEqual(res.confidence, 0.0)
            self.assertEqual(res.confidence_level, "low")
            self.assertIn("could not be found", res.explanation.lower())
        finally:
            self.qa.answer_pages = orig_answer_pages


if __name__ == "__main__":
    unittest.main()
