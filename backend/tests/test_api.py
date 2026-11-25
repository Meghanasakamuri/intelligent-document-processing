"""Comprehensive automated test suite for Intelligent Document Processing (IDP).

Covers:
- API endpoints (health, root, stats, evaluation, models)
- Document CRUD lifecycle & SQLite persistence
- Human-in-the-loop verification & persistent audit trail
- LayoutLMv3 extractor interface & semantic field mapping
- Document QA interface & unanswerable handling
- Confidence classification & temperature scaling calibration
- Heuristic baseline extraction (Invoices, Receipts, Forms)
- Upload format and size validation

Runnable with:
    python -m unittest tests/test_api.py
or (if pytest is installed):
    pytest tests/ -v
"""

from __future__ import annotations

import io
import os
import shutil
import unittest
from pathlib import Path

# Ensure backend root is on sys.path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Safe compatibility fallback for Windows environments where Application Control (Smart App Control)
# blocks unsigned compiled C-extensions (_regex.cp313-win_amd64.pyd)
try:
    import regex  # noqa: F401
except ImportError:
    import re
    sys.modules["regex"] = re

# Use an isolated test database and upload directory
TEST_DIR = Path(__file__).resolve().parent
TEST_DB = TEST_DIR / "test_documents.db"
TEST_UPLOADS = TEST_DIR / "test_uploads"
os.environ["IDP_DATABASE_PATH"] = str(TEST_DB)
os.environ["IDP_UPLOAD_DIR"] = str(TEST_UPLOADS)

from fastapi import HTTPException, UploadFile
from app import database, main, processing
from app.models.base import ExtractionResult, QAResult
from app.models.confidence import (
    TemperatureCalibrator,
    brier_score,
    classify_confidence,
    expected_calibration_error,
)
from app.models.document_qa import DocumentQA
from app.models.layoutlm_extractor import LayoutLMv3Extractor
from app.schemas import FieldUpdate, GroundTruthRequest, QARequest, VerifyRequest


class TestIDPSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_UPLOADS.mkdir(parents=True, exist_ok=True)
        database.init_db()

    @classmethod
    def tearDownClass(cls):
        if TEST_DB.exists():
            try:
                TEST_DB.unlink()
            except Exception:
                pass
        if TEST_UPLOADS.exists():
            try:
                shutil.rmtree(TEST_UPLOADS, ignore_errors=True)
            except Exception:
                pass

    # ── 1. Core Endpoints & Health ───────────────────────────────────

    def test_01_health_check(self):
        result = main.health()
        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["service"], "intelligent-document-processing-api")

    def test_02_home_root(self):
        result = main.home()
        self.assertIn("message", result)
        self.assertIn("objectives", result)
        self.assertIn("O2", result["objectives"])
        self.assertIn("O3", result["objectives"])

    def test_03_models_status(self):
        status = main.get_models_status()
        self.assertIsNotNone(status.layoutlm_model)
        self.assertIsNotNone(status.qa_model)
        self.assertIsInstance(status.layoutlm_available, bool)
        self.assertIsInstance(status.qa_available, bool)
        self.assertEqual(status.device, "cpu")

    def test_04_models_load(self):
        result = main.load_models("layoutlm")
        self.assertEqual(result["requested"], "layoutlm")
        self.assertIn("status", result)

    # ── 2. Baseline Extraction & Classification ──────────────────────

    def test_05_document_type_detection(self):
        invoice_text = "TAX INVOICE\nGSTIN: 27AAAAA0000A1Z5\nGRAND TOTAL: $500.00"
        self.assertEqual(processing.detect_document_type(invoice_text), "Invoice")

        receipt_text = "CASH RECEIPT\nAMOUNT RECEIVED: $25.00\nPAID BY CASH"
        self.assertEqual(processing.detect_document_type(receipt_text), "Receipt")

        form_text = "REGISTRATION FORM\nAPPLICATION SIGNATURE\nDATE OF BIRTH: 01/01/1990"
        self.assertEqual(processing.detect_document_type(form_text), "Form")

        unknown_text = "Random plain unstructured note"
        self.assertEqual(processing.detect_document_type(unknown_text), "Unknown")

    def test_06_invoice_field_extraction(self):
        text = (
            "ACME SUPPLIES INC\n"
            "TAX INVOICE\n"
            "INVOICE NO: INV-2026-999\n"
            "INVOICE DATE: 15/11/2026\n"
            "GSTIN: 29ABCDE1234F1Z5\n"
            "SUBTOTAL: 1,000.00\n"
            "TOTAL TAX: 180.00\n"
            "GRAND TOTAL: $1,180.00\n"
        )
        invoice_fields, structured = processing.extract_invoice_fields(text)
        self.assertIn("INV-2026-999", str(invoice_fields.get("invoice_number")))
        self.assertIn("15/11/2026", str(invoice_fields.get("invoice_date")))
        self.assertIn("1,180.00", str(invoice_fields.get("total_amount")))
        self.assertIn("29ABCDE1234F1Z5", str(invoice_fields.get("gst_number")))
        self.assertGreater(invoice_fields.get("confidence_score", 0), 50.0)

    def test_07_receipt_field_extraction(self):
        text = (
            "COFFEE SHOP\n"
            "RECEIPT # 4022\n"
            "DATE: 12/10/2026\n"
            "TAX AMOUNT: 2.50\n"
            "TOTAL AMOUNT: 28.50\n"
        )
        invoice_fields, structured = processing.extract_receipt_fields(text)
        self.assertEqual(structured.get("receipt_number"), "4022")
        self.assertEqual(structured.get("total_amount"), "28.50")
        self.assertGreater(invoice_fields.get("confidence_score", 0), 50.0)

    def test_08_form_field_extraction(self):
        text = (
            "APPLICATION FORM\n"
            "Full Name: Jane Doe\n"
            "Email: jane@example.com\n"
            "Phone: 555-0199\n"
        )
        invoice_fields, structured = processing.extract_form_fields(text)
        self.assertEqual(structured.get("form_type"), "APPLICATION FORM")
        fields = structured.get("fields", [])
        self.assertEqual(len(fields), 3)

    # ── 3. Confidence & Calibration Module ───────────────────────────

    def test_09_confidence_classification(self):
        self.assertEqual(classify_confidence(0.95), "high")
        self.assertEqual(classify_confidence(0.80), "high")
        self.assertEqual(classify_confidence(0.65), "medium")
        self.assertEqual(classify_confidence(0.50), "medium")
        self.assertEqual(classify_confidence(0.49), "low")
        self.assertEqual(classify_confidence(0.10), "low")

    def test_10_temperature_calibration(self):
        import numpy as np
        calibrator = TemperatureCalibrator()
        self.assertFalse(calibrator.is_fitted)

        # Before calibration, scores are unscaled identity
        self.assertAlmostEqual(calibrator.calibrate(0.75), 0.75)

        # Fit with synthetic overconfident logits and binary labels
        logits = np.array([[3.0, 1.0], [2.5, 1.5], [1.0, 2.0], [0.5, 3.0]])
        labels = np.array([0, 0, 1, 1])
        learned_t = calibrator.fit(logits, labels, lr=0.05, max_iter=50)
        self.assertTrue(calibrator.is_fitted)
        self.assertGreater(learned_t, 0.0)

        # Test persistence
        cal_path = TEST_DIR / "temp_calibration.json"
        calibrator.save(cal_path)
        self.assertTrue(cal_path.is_file())

        loaded_calibrator = TemperatureCalibrator()
        loaded_calibrator.load(cal_path)
        self.assertTrue(loaded_calibrator.is_fitted)
        self.assertAlmostEqual(loaded_calibrator.temperature, calibrator.temperature)
        cal_path.unlink()

    def test_11_calibration_metrics(self):
        import numpy as np
        confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.2])
        outcomes = np.array([1, 1, 1, 0, 0, 0])
        ece_res = expected_calibration_error(confidences, outcomes, n_bins=5)
        self.assertIn("ece", ece_res)
        self.assertIn("bins", ece_res)
        self.assertGreaterEqual(ece_res["ece"], 0.0)

        b_score = brier_score(confidences, outcomes)
        self.assertGreaterEqual(b_score, 0.0)

    # ── 4. Document CRUD & Human Verification Audit Trail ────────────

    def test_12_document_lifecycle_and_audit_trail(self):
        import uuid
        doc_id = f"test_doc_{uuid.uuid4().hex[:8]}"
        test_file = TEST_UPLOADS / f"{doc_id}_sample.txt"
        test_file.write_text("DUMMY DOCUMENT CONTENT")

        # 1. Insert
        record = {
            "id": doc_id,
            "filename": "sample_invoice.png",
            "stored_filename": f"{doc_id}_sample.txt",
            "file_path": str(test_file),
            "content_type": "image/png",
            "document_type": "Invoice",
            "uploaded_at": main._now(),
            "processing_status": "processed",
            "extracted_fields": {"vendor_name": "Acme", "total_amount": "100.00"},
            "invoice_fields": {
                "company_name": "Acme",
                "invoice_number": "INV-001",
                "total_amount": "100.00",
                "confidence_score": 75.0,
            },
            "extracted_text": "ACME SAMPLE INVOICE",
            "confidence_score": 75.0,
            "confidence_method": "heuristic_field_coverage",
            "verified": False,
            "verified_at": None,
            "processing_error": None,
            "ground_truth": None,
            "page_count": 1,
            "pages_processed": 1,
            "text_source": "tesseract",
            "message": "Processed successfully",
        }
        database.insert_document(record)

        # 2. Get Document by ID
        doc = main.get_document_by_id(doc_id)
        self.assertEqual(doc["id"], doc_id)
        self.assertEqual(doc["filename"], "sample_invoice.png")
        self.assertFalse(doc["verified"])

        # 3. List Documents
        doc_list = main.get_documents(q="Acme", limit=10)
        self.assertGreaterEqual(doc_list["total"], 1)

        # 4. Human Correction: Update fields (should write to audit_log!)
        update_req = FieldUpdate(
            invoice_fields={"invoice_number": "INV-001-CORRECTED", "total_amount": "120.00"}
        )
        updated_doc = main.update_document_fields(doc_id, update_req)
        self.assertEqual(updated_doc["invoice_fields"]["invoice_number"], "INV-001-CORRECTED")

        # Verify audit trail recorded changes
        audit_trail = main.get_document_audit_log(doc_id)
        self.assertGreaterEqual(len(audit_trail), 2)
        field_names = [entry["field_name"] for entry in audit_trail]
        self.assertIn("invoice_number", field_names)
        self.assertIn("total_amount", field_names)

        # 5. Human Verification
        verify_req = VerifyRequest(verified=True)
        verified_doc = main.verify_document(doc_id, verify_req)
        self.assertTrue(verified_doc["verified"])
        self.assertIsNotNone(verified_doc["verified_at"])

        # Verify audit trail recorded approval
        audit_trail = main.get_document_audit_log(doc_id)
        actions = [entry["action"] for entry in audit_trail]
        self.assertIn("approve", actions)

        # 6. Save Ground Truth
        gt_req = GroundTruthRequest(
            fields={"invoice_number": "INV-001-CORRECTED", "total_amount": "120.00"}
        )
        main.save_ground_truth(doc_id, gt_req)
        doc_after_gt = main.get_document_by_id(doc_id)
        self.assertIsNotNone(doc_after_gt["ground_truth"])

        # 7. Statistics & Evaluation
        stats = main.get_stats()
        self.assertGreaterEqual(stats["total_documents"], 1)
        self.assertGreaterEqual(stats["verified"], 1)

        eval_res = main.get_evaluation()
        self.assertTrue(eval_res["metrics_available"])
        self.assertEqual(eval_res["accuracy"], 1.0)
        self.assertEqual(eval_res["exact_matches"], 2)

        # 8. File Endpoint (Verify inline disposition to avoid forced downloads in review iframe)
        file_resp = main.download_document(doc_id)
        self.assertEqual(
            file_resp.headers.get("content-disposition"),
            'inline; filename="sample_invoice.png"',
        )

        # 9. Delete Document
        del_res = main.remove_document(doc_id)
        self.assertEqual(del_res["message"], "Document deleted")

        # 10. Verify 404 after deletion
        with self.assertRaises(HTTPException):
            main.get_document_by_id(doc_id)

    # ── 5. O2 & O3 Model Interfaces ──────────────────────────────────

    def test_13_layoutlmv3_extractor_interface(self):
        extractor = LayoutLMv3Extractor()
        self.assertIsNotNone(extractor.model_name)
        # Verify gracefully reports availability
        avail = extractor.is_available()
        self.assertIsInstance(avail, bool)

    def test_14_document_qa_interface(self):
        qa = DocumentQA()
        self.assertIsNotNone(qa.model_name)
        avail = qa.is_available()
        self.assertIsInstance(avail, bool)

    # ── 6. Semantic Field Mapping ────────────────────────────────────

    def test_15_semantic_mapping_layer(self):
        pairs = [
            ("Invoice #:", "INV-8899", 0.95),
            ("Invoice Date:", "01/12/2026", 0.92),
            ("Grand Total:", "$450.00", 0.88),
            ("GSTIN:", "27AABCT3518Q1ZV", 0.96),
            ("Department Code:", "D-42", 0.70),  # generic form pair
        ]
        headers = ["ACME LOGISTICS INC"]
        raw_confs = {p[0]: p[2] for p in pairs}

        canonical, confs, unmapped = LayoutLMv3Extractor._map_to_canonical_fields(
            pairs=pairs,
            headers=headers,
            document_type="Invoice",
            raw_confs=raw_confs,
        )

        self.assertEqual(canonical.get("invoice_number"), "INV-8899")
        self.assertEqual(canonical.get("invoice_date"), "01/12/2026")
        self.assertEqual(canonical.get("total_amount"), "450.00")
        self.assertEqual(canonical.get("gst_number"), "27AABCT3518Q1ZV")
        self.assertEqual(canonical.get("vendor_name"), "ACME LOGISTICS INC")
        # Arbitrary form words must NOT be injected into canonical fields
        self.assertNotIn("Department Code", canonical)
        self.assertEqual(unmapped.get("Department Code"), "D-42")

    # ── 7. LayoutLMv3 Error Handling & Extraction Tests ──────────────

    def test_16_layoutlmv3_empty_ocr(self):
        from PIL import Image
        img = Image.new("RGB", (200, 200), color="white")
        extractor = main.get_layoutlm()
        res = extractor.extract(img, words=[], boxes=[])
        self.assertEqual(res.fields, {})
        self.assertEqual(res.confidence_scores, {})
        self.assertIn("error", res.metadata)

    def test_17_layoutlmv3_malformed_boxes(self):
        from PIL import Image
        img = Image.new("RGB", (200, 200), color="white")
        extractor = main.get_layoutlm()
        # Words and boxes containing mismatched lengths, invalid dimensions, non-integers
        res = extractor.extract(
            img,
            words=["INVOICE", "INV-100", "DATE", "BAD1", "BAD2"],
            boxes=[
                [10, 10, 50, 30],
                [60, 10, 120, 30],
                [10, 20],               # invalid length (2 instead of 4)
                "not_a_box",            # invalid type
                [10, 10, "bad", 40],    # non-integer
            ],
            document_type="Invoice",
        )
        self.assertIsInstance(res.fields, dict)
        self.assertIsInstance(res.confidence_scores, dict)
        self.assertNotIn("ValueError: expected sequence of length 4", str(res.metadata))

    def test_18_layoutlmv3_normal_invoice_input(self):
        from PIL import Image
        img = Image.new("RGB", (400, 400), color="white")
        extractor = main.get_layoutlm()
        res = extractor.extract(
            img,
            words=["INVOICE", "INV-2024-001", "DATE", "2024-11-15", "TOTAL", "450.00"],
            boxes=[
                [10, 10, 50, 30],
                [60, 10, 120, 30],
                [10, 40, 50, 60],
                [60, 40, 120, 60],
                [10, 70, 50, 90],
                [60, 70, 120, 90],
            ],
            document_type="Invoice",
        )
        self.assertIn(res.method, ("layoutlmv3_fatura2", "layoutlmv3_funsd"))
        self.assertIsInstance(res.fields, dict)
        self.assertIsInstance(res.confidence_scores, dict)
        self.assertGreaterEqual(len(res.fields), 1)

    def test_19_layoutlm_api_endpoint(self):
        import uuid
        from PIL import Image
        doc_id = f"test_layoutlm_{uuid.uuid4().hex[:8]}"
        img_path = TEST_UPLOADS / f"{doc_id}.png"
        img = Image.new("RGB", (300, 300), color="white")
        img.save(img_path)

        record = {
            "id": doc_id,
            "filename": "test_invoice.png",
            "stored_filename": f"{doc_id}.png",
            "file_path": str(img_path),
            "content_type": "image/png",
            "document_type": "Invoice",
            "uploaded_at": main._now(),
            "processing_status": "processed",
            "extracted_fields": {},
            "invoice_fields": {},
            "extracted_text": "",
            "confidence_score": 0.0,
            "confidence_method": "layoutlm",
            "verified": False,
            "verified_at": None,
            "processing_error": None,
            "ground_truth": None,
            "page_count": 1,
            "pages_processed": 1,
            "text_source": "tesseract",
            "message": "Ready for layoutlm",
        }
        database.insert_document(record)

        # Call endpoint directly: must not raise HTTP 500
        res = main.extract_with_layoutlm(doc_id)
        self.assertEqual(res["document_id"], doc_id)
        self.assertEqual(res["method"], "layoutlmv3_funsd")
        self.assertIn("fields", res)
        self.assertIn("confidence_scores", res)
        self.assertIn("document", res)

        # Cleanup
        main.remove_document(doc_id)

    def test_20_gst_vendor_buyer_disambiguation(self):
        """Verify that Seller/Vendor GSTIN is prioritized over Buyer GSTIN and invalid GST is rejected."""
        pairs = [
            ("GSTIN NO:", "37COAPS3244E1Z1", 0.99, [117, 131, 196, 138], [201, 131, 329, 138]),
            ("To PRASANTHI PHARMACY GST:", "37BAPPK1430C1Z0", 0.98, [467, 155, 499, 162], [509, 155, 629, 162]),
            ("TAX INVOICE", None, 0.90, [448, 22, 551, 29], None),
            ("SCHOOL ROAD,BHAVANIPURAM", ",VIJAYAWADA-520012", 0.85, [400, 96, 647, 105], [653, 96, 829, 105]),
            ("Cell:", "9394875966", 0.99, [660, 112, 692, 119], [708, 112, 908, 121]),
        ]
        headers = []
        raw_confs = {p[0]: p[2] for p in pairs}

        canonical, confs, unmapped = LayoutLMv3Extractor._map_to_canonical_fields(
            pairs=pairs,
            headers=headers,
            document_type="Invoice",
            raw_confs=raw_confs,
        )

        # 1. Vendor GSTIN must be assigned, NOT buyer GSTIN
        self.assertEqual(canonical.get("gst_number"), "37COAPS3244E1Z1")
        self.assertNotEqual(canonical.get("gst_number"), "37BAPPK1430C1Z0")

        # 2. Arbitrary words (TAX, SCHOOL, ROAD, Cell) must NOT be in canonical fields
        for unwanted in ["TAX INVOICE", "TAX", "SCHOOL ROAD,BHAVANIPURAM", "Cell", "ROAD"]:
            self.assertNotIn(unwanted, canonical)

        # 3. Arbitrary pairs must be preserved in unmapped
        self.assertIn("Cell", unmapped)
        self.assertIn("SCHOOL ROAD,BHAVANIPURAM", unmapped)

    def test_21_canonical_invoice_monetary_and_vendor_fields(self):
        """Verify extraction and separation of vendor name, customer name, monetary breakdown fields, and currency."""
        pairs = [
            ("GSTIN NO:", "37COAPS3244E1Z1", 0.99, [117, 131, 196, 138], [201, 131, 329, 138]),
            ("To PRASANTHI PHARMACY", "43-107/1-24/3/7,NEAR SAT BABA", 0.95, [50, 155, 242, 162], [250, 155, 450, 162]),
            ("To PRASANTHI PHARMACY GST:", "37BAPPK1430C1Z0", 0.98, [467, 155, 499, 162], [509, 155, 629, 162]),
            ("INVOICE NO:", "GSTSAL/379", 0.95, [716, 155, 825, 162], [837, 155, 915, 163]),
            ("DATE:", "02/Jun/2026", 0.99, [716, 172, 762, 179], [837, 172, 929, 180]),
            ("Gross Amount", "16753.50", 0.99, [704, 719, 785, 726], [899, 719, 947, 725]),
            ("CGST Amount:", "418.84", 0.99, [704, 748, 795, 754], [910, 747, 947, 754]),
            ("SGST Amount :", "418.84", 0.99, [704, 762, 793, 768], [910, 762, 947, 768]),
            ("Net Amount :", "17591.00", 0.99, [704, 777, 805, 786], [881, 777, 946, 786]),
            ("Customer", None, 0.82, [159, 827, 233, 834], None),
        ]
        headers = ["TAX INVOICE", "ofltems:"]
        raw_confs = {p[0]: p[2] for p in pairs}
        valid_words = [
            "TAX", "INVOICE", "GSTIN", "NO:", "37COAPS3244E1Z1",
            "To", "PRASANTHI", "PHARMACY", "GST:", "37BAPPK1430C1Z0",
            "INVOICE", "NO:", "GSTSAL/379", "DATE:", "02/Jun/2026",
            "Gross", "Amount", "16753.50",
            "CGST", "Amount:", "418.84",
            "SGST", "Amount:", "418.84",
            "Net", "Amount:", "17591.00",
            "Rupees", "Seventeen", "Thousand", "Five", "Hundred", "Ninety", "One", "Only",
            "Customer", "'s", "Signature", "For", "SIVA", "SAI", "ENTERPRISES.",
        ]

        canonical, confs, unmapped = LayoutLMv3Extractor._map_to_canonical_fields(
            pairs=pairs,
            headers=headers,
            document_type="Invoice",
            raw_confs=raw_confs,
            valid_words=valid_words,
        )

        # 1. Vendor vs Customer Disambiguation
        self.assertEqual(canonical.get("vendor_name"), "SIVA SAI ENTERPRISES")
        self.assertEqual(canonical.get("customer_name"), "PRASANTHI PHARMACY")
        self.assertNotEqual(canonical.get("vendor_name"), canonical.get("customer_name"))

        # Reject generic labels or table headers as names
        self.assertNotEqual(canonical.get("customer_name"), "Customer")
        self.assertNotEqual(canonical.get("vendor_name"), "ofltems:")
        self.assertNotEqual(canonical.get("vendor_name"), "Noofltems:")

        # 2. Identification Fields
        self.assertEqual(canonical.get("gst_number"), "37COAPS3244E1Z1")
        self.assertEqual(canonical.get("invoice_number"), "GSTSAL/379")
        self.assertEqual(canonical.get("invoice_date"), "02/Jun/2026")

        # 3. Monetary Breakdown Fields
        self.assertEqual(canonical.get("gross_amount"), "16753.50")
        self.assertEqual(canonical.get("subtotal"), "16753.50")
        self.assertEqual(canonical.get("cgst_amount"), "418.84")
        self.assertEqual(canonical.get("sgst_amount"), "418.84")
        self.assertEqual(canonical.get("tax_amount"), "837.68")  # Composite CGST + SGST
        self.assertEqual(canonical.get("total_amount"), "17591.00")
        self.assertEqual(canonical.get("currency"), "INR")

        # 4. Missing fields remain None / absent without fabrication
        self.assertNotIn("due_date", canonical)
        self.assertNotIn("po_number", canonical)

    def test_22_generic_vendor_and_customer_name_patterns(self):
        """Verify generic extraction of customer and vendor names across standard invoice layouts."""
        pairs = [
            ("Bill To:", "APEX SOLUTIONS PVT LTD", 0.94, [100, 200, 180, 215], [200, 200, 450, 215]),
            ("Vendor:", "GLOBAL LOGISTICS INC", 0.96, [100, 100, 180, 115], [200, 100, 420, 115]),
            ("Customer 's Signature", None, 0.88, [100, 900, 250, 915], None),
        ]
        headers = ["COMMERCIAL INVOICE", "No. of Items:", "Description", "Total"]
        raw_confs = {p[0]: p[2] for p in pairs}

        canonical, confs, unmapped = LayoutLMv3Extractor._map_to_canonical_fields(
            pairs=pairs,
            headers=headers,
            document_type="Invoice",
            raw_confs=raw_confs,
            valid_words=["APEX", "SOLUTIONS", "PVT", "LTD", "GLOBAL", "LOGISTICS", "INC"],
        )

        self.assertEqual(canonical.get("customer_name"), "APEX SOLUTIONS PVT LTD")
        self.assertEqual(canonical.get("vendor_name"), "GLOBAL LOGISTICS INC")
        self.assertNotEqual(canonical.get("customer_name"), "Customer 's Signature")
        self.assertNotEqual(canonical.get("vendor_name"), "No. of Items:")

    def test_23_regression_noofltems_rejected_and_top_vendor_selected(self):
        """Regression test: ensure 'Noofltems' is rejected and top invoice company name is selected."""
        ocr_text = (
            "SIVA SAI ENTERPRISES\n"
            "20B : WLF20B2025AP000836\n"
            "TAX INVOICE\n"
            "GSTIN NO : 37COAPS3244E1Z1\n"
            "To PRASANTHI PHARMACY\n"
            "INVOICE NO : GSTSAL/379\n"
            "DATE : 02/Jun/2026\n"
            "Noofltems : 6\n"
            "Gross Amount : 16753.50\n"
            "Net Amount : 17591.00\n"
        )
        pairs = [
            ("To PRASANTHI PHARMACY", None, 0.95, [50, 155, 242, 162], None),
            ("INVOICE NO :", "GSTSAL/379", 0.95, [716, 155, 825, 162], [837, 155, 915, 163]),
            ("DATE :", "02/Jun/2026", 0.99, [716, 172, 762, 179], [837, 172, 929, 180]),
            ("Noofltems :", "6", 0.95, [74, 705, 126, 711], [143, 705, 149, 711]),
            ("Gross Amount :", "16753.50", 0.99, [704, 719, 785, 726], [899, 719, 947, 725]),
            ("Net Amount :", "17591.00", 0.99, [704, 777, 805, 786], [881, 777, 946, 786]),
        ]
        headers = ["TAX INVOICE", "Noofltems:", "Noofltems"]
        raw_confs = {p[0]: p[2] for p in pairs}

        canonical, confs, unmapped = LayoutLMv3Extractor._map_to_canonical_fields(
            pairs=pairs,
            headers=headers,
            document_type="Invoice",
            raw_confs=raw_confs,
            valid_words=["SIVA", "SAI", "ENTERPRISES", "PRASANTHI", "PHARMACY"],
            ocr_text=ocr_text,
        )

        # 1. 'Noofltems' or any variation must be rejected
        self.assertNotEqual(canonical.get("vendor_name"), "Noofltems")
        self.assertNotEqual(canonical.get("vendor_name"), "Noofltems:")
        self.assertNotIn("Noofltems", canonical.get("vendor_name", ""))

        # 2. Correct vendor name at top of invoice must be selected
        self.assertEqual(canonical.get("vendor_name"), "SIVA SAI ENTERPRISES")

        # 3. Customer name must be PRASANTHI PHARMACY
        self.assertEqual(canonical.get("customer_name"), "PRASANTHI PHARMACY")

    def test_24_fatura2_model_loading_and_47_labels(self):
        """Verify local FATURA2 model loading and 47 BIO labels vocabulary."""
        extractor = LayoutLMv3Extractor()
        extractor.load()
        self.assertTrue(extractor.is_loaded())
        self.assertIsNotNone(extractor._model)
        self.assertIsNotNone(extractor._processor)
        
        # Verify 47 BIO labels
        id2label = getattr(extractor._model.config, "id2label", {})
        self.assertEqual(len(id2label), 47)
        self.assertEqual(id2label.get(0) or id2label.get("0"), "O")
        self.assertIn("B-TABLE", id2label.values())
        self.assertIn("B-NUMBER", id2label.values())
        self.assertIn("B-DATE", id2label.values())
        self.assertIn("B-DUE_DATE", id2label.values())
        self.assertIn("B-SELLER", id2label.values())
        self.assertIn("B-BILL_TO", id2label.values())
        self.assertIn("B-TOTAL", id2label.values())
        self.assertIn("B-SUBTOTAL", id2label.values())
        self.assertIn("B-TAX", id2label.values())
        self.assertIn("B-AMOUNT_DUE", id2label.values())

    def test_25_fatura2_invoice_extraction_mappings(self):
        """Verify FATURA2 entity labels are mapped to canonical invoice schema fields."""
        extractor = LayoutLMv3Extractor()
        extractor.load()
        
        # Test synthetic invoice tokens matching FATURA2 vocabulary
        words = ["Acme", "Corp", "John", "Doe", "INV-9988", "2024-11-20", "2024-12-20", "1000.00", "100.00", "1100.00"]
        boxes = [[10, 10, 50, 30]] * len(words)
        
        # Map IDs using the loaded model's label2id
        label2id = getattr(extractor._model.config, "label2id", {})
        word_preds = {
            0: label2id.get("B-SELLER", 17),
            1: label2id.get("I-SELLER", 18),
            2: label2id.get("B-BILL_TO", 23),
            3: label2id.get("I-BILL_TO", 24),
            4: label2id.get("B-NUMBER", 7),
            5: label2id.get("B-DATE", 9),
            6: label2id.get("B-DUE_DATE", 11),
            7: label2id.get("B-SUBTOTAL", 27),
            8: label2id.get("B-TAX", 29),
            9: label2id.get("B-TOTAL", 35),
        }
        word_confs = {i: 0.95 for i in range(len(words))}
        id2label = getattr(extractor._model.config, "id2label", {})
        
        res = extractor._extract_invoice(
            valid_words=words,
            word_preds=word_preds,
            word_confs=word_confs,
            norm_boxes=boxes,
            id2label=id2label,
            ocr_text=" ".join(words),
            document_type="Invoice",
            start=0.0,
        )
        
        self.assertEqual(res.method, "layoutlmv3_fatura2")
        self.assertEqual(res.fields.get("vendor_name"), "Acme Corp")
        self.assertEqual(res.fields.get("customer_name"), "John Doe")
        self.assertEqual(res.fields.get("invoice_number"), "9988")
        self.assertEqual(res.fields.get("invoice_date"), "2024-11-20")
        self.assertEqual(res.fields.get("due_date"), "2024-12-20")
        self.assertEqual(res.fields.get("subtotal"), "1000.00")
        self.assertEqual(res.fields.get("tax_amount"), "100.00")
        self.assertEqual(res.fields.get("total_amount"), "1100.00")

    def test_26_funsd_fallback_preservation(self):
        """Verify graceful fallback to FUNSD when model path is invalid."""
        fallback_extractor = LayoutLMv3Extractor(
            model_name="nonexistent/corrupted/checkpoint/path",
            fallback_model="nielsr/layoutlmv3-finetuned-funsd",
        )
        self.assertEqual(fallback_extractor.fallback_model, "nielsr/layoutlmv3-finetuned-funsd")

    def test_27_document_qa_unchanged(self):
        """Verify Document QA model configuration and interface remain unchanged."""
        qa = main.get_qa()
        self.assertEqual(qa.model_name, "impira/layoutlm-document-qa")
        self.assertTrue(qa.is_available())

    def test_28_document_qa_successful_local_loading(self):
        """Verify Document QA loads successfully and quickly from local cache with local_files_only=True."""
        import time
        from PIL import Image
        qa = DocumentQA(local_files_only=True)
        t0 = time.time()
        qa.load()
        load_time = time.time() - t0

        self.assertTrue(qa.is_loaded())
        self.assertTrue(qa.is_available())
        self.assertIsNone(qa._load_error)
        self.assertLess(load_time, 10.0, f"Loading should not hang on network retries (took {load_time:.2f}s)")

        # Verify inference works with local model
        dummy_img = Image.new("RGB", (100, 100), color="white")
        result = qa.answer(dummy_img, "What is the invoice number?")
        self.assertIsNotNone(result)
        self.assertEqual(result.question, "What is the invoice number?")

    def test_29_document_qa_unavailable_local_cache(self):
        """Verify Document QA handles missing cached model files gracefully with unavailable status and error."""
        import time
        from PIL import Image
        qa = DocumentQA(
            model_name="nonexistent/uncached-mock-model-for-testing",
            local_files_only=True,
        )
        t0 = time.time()
        qa.load()
        elapsed = time.time() - t0

        self.assertFalse(qa.is_loaded())
        self.assertFalse(qa.is_available())
        self.assertIsNotNone(qa._load_error)
        self.assertIn("unavailable in local cache", qa._load_error.lower())
        self.assertLess(elapsed, 3.0, f"Failure should be immediate without retry delays (took {elapsed:.2f}s)")

        # Calling answer() should return structured QAResult with explanation and answer=None
        dummy_img = Image.new("RGB", (50, 50), color="white")
        result = qa.answer(dummy_img, "What is the date?")
        self.assertIsNone(result.answer)
        self.assertEqual(result.confidence, 0.0)
        self.assertIsNotNone(result.explanation)

    def test_30_document_qa_offline_failure_handling(self):
        """Verify Document QA respects offline mode toggle and handles failure gracefully."""
        qa = DocumentQA(
            model_name="invalid/checkpoint-path-not-existing",
            offline=True,
        )
        self.assertTrue(qa.is_offline_mode())
        qa.load()
        self.assertFalse(qa.is_loaded())
        self.assertFalse(qa.is_available())
        self.assertIsNotNone(qa._load_error)
        self.assertIn("unavailable in local cache", qa._load_error.lower())

    # ── 12. Multi-Page Document Pipeline Handling ────────────────────

    def test_31_multipage_pdf_layoutlm_extraction(self):
        """Verify LayoutLMv3 extracts fields across all pages of a multi-page PDF."""
        import pymupdf
        doc_id = "test_multipage_layoutlm_doc"
        database.delete_document(doc_id)

        # Create synthetic 2-page PDF
        doc = pymupdf.open()
        p1 = doc.new_page(width=600, height=800)
        p1.insert_text((50, 100), "INVOICE\nVendor: Global Tech Inc\nInvoice Number: INV-889911\nDate: 2026-03-01", fontsize=14)

        p2 = doc.new_page(width=600, height=800)
        p2.insert_text((50, 100), "SUMMARY & TOTALS\nSubtotal: $2500.00\nSales Tax: $250.00\nTOTAL AMOUNT: 2750.00", fontsize=14)

        pdf_bytes = doc.tobytes()
        doc.close()

        dest = main.UPLOAD_DIR / f"{doc_id}_multipage.pdf"
        dest.write_bytes(pdf_bytes)

        record = {
            "id": doc_id,
            "filename": "multipage.pdf",
            "stored_filename": dest.name,
            "file_path": str(dest),
            "file_size": len(pdf_bytes),
            "content_type": "application/pdf",
            "document_type": "Invoice",
            "uploaded_at": main._now(),
            "processing_status": "processed",
            "extracted_fields": {},
            "invoice_fields": {},
            "extracted_text": "Page 1\n\nPage 2",
            "confidence_score": 0.0,
            "confidence_method": "baseline",
            "verified": False,
            "verified_at": None,
            "processing_error": None,
            "ground_truth": None,
            "page_count": 2,
            "pages_processed": 2,
            "text_source": "pymupdf",
            "message": "Ready for multi-page extraction",
        }
        database.insert_document(record)

        try:
            res = main.extract_with_layoutlm(doc_id)
            self.assertEqual(res["document_id"], doc_id)
            self.assertEqual(res["metadata"]["pages_processed"], 2)

            # Check that fields from Page 1 are present
            fields = res["fields"]
            self.assertIn("vendor_name", fields)
            self.assertEqual(fields["vendor_name"], "Global Tech Inc")
            self.assertIn("invoice_date", fields)

            # Check that fields from Page 2 are present
            self.assertIn("total_amount", fields)
            self.assertEqual(fields["total_amount"], "2750.00")

            # Check that field source provenance is tracked
            source_pages = res["metadata"]["field_source_pages"]
            self.assertEqual(source_pages.get("vendor_name"), 1)
            self.assertEqual(source_pages.get("total_amount"), 2)

            # Check that page_results contains both pages
            page_results = res["metadata"]["page_results"]
            self.assertEqual(len(page_results), 2)
            self.assertEqual(page_results[0]["page_number"], 1)
            self.assertEqual(page_results[1]["page_number"], 2)
        finally:
            main.remove_document(doc_id)

    def test_32_multipage_pdf_document_qa(self):
        """Verify Document QA evaluates all pages and returns answers from Page 1 and Page 2."""
        import pymupdf
        doc_id = "test_multipage_qa_doc"
        database.delete_document(doc_id)

        doc = pymupdf.open()
        p1 = doc.new_page(width=600, height=800)
        p1.insert_text((50, 100), "INVOICE\nVendor: Global Tech Inc\nInvoice Number: INV-889911\nDate: 2026-03-01", fontsize=14)

        p2 = doc.new_page(width=600, height=800)
        p2.insert_text((50, 100), "SUMMARY & TERMS\nSubtotal: $2500.00\nSales Tax: $250.00\nTOTAL AMOUNT: 2750.00\nWarranty: 3 Years Extended Coverage", fontsize=14)

        pdf_bytes = doc.tobytes()
        doc.close()

        dest = main.UPLOAD_DIR / f"{doc_id}_multipage_qa.pdf"
        dest.write_bytes(pdf_bytes)

        record = {
            "id": doc_id,
            "filename": "multipage_qa.pdf",
            "stored_filename": dest.name,
            "file_path": str(dest),
            "file_size": len(pdf_bytes),
            "content_type": "application/pdf",
            "document_type": "Invoice",
            "uploaded_at": main._now(),
            "processing_status": "processed",
            "extracted_fields": {},
            "invoice_fields": {},
            "extracted_text": "Page 1\n\nPage 2",
            "confidence_score": 0.0,
            "confidence_method": "baseline",
            "verified": False,
            "verified_at": None,
            "processing_error": None,
            "ground_truth": None,
            "page_count": 2,
            "pages_processed": 2,
            "text_source": "pymupdf",
            "message": "Ready for multi-page QA",
        }
        database.insert_document(record)

        try:
            # Query Page 1 info: Invoice Number
            qa1 = main.ask_document_question(doc_id, QARequest(question="What is the invoice number?"))
            self.assertEqual(qa1["pages_searched"], 2)
            self.assertEqual(qa1["page_number"], 1)
            self.assertIn("INV-8899", qa1["answer"] or "")

            # Query Page 2 info: Warranty
            qa2 = main.ask_document_question(doc_id, QARequest(question="What is the warranty?"))
            self.assertEqual(qa2["pages_searched"], 2)
            self.assertEqual(qa2["page_number"], 2)
            self.assertIn("3 Years Extended Coverage", qa2["answer"] or "")
        finally:
            main.remove_document(doc_id)

    def test_33_single_page_processing_preservation(self):
        """Verify single-page processing remains functional and unaffected."""
        from PIL import Image
        doc_id = "test_singlepage_doc"
        database.delete_document(doc_id)

        # Single page image
        img = Image.new("RGB", (300, 300), color="white")
        dest = main.UPLOAD_DIR / f"{doc_id}_single.png"
        img.save(dest)

        record = {
            "id": doc_id,
            "filename": "single.png",
            "stored_filename": dest.name,
            "file_path": str(dest),
            "file_size": 500,
            "content_type": "image/png",
            "document_type": "Invoice",
            "uploaded_at": main._now(),
            "processing_status": "processed",
            "extracted_fields": {},
            "invoice_fields": {},
            "extracted_text": "Single page text",
            "confidence_score": 0.0,
            "confidence_method": "baseline",
            "verified": False,
            "verified_at": None,
            "processing_error": None,
            "ground_truth": None,
            "page_count": 1,
            "pages_processed": 1,
            "text_source": "tesseract",
            "message": "Ready for single-page",
        }
        database.insert_document(record)

        try:
            # Multi-image loader returns 1 page
            pages = main._load_document_images(str(dest))
            self.assertEqual(len(pages), 1)
            self.assertEqual(pages[0][0], 1)

            # Single-image loader returns Image object
            single_img = main._load_document_image(str(dest))
            self.assertIsInstance(single_img, Image.Image)

            # LayoutLM on single page
            res = main.extract_with_layoutlm(doc_id)
            self.assertEqual(res["metadata"]["pages_processed"], 1)

            # QA on single page
            qa_res = main.ask_document_question(doc_id, QARequest(question="What is the total?"))
            self.assertEqual(qa_res["pages_searched"], 1)
            self.assertEqual(qa_res["page_number"], 1)
        finally:
            main.remove_document(doc_id)

    def test_34_multipage_edge_cases_and_error_handling(self):
        """Verify edge case handling for empty PDFs, invalid paths, and blank pages."""
        import pymupdf
        # 1. Non-existent file path
        with self.assertRaises(HTTPException) as ctx404:
            main._load_document_images("nonexistent/invalid_file_path.pdf")
        self.assertEqual(ctx404.exception.status_code, 404)

        # 2. Corrupted / 0-page PDF
        empty_dest = main.UPLOAD_DIR / "empty_doc.pdf"
        empty_dest.write_bytes(b"%PDF-1.4\n%%EOF\n")

        try:
            with self.assertRaises(HTTPException) as ctx400:
                main._load_document_images(str(empty_dest))
            self.assertEqual(ctx400.exception.status_code, 400)
        finally:
            import gc
            gc.collect()
            if empty_dest.exists():
                try:
                    empty_dest.unlink()
                except Exception:
                    pass

        # 3. Multi-page PDF with blank second page
        doc2 = pymupdf.open()
        p1 = doc2.new_page(width=600, height=800)
        p1.insert_text((50, 100), "INVOICE\nInvoice Number: INV-BLANK-TEST\nTOTAL AMOUNT: 500.00", fontsize=14)
        doc2.new_page(width=600, height=800)  # Blank page 2
        blank_p2_bytes = doc2.tobytes()
        doc2.close()

        doc_id = "test_blank_p2_doc"
        dest2 = main.UPLOAD_DIR / f"{doc_id}.pdf"
        dest2.write_bytes(blank_p2_bytes)

        record = {
            "id": doc_id,
            "filename": "blank_p2.pdf",
            "stored_filename": dest2.name,
            "file_path": str(dest2),
            "file_size": len(blank_p2_bytes),
            "content_type": "application/pdf",
            "document_type": "Invoice",
            "uploaded_at": main._now(),
            "processing_status": "processed",
            "extracted_fields": {},
            "invoice_fields": {},
            "extracted_text": "Invoice Number: INV-BLANK-TEST",
            "confidence_score": 0.0,
            "confidence_method": "baseline",
            "verified": False,
            "verified_at": None,
            "processing_error": None,
            "ground_truth": None,
            "page_count": 2,
            "pages_processed": 2,
            "text_source": "pymupdf",
            "message": "Ready",
        }
        database.insert_document(record)

        try:
            res = main.extract_with_layoutlm(doc_id)
            self.assertEqual(res["metadata"]["pages_processed"], 2)
            self.assertIn("invoice_number", res["fields"])
        finally:
            main.remove_document(doc_id)

    # ── 12. Priority 3: Structured Line-Item Extraction Tests ────────

    def test_35_line_items_single_page_multi_row(self):
        """Verify extraction of multiple structured table rows on a single page."""
        extractor = LayoutLMv3Extractor()
        entities = [
            # Row 1: Widget A (2 x 15.00 = 30.00)
            {"label": "ITEM_DESCRIPTION", "text": "Widget A", "box": [50, 200, 250, 220], "confidence": 0.95},
            {"label": "ITEM_QUANTITY", "text": "2", "box": [300, 200, 320, 220], "confidence": 0.92},
            {"label": "ITEM_UNIT_PRICE", "text": "$15.00", "box": [400, 200, 460, 220], "confidence": 0.94},
            {"label": "ITEM_TOTAL", "text": "$30.00", "box": [550, 200, 620, 220], "confidence": 0.96},
            # Row 2: Gadget B (5 x 10.00 = 50.00)
            {"label": "ITEM_DESCRIPTION", "text": "Gadget B", "box": [50, 240, 250, 260], "confidence": 0.91},
            {"label": "ITEM_QUANTITY", "text": "5 pcs", "box": [300, 240, 340, 260], "confidence": 0.90},
            {"label": "ITEM_UNIT_PRICE", "text": "$10.00", "box": [400, 240, 460, 260], "confidence": 0.93},
            {"label": "ITEM_TOTAL", "text": "$50.00", "box": [550, 240, 620, 260], "confidence": 0.95},
            # Row 3: Consulting Service (10 x 85.00 = 850.00)
            {"label": "ITEM_DESCRIPTION", "text": "Consulting Service", "box": [50, 280, 260, 300], "confidence": 0.97},
            {"label": "ITEM_QUANTITY", "text": "10 hrs", "box": [300, 280, 345, 300], "confidence": 0.93},
            {"label": "ITEM_UNIT_PRICE", "text": "$85.00", "box": [400, 280, 460, 300], "confidence": 0.96},
            {"label": "ITEM_TOTAL", "text": "$850.00", "box": [550, 280, 620, 300], "confidence": 0.98},
        ]

        items = extractor._extract_table_line_items(entities, page_number=1)
        self.assertEqual(len(items), 3)

        # Row 1 checks
        r1 = items[0]
        self.assertEqual(r1["row_index"], 1)
        self.assertEqual(r1["page"], 1)
        self.assertEqual(r1["description"], "Widget A")
        self.assertEqual(r1["quantity"], 2.0)
        self.assertEqual(r1["unit_price"], 15.0)
        self.assertEqual(r1["item_total"], 30.0)
        self.assertEqual(r1["raw_description"], "Widget A")
        self.assertEqual(r1["raw_quantity"], "2")
        self.assertEqual(r1["raw_unit_price"], "$15.00")
        self.assertEqual(r1["raw_item_total"], "$30.00")
        self.assertTrue(r1["is_valid"])
        self.assertFalse(r1["suspicious"])
        self.assertIsNone(r1["validation_error"])

        # Row 2 checks
        r2 = items[1]
        self.assertEqual(r2["row_index"], 2)
        self.assertEqual(r2["description"], "Gadget B")
        self.assertEqual(r2["quantity"], 5.0)
        self.assertEqual(r2["unit_price"], 10.0)
        self.assertEqual(r2["item_total"], 50.0)
        self.assertTrue(r2["is_valid"])
        self.assertFalse(r2["suspicious"])

        # Row 3 checks
        r3 = items[2]
        self.assertEqual(r3["row_index"], 3)
        self.assertEqual(r3["description"], "Consulting Service")
        self.assertEqual(r3["quantity"], 10.0)
        self.assertEqual(r3["unit_price"], 85.0)
        self.assertEqual(r3["item_total"], 850.0)
        self.assertTrue(r3["is_valid"])
        self.assertFalse(r3["suspicious"])

    def test_36_line_items_multi_page_distribution(self):
        """Verify line items spread across multiple PDF pages maintain page numbers and sequential ordering."""
        extractor = LayoutLMv3Extractor()

        # Mock page 1 result with 2 items
        p1_res = ExtractionResult(
            method="layoutlmv3_fatura2",
            fields={"invoice_number": "INV-P1-P2"},
            confidence_scores={"invoice_number": 0.99},
            line_items=[
                {"row_index": 1, "page": 1, "description": "Page 1 Item 1", "quantity": 1.0, "unit_price": 50.0, "item_total": 50.0, "is_valid": True, "suspicious": False, "validation_error": None},
                {"row_index": 2, "page": 1, "description": "Page 1 Item 2", "quantity": 2.0, "unit_price": 25.0, "item_total": 50.0, "is_valid": True, "suspicious": False, "validation_error": None},
            ],
            processing_time=0.5,
        )

        # Mock page 2 result with 2 items
        p2_res = ExtractionResult(
            method="layoutlmv3_fatura2",
            fields={"total_amount": "200.00"},
            confidence_scores={"total_amount": 0.98},
            line_items=[
                {"row_index": 1, "page": 2, "description": "Page 2 Item 3", "quantity": 1.0, "unit_price": 75.0, "item_total": 75.0, "is_valid": True, "suspicious": False, "validation_error": None},
                {"row_index": 2, "page": 2, "description": "Page 2 Item 4", "quantity": 1.0, "unit_price": 25.0, "item_total": 25.0, "is_valid": True, "suspicious": False, "validation_error": None},
            ],
            processing_time=0.4,
        )

        from unittest.mock import patch
        with patch.object(extractor, "extract", side_effect=[p1_res, p2_res]):
            from PIL import Image
            img1 = Image.new("RGB", (100, 100), color="white")
            img2 = Image.new("RGB", (100, 100), color="white")
            combined = extractor.extract_pages([img1, img2], ocr_text="")

        items = combined.line_items
        self.assertEqual(len(items), 4)

        # Verify sequential row indices across document
        self.assertEqual([it["row_index"] for it in items], [1, 2, 3, 4])
        # Verify page numbers match provenance
        self.assertEqual([it["page"] for it in items], [1, 1, 2, 2])
        # Verify descriptions
        self.assertEqual(items[0]["description"], "Page 1 Item 1")
        self.assertEqual(items[1]["description"], "Page 1 Item 2")
        self.assertEqual(items[2]["description"], "Page 2 Item 3")
        self.assertEqual(items[3]["description"], "Page 2 Item 4")

        # Verify page-level metadata
        page_results = combined.metadata["page_results"]
        self.assertEqual(len(page_results), 2)
        self.assertEqual(len(page_results[0]["line_items"]), 2)
        self.assertEqual(len(page_results[1]["line_items"]), 2)
        self.assertEqual(combined.metadata["total_line_items"], 4)

    def test_37_line_items_partial_rows_preserved(self):
        """Verify partial rows (missing quantity or unit price) are preserved gracefully without hallucination."""
        extractor = LayoutLMv3Extractor()
        entities = [
            # Row 1: Description and total only (e.g. shipping/labor flat fee)
            {"label": "ITEM_DESCRIPTION", "text": "Standard Shipping", "box": [50, 200, 250, 220], "confidence": 0.95},
            {"label": "ITEM_TOTAL", "text": "15.00", "box": [550, 200, 620, 220], "confidence": 0.96},
            # Row 2: Description and unit price only
            {"label": "ITEM_DESCRIPTION", "text": "Maintenance Retainer", "box": [50, 250, 250, 270], "confidence": 0.93},
            {"label": "ITEM_UNIT_PRICE", "text": "120.00", "box": [400, 250, 460, 270], "confidence": 0.91},
        ]

        items = extractor._extract_table_line_items(entities, page_number=1)
        self.assertEqual(len(items), 2)

        # Row 1: missing quantity & unit_price should be None, never hallucinated
        r1 = items[0]
        self.assertEqual(r1["description"], "Standard Shipping")
        self.assertIsNone(r1["quantity"])
        self.assertIsNone(r1["unit_price"])
        self.assertEqual(r1["item_total"], 15.0)
        self.assertTrue(r1["is_valid"])
        self.assertFalse(r1["suspicious"])

        # Row 2: missing quantity & item_total should be None
        r2 = items[1]
        self.assertEqual(r2["description"], "Maintenance Retainer")
        self.assertIsNone(r2["quantity"])
        self.assertEqual(r2["unit_price"], 120.0)
        self.assertIsNone(r2["item_total"])
        self.assertTrue(r2["is_valid"])
        self.assertFalse(r2["suspicious"])

    def test_38_line_items_arithmetic_mismatch_and_ocr_errors(self):
        """Verify suspicious row detection: arithmetic mismatch and incomplete/corrupted OCR values."""
        extractor = LayoutLMv3Extractor()
        entities = [
            # Row 1: Arithmetic mismatch (2 x 15.00 = 30.00, but total says 99.00)
            {"label": "ITEM_DESCRIPTION", "text": "Wrong Calculation Item", "box": [50, 200, 250, 220], "confidence": 0.95},
            {"label": "ITEM_QUANTITY", "text": "2", "box": [300, 200, 320, 220], "confidence": 0.92},
            {"label": "ITEM_UNIT_PRICE", "text": "15.00", "box": [400, 200, 460, 220], "confidence": 0.94},
            {"label": "ITEM_TOTAL", "text": "99.00", "box": [550, 200, 620, 220], "confidence": 0.96},
            # Row 2: Corrupted numeric quantity (OCR noise: 'INVALID_QTY_TEXT')
            {"label": "ITEM_DESCRIPTION", "text": "Corrupted Qty Item", "box": [50, 250, 250, 270], "confidence": 0.90},
            {"label": "ITEM_QUANTITY", "text": "INVALID_QTY_TEXT", "box": [300, 250, 380, 270], "confidence": 0.88},
            {"label": "ITEM_UNIT_PRICE", "text": "20.00", "box": [400, 250, 460, 270], "confidence": 0.91},
            {"label": "ITEM_TOTAL", "text": "40.00", "box": [550, 250, 620, 270], "confidence": 0.92},
            # Row 3: Missing description (numbers present but no item name)
            {"label": "ITEM_QUANTITY", "text": "1", "box": [300, 300, 320, 320], "confidence": 0.90},
            {"label": "ITEM_UNIT_PRICE", "text": "50.00", "box": [400, 300, 460, 320], "confidence": 0.91},
            {"label": "ITEM_TOTAL", "text": "50.00", "box": [550, 300, 620, 320], "confidence": 0.92},
        ]

        items = extractor._extract_table_line_items(entities, page_number=1)
        self.assertEqual(len(items), 3)

        # Row 1: flagged as arithmetic mismatch
        r1 = items[0]
        self.assertFalse(r1["is_valid"])
        self.assertTrue(r1["suspicious"])
        self.assertIn("Arithmetic mismatch", r1["validation_error"])

        # Row 2: flagged as invalid numeric format
        r2 = items[1]
        self.assertFalse(r2["is_valid"])
        self.assertTrue(r2["suspicious"])
        self.assertIsNone(r2["quantity"])
        self.assertEqual(r2["raw_quantity"], "INVALID_QTY_TEXT")
        self.assertIn("Invalid numeric format for quantity", r2["validation_error"])

        # Row 3: flagged as missing description
        r3 = items[2]
        self.assertFalse(r3["is_valid"])
        self.assertTrue(r3["suspicious"])
        self.assertIsNone(r3["description"])
        self.assertIn("Missing item description", r3["validation_error"])

    def test_39_line_items_api_endpoint_and_database_persistence(self):
        """Verify line_items are saved to SQLite and returned through API extraction endpoints."""
        doc_id = "test_line_items_persistence_doc"
        database.delete_document(doc_id)

        from PIL import Image
        img = Image.new("RGB", (300, 300), color="white")
        dest = main.UPLOAD_DIR / f"{doc_id}.png"
        img.save(dest)

        record = {
            "id": doc_id,
            "filename": "invoice_lineitems.png",
            "stored_filename": dest.name,
            "file_path": str(dest),
            "file_size": 500,
            "content_type": "image/png",
            "document_type": "Invoice",
            "uploaded_at": main._now(),
            "processing_status": "processed",
            "extracted_fields": {},
            "invoice_fields": {},
            "extracted_text": "Sample Invoice",
            "confidence_score": 0.0,
            "confidence_method": "baseline",
            "verified": False,
            "verified_at": None,
            "processing_error": None,
            "ground_truth": None,
            "page_count": 1,
            "pages_processed": 1,
            "text_source": "tesseract",
            "message": "Ready",
        }
        database.insert_document(record)

        mock_line_items = [
            {
                "row_index": 1,
                "page": 1,
                "description": "Item Alpha",
                "quantity": 2.0,
                "unit_price": 50.0,
                "item_total": 100.0,
                "raw_description": "Item Alpha",
                "raw_quantity": "2",
                "raw_unit_price": "50.00",
                "raw_item_total": "100.00",
                "confidence": 0.95,
                "box": [50, 200, 600, 220],
                "is_valid": True,
                "suspicious": False,
                "validation_error": None,
            }
        ]

        from unittest.mock import patch
        mock_result = ExtractionResult(
            method="layoutlmv3_fatura2",
            fields={"invoice_number": "INV-MOCK-LI"},
            confidence_scores={"invoice_number": 0.98},
            line_items=mock_line_items,
            processing_time=0.25,
            metadata={"pages_processed": 1, "total_line_items": 1},
        )

        try:
            with patch.object(main.get_layoutlm(), "extract_pages", return_value=mock_result):
                api_response = main.extract_with_layoutlm(doc_id)

            # API response should contain line_items at top level
            self.assertIn("line_items", api_response)
            self.assertEqual(len(api_response["line_items"]), 1)
            self.assertEqual(api_response["line_items"][0]["description"], "Item Alpha")

            # Database persistence check via get_document()
            doc, _ = database.get_document(doc_id)
            self.assertIn("line_items", doc)
            self.assertEqual(len(doc["line_items"]), 1)
            self.assertEqual(doc["line_items"][0]["item_total"], 100.0)

            # Public document serialization check
            pub = main._public_document(doc)
            self.assertIn("line_items", pub)
            self.assertEqual(pub["line_items"][0]["quantity"], 2.0)
        finally:
            main.remove_document(doc_id)

    def test_40_line_items_backward_compatibility_empty(self):
        """Verify documents without table rows return empty line_items without error."""
        extractor = LayoutLMv3Extractor()
        # Empty entities list
        items = extractor._extract_table_line_items([], page_number=1)
        self.assertEqual(items, [])

        # Non-table entities only (e.g. only header, date, vendor)
        non_table_ents = [
            {"label": "SELLER", "text": "Acme Corp", "box": [50, 50, 200, 70], "confidence": 0.95},
            {"label": "DATE", "text": "2024-01-15", "box": [400, 50, 480, 70], "confidence": 0.96},
        ]
        items2 = extractor._extract_table_line_items(non_table_ents, page_number=1)
        self.assertEqual(items2, [])

    # ── 13. Priority 4: Date Normalization & Financial Validation ───

    def test_41_date_normalization_valid_formats(self):
        """Verify date normalization for all required and standard formats into ISO-8601."""
        from app.validation import normalize_date

        cases = [
            ("18/11/2026", "2026-11-18"),
            ("18-11-2026", "2026-11-18"),
            ("2026-11-18", "2026-11-18"),
            ("Nov 18, 2026", "2026-11-18"),
            ("18 November 2026", "2026-11-18"),
            ("2026/11/18", "2026-11-18"),
            ("18.11.2026", "2026-11-18"),
            ("18-Nov-2026", "2026-11-18"),
            ("November 18, 2026", "2026-11-18"),
            ("2026-Nov-18", "2026-11-18"),
            ("11/18/2026", "2026-11-18"),
            ("18th November 2026", "2026-11-18"),
            ("05/05/2026", "2026-05-05"),
        ]

        for raw_val, expected_iso in cases:
            res = normalize_date(raw_val)
            self.assertTrue(res.is_valid, f"Failed validity for {raw_val}")
            self.assertFalse(res.is_ambiguous, f"Unexpected ambiguity for {raw_val}")
            self.assertEqual(res.normalized, expected_iso, f"Mismatch for {raw_val}")
            self.assertEqual(res.raw, raw_val)

    def test_42_date_normalization_ambiguous(self):
        """Verify ambiguous dates (e.g. 03/04/2026) are flagged without guessing unless format specified."""
        from app.validation import normalize_date

        # Default without format preference: must NOT guess silently
        ambig_res = normalize_date("03/04/2026")
        self.assertTrue(ambig_res.is_ambiguous)
        self.assertIsNone(ambig_res.normalized)
        self.assertFalse(ambig_res.is_valid)
        self.assertIn("2026-04-03", ambig_res.possible_dates)
        self.assertIn("2026-03-04", ambig_res.possible_dates)
        self.assertIn("ambiguous", (ambig_res.error or "").lower())

        # Resolved with explicit DMY convention
        dmy_res = normalize_date("03/04/2026", format_preference="DMY")
        self.assertTrue(dmy_res.is_ambiguous)
        self.assertTrue(dmy_res.is_valid)
        self.assertEqual(dmy_res.normalized, "2026-04-03")

        # Resolved with explicit MDY convention
        mdy_res = normalize_date("03/04/2026", format_preference="MDY")
        self.assertTrue(mdy_res.is_ambiguous)
        self.assertTrue(mdy_res.is_valid)
        self.assertEqual(mdy_res.normalized, "2026-03-04")

    def test_43_date_normalization_invalid_and_missing(self):
        """Verify calendar validation rejects impossible dates and preserves missing dates without hallucination."""
        from app.validation import normalize_date

        # Impossible calendar date: Feb 31
        feb31 = normalize_date("31/02/2026")
        self.assertFalse(feb31.is_valid)
        self.assertIsNone(feb31.normalized)
        self.assertFalse(feb31.is_ambiguous)
        self.assertIsNotNone(feb31.error)

        # Non-leap year Feb 29
        feb29_nonleap = normalize_date("29/02/2025")
        self.assertFalse(feb29_nonleap.is_valid)
        self.assertIsNone(feb29_nonleap.normalized)

        # Leap year Feb 29
        feb29_leap = normalize_date("29/02/2024")
        self.assertTrue(feb29_leap.is_valid)
        self.assertEqual(feb29_leap.normalized, "2024-02-29")

        # Missing dates: must not invent dates
        missing_none = normalize_date(None)
        self.assertTrue(missing_none.is_valid)
        self.assertIsNone(missing_none.normalized)
        self.assertIsNone(missing_none.raw)

        missing_empty = normalize_date("")
        self.assertTrue(missing_empty.is_valid)
        self.assertIsNone(missing_empty.normalized)

        missing_not_found = normalize_date("Not found")
        self.assertTrue(missing_not_found.is_valid)
        self.assertIsNone(missing_not_found.normalized)

    def test_44_financial_validation_consistent_and_tolerance(self):
        """Verify mathematical consistency check and configurable rounding tolerance."""
        from app.validation import validate_financials

        # 1. Standard matching: subtotal + tax = total
        f1 = {
            "subtotal": "100.00",
            "tax_amount": "10.00",
            "total_amount": "110.00",
        }
        res1 = validate_financials(f1, tolerance=0.05)
        self.assertTrue(res1.is_valid)
        self.assertFalse(res1.suspicious)
        self.assertEqual(res1.calculated_total, 110.0)
        self.assertEqual(res1.difference, 0.0)
        self.assertEqual(len(res1.validation_errors), 0)

        # 2. Matching with discount and shipping: subtotal + tax - discount + shipping = total
        f2 = {
            "subtotal": "200.00",
            "tax_amount": "20.00",
            "discount": "30.00",
            "shipping_amount": "15.00",
            "total_amount": "205.00",
        }
        res2 = validate_financials(f2, tolerance=0.05)
        self.assertTrue(res2.is_valid)
        self.assertFalse(res2.suspicious)
        self.assertEqual(res2.calculated_total, 205.0)
        self.assertEqual(res2.difference, 0.0)

        # 3. Rounding difference within tolerance (diff=0.02 <= 0.05)
        f3 = {
            "subtotal": "99.98",
            "tax_amount": "10.00",
            "total_amount": "110.00",
        }
        res3 = validate_financials(f3, tolerance=0.05)
        self.assertTrue(res3.is_valid)
        self.assertFalse(res3.suspicious)
        self.assertEqual(res3.calculated_total, 109.98)
        self.assertEqual(res3.difference, 0.02)

    def test_45_financial_validation_mismatches_and_line_items_sum(self):
        """Verify detection of invoice total mismatch and line-item sum mismatch."""
        from app.validation import validate_financials

        # 1. Total mismatch: subtotal (100) + tax (10) != total (150)
        f1 = {
            "subtotal": "100.00",
            "tax_amount": "10.00",
            "total_amount": "150.00",
        }
        res1 = validate_financials(f1, tolerance=0.05)
        self.assertFalse(res1.is_valid)
        self.assertTrue(res1.suspicious)
        self.assertEqual(res1.calculated_total, 110.0)
        self.assertEqual(res1.difference, 40.0)
        self.assertTrue(any("mismatch" in e.lower() for e in res1.validation_errors))

        # 2. Line items sum matching subtotal
        items_matching = [
            {"item_total": 45.0},
            {"item_total": 55.0},
        ]
        res2 = validate_financials({"subtotal": "100.00", "total_amount": "100.00"}, line_items=items_matching)
        self.assertTrue(res2.is_valid)
        self.assertFalse(res2.suspicious)
        self.assertEqual(res2.line_items_sum, 100.0)
        self.assertTrue(res2.line_items_consistent)

        # 3. Line items sum NOT matching subtotal
        items_mismatch = [
            {"item_total": 30.0},
            {"item_total": 40.0},
        ]
        res3 = validate_financials({"subtotal": "100.00", "total_amount": "100.00"}, line_items=items_mismatch)
        self.assertFalse(res3.is_valid)
        self.assertTrue(res3.suspicious)
        self.assertEqual(res3.line_items_sum, 70.0)
        self.assertFalse(res3.line_items_consistent)
        self.assertTrue(any("Line items sum" in e for e in res3.validation_errors))

    def test_46_financial_validation_missing_optional_fields(self):
        """Verify missing optional financial fields are not flagged as errors."""
        from app.validation import validate_financials

        # Only total amount present (no subtotal, tax, discount, or line items)
        f_minimal = {"total_amount": "500.00"}
        res = validate_financials(f_minimal)
        self.assertTrue(res.is_valid)
        self.assertFalse(res.suspicious)
        self.assertIsNone(res.calculated_total)
        self.assertIsNone(res.difference)
        self.assertEqual(len(res.validation_errors), 0)

        # Corrupted numeric field is flagged as error
        f_corrupt = {"total_amount": "CORRUPTED_MONEY_TXT"}
        res_corrupt = validate_financials(f_corrupt)
        self.assertFalse(res_corrupt.is_valid)
        self.assertTrue(res_corrupt.suspicious)
        self.assertTrue(any("Invalid numeric format" in e for e in res_corrupt.validation_errors))

    def test_47_idempotent_database_initialization(self):
        """Verify repeated calls to init_db() are strictly idempotent and do not fail or duplicate columns."""
        # Call init_db() 5 times consecutively
        for _ in range(5):
            database.init_db()

        # Verify documents table can still be queried cleanly
        stats = database.dashboard_stats()
        self.assertIn("total_documents", stats)

    def test_48_validation_api_endpoint_and_persistence(self):
        """Verify validation results are returned via API and persisted in SQLite."""
        doc_id = "test_validation_persistence_doc"
        database.delete_document(doc_id)

        from PIL import Image
        img = Image.new("RGB", (300, 300), color="white")
        dest = main.UPLOAD_DIR / f"{doc_id}.png"
        img.save(dest)

        record = {
            "id": doc_id,
            "filename": "validated_invoice.png",
            "stored_filename": dest.name,
            "file_path": str(dest),
            "file_size": 500,
            "content_type": "image/png",
            "document_type": "Invoice",
            "uploaded_at": main._now(),
            "processing_status": "processed",
            "extracted_fields": {},
            "invoice_fields": {},
            "extracted_text": "Sample Invoice",
            "confidence_score": 0.0,
            "confidence_method": "baseline",
            "verified": False,
            "verified_at": None,
            "processing_error": None,
            "ground_truth": None,
            "page_count": 1,
            "pages_processed": 1,
            "text_source": "tesseract",
            "message": "Ready",
        }
        database.insert_document(record)

        mock_validation = {
            "is_valid": True,
            "suspicious": False,
            "validation_errors": [],
            "financials": {
                "subtotal": 100.0,
                "tax_amount": 10.0,
                "total_amount": 110.0,
                "calculated_total": 110.0,
                "difference": 0.0,
            },
            "dates": {
                "invoice_date": {
                    "raw": "18/11/2026",
                    "normalized": "2026-11-18",
                    "is_valid": True,
                    "is_ambiguous": False,
                }
            },
        }

        mock_result = ExtractionResult(
            method="layoutlmv3_fatura2",
            fields={"invoice_date": "2026-11-18", "raw_invoice_date": "18/11/2026", "total_amount": "110.00"},
            confidence_scores={"total_amount": 0.98},
            line_items=[],
            validation=mock_validation,
            processing_time=0.3,
            metadata={"pages_processed": 1, "validation": mock_validation},
        )

        from unittest.mock import patch
        try:
            with patch.object(main.get_layoutlm(), "extract_pages", return_value=mock_result):
                api_response = main.extract_with_layoutlm(doc_id)

            # API response should contain validation object
            self.assertIn("validation", api_response)
            self.assertTrue(api_response["validation"]["is_valid"])
            self.assertEqual(api_response["validation"]["financials"]["calculated_total"], 110.0)

            # Database persistence check via get_document()
            doc, _ = database.get_document(doc_id)
            self.assertIn("validation_results", doc)
            self.assertIsNotNone(doc["validation_results"])
            self.assertTrue(doc["validation_results"]["is_valid"])

            # Public document serialization check
            pub = main._public_document(doc)
            self.assertIn("validation_results", pub)
            self.assertTrue(pub["validation_results"]["is_valid"])
        finally:
            main.remove_document(doc_id)



if __name__ == "__main__":
    unittest.main()
