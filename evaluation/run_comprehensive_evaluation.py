"""Comprehensive Invoice Field Accuracy Evaluation Runner.

Evaluates BOTH:
1. Indian GST Invoices (31 invoices from evaluation/invoice_evaluation)
2. FATURA2 Synthetic Invoices (50 held-out invoices from evaluation/fatura2_test50)

Both datasets are processed through the EXACT real application workflow:
- POST /upload -> ingest_document()
- Token classification & statutory/spatial extraction -> _fuse_hybrid_fields()
- Financial validation -> validate_financials()
- Public document extraction -> _public_document()
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "backend"))

from app import database, main
from app.validation import normalize_date, parse_numeric


def evaluate_indian_gst_invoices() -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("RUNNING INDIAN GST INVOICE EVALUATION (31 INVOICES)")
    print("Application Workflow: ingest_document -> LayoutLMv3 + Native PyMuPDF + Hybrid Fusion")
    print("=" * 75 + "\n")

    annotations_dir = BASE_DIR / "evaluation" / "annotations"
    invoices_dir = BASE_DIR / "evaluation" / "invoice_evaluation"

    # Locate all invoice annotations
    ann_files = sorted(annotations_dir.glob("invoice_*_page_1.json"))
    if not ann_files:
        raise FileNotFoundError(f"No annotation files found in {annotations_dir}")

    eval_fields = [
        "vendor_name",
        "customer_name",
        "vendor_gstin",
        "customer_gstin",
        "invoice_number",
        "invoice_date",
        "subtotal",
        "cgst_amount",
        "sgst_amount",
        "tax_amount",
        "total_amount",
    ]

    stats = {
        f: {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "missing": 0,
            "accuracy_pct": 0.0,
        }
        for f in eval_fields
    }

    per_invoice_results = []
    database.init_db()

    for ann_file in ann_files:
        with open(ann_file, "r", encoding="utf-8") as f:
            ann_data = json.load(f)

        orig_filename = ann_data.get("original_filename")
        pdf_path = invoices_dir / orig_filename
        if not pdf_path.exists():
            continue

        supported = ann_data.get("supported_fields", {})
        missing_entries = {m["field_name"]: m for m in ann_data.get("missing_fields", [])}

        # Build ground-truth dictionary
        gt: Dict[str, Any] = {
            "vendor_name": supported.get("vendor_name"),
            "customer_name": supported.get("customer_name"),
            "invoice_number": supported.get("invoice_number"),
            "invoice_date": supported.get("invoice_date"),
            "subtotal": supported.get("subtotal"),
            "cgst_amount": supported.get("cgst_amount"),
            "sgst_amount": supported.get("sgst_amount"),
            "tax_amount": supported.get("tax_amount"),
            "total_amount": supported.get("total_amount"),
            "vendor_gstin": None,
            "customer_gstin": None,
        }

        # GST numbers from missing_fields
        if "gst_number" in missing_entries:
            vals = missing_entries["gst_number"].get("values", [])
            if vals:
                gt["vendor_gstin"] = vals[0]
            if len(vals) > 1:
                gt["customer_gstin"] = vals[1]

        # Execute application workflow
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        ingest_res = main.ingest_document(orig_filename, pdf_bytes)
        pred_fields = ingest_res.get("invoice_fields", {})

        invoice_eval = {
            "filename": orig_filename,
            "fields": {},
            "validation_is_valid": ingest_res.get("validation_results", {}).get("is_valid"),
        }

        for f in eval_fields:
            gt_val = gt.get(f)
            pred_val = pred_fields.get(f)

            if gt_val is None:
                continue

            stats[f]["total"] += 1
            is_match = False
            match_reason = ""

            if pred_val is None or str(pred_val).strip() in ("", "None", "Not found"):
                stats[f]["missing"] += 1
                match_reason = "missing"
            else:
                s_pred = str(pred_val).strip()
                s_gt = str(gt_val).strip()

                if f in ("subtotal", "cgst_amount", "sgst_amount", "tax_amount", "total_amount"):
                    p_num = parse_numeric(s_pred)
                    g_num = parse_numeric(s_gt)
                    if p_num is not None and g_num is not None and abs(p_num - g_num) <= 0.05:
                        is_match = True
                    else:
                        match_reason = f"numeric mismatch ({s_pred} vs {s_gt})"
                elif f == "invoice_date":
                    p_norm = normalize_date(s_pred).normalized
                    g_norm = normalize_date(s_gt).normalized
                    if p_norm and g_norm and p_norm == g_norm:
                        is_match = True
                    elif s_pred == s_gt or s_pred in s_gt or s_gt in s_pred:
                        is_match = True
                    else:
                        match_reason = f"date mismatch ({s_pred} vs {s_gt})"
                elif f in ("vendor_gstin", "customer_gstin"):
                    # Normalize O vs 0 in GSTIN if OCR confused
                    clean_p = s_pred.upper().replace(" ", "")
                    clean_g = s_gt.upper().replace(" ", "")
                    if clean_p == clean_g or clean_p.replace("O", "0") == clean_g.replace("O", "0"):
                        is_match = True
                    else:
                        match_reason = f"gstin mismatch ({s_pred} vs {s_gt})"
                elif f in ("vendor_name", "customer_name"):
                    c_p = s_pred.upper().strip()
                    c_g = s_gt.upper().strip()
                    ratio = SequenceMatcher(None, c_p, c_g).ratio()
                    if c_p == c_g or c_p in c_g or c_g in c_p or ratio >= 0.75:
                        is_match = True
                    else:
                        match_reason = f"string mismatch ({s_pred} vs {s_gt}, ratio {ratio:.2f})"
                elif f == "invoice_number":
                    c_p = s_pred.upper().replace(" ", "")
                    c_g = s_gt.upper().replace(" ", "")
                    if c_p == c_g or c_g in c_p or c_p in c_g:
                        is_match = True
                    else:
                        match_reason = f"inv mismatch ({s_pred} vs {s_gt})"

            if is_match:
                stats[f]["correct"] += 1
            elif match_reason != "missing":
                stats[f]["incorrect"] += 1

            invoice_eval["fields"][f] = {
                "gt": gt_val,
                "pred": pred_val,
                "match": is_match,
                "reason": match_reason,
            }

        per_invoice_results.append(invoice_eval)

    # Compute percentage accuracies
    total_eval_points = sum(v["total"] for v in stats.values())
    total_correct = sum(v["correct"] for v in stats.values())
    overall_accuracy = round((total_correct / total_eval_points * 100.0) if total_eval_points else 0.0, 1)

    for f in eval_fields:
        t = stats[f]["total"]
        c = stats[f]["correct"]
        stats[f]["accuracy_pct"] = round((c / t * 100.0) if t else 0.0, 1)

    report = {
        "dataset": "Indian GST Invoices",
        "total_invoices": len(per_invoice_results),
        "total_fields_evaluated": total_eval_points,
        "overall_accuracy_pct": overall_accuracy,
        "field_statistics": stats,
        "per_invoice_results": per_invoice_results,
    }

    out_file = BASE_DIR / "evaluation" / "indian_gst_repaired_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Results saved to: {out_file}")
    return report


def evaluate_fatura2_dev_invoices() -> Dict[str, Any]:
    from evaluation.run_app_workflow_evaluation import run_app_workflow_evaluation
    eval_dir = BASE_DIR / "evaluation" / "fatura2_test50"
    report = run_app_workflow_evaluation(eval_dir=eval_dir, report_tag="fatura2_dev50_repaired")
    return report


if __name__ == "__main__":
    gst_report = evaluate_indian_gst_invoices()
    print("\n--- INDIAN GST FIELD ACCURACY SUMMARY ---")
    for f, d in gst_report["field_statistics"].items():
        print(f"  {f:18}: {d['correct']:2d} / {d['total']:2d} ({d['accuracy_pct']:5.1f}%)")
    print(f"OVERALL INDIAN GST ACCURACY: {gst_report['overall_accuracy_pct']}%\n")

    fat_report = evaluate_fatura2_dev_invoices()
    print("\n--- FATURA2 FIELD ACCURACY SUMMARY ---")
    for f, d in fat_report.get("field_accuracy", {}).items():
        print(f"  {f:18}: {d['correct']:2d} / {d['total']:2d} ({d['accuracy_pct']:5.1f}%)")
    print(f"OVERALL FATURA2 ACCURACY: {fat_report.get('overall_accuracy_pct', 0.0)}%\n")
