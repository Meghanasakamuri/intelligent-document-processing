"""End-to-end evaluation harness for FATURA2 production invoice extraction.

Runs the complete production pipeline (OCR + LayoutLMv3 + Field Mapping + Validation)
on original invoice images and compares against verified ground-truth annotations.
"""

import sys
import os
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

# Ensure backend modules are importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))

from PIL import Image
from app.models.layoutlm_extractor import LayoutLMv3Extractor
from app.validation import normalize_date


def parse_ground_truth(gt_path: Path) -> Dict[str, Any]:
    """Parse canonical ground-truth fields from FATURA2 Original_Format annotation."""
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt: Dict[str, Any] = {
        "raw": data,
        "vendor_name": None,
        "customer_name": None,
        "invoice_number": None,
        "po_number": None,
        "invoice_date": None,
        "due_date": None,
        "subtotal": None,
        "tax_amount": None,
        "discount": None,
        "total_amount": None,
        "currency": None,
        "table_bbox": None,
        "full_text": data.get("OTHER", {}).get("text", "") if isinstance(data.get("OTHER"), dict) else "",
    }

    # 1. Vendor Name
    if "SELLER_NAME" in data and isinstance(data["SELLER_NAME"], dict):
        gt["vendor_name"] = data["SELLER_NAME"].get("text", "").strip()
    elif "SELLER" in data and isinstance(data["SELLER"], dict):
        s = data["SELLER"].get("text", "").strip()
        gt["vendor_name"] = s.split("\n")[0].strip()
    elif "SELLER_SITE" in data and isinstance(data["SELLER_SITE"], dict):
        site = data["SELLER_SITE"].get("text", "").strip()
        domain = re.sub(r"^(?:https?://)?(?:www\.)?", "", site).split(".")[0]
        # CamelCase splitting
        spaced = re.sub(r"([a-zA-Z])(and)([A-Z])", r"\1 \2 \3", domain, flags=re.IGNORECASE)
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", spaced)
        gt["vendor_name"] = spaced

    # 2. Customer Name
    for buyer_key in ["BUYER", "BILL_TO"]:
        if buyer_key in data and isinstance(data[buyer_key], dict):
            btxt = data[buyer_key].get("text", "").strip()
            lines = [line.strip() for line in btxt.split("\n") if line.strip()]
            for line in lines:
                name_cand = re.sub(r"^(?:Bill[_\s]*to|Customer|Buyer|To)[:\s]*", "", line, flags=re.IGNORECASE).strip()
                if name_cand and name_cand.upper() not in ["BILL_TO", "BILL TO", "BUYER", "CUSTOMER"]:
                    gt["customer_name"] = name_cand
                    break
            if gt.get("customer_name"):
                break

    # 3. Invoice Number
    if "NUMBER" in data and isinstance(data["NUMBER"], dict):
        ntxt = data["NUMBER"].get("text", "").strip()
        inv_no = re.sub(
            r"^(?:INVOICE\s*(?:ID|NUMBER|NO|#|:|-)|INV\s*(?:ID|NUMBER|NO|#|:|-)|BILL\s*(?:NUMBER|NO|#|:|-)|NUMBER|NO|#|:|\s|-)+",
            "",
            ntxt,
            flags=re.IGNORECASE,
        ).strip(" :-#")
        gt["invoice_number"] = inv_no if inv_no else None

    # 4. PO Number
    if "PO_NUMBER" in data and isinstance(data["PO_NUMBER"], dict):
        ptxt = data["PO_NUMBER"].get("text", "").strip()
        po_no = re.sub(
            r"^(?:PO\s*(?:NUMBER|NO|#|:|-)|P\.O\.\s*(?:NUMBER|NO|#|:|-)|ORDER\s*(?:NUMBER|NO|#|:|-)|NUMBER|NO|#|:|\s|-)+",
            "",
            ptxt,
            flags=re.IGNORECASE,
        ).strip(" :-#")
        gt["po_number"] = po_no if po_no else None

    # 5. Dates
    date_regex = re.compile(r"(\d{1,4}[-/.][A-Za-z]{3,9}[-/.]\d{2,4}|\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4})")
    if "DATE" in data and isinstance(data["DATE"], dict):
        m = date_regex.search(data["DATE"].get("text", ""))
        if m:
            res = normalize_date(m.group(1))
            gt["invoice_date"] = res.normalized

    if "DUE_DATE" in data and isinstance(data["DUE_DATE"], dict):
        m = date_regex.search(data["DUE_DATE"].get("text", ""))
        if m:
            res = normalize_date(m.group(1))
            gt["due_date"] = res.normalized

    # 6. Financials
    amt_regex = re.compile(r"[-+]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})|[-+]?[0-9]+\.[0-9]{1,2}")
    if "TOTAL" in data and isinstance(data["TOTAL"], dict):
        ttxt = data["TOTAL"].get("text", "").strip()
        m = amt_regex.search(ttxt)
        if m:
            gt["total_amount"] = m.group(0).replace(",", "")
        if "EUR" in ttxt.upper() or "€" in ttxt:
            gt["currency"] = "EUR"
        elif "USD" in ttxt.upper() or "$" in ttxt:
            gt["currency"] = "USD"
        elif "GBP" in ttxt.upper() or "£" in ttxt:
            gt["currency"] = "GBP"

    if "SUB_TOTAL" in data and isinstance(data["SUB_TOTAL"], dict):
        m = amt_regex.search(data["SUB_TOTAL"].get("text", ""))
        if m:
            gt["subtotal"] = m.group(0).replace(",", "")

    # Tax (TAX or GST)
    for tax_key in ["TAX", "GST(9%)", "VAT"]:
        if tax_key in data and isinstance(data[tax_key], dict):
            tax_clean = re.sub(r"\([^)]*%\)", "", data[tax_key].get("text", ""))
            tax_clean = re.sub(r"\b\d+(?:\.\d+)?%", "", tax_clean)
            m = amt_regex.search(tax_clean)
            if m:
                gt["tax_amount"] = m.group(0).replace(",", "")
                break

    if "DISCOUNT" in data and isinstance(data["DISCOUNT"], dict):
        disc_clean = re.sub(r"\([^)]*%\)", "", data["DISCOUNT"].get("text", ""))
        disc_clean = re.sub(r"\b\d+(?:\.\d+)?%", "", disc_clean)
        m = amt_regex.search(disc_clean)
        if m:
            gt["discount"] = m.group(0).replace(",", "")

    # 7. Table BBox
    if "TABLE" in data and isinstance(data["TABLE"], list) and len(data["TABLE"]) > 0:
        first_row = data["TABLE"][0]
        if isinstance(first_row, list) and len(first_row) > 0 and isinstance(first_row[0], dict):
            gt["table_bbox"] = first_row[0].get("bbox")

    return gt


def evaluate_field_match(
    field_name: str,
    extracted_val: Any,
    gt_val: Any,
    raw_ocr_text: str = "",
) -> Tuple[bool, str]:
    """Compare extracted value against ground truth and return (is_match, failure_reason)."""
    if gt_val is None:
        if extracted_val is None:
            return True, "both_none"
        return False, "spurious_extraction"

    if extracted_val is None:
        # Check if GT text appears in raw OCR
        gt_str = str(gt_val).strip()
        if gt_str.lower() in raw_ocr_text.lower():
            return False, "model_prediction_or_mapping"
        return False, "ocr_missing"

    s_ext = str(extracted_val).strip()
    s_gt = str(gt_val).strip()

    # Exact string match
    if s_ext.lower() == s_gt.lower():
        return True, "exact_match"

    # Numeric match (for financial fields)
    if field_name in ["total_amount", "subtotal", "tax_amount", "discount"]:
        try:
            f_ext = float(s_ext.replace(",", ""))
            f_gt = float(s_gt.replace(",", ""))
            if abs(f_ext - f_gt) <= 0.05:
                return True, "numeric_match"
            return False, f"amount_mismatch (got {f_ext}, expected {f_gt})"
        except ValueError:
            return False, f"unparseable_number (got {s_ext})"

    # Date match
    if field_name in ["invoice_date", "due_date"]:
        norm_ext = normalize_date(s_ext).normalized
        norm_gt = normalize_date(s_gt).normalized
        if norm_ext and norm_gt and norm_ext == norm_gt:
            return True, "date_match"
        return False, f"date_mismatch (got {norm_ext}, expected {norm_gt})"

    # Substring / partial match for long vendor or customer names
    if field_name in ["vendor_name", "customer_name"]:
        c_ext = re.sub(r"[^\w\s]", "", s_ext).strip().lower()
        c_gt = re.sub(r"[^\w\s]", "", s_gt).strip().lower()
        if c_ext == c_gt:
            return True, "normalized_name_match"
        if len(c_ext) >= 4 and (c_ext in c_gt or c_gt in c_ext):
            return True, "name_substring_match"
        return False, f"name_mismatch (got '{s_ext}', expected '{s_gt}')"

    # Identifier / invoice number match
    if field_name in ["invoice_number", "po_number"]:
        clean_ext = re.sub(r"[\s:-]", "", s_ext).lower()
        clean_gt = re.sub(r"[\s:-]", "", s_gt).lower()
        if clean_ext == clean_gt:
            return True, "identifier_match"
        return False, f"identifier_mismatch (got '{s_ext}', expected '{s_gt}')"

    return False, f"mismatch (got '{s_ext}', expected '{s_gt}')"


def run_evaluation(
    eval_dir: Path = Path("evaluation/fatura2_test50"),
    report_tag: str = "baseline",
) -> Dict[str, Any]:
    """Run full production evaluation on the 50 test invoices."""
    img_dir = eval_dir / "images"
    gt_dir = eval_dir / "ground_truth"
    manifest_file = eval_dir / "gt_download_list.json"

    with open(manifest_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"\n=======================================================")
    print(f"RUNNING FATURA2 END-TO-END EVALUATION ({report_tag.upper()})")
    print(f"Evaluating {len(items)} invoices from unseen templates 44-50")
    print(f"=======================================================\n")

    extractor = LayoutLMv3Extractor()
    extractor.load()

    target_fields = [
        "vendor_name",
        "customer_name",
        "invoice_number",
        "po_number",
        "invoice_date",
        "due_date",
        "subtotal",
        "tax_amount",
        "discount",
        "total_amount",
        "currency",
    ]

    field_stats = {
        f: {
            "gt_present": 0,
            "correct": 0,
            "incorrect": 0,
            "missing": 0,
            "spurious": 0,
            "derived_correct": 0,
            "direct_correct": 0,
            "failure_reasons": defaultdict(int),
        }
        for f in target_fields
    }

    # Error analysis buckets
    error_causes = defaultdict(int)

    invoice_results = []
    total_invoices = len(items)
    perfect_invoices = 0
    total_line_items_extracted = 0

    ocr_word_precision_list = []
    ocr_word_recall_list = []

    start_time = time.time()

    for idx, item in enumerate(items):
        inv_id = item["invoice_id"]
        tpl = item["template"]
        img_path = Path(item["image_file"])
        gt_path = Path(item["out_file"])

        if not img_path.exists() or not gt_path.exists():
            print(f"[{idx+1}/{total_invoices}] Missing file for {inv_id}, skipping")
            continue

        gt = parse_ground_truth(gt_path)

        img = Image.open(img_path).convert("RGB")
        # Run production OCR and model inference
        t0 = time.time()
        res = extractor.extract(img, psm=3)
        proc_time = round(time.time() - t0, 3)

        raw_ocr = res.raw_text or ""
        field_sources = res.metadata.get("field_sources", {})

        # 1. OCR word-level recall against GT full text
        gt_full_text = gt["full_text"]
        if gt_full_text:
            gt_words = set(re.findall(r"\b[A-Za-z0-9_/-]{2,}\b", gt_full_text.lower()))
            ocr_words = set(re.findall(r"\b[A-Za-z0-9_/-]{2,}\b", raw_ocr.lower()))
            if gt_words:
                rec = len(gt_words.intersection(ocr_words)) / len(gt_words)
                ocr_word_recall_list.append(rec)
            if ocr_words:
                prec = len(gt_words.intersection(ocr_words)) / len(ocr_words)
                ocr_word_precision_list.append(prec)

        # 2. Field-level matching
        inv_record = {
            "invoice_id": inv_id,
            "template": tpl,
            "processing_time": proc_time,
            "fields": {},
            "all_fields_correct": True,
            "line_items_count": len(res.line_items),
        }

        total_line_items_extracted += len(res.line_items)

        for fld in target_fields:
            ext_val = res.fields.get(fld)
            gt_val = gt.get(fld)
            src = field_sources.get(fld, "unknown")

            is_match, reason = evaluate_field_match(fld, ext_val, gt_val, raw_ocr_text=raw_ocr)

            inv_record["fields"][fld] = {
                "extracted": ext_val,
                "ground_truth": gt_val,
                "is_match": is_match,
                "source": src,
                "reason": reason,
            }

            stats = field_stats[fld]

            if gt_val is not None:
                stats["gt_present"] += 1
                if is_match:
                    stats["correct"] += 1
                    if src == "derived":
                        stats["derived_correct"] += 1
                    else:
                        stats["direct_correct"] += 1
                else:
                    inv_record["all_fields_correct"] = False
                    if ext_val is None:
                        stats["missing"] += 1
                        error_causes[f"missing_{fld}:{reason}"] += 1
                    else:
                        stats["incorrect"] += 1
                        error_causes[f"incorrect_{fld}:{reason}"] += 1
                    stats["failure_reasons"][reason] += 1
            else:
                if ext_val is not None:
                    stats["spurious"] += 1
                    # Spurious is not penalized if it's derived or harmless, but noted

        if inv_record["all_fields_correct"]:
            perfect_invoices += 1

        invoice_results.append(inv_record)

        if (idx + 1) % 10 == 0 or (idx + 1) == total_invoices:
            print(f"  Processed {idx + 1}/{total_invoices} invoices ({proc_time}s/doc)")

    elapsed = round(time.time() - start_time, 2)

    # 3. Compute aggregate metrics
    avg_ocr_recall = round(sum(ocr_word_recall_list) / len(ocr_word_recall_list) * 100, 2) if ocr_word_recall_list else 0.0
    avg_ocr_precision = round(sum(ocr_word_precision_list) / len(ocr_word_precision_list) * 100, 2) if ocr_word_precision_list else 0.0

    summary_metrics = {
        "evaluation_tag": report_tag,
        "total_invoices": len(invoice_results),
        "total_time_seconds": elapsed,
        "avg_time_per_invoice": round(elapsed / max(1, len(invoice_results)), 3),
        "ocr_word_recall_pct": avg_ocr_recall,
        "ocr_word_precision_pct": avg_ocr_precision,
        "complete_invoice_exact_match_pct": round(perfect_invoices / max(1, len(invoice_results)) * 100, 2),
        "total_line_items_extracted": total_line_items_extracted,
        "avg_line_items_per_invoice": round(total_line_items_extracted / max(1, len(invoice_results)), 2),
        "field_accuracies": {},
        "error_root_causes": dict(sorted(error_causes.items(), key=lambda x: x[1], reverse=True)),
    }

    print("\n-------------------------------------------------------")
    print(f"ACCURACY BENCHMARK SUMMARY ({report_tag.upper()}):")
    print("-------------------------------------------------------")
    print(f"Complete Invoice Exact Match: {summary_metrics['complete_invoice_exact_match_pct']}% ({perfect_invoices}/{len(invoice_results)})")
    print(f"OCR Word Recall: {avg_ocr_recall}%, Word Precision: {avg_ocr_precision}%")
    print(f"Total Line Items Extracted: {total_line_items_extracted}\n")

    print(f"{'Field':20s} | {'Support':7s} | {'Correct':7s} | {'Missing':7s} | {'Incorrect':9s} | {'Accuracy':8s} | {'Direct/Derived'}")
    print("-" * 85)

    for fld in target_fields:
        st = field_stats[fld]
        supp = st["gt_present"]
        corr = st["correct"]
        acc = round((corr / supp * 100), 2) if supp > 0 else 0.0
        summary_metrics["field_accuracies"][fld] = {
            "support": supp,
            "correct": corr,
            "missing": st["missing"],
            "incorrect": st["incorrect"],
            "accuracy_pct": acc,
            "direct_correct": st["direct_correct"],
            "derived_correct": st["derived_correct"],
            "failure_reasons": dict(st["failure_reasons"]),
        }
        print(f"{fld:20s} | {supp:7d} | {corr:7d} | {st['missing']:7d} | {st['incorrect']:9d} | {acc:7.2f}% | {st['direct_correct']} direct / {st['derived_correct']} derived")

    print("-" * 85)

    # 4. Save results to reports directory
    rep_dir = Path(f"evaluation/reports/fatura2_end_to_end")
    rep_dir.mkdir(parents=True, exist_ok=True)

    with open(rep_dir / f"fatura2_{report_tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    with open(rep_dir / f"fatura2_{report_tag}_invoices.json", "w", encoding="utf-8") as f:
        json.dump(invoice_results, f, indent=2)

    print(f"\nSaved evaluation reports to {rep_dir / f'fatura2_{report_tag}_summary.json'}")

    return summary_metrics


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    run_evaluation(report_tag=tag)
