"""Complete Application Workflow Evaluation Harness for FATURA2 Invoices.

Executes the REAL application workflow:
1. Ingest document (POST /upload -> main.ingest_document)
2. Run LayoutLMv3 extraction (POST /documents/{id}/extract/layoutlm -> main.extract_with_layoutlm)
3. Persist to database (SQLite)
4. Retrieve public document (GET /documents/{id} -> main.get_document_by_id)
5. Map fields to frontend review state (DocumentReview.jsx)
6. Compare saved API and UI values against ground truth across all 12 canonical fields.
"""

from __future__ import annotations

import io
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
sys.path.insert(0, str(BASE_DIR / "backend"))

from app import database, main
from app.validation import normalize_date

TARGET_FIELDS = [
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
    "line_items",
]


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
        "line_items_count": 0,
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
            raw_d = m.group(1)
            res = normalize_date(raw_d)
            gt["invoice_date"] = res.normalized or raw_d

    if "DUE_DATE" in data and isinstance(data["DUE_DATE"], dict):
        m = date_regex.search(data["DUE_DATE"].get("text", ""))
        if m:
            raw_d = m.group(1)
            res = normalize_date(raw_d)
            gt["due_date"] = res.normalized or raw_d

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

    # 7. Table and Line Items
    if "TABLE" in data and isinstance(data["TABLE"], list) and len(data["TABLE"]) > 0:
        first_row = data["TABLE"][0]
        if isinstance(first_row, list) and len(first_row) > 0 and isinstance(first_row[0], dict):
            gt["table_bbox"] = first_row[0].get("bbox")

    # Count table line items from OTHER text
    lines = [l.strip() for l in gt["full_text"].split("\n") if l.strip()]
    in_t = False
    items = []
    for l in lines:
        lu = l.upper()
        if any(h in lu for h in ["DESCRIPTION", "ITEMS", "ITEM", "PARTICULARS", "QTY", "QUANTITY"]) and any(h2 in lu for h2 in ["PRICE", "RATE", "AMOUNT", "TAX"]):
            in_t = True
            continue
        if in_t:
            if any(lu.startswith(end) for end in [
                "SUB_TOTAL", "SUBTOTAL", "TOTAL", "TAX:", "TAX :", "TAX ", "GST", "VAT",
                "BALANCE DUE", "BANK NAME", "THANK YOU", "TOTAL IN WORDS", "NOTE:"
            ]):
                in_t = False
                break
            if re.search(r"\d+\.\d{2}", l):
                items.append(l)
    gt["line_items_count"] = len(items) if items else (1 if gt["table_bbox"] else 0)

    return gt


def parse_gt_line_items(gt_data: dict, tpl_name: str) -> list[dict]:
    """Parse structured row-level ground-truth line items across all FATURA2 templates."""
    txt = gt_data.get('OTHER', {}).get('text', '')
    lines = [l.strip() for l in txt.split('\n') if l.strip()]

    # Special handling for column-wise format in Template46
    if tpl_name == 'Template46' and '\nQUANTITY\n' in txt and '\nPRICE\n' in txt:
        items_m = re.search(r'\nITEMS\n(.*?)\n(?:QUANTITY|PRICE|Invoice Date)', txt, re.DOTALL)
        qty_m = re.search(r'\nQUANTITY\n(.*?)\n(?:PRICE|TAX|Bank Name)', txt, re.DOTALL)
        price_m = re.search(r'\nPRICE\n(.*?)\n(?:TAX|Due Date|TOTAL)', txt, re.DOTALL)
        if items_m and qty_m:
            descs = [l.strip(' .') for l in items_m.group(1).strip().split('\n') if l.strip()]
            qtys = [float(re.search(r'\d+(?:\.\d+)?', l).group(0)) for l in qty_m.group(1).strip().split('\n') if re.search(r'\d+(?:\.\d+)?', l)]
            prices = [float(re.search(r'\d+(?:\.\d+)?', l).group(0)) for l in price_m.group(1).strip().split('\n') if re.search(r'\d+(?:\.\d+)?', l)] if price_m else []
            out = []
            for i in range(min(len(descs), len(qtys))):
                p = prices[i] if i < len(prices) else None
                tot = round(qtys[i] * p, 2) if p else None
                out.append({'description': descs[i], 'quantity': qtys[i], 'unit_price': p, 'item_total': tot})
            if out:
                return out

    start_idx = None
    for idx, l in enumerate(lines):
        lu = l.upper()
        if any(h in lu for h in ['DESCRIPTION', 'ITEMS', 'ITEM', 'PARTICULARS', 'QTY', 'QUANTITY']) and any(h2 in lu for h2 in ['PRICE', 'RATE', 'AMOUNT', 'TAX', 'TOTAL', 'ID', '()']):
            start_idx = idx + 1
            break
        if tpl_name == 'Template45' and ('aide' in lu or 'site:' in lu.lower()):
            start_idx = idx + 1
        if tpl_name == 'Template48' and re.match(r'^\d{5,8}\s*[_|]', l):
            start_idx = idx
            break

    if start_idx is None:
        for idx, l in enumerate(lines):
            if len(re.findall(r'\d+\.\d{2}', l)) >= 2 and not any(k in l.upper() for k in ['TEL:', 'DATE:', 'TOTAL', 'SUB_TOTAL']):
                start_idx = idx
                break

    if start_idx is None:
        return []

    items = []
    end_prefixes = ['SUB_TOTAL', 'SUBTOTAL', 'TOTAL :', 'TOTAL:', 'TOTAL AMOUNT', 'TOTAL IN WORDS', 'TAX:', 'TAX :', 'TAX ', 'GST', 'VAT', 'BALANCE DUE', 'BANK NAME', 'NOTE:', 'TERMS AND CONDITIONS']
    for l in lines[start_idx:]:
        lu = l.upper()
        if any(lu.startswith(k) for k in end_prefixes) or "THANK YOU" in lu:
            break

        nums = [float(x.replace('$', '')) for x in re.findall(r'[-+]?\$?\d+\.\d{2}', l)]

        if tpl_name == 'Template48' and re.search(r'\d{5,8}', l):
            m = re.search(r'^\d{5,8}\s*[_|]*\s*(.*?)$', l)
            desc = m.group(1).strip() if m else l
            items.append({'description': desc, 'quantity': None, 'unit_price': None, 'item_total': None})
            continue

        if not nums:
            if tpl_name == 'Template48' and len(l) > 3:
                items.append({'description': l.strip(' _|.-'), 'quantity': None, 'unit_price': None, 'item_total': None})
            continue

        if tpl_name in ['Template44', 'Template45']:
            m = re.match(r'^(.*?)\s+(\d+(?:\.\d+)?)\s+\$?(\d+(?:\.\d+)?)\s+\$?(\d+(?:\.\d+)?)$', l)
            if m:
                desc, q, u, tot = m.groups()
                items.append({'description': desc.strip(), 'quantity': float(q), 'unit_price': float(u), 'item_total': float(tot)})
            elif len(nums) >= 2:
                tot = nums[-1]
                q = nums[0] if len(nums) >= 3 else 1.0
                u = nums[1] if len(nums) >= 3 else nums[0]
                desc = re.sub(r'[-+]?\$?\d+\.\d{2}', '', l).strip(' .')
                items.append({'description': desc, 'quantity': q, 'unit_price': u, 'item_total': tot})
        elif tpl_name == 'Template46':
            m = re.match(r'^(.*?)\s+(\d+(?:\.\d+)?)\s+\$?(\d+(?:\.\d+)?)\s+\$?(\d+(?:\.\d+)?)\s+\$?(\d+(?:\.\d+)?)$', l)
            if m:
                desc, q, u, tx, tot = m.groups()
                items.append({'description': desc.strip(), 'quantity': float(q), 'unit_price': float(u), 'tax_amount': float(tx), 'item_total': float(tot)})
            elif len(nums) == 1:
                desc = re.sub(r'[-+]?\$?\d+(?:\.\d+)?', '', l).strip(' .')
                items.append({'description': desc, 'quantity': nums[0], 'unit_price': None, 'item_total': None})
            else:
                desc = re.sub(r'[-+]?\$?\d+\.\d{2}', '', l).strip(' .')
                items.append({'description': desc, 'quantity': nums[0], 'unit_price': nums[1] if len(nums)>1 else None, 'item_total': nums[-1]})
        elif tpl_name == 'Template47':
            m = re.match(r'^(\d+(?:\.\d+)?)\s+(\d{5,8})\s*\|\s*(.*?)\s+(\d+(?:\.\d+)?)$', l)
            if m:
                q, id_val, desc, tot = m.groups()
                q_f, tot_f = float(q), float(tot)
                items.append({'description': desc.strip(), 'quantity': q_f, 'unit_price': round(tot_f/q_f, 2) if q_f else tot_f, 'item_total': tot_f})
            else:
                desc = re.sub(r'[-+]?\$?\d+\.\d{2}', '', l).strip(' .|')
                items.append({'description': desc, 'quantity': nums[0], 'unit_price': nums[-1], 'item_total': nums[-1]})
        elif tpl_name in ['Template49', 'Template50']:
            m = re.match(r'^(.*?)\s+\$?(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+\$?(\d+(?:\.\d+)?)$', l)
            if m:
                desc, u, q, tot = m.groups()
                items.append({'description': desc.strip(' .|'), 'quantity': float(q), 'unit_price': float(u), 'item_total': float(tot)})
            else:
                desc = re.sub(r'[-+]?\$?\d+\.\d{2}', '', l).strip(' .|')
                items.append({'description': desc, 'quantity': nums[-2] if len(nums)>=2 else 1.0, 'unit_price': nums[-3] if len(nums)>=3 else nums[0], 'item_total': nums[-1]})
        else:
            desc = re.sub(r'[-+]?\$?\d+\.\d{2}', '', l).strip(' .')
            items.append({'description': desc, 'quantity': nums[0] if len(nums)>=2 else 1.0, 'unit_price': nums[-2] if len(nums)>=3 else nums[0], 'item_total': nums[-1]})
    return items


def match_rows(pred_rows: list[dict], gt_rows: list[dict]) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Match predicted rows to ground-truth rows using optimal scoring."""
    if not pred_rows:
        return [], [], gt_rows
    if not gt_rows:
        return [], pred_rows, []

    scores = []
    for p_idx, p in enumerate(pred_rows):
        for g_idx, g in enumerate(gt_rows):
            score = 0.0
            p_tot = p.get("item_total")
            g_tot = g.get("item_total")
            p_desc = (p.get("description") or "").lower().strip(" .|_-")
            g_desc = (g.get("description") or "").lower().strip(" .|_-")
            ratio = SequenceMatcher(None, p_desc, g_desc).ratio()

            if g_tot is None:
                if ratio >= 0.65 or g_desc in p_desc or p_desc in g_desc:
                    score += 0.85
                else:
                    score += 0.40 * ratio
            else:
                # 1. Total match
                if p_tot is not None and abs(p_tot - g_tot) <= 0.05:
                    score += 0.50
                elif p_tot is not None and abs(p_tot - g_tot) <= 0.50:
                    score += 0.30

                # 2. Description match
                score += 0.30 * ratio

                # 3. Quantity match
                p_q = p.get("quantity")
                g_q = g.get("quantity")
                if p_q is not None and g_q is not None:
                    if abs(p_q - g_q) <= 0.01:
                        score += 0.10

                # 4. Price match
                p_u = p.get("unit_price")
                g_u = g.get("unit_price")
                if p_u is not None and g_u is not None:
                    if abs(p_u - g_u) <= 0.05:
                        score += 0.10

            scores.append((score, p_idx, g_idx))

    scores.sort(key=lambda x: x[0], reverse=True)
    matched_pairs = []
    used_p = set()
    used_g = set()

    for s, p_idx, g_idx in scores:
        if p_idx in used_p or g_idx in used_g:
            continue
        if s >= 0.35:
            matched_pairs.append((pred_rows[p_idx], gt_rows[g_idx]))
            used_p.add(p_idx)
            used_g.add(g_idx)

    unmatched_p = [p for i, p in enumerate(pred_rows) if i not in used_p]
    unmatched_g = [g for i, g in enumerate(gt_rows) if i not in used_g]

    return matched_pairs, unmatched_p, unmatched_g


def evaluate_invoice_line_items(pred_items: list[dict], gt_items: list[dict]) -> dict:
    """Evaluate extracted line items against ground truth rows for a single invoice."""
    matched, un_p, un_g = match_rows(pred_items, gt_items)

    n_gt = len(gt_items)
    n_pred = len(pred_items)
    n_matched = len(matched)

    desc_matches = 0
    qty_matches = 0
    price_matches = 0
    total_matches = 0
    row_exact_matches = 0

    for p, g in matched:
        p_desc = (p.get("description") or "").lower().strip(" .|_-")
        g_desc = (g.get("description") or "").lower().strip(" .|_-")
        ratio = SequenceMatcher(None, p_desc, g_desc).ratio()
        desc_ok = bool(ratio >= 0.65 or g_desc in p_desc or p_desc in g_desc or (not g_desc and not p_desc))

        p_q = p.get("quantity")
        g_q = g.get("quantity")
        qty_ok = bool((p_q is None and g_q is None) or (p_q is not None and g_q is not None and abs(p_q - g_q) <= 0.05))

        p_u = p.get("unit_price")
        g_u = g.get("unit_price")
        price_ok = bool((p_u is None and g_u is None) or (p_u is not None and g_u is not None and abs(p_u - g_u) <= 0.05))

        p_tot = p.get("item_total")
        g_tot = g.get("item_total")
        tot_ok = bool((p_tot is None and g_tot is None) or (p_tot is not None and g_tot is not None and abs(p_tot - g_tot) <= 0.05))

        if desc_ok:
            desc_matches += 1
        if qty_ok:
            qty_matches += 1
        if price_ok:
            price_matches += 1
        if tot_ok:
            total_matches += 1
        if desc_ok and qty_ok and price_ok and tot_ok:
            row_exact_matches += 1

    invoice_exact = bool(n_gt > 0 and n_matched == n_gt and len(un_p) == 0 and row_exact_matches == n_gt)

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_matched": n_matched,
        "desc_matches": desc_matches,
        "qty_matches": qty_matches,
        "price_matches": price_matches,
        "total_matches": total_matches,
        "row_exact_matches": row_exact_matches,
        "invoice_exact": invoice_exact,
        "table_detected": n_pred > 0,
        "matched": matched,
        "unmatched_pred": un_p,
        "unmatched_gt": un_g,
    }


def simulate_frontend_review_state(api_doc: dict) -> dict:
    """Replicate frontend/src/pages/DocumentReview.jsx field mapping logic."""
    is_layoutlm = (
        api_doc.get("extraction_method") == "layoutlm"
        or bool(api_doc.get("layoutlm_fields") and len(api_doc["layoutlm_fields"]) > 0)
    )
    source_fields = (
        api_doc.get("layoutlm_fields")
        if is_layoutlm and api_doc.get("layoutlm_fields")
        else (api_doc.get("invoice_fields") or api_doc.get("extracted_fields") or {})
    )
    initial_fields = (
        api_doc.get("invoice_fields")
        if api_doc.get("verified") and api_doc.get("invoice_fields")
        else source_fields
    )
    return {
        "vendor_name": initial_fields.get("vendor_name") or initial_fields.get("company_name") or "",
        "customer_name": initial_fields.get("customer_name") or "",
        "gst_number": initial_fields.get("gst_number") or "",
        "customer_gst": initial_fields.get("customer_gst") or initial_fields.get("buyer_gstin") or "",
        "invoice_number": initial_fields.get("invoice_number") or "",
        "invoice_date": initial_fields.get("invoice_date") or "",
        "due_date": initial_fields.get("due_date") or "",
        "gross_amount": initial_fields.get("gross_amount") or "",
        "subtotal": initial_fields.get("subtotal") or "",
        "cgst_amount": initial_fields.get("cgst_amount") or "",
        "sgst_amount": initial_fields.get("sgst_amount") or "",
        "tax_amount": initial_fields.get("tax_amount") or "",
        "total_amount": initial_fields.get("total_amount") or "",
        "currency": initial_fields.get("currency") or "INR",
        "po_number": initial_fields.get("po_number") or "",
        "discount": initial_fields.get("discount") or "",
    }


def evaluate_field_match(
    field_name: str,
    extracted_val: Any,
    gt_val: Any,
) -> Tuple[bool, str]:
    """Compare extracted value against ground truth and return (is_match, reason)."""
    if gt_val is None:
        if extracted_val is None or str(extracted_val).strip() in ("", "Not found", "None", "null"):
            return True, "both_none"
        return False, "spurious_extraction"

    if extracted_val is None or str(extracted_val).strip() in ("", "Not found", "None", "null"):
        return False, "missing_extraction"

    s_ext = str(extracted_val).strip()
    s_gt = str(gt_val).strip()

    # Exact string match
    if s_ext.lower() == s_gt.lower():
        return True, "exact_match"

    # Numeric match (financial fields)
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
        norm_ext = normalize_date(s_ext).normalized or s_ext
        norm_gt = normalize_date(s_gt).normalized or s_gt
        if norm_ext == norm_gt:
            return True, "date_match"
        return False, f"date_mismatch (got '{norm_ext}', expected '{norm_gt}')"

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

    # Currency match
    if field_name == "currency":
        if s_ext.upper() == s_gt.upper() or (s_gt == "USD" and s_ext == "$") or (s_gt == "EUR" and s_ext == "€"):
            return True, "currency_match"
        return False, f"currency_mismatch (got '{s_ext}', expected '{s_gt}')"

    # Line items
    if field_name == "line_items":
        # extracted_val is list of line items, gt_val is int count
        ext_count = len(extracted_val) if isinstance(extracted_val, list) else 0
        gt_count = int(gt_val)
        if gt_count == 0 and ext_count == 0:
            return True, "both_zero"
        if ext_count > 0 and abs(ext_count - gt_count) <= 1:
            return True, "line_items_count_match"
        return False, f"line_items_mismatch (got {ext_count}, expected {gt_count})"

    return False, f"mismatch (got '{s_ext}', expected '{s_gt}')"


def run_app_workflow_evaluation(
    eval_dir: Path = Path("evaluation/fatura2_test50"),
    report_tag: str = "app_workflow_baseline",
) -> Dict[str, Any]:
    """Execute the full application workflow evaluation across test invoices."""
    database.init_db()

    manifest_file = eval_dir / "gt_download_list.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"\n{'='*70}")
    print(f"RUNNING FATURA2 APPLICATION WORKFLOW EVALUATION ({report_tag.upper()})")
    print(f"Source: Real Application Workflow (POST /upload -> POST /extract/layoutlm -> GET /documents/{'{id}'})")
    print(f"Evaluating {len(items)} invoices from unseen templates 44-50")
    print(f"{'='*70}\n")

    field_stats = {
        f: {
            "gt_present": 0,
            "predictions_returned": 0,
            "correct": 0,
            "incorrect": 0,
            "missing": 0,
            "spurious": 0,
            "direct": 0,
            "normalized": 0,
            "derived": 0,
            "failure_reasons": defaultdict(int),
        }
        for f in TARGET_FIELDS
    }

    # Date Forensic Audit Table
    date_forensics = []
    # Financial Derivation Table
    financial_audit = []
    # Discrepancy Audit (Model vs DB vs API vs UI)
    discrepancies = []
    # Granular Line Item Audit
    line_item_records = []

    invoice_records = []
    start_time = time.time()

    for idx, item in enumerate(items):
        inv_id = item["invoice_id"]
        img_path = Path(item["image_file"])
        gt_path = Path(item["out_file"])

        if not img_path.exists() or not gt_path.exists():
            print(f"[{idx+1}/{len(items)}] Missing file for {inv_id}, skipping")
            continue

        gt = parse_ground_truth(gt_path)

        with open(img_path, "rb") as fp:
            img_bytes = fp.read()

        t0 = time.time()
        # 1. Real Application Upload / Ingest
        upload_resp = main.ingest_document(img_path.name, img_bytes)
        doc_id = upload_resp["id"]

        # 2. Real Application LayoutLMv3 Extraction
        extract_resp = main.extract_with_layoutlm(doc_id)

        # 3. Real Application Document Retrieval Endpoint
        api_doc = main.get_document_by_id(doc_id)

        elapsed = round(time.time() - t0, 2)

        # 4. Frontend Review State
        ui_state = simulate_frontend_review_state(api_doc)

        # Field source metadata from extractor
        field_sources = extract_resp.get("metadata", {}).get("field_sources", {})
        raw_values = extract_resp.get("metadata", {}).get("raw_values", {})
        model_fields = extract_resp.get("fields", {})
        db_fields = api_doc.get("invoice_fields", {})
        line_items = api_doc.get("line_items", [])
        validation_res = api_doc.get("validation_results", {})
        ocr_text = api_doc.get("extracted_text", "")

        # 5. Evaluate all 12 fields
        invoice_field_eval = {}

        for f in TARGET_FIELDS:
            gt_val = gt.get(f) if f != "line_items" else gt.get("line_items_count", 0)
            ui_val = ui_state.get(f) if f != "line_items" else line_items
            api_val = db_fields.get(f) if f != "line_items" else line_items
            model_val = model_fields.get(f) if f != "line_items" else line_items

            # Check if GT present
            gt_is_present = (gt_val is not None and gt_val != "" and gt_val != 0)
            if gt_is_present:
                field_stats[f]["gt_present"] += 1

            # Check if prediction returned
            has_pred = False
            if f == "line_items":
                has_pred = len(line_items) > 0
            else:
                has_pred = (api_val is not None and str(api_val).strip() not in ("", "Not found", "None", "null"))

            if has_pred:
                field_stats[f]["predictions_returned"] += 1

            # Evaluate match against ground truth
            is_match, reason = evaluate_field_match(f, api_val, gt_val)
            invoice_field_eval[f] = {"is_match": is_match, "reason": reason, "api_val": api_val, "gt_val": gt_val}

            if gt_is_present:
                if is_match:
                    field_stats[f]["correct"] += 1
                else:
                    if has_pred:
                        field_stats[f]["incorrect"] += 1
                    else:
                        field_stats[f]["missing"] += 1
                    field_stats[f]["failure_reasons"][reason] += 1
            else:
                if has_pred:
                    field_stats[f]["spurious"] += 1

            # Financial provenance accounting
            if f in ["subtotal", "tax_amount", "discount", "total_amount"]:
                source_type = field_sources.get(f, "unknown")
                if source_type == "token_classification":
                    field_stats[f]["direct"] += 1
                elif source_type == "derived":
                    field_stats[f]["derived"] += 1
                elif source_type in ["spatial_anchor", "localized_ocr_patch"]:
                    field_stats[f]["normalized"] += 1
                elif not has_pred:
                    field_stats[f]["missing"] += 1

        # Date Forensic Audit for invoice_date
        raw_inv_date = raw_values.get("invoice_date") or ""
        date_forensics.append({
            "image": img_path.name,
            "template": item.get("template", ""),
            "gt_invoice_date": gt.get("invoice_date"),
            "gt_due_date": gt.get("due_date"),
            "ocr_raw_date": raw_inv_date,
            "model_prediction": model_fields.get("invoice_date"),
            "api_saved_date": db_fields.get("invoice_date"),
            "ui_displayed_date": ui_state.get("invoice_date"),
            "is_date_correct": invoice_field_eval["invoice_date"]["is_match"],
            "failure_reason": invoice_field_eval["invoice_date"]["reason"],
        })

        # Financial Derivation Audit
        financial_audit.append({
            "image": img_path.name,
            "subtotal": {"val": db_fields.get("subtotal"), "gt": gt.get("subtotal"), "source": field_sources.get("subtotal")},
            "tax": {"val": db_fields.get("tax_amount"), "gt": gt.get("tax_amount"), "source": field_sources.get("tax_amount")},
            "discount": {"val": db_fields.get("discount"), "gt": gt.get("discount"), "source": field_sources.get("discount")},
            "total": {"val": db_fields.get("total_amount"), "gt": gt.get("total_amount"), "source": field_sources.get("total_amount")},
            "arithmetic_valid": validation_res.get("is_valid"),
        })

        # Discrepancy Audit (Model vs DB vs API vs UI)
        doc_discrepancies = []
        for f in TARGET_FIELDS:
            if f == "line_items":
                continue
            m_v = str(model_fields.get(f) or "")
            a_v = str(db_fields.get(f) or "")
            u_v = str(ui_state.get(f) or "")
            if m_v != a_v or a_v != u_v:
                doc_discrepancies.append({"field": f, "model": m_v, "api_db": a_v, "ui": u_v})
        if doc_discrepancies:
            discrepancies.append({"image": img_path.name, "discrepancies": doc_discrepancies})

        # Clean up database record to keep test isolated
        main.remove_document(doc_id)

        # Granular Line Item Evaluation
        tpl_name = f"Template{item['template']}" if isinstance(item.get("template"), int) else (item.get("template") or img_path.name.split("_")[0])
        gt_line_items = parse_gt_line_items(gt["raw"], tpl_name)
        li_eval = evaluate_invoice_line_items(line_items, gt_line_items)
        line_item_records.append({
            "image": img_path.name,
            "template": tpl_name,
            **li_eval,
        })

        print(f"[{idx+1:02d}/{len(items):02d}] {img_path.name} ({elapsed}s) | "
              f"Vendor: {'OK' if invoice_field_eval['vendor_name']['is_match'] else 'FAIL'} | "
              f"Inv#: {'OK' if invoice_field_eval['invoice_number']['is_match'] else 'FAIL'} | "
              f"Date: {'OK' if invoice_field_eval['invoice_date']['is_match'] else 'FAIL'} | "
              f"Total: {'OK' if invoice_field_eval['total_amount']['is_match'] else 'FAIL'} | "
              f"Lines: {len(line_items)} (GT:{len(gt_line_items)} M:{li_eval['n_matched']} Ex:{'Y' if li_eval['invoice_exact'] else 'N'})",
              flush=True)

    total_time = round(time.time() - start_time, 2)

    # Compute Summary Metrics per Field
    results_summary = {}
    for f in TARGET_FIELDS:
        s = field_stats[f]
        gt_cnt = s["gt_present"]
        preds = s["predictions_returned"]
        corr = s["correct"]
        inc = s["incorrect"]
        miss = s["missing"]

        precision = round(corr / preds, 4) if preds > 0 else 0.0
        recall = round(corr / gt_cnt, 4) if gt_cnt > 0 else 0.0
        accuracy = round(corr / gt_cnt, 4) if gt_cnt > 0 else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

        results_summary[f] = {
            "gt_present": gt_cnt,
            "predictions_returned": preds,
            "correct": corr,
            "incorrect": inc,
            "missing": miss,
            "spurious": s["spurious"],
            "direct": s["direct"],
            "normalized": s["normalized"],
            "derived": s["derived"],
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1,
            "failure_reasons": dict(s["failure_reasons"]),
        }

    # Compute Line-Item Row-Level Performance
    total_gt_rows = sum(r["n_gt"] for r in line_item_records)
    total_pred_rows = sum(r["n_pred"] for r in line_item_records)
    total_matched_rows = sum(r["n_matched"] for r in line_item_records)
    table_detected_count = sum(1 for r in line_item_records if r["table_detected"])
    table_detection_rate = round(table_detected_count / len(items), 4) if items else 0.0

    row_precision = round(total_matched_rows / total_pred_rows, 4) if total_pred_rows > 0 else 0.0
    row_recall = round(total_matched_rows / total_gt_rows, 4) if total_gt_rows > 0 else 0.0
    row_f1 = round((2 * row_precision * row_recall) / (row_precision + row_recall), 4) if (row_precision + row_recall) > 0 else 0.0

    desc_accuracy = round(sum(r["desc_matches"] for r in line_item_records) / total_matched_rows, 4) if total_matched_rows > 0 else 0.0
    qty_accuracy = round(sum(r["qty_matches"] for r in line_item_records) / total_matched_rows, 4) if total_matched_rows > 0 else 0.0
    price_accuracy = round(sum(r["price_matches"] for r in line_item_records) / total_matched_rows, 4) if total_matched_rows > 0 else 0.0
    total_accuracy = round(sum(r["total_matches"] for r in line_item_records) / total_matched_rows, 4) if total_matched_rows > 0 else 0.0
    row_exact_rate = round(sum(r["row_exact_matches"] for r in line_item_records) / total_matched_rows, 4) if total_matched_rows > 0 else 0.0
    invoice_exact_count = sum(1 for r in line_item_records if r["invoice_exact"])
    invoice_exact_rate = round(invoice_exact_count / len(items), 4) if items else 0.0

    line_item_metrics = {
        "table_detection_rate": table_detection_rate,
        "table_detected_count": table_detected_count,
        "total_gt_rows": total_gt_rows,
        "total_pred_rows": total_pred_rows,
        "total_matched_rows": total_matched_rows,
        "row_precision": row_precision,
        "row_recall": row_recall,
        "row_f1": row_f1,
        "desc_accuracy": desc_accuracy,
        "qty_accuracy": qty_accuracy,
        "price_accuracy": price_accuracy,
        "total_accuracy": total_accuracy,
        "row_exact_rate": row_exact_rate,
        "invoice_exact_count": invoice_exact_count,
        "invoice_exact_rate": invoice_exact_rate,
    }

    # Print Table 1: Canonical Invoice Field Results
    print(f"\n{'='*105}")
    print(f"APPLICATION WORKFLOW EVALUATION RESULTS ({report_tag.upper()}) - {len(items)} INVOICES")
    print(f"{'='*105}")
    header = f"{'Field':<18} | {'GT':<4} | {'Pred':<5} | {'Corr':<5} | {'Inc':<4} | {'Miss':<5} | {'Prec (%)':<9} | {'Rec (%)':<8} | {'Acc (%)':<8} | {'F1':<6}"
    print(header)
    print("-" * 105)
    for f in TARGET_FIELDS:
        res = results_summary[f]
        print(
            f"{f:<18} | {res['gt_present']:<4} | {res['predictions_returned']:<5} | "
            f"{res['correct']:<5} | {res['incorrect']:<4} | {res['missing']:<5} | "
            f"{res['precision']*100:<9.1f} | {res['recall']*100:<8.1f} | {res['accuracy']*100:<8.1f} | {res['f1']:<6.4f}"
        )
    print(f"{'='*105}")

    # Print Table 2: Granular Line Items Row-Level Results
    print(f"\n{'='*105}")
    print(f"GRANULAR LINE-ITEM EXTRACTION PERFORMANCE (ROW-LEVEL) - {len(items)} INVOICES")
    print(f"{'='*105}")
    print(f"{'Metric':<42} | {'Value':<18} | {'Percentage / Ratio':<20}")
    print("-" * 105)
    print(f"{'Invoices Evaluated':<42} | {len(items):<18} | -")
    print(f"{'Table Detection Rate':<42} | {table_detected_count}/{len(items):<16} | {table_detection_rate*100:.1f}%")
    print(f"{'Total Ground Truth Rows':<42} | {total_gt_rows:<18} | -")
    print(f"{'Total Extracted Rows':<42} | {total_pred_rows:<18} | -")
    print(f"{'Matched Rows':<42} | {total_matched_rows:<18} | -")
    print(f"{'Row Precision':<42} | {total_matched_rows}/{total_pred_rows:<16} | {row_precision*100:.1f}%")
    print(f"{'Row Recall':<42} | {total_matched_rows}/{total_gt_rows:<16} | {row_recall*100:.1f}%")
    print(f"{'Row F1 Score':<42} | {row_f1:<18.4f} | -")
    print(f"{'Description Accuracy (matched rows)':<42} | {sum(r['desc_matches'] for r in line_item_records)}/{total_matched_rows:<16} | {desc_accuracy*100:.1f}%")
    print(f"{'Quantity Accuracy (matched rows)':<42} | {sum(r['qty_matches'] for r in line_item_records)}/{total_matched_rows:<16} | {qty_accuracy*100:.1f}%")
    print(f"{'Unit Price Accuracy (matched rows)':<42} | {sum(r['price_matches'] for r in line_item_records)}/{total_matched_rows:<16} | {price_accuracy*100:.1f}%")
    print(f"{'Line Total Accuracy (matched rows)':<42} | {sum(r['total_matches'] for r in line_item_records)}/{total_matched_rows:<16} | {total_accuracy*100:.1f}%")
    print(f"{'Complete Row Exact Match Rate':<42} | {sum(r['row_exact_matches'] for r in line_item_records)}/{total_matched_rows:<16} | {row_exact_rate*100:.1f}%")
    print(f"{'Full Invoice Line-Items Exact Match Rate':<42} | {invoice_exact_count}/{len(items):<16} | {invoice_exact_rate*100:.1f}%")
    print(f"{'='*105}")
    print(f"Total Evaluation Time: {total_time:.1f}s (Average {total_time/len(items):.2f}s per invoice)\n")

    output_payload = {
        "report_tag": report_tag,
        "total_invoices": len(items),
        "total_time_seconds": total_time,
        "results_summary": results_summary,
        "line_item_metrics": line_item_metrics,
        "date_forensics": date_forensics,
        "financial_audit": financial_audit,
        "discrepancies": discrepancies,
    }

    out_file = BASE_DIR / "evaluation" / f"{report_tag}_report.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(output_payload, fp, indent=2)
    print(f"Detailed evaluation payload saved to: {out_file}\n")

    return output_payload


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "app_workflow_baseline"
    edir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("evaluation/fatura2_test50")
    run_app_workflow_evaluation(eval_dir=edir, report_tag=tag)
