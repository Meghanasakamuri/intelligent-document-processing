"""OCR and heuristic field extraction.

Text comes from Tesseract OCR or from text already embedded in a PDF.
Field values come from regular expressions and line parsing. This module
does not run a document-understanding model, and the coverage percentage
is only a count of fields that were found.
"""

import os
import re

import pymupdf as fitz
import pytesseract
from PIL import Image, ImageOps

from app.config import TESSERACT_CMD

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

NOT_FOUND = "Not found"
MAX_PDF_PAGES = 10
CONFIDENCE_METHOD = "heuristic_field_coverage"

INVOICE_FIELD_KEYS = [
    "company_name",
    "gst_number",
    "invoice_number",
    "invoice_date",
    "subtotal",
    "tax_amount",
    "total_amount",
    "currency",
]

RECEIPT_FIELD_KEYS = [
    "merchant_name",
    "receipt_number",
    "date",
    "tax_amount",
    "total_amount",
]


def empty_invoice_fields():
    fields = {key: NOT_FOUND for key in INVOICE_FIELD_KEYS}
    fields["confidence_score"] = 0.0
    fields["confidence_method"] = CONFIDENCE_METHOD
    return fields


def coverage_score(values):
    if not values:
        return 0.0
    found = sum(1 for value in values if value_present(value))
    return round((found / len(values)) * 100, 2)


def value_present(value):
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "not found",
        "none",
        "null",
    }:
        return False
    return True


def _display(value):
    return value if value_present(value) else NOT_FOUND


def _null_if_missing(value):
    return value if value_present(value) else None


def preprocess_image(image):
    """Improve contrast for OCR without changing the stored original."""
    prepared = image.convert("L")
    prepared = ImageOps.autocontrast(prepared)
    if prepared.width < 1200:
        prepared = prepared.resize(
            (prepared.width * 2, prepared.height * 2),
            Image.Resampling.LANCZOS,
        )
    return prepared


def ocr_image(image):
    prepared = preprocess_image(image)
    try:
        from app.config import TESSERACT_OEM, TESSERACT_PSM
        text = pytesseract.image_to_string(
            prepared,
            config=f"--oem {TESSERACT_OEM} --psm {TESSERACT_PSM}",
            timeout=60,
        )
        return text or "", None
    except Exception as first_error:
        try:
            text = pytesseract.image_to_string(prepared, timeout=60)
            return text or "", None
        except Exception as second_error:
            return "", str(second_error or first_error)


def _page_image(page):
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    return Image.frombytes(
        "RGB",
        [pixmap.width, pixmap.height],
        pixmap.samples,
    )


def extract_pdf(file_path):
    document = fitz.open(file_path)
    try:
        page_count = document.page_count
        if page_count == 0:
            return {
                "text": "",
                "page_count": 0,
                "pages_processed": 0,
                "text_source": "none",
                "error": "PDF does not contain any pages",
            }

        pages_processed = min(page_count, MAX_PDF_PAGES)
        parts = []
        sources = set()
        errors = []

        for index in range(pages_processed):
            page = document.load_page(index)
            embedded = page.get_text("text") or ""
            if len(embedded.strip()) >= 40:
                parts.append(embedded)
                sources.add("pymupdf")
                continue

            try:
                text, error = ocr_image(_page_image(page))
            except Exception as exc:
                text, error = "", str(exc)

            if error:
                errors.append(f"Page {index + 1}: {error}")
            if text.strip():
                parts.append(text)
                sources.add("tesseract")

        if not sources:
            text_source = "none"
        elif sources == {"pymupdf"}:
            text_source = "pymupdf"
        elif sources == {"tesseract"}:
            text_source = "tesseract"
        else:
            text_source = "pymupdf+tesseract"

        notes = list(errors)
        if page_count > pages_processed:
            notes.append(
                f"Processed the first {pages_processed} of {page_count} pages"
            )

        return {
            "text": "\n\n".join(parts),
            "page_count": page_count,
            "pages_processed": pages_processed,
            "text_source": text_source,
            "error": "; ".join(notes) or None,
        }
    finally:
        document.close()


def extract_image(file_path):
    with Image.open(file_path) as image:
        image.load()
        text, error = ocr_image(image.copy())
    return {
        "text": text,
        "page_count": 1,
        "pages_processed": 1,
        "text_source": "tesseract" if text.strip() else "none",
        "error": error,
    }


def detect_document_type(text):
    upper_text = text.upper()

    invoice_keywords = [
        "INVOICE",
        "TAX INVOICE",
        "GST",
        "GSTIN",
        "GRAND TOTAL",
        "INVOICE NO",
        "INVOICE NUMBER",
    ]
    receipt_keywords = [
        "RECEIPT",
        "CASH MEMO",
        "AMOUNT RECEIVED",
        "PAID BY",
        "CHANGE DUE",
    ]
    form_keywords = [
        "FORM",
        "APPLICATION",
        "DECLARATION",
        "REGISTRATION",
        "SIGNATURE",
        "DATE OF BIRTH",
    ]

    invoice_score = sum(
        1 for keyword in invoice_keywords if keyword in upper_text
    )
    receipt_score = sum(
        1 for keyword in receipt_keywords if keyword in upper_text
    )
    form_score = sum(
        1 for keyword in form_keywords if keyword in upper_text
    )

    if "TAX INVOICE" in upper_text or invoice_score >= 2:
        if receipt_score > invoice_score and "INVOICE" not in upper_text:
            return "Receipt"
        return "Invoice"

    if receipt_score >= 1 and receipt_score >= form_score:
        return "Receipt"

    if form_score >= 2:
        return "Form"

    return "Unknown"


def _search_amount(upper_text, labels):
    amount = (
        r"(?:RS\.?|INR|USD|EUR|₹|\$|€)?\s*([\d,]+(?:\.\d{1,2})?)"
    )
    for label in labels:
        match = re.search(
            rf"{label}\s*[:.\-]?\s*{amount}",
            upper_text,
        )
        if match:
            return match.group(1)
    return None


def _search_date(text):
    # Support DD/Mon/YYYY (e.g. 21/Sep/2026), DD-MM-YYYY, YYYY-MM-DD
    labeled = re.search(
        r"(?:INVOICE\s*DATE|BILL\s*DATE|DATE\s*OF\s*ISSUE|RECEIPT\s*DATE|DATED|DATE)\s*[:.\-]?\s*\n?\s*"
        r"(\d{1,2}[-/](?:[A-Za-z]{3}|\d{1,2})[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})",
        text,
        re.IGNORECASE,
    )
    if labeled:
        return labeled.group(1).strip()

    # Fallback to date patterns with valid day (1-31) and year (1990-2040)
    for m in re.finditer(r"\b(\d{1,2}[-/](?:[A-Za-z]{3}|\d{1,2})[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", text):
        cand = m.group(1).strip()
        parts = re.split(r"[-/]", cand)
        if len(parts) == 3:
            # Avoid door numbers like 76-14-95
            try:
                if len(parts[0]) == 4: # YYYY-MM-DD
                    if 1990 <= int(parts[0]) <= 2040 and 1 <= int(parts[1]) <= 12:
                        return cand
                else: # DD-MM-YYYY or DD-Mon-YYYY
                    if int(parts[0]) <= 31 and 1990 <= int(parts[2]) <= 2040:
                        return cand
            except ValueError:
                # Month name like Sep
                return cand
    return None


def _party_name(text):
    skip = re.compile(
        r"^(?:TAX\s+)?INVOICE\b|^(?:CASH\s+)?RECEIPT\b|^CASH\s+MEMO\b|"
        r"^ORIGINAL\b|^DUPLICATE\b|^PAGE\b|^GSTIN\b|^GST\s*NO\b|^GST:\b|"
        r"^EMAIL\b|^CELL\b|^PHONE\b|^TEL\b|^D\.NO\b|^PLOT\b|^ADDRESS\b|"
        r"^20B\b|^21B\b|^WLF\b|^PAN\s*NO\b|^STATE\s*CODE\b|^STATE\s*NAME\b|"
        r"^\(|^DISTRIBUTOR\b|^MANUFACTURER\b|^DEALERS?\b",
        re.IGNORECASE,
    )
    lines = text.splitlines()
    # Check top lines first
    for line in lines[:15]:
        clean = re.sub(r"\s+", " ", line).strip()
        if len(clean) < 3 or skip.match(clean) or clean.startswith("(") or clean.endswith(")"):
            continue
        if re.fullmatch(r"[\d\W_]+", clean):
            continue
        # Strip leading signatory noise e.g. "For SIVA SAI ENTERPRISES"
        cleaned_vendor = re.sub(r"^(?:FOR|M/S\.?|MESSRS)\s+", "", clean, flags=re.IGNORECASE).strip()
        if any(c.isalpha() for c in cleaned_vendor) and len(cleaned_vendor) >= 3:
            return cleaned_vendor

    # Check footer signatory line if top header was missed (e.g. "For SIVA SAI ENTERPRISES")
    for line in lines:
        m_for = re.match(r"^\s*FOR\s+([A-Za-z0-9\s&.,'()-]+)$", line, re.IGNORECASE)
        if m_for:
            name = m_for.group(1).strip(" :.-#")
            if len(name) >= 3 and not any(k in name.upper() for k in ["SIGNATURE", "AUTHORISED", "CUSTOMER", "RECEIVER"]):
                return name
    return None


def _customer_name(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    strict_addr_indicators = [
        "ADDRESS", "GSTIN", "DATE", "INVOICE", "20B", "21B", "SIGNATURE", "SIGN", "SEAL", "RECEIVER",
        "GROUND", "FLOOR", "D.NO", "H.NO", "PLOT", "FLAT", "APARTMENT", "RESIDENCY", "NEAR", "OPP", "BEHIND", "BESIDE"
    ]
    locality_indicators = [
        "ROAD", "STREET", "NAGAR", "COLONY", "TEMPLE", "MARG", "LANE", "CROSS", "ENCLAVE",
        "AVENUE", "LAYOUT", "SECTOR", "VILLAGE", "DISTRICT"
    ]
    comp_suffix_pattern = re.compile(
        r"\b(?:HEALTH\s*CARE|HEALTHCARE|PHARMA(?:CY|CEUTICALS?)?|MEDICALS?|CLINIC|HOSPITAL|SURGICALS?|"
        r"LABS?|DIAGNOSTICS?|ENTERPRISES?|LTD|PVT|LIMITED|CORP|INC|INDUSTRIES|AGENC(?:Y|IES)|SOLUTIONS|"
        r"SYSTEMS|SERVICES|COMPANY|CO\b|LLP|PLC|GMBH|BROTHERS?|PARTNERS?|STORE|STORES)\b",
        re.IGNORECASE,
    )

    def _is_invalid_candidate(cand_str: str) -> bool:
        if len(cand_str) < 3 or re.match(r"^\d+[-/]\d+", cand_str):
            return True
        c_up = cand_str.upper()
        if any(k in c_up for k in ["CELL", "PHONE", "EMAIL", "DISTRIBUTOR", "PRODUCTS", "GSTIN", "20B", "21B", "SIGN", "ENTERPRISES"]):
            return True
        if any(k in c_up for k in strict_addr_indicators):
            return True
        if not comp_suffix_pattern.search(cand_str) and any(k in c_up for k in locality_indicators):
            return True
        return False

    # 1. Preceding INVOICE NO in multi-column Indian layouts (e.g. SINGH NAGAR P N P HEALTH CARE / PRASANTHI PHARMACY)
    for idx, line in enumerate(lines):
        if line.upper() in ("INVOICE NO", "INVOICE NO:", "TAX INVOICE") and idx > 0:
            cand = lines[idx - 1].strip(" :.-#")
            cand_clean = re.sub(r"^(?:Mr\.?|Ms\.?|Mrs\.?|M/s\.?)\s+", "", cand, flags=re.IGNORECASE).strip()
            if not _is_invalid_candidate(cand_clean):
                return cand_clean

    # 2. Anchor check: Client Name, Bill To, Billed To, Customer (not Customer Signature), Buyer, Messrs, M/s
    m_cust = re.search(
        r"\b(?:Client\s*Name|Bill\s*To|Billed\s*To|Customer(?!\s*['’]s?\s*Sign)|Buyer|Messrs|M/s)\b\s*[:.\-]?\s*([A-Za-z0-9\s&.,'()-]+)",
        text,
        re.IGNORECASE,
    )
    if m_cust:
        cand = m_cust.group(1).splitlines()[0].strip(" :.-#")
        cand_clean = re.sub(r"^(?:Name\s*[:.\-]?\s*|Mr\.?|Ms\.?|Mrs\.?|M/s\.?)\s*", "", cand, flags=re.IGNORECASE).strip()
        if not _is_invalid_candidate(cand_clean):
            return cand_clean

    # 3. Pattern check: "To" anchor on its own line followed by recipient name
    for idx, line in enumerate(lines):
        if line.lower() in ("to", "to:") and idx + 1 < len(lines):
            cand = lines[idx + 1].strip(" :.-#")
            cand_clean = re.sub(r"^(?:Mr\.?|Ms\.?|Mrs\.?|M/s\.?)\s+", "", cand, flags=re.IGNORECASE).strip()
            if not _is_invalid_candidate(cand_clean):
                return cand_clean

    return None


def _gst_numbers(text):
    upper_text = text.upper()
    vendor_gst = None
    customer_gst = None

    # Split text into vendor block (before Client/Bill To/Buyer/To) and customer block
    split_m = re.search(r"\b(?:Client\s*Name|Bill\s*To|Billed\s*To|Buyer|Customer|Consignee|To\b)\b", text, re.IGNORECASE)
    if split_m:
        vendor_block = text[:split_m.start()]
        customer_block = text[split_m.start():]
    else:
        vendor_block = text
        customer_block = ""

    # 1. Vendor GST: look in vendor block first
    m_v = re.search(r"\b(?:GSTIN|GST\s*NO|GSTIN\s*NO|GST)\s*[:.\-]?\s*([0-9A-Z]{10,15})\b", vendor_block, re.IGNORECASE)
    if m_v:
        vendor_gst = m_v.group(1).strip()
    else:
        v_gstins = re.findall(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b", vendor_block.upper())
        if v_gstins:
            vendor_gst = v_gstins[0]

    # 2. Customer GST: look in customer block
    if customer_block:
        m_c = re.search(r"\b(?:GSTIN|GST\s*NO|GSTIN\s*NO|GST)\s*[:.\-]?\s*([0-9A-Z]{10,15})\b", customer_block, re.IGNORECASE)
        if m_c:
            customer_gst = m_c.group(1).strip()
        else:
            c_gstins = re.findall(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b", customer_block.upper())
            if c_gstins:
                customer_gst = c_gstins[0]

    all_gstins = re.findall(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b", upper_text)
    if not vendor_gst and all_gstins:
        vendor_gst = all_gstins[0]
    if not customer_gst:
        if len(all_gstins) > 1:
            customer_gst = all_gstins[1]
        elif all_gstins and vendor_gst and all_gstins[0] != vendor_gst:
            customer_gst = all_gstins[0]

    return vendor_gst, customer_gst


def _reference_number(upper_text, labels):
    bad_pat = re.compile(r"\b(?:AP/\d{2}/\d{2}/\d{4}-\d+|WLF\d{2}[A-Z]\d+|20B|21B|D\.?\s*NO|DOOR\s*NO|H\.?\s*NO|PLOT\s*NO)\b", re.IGNORECASE)
    for label in labels:
        # Check on same line
        match = re.search(
            rf"{label}\s*[:.#]?\s*([A-Z0-9][A-Z0-9/-]{{1,30}})",
            upper_text,
        )
        if match:
            cand = match.group(1).strip()
            # Discard drug licenses, PAN, GSTIN, door numbers
            if not bad_pat.search(cand) and not re.match(r"^[A-Z]{5}\d{4}[A-Z]$", cand):
                if not re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", cand):
                    return cand

        # Check on next line e.g. "INVOICE NO\nGSTSAL/996"
        match_nl = re.search(
            rf"{label}\s*[:.#]?\s*\n\s*([A-Z0-9][A-Z0-9/-]{{1,30}})",
            upper_text,
        )
        if match_nl:
            cand = match_nl.group(1).strip()
            if not bad_pat.search(cand) and not re.match(r"^[A-Z]{5}\d{4}[A-Z]$", cand):
                if not re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", cand):
                    return cand
    return None


def _currency(text):
    upper_text = text.upper()
    if "₹" in text or re.search(r"\bINR\b|\bRS\.?\b", upper_text):
        return "INR"
    if "$" in text or re.search(r"\bUSD\b", upper_text):
        return "USD"
    if "€" in text or re.search(r"\bEUR\b", upper_text):
        return "EUR"
    return None


def extract_invoice_fields(text):
    upper_text = text.upper()
    vendor = _party_name(text)
    customer = _customer_name(text)
    invoice_number = _reference_number(
        upper_text,
        [
            r"INVOICE\s*(?:NO|NUMBER|#)",
            r"INV\s*(?:NO|NUMBER|#)",
            r"BILL\s*(?:NO|NUMBER|#)",
        ],
    )
    invoice_date = _search_date(text)
    vendor_gst, customer_gst = _gst_numbers(text)

    # Subtotal / Taxable amount
    subtotal = _search_amount(
        upper_text,
        [
            r"TAXABLE\s*VALUE",
            r"TAXABLE\s*AMOUNT",
            r"TAXABLE\s*AMT",
            r"GROSS\s*AMOUNT",
            r"TOTAL\s*VALUE",
            r"SUB\s*TOTAL",
        ],
    )

    # Taxes
    cgst_amount = None
    sgst_amount = None
    igst_amount = None

    m_cgst = re.search(r"\bCGST\b[^\n\d]{0,40}?([0-9,]+\.[0-9]{2})", text, re.IGNORECASE)
    if m_cgst:
        cgst_amount = m_cgst.group(1).replace(",", "")
    m_sgst = re.search(r"\bSGST\b[^\n\d]{0,40}?([0-9,]+\.[0-9]{2})", text, re.IGNORECASE)
    if m_sgst:
        sgst_amount = m_sgst.group(1).replace(",", "")
    m_igst = re.search(r"\bIGST\b[^\n\d]{0,40}?([0-9,]+\.[0-9]{2})", text, re.IGNORECASE)
    if m_igst:
        igst_amount = m_igst.group(1).replace(",", "")

    cgst_rate = None
    sgst_rate = None
    igst_rate = None
    m_cgst_r = re.search(r"\bCGST\b[^\n\d]*?(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
    if m_cgst_r:
        cgst_rate = f"{m_cgst_r.group(1)}%"
    m_sgst_r = re.search(r"\bSGST\b[^\n\d]*?(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
    if m_sgst_r:
        sgst_rate = f"{m_sgst_r.group(1)}%"
    m_igst_r = re.search(r"\bIGST\b[^\n\d]*?(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
    if m_igst_r:
        igst_rate = f"{m_igst_r.group(1)}%"

    # Total tax
    tax_amount = None
    if cgst_amount and sgst_amount:
        try:
            tax_amount = f"{float(cgst_amount) + float(sgst_amount):.2f}"
        except ValueError:
            pass
    if not tax_amount and igst_amount:
        tax_amount = igst_amount

    if not tax_amount:
        tax_amount = _search_amount(
            upper_text,
            [r"TOTAL\s*TAX", r"TAX\s*AMOUNT", r"GST\s*AMOUNT", r"\bVAT\b"],
        )

    total_amount = _search_amount(
        upper_text,
        [
            r"NET\s*AMOUNT",
            r"GRAND\s*TOTAL",
            r"TOTAL\s*AMOUNT",
            r"AMOUNT\s*PAYABLE",
            r"AMOUNT\s*DUE",
            r"BALANCE\s*DUE",
            r"NET\s*PAYABLE",
            r"(?<!SUB)(?<!GRAND\s)TOTAL",
        ],
    )
    if not total_amount:
        m_net = re.search(
            r"\b(?:NET\s*AMOUNT|TOTAL\s*AMOUNT|GRAND\s*TOTAL|AMOUNT\s*PAYABLE|NET\s*PAYABLE)\b\s*[:.\-]?\s*(?:[^\n\d]*\n)*?[^\n\d]*?([0-9,]+(?:\.[0-9]{2})?)",
            upper_text,
        )
        if m_net:
            total_amount = m_net.group(1).replace(",", "")
    currency = _currency(text)

    structured = {
        "vendor_name": _null_if_missing(vendor),
        "customer_name": _null_if_missing(customer),
        "invoice_number": _null_if_missing(invoice_number),
        "invoice_date": _null_if_missing(invoice_date),
        "gst_number": _null_if_missing(vendor_gst),
        "customer_gstin": _null_if_missing(customer_gst),
        "subtotal": _null_if_missing(subtotal),
        "tax_amount": _null_if_missing(tax_amount),
        "cgst_amount": _null_if_missing(cgst_amount),
        "sgst_amount": _null_if_missing(sgst_amount),
        "igst_amount": _null_if_missing(igst_amount),
        "total_amount": _null_if_missing(total_amount),
        "currency": _null_if_missing(currency),
        "cgst_rate": _null_if_missing(cgst_rate),
        "sgst_rate": _null_if_missing(sgst_rate),
        "igst_rate": _null_if_missing(igst_rate),
    }
    invoice_fields = {
        "vendor_name": _display(vendor),
        "company_name": _display(vendor),
        "customer_name": _display(customer),
        "gst_number": _display(vendor_gst),
        "customer_gstin": _display(customer_gst),
        "invoice_number": _display(invoice_number),
        "invoice_date": _display(invoice_date),
        "subtotal": _display(subtotal),
        "tax_amount": _display(tax_amount),
        "cgst_amount": _display(cgst_amount),
        "sgst_amount": _display(sgst_amount),
        "igst_amount": _display(igst_amount),
        "cgst_rate": _display(cgst_rate),
        "sgst_rate": _display(sgst_rate),
        "igst_rate": _display(igst_rate),
        "total_amount": _display(total_amount),
        "currency": _display(currency),
    }
    invoice_fields["confidence_score"] = coverage_score(
        [invoice_fields[key] for key in INVOICE_FIELD_KEYS if key in invoice_fields]
    )
    invoice_fields["confidence_method"] = CONFIDENCE_METHOD
    return invoice_fields, structured


def extract_receipt_fields(text):
    upper_text = text.upper()
    merchant = _party_name(text)
    receipt_number = _reference_number(
        upper_text,
        [
            r"RECEIPT\s*(?:NO|NUMBER|#)",
            r"BILL\s*(?:NO|NUMBER|#)",
            r"TRANS(?:ACTION)?\s*(?:NO|ID|#)",
        ],
    )
    receipt_date = _search_date(text)
    tax_amount = _search_amount(
        upper_text,
        [r"TAX\s*AMOUNT", r"TOTAL\s*TAX", r"GST\s*AMOUNT", r"\bVAT\b"],
    )
    total_amount = _search_amount(
        upper_text,
        [
            r"GRAND\s*TOTAL",
            r"TOTAL\s*AMOUNT",
            r"AMOUNT\s*PAID",
            r"AMOUNT\s*DUE",
            r"BALANCE\s*DUE",
            r"(?<!SUB)(?<!GRAND\s)TOTAL",
        ],
    )

    structured = {
        "merchant_name": _null_if_missing(merchant),
        "receipt_number": _null_if_missing(receipt_number),
        "date": _null_if_missing(receipt_date),
        "tax_amount": _null_if_missing(tax_amount),
        "total_amount": _null_if_missing(total_amount),
    }
    invoice_fields = empty_invoice_fields()
    invoice_fields.update(
        {
            "company_name": _display(merchant),
            "invoice_number": _display(receipt_number),
            "invoice_date": _display(receipt_date),
            "tax_amount": _display(tax_amount),
            "total_amount": _display(total_amount),
        }
    )
    invoice_fields["confidence_score"] = coverage_score(
        [structured[key] for key in RECEIPT_FIELD_KEYS]
    )
    return invoice_fields, structured


def extract_form_fields(text):
    upper_text = text.upper()
    form_type = None
    typed = re.search(
        r"\b([A-Z][A-Z0-9 /-]{0,40}\bFORM|FORM\s*[:#-]?\s*[A-Z0-9-]{1,20})\b",
        upper_text,
    )
    if typed:
        form_type = re.sub(r"\s+", " ", typed.group(1)).strip()

    pairs = []
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        label = re.sub(r"\s+", " ", label).strip(" -\t")
        value = re.sub(r"\s+", " ", value).strip()
        if not label or len(label) > 80:
            continue
        pairs.append(
            {
                "label": label,
                "value": value or None,
            }
        )

    structured = {
        "form_type": form_type,
        "fields": pairs,
    }
    invoice_fields = empty_invoice_fields()
    found_values = sum(1 for pair in pairs if value_present(pair["value"]))
    expected = max(len(pairs), 1)
    type_found = 1 if form_type else 0
    invoice_fields["confidence_score"] = round(
        ((found_values + type_found) / (expected + 1)) * 100,
        2,
    )
    if pairs:
        invoice_fields["company_name"] = _display(pairs[0]["value"])
    return invoice_fields, structured


def extract_fields(text, document_type):
    if document_type == "Invoice":
        return extract_invoice_fields(text)
    if document_type == "Receipt":
        return extract_receipt_fields(text)
    if document_type == "Form":
        return extract_form_fields(text)

    invoice_fields, structured = extract_invoice_fields(text)
    invoice_fields["confidence_score"] = coverage_score(
        [
            invoice_fields["company_name"],
            invoice_fields["invoice_number"],
            invoice_fields["invoice_date"],
            invoice_fields["total_amount"],
        ]
    )
    return invoice_fields, structured


def process_file(file_path, extension):
    try:
        if extension == ".pdf":
            extracted = extract_pdf(file_path)
        else:
            extracted = extract_image(file_path)
    except Exception as exc:
        invoice_fields = empty_invoice_fields()
        return {
            "document_type": "Unknown",
            "extracted_text": "",
            "invoice_fields": invoice_fields,
            "extracted_fields": {},
            "page_count": None,
            "pages_processed": 0,
            "text_source": "none",
            "processing_status": "failed",
            "processing_error": str(exc),
            "message": "Document stored, but it could not be read",
            "confidence_score": 0.0,
            "confidence_method": CONFIDENCE_METHOD,
        }

    text = extracted["text"] or ""
    if not text.strip():
        invoice_fields = empty_invoice_fields()
        error = extracted["error"] or "No readable text was extracted"
        return {
            "document_type": "Unknown",
            "extracted_text": "",
            "invoice_fields": invoice_fields,
            "extracted_fields": {},
            "page_count": extracted["page_count"],
            "pages_processed": extracted["pages_processed"],
            "text_source": extracted["text_source"],
            "processing_status": "failed",
            "processing_error": error,
            "message": "Document stored, but no readable text was extracted",
            "confidence_score": 0.0,
            "confidence_method": CONFIDENCE_METHOD,
        }

    document_type = detect_document_type(text)
    invoice_fields, structured = extract_fields(text, document_type)
    warning = extracted["error"]

    return {
        "document_type": document_type,
        "extracted_text": text,
        "invoice_fields": invoice_fields,
        "extracted_fields": structured,
        "page_count": extracted["page_count"],
        "pages_processed": extracted["pages_processed"],
        "text_source": extracted["text_source"],
        "processing_status": "processed",
        "processing_error": warning,
        "message": "Document uploaded and processed successfully",
        "confidence_score": invoice_fields.get("confidence_score", 0.0),
        "confidence_method": CONFIDENCE_METHOD,
    }
