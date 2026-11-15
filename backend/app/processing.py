"""OCR and heuristic field extraction.

Text comes from Tesseract OCR or from text already embedded in a PDF.
Field values come from regular expressions and line parsing. This module
does not run a document-understanding model, and the coverage percentage
is only a count of fields that were found.
"""

import os
import re

import fitz
import pytesseract
from PIL import Image, ImageOps


pytesseract.pytesseract.tesseract_cmd = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)

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
        text = pytesseract.image_to_string(
            prepared,
            config="--oem 3 --psm 6",
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
    date_pattern = (
        r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
        r"|\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
        r"[A-Z]*\s+\d{2,4})"
    )
    labeled = re.search(
        r"(?:INVOICE\s*DATE|BILL\s*DATE|DATE\s*OF\s*ISSUE|"
        r"RECEIPT\s*DATE|DATED|DATE)\s*[:.]?\s*"
        + date_pattern,
        text.upper(),
    )
    if labeled:
        return labeled.group(1)

    fallback = re.search(date_pattern, text.upper())
    if fallback:
        return fallback.group(1)
    return None


def _party_name(text):
    skip = re.compile(
        r"^(TAX\s+)?INVOICE\b|^(CASH\s+)?RECEIPT\b|^CASH\s+MEMO\b|"
        r"^ORIGINAL\b|^DUPLICATE\b|^PAGE\b",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        clean = re.sub(r"\s+", " ", line).strip()
        if len(clean) < 3 or skip.match(clean):
            continue
        if re.fullmatch(r"[\d\W_]+", clean):
            continue
        return clean
    return None


def _gst_number(upper_text):
    match = re.search(
        r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z]\d\b",
        upper_text,
    )
    return match.group(0) if match else None


def _reference_number(upper_text, labels):
    for label in labels:
        match = re.search(
            rf"{label}\s*[:.#]?\s*([A-Z0-9][A-Z0-9/-]{{1,30}})",
            upper_text,
        )
        if match:
            return match.group(1)
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
    invoice_number = _reference_number(
        upper_text,
        [
            r"INVOICE\s*(?:NO|NUMBER|#)",
            r"INV\s*(?:NO|NUMBER|#)",
            r"BILL\s*(?:NO|NUMBER|#)",
        ],
    )
    invoice_date = _search_date(text)
    gst_number = _gst_number(upper_text)
    subtotal = _search_amount(
        upper_text,
        [r"SUB\s*TOTAL", r"TAXABLE\s*VALUE", r"TAXABLE\s*AMOUNT"],
    )
    tax_amount = _search_amount(
        upper_text,
        [r"TAX\s*AMOUNT", r"TOTAL\s*TAX", r"GST\s*AMOUNT", r"\bVAT\b"],
    )
    total_amount = _search_amount(
        upper_text,
        [
            r"GRAND\s*TOTAL",
            r"TOTAL\s*AMOUNT",
            r"AMOUNT\s*PAYABLE",
            r"AMOUNT\s*DUE",
            r"BALANCE\s*DUE",
            r"NET\s*PAYABLE",
            r"(?<!SUB)(?<!GRAND\s)TOTAL",
        ],
    )
    currency = _currency(text)

    structured = {
        "vendor_name": _null_if_missing(vendor),
        "invoice_number": _null_if_missing(invoice_number),
        "invoice_date": _null_if_missing(invoice_date),
        "gst_number": _null_if_missing(gst_number),
        "subtotal": _null_if_missing(subtotal),
        "tax_amount": _null_if_missing(tax_amount),
        "total_amount": _null_if_missing(total_amount),
        "currency": _null_if_missing(currency),
    }
    invoice_fields = {
        "company_name": _display(vendor),
        "gst_number": _display(gst_number),
        "invoice_number": _display(invoice_number),
        "invoice_date": _display(invoice_date),
        "subtotal": _display(subtotal),
        "tax_amount": _display(tax_amount),
        "total_amount": _display(total_amount),
        "currency": _display(currency),
    }
    invoice_fields["confidence_score"] = coverage_score(
        [invoice_fields[key] for key in INVOICE_FIELD_KEYS]
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
