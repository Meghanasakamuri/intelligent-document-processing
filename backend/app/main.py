import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONTENT_TYPES,
    LAYOUTLM_MODEL,
    MAX_UPLOAD_BYTES,
    MODELS_CACHE_DIR,
    QA_MODEL,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
)
from app.database import (
    dashboard_stats,
    delete_document,
    get_audit_log,
    get_document,
    init_db,
    insert_audit_entry,
    insert_document,
    labeled_documents,
    list_documents,
    set_ground_truth,
    set_verified,
    update_extraction_method,
    update_fields,
)
from app.logging_config import setup_logging
from app.models.confidence import classify_confidence
from app.processing import (
    CONFIDENCE_METHOD,
    INVOICE_FIELD_KEYS,
    NOT_FOUND,
    RECEIPT_FIELD_KEYS,
    coverage_score,
    empty_invoice_fields,
    process_file,
    value_present,
)
from app.schemas import (
    FieldUpdate,
    GroundTruthRequest,
    ModelStatusResponse,
    QARequest,
    VerifyRequest,
)

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Intelligent Document Processing API",
    description="Research API for Layout-Aware Document Extraction (O2) and Document QA (O3)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# ── Model Singletons ──────────────────────────────────────────────────
_layoutlm_extractor = None
_document_qa = None


def get_layoutlm():
    """Lazy-load LayoutLMv3 extractor (O2)."""
    global _layoutlm_extractor
    if _layoutlm_extractor is None:
        from app.models.layoutlm_extractor import LayoutLMv3Extractor
        _layoutlm_extractor = LayoutLMv3Extractor(
            model_name=LAYOUTLM_MODEL,
            cache_dir=str(MODELS_CACHE_DIR),
        )
    return _layoutlm_extractor


def get_qa():
    """Lazy-load Document QA model (O3)."""
    global _document_qa
    if _document_qa is None:
        from app.models.document_qa import DocumentQA
        _document_qa = DocumentQA(
            model_name=QA_MODEL,
            cache_dir=str(MODELS_CACHE_DIR),
        )
    return _document_qa


def _now():
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(filename):
    base = os.path.basename(filename or "document")
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return (cleaned or "document")[:180]


def _load_document_images(file_path: str, max_pages: int = 10) -> list[tuple[int, Any]]:
    """Load all pages of a document (or single image) as a list of (page_num_1_based, PIL Image).

    Parameters
    ----------
    file_path : str
        Path to the PDF or image file.
    max_pages : int
        Maximum number of pages to render. Defaults to 10 (matching MAX_PDF_PAGES).

    Returns
    -------
    list[tuple[int, Image.Image]]
        List of (page_number, PIL Image) tuples, 1-indexed.
    """
    from PIL import Image
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document file not found on disk")

    if path.suffix.lower() == ".pdf":
        import pymupdf
        try:
            doc = pymupdf.open(str(path))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid or corrupted PDF file: {exc}")

        try:
            total_pages = doc.page_count
            if total_pages == 0:
                raise HTTPException(status_code=400, detail="PDF has no pages")

            pages_to_load = min(total_pages, max_pages)
            images: list[tuple[int, Image.Image]] = []

            for p_idx in range(pages_to_load):
                try:
                    page = doc[p_idx]
                    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                    images.append((p_idx + 1, img))
                except Exception as page_exc:
                    logger.warning("Failed to render PDF page %d: %s", p_idx + 1, page_exc)

            if not images:
                raise HTTPException(status_code=500, detail="Failed to render any pages from PDF")

            return images
        finally:
            doc.close()

    try:
        image = Image.open(str(path)).convert("RGB")
        return [(1, image)]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")


def _load_document_image(file_path: str):
    """Load a document file (image or first page of PDF) as a PIL Image (backward-compatible)."""
    pages = _load_document_images(file_path, max_pages=1)
    return pages[0][1]


def _fuse_hybrid_fields(llm_fields: dict, baseline_fields: dict, confs: dict | None = None) -> tuple[dict, dict]:
    """Fuse LayoutLM predictions with baseline rule-based / spatial extractions.
    
    LayoutLM has primary authority for learned spatial layout fields, but baseline
    provides high-precision regex/anchor statutory values and fallback for fields where
    the model produced invalid noise (such as drug licenses for invoice numbers).
    """
    fused = dict(llm_fields or {})
    fused_conf = dict(confs or {})

    # 1. Invoice Number
    inv_num = fused.get("invoice_number")
    is_bad_inv = False
    if inv_num:
        is_bad_inv = bool(
            re.search(r"\b(?:AP/\d{2}/\d{2}/\d{4}-\d+|WLF\d{2}[A-Z]\d+|20B|21B|D\.?\s*NO|DOOR\s*NO|H\.?\s*NO|PLOT\s*NO)\b", inv_num, re.IGNORECASE)
            or re.search(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b", inv_num)
            or re.match(r"^[A-Z]{5}\d{4}[A-Z]$", inv_num)
            or not any(c.isdigit() or c.isalpha() for c in inv_num)
        )
    if (not inv_num or is_bad_inv) and baseline_fields.get("invoice_number"):
        fused["invoice_number"] = baseline_fields["invoice_number"]
        fused_conf["invoice_number"] = 0.90

    # 2. Vendor Name / Company Name
    v_name = fused.get("vendor_name") or fused.get("company_name")
    b_vname = baseline_fields.get("vendor_name") or baseline_fields.get("company_name")

    comp_keyword_regex = re.compile(
        r"\b(?:LIMITED|LTD|PVT|ENTERPRISES?|CORP|INC|INDUSTRIES|PHARMA(?:CEUTICALS?)?|SURGICALS?|"
        r"HEALTH\s*CARE|HEALTHCARE|MEDICALS?|CLINIC|HOSPITAL|LABS?|DIAGNOSTICS?|STORE|STORES|"
        r"TRADERS?|AGENC(?:Y|IES)|SOLUTIONS|SYSTEMS|SERVICES|COMPANY|CO\b|LLP|PLC|GMBH|"
        r"SONS?|BROTHERS?|PARTNERS?|GROUP|HOLDINGS?|ASSOCIATES?)\b",
        re.IGNORECASE,
    )
    b_has_comp = bool(b_vname and comp_keyword_regex.search(b_vname))
    v_has_comp = bool(v_name and comp_keyword_regex.search(v_name))

    addr_indicators = [
        "ROAD", "STREET", "NAGAR", "COLONY", "TEMPLE", "NEAR", "BEHIND", "OPP", "BESIDE", "FLOOR", "BUILDING",
        "PLOT", "D.NO", "H.NO", "FLAT", "APARTMENT", "SUITE", "SIGNATURE", "SIGN", "AUTHORISED", "AVENUE",
        "LAYOUT", "SECTOR", "VILLAGE", "DISTRICT", "MARG", "LANE", "CROSS", "ENCLAVE"
    ]
    addr_pattern = re.compile(
        r"\b(?:ROAD|STREET|NAGAR|COLONY|TEMPLE|NEAR|OPP|BEHIND|BESIDE|FLOOR|BUILDING|PLOT|"
        r"AVENUE|LAYOUT|SECTOR|VILLAGE|DISTRICT|MARG|LANE|CROSS|ENCLAVE|APARTMENT|SUITE)\b",
        re.IGNORECASE,
    )
    job_title_regex = re.compile(
        r"\b(?:MANAGER|MANAGE|DIRECTOR|CASHIER|ADMIN(?:ISTRATOR)?|OFFICER|CLERK|SIGNATORY|EMPLOYEE|CONSULTANT|ACCOUNTANT|SUPERVISOR|ASSISTANT|OWNER)\b",
        re.IGNORECASE,
    )
    c_addr = fused.get("customer_address") or ""
    c_name_val = fused.get("customer_name") or ""
    is_v_in_caddr = bool(v_name and len(v_name.strip()) >= 3 and (v_name.upper() in c_addr.upper() or (c_name_val and v_name.upper() == c_name_val.upper())))
    v_has_addr = bool(v_name and (addr_pattern.search(v_name) or bool(re.match(r"^\d+[-/]\d+", v_name.strip()))) and not v_has_comp)
    is_v_job = bool(v_name and job_title_regex.search(v_name) and not v_has_comp)

    # Derive vendor from website if available
    website = fused.get("vendor_website") or baseline_fields.get("vendor_website")
    derived_name = None
    if website:
        cleaned_site = re.sub(r"^[‘'\"\s]*(?:https?://)?(?:www\.)?\s*", "", str(website)).strip()
        domain = cleaned_site.split(".")[0] if cleaned_site else ""
        if domain and len(domain) >= 3 and not any(k in domain.upper() for k in ["EXAMPLE", "TEST", "LOCALHOST", "GMAIL", "YAHOO", "HOTMAIL"]):
            spaced = re.sub(r"([a-zA-Z])(and)([A-Z])", r"\1 \2 \3", domain, flags=re.IGNORECASE)
            spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", spaced)
            derived_name = spaced

    is_vname_poor = (
        not v_name
        or len(v_name.strip()) < 2
        or "GST" in v_name.upper()
        or v_name.strip().startswith("(")
        or is_v_in_caddr
        or v_has_addr
        or is_v_job
        or any(k in v_name.upper() for k in [
            "CLIENT NAME", "CUSTOMER NAME", "BUYER NAME", "BILL TO", "SHIP TO", "CLIENT", "TO:",
            "DISTRIBUTOR", "MANUFACTURER", "DEALER", "WHOLESALER", "RETAILER", "PRODUCTS", "COMMERCIAL INVOICE", "TAX INVOICE"
        ])
        or (b_has_comp and not v_has_comp)
    )
    b_is_addr = bool(b_vname and (addr_pattern.search(b_vname) or bool(re.match(r"^\d+[-/]\d+", b_vname.strip()))) and not b_has_comp)
    b_is_generic = bool(b_vname and any(k in b_vname.upper() for k in ["COMMERCIAL INVOICE", "TAX INVOICE", "INVOICE", "BILL", "CASH MEMO"]))

    if is_vname_poor and b_vname and not b_is_addr and not b_is_generic:
        fused["vendor_name"] = b_vname
        fused["company_name"] = b_vname
        fused_conf["vendor_name"] = 0.90
        fused_conf["company_name"] = 0.90
    elif is_vname_poor and derived_name:
        fused["vendor_name"] = derived_name
        fused["company_name"] = derived_name
        fused_conf["vendor_name"] = 0.88
        fused_conf["company_name"] = 0.88
    elif v_name and not is_vname_poor:
        if "vendor_name" not in fused or fused.get("vendor_name") != v_name:
            fused["vendor_name"] = v_name
        if "company_name" not in fused or fused.get("company_name") != v_name:
            fused["company_name"] = v_name
    elif b_vname and not b_is_addr and not b_is_generic:
        fused["vendor_name"] = b_vname
        fused["company_name"] = b_vname
        fused_conf["vendor_name"] = 0.85
        fused_conf["company_name"] = 0.85
    elif derived_name:
        fused["vendor_name"] = derived_name
        fused["company_name"] = derived_name
        fused_conf["vendor_name"] = 0.85
        fused_conf["company_name"] = 0.85

    # 3. Customer Name
    c_name = fused.get("customer_name")
    b_cname = baseline_fields.get("customer_name")

    strict_c_addr = [
        "GROUND", "FLOOR", "D.NO", "H.NO", "PLOT", "FLAT", "APARTMENT", "SUITE", "SIGNATURE", "SIGN", "AUTHORISED"
    ]
    locality_c_addr = [
        "ROAD", "STREET", "NAGAR", "COLONY", "TEMPLE", "NEAR", "BEHIND", "OPP", "BUILDING", "MARG", "LANE", "CROSS", "ENCLAVE", "AVENUE", "LAYOUT", "SECTOR", "VILLAGE", "DISTRICT"
    ]
    c_has_comp = bool(c_name and comp_keyword_regex.search(c_name))
    b_has_comp = bool(b_cname and comp_keyword_regex.search(b_cname))

    c_is_addr = bool(c_name and (any(k in c_name.upper() for k in strict_c_addr) or (not c_has_comp and any(k in c_name.upper() for k in locality_c_addr))))
    b_is_addr = bool(b_cname and (any(k in b_cname.upper() for k in strict_c_addr) or (not b_has_comp and any(k in b_cname.upper() for k in locality_c_addr))))

    is_cname_poor = (
        not c_name
        or len(c_name.strip()) < 2
        or bool(re.match(r"^\d+[-/]\d+", c_name))
        or c_is_addr
        or any(k in c_name.upper() for k in ["CLIENT NAME", "CUSTOMER NAME", "BILL TO", "BUYER NAME", "SHIP TO"])
    )
    if is_cname_poor and b_cname and not b_is_addr and b_cname.lower() != "not found":
        fused["customer_name"] = b_cname
        fused_conf["customer_name"] = 0.90
    elif c_name:
        fused["customer_name"] = re.sub(r"^(?:Name|Client\s*Name|Customer\s*Name|Buyer|M/s\.?)\s*[:.\-]?\s*", "", c_name, flags=re.IGNORECASE).strip()

    # 4. GSTIN numbers
    b_vgst = baseline_fields.get("vendor_gstin") or baseline_fields.get("gst_number")
    b_cgst = baseline_fields.get("customer_gstin")
    if b_vgst and b_vgst != "Not found":
        fused["vendor_gstin"] = b_vgst
        fused["gst_number"] = b_vgst
        fused_conf["vendor_gstin"] = 0.95
        fused_conf["gst_number"] = 0.95
    elif not fused.get("vendor_gstin") and fused.get("gst_number"):
        fused["vendor_gstin"] = fused["gst_number"]

    if b_cgst and b_cgst != "Not found":
        fused["customer_gstin"] = b_cgst
        fused_conf["customer_gstin"] = 0.95

    # 5. Statutory taxes and rates (CGST, SGST, IGST)
    for tax_k in ("cgst_amount", "sgst_amount", "igst_amount", "cgst_rate", "sgst_rate", "igst_rate"):
        b_val = baseline_fields.get(tax_k)
        if b_val and b_val != "Not found" and not fused.get(tax_k):
            fused[tax_k] = b_val
            fused_conf[tax_k] = 0.90

    # 6. Total Amount
    b_tot_val = None
    try:
        if baseline_fields.get("total_amount") and baseline_fields["total_amount"] != "Not found":
            b_tot_val = float(baseline_fields["total_amount"])
    except (ValueError, TypeError):
        pass

    cur_tot_val = None
    try:
        if fused.get("total_amount"):
            cur_tot_val = float(fused["total_amount"])
    except (ValueError, TypeError):
        pass

    cur_sub_val = None
    try:
        if fused.get("subtotal"):
            cur_sub_val = float(fused["subtotal"])
    except (ValueError, TypeError):
        pass

    if not fused.get("total_amount") and b_tot_val is not None:
        fused["total_amount"] = baseline_fields["total_amount"]
        fused_conf["total_amount"] = 0.88
    elif cur_tot_val is not None and cur_sub_val is not None and abs(cur_tot_val - cur_sub_val) <= 0.05:
        if b_tot_val is not None and b_tot_val > cur_sub_val:
            fused["total_amount"] = f"{b_tot_val:.2f}"
            fused_conf["total_amount"] = 0.90

    # 7. Tax Amount & Subtotal Inversion Correction
    tot = 0.0
    try:
        tot = float(fused.get("total_amount") or baseline_fields.get("total_amount") or 0.0)
    except (ValueError, TypeError):
        pass

    cur_tax = 0.0
    try:
        cur_tax = float(fused.get("tax_amount") or 0.0)
    except (ValueError, TypeError):
        pass

    b_tax = None
    try:
        if baseline_fields.get("tax_amount"):
            b_tax = float(baseline_fields["tax_amount"])
    except (ValueError, TypeError):
        pass

    # If tax is missing or disproportionate (>= 70% of total) while baseline has a reasonable tax or cgst+sgst
    if tot > 0 and (cur_tax >= tot * 0.70 or cur_tax == 0):
        if b_tax is not None and 0 < b_tax < tot * 0.70:
            fused["tax_amount"] = f"{b_tax:.2f}"
            fused_conf["tax_amount"] = 0.90
        elif fused.get("cgst_amount") and fused.get("sgst_amount"):
            try:
                tot_c_s = round(float(fused["cgst_amount"]) + float(fused["sgst_amount"]), 2)
                if 0 < tot_c_s < tot * 0.70:
                    fused["tax_amount"] = f"{tot_c_s:.2f}"
                    fused_conf["tax_amount"] = 0.90
            except ValueError:
                pass

    # 8. Subtotal
    if not fused.get("subtotal") and baseline_fields.get("subtotal"):
        fused["subtotal"] = baseline_fields["subtotal"]
        fused_conf["subtotal"] = 0.88
    elif tot > 0 and fused.get("tax_amount"):
        try:
            cur_sub = float(fused.get("subtotal") or 0.0)
            cur_tax = float(fused["tax_amount"])
            disc = float(fused.get("discount") or 0.0)
            if cur_sub <= 0 or (cur_sub - disc > tot + 0.05):
                rec_sub = round(tot - cur_tax + disc, 2)
                if rec_sub > 0:
                    fused["subtotal"] = f"{rec_sub:.2f}"
                    fused_conf["subtotal"] = 0.85
        except (ValueError, TypeError):
            pass

    # 9. Invoice Date, Due Date, Currency
    if not fused.get("invoice_date") and baseline_fields.get("invoice_date"):
        fused["invoice_date"] = baseline_fields["invoice_date"]
        fused_conf["invoice_date"] = 0.88

    if not fused.get("due_date") and baseline_fields.get("due_date"):
        fused["due_date"] = baseline_fields["due_date"]
        fused_conf["due_date"] = 0.85

    if not fused.get("currency") and baseline_fields.get("currency"):
        fused["currency"] = baseline_fields["currency"]
        fused_conf["currency"] = 0.90

    return fused, fused_conf


def _calculate_field_coverage(fields: dict | None) -> float:
    """Calculate field coverage percentage against standard canonical invoice fields."""
    if not fields:
        return 0.0
    canonical_keys = [
        "company_name", "invoice_number", "invoice_date", "gst_number",
        "subtotal", "tax_amount", "total_amount", "currency"
    ]
    found = 0
    for key in canonical_keys:
        val = fields.get(key)
        if key == "company_name" and not val:
            val = fields.get("vendor_name")
        if val is not None and str(val).strip() and str(val).strip().lower() not in {"not found", "none", "null"}:
            found += 1
    return round((found / len(canonical_keys)) * 100.0, 1)


def _calculate_avg_confidence(confidences: dict | None) -> float | None:
    """Calculate average percentage confidence from model token classification scores."""
    if not confidences:
        return None
    vals = [float(v) for v in confidences.values() if isinstance(v, (int, float))]
    if not vals:
        return None
    avg = sum(vals) / len(vals)
    if avg <= 1.0:
        avg *= 100.0
    return round(avg, 1)


def _public_document(document, include_text=True):
    is_layoutlm = (
        document.get("extraction_method") == "layoutlm"
        or bool(document.get("layoutlm_fields"))
    )

    baseline_extracted = document.get("extracted_fields") or {}
    baseline_invoice = document.get("invoice_fields") or {}
    layoutlm_fields = document.get("layoutlm_fields")
    layoutlm_conf = document.get("layoutlm_confidence")

    # Determine active fields, coverage, and confidence based on extraction_method
    if is_layoutlm and layoutlm_fields:
        active_method = "layoutlm"
        # If document was human-verified with custom edits, preserve verified invoice_fields
        if document.get("verified") and baseline_invoice:
            active_fields = dict(baseline_invoice)
        elif baseline_invoice and baseline_invoice != baseline_extracted and baseline_invoice != layoutlm_fields:
            # Human edits saved as draft prior to verification
            active_fields = dict(baseline_invoice)
        else:
            active_fields = dict(layoutlm_fields)
            if "company_name" not in active_fields and "vendor_name" in active_fields:
                active_fields["company_name"] = active_fields["vendor_name"]
            if "vendor_name" not in active_fields and "company_name" in active_fields:
                active_fields["vendor_name"] = active_fields["company_name"]

        coverage_score = _calculate_field_coverage(active_fields)
        avg_model_conf = _calculate_avg_confidence(layoutlm_conf)
        conf_score = avg_model_conf if avg_model_conf is not None else coverage_score
        conf_method = "layoutlmv3_token_classification"
        ai_model = "LayoutLMv3"
    else:
        active_method = "baseline"
        active_fields = dict(baseline_invoice or baseline_extracted)
        coverage_score = document.get("confidence_score") or _calculate_field_coverage(active_fields)
        conf_score = coverage_score
        avg_model_conf = None
        conf_method = document.get("confidence_method") or "heuristic_field_coverage"
        ai_model = "Baseline OCR"

    payload = {
        "id": document["id"],
        "filename": document["filename"],
        "document_type": document["document_type"],
        "uploaded_at": document["uploaded_at"],
        "processing_status": document["processing_status"],
        "invoice_fields": active_fields,
        "extracted_fields": baseline_extracted,
        "baseline_fields": baseline_extracted,
        "baseline_invoice_fields": baseline_invoice,
        "layoutlm_fields": layoutlm_fields,
        "layoutlm_confidence": layoutlm_conf,
        "extraction_method": active_method,
        "line_items": document.get("line_items", []),
        "validation_results": document.get("validation_results"),
        "confidence_score": conf_score,
        "coverage_score": coverage_score,
        "confidence_method": conf_method,
        "model_confidence": avg_model_conf,
        "verified": document["verified"],
        "verified_at": document["verified_at"],
        "processing_error": document["processing_error"],
        "page_count": document["page_count"],
        "pages_processed": document["pages_processed"],
        "text_source": document["text_source"],
        "message": document["message"],
        "ai_model": ai_model,
        "ground_truth": document["ground_truth"],
    }
    if include_text:
        payload["extracted_text"] = document.get("extracted_text", "")
    else:
        payload["extracted_text_preview"] = document.get(
            "extracted_text_preview",
            (document.get("extracted_text") or "")[:240],
        )
    return payload


def _upload_response(document):
    payload = _public_document(document, include_text=True)
    return {
        "message": payload["message"],
        "filename": payload["filename"],
        "document_type": payload["document_type"],
        "extracted_text": payload["extracted_text"],
        "invoice_fields": payload["invoice_fields"],
        **payload,
    }


def _require_document(document_id):
    document, file_path = get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document, file_path


def _recalculate(document_type, invoice_fields, extracted_fields):
    if document_type == "Receipt":
        values = [extracted_fields.get(key) for key in RECEIPT_FIELD_KEYS]
    elif document_type == "Form":
        pairs = extracted_fields.get("fields") or []
        found = sum(
            1
            for pair in pairs
            if isinstance(pair, dict) and value_present(pair.get("value"))
        )
        expected = max(len(pairs), 1) + 1
        type_found = 1 if value_present(extracted_fields.get("form_type")) else 0
        score = round(((found + type_found) / expected) * 100, 2)
        invoice_fields["confidence_score"] = score
        invoice_fields["confidence_method"] = CONFIDENCE_METHOD
        return score
    else:
        values = [invoice_fields.get(key) for key in INVOICE_FIELD_KEYS]

    score = coverage_score(values)
    invoice_fields["confidence_score"] = score
    invoice_fields["confidence_method"] = CONFIDENCE_METHOD
    return score


def _sync_structured_fields(document_type, invoice_fields, extracted_fields):
    mapping = {
        "company_name": "vendor_name",
        "invoice_number": "invoice_number",
        "invoice_date": "invoice_date",
        "gst_number": "gst_number",
        "subtotal": "subtotal",
        "tax_amount": "tax_amount",
        "total_amount": "total_amount",
        "currency": "currency",
    }
    if document_type == "Receipt":
        mapping = {
            "company_name": "merchant_name",
            "invoice_number": "receipt_number",
            "invoice_date": "date",
            "tax_amount": "tax_amount",
            "total_amount": "total_amount",
        }

    for legacy_key, structured_key in mapping.items():
        if legacy_key not in invoice_fields:
            continue
        value = invoice_fields.get(legacy_key)
        extracted_fields[structured_key] = (
            None if not value_present(value) or value == NOT_FOUND else value
        )


def _normalize(value):
    if not value_present(value):
        return ""
    text = str(value).strip().upper()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text


def _flatten_prediction(document):
    predicted = {}
    extracted = document["extracted_fields"] or {}
    for key, value in extracted.items():
        if key == "fields" and isinstance(value, list):
            for pair in value:
                if isinstance(pair, dict) and pair.get("label"):
                    predicted[str(pair["label"])] = pair.get("value")
        else:
            predicted[key] = value

    aliases = {
        "vendor_name": "company_name",
        "merchant_name": "company_name",
        "receipt_number": "invoice_number",
        "date": "invoice_date",
    }
    for source_key, alias in aliases.items():
        if source_key in predicted and alias not in predicted:
            predicted[alias] = predicted[source_key]

    for key, value in (document["invoice_fields"] or {}).items():
        predicted.setdefault(key, value)
    return predicted


def _evaluate():
    documents = labeled_documents()
    compared = 0
    exact_matches = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    per_document = []

    for document in documents:
        ground_truth = document["ground_truth"]
        fields = ground_truth.get("fields", ground_truth)
        if not isinstance(fields, dict):
            continue

        predicted = _flatten_prediction(document)
        document_compared = 0
        document_matches = 0

        for key, expected in fields.items():
            if not value_present(expected):
                continue
            document_compared += 1
            actual = predicted.get(key)
            if _normalize(actual) == _normalize(expected):
                exact_matches += 1
                true_positives += 1
                document_matches += 1
            elif value_present(actual):
                false_positives += 1
                false_negatives += 1
            else:
                false_negatives += 1

        compared += document_compared
        if document_compared:
            per_document.append(
                {
                    "id": document["id"],
                    "filename": document["filename"],
                    "compared_fields": document_compared,
                    "exact_matches": document_matches,
                }
            )

    if compared == 0:
        return {
            "labeled_documents": len(documents),
            "metrics_available": False,
            "message": (
                "Evaluation cannot be calculated because no labeled "
                "ground-truth field values are available."
            ),
            "compared_fields": 0,
            "exact_matches": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "documents": per_document,
        }

    accuracy = round(exact_matches / compared, 4)
    precision = (
        round(true_positives / (true_positives + false_positives), 4)
        if true_positives + false_positives
        else None
    )
    recall = (
        round(true_positives / (true_positives + false_negatives), 4)
        if true_positives + false_negatives
        else None
    )
    if precision is None or recall is None or precision + recall == 0:
        f1 = 0.0 if precision == 0 and recall == 0 else None
    else:
        f1 = round(2 * precision * recall / (precision + recall), 4)

    return {
        "labeled_documents": len(per_document),
        "metrics_available": True,
        "message": "Metrics are exact-match scores against saved ground truth.",
        "compared_fields": compared,
        "exact_matches": exact_matches,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "documents": per_document,
    }


# ── Core Endpoints ────────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message": "Document Processing API is running",
        "version": "2.0.0",
        "objectives": {
            "O1": "Multi-format OCR ingestion & heuristic baseline",
            "O2": "LayoutLMv3 token classification & key-field extraction",
            "O3": "Document question answering & confidence scoring",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "intelligent-document-processing-api",
        "version": "2.0.0",
    }


def ingest_document(filename: str, contents: bytes) -> dict:
    """Ingest, process, and persist an uploaded document (core logic for /upload)."""
    original_name = _safe_filename(filename)
    extension = Path(original_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported file type",
                "filename": original_name,
            },
        )

    if not contents:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Uploaded file is empty",
                "filename": original_name,
            },
        )
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "message": f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                "filename": original_name,
            },
        )

    document_id = uuid.uuid4().hex
    stored_filename = f"{document_id}_{original_name}"
    file_path = UPLOAD_DIR / stored_filename
    file_path.write_bytes(contents)

    processed = process_file(str(file_path), extension)
    record = {
        "id": document_id,
        "filename": original_name,
        "stored_filename": stored_filename,
        "file_path": str(file_path),
        "content_type": CONTENT_TYPES.get(extension),
        "uploaded_at": _now(),
        "verified": False,
        "verified_at": None,
        "ground_truth": None,
        **processed,
    }
    insert_document(record)

    # Automatically execute LayoutLMv3 if model is available, so that uploaded documents
    # immediately benefit from spatial AI extraction without needing manual trigger
    auto_extract = os.environ.get("IDP_AUTO_LAYOUTLM_UPLOAD", "1") == "1"
    if auto_extract:
        try:
            llm = get_layoutlm()
            if llm and llm.is_available():
                page_entries = _load_document_images(str(file_path))
                llm_result = llm.extract_pages(
                    images=[img for _, img in page_entries],
                    ocr_text=processed.get("extracted_text", ""),
                    document_type=processed.get("document_type", "Unknown"),
                )
                base_fields = {**(processed.get("extracted_fields") or {}), **(processed.get("invoice_fields") or {})}
                fused_fields, fused_confs = _fuse_hybrid_fields(llm_result.fields, base_fields, llm_result.confidence_scores)

                avg_conf = None
                if fused_confs:
                    scores = [v for v in fused_confs.values() if isinstance(v, (int, float))]
                    if scores:
                        val = sum(scores) / len(scores)
                        avg_conf = round(val * 100.0 if val <= 1.0 else val, 1)

                val_dict = getattr(llm_result, "validation", None)
                if not val_dict or (fused_fields != llm_result.fields):
                    from app.validation import validate_financials
                    fin_res = validate_financials(fused_fields, line_items=getattr(llm_result, "line_items", None))
                    val_dict = dict(val_dict or {})
                    val_dict["financials"] = fin_res.to_dict()
                    val_dict["is_valid"] = fin_res.is_valid and all(d.get("is_valid", True) for d in val_dict.get("dates", {}).values())
                    val_dict["suspicious"] = fin_res.suspicious or any(d.get("is_ambiguous", False) or not d.get("is_valid", True) for d in val_dict.get("dates", {}).values())
                    val_dict["validation_errors"] = list(fin_res.validation_errors)

                update_extraction_method(
                    document_id=document_id,
                    method="layoutlm",
                    layoutlm_fields=fused_fields,
                    layoutlm_confidence=fused_confs,
                    confidence_score=avg_conf,
                    confidence_method="layoutlmv3_token_classification",
                    line_items=getattr(llm_result, "line_items", None),
                    validation_results=val_dict,
                )
        except Exception as exc:
            logger.warning("Auto LayoutLM on upload failed: %s", exc)

    document, _ = get_document(document_id)
    return _upload_response(document)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()
    return ingest_document(file.filename or "document", contents)


@app.get("/documents")
def get_documents(
    q: str | None = Query(default=None),
    document_type: str | None = Query(default=None),
    verified: bool | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    if hasattr(status, "default"):
        status = status.default
    if hasattr(q, "default"):
        q = q.default
    if hasattr(document_type, "default"):
        document_type = document_type.default
    if hasattr(verified, "default"):
        verified = verified.default
    if hasattr(limit, "default"):
        limit = limit.default
    if hasattr(offset, "default"):
        offset = offset.default

    if status and status not in {"processed", "failed"}:
        raise HTTPException(
            status_code=400,
            detail="status must be 'processed' or 'failed'",
        )

    result = list_documents(
        query=q,
        document_type=document_type,
        verified=verified,
        status=status,
        limit=limit,
        offset=offset,
    )
    result["documents"] = [
        _public_document(document, include_text=False)
        for document in result["documents"]
    ]
    return result


@app.get("/documents/{document_id}")
def get_document_by_id(document_id: str):
    document, _ = _require_document(document_id)
    return _public_document(document, include_text=True)


@app.patch("/documents/{document_id}")
def update_document_fields(document_id: str, update: FieldUpdate):
    if update.invoice_fields is None and update.extracted_fields is None:
        raise HTTPException(
            status_code=400,
            detail="Provide invoice_fields or extracted_fields",
        )

    document, _ = _require_document(document_id)
    invoice_fields = dict(document["invoice_fields"] or empty_invoice_fields())
    extracted_fields = dict(document["extracted_fields"] or {})

    # Audit logging for human corrections
    now_ts = _now()
    if update.invoice_fields:
        for field_name, new_val in update.invoice_fields.items():
            old_val = invoice_fields.get(field_name)
            if str(old_val) != str(new_val):
                insert_audit_entry(
                    document_id=document_id,
                    field_name=field_name,
                    old_value=str(old_val) if old_val is not None else None,
                    new_value=str(new_val) if new_val is not None else None,
                    confidence=invoice_fields.get("confidence_score"),
                    action="edit",
                    timestamp=now_ts,
                )
        invoice_fields.update(update.invoice_fields)
        _sync_structured_fields(
            document["document_type"],
            invoice_fields,
            extracted_fields,
        )

    if update.extracted_fields:
        for field_name, new_val in update.extracted_fields.items():
            old_val = extracted_fields.get(field_name)
            if str(old_val) != str(new_val):
                insert_audit_entry(
                    document_id=document_id,
                    field_name=f"structured.{field_name}",
                    old_value=str(old_val) if old_val is not None else None,
                    new_value=str(new_val) if new_val is not None else None,
                    confidence=None,
                    action="edit",
                    timestamp=now_ts,
                )
        extracted_fields.update(update.extracted_fields)

    score = _recalculate(
        document["document_type"],
        invoice_fields,
        extracted_fields,
    )
    update_fields(document_id, invoice_fields, extracted_fields, score)
    updated, _ = get_document(document_id)
    return _public_document(updated, include_text=True)


@app.post("/documents/{document_id}/verify")
def verify_document(
    document_id: str,
    request: VerifyRequest | None = Body(default=None),
):
    document, _ = _require_document(document_id)
    verified = True if request is None else request.verified
    now_ts = _now()
    set_verified(document_id, verified, now_ts if verified else None)

    insert_audit_entry(
        document_id=document_id,
        field_name="__all__",
        old_value="unverified" if not document["verified"] else "verified",
        new_value="verified" if verified else "unverified",
        confidence=document.get("confidence_score"),
        action="approve" if verified else "reject",
        timestamp=now_ts,
    )

    updated, _ = get_document(document_id)
    return _public_document(updated, include_text=True)


@app.put("/documents/{document_id}/ground-truth")
def save_ground_truth(document_id: str, request: GroundTruthRequest):
    _require_document(document_id)
    if not request.fields:
        raise HTTPException(
            status_code=400,
            detail="Ground truth must include at least one field",
        )
    set_ground_truth(document_id, {"fields": request.fields})
    document, _ = get_document(document_id)
    return _public_document(document, include_text=False)


@app.get("/documents/{document_id}/file")
def download_document(document_id: str):
    document, file_path = _require_document(document_id)
    path = Path(file_path)
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Original file is no longer available",
        )
    return FileResponse(
        path,
        media_type=document["content_type"] or "application/octet-stream",
        content_disposition_type="inline",
        filename=document["filename"],
    )


@app.delete("/documents/{document_id}")
def remove_document(document_id: str):
    file_path = delete_document(document_id)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Document not found")

    path = Path(file_path)
    if path.is_file():
        path.unlink()

    return {
        "message": "Document deleted",
        "id": document_id,
    }


@app.get("/stats")
def get_stats():
    return dashboard_stats()


@app.get("/evaluation")
def get_evaluation():
    return _evaluate()


# ── O3: Document Question Answering Endpoint ──────────────────────────

@app.post("/documents/{document_id}/qa")
def ask_document_question(document_id: str, request: QARequest):
    """Answer questions about a document using hybrid extractive QA (O3), searching across structured fields, document text, and multi-page visual layout."""
    document, file_path = _require_document(document_id)

    try:
        page_entries = _load_document_images(file_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load document image for QA: {exc}",
        )

    qa_engine = get_qa()
    result = qa_engine.answer_document(
        document=document,
        file_path=file_path,
        question=request.question,
        page_entries=page_entries,
    )

    conf_level = classify_confidence(result.confidence)

    return {
        "document_id": document_id,
        "question": result.question,
        "answer": result.answer,
        "confidence": result.confidence,
        "confidence_level": conf_level,
        "source_text": result.source_text,
        "explanation": result.explanation,
        "model": result.model_name,
        "source_type": getattr(result, "source_type", None) or "layoutlm_docqa",
        "field_name": getattr(result, "field_name", None),
        "page_number": result.page or 1,
        "pages_searched": len(page_entries),
    }


# ── O2: LayoutLMv3 Advanced Extraction Endpoint ───────────────────────

@app.post("/documents/{document_id}/extract/layoutlm")
def extract_with_layoutlm(document_id: str):
    """Run Layout-Aware LayoutLMv3 Extraction on a document across all pages (O2)."""
    document, file_path = _require_document(document_id)

    try:
        page_entries = _load_document_images(file_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load document image for LayoutLM: {exc}",
        )

    extractor = get_layoutlm()
    result = extractor.extract_pages(
        images=[img for _, img in page_entries],
        ocr_text=document.get("extracted_text", ""),
        document_type=document.get("document_type", "Unknown"),
    )

    base_fields = {**(document.get("extracted_fields") or {}), **(document.get("invoice_fields") or {})}
    fused_fields, fused_confs = _fuse_hybrid_fields(result.fields, base_fields, result.confidence_scores)

    avg_conf = None
    if fused_confs:
        scores = [v for v in fused_confs.values() if isinstance(v, (int, float))]
        if scores:
            val = sum(scores) / len(scores)
            avg_conf = round(val * 100.0 if val <= 1.0 else val, 1)

    val_dict = getattr(result, "validation", None)
    if not val_dict or (fused_fields != result.fields):
        from app.validation import validate_financials
        fin_res = validate_financials(fused_fields, line_items=getattr(result, "line_items", None))
        val_dict = dict(val_dict or {})
        val_dict["financials"] = fin_res.to_dict()
        val_dict["is_valid"] = fin_res.is_valid and all(d.get("is_valid", True) for d in val_dict.get("dates", {}).values())
        val_dict["suspicious"] = fin_res.suspicious or any(d.get("is_ambiguous", False) or not d.get("is_valid", True) for d in val_dict.get("dates", {}).values())
        val_dict["validation_errors"] = list(fin_res.validation_errors)

    # Persist the advanced extraction alongside the baseline
    update_extraction_method(
        document_id=document_id,
        method="layoutlm",
        layoutlm_fields=fused_fields,
        layoutlm_confidence=fused_confs,
        confidence_score=avg_conf,
        confidence_method="layoutlmv3_token_classification",
        line_items=getattr(result, "line_items", None),
        validation_results=val_dict,
    )

    updated, _ = get_document(document_id)
    return {
        "document_id": document_id,
        "method": result.method,
        "model_name": result.model_name,
        "fields": result.fields,
        "confidence_scores": result.confidence_scores,
        "line_items": getattr(result, "line_items", []),
        "validation": getattr(result, "validation", None),
        "processing_time": result.processing_time,
        "metadata": result.metadata,
        "document": _public_document(updated, include_text=False),
    }


# ── Audit Trail Endpoint ──────────────────────────────────────────────

@app.get("/documents/{document_id}/audit-log")
def get_document_audit_log(document_id: str):
    """Retrieve the human-in-the-loop audit history for a document."""
    _require_document(document_id)
    return get_audit_log(document_id)


# ── Model Status & Lifecycle Endpoints ────────────────────────────────

@app.get("/models/status", response_model=ModelStatusResponse)
def get_models_status():
    """Report availability and loaded status of ML models."""
    layoutlm = get_layoutlm()
    qa = get_qa()
    return ModelStatusResponse(
        layoutlm_available=layoutlm.is_available(),
        layoutlm_loaded=layoutlm.is_loaded(),
        layoutlm_model=LAYOUTLM_MODEL,
        qa_available=qa.is_available(),
        qa_loaded=qa.is_loaded(),
        qa_model=QA_MODEL,
        device="cpu",
    )


@app.post("/models/load")
def load_models(model: str = Body(default="all", embed=True)):
    """Pre-load ML models into memory to reduce first-request latency."""
    loaded = []
    if model in {"all", "layoutlm"}:
        llm = get_layoutlm()
        if llm.is_available():
            llm.load()
            loaded.append("layoutlm")
    if model in {"all", "qa"}:
        qa = get_qa()
        if qa.is_available():
            qa.load()
            loaded.append("qa")

    return {
        "requested": model,
        "loaded": loaded,
        "status": "ready" if loaded else "unavailable (install torch & transformers to enable)",
    }
