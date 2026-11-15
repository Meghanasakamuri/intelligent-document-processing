import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.database import (
    dashboard_stats,
    delete_document,
    get_document,
    init_db,
    insert_document,
    labeled_documents,
    list_documents,
    set_ground_truth,
    set_verified,
    update_fields,
    UPLOAD_DIR,
)
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


app = FastAPI(
    title="Intelligent Document Processing API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".pdf": "application/pdf",
}


class FieldUpdate(BaseModel):
    invoice_fields: dict | None = None
    extracted_fields: dict | None = None


class VerifyRequest(BaseModel):
    verified: bool = True


class GroundTruthRequest(BaseModel):
    fields: dict = Field(default_factory=dict)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(filename):
    base = os.path.basename(filename or "document")
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return (cleaned or "document")[:180]


def _public_document(document, include_text=True):
    payload = {
        "id": document["id"],
        "filename": document["filename"],
        "document_type": document["document_type"],
        "uploaded_at": document["uploaded_at"],
        "processing_status": document["processing_status"],
        "invoice_fields": document["invoice_fields"],
        "extracted_fields": document["extracted_fields"],
        "confidence_score": document["confidence_score"],
        "confidence_method": document["confidence_method"],
        "model_confidence": None,
        "verified": document["verified"],
        "verified_at": document["verified_at"],
        "processing_error": document["processing_error"],
        "page_count": document["page_count"],
        "pages_processed": document["pages_processed"],
        "text_source": document["text_source"],
        "message": document["message"],
        "ai_model": None,
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
    """Keep the fields the existing frontend already reads."""
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


@app.get("/")
def home():
    return {
        "message": "Document Processing API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "intelligent-document-processing-api"
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    original_name = _safe_filename(file.filename)
    extension = Path(original_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported file type",
                "filename": original_name,
            },
        )

    contents = await file.read()
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
                "message": "File exceeds the 20 MB upload limit",
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
    document, _ = get_document(document_id)
    return _upload_response(document)


@app.get("/documents")
def get_documents(
    q: str | None = Query(default=None),
    document_type: str | None = Query(default=None),
    verified: bool | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
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

    if update.invoice_fields:
        invoice_fields.update(update.invoice_fields)
        _sync_structured_fields(
            document["document_type"],
            invoice_fields,
            extracted_fields,
        )
    if update.extracted_fields:
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
    _require_document(document_id)
    verified = True if request is None else request.verified
    set_verified(document_id, verified, _now() if verified else None)
    document, _ = get_document(document_id)
    return _public_document(document, include_text=True)


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
