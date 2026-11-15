"""SQLite storage for processed documents.

Queries stay in this module so the storage layer can later be swapped
for PostgreSQL without changing the API routes. Placeholders use SQLite's
``?`` style.
"""

import json
import os
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(
    os.environ.get("IDP_DATA_DIR", str(BASE_DIR / "data"))
)
UPLOAD_DIR = Path(
    os.environ.get("IDP_UPLOAD_DIR", str(BASE_DIR / "uploads"))
)
DATABASE_PATH = Path(
    os.environ.get("IDP_DATABASE_PATH", str(DATA_DIR / "documents.db"))
)


def _connect():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                content_type TEXT,
                document_type TEXT,
                uploaded_at TEXT NOT NULL,
                processing_status TEXT NOT NULL,
                extracted_fields TEXT,
                invoice_fields TEXT,
                ocr_text TEXT,
                confidence_score REAL,
                confidence_method TEXT,
                verified INTEGER NOT NULL DEFAULT 0,
                verified_at TEXT,
                processing_error TEXT,
                ground_truth TEXT,
                page_count INTEGER,
                pages_processed INTEGER,
                text_source TEXT,
                message TEXT
            )
            """
        )


def _loads(value, fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def row_to_document(row, include_text=True):
    if row is None:
        return None

    text = row["ocr_text"] or ""
    invoice_fields = _loads(row["invoice_fields"], {})
    extracted_fields = _loads(row["extracted_fields"], {})

    document = {
        "id": row["id"],
        "filename": row["original_filename"],
        "stored_filename": row["stored_filename"],
        "content_type": row["content_type"],
        "document_type": row["document_type"],
        "uploaded_at": row["uploaded_at"],
        "processing_status": row["processing_status"],
        "extracted_fields": extracted_fields,
        "invoice_fields": invoice_fields,
        "confidence_score": row["confidence_score"],
        "confidence_method": row["confidence_method"],
        "model_confidence": None,
        "verified": bool(row["verified"]),
        "verified_at": row["verified_at"],
        "processing_error": row["processing_error"],
        "ground_truth": _loads(row["ground_truth"], None),
        "page_count": row["page_count"],
        "pages_processed": row["pages_processed"],
        "text_source": row["text_source"],
        "message": row["message"],
        "ai_model": None,
    }

    if include_text:
        document["extracted_text"] = text
    else:
        document["extracted_text_preview"] = text[:240]

    return document


def insert_document(record):
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                id, original_filename, stored_filename, file_path,
                content_type, document_type, uploaded_at,
                processing_status, extracted_fields, invoice_fields,
                ocr_text, confidence_score, confidence_method,
                verified, verified_at, processing_error, ground_truth,
                page_count, pages_processed, text_source, message
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                record["id"],
                record["filename"],
                record["stored_filename"],
                record["file_path"],
                record.get("content_type"),
                record.get("document_type"),
                record["uploaded_at"],
                record["processing_status"],
                json.dumps(record.get("extracted_fields") or {}),
                json.dumps(record.get("invoice_fields") or {}),
                record.get("extracted_text") or "",
                record.get("confidence_score"),
                record.get("confidence_method"),
                1 if record.get("verified") else 0,
                record.get("verified_at"),
                record.get("processing_error"),
                json.dumps(record["ground_truth"])
                if record.get("ground_truth") is not None
                else None,
                record.get("page_count"),
                record.get("pages_processed"),
                record.get("text_source"),
                record.get("message"),
            ),
        )


def get_document(document_id):
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    return row_to_document(row, include_text=True), (
        None if row is None else row["file_path"]
    )


def list_documents(
    query=None,
    document_type=None,
    verified=None,
    status=None,
    limit=50,
    offset=0,
):
    clauses = []
    parameters = []

    if query:
        escaped = (
            query.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        like = f"%{escaped}%"
        clauses.append(
            """
            (
                original_filename LIKE ? ESCAPE '\\'
                OR document_type LIKE ? ESCAPE '\\'
                OR IFNULL(ocr_text, '') LIKE ? ESCAPE '\\'
                OR IFNULL(extracted_fields, '') LIKE ? ESCAPE '\\'
                OR IFNULL(invoice_fields, '') LIKE ? ESCAPE '\\'
            )
            """
        )
        parameters.extend([like, like, like, like, like])

    if document_type:
        clauses.append("LOWER(document_type) = LOWER(?)")
        parameters.append(document_type)

    if verified is not None:
        clauses.append("verified = ?")
        parameters.append(1 if verified else 0)

    if status:
        clauses.append("processing_status = ?")
        parameters.append(status)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    parameters.extend([limit, offset])

    with _connect() as connection:
        total = connection.execute(
            f"SELECT COUNT(*) AS count FROM documents {where}",
            parameters[:-2],
        ).fetchone()["count"]
        rows = connection.execute(
            f"""
            SELECT * FROM documents
            {where}
            ORDER BY uploaded_at DESC
            LIMIT ? OFFSET ?
            """,
            parameters,
        ).fetchall()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "documents": [
            row_to_document(row, include_text=False) for row in rows
        ],
    }


def update_fields(document_id, invoice_fields, extracted_fields, confidence_score):
    with _connect() as connection:
        cursor = connection.execute(
            """
            UPDATE documents
            SET invoice_fields = ?,
                extracted_fields = ?,
                confidence_score = ?,
                verified = 0,
                verified_at = NULL
            WHERE id = ?
            """,
            (
                json.dumps(invoice_fields),
                json.dumps(extracted_fields),
                confidence_score,
                document_id,
            ),
        )
        return cursor.rowcount > 0


def set_verified(document_id, verified, verified_at):
    with _connect() as connection:
        cursor = connection.execute(
            """
            UPDATE documents
            SET verified = ?, verified_at = ?
            WHERE id = ?
            """,
            (1 if verified else 0, verified_at, document_id),
        )
        return cursor.rowcount > 0


def set_ground_truth(document_id, ground_truth):
    with _connect() as connection:
        cursor = connection.execute(
            """
            UPDATE documents
            SET ground_truth = ?
            WHERE id = ?
            """,
            (json.dumps(ground_truth), document_id),
        )
        return cursor.rowcount > 0


def delete_document(document_id):
    with _connect() as connection:
        row = connection.execute(
            "SELECT file_path FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            return None
        connection.execute(
            "DELETE FROM documents WHERE id = ?",
            (document_id,),
        )
        return row["file_path"]


def dashboard_stats():
    with _connect() as connection:
        total = connection.execute(
            "SELECT COUNT(*) AS count FROM documents"
        ).fetchone()["count"]
        verified = connection.execute(
            "SELECT COUNT(*) AS count FROM documents WHERE verified = 1"
        ).fetchone()["count"]
        status_rows = connection.execute(
            """
            SELECT processing_status, COUNT(*) AS count
            FROM documents
            GROUP BY processing_status
            """
        ).fetchall()
        type_rows = connection.execute(
            """
            SELECT COALESCE(document_type, 'Unknown') AS document_type,
                   COUNT(*) AS count
            FROM documents
            GROUP BY document_type
            """
        ).fetchall()
        average = connection.execute(
            """
            SELECT AVG(confidence_score) AS average
            FROM documents
            WHERE confidence_score IS NOT NULL
            """
        ).fetchone()["average"]

    return {
        "total_documents": total,
        "verified": verified,
        "unverified": total - verified,
        "by_status": {
            row["processing_status"]: row["count"] for row in status_rows
        },
        "by_type": {
            row["document_type"]: row["count"] for row in type_rows
        },
        "average_heuristic_coverage": (
            round(average, 2) if average is not None else None
        ),
        "confidence_method": "heuristic_field_coverage",
        "model_confidence": None,
    }


def labeled_documents():
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, original_filename, invoice_fields,
                   extracted_fields, ground_truth
            FROM documents
            WHERE ground_truth IS NOT NULL
            """
        ).fetchall()

    labeled = []
    for row in rows:
        ground_truth = _loads(row["ground_truth"], None)
        if not ground_truth:
            continue
        labeled.append(
            {
                "id": row["id"],
                "filename": row["original_filename"],
                "invoice_fields": _loads(row["invoice_fields"], {}),
                "extracted_fields": _loads(row["extracted_fields"], {}),
                "ground_truth": ground_truth,
            }
        )
    return labeled
