"""Centralised configuration loaded from environment variables.

Every configurable value lives here so the rest of the application
imports ``config.<NAME>`` instead of reading ``os.environ`` directly.
A ``.env`` file in the backend root is loaded automatically when
python-dotenv is installed; otherwise the system environment is used.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path, override=False)
except ImportError:
    pass

# Guard pyarrow if blocked by system security policies on Windows
try:
    import pyarrow.compute  # noqa: F401
except (ImportError, OSError):
    import sys
    sys.modules["pyarrow"] = None
    sys.modules["pyarrow.compute"] = None

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("IDP_DATA_DIR", str(BASE_DIR / "data")))
UPLOAD_DIR = Path(os.environ.get("IDP_UPLOAD_DIR", str(BASE_DIR / "uploads")))
DATABASE_PATH = Path(
    os.environ.get("IDP_DATABASE_PATH", str(DATA_DIR / "documents.db"))
)

# ── Tesseract ────────────────────────────────────────────────────────
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
TESSERACT_PSM = int(os.environ.get("IDP_TESSERACT_PSM", "3"))
TESSERACT_OEM = int(os.environ.get("IDP_TESSERACT_OEM", "3"))

# ── Model configuration (O2 + O3) ───────────────────────────────────
LAYOUTLM_MODEL = os.environ.get(
    "IDP_LAYOUTLM_MODEL",
    "checkpoints/layoutlmv3-fatura2",
)
FALLBACK_LAYOUTLM_MODEL = os.environ.get(
    "IDP_FALLBACK_LAYOUTLM_MODEL",
    "nielsr/layoutlmv3-finetuned-funsd",
)
INVOICE_MODEL_DIR = Path(
    os.environ.get("IDP_INVOICE_MODEL_DIR", str(BASE_DIR / "checkpoints" / "layoutlmv3-fatura2"))
)


def resolve_model_path(model_identifier: str) -> str:
    """Resolve a model identifier to an absolute directory path if local, else return as-is."""
    if not model_identifier:
        return model_identifier
    p = Path(model_identifier)
    if p.is_dir():
        return str(p.resolve())
    base_p = BASE_DIR / p
    if base_p.is_dir():
        return str(base_p.resolve())
    return model_identifier

QA_MODEL = os.environ.get(
    "IDP_QA_MODEL",
    "impira/layoutlm-document-qa",
)
MODELS_CACHE_DIR = Path(
    os.environ.get("IDP_MODELS_CACHE", str(BASE_DIR / "model_cache"))
)

# ── Confidence thresholds ────────────────────────────────────────────
CONFIDENCE_HIGH = float(os.environ.get("IDP_CONFIDENCE_HIGH", "0.80"))
CONFIDENCE_MEDIUM = float(os.environ.get("IDP_CONFIDENCE_MEDIUM", "0.50"))

# ── Upload limits ────────────────────────────────────────────────────
MAX_UPLOAD_MB = int(os.environ.get("IDP_MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".pdf": "application/pdf",
}

# ── Server ───────────────────────────────────────────────────────────
HOST = os.environ.get("IDP_HOST", "127.0.0.1")
PORT = int(os.environ.get("IDP_PORT", "8000"))
LOG_LEVEL = os.environ.get("IDP_LOG_LEVEL", "INFO").upper()
