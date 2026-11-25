# Intelligent Document Processing for Forms and Invoices

A research-grade Intelligent Document Processing (IDP) system that extracts structured information from invoices, forms, and receipts using OCR, layout-aware deep learning models (LayoutLMv3 fine-tuned on the FATURA2 dataset with 47 BIO labels), multi-page document pipelines, structured line-item extraction, mathematical validation, and extractive Document Question Answering with source page attribution.

---

## Research Objectives

| Objective | Description | Implementation |
|---|---|---|
| **O1** | Multi-format OCR ingestion & heuristic baseline | PyMuPDF text stream extraction + Tesseract OCR fallback with contrast preprocessing |
| **O2** | Layout-aware document understanding & key-field extraction | LayoutLMv3 token classification fine-tuned on FATURA2 (47 BIO labels) with structured line-item extraction |
| **O3** | Multi-page document question answering with confidence scoring | LayoutLM extractive QA (`impira/layoutlm-document-qa`) searching across all document pages with source page attribution |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        React 19 Frontend (Vite)                        │
│  Dashboard │ Upload │ Review (Line Items & Validation) │ Q&A │ History  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ REST API
┌───────────────────────────────────┴────────────────────────────────────┐
│                       FastAPI Backend (Uvicorn)                        │
│  ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌─────────────────┐ ┌─────┐  │
│  │  Upload  │ │ Extraction │ │   Q&A    │ │  Verification   │ │Audit│  │
│  │ Pipeline │ │  Pipeline  │ │ Pipeline │ │    & Review     │ │Trail│  │
│  └──────────┘ └────────────┘ └──────────┘ └─────────────────┘ └─────┘  │
│       │        ┌─────┴──────┐     │              │              │      │
│       │        │            │     │              │              │      │
│  ┌────┴─────┐ ┌┴─────┐ ┌───┴──┐ ┌┴────────┐ ┌──┴───────────┐  │      │
│  │Tesseract │ │OCR+  │ │FATURA│ │LayoutLM │ │  Confidence  │  │      │
│  │  OCR &   │ │Regex │ │2-LMv3│ │Doc QA   │ │  Calibration │  │      │
│  │ PyMuPDF  │ │Base- │ │(47   │ │(Multi-  │ │  (Temp.      │  │      │
│  │(Per-Page)│ │line  │ │Labels│ │ Page)   │ │   Scaling)   │  │      │
│  └──────────┘ └──────┘ └──────┘ └─────────┘ └──────────────┘  │      │
│                           │                              │      │      │
│                    ┌──────┴──────────────────────────────┴──────┴────┐ │
│                    │     Validation (Dates & Financial Math)         │ │
│                    └──────────────────────┬──────────────────────────┘ │
│                                           │                            │
│                                    ┌──────┴──────┐                     │
│                                    │   SQLite    │                     │
│                                    │  Database   │                     │
│                                    └─────────────┘                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Backend
- **Language**: Python 3.13
- **Framework**: FastAPI + Uvicorn
- **Database**: SQLite (WAL mode, schema auto-migration)
- **OCR Engine**: Tesseract 5.4 (`pytesseract`) + PyMuPDF (`pymupdf`)
- **O2 Extraction Model**: `LayoutLMv3ForTokenClassification` fine-tuned on the **FATURA2** invoice dataset (47 BIO labels)
- **O3 DocQA Model**: `LayoutLMForQuestionAnswering` (`impira/layoutlm-document-qa`) with local-first offline caching
- **Computer Vision & Preprocessing**: Pillow (`PIL.Image`, `PIL.ImageOps.autocontrast`)
- **ML Framework**: PyTorch (CPU-optimized) + Hugging Face Transformers

### Frontend
- **Framework**: React 19
- **Build Tool**: Vite 8
- **Routing**: React Router DOM 7
- **Styling**: Modern, responsive CSS modules & inline styles (clean neutral theme, dark-mode compatible)

---

## Core Capabilities & Implemented Pipelines

### 1. Multi-Page Document Pipeline
- Ingests single-page and multi-page PDFs and images (up to 20 MB).
- Performs independent OCR and spatial bounding box extraction per page.
- Aggregates LayoutLMv3 token predictions and entity spans across all document pages into a consolidated document-level response.
- Preserves page provenance (`page_number`, `pages_searched`) for all extracted line items and QA answers.

### 2. FATURA2 LayoutLMv3 Token Classification (47 BIO Labels)
- Checkpoint: `backend/checkpoints/layoutlmv3-fatura2/`
- Full 47-label schema:
  - Header & Identification: `B/I-TITLE`, `B/I-NUMBER`, `B/I-DATE`, `B/I-DUE_DATE`, `B/I-PO_NUMBER`
  - Commercial Parties: `B/I-SELLER`, `B/I-SELLER_ADDRESS`, `B/I-SELLER_SITE`, `B/I-BILL_TO`, `B/I-SEND_TO`
  - Financial Totals: `B/I-SUBTOTAL`, `B/I-TAX`, `B/I-DISCOUNT`, `B/I-AMOUNT_DUE`, `B/I-TOTAL`
  - Structured Line Items: `B/I-TABLE`, `B/I-ITEM_DESCRIPTION`, `B/I-ITEM_QUANTITY`, `B/I-ITEM_UNIT_PRICE`, `B/I-ITEM_TOTAL`
  - Metadata: `B/I-LOGO`, `B/I-CONDITIONS`, `B/I-NOTE`, `O`
- High-speed CPU inference: ~0.8s–1.2s per page with zero network latency.

### 3. Structured Invoice Line-Item Extraction
- Reconstructs table rows by associating descriptions, quantities, unit prices, and item totals on matching horizontal bands.
- Retains sequential document-level row ordering (`row_index`) and source page numbers (`page`).
- Gracefully handles partial rows (e.g. flat fees missing quantity or unit price) without hallucination.
- Performs row-level arithmetic validation: flags discrepancies where `quantity × unit_price ≠ item_total` or when numeric values are corrupted by OCR noise.

### 4. Date Normalization & Financial Validation
- **Date Normalization**:
  - Converts recognized invoice and due dates into ISO-8601 (`YYYY-MM-DD`).
  - Supports multiple conventions: `DD/MM/YYYY`, `MM/DD/YYYY`, `YYYY-MM-DD`, `DD Month YYYY`, etc.
  - Automatically flags ambiguous calendar dates (e.g., `03/04/2026`) without guessing, warning the reviewer.
  - Preserves original raw OCR text alongside normalized dates.
- **Financial Validation**:
  - Validates invoice arithmetic: `subtotal + tax - discount + additional_charges == total`.
  - Compares the sum of line-item totals against the extracted subtotal.
  - Configurable rounding tolerance (default: ±0.05).
  - Emits descriptive validation errors and flags suspicious totals for human review.

### 5. Multi-Page Document Question Answering (O3)
- Natural-language question answering grounded in document content.
- Searches across every page of multi-page documents, returning the highest-confidence extractive answer span.
- Returns explicit page attribution: `page_number` and `pages_searched`.
- Optimized offline loading (`local_files_only=True`) eliminating unnecessary Hugging Face Hub retries.

### 6. Interactive Frontend Review & Audit Trail
- **Document Review UI** (`DocumentReview.jsx`):
  - Interactive line-items table with page indicators, quantity, price, totals, and validation status badges.
  - Financial validation card displaying extracted vs. calculated totals, difference, and arithmetic errors.
  - Date normalization pills with ambiguity warnings.
  - Draft editing and human correction workflows.
- **Document QA UI** (`DocumentQA.jsx`):
  - Multi-page answer bubble with prominent `Page X of Y` attribution and `Source: Page X (N pages searched)` footer.
- **Audit Trail**:
  - Granular change tracking in SQLite (`audit_log` table): old value, new value, confidence, action, and timestamp.

---

## Installation & Setup

### Prerequisites
- **Python**: 3.11, 3.12, or 3.13
- **Node.js**: 18+ and npm
- **Tesseract OCR**: 5.x installed and added to system `PATH`
  - Default Windows path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

### 1. Clone & Navigate
```powershell
cd c:\Users\megha\intelligent-document-processing
```

### 2. Backend Setup
```powershell
cd backend

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt
```

> [!NOTE]
> **Windows Smart App Control (SAC) Compatibility**:
> On Windows 11 with Smart App Control or Application Control policies enabled, unsigned compiled C-extensions (such as `_regex.pyd`) can be blocked by the OS. The backend includes a built-in, safe in-process fallback to standard library `re` inside `app/__init__.py`, allowing all Transformers tokenizers and models to load without compromising system security.

### 3. Frontend Setup
```powershell
cd ..\frontend
npm install
```

---

## Running the Application

### Start Backend API Server
```powershell
cd backend
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- API Base: `http://127.0.0.1:8000`
- Swagger Interactive Docs: `http://127.0.0.1:8000/docs`
- Health Endpoint: `http://127.0.0.1:8000/health`

### Start Frontend UI
```powershell
cd frontend
npm run dev
```
- Web Application: `http://localhost:5173`

---

## Testing & Verification

### Run Backend Unit Tests (48/48 Passing)
```powershell
cd backend
.\venv\Scripts\Activate.ps1
python -m unittest discover -s tests -p "test_*.py"
```

### Run 14-Step End-to-End Workflow Verification
```powershell
python scratch/verify_e2e.py
```

### Frontend Linting & Build Verification
```powershell
cd frontend
npm run lint
npm run build
```

---

## API Reference Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status |
| `GET` | `/stats` | Document counts, processing rates, average confidence |
| `POST` | `/upload` | Ingest and OCR multi-page PDF or image document |
| `GET` | `/documents` | Query paginated documents with filtering |
| `GET` | `/documents/{id}` | Retrieve document details, fields, line items, and validation results |
| `PATCH` | `/documents/{id}` | Update draft fields (recorded in audit log) |
| `POST` | `/documents/{id}/verify` | Approve/verify document and promote human edits |
| `POST` | `/documents/{id}/extract/layoutlm` | Run LayoutLMv3 extraction across all pages |
| `POST` | `/documents/{id}/qa` | Ask natural language question with multi-page search & attribution |
| `GET` | `/documents/{id}/audit-log` | Retrieve document audit trail |
| `DELETE` | `/documents/{id}` | Delete document and stored files |

---

## Scope & Limitations

1. **Synthetic FATURA2 Training Domain**:
   - The primary LayoutLMv3 model is trained on synthetic FATURA2 invoice datasets.
   - **Zero real Indian tax invoices or handwritten documents were used for training**. Real-world Indian GST invoices (e.g., thermal receipts, multi-tax CGST/SGST/IGST breakdown tables) are out of domain and not validated for production use.
2. **CPU Inference Latency**:
   - Document QA: ~0.6s–0.9s per query.
   - LayoutLMv3 Token Classification: ~0.8s–1.2s per page on CPU.
3. **Single-User Architecture**:
   - Relies on SQLite with WAL mode. For enterprise multi-tenant deployments, migrate to PostgreSQL and Redis job queues.

---

## License

This project is developed for academic and engineering research purposes.
