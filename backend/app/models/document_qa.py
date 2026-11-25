"""O3 — Document question answering with LayoutLM.

Uses ``impira/layoutlm-document-qa``, a LayoutLM v1 checkpoint
fine-tuned on DocVQA and SQuAD 2.0.  This is an **extractive** QA
model: it selects an answer span from the document's text rather than
generating one.

Architecture
------------
Document Image + Question
  → Tesseract OCR (words + bounding boxes)
  → LayoutLM tokeniser (text + layout features)
  → LayoutLM encoder → start / end span logits
  → Answer extraction + raw confidence score

Important
---------
* This model is LayoutLM **v1**, *not* LayoutLMv3.  It is used here
  specifically for the O3 (document QA) objective and is completely
  separate from the O2 LayoutLMv3 extraction model.
* The raw ``score`` returned by the pipeline is a softmax probability
  over candidate spans.  It is **not** a calibrated confidence.
  Use the ``confidence`` module for calibration.
* If the model cannot find an answer it returns a low-confidence empty
  string; we convert this to ``None`` with an explanation.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from PIL import Image

from app.models.base import BaseQA, QAResult

logger = logging.getLogger(__name__)


# ── Natural Language Question Intent Matchers ─────────────────────────────────

_INTENT_PATTERNS = [
    # Tax IDs first to avoid being intercepted by party name queries
    ("vendor_gstin", [
        r"\b(vendor|seller|supplier|company)\s*(gstin|gst|tin|vat|tax\s*id|gst\s*number)\b",
        r"\b(gstin|gst\s*number|tax\s*id)\s+of\s+(the\s+)?(vendor|seller|supplier|company)\b",
        r"^\s*(vendor|seller|supplier)\s+gst(in)?\s*(\?)?\s*$",
    ]),
    ("customer_gstin", [
        r"\b(customer|buyer|client|purchaser)\s*(gstin|gst|tin|vat|tax\s*id|gst\s*number)\b",
        r"\b(gstin|gst\s*number|tax\s*id)\s+of\s+(the\s+)?(customer|buyer|client)\b",
        r"^\s*(customer|buyer|client)\s+gst(in)?\s*(\?)?\s*$",
    ]),
    ("generic_gstin", [
        r"\b(what\s+is\s+(the\s+)?)?(gstin|gst\s*number|gst\s*no)\b",
    ]),
    # Taxes
    ("cgst_and_sgst", [
        r"\b(cgst\s*(and|&)\s*sgst|central\s*and\s*state\s*tax|central\s*gst\s*and\s*state\s*gst)\b",
    ]),
    ("cgst_amount", [
        r"\b(cgst|central\s+gst|central\s+tax)\b",
    ]),
    ("sgst_amount", [
        r"\b(sgst|state\s+gst|state\s+tax)\b",
    ]),
    ("igst_amount", [
        r"\b(igst|integrated\s+gst|integrated\s+tax)\b",
    ]),
    ("tax_amount", [
        r"\b(total\s+tax|tax\s+amount|tax\s+total|how\s+much\s+tax|sales\s+tax|vat\s+amount|gst\s+amount|tax\s+value)\b",
        r"\bwhat\s+is\s+the\s+(total\s+)?tax\b",
    ]),
    # Subtotal
    ("subtotal", [
        r"\b(subtotal|sub\s*total|taxable\s*amount|taxable\s*value|gross\s*amount|amount\s+before\s+tax|pre-?tax\s*amount|total\s+before\s+tax)\b",
    ]),
    # Total Amount
    ("total_amount", [
        r"\b(total\s+invoice\s+amount|grand\s+total|net\s+amount|net\s+payable|total\s+payable|final\s+amount|final\s+total|bill\s+total|invoice\s+total|total\s+amount|total\s+price|total\s+cost)\b",
        r"\bhow\s+much\s+(is\s+(the\s+)?(total|due|payable|bill|invoice)|do\s+i\s+(have\s+to\s+)?pay|was\s+charged)\b",
        r"\bwhat\s+is\s+the\s+(total|grand\s+total|net\s+amount|final\s+amount|payable\s+amount|total\s+price)\b",
        r"^\s*(total|grand\s*total|net\s*amount)\s*(\?)?\s*$",
    ]),
    # Parties
    ("vendor_name", [
        r"\b(who\s+is\s+(the\s+)?(vendor|seller|supplier|merchant|biller|issuer))\b",
        r"\b(who\s+(supplied|issued|sent|billed|provided)\s+(this|the)\s+(invoice|bill|document|goods|items)?)\b",
        r"\b(who\s+is\s+it\s+from|from\s+whom|who\s+sold\s+(this|to\s+us))\b",
        r"\b(vendor|seller|supplier|issuer)\s+(name|company|identity)\b",
        r"\bwhat\s+is\s+the\s+(vendor|seller|supplier|company)\s*(name)?\b",
        r"^\s*(vendor|seller|supplier|company)\s*(\?)?\s*$",
    ]),
    ("customer_name", [
        r"\b(who\s+is\s+(the\s+)?(customer|buyer|client|purchaser|recipient|consignee))\b",
        r"\b(who\s+(bought|purchased|received)\s+(this|the))\b",
        r"\b(who\s+(was|is)\s+billed|billed\s+to|bill\s+to|invoice\s+to|sold\s+to|shipped\s+to)\b",
        r"\b(customer|buyer|client|purchaser|recipient)\s+(name|identity)\b",
        r"\bwhat\s+is\s+the\s+(customer|buyer|client|purchaser)\s*(name)?\b",
        r"^\s*(customer|buyer|client|bill\s*to)\s*(\?)?\s*$",
    ]),
    # Numbers & Dates
    ("due_date", [
        r"\b(when\s+is\s+(the\s+)?payment\s+due|due\s+date|payment\s+due\s+date|pay\s+(by|before)|payment\s+deadline)\b",
        r"\bwhat\s+is\s+the\s+due\s*date\b",
    ]),
    ("invoice_date", [
        r"\b(when\s+was\s+(this|the)\s+(invoice|bill)\s+(issued|created|dated|generated|made))\b",
        r"\b(invoice|bill|issue|billing)\s*date\b",
        r"\bdate\s+of\s+(the\s+)?(invoice|bill|issue)\b",
        r"\bwhat\s+is\s+the\s+(invoice|bill|issue)?\s*date\b",
        r"^\s*(invoice|bill)?\s*date\s*(\?)?\s*$",
    ]),
    ("invoice_number", [
        r"\b(invoice|bill|receipt|inv)\s*(number|no|num|#|id|code)\b",
        r"\bwhat\s+is\s+the\s+(invoice|bill|receipt)\s*(number|no|num|#|id|code)\b",
        r"^\s*(invoice|bill|receipt)\s*(no|num|#|number)\s*(\?)?\s*$",
    ]),
    # Discount & Currency
    ("discount", [
        r"\b(discount|rebate|concession)\s*(amount)?\b",
        r"\bhow\s+much\s+discount\b",
    ]),
    ("currency", [
        r"\b(what\s+currency|which\s+currency|currency\s+used|currency\s+of\s+(the\s+)?invoice)\b",
        r"^\s*currency\s*(\?)?\s*$",
    ]),
    # Addresses
    ("customer_address", [
        r"\b(customer|buyer|client|billing|bill\s+to|consignee)\s+address\b",
        r"\bwhere\s+is\s+the\s+(customer|buyer|client)\s+located\b",
    ]),
    ("vendor_address", [
        r"\b(vendor|seller|supplier|company)\s+address\b",
        r"\bwhere\s+is\s+the\s+(vendor|seller|supplier|company)\s+located\b",
    ]),
    # Line items
    ("line_items", [
        r"\b(what\s+products|which\s+products|what\s+items|products\s+listed|line\s*items|items\s+listed|items\s+purchased|what\s+was\s+bought|what\s+was\s+purchased|list\s+the\s+(products|items)|items\s+on\s+(this|the)\s+invoice)\b",
        r"\bhow\s+many\s+(items|products|line\s*items)\b",
        r"^\s*(products|items|line\s*items)\s*(\?)?\s*$",
    ]),
    # Contextual fields
    ("ifsc_code", [r"\b(ifsc|ifsc\s*code)\b"]),
    ("account_number", [r"\b(account\s*number|account\s*no|a/c\s*no|bank\s*acc)\b"]),
    ("bank_details", [r"\b(bank\s+details|bank\s+account|bank\s+name|wire\s+transfer|payment\s+instructions)\b"]),
    ("drug_licence", [r"\b(drug\s*licen[sc]e|d\.?l\.?\s*no|20b|21b)\b"]),
    ("email", [r"\b(email|email\s*address|e-mail)\b"]),
    ("phone", [r"\b(phone|phone\s*number|telephone|cell|mobile|contact\s*number)\b"]),
    ("jurisdiction", [r"\b(jurisdiction|subject\s+to\s+jurisdiction)\b"]),
    ("payment_terms", [r"\b(payment\s+terms|terms\s+of\s+payment|credit\s+terms)\b"]),
]


def classify_question_intent(question: str) -> str:
    """Map natural language question variations to a canonical field or intent."""
    qn = question.lower().strip()
    for intent, patterns in _INTENT_PATTERNS:
        for pat in patterns:
            if re.search(pat, qn):
                return intent
    return "unknown"


def _find_grounding_evidence(text: str, val: str, fallback_field: str | None = None) -> str:
    """Extract grounding evidence from document text for an extracted answer."""
    if not text or not val:
        return f"Verified invoice canonical record: {fallback_field} = {val}" if fallback_field else str(val)
    v_clean = str(val).strip()
    idx = text.lower().find(v_clean.lower())
    if idx != -1:
        start = max(0, idx - 50)
        end = min(len(text), idx + len(v_clean) + 50)
        snippet = text[start:end].replace("\n", " ").strip()
        return f"...{snippet}..."
    words = v_clean.split()
    if len(words) > 1:
        sub = " ".join(words[:2])
        idx = text.lower().find(sub.lower())
        if idx != -1:
            start = max(0, idx - 50)
            end = min(len(text), idx + len(v_clean) + 50)
            snippet = text[start:end].replace("\n", " ").strip()
            return f"...{snippet}..."
    return f"Verified canonical invoice record: '{fallback_field}' = {val}" if fallback_field else f"Extracted value: {val}"


def _is_offline_configured() -> bool:
    """Check if offline operation is explicitly required by application configuration or environment."""
    try:
        from app.config import IDP_OFFLINE
        if IDP_OFFLINE:
            return True
    except (ImportError, AttributeError):
        pass

    for var in ("IDP_OFFLINE", "IDP_QA_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        val = os.environ.get(var, "").strip().lower()
        if val in ("1", "true", "yes", "on"):
            return True

    return False


class DocumentQA(BaseQA):
    """Extractive document QA using the Hugging Face pipeline.

    Parameters
    ----------
    model_name : str
        HF model identifier or local checkpoint path (must support ``document-question-answering``).
    cache_dir : str | None
        Where to cache downloaded weights.
    local_files_only : bool | None
        Whether to restrict model loading exclusively to local cached files.
    offline : bool | None
        Explicit toggle for offline mode.
    """

    def __init__(
        self,
        model_name: str = "impira/layoutlm-document-qa",
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
        offline: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.offline = offline
        self._pipeline = None
        self._loaded = False
        self._load_error: str | None = None

    # ── public interface ─────────────────────────────────────────────

    def is_offline_mode(self) -> bool:
        """Determine whether offline mode / local-files-only is active."""
        if self.local_files_only is not None:
            return bool(self.local_files_only)
        if self.offline is not None:
            return bool(self.offline)

        # Check explicit application configuration or environment variables
        if _is_offline_configured():
            return True

        # Check if model_name is a local directory
        try:
            from app.config import resolve_model_path
            resolved = resolve_model_path(self.model_name)
            if os.path.isdir(resolved):
                return True
        except Exception:
            if os.path.isdir(self.model_name):
                return True

        return False

    def is_available(self) -> bool:
        """Check whether runtime dependencies are present and model is not in an error state."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            if self._load_error is not None:
                return False
            return True
        except (ImportError, OSError, Exception):
            return False

    def is_loaded(self) -> bool:
        return self._loaded

    def is_cached(self) -> bool:
        """Check if model configuration is available locally without network access."""
        from transformers import AutoConfig
        if self.cache_dir:
            try:
                AutoConfig.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=True,
                )
                return True
            except Exception:
                pass
        try:
            AutoConfig.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            return True
        except Exception:
            return False

    def load(self, local_files_only: bool | None = None) -> None:
        """Eagerly load the model (called lazily on first ``answer``)."""
        if self._loaded:
            return
        if self._load_error and local_files_only is None:
            return

        self._load_error = None

        # Ensure Tesseract path is set from config if available
        try:
            import pytesseract
            from app.config import TESSERACT_CMD
            if TESSERACT_CMD and os.path.exists(TESSERACT_CMD):
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        except Exception:
            pass

        try:
            from transformers import pipeline

            use_local_only = (
                local_files_only
                if local_files_only is not None
                else self.is_offline_mode()
            )

            # If not explicitly specified otherwise and model is cached locally,
            # use local files to prevent unnecessary network latency/retries.
            if local_files_only is None and not use_local_only and self.is_cached():
                use_local_only = True

            logger.info(
                "Loading Document QA model: %s (local_files_only=%s) …",
                self.model_name,
                use_local_only,
            )
            start = time.time()

            pipeline_kwargs = {
                "task": "document-question-answering",
                "model": self.model_name,
                "cache_dir": self.cache_dir,
            }
            if use_local_only:
                pipeline_kwargs["local_files_only"] = True

            self._pipeline = pipeline(**pipeline_kwargs)

            elapsed = time.time() - start
            logger.info("Document QA loaded in %.1fs", elapsed)
            self._loaded = True
            self._load_error = None

        except Exception as exc:
            self._loaded = False
            self._pipeline = None
            exc_str = str(exc)
            if any(term in exc_str.lower() for term in ("couldn't find", "couldn't connect", "not found", "local_files_only")):
                self._load_error = (
                    f"Document QA model '{self.model_name}' is unavailable in local cache (offline mode). "
                    f"Ensure weights are pre-downloaded or allow network access. Error: {exc}"
                )
            else:
                self._load_error = f"Failed to load Document QA model '{self.model_name}': {exc}"
            logger.error("Document QA load error: %s", self._load_error)

    def answer(
        self,
        image: Image.Image,
        question: str,
        ocr_text: str = "",
    ) -> QAResult:
        self.load()

        if not self._loaded:
            return QAResult(
                question=question,
                answer=None,
                confidence=0.0,
                explanation=self._load_error or "QA model not loaded",
                model_name=self.model_name,
            )

        start = time.time()

        try:
            results = self._pipeline(image, question, top_k=1)
        except Exception as exc:
            logger.error("QA inference error: %s", exc)
            return QAResult(
                question=question,
                answer=None,
                confidence=0.0,
                explanation=f"Inference error: {exc}",
                model_name=self.model_name,
            )

        elapsed = time.time() - start

        if not results:
            return QAResult(
                question=question,
                answer=None,
                confidence=0.0,
                explanation="The model could not find an answer in this document.",
                model_name=self.model_name,
            )

        top = results[0] if isinstance(results, list) else results
        raw_answer = top.get("answer", "").strip()
        raw_score = float(top.get("score", 0.0))

        # Low-confidence or empty answer → report as not found
        if not raw_answer or raw_score < 0.01:
            return QAResult(
                question=question,
                answer=None,
                confidence=round(raw_score, 4),
                explanation=(
                    "The model could not reliably extract an answer "
                    "from this document."
                ),
                model_name=self.model_name,
            )

        # Try to locate the answer in the OCR text for context
        source_text = None
        if ocr_text and raw_answer.lower() in ocr_text.lower():
            idx = ocr_text.lower().index(raw_answer.lower())
            context_start = max(0, idx - 60)
            context_end = min(len(ocr_text), idx + len(raw_answer) + 60)
            source_text = ocr_text[context_start:context_end]

        return QAResult(
            question=question,
            answer=raw_answer,
            confidence=round(raw_score, 4),
            source_text=source_text,
            explanation=None,
            model_name=self.model_name,
        )

    def answer_pages(
        self,
        images: list[Image.Image],
        question: str,
        ocr_text: str = "",
    ) -> tuple[int, QAResult]:
        """Answer a question across multiple document page images.

        Parameters
        ----------
        images : list[Image.Image]
            List of PIL page images in page order.
        question : str
            The question to answer.
        ocr_text : str
            Full document OCR text (optional context).

        Returns
        -------
        tuple[int, QAResult]
            (best_page_number_1_based, best_qa_result)
        """
        if not images:
            return 1, QAResult(
                question=question,
                answer=None,
                confidence=0.0,
                explanation="No page images provided for Document QA.",
                model_name=self.model_name,
            )

        if len(images) == 1:
            return 1, self.answer(images[0], question, ocr_text=ocr_text)

        all_results: list[tuple[int, QAResult]] = []
        for idx, img in enumerate(images):
            res = self.answer(img, question, ocr_text=ocr_text)
            all_results.append((idx + 1, res))

        # Filter candidates with non-empty answers and positive confidence
        valid_candidates = [
            (p, r) for p, r in all_results
            if r.answer and r.confidence > 0.0
        ]
        if valid_candidates:
            # Sort by confidence descending
            valid_candidates.sort(key=lambda item: item[1].confidence, reverse=True)
            return valid_candidates[0]

        # If no page found an answer, return the first page's result
        return all_results[0]

    def answer_document(
        self,
        document: dict,
        file_path: str | os.PathLike | None = None,
        question: str = "",
        page_entries: list[tuple[int, Image.Image]] | None = None,
    ) -> QAResult:
        """Answer questions about an invoice using a hybrid retrieval architecture:
        1. Canonical structured invoice fields (vendor, customer, dates, amounts, taxes, GSTINs).
        2. Line-item tabular data and specific product queries.
        3. Contextual / spatial text search (bank details, IFSC, contacts, terms).
        4. LayoutLM Document QA extractive comprehension fallback across all pages.
        """
        doc_fields: dict[str, Any] = {}
        doc_fields.update(document.get("extracted_fields") or {})
        doc_fields.update(document.get("invoice_fields") or {})
        doc_fields.update(document.get("layoutlm_fields") or {})
        ocr_text = document.get("extracted_text") or ""
        line_items = document.get("line_items") or []
        currency = doc_fields.get("currency") or "INR"
        sym = "₹" if currency in ("INR", "₹") else ("$" if currency in ("USD", "$") else ("€" if currency in ("EUR", "€") else ""))

        intent = classify_question_intent(question)
        qn_lower = question.lower().strip()

        # ── 1. Structured Canonical Fields ──────────────────────────────────
        if intent == "vendor_name":
            for k in ("vendor_name", "company_name", "seller_name"):
                v = doc_fields.get(k)
                if v and v not in ("Not found", "null", "None"):
                    evidence = _find_grounding_evidence(ocr_text, str(v), "vendor_name")
                    return QAResult(
                        question=question,
                        answer=str(v),
                        confidence=0.98,
                        confidence_level="high",
                        source_text=evidence,
                        page=1,
                        source_type="structured_field",
                        field_name="vendor_name",
                        model_name="Structured Invoice Field: vendor_name",
                    )

        elif intent == "customer_name":
            for k in ("customer_name", "buyer_name", "client_name"):
                v = doc_fields.get(k)
                if v and v not in ("Not found", "null", "None"):
                    evidence = _find_grounding_evidence(ocr_text, str(v), "customer_name")
                    return QAResult(
                        question=question,
                        answer=str(v),
                        confidence=0.98,
                        confidence_level="high",
                        source_text=evidence,
                        page=1,
                        source_type="structured_field",
                        field_name="customer_name",
                        model_name="Structured Invoice Field: customer_name",
                    )

        elif intent == "invoice_number":
            for k in ("invoice_number", "bill_number", "receipt_number"):
                v = doc_fields.get(k)
                if v and v not in ("Not found", "null", "None"):
                    evidence = _find_grounding_evidence(ocr_text, str(v), "invoice_number")
                    return QAResult(
                        question=question,
                        answer=str(v),
                        confidence=0.98,
                        confidence_level="high",
                        source_text=evidence,
                        page=1,
                        source_type="structured_field",
                        field_name="invoice_number",
                        model_name="Structured Invoice Field: invoice_number",
                    )

        elif intent == "invoice_date":
            for k in ("raw_invoice_date", "invoice_date"):
                v = doc_fields.get(k)
                if v and v not in ("Not found", "null", "None"):
                    evidence = _find_grounding_evidence(ocr_text, str(v), "invoice_date")
                    return QAResult(
                        question=question,
                        answer=str(v),
                        confidence=0.98,
                        confidence_level="high",
                        source_text=evidence,
                        page=1,
                        source_type="structured_field",
                        field_name="invoice_date",
                        model_name="Structured Invoice Field: invoice_date",
                    )

        elif intent == "due_date":
            for k in ("due_date", "payment_due_date"):
                v = doc_fields.get(k)
                if v and v not in ("Not found", "null", "None"):
                    evidence = _find_grounding_evidence(ocr_text, str(v), "due_date")
                    return QAResult(
                        question=question,
                        answer=str(v),
                        confidence=0.98,
                        confidence_level="high",
                        source_text=evidence,
                        page=1,
                        source_type="structured_field",
                        field_name="due_date",
                        model_name="Structured Invoice Field: due_date",
                    )

        elif intent == "total_amount":
            for k in ("total_amount", "net_amount", "grand_total"):
                v = doc_fields.get(k)
                if v and v not in ("Not found", "null", "None"):
                    ans_str = str(v)
                    if sym and not ans_str.startswith(sym):
                        ans_str = f"{sym}{ans_str}"
                    evidence = _find_grounding_evidence(ocr_text, str(v), "total_amount")
                    return QAResult(
                        question=question,
                        answer=ans_str,
                        confidence=0.98,
                        confidence_level="high",
                        source_text=evidence,
                        page=1,
                        source_type="structured_field",
                        field_name="total_amount",
                        model_name="Structured Invoice Field: total_amount",
                    )

        elif intent == "subtotal":
            for k in ("subtotal", "taxable_amount", "gross_amount"):
                v = doc_fields.get(k)
                if v and v not in ("Not found", "null", "None"):
                    ans_str = str(v)
                    if sym and not ans_str.startswith(sym):
                        ans_str = f"{sym}{ans_str}"
                    evidence = _find_grounding_evidence(ocr_text, str(v), "subtotal")
                    return QAResult(
                        question=question,
                        answer=ans_str,
                        confidence=0.98,
                        confidence_level="high",
                        source_text=evidence,
                        page=1,
                        source_type="structured_field",
                        field_name="subtotal",
                        model_name="Structured Invoice Field: subtotal",
                    )

        elif intent == "cgst_and_sgst":
            cgst = doc_fields.get("cgst_amount")
            sgst = doc_fields.get("sgst_amount")
            parts = []
            if cgst and cgst not in ("Not found", "null", "None"):
                parts.append(f"CGST: {sym}{cgst}" if sym and not str(cgst).startswith(sym) else f"CGST: {cgst}")
            if sgst and sgst not in ("Not found", "null", "None"):
                parts.append(f"SGST: {sym}{sgst}" if sym and not str(sgst).startswith(sym) else f"SGST: {sgst}")
            if parts:
                ans_str = ", ".join(parts)
                evidence = _find_grounding_evidence(ocr_text, str(cgst or sgst), "taxes")
                return QAResult(
                    question=question,
                    answer=ans_str,
                    confidence=0.98,
                    confidence_level="high",
                    source_text=evidence,
                    page=1,
                    source_type="structured_field",
                    field_name="cgst_and_sgst",
                    model_name="Structured Invoice Field: cgst_and_sgst",
                )

        elif intent == "cgst_amount":
            v = doc_fields.get("cgst_amount")
            if v and v not in ("Not found", "null", "None"):
                ans_str = f"{sym}{v}" if sym and not str(v).startswith(sym) else str(v)
                evidence = _find_grounding_evidence(ocr_text, str(v), "cgst_amount")
                return QAResult(
                    question=question,
                    answer=ans_str,
                    confidence=0.98,
                    confidence_level="high",
                    source_text=evidence,
                    page=1,
                    source_type="structured_field",
                    field_name="cgst_amount",
                    model_name="Structured Invoice Field: cgst_amount",
                )

        elif intent == "sgst_amount":
            v = doc_fields.get("sgst_amount")
            if v and v not in ("Not found", "null", "None"):
                ans_str = f"{sym}{v}" if sym and not str(v).startswith(sym) else str(v)
                evidence = _find_grounding_evidence(ocr_text, str(v), "sgst_amount")
                return QAResult(
                    question=question,
                    answer=ans_str,
                    confidence=0.98,
                    confidence_level="high",
                    source_text=evidence,
                    page=1,
                    source_type="structured_field",
                    field_name="sgst_amount",
                    model_name="Structured Invoice Field: sgst_amount",
                )

        elif intent == "igst_amount":
            v = doc_fields.get("igst_amount")
            if v and v not in ("Not found", "null", "None"):
                ans_str = f"{sym}{v}" if sym and not str(v).startswith(sym) else str(v)
                evidence = _find_grounding_evidence(ocr_text, str(v), "igst_amount")
                return QAResult(
                    question=question,
                    answer=ans_str,
                    confidence=0.98,
                    confidence_level="high",
                    source_text=evidence,
                    page=1,
                    source_type="structured_field",
                    field_name="igst_amount",
                    model_name="Structured Invoice Field: igst_amount",
                )

        elif intent == "tax_amount":
            for k in ("tax_amount", "total_tax"):
                v = doc_fields.get(k)
                if v and v not in ("Not found", "null", "None"):
                    ans_str = f"{sym}{v}" if sym and not str(v).startswith(sym) else str(v)
                    evidence = _find_grounding_evidence(ocr_text, str(v), "tax_amount")
                    return QAResult(
                        question=question,
                        answer=ans_str,
                        confidence=0.98,
                        confidence_level="high",
                        source_text=evidence,
                        page=1,
                        source_type="structured_field",
                        field_name="tax_amount",
                        model_name="Structured Invoice Field: tax_amount",
                    )

        elif intent == "vendor_gstin":
            for k in ("vendor_gstin", "gst_number"):
                v = doc_fields.get(k)
                if v and v not in ("Not found", "null", "None"):
                    evidence = _find_grounding_evidence(ocr_text, str(v), "vendor_gstin")
                    return QAResult(
                        question=question,
                        answer=str(v),
                        confidence=0.98,
                        confidence_level="high",
                        source_text=evidence,
                        page=1,
                        source_type="structured_field",
                        field_name="vendor_gstin",
                        model_name="Structured Invoice Field: vendor_gstin",
                    )

        elif intent == "customer_gstin":
            v = doc_fields.get("customer_gstin")
            if v and v not in ("Not found", "null", "None"):
                evidence = _find_grounding_evidence(ocr_text, str(v), "customer_gstin")
                return QAResult(
                    question=question,
                    answer=str(v),
                    confidence=0.98,
                    confidence_level="high",
                    source_text=evidence,
                    page=1,
                    source_type="structured_field",
                    field_name="customer_gstin",
                    model_name="Structured Invoice Field: customer_gstin",
                )

        elif intent == "generic_gstin":
            vg = doc_fields.get("vendor_gstin") or doc_fields.get("gst_number")
            cg = doc_fields.get("customer_gstin")
            parts = []
            if vg and vg not in ("Not found", "null", "None"):
                parts.append(f"Vendor GSTIN: {vg}")
            if cg and cg not in ("Not found", "null", "None"):
                parts.append(f"Customer GSTIN: {cg}")
            if parts:
                ans_str = ", ".join(parts)
                evidence = _find_grounding_evidence(ocr_text, str(vg or cg), "gstin")
                return QAResult(
                    question=question,
                    answer=ans_str,
                    confidence=0.98,
                    confidence_level="high",
                    source_text=evidence,
                    page=1,
                    source_type="structured_field",
                    field_name="gstin",
                    model_name="Structured Invoice Field: gstin",
                )

        elif intent == "discount":
            v = doc_fields.get("discount")
            if v and v not in ("Not found", "null", "None"):
                ans_str = f"{sym}{v}" if sym and not str(v).startswith(sym) else str(v)
                evidence = _find_grounding_evidence(ocr_text, str(v), "discount")
                return QAResult(
                    question=question,
                    answer=ans_str,
                    confidence=0.98,
                    confidence_level="high",
                    source_text=evidence,
                    page=1,
                    source_type="structured_field",
                    field_name="discount",
                    model_name="Structured Invoice Field: discount",
                )

        elif intent == "currency":
            v = doc_fields.get("currency")
            if v and v not in ("Not found", "null", "None"):
                return QAResult(
                    question=question,
                    answer=str(v),
                    confidence=0.98,
                    confidence_level="high",
                    source_text=f"Currency: {v}",
                    page=1,
                    source_type="structured_field",
                    field_name="currency",
                    model_name="Structured Invoice Field: currency",
                )

        elif intent == "customer_address":
            v = doc_fields.get("customer_address")
            if v and v not in ("Not found", "null", "None"):
                evidence = _find_grounding_evidence(ocr_text, str(v), "customer_address")
                return QAResult(
                    question=question,
                    answer=str(v),
                    confidence=0.95,
                    confidence_level="high",
                    source_text=evidence,
                    page=1,
                    source_type="structured_field",
                    field_name="customer_address",
                    model_name="Structured Invoice Field: customer_address",
                )

        elif intent == "vendor_address":
            v = doc_fields.get("vendor_address")
            if v and v not in ("Not found", "null", "None"):
                evidence = _find_grounding_evidence(ocr_text, str(v), "vendor_address")
                return QAResult(
                    question=question,
                    answer=str(v),
                    confidence=0.95,
                    confidence_level="high",
                    source_text=evidence,
                    page=1,
                    source_type="structured_field",
                    field_name="vendor_address",
                    model_name="Structured Invoice Field: vendor_address",
                )

        # ── 2. Line Items Table & Product Extraction ──────────────────────
        if intent == "line_items":
            if line_items:
                lines = [f"{len(line_items)} line items found:"]
                for i, it in enumerate(line_items, 1):
                    desc = it.get("description") or f"Item {i}"
                    qty = it.get("quantity")
                    price = it.get("unit_price")
                    tot = it.get("item_total")
                    bits = []
                    if qty is not None:
                        bits.append(f"Qty: {qty}")
                    if price is not None:
                        bits.append(f"Price: {sym}{price}" if sym else f"Price: {price}")
                    if tot is not None:
                        bits.append(f"Total: {sym}{tot}" if sym else f"Total: {tot}")
                    lines.append(f"{i}. {desc} ({', '.join(bits)})" if bits else f"{i}. {desc}")
                summary = "\n".join(lines)
                evidence = f"Extracted {len(line_items)} verified rows from invoice table."
                return QAResult(
                    question=question,
                    answer=summary,
                    confidence=0.95,
                    confidence_level="high",
                    source_text=evidence,
                    page=1,
                    source_type="line_items",
                    field_name="line_items",
                    model_name="Structured Line-Item Extraction",
                )

        # Check for specific item queries (e.g., "quantity of LIBRIUM", "price of ADVACON")
        if line_items:
            prod_query = re.search(r"\b(quantity|qty|price|rate|cost|total|amount)\s+(?:of|for)\s+([A-Za-z0-9\s\-_]+)", qn_lower)
            if prod_query:
                metric = prod_query.group(1).lower()
                target = prod_query.group(2).strip().lower()
                for it in line_items:
                    desc = (it.get("description") or "").lower()
                    if target in desc:
                        if metric in ("quantity", "qty"):
                            val = it.get("quantity")
                            if val is not None:
                                return QAResult(
                                    question=question,
                                    answer=f"{val} (for {it.get('description')})",
                                    confidence=0.95,
                                    confidence_level="high",
                                    source_text=f"Table row: {it.get('description')} | Qty: {val}",
                                    page=1,
                                    source_type="line_items",
                                    field_name="line_item_quantity",
                                    model_name="Structured Line-Item Extraction",
                                )
                        elif metric in ("price", "rate", "cost"):
                            val = it.get("unit_price")
                            if val is not None:
                                p_str = f"{sym}{val}" if sym else str(val)
                                return QAResult(
                                    question=question,
                                    answer=f"{p_str} (Unit price for {it.get('description')})",
                                    confidence=0.95,
                                    confidence_level="high",
                                    source_text=f"Table row: {it.get('description')} | Unit Price: {p_str}",
                                    page=1,
                                    source_type="line_items",
                                    field_name="line_item_unit_price",
                                    model_name="Structured Line-Item Extraction",
                                )
                        elif metric in ("total", "amount"):
                            val = it.get("item_total")
                            if val is not None:
                                t_str = f"{sym}{val}" if sym else str(val)
                                return QAResult(
                                    question=question,
                                    answer=f"{t_str} (Item total for {it.get('description')})",
                                    confidence=0.95,
                                    confidence_level="high",
                                    source_text=f"Table row: {it.get('description')} | Total: {t_str}",
                                    page=1,
                                    source_type="line_items",
                                    field_name="line_item_total",
                                    model_name="Structured Line-Item Extraction",
                                )

        # ── 3. Contextual Document Text Matches (Regex / Proximity) ─────────
        if ocr_text:
            if intent == "bank_details":
                m = re.search(r"BANK\s+DETAILS[:\s]+([^\n\r]+(?:\n[^\n\r]+)?)", ocr_text, re.IGNORECASE)
                if m:
                    ans = m.group(1).strip()
                    return QAResult(
                        question=question,
                        answer=ans,
                        confidence=0.95,
                        confidence_level="high",
                        source_text=f"...{m.group(0)}...",
                        page=1,
                        source_type="document_text",
                        field_name="bank_details",
                        model_name="Document Text Match",
                    )

            elif intent == "ifsc_code":
                m = re.search(r"IFSC(?:\s+CODE)?\s*[:\s]+([A-Z]{4}0[A-Z0-9]{6})", ocr_text, re.IGNORECASE)
                if m:
                    ans = m.group(1).strip()
                    return QAResult(
                        question=question,
                        answer=ans,
                        confidence=0.95,
                        confidence_level="high",
                        source_text=f"...{m.group(0)}...",
                        page=1,
                        source_type="document_text",
                        field_name="ifsc_code",
                        model_name="Document Text Match",
                    )

            elif intent == "account_number":
                m = re.search(r"A/?C\s*(?:NO|NUM|NUMBER)?\s*[:\s]*([0-9]{9,18})", ocr_text, re.IGNORECASE)
                if m:
                    ans = m.group(1).strip()
                    return QAResult(
                        question=question,
                        answer=ans,
                        confidence=0.95,
                        confidence_level="high",
                        source_text=f"...{m.group(0)}...",
                        page=1,
                        source_type="document_text",
                        field_name="account_number",
                        model_name="Document Text Match",
                    )

            elif intent == "drug_licence":
                dl_matches = re.findall(r"(2[01]B\s*:\s*[A-Za-z0-9/-]+)", ocr_text, re.IGNORECASE)
                if dl_matches:
                    ans = ", ".join(dl_matches)
                    return QAResult(
                        question=question,
                        answer=ans,
                        confidence=0.95,
                        confidence_level="high",
                        source_text=f"...{ans}...",
                        page=1,
                        source_type="document_text",
                        field_name="drug_licence",
                        model_name="Document Text Match",
                    )

            elif intent == "email":
                m = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", ocr_text)
                if m:
                    ans = m.group(1).strip()
                    return QAResult(
                        question=question,
                        answer=ans,
                        confidence=0.95,
                        confidence_level="high",
                        source_text=f"...{m.group(0)}...",
                        page=1,
                        source_type="document_text",
                        field_name="email",
                        model_name="Document Text Match",
                    )

            elif intent == "phone":
                m = re.search(r"(?:Cell|Ph|Phone|Mobile|Tel)[:\s]*([0-9]{10}(?:[,\s]+[0-9]{10})*)", ocr_text, re.IGNORECASE)
                if m:
                    ans = m.group(1).strip()
                    return QAResult(
                        question=question,
                        answer=ans,
                        confidence=0.95,
                        confidence_level="high",
                        source_text=f"...{m.group(0)}...",
                        page=1,
                        source_type="document_text",
                        field_name="phone",
                        model_name="Document Text Match",
                    )

            elif intent == "jurisdiction":
                m = re.search(r"(Subject\s+to\s+[A-Za-z\s]+Jurisdiction\s+only)", ocr_text, re.IGNORECASE)
                if m:
                    ans = m.group(1).strip()
                    return QAResult(
                        question=question,
                        answer=ans,
                        confidence=0.95,
                        confidence_level="high",
                        source_text=f"...{m.group(0)}...",
                        page=1,
                        source_type="document_text",
                        field_name="jurisdiction",
                        model_name="Document Text Match",
                    )

        # ── 4. Document-Level Extractive LayoutLM QA Fallback ──────────────
        if page_entries:
            images = [img for _, img in page_entries]
            best_page, qa_res = self.answer_pages(images, question, ocr_text=ocr_text)
            if qa_res.answer and qa_res.confidence >= 0.15:
                qa_res.page = best_page
                qa_res.source_type = "layoutlm_docqa"
                return qa_res

        # ── 5. Unanswerable Guard (Prevent Fabrications / Hallucinations) ───
        return QAResult(
            question=question,
            answer=None,
            confidence=0.0,
            confidence_level="low",
            explanation="The requested information could not be found or verified in this document.",
            source_text=None,
            page=1,
            source_type="unanswerable",
            model_name=self.model_name,
        )

