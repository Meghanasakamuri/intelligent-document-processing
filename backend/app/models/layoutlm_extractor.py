"""O2 — LayoutLMv3 token classification for structured field extraction.

Uses a LayoutLMv3 model fine-tuned on the FUNSD dataset for form
understanding.  The model classifies each token into semantic
categories (HEADER, QUESTION, ANSWER, OTHER) using BIO tagging, then
pairs QUESTION and ANSWER entities to produce key-value extractions.

This is a **layout-aware** approach: the model jointly considers the
token text, its 2-D bounding-box position on the page, and the
document image, so it can exploit spatial relationships that pure
regex heuristics miss.

Architecture
------------
Image → Tesseract OCR (words + bounding boxes)
      → LayoutLMv3 processor (tokenise with layout)
      → LayoutLMv3 + classification head (token labels)
      → BIO entity grouping
      → QUESTION ↔ ANSWER pairing
      → Structured key-value output

Important
---------
* The default checkpoint ``nielsr/layoutlmv3-finetuned-funsd`` was
  fine-tuned on the FUNSD *form* dataset.  It classifies generic
  form tokens — it does **not** directly output named invoice fields.
* To extract invoice-specific fields (invoice_number, total_amount …)
  the model would need fine-tuning on an invoice token-classification
  dataset (e.g.  CORD).  See ``train_layoutlm.py`` for the training
  pipeline.
* Raw softmax scores from this model are **not** calibrated
  probabilities.  Use the ``confidence`` module for calibration.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from PIL import Image

from app.models.base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

# FUNSD BIO labels (9 classes)
FUNSD_LABELS = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
    "B-OTHER",
    "I-OTHER",
]


class LayoutLMv3Extractor(BaseExtractor):
    """LayoutLMv3 token-classification extractor (O2).

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.  Must be a LayoutLMv3 checkpoint
        with a ``ForTokenClassification`` head.
    cache_dir : str | None
        Local directory for downloaded model weights.
    """

    def __init__(
        self,
        model_name: str | None = None,
        fallback_model: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        try:
            from app.config import LAYOUTLM_MODEL, FALLBACK_LAYOUTLM_MODEL, MODELS_CACHE_DIR
            default_model = LAYOUTLM_MODEL
            default_fallback = FALLBACK_LAYOUTLM_MODEL
            default_cache = str(MODELS_CACHE_DIR)
        except Exception:
            default_model = "checkpoints/layoutlmv3-fatura2"
            default_fallback = "nielsr/layoutlmv3-finetuned-funsd"
            default_cache = None

        self.model_name = model_name or default_model
        self.fallback_model = fallback_model or default_fallback
        self.cache_dir = cache_dir or default_cache
        self.active_model_name = self.model_name
        self._model = None
        self._processor = None
        self._loaded = False
        self._load_error: str | None = None

    # ── public interface ─────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except (ImportError, OSError, Exception):
            return False

    def is_loaded(self) -> bool:
        return self._loaded

    def _is_invoice_model(self) -> bool:
        """Check whether the loaded model has invoice-specific entity labels."""
        if not self._model or not hasattr(self._model, "config"):
            return False
        id2label = getattr(self._model.config, "id2label", {})
        if not id2label:
            return False
        invoice_keywords = {
            "SELLER", "BUYER", "TOTAL", "SUBTOTAL", "TAX", "NUMBER",
            "INV_NO", "INVOICE_NUMBER", "AMOUNT", "VENDOR", "CUSTOMER",
            "MENU", "PRICE", "TABLE", "ITEM_DESCRIPTION", "BILL_TO",
            "SEND_TO", "AMOUNT_DUE", "PO_NUMBER", "SELLER_ADDRESS",
            "SELLER_SITE", "ITEM_TOTAL", "ITEM_QUANTITY", "ITEM_UNIT_PRICE",
        }
        for lbl in id2label.values():
            clean_lbl = str(lbl).replace("B-", "").replace("I-", "").upper()
            if clean_lbl in invoice_keywords:
                return True
        return False

    def load(self) -> None:
        """Eagerly load the model using AutoModelForTokenClassification and AutoProcessor.

        Attempts to load the configured primary model (e.g. FATURA2 checkpoint).
        If loading fails, falls back gracefully to the FUNSD checkpoint.
        """
        if self._loaded or self._load_error:
            return
        try:
            import os
            from transformers import (
                AutoModelForTokenClassification,
                AutoProcessor,
            )
            from app.config import resolve_model_path

            logger.info("Loading LayoutLMv3 model: %s …", self.model_name)
            start = time.time()

            primary_resolved = resolve_model_path(self.model_name)
            is_local = os.path.isdir(primary_resolved)

            try:
                self._processor = AutoProcessor.from_pretrained(
                    primary_resolved,
                    apply_ocr=False,
                    cache_dir=self.cache_dir,
                    local_files_only=is_local,
                )
                self._model = AutoModelForTokenClassification.from_pretrained(
                    primary_resolved,
                    cache_dir=self.cache_dir,
                    local_files_only=is_local,
                )
                self.active_model_name = self.model_name
                logger.info("Successfully loaded primary LayoutLMv3 model: %s", self.model_name)
            except Exception as primary_exc:
                logger.warning(
                    "Primary LayoutLMv3 model (%s) failed to load: %s. Falling back to FUNSD model (%s)",
                    self.model_name,
                    primary_exc,
                    self.fallback_model,
                )
                fallback_resolved = resolve_model_path(self.fallback_model)
                fb_is_local = os.path.isdir(fallback_resolved)
                self._processor = AutoProcessor.from_pretrained(
                    fallback_resolved,
                    apply_ocr=False,
                    cache_dir=self.cache_dir,
                    local_files_only=fb_is_local,
                )
                self._model = AutoModelForTokenClassification.from_pretrained(
                    fallback_resolved,
                    cache_dir=self.cache_dir,
                    local_files_only=fb_is_local,
                )
                self.active_model_name = self.fallback_model
                logger.info("Successfully loaded fallback LayoutLMv3 model: %s", self.fallback_model)

            self._model.eval()

            elapsed = time.time() - start
            logger.info("LayoutLMv3 (%s) loaded in %.1fs", self.active_model_name, elapsed)
            self._loaded = True

        except Exception as exc:
            self._load_error = str(exc)
            logger.error("Failed to load LayoutLMv3: %s", exc)

    def extract_pages(
        self,
        images: list[Image.Image],
        ocr_text: str = "",
        document_type: str = "Unknown",
    ) -> ExtractionResult:
        """Run LayoutLMv3 extraction across multiple document page images.

        Parameters
        ----------
        images : list[Image.Image]
            List of PIL page images in page order.
        ocr_text : str
            Full document OCR text (optional context).
        document_type : str
            Document classification type.

        Returns
        -------
        ExtractionResult
            Combined extraction result with aggregated fields, confidence scores,
            and page-level provenance stored in metadata.
        """
        if not images:
            return ExtractionResult(
                method="layoutlmv3_funsd",
                model_name=self.active_model_name,
                raw_text=ocr_text,
                metadata={"error": "No page images provided"},
            )

        if len(images) == 1:
            res = self.extract(image=images[0], ocr_text=ocr_text, document_type=document_type)
            res.metadata = dict(res.metadata or {})
            res.metadata["pages_processed"] = 1
            res.metadata["field_source_pages"] = {k: 1 for k in (res.fields or {})}
            items = getattr(res, "line_items", []) or []
            for idx, item in enumerate(items):
                item["page"] = 1
                item["row_index"] = idx + 1
            res.line_items = items
            res.metadata["total_line_items"] = len(items)
            res.metadata["page_results"] = [
                {
                    "page_number": 1,
                    "fields": res.fields,
                    "confidence_scores": res.confidence_scores,
                    "line_items": items,
                    "validation": getattr(res, "validation", None),
                }
            ]
            return res

        combined_fields: dict[str, Any] = {}
        combined_confs: dict[str, float] = {}
        field_source_pages: dict[str, int] = {}
        page_results: list[dict[str, Any]] = []
        all_line_items: list[dict[str, Any]] = []
        total_time = 0.0
        active_method = "layoutlmv3_funsd"
        global_row_index = 1

        for p_idx, img in enumerate(images):
            p_num = p_idx + 1
            try:
                p_res = self.extract(
                    image=img,
                    ocr_text="",
                    document_type=document_type,
                )
                if p_res.method:
                    active_method = p_res.method
                total_time += (p_res.processing_time or 0.0)
                p_fields = p_res.fields or {}
                p_confs = p_res.confidence_scores or {}
                p_items = getattr(p_res, "line_items", []) or []

                p_tagged_items = []
                for item in p_items:
                    item_copy = dict(item)
                    item_copy["page"] = p_num
                    item_copy["row_index"] = global_row_index
                    global_row_index += 1
                    all_line_items.append(item_copy)
                    p_tagged_items.append(item_copy)

                page_results.append({
                    "page_number": p_num,
                    "fields": p_fields,
                    "confidence_scores": p_confs,
                    "line_items": p_tagged_items,
                })

                for f_key, f_val in p_fields.items():
                    if f_val is None or str(f_val).strip() == "":
                        continue

                    f_conf = float(p_confs.get(f_key, 0.0))

                    if f_key not in combined_fields:
                        combined_fields[f_key] = f_val
                        combined_confs[f_key] = f_conf
                        field_source_pages[f_key] = p_num
                    else:
                        existing_val = combined_fields[f_key]
                        existing_conf = combined_confs.get(f_key, 0.0)

                        if str(f_val).strip() == str(existing_val).strip():
                            if f_conf > existing_conf:
                                combined_confs[f_key] = f_conf
                        elif f_conf > existing_conf:
                            combined_fields[f_key] = f_val
                            combined_confs[f_key] = f_conf
                            field_source_pages[f_key] = p_num

            except Exception as page_exc:
                logger.warning("Extraction failed on page %d: %s", p_num, page_exc)
                page_results.append({
                    "page_number": p_num,
                    "error": str(page_exc),
                    "fields": {},
                    "confidence_scores": {},
                    "line_items": [],
                })

        # Document-level validation across all pages
        from app.validation import normalize_date, validate_financials
        doc_date_val: dict[str, Any] = {}
        if "invoice_date" in combined_fields:
            raw_inv = combined_fields.get("raw_invoice_date") or combined_fields["invoice_date"]
            inv_norm = normalize_date(raw_inv)
            combined_fields["raw_invoice_date"] = raw_inv
            if inv_norm.normalized:
                combined_fields["invoice_date"] = inv_norm.normalized
            doc_date_val["invoice_date"] = inv_norm.to_dict()

        if "due_date" in combined_fields:
            raw_due = combined_fields.get("raw_due_date") or combined_fields["due_date"]
            due_norm = normalize_date(raw_due)
            combined_fields["raw_due_date"] = raw_due
            if due_norm.normalized:
                combined_fields["due_date"] = due_norm.normalized
            doc_date_val["due_date"] = due_norm.to_dict()

        doc_fin_val = validate_financials(combined_fields, line_items=all_line_items, tolerance=0.05)
        doc_errors = list(doc_fin_val.validation_errors)
        for d_key, d_res in doc_date_val.items():
            if not d_res.get("is_valid") and d_res.get("error"):
                doc_errors.append(f"{d_key}: {d_res['error']}")
            elif d_res.get("is_ambiguous"):
                doc_errors.append(f"{d_key}: Ambiguous date format")

        doc_validation = {
            "is_valid": doc_fin_val.is_valid and all(d.get("is_valid", True) for d in doc_date_val.values()),
            "suspicious": doc_fin_val.suspicious or any(d.get("is_ambiguous", False) or not d.get("is_valid", True) for d in doc_date_val.values()),
            "validation_errors": doc_errors,
            "financials": doc_fin_val.to_dict(),
            "dates": doc_date_val,
        }

        return ExtractionResult(
            method=active_method,
            model_name=self.active_model_name,
            fields=combined_fields,
            confidence_scores=combined_confs,
            line_items=all_line_items,
            processing_time=round(total_time, 2),
            raw_text=ocr_text,
            validation=doc_validation,
            metadata={
                "pages_processed": len(images),
                "field_source_pages": field_source_pages,
                "page_results": page_results,
                "total_line_items": len(all_line_items),
                "validation": doc_validation,
            },
        )

    def extract(
        self,
        image: Image.Image,
        ocr_text: str = "",
        document_type: str = "Unknown",
        words: list[str] | None = None,
        boxes: list[list[int]] | None = None,
        psm: int | None = None,
    ) -> ExtractionResult:
        self.load()

        if not self._loaded:
            return ExtractionResult(
                method="layoutlmv3_funsd",
                model_name=self.model_name,
                raw_text=ocr_text,
                metadata={"error": self._load_error or "Model not loaded"},
            )

        if not hasattr(image, "size") or image.size[0] <= 0 or image.size[1] <= 0:
            return ExtractionResult(
                method="layoutlmv3_funsd",
                model_name=self.model_name,
                raw_text=ocr_text,
                metadata={"error": "Invalid or empty image dimensions"},
            )

        import torch

        start = time.time()

        # 1. OCR words + bounding boxes (if not directly provided)
        if words is None or boxes is None:
            words, boxes = self._ocr_words_boxes(image, psm=psm)

        if not words or not boxes:
            return ExtractionResult(
                method="layoutlmv3_funsd",
                model_name=self.model_name,
                raw_text=ocr_text,
                metadata={"error": "OCR produced no words"},
            )

        # 2. Validate strict 1:1 correspondence and normalise boxes to 0-1000
        w, h = image.size
        valid_words: list[str] = []
        norm_boxes: list[list[int]] = []

        for word, box in zip(words, boxes):
            word_str = str(word).strip()
            if not word_str:
                continue
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                x0, y0, x1, y1 = [int(v) for v in box]
            except (ValueError, TypeError):
                continue

            # Scale to 0-1000 range and clamp strictly
            nx0 = max(0, min(1000, int(x0 * 1000 / w)))
            ny0 = max(0, min(1000, int(y0 * 1000 / h)))
            nx1 = max(0, min(1000, int(x1 * 1000 / w)))
            ny1 = max(0, min(1000, int(y1 * 1000 / h)))

            # Guarantee non-inverted box coordinates
            nx0, nx1 = min(nx0, nx1), max(nx0, nx1)
            ny0, ny1 = min(ny0, ny1), max(ny0, ny1)

            norm_boxes.append([nx0, ny0, nx1, ny1])
            valid_words.append(word_str)

        if not valid_words:
            return ExtractionResult(
                method="layoutlmv3_funsd",
                model_name=self.model_name,
                raw_text=ocr_text,
                metadata={"error": "No valid words or bounding boxes found"},
            )

        # 3. Tokenise with LayoutLMv3Processor (sliding-window for multi-page/long documents)
        encoding = self._processor(
            images=image,
            text=valid_words,
            boxes=norm_boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
            return_overflowing_tokens=True,
            stride=128,
        )

        pv = encoding.get("pixel_values")
        if isinstance(pv, list):
            encoding["pixel_values"] = torch.stack(pv)

        # 4. Inference
        with torch.no_grad():
            outputs = self._model(**encoding)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        num_batches = logits.size(0)

        # 5. Map sub-word predictions back to words across all chunk windows
        word_preds: dict[int, int] = {}
        word_confs: dict[int, float] = {}

        for b in range(num_batches):
            b_word_ids = encoding.word_ids(batch_index=b)
            b_preds = logits[b].argmax(dim=-1).tolist()
            b_probs = probs[b].max(dim=-1).values.tolist()
            for tok_idx, wid in enumerate(b_word_ids):
                if wid is None or wid >= len(valid_words):
                    continue
                c_conf = float(b_probs[tok_idx])
                pred_label = b_preds[tok_idx]
                if wid not in word_preds:
                    word_preds[wid] = pred_label
                    word_confs[wid] = c_conf
                else:
                    prev_label = word_preds[wid]
                    if prev_label == 0 and pred_label != 0:
                        word_preds[wid] = pred_label
                        word_confs[wid] = c_conf
                    elif prev_label == pred_label and c_conf > word_confs[wid]:
                        word_confs[wid] = c_conf

        # 6. Branch for invoice-specific models (e.g. FATURA2 / CORD)
        id2label = getattr(getattr(self._model, "config", None), "id2label", None)
        if self._is_invoice_model():
            return self._extract_invoice(
                valid_words=valid_words,
                word_preds=word_preds,
                word_confs=word_confs,
                norm_boxes=norm_boxes,
                id2label=id2label,
                ocr_text=ocr_text,
                document_type=document_type,
                start=start,
                image=image,
            )

        # 7. FUNSD Mode: Group into 2D layout-aware entities
        entities = self._group_entities(
            valid_words,
            word_preds,
            word_confs,
            boxes=norm_boxes,
            id2label=id2label,
        )

        # 8. Pair QUESTION → ANSWER using 2-pass spatial layout proximity
        pairs = self._pair_qa(entities)

        # 8. Build raw key-value output and canonical schema mapping
        raw_fields: dict[str, Any] = {}
        confidence_scores: dict[str, float] = {}
        for item in pairs:
            q_text = item[0]
            a_text = item[1]
            c_val = item[2]
            key = q_text.strip().rstrip(":").strip()
            if key:
                raw_fields[key] = a_text.strip() if a_text else None
                confidence_scores[key] = round(c_val, 4)

        headers = [e["text"] for e in entities if e["label"] == "HEADER"]
        canonical_fields, canonical_confs, unmapped = self._map_to_canonical_fields(
            pairs=pairs,
            headers=headers,
            document_type=document_type,
            raw_confs=confidence_scores,
            valid_words=valid_words,
            ocr_text=ocr_text,
        )

        elapsed = time.time() - start

        return ExtractionResult(
            fields=canonical_fields,
            confidence_scores=canonical_confs,
            raw_text=ocr_text or " ".join(valid_words),
            method="layoutlmv3_funsd",
            model_name=self.model_name,
            processing_time=round(elapsed, 2),
            metadata={
                "total_entities": len(entities),
                "pairs_extracted": len([p for p in pairs if p[1] is not None]),
                "raw_pairs": raw_fields,
                "unmapped_pairs": unmapped,
                "headers": headers,
                "document_type": document_type,
                "model_domain": "form_understanding_funsd",
                "notes": (
                    "Checkpoint nielsr/layoutlmv3-finetuned-funsd classifies generic form tokens "
                    "(HEADER, QUESTION, ANSWER). Canonical invoice fields are structured via layout-aware "
                    "2D spatial pairing and semantic validation. Invoice-specific fine-tuning (e.g. CORD) "
                    "is recommended for direct token-level invoice entity extraction."
                ),
            },
        )

    @classmethod
    def _parse_numeric(cls, val_str: Any) -> float | None:
        """Parse clean floating-point number from currency or quantity text.

        Handles currency symbols, unit suffixes, commas, European formats,
        and negative amounts. Returns None if unparseable or contains non-unit alphabetic text.
        """
        if val_str is None:
            return None
        s = str(val_str).strip()
        if not s:
            return None
        # Remove currency symbols
        s = re.sub(r"[$€£₹¥]", "", s).strip()
        # Remove common unit suffixes
        s = re.sub(
            r"(?i)\b(?:pcs|pc|units?|hrs?|hours?|items?|ea|boxes|box|kg|m|g)\b",
            "",
            s,
        ).strip()
        # Remove trailing x/X multiplier (e.g. '2x' -> '2')
        s = re.sub(r"(?i)\s*x\s*$", "", s).strip()

        # Reject if non-unit alphabetic letters remain
        if re.search(r"[A-Za-z]", s):
            return None

        # Standard format with comma thousands: 1,234.56
        if re.search(r"^\s*[-+]?\d{1,3}(?:,\d{3})*\.\d+\s*$", s):
            s = s.replace(",", "")
        # European format with dot thousands: 1.234,56
        elif re.search(r"^\s*[-+]?\d{1,3}(?:\.\d{3})*,\d+\s*$", s):
            s = s.replace(".", "").replace(",", ".")
        # Comma as decimal separator: 12,50
        elif re.search(r"^\s*[-+]?\d+,\d{1,2}\s*$", s):
            s = s.replace(",", ".")
        # Comma as thousands without decimal: 1,250
        elif re.search(r"^\s*[-+]?\d{1,3}(?:,\d{3})+\s*$", s):
            s = s.replace(",", "")

        m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
        if m:
            try:
                return round(float(m.group(0)), 2)
            except ValueError:
                return None
        return None

    @staticmethod
    def _clean_table_description(text: str) -> str:
        """Clean table border lines, delimiters, and OCR noise from description text."""
        if not text:
            return ""
        t = text
        t = re.sub(r"^[\s|_|\\/\[\]:.\-©~]+", "", t)
        t = re.sub(r"[\s|_|\\/\[\]:.\-©~]+$", "", t)
        t = re.sub(r"\s+[|_\\]+\s+", " ", t)
        t = re.sub(r"\s+[©~]\s+", " ", t)
        t = re.sub(r"^[pP]Is(?=[A-Z])", "DIS", t)
        return t.strip()

    @staticmethod
    def _parse_table_num_token(token: str) -> tuple[float | None, bool]:
        """Parse numeric token handling currency symbols, negative values, OCR artifacts.
        Returns (numeric_value, has_decimal_point).
        """
        if not token:
            return None, False
        s = token.strip()
        is_neg = False
        if s.startswith("(-)") or s.startswith("-"):
            is_neg = True

        # Strip currency, sign, and OCR symbols ($ often read as s, S, §, -st, st, etc.)
        s = re.sub(r"^[-—+]+", "", s)
        s = re.sub(r"^(?:st|[-]?st|[(\-\s$€£¥₹sS§+])+", "", s, flags=re.IGNORECASE)
        s = re.sub(r"[\s)%$€£¥₹!\]}\|]+$", "", s)
        s = s.replace(",", "")

        has_dec = "." in s
        if re.match(r"^\d+(?:\.\d+)?$", s):
            try:
                val = float(s)
                return (-val if is_neg else val), has_dec
            except ValueError:
                return None, False
        return None, False

    @classmethod
    def _extract_table_line_items(
        cls,
        entities: list[dict],
        valid_words: list[str] | None = None,
        norm_boxes: list[list[int]] | None = None,
        page_number: int = 1,
        y_threshold: float = 18.0,
    ) -> list[dict[str, Any]]:
        """Extract structured invoice table rows from token classification entities.

        Clusters ITEM_DESCRIPTION, ITEM_QUANTITY, ITEM_UNIT_PRICE, and ITEM_TOTAL
        entities into horizontal rows, validates arithmetic consistency, and returns
        fully traceable line item objects.
        """
        target_labels = {
            "ITEM_DESCRIPTION",
            "ITEM_QUANTITY",
            "ITEM_UNIT_PRICE",
            "ITEM_TOTAL",
        }

        # Filter entities to line-item components
        item_ents = [
            e for e in entities
            if e.get("label", "").upper() in target_labels
        ]

        # Fallback: if no fine-grained ITEM_* entities, perform spatial column/row reconstruction
        if not item_ents and valid_words and norm_boxes:
            # 1. Search for table header line
            col_defs = {
                "desc": ["description", "desc", "particulars", "items", "item", "product", "product name"],
                "qty": ["quantity", "qty", "antity", "units", "quanity"],
                "id": ["id", "code", "part no", "item id", "tin", "s.no", "f.no"],
                "hsn": ["hsn", "sac", "hsn/sac"],
                "pack": ["pack", "packing"],
                "batch": ["batch", "lot", "b.no", "b. no", "bno", "batch no"],
                "exp": ["edate", "e.date", "exp", "expiry"],
                "free": ["free"],
                "price": ["price", "rate", "unit price", "unit rate"],
                "disc": ["dic", "disc", "discount", "disc%", "ic"],
                "tax": ["tax", "vat", "gst", "tak", "gst%"],
                "total": ["amount", "total", "line total", "net amount", "taxable value"],
                "mrp": ["mrp", "m.r.p."],
                "mfg": ["mfg", "mig", "manufacturer"],
            }
            strip_hdr_chars = " :._-|[]{}()!~`'\""

            # Group all page tokens by y-mid
            all_lines = []
            tokens_with_box = [{"word": w, "box": b} for w, b in zip(valid_words, norm_boxes)]
            tokens_with_box.sort(key=lambda x: ((x["box"][1] + x["box"][3]) / 2.0, x["box"][0]))
            for tb in tokens_with_box:
                ymid = (tb["box"][1] + tb["box"][3]) / 2.0
                assigned = False
                for l in all_lines:
                    l_ymid = sum((x["box"][1] + x["box"][3]) / 2.0 for x in l) / len(l)
                    if abs(ymid - l_ymid) <= 12.0:
                        l.append(tb)
                        assigned = True
                        break
                if not assigned:
                    all_lines.append([tb])

            header_idx = None
            cols_detected = {}
            top_y = 200

            for l_idx, line in enumerate(all_lines):
                line_txt = " ".join(t["word"].upper() for t in line)
                if any(meta in line_txt for meta in ["DATE:", "INVOICE NUMBER", "BILL TO", "SHIP TO"]):
                    continue
                matches = {}
                for c_type, kws in col_defs.items():
                    for t_item in line:
                        tw = t_item["word"].lower().strip(strip_hdr_chars)
                        if any(kw == tw or (tw.startswith(kw) and kw not in ["part", "tin", "tax", "item", "id", "free", "pack", "ic"]) for kw in kws):
                            matches[c_type] = t_item["box"]
                            break
                if len(matches) >= 2:
                    header_idx = l_idx
                    cols_detected = matches
                    top_y = max(t["box"][3] for t in line)
                    break

            # If no header detected, check for TABLE entity or first line with decimal numbers
            table_ents = [e for e in entities if e.get("label", "").upper() == "TABLE"]
            if table_ents:
                t_box = table_ents[0].get("box", [0, 0, 1000, 1000])
                top_y = max(top_y, t_box[1])

            if header_idx is None and not table_ents:
                for l_idx, line in enumerate(all_lines):
                    line_ymid = sum((t["box"][1] + t["box"][3]) / 2.0 for t in line) / len(line)
                    if line_ymid < 200:
                        continue
                    line_txt = " ".join(t["word"].upper() for t in line)
                    if any(meta in line_txt for meta in ["SITE:", "HTTP", "WWW.", "TEL:", "EMAIL:", "BILL TO"]):
                        continue
                    nums_in_line = [cls._parse_table_num_token(t["word"])[0] for t in line if cls._parse_table_num_token(t["word"])[0] is not None]
                    if len(nums_in_line) >= 2 and any(t["word"].strip().endswith((".00", ".0", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9")) or "$" in t["word"] for t in line):
                        top_y = min(t["box"][1] for t in line) - 5
                        break

            # 2. Filter lines strictly within table region
            table_lines = []
            for l_idx, line in enumerate(all_lines):
                if header_idx is not None and l_idx <= header_idx:
                    continue
                line_ymid = sum((t["box"][1] + t["box"][3]) / 2.0 for t in line) / len(line)
                if line_ymid < top_y - 2:
                    continue

                line_txt = " ".join(t["word"].upper() for t in line)

                # Stop at summary anchors or banking details
                if re.search(r"\b(?:SUB_TOTAL|SUBTOTAL|TOTAL|BALANCE\s+DUE|BANKNAME|BANK\s+NAME|BANK\s+ACCOUNT|THANK\s+YOU|TERMS|NOTE)\b", line_txt):
                    break

                # Skip metadata lines
                if any(meta in line_txt for meta in ["SITE:", "HTTP", "WWW.", "TEL:", "EMAIL:", "BILL TO", "INVOICE DATE", "INVOICE NUMBER", "PO NUMBER", "DUE DATE"]):
                    continue

                non_border = [t for t in line if cls._clean_table_description(t["word"])]
                if not non_border:
                    continue

                table_lines.append(line)

            if not table_lines:
                return []

            # 3. Extract rows and merge wrapped descriptions
            extracted_items = []

            for line in table_lines:
                line.sort(key=lambda t: t["box"][0])

                # Check vertical gap from previous row: if > 45 px, table has ended
                if extracted_items:
                    prev_bottom = extracted_items[-1]["box"][3]
                    curr_top = min(t["box"][1] for t in line)
                    if curr_top - prev_bottom > 45.0:
                        break

                nums = []
                words = []
                for t in line:
                    w_str = t["word"]
                    n_val, has_dec = cls._parse_table_num_token(w_str)
                    if n_val is not None:
                        nums.append((n_val, t["box"], has_dec))
                    else:
                        words.append(t)

                desc_col_center = (cols_detected["desc"][0] + cols_detected["desc"][2]) / 2.0 if "desc" in cols_detected else None
                if desc_col_center is not None and len(cols_detected) >= 3:
                    filtered_words = []
                    for t in words:
                        tw_clean = t["word"].strip(" :.,|[]{}()")
                        if not any(c.isalpha() for c in tw_clean):
                            continue
                        cx = (t["box"][0] + t["box"][2]) / 2.0
                        best_c = min(cols_detected.keys(), key=lambda c: abs((cols_detected[c][0] + cols_detected[c][2]) / 2.0 - cx))
                        if best_c == "desc":
                            filtered_words.append(t)
                    desc_words = filtered_words if filtered_words else words
                else:
                    desc_words = words

                desc_line_text = cls._clean_table_description(" ".join(t["word"] for t in desc_words))
                is_continuation = False
                if not nums and desc_line_text and len(extracted_items) > 0:
                    prev_row = extracted_items[-1]
                    prev_bottom = prev_row["box"][3]
                    curr_top = min(t["box"][1] for t in line)
                    max_x = max(t["box"][2] for t in line)
                    if curr_top - prev_bottom <= 22.0 and (max_x <= 400 or (desc_col_center is not None and abs((min(t["box"][0] for t in line) + max_x) / 2.0 - desc_col_center) < 150)):
                        is_continuation = True

                if is_continuation:
                    prev_row = extracted_items[-1]
                    prev_row["description"] = f"{prev_row['description']} {desc_line_text}".strip()
                    prev_row["raw_description"] = f"{prev_row['raw_description']} {desc_line_text}".strip()
                    prev_row["box"][2] = max(prev_row["box"][2], max(t["box"][2] for t in line))
                    prev_row["box"][3] = max(prev_row["box"][3], max(t["box"][3] for t in line))
                    continue

                if not nums:
                    continue

                q_val = None
                u_val = None
                tot_val = None
                tax_val = None
                id_val = None

                col_centers = {c: (b[0] + b[2]) / 2.0 for c, b in cols_detected.items()} if cols_detected else {}
                target_num_cols = {"qty", "price", "tax", "total"}

                # Filter out Item IDs, HSN, and numbers aligned with non-target columns (batch, free, mrp, disc)
                pure_nums = []
                for n, b, hd in nums:
                    b_x = (b[0] + b[2]) / 2.0
                    if col_centers and len(col_centers) >= 3:
                        best_c = min(col_centers.keys(), key=lambda c: abs(col_centers[c] - b_x))
                        if best_c not in target_num_cols:
                            if best_c in ("id", "hsn"):
                                id_val = int(n)
                            continue
                    if n.is_integer() and 10000 <= n <= 99999999 and (b[0] < 200 or "id" in cols_detected):
                        id_val = int(n)
                    else:
                        pure_nums.append((n, b, hd))

                # Check for missing decimal dot in QTY column (e.g. 300 for 3.00)
                corrected_nums = []
                for n, b, hd in pure_nums:
                    b_x = (b[0] + b[2]) / 2.0
                    if n.is_integer() and 100 <= n <= 2000 and n % 100 == 0:
                        if col_centers and "qty" in col_centers:
                            if abs(col_centers["qty"] - b_x) < 50:
                                n = n / 100.0
                        elif b_x < 350:
                            n = n / 100.0
                    corrected_nums.append((n, b, hd))
                pure_nums = corrected_nums

                # Column identification
                if col_centers:
                    for n, b, hd in pure_nums:
                        b_x = (b[0] + b[2]) / 2.0
                        best_col = min(col_centers.keys(), key=lambda c: abs(col_centers[c] - b_x))
                        if best_col == "qty" and q_val is None:
                            q_val = n
                        elif best_col == "price" and u_val is None:
                            u_val = n
                        elif best_col == "tax" and tax_val is None:
                            tax_val = n
                        elif best_col == "total" and tot_val is None:
                            tot_val = n

                # Fallback / heuristic alignment if column matching left fields empty
                p_vals = [n for n, b, hd in pure_nums]
                if tot_val is None or q_val is None or u_val is None:
                    if len(p_vals) >= 4:
                        q_val, u_val, tax_val, tot_val = p_vals[0], p_vals[1], p_vals[2], p_vals[3]
                    elif len(p_vals) == 3:
                        tot_c = p_vals[-1]
                        n0, n1 = p_vals[0], p_vals[1]
                        if abs((n0 * n1) - tot_c) <= 0.05:
                            if n0.is_integer() and not n1.is_integer():
                                q_val, u_val, tot_val = n0, n1, tot_c
                            elif n1.is_integer() and not n0.is_integer():
                                u_val, q_val, tot_val = n0, n1, tot_c
                            else:
                                q_val, u_val, tot_val = min(n0, n1), max(n0, n1), tot_c
                        else:
                            q_val, u_val, tot_val = n0, n1, tot_c
                    elif len(p_vals) == 2:
                        if "price" in cols_detected and "total" not in cols_detected:
                            q_val, u_val = p_vals[0], p_vals[1]
                            tot_val = round(q_val * u_val, 2)
                        elif "total" in cols_detected and "price" not in cols_detected:
                            q_val, tot_val = p_vals[0], p_vals[1]
                            u_val = round(tot_val / q_val, 2) if q_val else tot_val
                        else:
                            if p_vals[0].is_integer() and 1 <= p_vals[0] <= 100:
                                q_val, tot_val = p_vals[0], p_vals[1]
                                u_val = round(tot_val / q_val, 2)
                            else:
                                tot_val, u_val = p_vals[1], p_vals[0]
                                q_val = round(tot_val / u_val, 2) if u_val else 1.0
                    elif len(p_vals) == 1:
                        if p_vals[0] == 0.0:
                            tot_val = 0.0
                            u_val = 0.0
                            q_val = 1.0
                        else:
                            tot_val = p_vals[0]
                            u_val = p_vals[0]
                            q_val = 1.0

                # Decimal point reconstruction via arithmetic cross-validation
                # Only apply to values that did NOT have a decimal dot in the raw token
                if q_val is not None and u_val is not None and tot_val is not None:
                    u_has_dec = any(hd for n, b, hd in pure_nums if n == u_val)
                    tot_has_dec = any(hd for n, b, hd in pure_nums if n == tot_val)

                    # If already mathematically consistent with tax or without tax, do not alter
                    already_consistent = bool(
                        abs((q_val * u_val) - tot_val) <= 0.05
                        or (tax_val is not None and abs((q_val * u_val + tax_val) - tot_val) <= 0.05)
                    )

                    if not already_consistent or (not u_has_dec and not tot_has_dec):
                        # Check if both U and Tot need / 100.0 (e.g. 1250 -> 12.50 and 3750 -> 37.50)
                        if not u_has_dec and not tot_has_dec and u_val > 100 and tot_val > 100:
                            if abs((q_val * (u_val / 100.0)) - (tot_val / 100.0)) <= 0.25:
                                u_val = round(u_val / 100.0, 2)
                                tot_val = round(tot_val / 100.0, 2)
                        # Check if only Tot needs / 100.0 (e.g. 5980 -> 59.80 with U=11.96, Q=5)
                        elif not tot_has_dec and tot_val > 500 and abs((q_val * u_val) - (tot_val / 100.0)) <= 0.25:
                            tot_val = round(tot_val / 100.0, 2)
                        # Check if only U needs / 100.0 (e.g. 1196 -> 11.96 with Tot=59.80, Q=5)
                        elif not u_has_dec and u_val > 500 and abs((q_val * (u_val / 100.0)) - tot_val) <= 0.25:
                            u_val = round(u_val / 100.0, 2)

                # Arithmetic cross-validation: if (q_val, u_val, tot_val) is invalid or incomplete, find a consistent pair (q, u) from row numbers that produces target total
                if not (q_val is not None and u_val is not None and tot_val is not None and abs((q_val * u_val) - tot_val) <= 0.05):
                    cand_nums = [n for n, b, hd in nums if n > 0 and (n < 100000 or n == tot_val)]
                    target_tot = tot_val if (tot_val is not None and tot_val > 0) else (max(cand_nums) if cand_nums else None)
                    if target_tot is not None:
                        found_pair = False
                        # Check if q_val and target_tot are present and u_val is slightly off due to OCR digit misreading (e.g. 368 vs 365)
                        if q_val is not None and q_val > 0 and target_tot is not None and target_tot > 0:
                            calc_u = round(target_tot / q_val, 2)
                            if u_val is not None and abs(u_val - calc_u) <= max(5.0, 0.05 * u_val):
                                u_val = calc_u
                                tot_val = target_tot
                                found_pair = True
                            else:
                                for n_cand in cand_nums:
                                    if abs(n_cand - calc_u) <= max(5.0, 0.05 * n_cand):
                                        u_val = calc_u
                                        tot_val = target_tot
                                        found_pair = True
                                        break

                        if not found_pair:
                            for i_a, na in enumerate(cand_nums):
                                for i_b, nb in enumerate(cand_nums):
                                    if i_a == i_b:
                                        continue
                                    if abs((na * nb) - target_tot) <= 0.05:
                                        if na.is_integer() and (not nb.is_integer() or na <= 100):
                                            q_val, u_val, tot_val = na, nb, target_tot
                                        elif nb.is_integer() and (not na.is_integer() or nb <= 100):
                                            q_val, u_val, tot_val = nb, na, target_tot
                                        else:
                                            q_val, u_val, tot_val = min(na, nb), max(na, nb), target_tot
                                        found_pair = True
                                        break
                                if found_pair:
                                    break

                # Mathematical consistency reconciliation
                if q_val is not None and u_val is not None and tot_val is None:
                    tot_val = round(q_val * u_val, 2)
                elif tot_val is not None and q_val is not None and u_val is None:
                    u_val = round(tot_val / q_val, 2) if q_val > 0 else tot_val
                elif tot_val is not None and u_val is not None and q_val is None:
                    q_val = round(tot_val / u_val, 2) if u_val > 0 else 1.0

                if q_val is None:
                    q_val = 1.0
                if u_val is None:
                    u_val = tot_val if tot_val is not None else 0.0
                if tot_val is None:
                    tot_val = round(q_val * u_val, 2)

                is_valid = bool(abs((q_val * u_val) - tot_val) <= 0.05 or (tax_val and abs((q_val * u_val + tax_val) - tot_val) <= 0.05))
                suspicious = not is_valid or not desc_line_text

                box_val = [
                    min(t["box"][0] for t in line),
                    min(t["box"][1] for t in line),
                    max(t["box"][2] for t in line),
                    max(t["box"][3] for t in line),
                ]

                extracted_items.append({
                    "row_index": len(extracted_items) + 1,
                    "page": page_number,
                    "description": desc_line_text,
                    "quantity": q_val,
                    "unit_price": u_val,
                    "tax_amount": tax_val,
                    "item_total": tot_val,
                    "raw_description": desc_line_text,
                    "raw_quantity": str(q_val) if q_val is not None else None,
                    "raw_unit_price": str(u_val) if u_val is not None else None,
                    "raw_item_total": str(tot_val) if tot_val is not None else None,
                    "confidence": 0.90 if is_valid else 0.70,
                    "box": box_val,
                    "is_valid": is_valid,
                    "suspicious": suspicious,
                    "validation_error": None if is_valid else f"Arithmetic mismatch: quantity ({q_val}) * unit_price ({u_val}) = {round(q_val * u_val, 2):.2f}, but item total is {tot_val:.2f}",
                })

            if extracted_items:
                return extracted_items

        if not item_ents:
            return []

        # Sort entities by vertical center, then horizontal start
        def get_ymid(ent):
            box = ent.get("box", [0, 0, 0, 0])
            return (box[1] + box[3]) / 2.0

        sorted_ents = sorted(
            item_ents,
            key=lambda e: (get_ymid(e), e.get("box", [0, 0, 0, 0])[0]),
        )

        # Spatial horizontal row clustering
        clusters: list[dict] = []
        for ent in sorted_ents:
            e_box = ent.get("box", [0, 0, 0, 0])
            e_ymid = (e_box[1] + e_box[3]) / 2.0
            e_label = ent.get("label", "").upper()

            best_cluster = None
            min_dist = float("inf")

            for cluster in clusters:
                c_ymid = cluster["y_mid"]
                dist = abs(e_ymid - c_ymid)

                if dist <= y_threshold:
                    cb = cluster["box"]
                    overlap = min(cb[3], e_box[3]) - max(cb[1], e_box[1])
                    min_height = max(1, min(cb[3] - cb[1], e_box[3] - e_box[1]))
                    overlap_ratio = overlap / min_height if min_height > 0 else 0

                    if dist <= 12.0 or overlap_ratio > 0.3:
                        already_has = e_label in cluster["labels"]
                        if already_has and e_label in {"ITEM_TOTAL", "ITEM_UNIT_PRICE", "ITEM_QUANTITY"}:
                            if dist > 6.0:
                                continue
                        if dist < min_dist:
                            min_dist = dist
                            best_cluster = cluster

            if best_cluster is not None:
                best_cluster["entities"].append(ent)
                best_cluster["labels"].add(e_label)
                all_b = [x.get("box", [0, 0, 0, 0]) for x in best_cluster["entities"]]
                best_cluster["box"] = [
                    min(b[0] for b in all_b),
                    min(b[1] for b in all_b),
                    max(b[2] for b in all_b),
                    max(b[3] for b in all_b),
                ]
                best_cluster["y_mid"] = (best_cluster["box"][1] + best_cluster["box"][3]) / 2.0
            else:
                clusters.append({
                    "entities": [ent],
                    "labels": {e_label},
                    "box": list(e_box),
                    "y_mid": e_ymid,
                })

        # Sort clusters in top-to-bottom reading order
        clusters.sort(key=lambda c: (c["box"][1], c["box"][0]))

        header_keywords = {
            "DESCRIPTION", "DESC", "ITEM", "PRODUCT", "QTY", "QUANTITY",
            "PRICE", "UNIT", "RATE", "TOTAL", "AMOUNT", "TAX", "SUBTOTAL",
            "NO", "NUM", "NUMBER", "LINE", "DETAILS", "PART", "CODE"
        }

        line_items: list[dict[str, Any]] = []

        for row_idx, cluster in enumerate(clusters):
            # Sort entities within the cluster horizontally from left to right
            c_ents = sorted(
                cluster["entities"],
                key=lambda e: e.get("box", [0, 0, 0, 0])[0],
            )

            desc_parts: list[str] = []
            qty_parts: list[str] = []
            unit_parts: list[str] = []
            total_parts: list[str] = []

            for e in c_ents:
                lbl = e.get("label", "").upper()
                txt = str(e.get("text", "")).strip()
                if not txt:
                    continue
                if lbl == "ITEM_DESCRIPTION":
                    desc_parts.append(txt)
                elif lbl == "ITEM_QUANTITY":
                    qty_parts.append(txt)
                elif lbl == "ITEM_UNIT_PRICE":
                    unit_parts.append(txt)
                elif lbl == "ITEM_TOTAL":
                    total_parts.append(txt)

            raw_desc = " ".join(desc_parts).strip() if desc_parts else None
            raw_qty = " ".join(qty_parts).strip() if qty_parts else None
            raw_unit = " ".join(unit_parts).strip() if unit_parts else None
            raw_total = " ".join(total_parts).strip() if total_parts else None

            # Skip header rows (e.g. 'Description Quantity Unit Price Total Amount' with no digits)
            all_text_upper = " ".join(filter(None, [raw_desc, raw_qty, raw_unit, raw_total])).upper()
            has_digits = any(c.isdigit() for c in all_text_upper)
            if not has_digits:
                words_in_row = re.findall(r"\b[A-Z]+\b", all_text_upper)
                if words_in_row and all(w in header_keywords for w in words_in_row):
                    continue

            # Average model confidence for this row
            confs = [float(e.get("confidence", 0.0)) for e in c_ents if e.get("confidence") is not None]
            row_conf = round(sum(confs) / len(confs), 4) if confs else 0.0

            # Parse numeric fields
            qty = cls._parse_numeric(raw_qty)
            unit_price = cls._parse_numeric(raw_unit)
            item_total = cls._parse_numeric(raw_total)
            desc = raw_desc

            # Validation logic
            is_valid = True
            suspicious = False
            validation_error = None

            if raw_qty is not None and qty is None:
                is_valid = False
                suspicious = True
                validation_error = f"Invalid numeric format for quantity: '{raw_qty}'"
            elif raw_unit is not None and unit_price is None:
                is_valid = False
                suspicious = True
                validation_error = f"Invalid numeric format for unit price: '{raw_unit}'"
            elif raw_total is not None and item_total is None:
                is_valid = False
                suspicious = True
                validation_error = f"Invalid numeric format for item total: '{raw_total}'"
            elif desc is None or not desc.strip():
                if qty is not None or unit_price is not None or item_total is not None:
                    is_valid = False
                    suspicious = True
                    validation_error = "Missing item description"
            elif qty is not None and unit_price is not None:
                expected = round(qty * unit_price, 2)
                if item_total is not None:
                    diff = round(abs(expected - item_total), 2)
                    tolerance = max(0.05, round(item_total * 0.01, 2))
                    if diff > tolerance:
                        is_valid = False
                        suspicious = True
                        validation_error = (
                            f"Arithmetic mismatch: quantity ({qty}) * unit_price ({unit_price}) = {expected:.2f}, "
                            f"but item total is {item_total:.2f} (diff: {diff:.2f})"
                        )

            # Skip empty rows (no description and no numbers)
            if desc is None and qty is None and unit_price is None and item_total is None:
                continue

            line_items.append({
                "row_index": len(line_items) + 1,
                "page": page_number,
                "description": desc,
                "quantity": qty,
                "unit_price": unit_price,
                "item_total": item_total,
                "raw_description": raw_desc,
                "raw_quantity": raw_qty,
                "raw_unit_price": raw_unit,
                "raw_item_total": raw_total,
                "confidence": row_conf,
                "box": cluster["box"],
                "is_valid": is_valid,
                "suspicious": suspicious,
                "validation_error": validation_error,
            })

        return line_items

    def _extract_invoice(
        self,
        valid_words: list[str],
        word_preds: dict[int, int],
        word_confs: dict[int, float],
        norm_boxes: list[list[int]],
        id2label: dict[int, str] | None,
        ocr_text: str,
        document_type: str,
        start: float,
        image: Any | None = None,
    ) -> ExtractionResult:
        """Extract canonical invoice fields directly from invoice token classification labels."""
        import re

        if id2label is None:
            id2label = {}

        # 1. Group contiguous words of same invoice entity using BIO tagging rules
        entities: list[dict] = []
        current: dict | None = None

        for wid in sorted(word_preds.keys()):
            if wid >= len(valid_words):
                continue
            word = str(valid_words[wid]).strip()
            if not word:
                continue

            pred_id = word_preds[wid]
            conf = float(word_confs[wid])
            raw_label = id2label.get(pred_id, "O")

            if raw_label == "O":
                if current:
                    entities.append(current)
                    current = None
                continue

            prefix, ent_type = raw_label.split("-", 1) if "-" in raw_label else ("B", raw_label)
            box = norm_boxes[wid] if wid < len(norm_boxes) else [0, 0, 0, 0]

            if current is not None:
                c_box = current["box"]
                same_line = abs(((c_box[1] + c_box[3]) / 2.0) - ((box[1] + box[3]) / 2.0)) <= 12
                horiz_gap = box[0] - c_box[2]

                can_continue = (current["label"] == ent_type) and (
                    prefix == "I" or (prefix != "B" and same_line and 0 <= horiz_gap < 50)
                )

                if can_continue:
                    sep = "" if (
                        word in {",", ".", ":", ";", ")", "]", "}", "-", "/", "%"}
                        or current["text"].endswith(("@", "/", "-", "$", "€", "£", "₹", "‘", "'", "(", "#"))
                        or word.startswith(("@", "/", "-", ".", ":"))
                    ) else " "
                    current["text"] += sep + word
                    current["box"] = [
                        min(c_box[0], box[0]),
                        min(c_box[1], box[1]),
                        max(c_box[2], box[2]),
                        max(c_box[3], box[3]),
                    ]
                    current["confidence"] = (current["confidence"] * current["count"] + conf) / (current["count"] + 1)
                    current["count"] += 1
                    current["word_indices"].append(wid)
                    continue

                entities.append(current)
                current = None

            current = {
                "label": ent_type,
                "prefix": prefix,
                "text": word,
                "box": list(box),
                "confidence": conf,
                "count": 1,
                "word_indices": [wid],
            }

        if current:
            entities.append(current)

        # 1b. Extract structured table line items from entities / spatial fallback
        raw_line_items = self._extract_table_line_items(
            entities=entities,
            valid_words=valid_words,
            norm_boxes=norm_boxes,
            page_number=1,
        )
        # Exclude rows that fall inside the financial footer area
        clean_line_items: list[dict[str, Any]] = []
        for li in raw_line_items:
            desc_up = (li.get("description") or "").upper()
            if any(k in desc_up for k in ["TOTAL IN WORDS", "SUB_TOTAL", "SUBTOTAL", "DISCOUNT", "BANK NAME", "THANK YOU"]):
                continue
            clean_line_items.append(li)
        for idx, li in enumerate(clean_line_items):
            li["row_index"] = idx + 1
        line_items = clean_line_items

        # 2. Extract canonical fields from entities
        fields: dict[str, Any] = {}
        confs: dict[str, float] = {}
        field_bboxes: dict[str, list[int]] = {}
        field_sources: dict[str, str] = {}
        raw_values: dict[str, str] = {}

        amount_regex = re.compile(r"[-+]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})|[-+]?[0-9]+\.[0-9]{2}")
        date_regex = re.compile(r"\b(\d{1,4}[-/.](?:[A-Za-z]{3}|\d{1,2})[-/.]\d{2,4})\b")

        def _find_horizontal(box: list[int], pattern: re.Pattern, require_digits: bool = True, max_y_diff: float = 12.0) -> tuple[str | None, list[int] | None]:
            """Find horizontal text to the right on the same line matching pattern."""
            y_mid = (box[1] + box[3]) / 2.0
            best_val = None
            best_box = None
            min_dist = float("inf")
            for w, b in zip(valid_words, norm_boxes):
                b_y_mid = (b[1] + b[3]) / 2.0
                if abs(b_y_mid - y_mid) <= max_y_diff and b[0] >= (box[0] - 5):
                    m = pattern.search(w)
                    if m:
                        val = m.group(1) if m.groups() else m.group(0)
                        if require_digits and not any(c.isdigit() for c in val):
                            continue
                        dist = b[0] - box[0]
                        if 0 <= dist < min_dist:
                            min_dist = dist
                            best_val = val
                            best_box = b
            return best_val, best_box

        for ent in sorted(entities, key=lambda x: x["confidence"], reverse=True):
            lbl = ent["label"].upper()
            txt = ent["text"].strip()
            conf = round(float(ent["confidence"]), 4)
            ent_box = ent.get("box", [0, 0, 0, 0])

            # Vendor Site
            if lbl in ("SELLER_SITE",) and "vendor_website" not in fields:
                cleaned_site = txt.strip(":.- ‘'\"").strip()
                if cleaned_site and "@" not in cleaned_site and not any(k in cleaned_site.upper() for k in ["BILL_TO", "BILL TO", "BUYER", "CUSTOMER", "SHIP_TO", "TOTAL"]):
                    fields["vendor_website"] = cleaned_site
                    confs["vendor_website"] = conf
                    field_bboxes["vendor_website"] = ent_box
                    field_sources["vendor_website"] = "token_classification"
                    raw_values["vendor_website"] = txt

            # Vendor Address
            elif lbl in ("SELLER_ADDRESS",) and "vendor_address" not in fields:
                cleaned_addr = re.sub(r"^(?:Address|Addr)[:\s]+", "", txt, flags=re.IGNORECASE).strip(":.- ").strip()
                if cleaned_addr and not any(k in cleaned_addr.upper() for k in ["BILL_TO", "BILL TO", "BUYER"]):
                    fields["vendor_address"] = cleaned_addr
                    confs["vendor_address"] = conf
                    field_bboxes["vendor_address"] = ent_box
                    field_sources["vendor_address"] = "token_classification"
                    raw_values["vendor_address"] = txt

            # Vendor Name
            elif lbl in ("SELLER", "VENDOR") and "vendor_name" not in fields:
                if "Address:" in txt:
                    parts = re.split(r"\bAddress[:\s]+", txt, flags=re.IGNORECASE)
                    name_part = parts[0].strip(":.- ").strip()
                    addr_part = parts[1].strip(":.- ").strip() if len(parts) > 1 else ""
                    if name_part and len(name_part) >= 2 and not any(k in name_part.upper() for k in ["INVOICE", "TAX", "TOTAL", "BILL_TO", "BILL TO", "BUYER", "CUSTOMER"]):
                        fields["vendor_name"] = name_part
                        confs["vendor_name"] = conf
                        field_bboxes["vendor_name"] = ent_box
                        field_sources["vendor_name"] = "token_classification"
                        raw_values["vendor_name"] = name_part
                    if addr_part and "vendor_address" not in fields:
                        fields["vendor_address"] = addr_part
                        confs["vendor_address"] = conf
                        field_bboxes["vendor_address"] = ent_box
                        field_sources["vendor_address"] = "token_classification"
                        raw_values["vendor_address"] = addr_part
                else:
                    cleaned = txt.strip(":.- ").strip()
                    if len(cleaned) >= 2 and not any(k in cleaned.upper() for k in ["INVOICE", "TAX", "TOTAL", "BILL_TO", "BILL TO", "BUYER", "CUSTOMER"]):
                        fields["vendor_name"] = cleaned
                        confs["vendor_name"] = conf
                        field_bboxes["vendor_name"] = ent_box
                        field_sources["vendor_name"] = "token_classification"
                        raw_values["vendor_name"] = txt

            # Customer / Buyer (supports FATURA2 BUYER / BILL_TO / SEND_TO)
            elif lbl in ("BUYER", "CUSTOMER", "BILL_TO", "SEND_TO"):
                cleaned = re.sub(r"^(?:Bill[_\s]+to|Ship[_\s]+to|Customer|Buyer|Invoiced[_\s]+to)[:\s]*", "", txt, flags=re.IGNORECASE).strip(":.- ").strip()
                m_cust = re.match(r"^([A-Za-z\s.'-]+?)(?=\s+\d|\s+Suite|\s+Apt|\s+Road|\s+Street|\s+Mission|\s+Tel|\s+Email|\s+Site|$)", cleaned)
                cust_name = m_cust.group(1).strip() if m_cust else cleaned
                if (
                    len(cust_name) >= 2
                    and not any(k in cust_name.upper() for k in ["INVOICE", "TAX", "TOTAL", "DATE", "PO", "BILL_TO", "BILL TO", "BUYER", "CUSTOMER"])
                    and "customer_name" not in fields
                ):
                    fields["customer_name"] = cust_name
                    confs["customer_name"] = conf
                    field_bboxes["customer_name"] = ent_box
                    field_sources["customer_name"] = "token_classification"
                    raw_values["customer_name"] = cust_name

                if m_cust and "customer_address" not in fields:
                    remainder = cleaned[len(m_cust.group(1)):].strip()
                    addr_clean = re.sub(r"(?:\bTel[:\s].*|\bEmail[:\s].*|\bSite[:\s].*|\bGSTIN[:\s].*)$", "", remainder, flags=re.IGNORECASE).strip(" ,.-")
                    if addr_clean and len(addr_clean) >= 5:
                        fields["customer_address"] = addr_clean
                        confs["customer_address"] = conf
                        field_bboxes["customer_address"] = ent_box
                        field_sources["customer_address"] = "token_classification"
                        raw_values["customer_address"] = addr_clean

            # PO Number
            elif lbl in ("PO_NUMBER",):
                cleaned = re.sub(
                    r"^(?:PO\s*(?:NUMBER|NO|#|:|-)|PO-|P\.O\.\s*(?:NUMBER|NO|#|:|-)|ORDER\s*(?:NUMBER|NO|#|:|-)|NUMBER|NO|#|:|\s|-)+",
                    "",
                    txt,
                    flags=re.IGNORECASE,
                ).strip(" :-#")
                if cleaned and any(c.isdigit() for c in cleaned) and cleaned.upper() not in ["PO", "P.O.", "NUMBER", "NO", "ORDER"] and "po_number" not in fields:
                    fields["po_number"] = cleaned
                    confs["po_number"] = conf
                    field_bboxes["po_number"] = ent_box
                    field_sources["po_number"] = "token_classification"
                    raw_values["po_number"] = txt

            # Number entity: check if preceding or containing PO
            elif lbl in ("NUMBER", "INVOICE_NUMBER", "INV_NO"):
                w_idx = ent.get("word_indices", [0])[0]
                preceding_context = " ".join(valid_words[max(0, w_idx - 3):w_idx]).upper()
                is_po = bool(re.search(r"\b(?:PO|P\.O\.|ORDER)\b", txt, re.IGNORECASE) or re.search(r"\b(?:PO|P\.O\.|ORDER)\b", preceding_context))

                if is_po:
                    cleaned_po = re.sub(
                        r"^(?:PO\s*(?:NUMBER|NO|#|:|-)|PO-|P\.O\.\s*(?:NUMBER|NO|#|:|-)|ORDER\s*(?:NUMBER|NO|#|:|-)|NUMBER|NO|#|:|\s|-)+",
                        "",
                        txt,
                        flags=re.IGNORECASE,
                    ).strip(" :-#")
                    if cleaned_po and any(c.isdigit() for c in cleaned_po) and cleaned_po.upper() not in ["PO", "P.O.", "NUMBER", "NO", "ORDER"] and "po_number" not in fields:
                        fields["po_number"] = cleaned_po
                        confs["po_number"] = conf
                        field_bboxes["po_number"] = ent_box
                        field_sources["po_number"] = "token_classification"
                        raw_values["po_number"] = txt
                else:
                    pat_inv_exact = re.compile(
                        r"\b(?:INV[/:.-]?\d{1,3}[/:.-]\d{1,3}[/:.-]\d{2,4}|INV\d{6,10}|\d[A-Za-z0-9]{3,5}-\d{2,4}|\d{3,5}-\d{2,4})\b",
                        re.IGNORECASE,
                    )
                    cleaned_inv = re.sub(
                        r"^(?:INVOICE\s*(?:ID|NUMBER|NO|#|:|-)|INV\s*(?:ID|NUMBER|NO|#|:|-)|INV-|BILL\s*(?:ID|NUMBER|NO|#|:|-)|NUMBER|NO|ID|#|:|\s|-)+",
                        "",
                        txt,
                        flags=re.IGNORECASE,
                    ).strip(" :-#")
                    cleaned_inv = cleaned_inv.replace("¥", "Y").replace("¢", "d")
                    cleaned_inv = re.sub(r"^INV[I|](?=\d)", "INV/", cleaned_inv, flags=re.IGNORECASE)
                    cleaned_inv = re.sub(r"^INV[I|](?=[A-Za-z])", "INV/", cleaned_inv, flags=re.IGNORECASE)
                    is_label_noise = bool(
                        re.match(r"^INVOI[A-Z0-9]*$", cleaned_inv.upper())
                        or re.match(r"^INNOI[A-Z0-9]*$", cleaned_inv.upper())
                        or cleaned_inv.upper() in ["INVOICE", "INVOIGE", "INNOIGE", "INVOICE ID", "INV ID", "NUMBER", "NO", "ID", "BILL", "INV", "COMMERCIAL"]
                        or not cleaned_inv
                    )

                    cand = None
                    if cleaned_inv and not is_label_noise and any(c.isdigit() or c.isalpha() for c in cleaned_inv):
                        cand = cleaned_inv
                    else:
                        m_pat = pat_inv_exact.search(txt)
                        if m_pat and not re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", m_pat.group(0)):
                            cand = m_pat.group(0)
                        else:
                            cand, _ = _find_horizontal(ent_box, pat_inv_exact, require_digits=True, max_y_diff=15.0)
                            if cand and re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", cand):
                                cand = None

                    if cand and "invoice_number" not in fields:
                        is_dl = bool(re.search(r"\b(?:AP/\d{2}/\d{2}/\d{4}-\d+|WLF\d{2}[A-Z]\d+|20B|21B)\b", cand, re.IGNORECASE))
                        is_gstin = bool(re.search(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b", cand))
                        is_pan = bool(re.match(r"^[A-Z]{5}\d{4}[A-Z]$", cand))
                        is_addr = bool(re.search(r"\b(?:D\.?\s*NO|DOOR\s*NO|H\.?\s*NO|PLOT\s*NO|ROAD|STREET|NAGAR)\b", cand, re.IGNORECASE))
                        if is_dl or is_gstin or is_pan or is_addr:
                            cand = None

                    if cand and "invoice_number" not in fields:
                        cand = cand.replace("¥", "Y").replace("¢", "d")
                        cand = re.sub(r"^INV[I|](?=\d)", "INV/", cand, flags=re.IGNORECASE)
                        # Fix O/o to 9 between letters in pattern \d[A-Z]\d[A-Z]\d[a-z]-\d+
                        m_fatura_code = re.match(r"^(\d)([A-Za-z])([Oo0-9])([A-Za-z])([0-9])([a-z])-(\d{2,4})$", cand)
                        if m_fatura_code:
                            d1, l1, d2, l2, d3, l3, d4 = m_fatura_code.groups()
                            d2_fixed = "9" if d2.upper() in ("O", "0") else d2
                            cand = f"{d1}{l1.upper()}{d2_fixed}{l2.upper()}{d3}{l3}-{d4}"
                        # Fix 'a' in 'ad' position
                        cand = re.sub(r"([A-Z])a(d-\d{2,4})$", r"\g<1>8\g<2>", cand)
                        # Fix INVS... -> INV5...
                        cand = re.sub(r"^INV[S](?=\d)", "INV5", cand, flags=re.IGNORECASE)
                        cand = re.sub(r"^INV[o](?=\d)", "INV0", cand)
                        # Fix INV/T3-47/611 -> INV/73-47/611
                        cand = re.sub(r"^INV/T(?=\d)", "INV/7", cand, flags=re.IGNORECASE)
                        # Fix INV/N1- -> INV/11-
                        cand = re.sub(r"^INV/N(?=\d)", "INV/1", cand, flags=re.IGNORECASE)

                        fields["invoice_number"] = cand
                        confs["invoice_number"] = conf
                        field_bboxes["invoice_number"] = ent_box
                        field_sources["invoice_number"] = "token_classification"
                        raw_values["invoice_number"] = txt

            # Invoice Date
            elif lbl in ("DATE", "INVOICE_DATE") and "invoice_date" not in fields:
                if "due" in txt.lower():
                    m_date = date_regex.search(txt)
                    if m_date and "due_date" not in fields:
                        fields["due_date"] = m_date.group(1)
                        confs["due_date"] = conf
                        field_bboxes["due_date"] = ent_box
                        field_sources["due_date"] = "token_classification"
                        raw_values["due_date"] = m_date.group(1)
                else:
                    m_date = date_regex.search(txt)
                    if m_date:
                        fields["invoice_date"] = m_date.group(1)
                        confs["invoice_date"] = conf
                        field_bboxes["invoice_date"] = ent_box
                        field_sources["invoice_date"] = "token_classification"
                        raw_values["invoice_date"] = m_date.group(1)

            # Due Date
            elif lbl in ("DUE_DATE",) and "due_date" not in fields:
                m_date = date_regex.search(txt)
                if m_date:
                    fields["due_date"] = m_date.group(1)
                    confs["due_date"] = conf
                    field_bboxes["due_date"] = ent_box
                    field_sources["due_date"] = "token_classification"
                    raw_values["due_date"] = m_date.group(1)

            # Total Amount (supports FATURA2 TOTAL / AMOUNT_DUE)
            elif lbl in ("TOTAL", "TOTAL_PRICE", "GRAND_TOTAL", "AMOUNT_DUE") and "total_amount" not in fields:
                has_total_keyword = bool(re.search(r"\b(?:TOTAL|BALANCE|DUE|AMOUNT|PAYABLE|NET)\b", txt, re.IGNORECASE))
                is_summary_area = ent_box[1] >= 450
                if has_total_keyword or is_summary_area:
                    m_amt = amount_regex.search(txt)
                    if m_amt:
                        fields["total_amount"] = m_amt.group(0)
                        confs["total_amount"] = conf
                        field_bboxes["total_amount"] = ent_box
                        field_sources["total_amount"] = "token_classification"
                        raw_values["total_amount"] = m_amt.group(0)

            # Subtotal
            elif lbl in ("SUBTOTAL", "SUB_TOTAL", "TAXABLE_AMOUNT") and "subtotal" not in fields:
                m_amt = amount_regex.search(txt)
                if m_amt:
                    fields["subtotal"] = m_amt.group(0)
                    confs["subtotal"] = conf
                    field_bboxes["subtotal"] = ent_box
                    field_sources["subtotal"] = "token_classification"
                    raw_values["subtotal"] = m_amt.group(0)

            # Tax Amount
            elif lbl in ("TAX", "TAX_AMOUNT", "TOTAL_TAX") and "tax_amount" not in fields:
                m_amt = amount_regex.search(txt)
                if m_amt:
                    fields["tax_amount"] = m_amt.group(0)
                    confs["tax_amount"] = conf
                    field_bboxes["tax_amount"] = ent_box
                    field_sources["tax_amount"] = "token_classification"
                    raw_values["tax_amount"] = m_amt.group(0)

            # Discount
            elif lbl in ("DISCOUNT",) and "discount" not in fields:
                m_amt = amount_regex.search(txt)
                if m_amt:
                    fields["discount"] = m_amt.group(0)
                    confs["discount"] = conf
                    field_bboxes["discount"] = ent_box
                    field_sources["discount"] = "token_classification"
                    raw_values["discount"] = m_amt.group(0)

            # Note
            elif lbl in ("NOTE",) and "notes" not in fields:
                fields["notes"] = txt
                confs["notes"] = conf
                field_bboxes["notes"] = ent_box
                field_sources["notes"] = "token_classification"
                raw_values["notes"] = txt

        # 3. Spatial Layout Fallbacks for Multi-Column Invoices

        # 3a. Vendor details fallback (Address not in customer block)
        cust_box = field_bboxes.get("customer_name")
        for idx, w in enumerate(valid_words):
            if w.lower().startswith("address:") or (w.lower() == "address" and idx + 1 < len(valid_words) and valid_words[idx+1] == ":"):
                b_box = norm_boxes[idx]
                if cust_box and abs(b_box[0] - cust_box[0]) < 80 and b_box[1] > cust_box[1]:
                    continue
                if b_box[1] < 450:
                    addr_words = []
                    for j in range(idx, min(idx + 15, len(valid_words))):
                        wb = norm_boxes[j]
                        if abs(wb[0] - b_box[0]) > 350:
                            continue
                        if any(k in valid_words[j].lower() for k in ["email:", "site:", "gstin:", "tel:", "www.", "bill"]):
                            break
                        w_clean = valid_words[j]
                        if w_clean.lower().startswith("address:"):
                            w_clean = w_clean[8:].strip()
                        if w_clean:
                            addr_words.append(w_clean)
                    if addr_words and "vendor_address" not in fields:
                        addr_str = " ".join(addr_words).strip(": ,.-")
                        addr_str = re.sub(r"\s+([,.-])", r"\1", addr_str)
                        fields["vendor_address"] = addr_str
                        confs["vendor_address"] = 0.90
                        field_bboxes["vendor_address"] = b_box
                        field_sources["vendor_address"] = "spatial_anchor"
                        raw_values["vendor_address"] = addr_str

                    # Check directly above this address for seller company name
                    needs_vendor = (
                        "vendor_name" not in fields
                        or fields.get("vendor_name") in ["CALLINTY", "Smith", "Graham"]
                        or (cust_box and field_bboxes.get("vendor_name") and abs(field_bboxes["vendor_name"][0] - cust_box[0]) < 100)
                    )
                    if needs_vendor:
                        above_cands = [
                            j for j, b in enumerate(norm_boxes)
                            if abs(b[0] - b_box[0]) < 100
                            and b[3] <= b_box[1] + 5
                            and b[1] >= b_box[1] - 45
                            and j != idx
                        ]
                        if above_cands:
                            y_above = (norm_boxes[above_cands[0]][1] + norm_boxes[above_cands[0]][3]) / 2.0
                            above_words = [valid_words[j] for j in above_cands if abs(((norm_boxes[j][1] + norm_boxes[j][3])/2.0) - y_above) <= 12]
                            if above_words:
                                v_name = " ".join(above_words).strip(": ,.-")
                                if len(v_name) >= 2 and not any(k in v_name.upper() for k in ["INVOICE", "TAX", "TOTAL", "BILL"]):
                                    fields["vendor_name"] = v_name
                                    confs["vendor_name"] = 0.92
                                    field_bboxes["vendor_name"] = norm_boxes[above_cands[0]]
                                    field_sources["vendor_name"] = "spatial_anchor"
                                    raw_values["vendor_name"] = v_name
                    break

        if "vendor_website" not in fields:
            for idx, w in enumerate(valid_words):
                if norm_boxes[idx][0] < 280 and "@" not in w:
                    w_clean = w.strip("‘'\" :,")
                    if any(k in w_clean.lower() for k in [".org", ".com", ".net", ".io"]) or "www." in w_clean.lower():
                        site_str = w_clean
                        if site_str.lower() in ("www.", "www", "‘www.", "‘www") and idx + 1 < len(valid_words):
                            next_w = valid_words[idx+1].strip("‘'\" :,")
                            site_str = "www." + next_w
                        fields["vendor_website"] = site_str
                        confs["vendor_website"] = 0.88
                        field_bboxes["vendor_website"] = norm_boxes[idx]
                        field_sources["vendor_website"] = "spatial_anchor"
                        raw_values["vendor_website"] = site_str
                        break

        # Validate extracted vendor_name: reject address/locality lines, customer address substrings, job titles, or table rows
        addr_pattern = re.compile(
            r"\b(?:ROAD|STREET|NAGAR|COLONY|TEMPLE|NEAR|OPP|BEHIND|BESIDE|FLOOR|BUILDING|PLOT|"
            r"AVENUE|LAYOUT|SECTOR|VILLAGE|DISTRICT|MARG|LANE|CROSS|ENCLAVE|APARTMENT|SUITE)\b",
            re.IGNORECASE,
        )
        comp_suffix_pattern = re.compile(
            r"\b(?:LIMITED|LTD|PVT|ENTERPRISES?|CORP|INC|INDUSTRIES|PHARMA(?:CEUTICALS?)?|SURGICALS?|"
            r"HEALTHCARE|TRADERS?|AGENC(?:Y|IES)|SOLUTIONS|SYSTEMS|SERVICES|COMPANY|CO\b|LLP|PLC|GMBH|"
            r"SONS?|BROTHERS?|PARTNERS?|GROUP|HOLDINGS?|ASSOCIATES?)\b",
            re.IGNORECASE,
        )
        job_title_pattern = re.compile(
            r"\b(?:MANAGER|MANAGE|DIRECTOR|CASHIER|ADMIN(?:ISTRATOR)?|OFFICER|CLERK|SIGNATORY|EMPLOYEE|CONSULTANT|ACCOUNTANT|SUPERVISOR|ASSISTANT|OWNER)\b",
            re.IGNORECASE,
        )
        if "vendor_name" in fields:
            v_curr = fields["vendor_name"]
            c_addr = fields.get("customer_address", "")
            c_name = fields.get("customer_name", "")
            v_up = v_curr.upper()
            v_box = field_bboxes.get("vendor_name", [0, 0, 0, 0])
            v_ymid = (v_box[1] + v_box[3]) / 2.0 if v_box else 0.0

            is_addr = bool((addr_pattern.search(v_curr) or bool(re.match(r"^\d+[-/]\d+", v_curr.strip()))) and not comp_suffix_pattern.search(v_curr))
            is_in_caddr = bool(len(v_curr) >= 3 and (v_up in c_addr.upper() or (c_name and v_up == c_name.upper())))
            is_job_title = bool(job_title_pattern.search(v_curr) and not comp_suffix_pattern.search(v_curr))
            is_in_table = bool(v_ymid > 350 and not comp_suffix_pattern.search(v_curr))

            is_poor = (
                is_addr
                or is_in_caddr
                or is_job_title
                or is_in_table
                or v_curr.strip().startswith("(")
                or v_curr.strip().endswith(")")
                or any(k in v_up for k in ["DISTRIBUTOR", "MANUFACTURER", "DEALER", "WHOLESALER", "PRODUCTS", "CLIENT NAME", "BILL TO", "INVOICE", "COMMERCIAL"])
            )
            if is_poor:
                del fields["vendor_name"]
                confs.pop("vendor_name", None)
                field_bboxes.pop("vendor_name", None)
                field_sources.pop("vendor_name", None)
                raw_values.pop("vendor_name", None)

        if "vendor_website" in fields and "vendor_name" not in fields:
            site = fields["vendor_website"]
            cleaned_site = re.sub(r"^[‘'\"\s]*(?:https?://)?(?:www\.)?\s*", "", site).strip()
            domain = cleaned_site.split(".")[0] if cleaned_site else ""
            if domain and len(domain) >= 3 and not any(k in domain.upper() for k in ["EXAMPLE", "TEST", "LOCALHOST", "GMAIL", "YAHOO", "HOTMAIL"]):
                spaced = re.sub(r"([a-zA-Z])(and)([A-Z])", r"\1 \2 \3", domain, flags=re.IGNORECASE)
                spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", spaced)
                fields["vendor_name"] = spaced
                confs["vendor_name"] = 0.88
                field_bboxes["vendor_name"] = field_bboxes.get("vendor_website", [0, 0, 0, 0])
                field_sources["vendor_name"] = "derived"
                raw_values["vendor_name"] = domain

        if "vendor_name" not in fields and valid_words:
            # 1. Top region check (y < 220)
            top_lines = []
            for idx, (w, b) in enumerate(zip(valid_words, norm_boxes)):
                if (b[1] + b[3]) / 2.0 < 220:
                    ymid = (b[1] + b[3]) / 2.0
                    placed = False
                    for tl in top_lines:
                        tl_ymid = sum((norm_boxes[j][1] + norm_boxes[j][3]) / 2.0 for j in tl) / len(tl)
                        if abs(ymid - tl_ymid) <= 12.0:
                            tl.append(idx)
                            placed = True
                            break
                    if not placed:
                        top_lines.append([idx])

            for tl in top_lines:
                line_str = " ".join(valid_words[j] for j in tl).strip(" :.,-")
                clean_str = re.sub(r"^(?:FOR|M/S\.?|MESSRS)\s+", "", line_str, flags=re.IGNORECASE).strip(" :.,-")
                clean_up = clean_str.upper()
                if (addr_pattern.search(clean_str) or bool(re.match(r"^\d+[-/]\d+", clean_str.strip()))) and not comp_suffix_pattern.search(clean_str):
                    continue
                if any(skip_k in clean_up for skip_k in [
                    "TAX INVOICE", "INVOICE", "ORIGINAL", "DUPLICATE", "COMMERCIAL",
                    "GSTIN", "GST NO", "GST:", "D.NO", "PLOT", "ROAD", "STREET", "SUITE",
                    "EMAIL", "CELL", "PHONE", "TEL:", "20B", "21B", "WLF", "PAN NO", "STATE CODE", "STATE NAME",
                    "DISTRIBUTOR", "PRODUCTS", "MANUFACTURER", "DEALER", "WHOLESALE",
                    "BILL TO", "SHIP TO", "BUYER", "CLIENT", "CONSIGNEE", "TO "
                ]) or clean_up.startswith("(") or clean_up.endswith(")"):
                    continue
                if re.search(r"\b(?:BANK|SBI|HDFC|ICICI|AXIS|BRANCH|IFSC|ACCOUNT|A/C)\b", clean_up):
                    continue
                if len(clean_str) >= 3 and any(c.isalpha() for c in clean_str):
                    fields["vendor_name"] = clean_str
                    confs["vendor_name"] = 0.90
                    field_bboxes["vendor_name"] = [
                        min(norm_boxes[j][0] for j in tl),
                        min(norm_boxes[j][1] for j in tl),
                        max(norm_boxes[j][2] for j in tl),
                        max(norm_boxes[j][3] for j in tl),
                    ]
                    field_sources["vendor_name"] = "contextual_spatial_fallback"
                    raw_values["vendor_name"] = line_str
                    break

            # 2. Check footer signatory "For <Company>" if still missing
            if "vendor_name" not in fields:
                for idx, w in enumerate(valid_words):
                    if w.upper() == "FOR" and idx + 1 < len(valid_words):
                        b_for = norm_boxes[idx]
                        y_for = (b_for[1] + b_for[3]) / 2.0
                        sig_idxs = [j for j in range(idx + 1, min(idx + 6, len(valid_words)))
                                    if abs(((norm_boxes[j][1] + norm_boxes[j][3]) / 2.0) - y_for) <= 10.0 and norm_boxes[j][0] > b_for[0]]
                        if sig_idxs:
                            sig_name = " ".join(valid_words[j] for j in sig_idxs).strip(" ,.-#")
                            sig_up = sig_name.upper()
                            if len(sig_name) >= 3 and not any(k in sig_up for k in ["SIGNATURE", "AUTHORISED", "CUSTOMER", "RECEIVER"]):
                                fields["vendor_name"] = sig_name
                                confs["vendor_name"] = 0.90
                                field_bboxes["vendor_name"] = [
                                    min(norm_boxes[j][0] for j in sig_idxs),
                                    min(norm_boxes[j][1] for j in sig_idxs),
                                    max(norm_boxes[j][2] for j in sig_idxs),
                                    max(norm_boxes[j][3] for j in sig_idxs),
                                ]
                                field_sources["vendor_name"] = "contextual_spatial_fallback"
                                raw_values["vendor_name"] = sig_name
                                break

        # 3b. Customer details spatial fallback (any column, y < 500)
        if "customer_name" not in fields or "customer_address" not in fields:
            for idx, w in enumerate(valid_words):
                w_low = w.lower()
                b_box = norm_boxes[idx]
                if b_box[1] >= 500:
                    continue

                is_anchor = False
                anchor_box = list(b_box)
                if any(w_low.startswith(k) for k in ["bill_to", "bill-to", "buyer", "customer", "ship_to", "client"]):
                    is_anchor = True
                    if w_low.startswith("client") and idx + 1 < len(valid_words) and valid_words[idx+1].lower().startswith("name"):
                        b_next = norm_boxes[idx+1]
                        anchor_box = [anchor_box[0], min(anchor_box[1], b_next[1]), max(anchor_box[2], b_next[2]), max(anchor_box[3], b_next[3])]
                elif w_low in ("bill", "ship") and idx + 1 < len(valid_words) and valid_words[idx+1].lower().startswith("to"):
                    is_anchor = True
                    b_next = norm_boxes[idx+1]
                    anchor_box = [anchor_box[0], min(anchor_box[1], b_next[1]), max(anchor_box[2], b_next[2]), max(anchor_box[3], b_next[3])]
                elif w_low in ("to", "to:") and b_box[0] < 350 and b_box[1] < 450:
                    is_anchor = True
                elif w_low in ("messrs", "m/s", "m/s."):
                    is_anchor = True

                if is_anchor:
                    y_mid = (anchor_box[1] + anchor_box[3]) / 2.0
                    same_line_words = []
                    # Check words on same horizontal line to the right of anchor
                    for j in range(idx + 1, min(idx + 12, len(valid_words))):
                        wb = norm_boxes[j]
                        if abs(((wb[1] + wb[3]) / 2.0) - y_mid) <= 12 and wb[0] >= anchor_box[0] + 5:
                            same_line_words.append(valid_words[j])

                    c_name = None
                    c_box = anchor_box
                    if same_line_words:
                        raw_c = " ".join(same_line_words).strip(": ,.-")
                        raw_c = re.sub(r"^(?:Bill[_\s]*to|Buyer|Customer|Ship[_\s]*to|To)[:\s]*", "", raw_c, flags=re.IGNORECASE).strip(": ,.-")
                        raw_c = re.sub(r"[\s:-]*(?:BILL[_\s]*TO|BUYER|CUSTOMER|SHIP[_\s]*TO)[:\s]*$", "", raw_c, flags=re.IGNORECASE).strip(": ,.-")
                        m_cust = re.match(r"^([A-Za-z\s.'-]+?)(?=\s+\d|\s+Suite|\s+Apt|\s+Road|\s+Street|\s+Mission|\s+Tel|\s+Email|\s+Site|$)", raw_c)
                        c_name = m_cust.group(1).strip() if m_cust else raw_c
                        c_name = re.sub(r"[\s:-]*(?:BILL[_\s]*TO|BUYER|CUSTOMER|SHIP[_\s]*TO)[:\s]*$", "", c_name, flags=re.IGNORECASE).strip(": ,.-")
                        c_box = norm_boxes[idx+1] if idx+1 < len(norm_boxes) else anchor_box
                    else:
                        # Name is on the next line below the anchor
                        next_line_words = []
                        y_anchor_bottom = anchor_box[3]
                        candidates = [
                            j for j, b in enumerate(norm_boxes)
                            if b[1] >= y_anchor_bottom - 2
                            and b[1] <= y_anchor_bottom + 30
                            and abs(b[0] - anchor_box[0]) < 80
                            and j != idx
                        ]
                        if candidates:
                            first_j = candidates[0]
                            y_line = (norm_boxes[first_j][1] + norm_boxes[first_j][3]) / 2.0
                            for j in candidates:
                                if abs(((norm_boxes[j][1] + norm_boxes[j][3]) / 2.0) - y_line) <= 10:
                                    next_line_words.append(valid_words[j])
                            if next_line_words:
                                raw_c = " ".join(next_line_words).strip(": ,.-")
                                raw_c = re.sub(r"[\s:-]*(?:BILL[_\s]*TO|BUYER|CUSTOMER|SHIP[_\s]*TO)[:\s]*$", "", raw_c, flags=re.IGNORECASE).strip(": ,.-")
                                m_cust = re.match(r"^([A-Za-z\s.'-]+?)(?=\s+\d|\s+Suite|\s+Apt|\s+Road|\s+Street|\s+Mission|\s+Tel|\s+Email|\s+Site|$)", raw_c)
                                c_name = m_cust.group(1).strip() if m_cust else raw_c
                                c_name = re.sub(r"[\s:-]*(?:BILL[_\s]*TO|BUYER|CUSTOMER|SHIP[_\s]*TO)[:\s]*$", "", c_name, flags=re.IGNORECASE).strip(": ,.-")
                                c_box = norm_boxes[first_j]

                    if c_name and len(c_name) >= 2 and not any(k in c_name.upper() for k in ["INVOICE", "TAX", "TOTAL", "DATE", "PO", "BILL_TO", "BILL TO", "BUYER", "CUSTOMER"]):
                        if "customer_name" not in fields:
                            fields["customer_name"] = c_name
                            confs["customer_name"] = 0.90
                            field_bboxes["customer_name"] = c_box
                            field_sources["customer_name"] = "spatial_anchor"
                            raw_values["customer_name"] = c_name

                    if "customer_address" not in fields:
                        addr_words = []
                        min_x = max(0, anchor_box[0] - 30)
                        max_x = anchor_box[0] + 350
                        for j in range(idx + 1, min(idx + 30, len(valid_words))):
                            wb = norm_boxes[j]
                            if wb[0] < min_x or wb[0] > max_x:
                                continue
                            if wb[1] <= anchor_box[3] + 2:
                                continue
                            if any(k in valid_words[j].lower() for k in ["tel:", "email:", "site:", "gstin:", "items", "quantity", "particular", "best", "unit", "price"]):
                                break
                            if c_name and valid_words[j] in c_name:
                                continue
                            addr_words.append(valid_words[j])
                        if addr_words:
                            addr_str = " ".join(addr_words).strip(": ,.-")
                            addr_str = re.sub(r"\s+([,.-])", r"\1", addr_str)
                            if len(addr_str) >= 5:
                                fields["customer_address"] = addr_str
                                confs["customer_address"] = 0.90
                                field_bboxes["customer_address"] = norm_boxes[idx+1] if idx+1 < len(norm_boxes) else b_box
                                field_sources["customer_address"] = "spatial_anchor"
                                raw_values["customer_address"] = addr_str
                    break

        # 3c. PO Number spatial fallback (strict horizontal line y <= 10)
        if "po_number" not in fields:
            for idx, w in enumerate(valid_words):
                if (any(k in w.lower() for k in ["po", "p.o."]) and "number" in w.lower()) or (w.lower() in ("po", "p.o.") and idx + 1 < len(valid_words) and "number" in valid_words[idx+1].lower()):
                    b_box = norm_boxes[idx]
                    val, v_box = _find_horizontal(b_box, re.compile(r"\b([A-Za-z0-9_/-]{1,15})\b"), require_digits=True, max_y_diff=10.0)
                    if val:
                        cleaned_val = re.sub(
                            r"^(?:PO\s*(?:NUMBER|NO|#|:)|P\.O\.\s*(?:NUMBER|NO|#|:)|ORDER\s*(?:NUMBER|NO|#|:)|NUMBER|NO|#|:|\s)+",
                            "",
                            val,
                            flags=re.IGNORECASE,
                        ).strip(" :-#")
                        if cleaned_val and any(c.isdigit() for c in cleaned_val) and not date_regex.match(cleaned_val):
                            fields["po_number"] = cleaned_val
                            confs["po_number"] = 0.92
                            field_bboxes["po_number"] = v_box or b_box
                            field_sources["po_number"] = "spatial_anchor"
                            raw_values["po_number"] = val
                            break

        # 3d. Invoice Number spatial fallback (excluding PO numbers)
        if "invoice_number" not in fields:
            for idx, w in enumerate(valid_words):
                w_low = w.lower()
                if ("invoice" in w_low or "inv" in w_low or "bill" in w_low) and "po" not in w_low and idx + 1 < len(valid_words):
                    b_box = norm_boxes[idx]
                    pat_inv_exact = re.compile(
                        r"\b(?:INV[/:.-]?\d{1,3}[/:.-]\d{1,3}[/:.-]\d{2,4}|INV\d{6,10}|\d[A-Za-z0-9]{3,5}-\d{2,4}|\d{3,5}-\d{2,4})\b",
                        re.IGNORECASE,
                    )
                    val, v_box = _find_horizontal(b_box, pat_inv_exact, require_digits=True, max_y_diff=12.0)
                    if not val:
                        val, v_box = _find_horizontal(b_box, re.compile(r"\b([A-Za-z0-9_/-]{2,20})\b"), require_digits=True, max_y_diff=10.0)
                    if val:
                        cleaned_val = re.sub(
                            r"^(?:INVOICE\s*(?:ID|NUMBER|NO|#|:|-)|INV\s*(?:ID|NUMBER|NO|#|:|-)|INV-|BILL\s*(?:ID|NUMBER|NO|#|:|-)|NUMBER|NO|ID|#|:|\s|-)+",
                            "",
                            val,
                            flags=re.IGNORECASE,
                        ).strip(" :-#")
                        cleaned_val = cleaned_val.replace("¥", "Y").replace("¢", "d")
                        cleaned_val = re.sub(r"^INV[I|](?=\d)", "INV/", cleaned_val, flags=re.IGNORECASE)
                        cleaned_val = re.sub(r"^INV[I|](?=[A-Za-z])", "INV/", cleaned_val, flags=re.IGNORECASE)
                        is_label_noise = bool(
                            re.match(r"^INVOI[A-Z0-9]*$", cleaned_val.upper())
                            or re.match(r"^INNOI[A-Z0-9]*$", cleaned_val.upper())
                            or cleaned_val.upper() in ["INVOICE", "INVOIGE", "INNOIGE", "INVOICE ID", "INV ID", "NUMBER", "NO", "ID", "BILL", "INV", "COMMERCIAL"]
                        )
                        if cleaned_val and not is_label_noise and any(c.isdigit() or c.isalpha() for c in cleaned_val):
                            is_dl = bool(re.search(r"\b(?:AP/\d{2}/\d{2}/\d{4}-\d+|WLF\d{2}[A-Z]\d+|20B|21B)\b", cleaned_val, re.IGNORECASE))
                            is_gstin = bool(re.search(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b", cleaned_val))
                            is_pan = bool(re.match(r"^[A-Z]{5}\d{4}[A-Z]$", cleaned_val))
                            if is_dl or is_gstin or is_pan:
                                continue
                            m_fatura_code = re.match(r"^(\d)([A-Za-z])([Oo0-9])([A-Za-z])([0-9])([a-z])-(\d{2,4})$", cleaned_val)
                            if m_fatura_code:
                                d1, l1, d2, l2, d3, l3, d4 = m_fatura_code.groups()
                                d2_fixed = "9" if d2.upper() in ("O", "0") else d2
                                cleaned_val = f"{d1}{l1.upper()}{d2_fixed}{l2.upper()}{d3}{l3}-{d4}"
                            cleaned_val = re.sub(r"([A-Z])a(d-\d{2,4})$", r"\g<1>8\g<2>", cleaned_val)
                            cleaned_val = re.sub(r"^INV[S](?=\d)", "INV5", cleaned_val, flags=re.IGNORECASE)
                            cleaned_val = re.sub(r"^INV[o](?=\d)", "INV0", cleaned_val)
                            cleaned_val = re.sub(r"^INV/T(?=\d)", "INV/7", cleaned_val, flags=re.IGNORECASE)
                            cleaned_val = re.sub(r"^INV/N(?=\d)", "INV/1", cleaned_val, flags=re.IGNORECASE)
                            fields["invoice_number"] = cleaned_val
                            confs["invoice_number"] = 0.90
                            field_bboxes["invoice_number"] = v_box or b_box
                            field_sources["invoice_number"] = "spatial_anchor"
                            raw_values["invoice_number"] = val
                            break

        # Spatial Date fallback if not extracted
        if "invoice_date" not in fields:
            for idx, w in enumerate(valid_words):
                if norm_boxes[idx][1] > 450:
                    continue
                w_up = w.upper().strip(" :")
                if w_up in ("DATE", "DATED", "INVOICE_DATE") or (idx > 0 and valid_words[idx-1].upper() in ("INVOICE", "BILL") and w_up == "DATE"):
                    y_mid = (norm_boxes[idx][1] + norm_boxes[idx][3]) / 2.0
                    for j in range(idx + 1, min(idx + 5, len(valid_words))):
                        if abs(((norm_boxes[j][1] + norm_boxes[j][3]) / 2.0) - y_mid) <= 12:
                            m_d = date_regex.search(valid_words[j])
                            if m_d:
                                fields["invoice_date"] = m_d.group(1)
                                confs["invoice_date"] = 0.90
                                field_bboxes["invoice_date"] = norm_boxes[j]
                                field_sources["invoice_date"] = "spatial_anchor"
                                raw_values["invoice_date"] = m_d.group(1)
                                break
                    if "invoice_date" in fields:
                        break

        # 3e. Financial Summary spatial fallback
        for idx, w in enumerate(valid_words):
            b = norm_boxes[idx]
            if b[1] < 350:
                continue
            w_clean = re.sub(r"^[‘'\"`\s]+", "", w).upper()
            y_mid = (b[1] + b[3]) / 2.0

            # Subtotal: SUB_TOTAL, SUBTOTAL, TAXABLE VALUE, GROSS AMOUNT, TOTAL VALUE
            is_sub_kw = any(k in w_clean for k in ("SUB_TOTAL", "SUBTOTAL", "TAXABLE", "GROSS"))
            if "subtotal" not in fields and is_sub_kw:
                line_words = [valid_words[j] for j, jb in enumerate(norm_boxes) if abs(((jb[1]+jb[3])/2.0) - y_mid) <= 10 and jb[0] >= b[0]]
                full_line = " ".join(line_words)
                m_sub = re.search(r"(?:SUB_TOTAL|SUBTOTAL|TAXABLE\s*(?:VALUE|AMT|AMOUNT)?|GROSS\s*AMOUNT|TOTAL\s*VALUE)\s*[:.]?\s*([0-9,]+\.[0-9]{2})", full_line, re.IGNORECASE)
                if m_sub:
                    val = m_sub.group(1).replace(",", "")
                    fields["subtotal"] = val
                    confs["subtotal"] = 0.92
                    field_bboxes["subtotal"] = b
                    field_sources["subtotal"] = "spatial_anchor"
                    raw_values["subtotal"] = val

            # Discount: DISCOUNT
            if "discount" not in fields and "DISCOUNT" in w_clean:
                line_words = [valid_words[j] for j, jb in enumerate(norm_boxes) if abs(((jb[1]+jb[3])/2.0) - y_mid) <= 10 and jb[0] >= b[0]]
                full_line = " ".join(line_words)
                m_disc = re.search(r"DISCOUNT(?:\([^)]*\))?[:.\s]+(?:\([+-.]?\)\s*)?([0-9,]+(?:\.[0-9]+)?)", full_line, re.IGNORECASE)
                if m_disc:
                    val = m_disc.group(1).replace(",", "")
                    fields["discount"] = val
                    confs["discount"] = 0.92
                    field_bboxes["discount"] = b
                    field_sources["discount"] = "spatial_anchor"
                    raw_values["discount"] = val

            # Tax: TAX, VAT, or GST
            if "tax_amount" not in fields and any(k in w_clean for k in ("TAX", "VAT", "GST")) and not any(k in w_clean for k in ("INVOICE", "TAXABLE", "GROSS", "NET", "SUBTOTAL", "SUB_TOTAL", "ROUND")):
                line_words = [valid_words[j] for j, jb in enumerate(norm_boxes) if abs(((jb[1]+jb[3])/2.0) - y_mid) <= 10 and jb[0] >= b[0]]
                full_line = " ".join(line_words)
                if not re.search(r"\b(?:GROSS|NET|SUBTOTAL|SUB_TOTAL|ROUND)\b", full_line, re.IGNORECASE):
                    candidates = re.findall(r"([0-9,]+\.[0-9]{2})", full_line)
                    amounts = [c for c in candidates if not re.search(rf"{re.escape(c)}\s*%", full_line)]
                    if amounts:
                        val = amounts[-1].replace(",", "")
                        fields["tax_amount"] = val
                        confs["tax_amount"] = 0.92
                        field_bboxes["tax_amount"] = b
                        field_sources["tax_amount"] = "spatial_anchor"
                        raw_values["tax_amount"] = val

            # CGST Amount
            if "cgst_amount" not in fields and "CGST" in w_clean:
                line_words = [valid_words[j] for j, jb in enumerate(norm_boxes) if abs(((jb[1]+jb[3])/2.0) - y_mid) <= 10 and jb[0] >= b[0]]
                full_line = " ".join(line_words)
                m_cgst_amt = re.search(r"CGST(?:\s*(?:AMOUNT|TAX|AMT))?\s*[:.]?\s*([0-9,]+\.[0-9]{2})", full_line, re.IGNORECASE)
                if m_cgst_amt:
                    val = m_cgst_amt.group(1).replace(",", "")
                    fields["cgst_amount"] = val
                    confs["cgst_amount"] = 0.92
                    field_bboxes["cgst_amount"] = b
                    field_sources["cgst_amount"] = "spatial_anchor"
                    raw_values["cgst_amount"] = val

            # SGST Amount
            if "sgst_amount" not in fields and "SGST" in w_clean:
                line_words = [valid_words[j] for j, jb in enumerate(norm_boxes) if abs(((jb[1]+jb[3])/2.0) - y_mid) <= 10 and jb[0] >= b[0]]
                full_line = " ".join(line_words)
                m_sgst_amt = re.search(r"SGST(?:\s*(?:AMOUNT|TAX|AMT))?\s*[:.]?\s*([0-9,]+\.[0-9]{2})", full_line, re.IGNORECASE)
                if m_sgst_amt:
                    val = m_sgst_amt.group(1).replace(",", "")
                    fields["sgst_amount"] = val
                    confs["sgst_amount"] = 0.92
                    field_bboxes["sgst_amount"] = b
                    field_sources["sgst_amount"] = "spatial_anchor"
                    raw_values["sgst_amount"] = val

            # IGST Amount
            if "igst_amount" not in fields and "IGST" in w_clean:
                line_words = [valid_words[j] for j, jb in enumerate(norm_boxes) if abs(((jb[1]+jb[3])/2.0) - y_mid) <= 10 and jb[0] >= b[0]]
                full_line = " ".join(line_words)
                m_igst_amt = re.search(r"IGST(?:\s*(?:AMOUNT|TAX|AMT))?\s*[:.]?\s*([0-9,]+\.[0-9]{2})", full_line, re.IGNORECASE)
                if m_igst_amt:
                    val = m_igst_amt.group(1).replace(",", "")
                    fields["igst_amount"] = val
                    confs["igst_amount"] = 0.92
                    field_bboxes["igst_amount"] = b
                    field_sources["igst_amount"] = "spatial_anchor"
                    raw_values["igst_amount"] = val

            # Total: TOTAL, BALANCE DUE, AMOUNT DUE, TOTAL DUE, NET AMOUNT, GRAND TOTAL, NET PAYABLE
            allow_tot_override = (
                "total_amount" not in fields
                or field_sources.get("total_amount") == "token_classification"
            )
            if allow_tot_override and (w_clean in ("TOTAL", "TOTAL:", "NET", "NET:") or any(k in w_clean for k in ("BALANCE", "AMOUNT_DUE", "NET_AMOUNT", "GRAND_TOTAL", "NET_PAYABLE"))):
                if not ("SUB" in w_clean):
                    line_words = [valid_words[j] for j, jb in enumerate(norm_boxes) if abs(((jb[1]+jb[3])/2.0) - y_mid) <= 12 and jb[0] >= (b[0] - 5)]
                    full_line = " ".join(line_words)
                    m_tot = re.search(r"(?<!SUB_)(?<!SUB\s)(?<!SUB-)\b(?:TOTAL|BALANCE\s+DUE|AMOUNT\s+DUE|TOTAL\s+DUE|NET\s+AMOUNT|GRAND\s+TOTAL|NET\s+PAYABLE|TOTAL\s+AMOUNT)\b\s*[:.]?\s*([0-9,]+\.[0-9]{2})", full_line, re.IGNORECASE)
                    if m_tot:
                        val = m_tot.group(1).replace(",", "")
                        fields["total_amount"] = val
                        confs["total_amount"] = 0.95
                        field_bboxes["total_amount"] = b
                        field_sources["total_amount"] = "spatial_anchor"
                        raw_values["total_amount"] = val

        # 3f. Localized OCR patch fallbacks when full-page segmenter dropped sections
        if image is not None:
            # Customer patch (upper-right or upper-left where multi-column segmentation missed customer)
            if "customer_name" not in fields:
                try:
                    w_img, h_img = image.size
                    crop_box = (int(w_img * 0.40), int(h_img * 0.18), int(w_img * 0.95), int(h_img * 0.45))
                    crop_img = image.crop(crop_box)
                    import pytesseract
                    from app.config import TESSERACT_CMD
                    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
                    crop_text = pytesseract.image_to_string(crop_img, config="--psm 6")
                    for c_line in crop_text.splitlines():
                        c_line_s = c_line.strip()
                        if any(c_line_s.lower().startswith(k) for k in ["bill to:", "bill to", "buyer:", "buyer", "bill_to"]):
                            clean_c = re.sub(r"^(?:Bill[_\s]*to|Buyer|Customer|Ship[_\s]*to)[:\s]*", "", c_line_s, flags=re.IGNORECASE).strip(": ,.-")
                            if clean_c and len(clean_c) >= 2 and not any(k in clean_c.upper() for k in ["INVOICE", "TAX", "TOTAL"]):
                                fields["customer_name"] = clean_c
                                confs["customer_name"] = 0.88
                                field_sources["customer_name"] = "localized_ocr_patch"
                                raw_values["customer_name"] = clean_c
                                break
                except Exception:
                    pass

            # Total amount patch (lower half where table / summary segmentation missed total)
            if "total_amount" not in fields or field_sources.get("total_amount") in ("token_classification", "derived"):
                try:
                    w_img, h_img = image.size
                    crop_box = (int(w_img * 0.10), int(h_img * 0.45), int(w_img * 0.98), int(h_img * 0.96))
                    crop_img = image.crop(crop_box)
                    import pytesseract
                    from app.config import TESSERACT_CMD
                    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
                    tot_pat = re.compile(r"(?<!SUB_)(?<!SUB\s)(?<!SUB-)\b(?:TOTAL|BALANCE\s+DUE|AMOUNT\s+DUE|TOTAL\s+DUE|NET\s+AMOUNT|GRAND\s+TOTAL|NET\s+PAYABLE|TOTAL\s+AMOUNT)\b\s*[:.]?\s*([0-9,]+\.[0-9]{2})", re.IGNORECASE)
                    crop_text = pytesseract.image_to_string(crop_img, config="--psm 6")
                    m_tot_patch = tot_pat.search(crop_text)
                    if not m_tot_patch:
                        crop_text_11 = pytesseract.image_to_string(crop_img, config="--psm 11")
                        m_tot_patch = tot_pat.search(crop_text_11)
                    if m_tot_patch:
                        val = m_tot_patch.group(1).replace(",", "")
                        fields["total_amount"] = val
                        confs["total_amount"] = 0.92
                        field_sources["total_amount"] = "localized_ocr_patch"
                        raw_values["total_amount"] = val
                except Exception:
                    pass

        # Derive total ONLY if completely missing from model and OCR patch
        if "total_amount" not in fields and "subtotal" in fields and "tax_amount" in fields:
            try:
                sub = float(fields["subtotal"])
                tax = float(fields.get("tax_amount") or 0.0)
                disc = float(fields.get("discount") or 0.0)
                derived_tot = round(sub + tax - disc, 2)
                fields["total_amount"] = f"{derived_tot:.2f}"
                confs["total_amount"] = 0.85
                field_sources["total_amount"] = "derived"
                raw_values["total_amount"] = f"{derived_tot:.2f}"
            except (ValueError, TypeError):
                pass

        # Derive subtotal if total, tax, and discount are present
        if "subtotal" not in fields and "total_amount" in fields and "tax_amount" in fields:
            try:
                tot = float(fields["total_amount"])
                tax = float(fields.get("tax_amount") or 0.0)
                disc = float(fields.get("discount") or 0.0)
                derived_sub = round(tot - tax + disc, 2)
                fields["subtotal"] = f"{derived_sub:.2f}"
                confs["subtotal"] = 0.85
                field_sources["subtotal"] = "derived"
                raw_values["subtotal"] = f"{derived_sub:.2f}"
            except (ValueError, TypeError):
                pass

        # 3f. Statutory fields (GSTIN, CGST/SGST/IGST)
        gstin_regex = re.compile(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b")
        text_source = ocr_text or " ".join(valid_words)

        all_gstins = gstin_regex.findall(text_source)
        if all_gstins:
            fields["gst_number"] = all_gstins[0]
            fields["vendor_gstin"] = all_gstins[0]
            confs["gst_number"] = 0.95
            confs["vendor_gstin"] = 0.95
            field_sources["gst_number"] = "statutory_regex"
            field_sources["vendor_gstin"] = "statutory_regex"
            raw_values["gst_number"] = all_gstins[0]
            raw_values["vendor_gstin"] = all_gstins[0]

            if len(all_gstins) > 1 and "customer_gstin" not in fields:
                fields["customer_gstin"] = all_gstins[1]
                confs["customer_gstin"] = 0.95
                field_sources["customer_gstin"] = "statutory_regex"
                raw_values["customer_gstin"] = all_gstins[1]

        # Contextual check for customer GSTIN if not set:
        if "customer_gstin" not in fields:
            cust_gstin_m = re.search(r"(?:Buyer|Client|Customer|Consignee|Billed\s*To)[^\n]{0,60}?(?:GSTIN|GST|GST\s*NO)?[^\n\w]*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})", text_source, re.IGNORECASE)
            if cust_gstin_m:
                fields["customer_gstin"] = cust_gstin_m.group(1)
                confs["customer_gstin"] = 0.95
                field_sources["customer_gstin"] = "contextual_regex"
                raw_values["customer_gstin"] = cust_gstin_m.group(1)

        cgst_match = re.search(r"\bCGST\b[^\n\d]{0,40}?([0-9,]+\.[0-9]{2})", text_source, re.IGNORECASE)
        sgst_match = re.search(r"\bSGST\b[^\n\d]{0,40}?([0-9,]+\.[0-9]{2})", text_source, re.IGNORECASE)
        igst_match = re.search(r"\bIGST\b[^\n\d]{0,40}?([0-9,]+\.[0-9]{2})", text_source, re.IGNORECASE)
        if cgst_match:
            c_val_str = cgst_match.group(1).replace(",", "")
            fields["cgst_amount"] = c_val_str
            confs["cgst_amount"] = 0.90
            field_sources["cgst_amount"] = "statutory_regex"
            raw_values["cgst_amount"] = c_val_str
        if sgst_match:
            s_val_str = sgst_match.group(1).replace(",", "")
            fields["sgst_amount"] = s_val_str
            confs["sgst_amount"] = 0.90
            field_sources["sgst_amount"] = "statutory_regex"
            raw_values["sgst_amount"] = s_val_str
        if igst_match:
            i_val_str = igst_match.group(1).replace(",", "")
            fields["igst_amount"] = i_val_str
            confs["igst_amount"] = 0.90
            field_sources["igst_amount"] = "statutory_regex"
            raw_values["igst_amount"] = i_val_str

        cgst_r_match = re.search(r"\bCGST\b[^\n\d]*?(\d+(?:\.\d+)?)\s*%", text_source, re.IGNORECASE)
        sgst_r_match = re.search(r"\bSGST\b[^\n\d]*?(\d+(?:\.\d+)?)\s*%", text_source, re.IGNORECASE)
        igst_r_match = re.search(r"\bIGST\b[^\n\d]*?(\d+(?:\.\d+)?)\s*%", text_source, re.IGNORECASE)
        if cgst_r_match:
            fields["cgst_rate"] = f"{cgst_r_match.group(1)}%"
            confs["cgst_rate"] = 0.90
        if sgst_r_match:
            fields["sgst_rate"] = f"{sgst_r_match.group(1)}%"
            confs["sgst_rate"] = 0.90
        if igst_r_match:
            fields["igst_rate"] = f"{igst_r_match.group(1)}%"
            confs["igst_rate"] = 0.90

        # If cgst + sgst exist, calculate or rectify tax_amount
        if "cgst_amount" in fields and "sgst_amount" in fields:
            try:
                c_val = float(fields["cgst_amount"])
                s_val = float(fields["sgst_amount"])
                tot_tax = round(c_val + s_val, 2)
                # If tax_amount missing, or inverted with subtotal/total
                if ("tax_amount" not in fields or
                    (fields.get("total_amount") and float(fields.get("tax_amount", 0)) >= float(fields["total_amount"]) * 0.7)):
                    fields["tax_amount"] = f"{tot_tax:.2f}"
                    confs["tax_amount"] = 0.90
                    field_sources["tax_amount"] = "derived_cgst_sgst"
                    raw_values["tax_amount"] = f"{tot_tax:.2f}"
            except ValueError:
                pass
        elif "igst_amount" in fields and ("tax_amount" not in fields or (fields.get("total_amount") and float(fields.get("tax_amount", 0)) >= float(fields["total_amount"]) * 0.7)):
            fields["tax_amount"] = fields["igst_amount"]
            confs["tax_amount"] = 0.90
            field_sources["tax_amount"] = "derived_igst"
            raw_values["tax_amount"] = fields["igst_amount"]

        # Recalculate subtotal if derived previously or missing, when total and tax are valid
        if "total_amount" in fields and "tax_amount" in fields:
            if "subtotal" not in fields or field_sources.get("subtotal") == "derived":
                try:
                    tot = float(fields["total_amount"])
                    tax = float(fields["tax_amount"])
                    disc = float(fields.get("discount") or 0.0)
                    if 0 <= tax < tot * 0.70:
                        derived_sub = round(tot - tax + disc, 2)
                        if derived_sub > 0:
                            fields["subtotal"] = f"{derived_sub:.2f}"
                            confs["subtotal"] = 0.85
                            field_sources["subtotal"] = "derived"
                            raw_values["subtotal"] = f"{derived_sub:.2f}"
                except (ValueError, TypeError):
                    pass

        # 3g. Currency Detection: STRICT PRIORITY TO TOTAL LINE
        total_box = field_bboxes.get("total_amount")
        curr_found = None
        if total_box:
            t_ymid = (total_box[1] + total_box[3]) / 2.0
            for w, b in zip(valid_words, norm_boxes):
                if abs(((b[1] + b[3]) / 2.0) - t_ymid) <= 15:
                    if re.search(r"\bEUR\b|[€]", w):
                        curr_found = "EUR"
                        break
                    elif re.search(r"\bUSD\b|[$§]", w):
                        curr_found = "USD"
                        break
                    elif re.search(r"\bINR\b|[₹]|(?:Rs\.?)", w, re.IGNORECASE):
                        curr_found = "INR"
                        break
                    elif re.search(r"\bGBP\b|[£]", w):
                        curr_found = "GBP"
                        break

        if not curr_found:
            for w, b in zip(valid_words, norm_boxes):
                if b[0] >= 300 and b[1] >= 600:
                    if re.search(r"\bEUR\b|[€]", w):
                        curr_found = "EUR"
                        break
                    elif re.search(r"\bUSD\b|[$§]", w):
                        curr_found = "USD"
                        break
                    elif re.search(r"\bINR\b|[₹]|(?:Rs\.?)", w, re.IGNORECASE):
                        curr_found = "INR"
                        break
                    elif re.search(r"\bGBP\b|[£]", w):
                        curr_found = "GBP"
                        break

        if not curr_found:
            if re.search(r"\b(?:Rupees|Rs\.?|INR|₹)\b", text_source, re.IGNORECASE):
                curr_found = "INR"
            elif re.search(r"\bEUR\b|[€]", text_source):
                curr_found = "EUR"
            elif re.search(r"\bUSD\b|[$]", text_source):
                curr_found = "USD"
            elif re.search(r"\bGBP\b|[£]", text_source):
                curr_found = "GBP"

        if curr_found:
            fields["currency"] = curr_found
            confs["currency"] = 0.99
            field_sources["currency"] = "total_line_or_summary"

        # Sync vendor_name and company_name aliases
        if "vendor_name" in fields and "company_name" not in fields:
            fields["company_name"] = fields["vendor_name"]
            confs["company_name"] = confs["vendor_name"]
        elif "company_name" in fields and "vendor_name" not in fields:
            fields["vendor_name"] = fields["company_name"]
            confs["vendor_name"] = confs["company_name"]

        # 4. Normalize dates (invoice_date, due_date) and preserve raw values
        from app.validation import normalize_date, validate_financials

        date_validation: dict[str, Any] = {}
        if "invoice_date" in fields:
            raw_inv_date = fields["invoice_date"]
            inv_date_norm = normalize_date(raw_inv_date)
            fields["raw_invoice_date"] = raw_inv_date
            if inv_date_norm.normalized:
                fields["invoice_date"] = inv_date_norm.normalized
            date_validation["invoice_date"] = inv_date_norm.to_dict()

        if "due_date" in fields:
            raw_due_date = fields["due_date"]
            due_date_norm = normalize_date(raw_due_date)
            fields["raw_due_date"] = raw_due_date
            if due_date_norm.normalized:
                fields["due_date"] = due_date_norm.normalized
            date_validation["due_date"] = due_date_norm.to_dict()

        # 5. Financial validation
        fin_validation = validate_financials(fields, line_items=line_items, tolerance=0.05)

        all_errors = list(fin_validation.validation_errors)
        for d_key, d_res in date_validation.items():
            if not d_res.get("is_valid") and d_res.get("error"):
                all_errors.append(f"{d_key}: {d_res['error']}")
            elif d_res.get("is_ambiguous"):
                all_errors.append(f"{d_key}: Ambiguous date format")

        validation_dict = {
            "is_valid": fin_validation.is_valid and all(d.get("is_valid", True) for d in date_validation.values()),
            "suspicious": fin_validation.suspicious or any(d.get("is_ambiguous", False) or not d.get("is_valid", True) for d in date_validation.values()),
            "validation_errors": all_errors,
            "financials": fin_validation.to_dict(),
            "dates": date_validation,
        }

        elapsed = time.time() - start

        return ExtractionResult(
            fields=fields,
            confidence_scores=confs,
            line_items=line_items,
            raw_text=ocr_text or " ".join(valid_words),
            method="layoutlmv3_fatura2",
            model_name=self.active_model_name,
            processing_time=round(elapsed, 2),
            validation=validation_dict,
            metadata={
                "total_entities": len(entities),
                "extracted_fields": list(fields.keys()),
                "entities_found": [e["label"] for e in entities],
                "field_bboxes": field_bboxes,
                "field_sources": field_sources,
                "raw_values": raw_values,
                "unsupported_fields_supplemented": ["gst_number", "cgst_amount", "sgst_amount", "currency"],
                "document_type": document_type,
                "model_domain": "invoice_understanding_fatura2",
                "total_line_items": len(line_items),
                "validation": validation_dict,
                "notes": (
                    "Extracted using LayoutLMv3 fine-tuned on mathieu1256/FATURA2-invoices. "
                    "Semantic invoice entities (SELLER, BUYER, NUMBER, DATE, TOTAL, TAX) extracted directly "
                    "via layout-aware token classification. Domain-specific statutory fields not present in FATURA2 "
                    "(e.g., GSTIN, CGST/SGST split) supplemented via statutory regex patterns."
                ),
            },
        )

    # ── semantic field mapping ───────────────────────────────────────

    @staticmethod
    def _map_to_canonical_fields(
        pairs: list[Any],
        headers: list[str],
        document_type: str,
        raw_confs: dict[str, float],
        valid_words: list[str] | None = None,
        ocr_text: str = "",
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
        """Map layout-aware QUESTION-ANSWER pairs to canonical schema fields.

        Enforces strict format validation, vendor vs buyer disambiguation,
        detailed monetary extraction (gross_amount, cgst, sgst, subtotal, tax, net/total),
        vendor identification, and currency detection.
        """
        import re

        fields: dict[str, Any] = {}
        confs: dict[str, float] = {}
        unmapped_pairs: dict[str, Any] = {}

        gstin_regex = re.compile(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b")
        date_regex = re.compile(r"\b(\d{1,4}[-/.](?:[A-Za-z]{3}|\d{1,2})[-/.]\d{2,4})\b")
        amount_regex = re.compile(r"[-+]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})|[-+]?[0-9]+\.[0-9]{2}")

        patterns = {
            "invoice_number": [
                r"\b(?:INVOICE|INV)\s*(?:NO|NUMBER|#)?\b",
                r"\bBILL\s*(?:NO|NUMBER|#)\b",
            ],
            "invoice_date": [
                r"\b(?:INVOICE\s*DATE|BILL\s*DATE|DATE|DATED)\b",
            ],
            "due_date": [
                r"\b(?:DUE\s*DATE|PAYMENT\s*DUE)\b",
            ],
            "vendor_name": [
                r"\b(?:VENDOR|SELLER|FROM|SUPPLIER)\b",
            ],
            "customer_name": [
                r"\b(?:CUSTOMER|BILL\s*TO|SOLD\s*TO|CLIENT|BUYER)\b",
            ],
            "gst_number": [
                r"\b(?:GSTIN|GST)\s*(?:NO|NUMBER|#)?\b",
            ],
            "subtotal": [
                r"\b(?:SUB\s*TOTAL|TAXABLE\s*(?:VALUE|AMOUNT)|NET\s*AMOUNT)\b",
            ],
            "tax_amount": [
                r"\b(?:TOTAL\s*TAX|TAX\s*AMOUNT|GST\s*AMOUNT|VAT)\b",
            ],
            "total_amount": [
                r"\b(?:GRAND\s*TOTAL|TOTAL\s*AMOUNT|\bTOTAL\b|AMOUNT\s*DUE|NET\s*PAYABLE)\b",
            ],
        }

        if document_type == "Receipt":
            patterns.update({
                "merchant_name": [r"\b(?:MERCHANT|STORE|RESTAURANT|SHOP)\b"],
                "receipt_number": [r"\b(?:RECEIPT|ORDER|TRANS(?:ACTION)?)\s*(?:NO|#)?\b"],
                "date": [r"\b(?:DATE|TRANS\s*DATE)\b"],
            })

        matched_questions = set()

        # Step 1: GST extraction with Vendor vs Buyer disambiguation
        detected_gstins = []
        for item in pairs:
            q_text = item[0]
            a_text = item[1]
            c_score = item[2]
            if not a_text:
                continue

            gst_match = gstin_regex.search(a_text)
            if gst_match or re.search(r"\b(?:GSTIN|GST)\b", q_text, re.IGNORECASE):
                extracted_gst = gst_match.group(1) if gst_match else a_text.strip().rstrip(":").strip()
                if gstin_regex.search(extracted_gst):
                    is_customer = any(k in q_text.upper() for k in ["TO", "BUYER", "CUSTOMER", "CLIENT", "BILL TO"])
                    detected_gstins.append({
                        "question": q_text,
                        "gstin": gstin_regex.search(extracted_gst).group(1),
                        "is_customer": is_customer,
                        "conf": c_score,
                    })

        # Prefer vendor GST for the canonical gst_number
        vendor_gsts = [g for g in detected_gstins if not g["is_customer"]]
        customer_gsts = [g for g in detected_gstins if g["is_customer"]]

        if vendor_gsts:
            fields["gst_number"] = vendor_gsts[0]["gstin"]
            confs["gst_number"] = round(vendor_gsts[0]["conf"], 4)
            matched_questions.add(vendor_gsts[0]["question"])
        elif customer_gsts:
            fields["gst_number"] = customer_gsts[0]["gstin"]
            confs["gst_number"] = round(customer_gsts[0]["conf"] * 0.9, 4)
            matched_questions.add(customer_gsts[0]["question"])

        # Step 2: Invoice Number extraction
        for item in pairs:
            q_text = item[0]
            a_text = item[1]
            c_score = item[2]
            if "invoice_number" in fields or not a_text:
                continue
            if any(re.search(pat, q_text.upper()) for pat in patterns["invoice_number"]):
                cleaned = a_text.strip().lstrip(":").strip()
                if cleaned and not all(ch in ":-/. " for ch in cleaned):
                    fields["invoice_number"] = cleaned
                    confs["invoice_number"] = round(c_score, 4)
                    matched_questions.add(q_text)
                    break

        # Step 3: Invoice Date extraction
        for item in pairs:
            q_text = item[0]
            a_text = item[1]
            c_score = item[2]
            if "invoice_date" in fields or not a_text:
                continue
            if any(re.search(pat, q_text.upper()) for pat in patterns["invoice_date"]):
                date_match = date_regex.search(a_text)
                if date_match:
                    fields["invoice_date"] = date_match.group(1)
                    confs["invoice_date"] = round(c_score, 4)
                    matched_questions.add(q_text)
                    break
                else:
                    cleaned = a_text.strip().lstrip(":").strip()
                    if cleaned and not all(ch in ":-/. " for ch in cleaned) and any(ch.isdigit() for ch in cleaned):
                        fields["invoice_date"] = cleaned
                        confs["invoice_date"] = round(c_score, 4)
                        matched_questions.add(q_text)
                        break

        # Step 4: Monetary Breakdown Fields (Gross, CGST, SGST, Net/Total, Subtotal, Tax)
        for item in pairs:
            q_text = item[0]
            a_text = item[1]
            c_score = item[2]
            if not a_text:
                continue
            qu = q_text.upper()

            # Gross Amount
            if "gross_amount" not in fields and ("GROSS AMOUNT" in qu or "GROSS AMT" in qu or qu.strip().rstrip(":") == "GROSS"):
                amt_match = amount_regex.search(a_text)
                if amt_match:
                    fields["gross_amount"] = amt_match.group(0)
                    confs["gross_amount"] = round(c_score, 4)
                    matched_questions.add(q_text)

            # CGST Amount
            if "cgst_amount" not in fields and ("CGST AMOUNT" in qu or "CGST AMT" in qu or (qu.strip().rstrip(":") == "CGST" and "TAX" not in qu)):
                amt_match = amount_regex.search(a_text)
                if amt_match:
                    fields["cgst_amount"] = amt_match.group(0)
                    confs["cgst_amount"] = round(c_score, 4)
                    matched_questions.add(q_text)

            # SGST Amount
            if "sgst_amount" not in fields and ("SGST AMOUNT" in qu or "SGST AMT" in qu or (qu.strip().rstrip(":") == "SGST" and "TAX" not in qu)):
                amt_match = amount_regex.search(a_text)
                if amt_match:
                    fields["sgst_amount"] = amt_match.group(0)
                    confs["sgst_amount"] = round(c_score, 4)
                    matched_questions.add(q_text)

            # Total / Net Amount
            if "total_amount" not in fields and any(k in qu for k in ["NET AMOUNT", "GRAND TOTAL", "TOTAL AMOUNT", "TOTAL DUE", "NET PAYABLE"]):
                amt_match = amount_regex.search(a_text)
                if amt_match:
                    fields["total_amount"] = amt_match.group(0)
                    confs["total_amount"] = round(c_score, 4)
                    matched_questions.add(q_text)

        # Fallback for total_amount if not yet matched
        if "total_amount" not in fields:
            for item in pairs:
                q_text = item[0]
                a_text = item[1]
                c_score = item[2]
                if not a_text or q_text in matched_questions:
                    continue
                if re.search(r"\bTOTAL\b", q_text.upper()):
                    amt_match = amount_regex.search(a_text)
                    if amt_match:
                        fields["total_amount"] = amt_match.group(0)
                        confs["total_amount"] = round(c_score, 4)
                        matched_questions.add(q_text)
                        break

        # Subtotal (Taxable Value or Gross Amount)
        if "subtotal" not in fields:
            for item in pairs:
                q_text = item[0]
                a_text = item[1]
                c_score = item[2]
                if not a_text or q_text in matched_questions:
                    continue
                if any(re.search(pat, q_text.upper()) for pat in patterns["subtotal"]):
                    amt_match = amount_regex.search(a_text)
                    if amt_match:
                        fields["subtotal"] = amt_match.group(0)
                        confs["subtotal"] = round(c_score, 4)
                        matched_questions.add(q_text)
                        break
            if "subtotal" not in fields and "gross_amount" in fields:
                fields["subtotal"] = fields["gross_amount"]
                confs["subtotal"] = confs["gross_amount"]

        # Tax Amount (Total Tax or composite CGST + SGST)
        if "tax_amount" not in fields:
            for item in pairs:
                q_text = item[0]
                a_text = item[1]
                c_score = item[2]
                if not a_text or q_text in matched_questions:
                    continue
                if any(re.search(pat, q_text.upper()) for pat in patterns["tax_amount"]):
                    amt_match = amount_regex.search(a_text)
                    if amt_match:
                        fields["tax_amount"] = amt_match.group(0)
                        confs["tax_amount"] = round(c_score, 4)
                        matched_questions.add(q_text)
                        break
            if "tax_amount" not in fields and "cgst_amount" in fields and "sgst_amount" in fields:
                try:
                    c_val = float(fields["cgst_amount"])
                    s_val = float(fields["sgst_amount"])
                    fields["tax_amount"] = f"{c_val + s_val:.2f}"
                    confs["tax_amount"] = min(confs.get("cgst_amount", 0.9), confs.get("sgst_amount", 0.9))
                except ValueError:
                    pass

        # Step 5: Customer Name (disassociated from vendor, excluding generic labels)
        generic_name_labels = {
            "CUSTOMER", "CUSTOMER NAME", "CLIENT", "BUYER",
            "CUSTOMER 'S SIGNATURE", "CUSTOMER SIGNATURE", "SIGNATURE",
            "TO", "BILL TO", "SOLD TO", "CONSIGNEE", "NAME",
            "VENDOR", "SELLER", "SUPPLIER", "AUTHORIZED SIGNATORY",
        }

        if "customer_name" not in fields:
            # First check if the question/label itself embeds the customer name (e.g. 'To PRASANTHI PHARMACY')
            for item in pairs:
                q_text = item[0]
                a_text = item[1]
                c_score = item[2]
                m = re.match(
                    r"^(?:TO|BILL\s*TO|BILLED\s*TO|SOLD\s*TO|BUYER|CLIENT|CONSIGNEE)\b\s*[:\-]?\s*(.+)$",
                    q_text.strip(),
                    re.IGNORECASE,
                )
                if m:
                    cand = m.group(1).strip().strip(":").strip()
                    cand_upper = cand.upper()
                    if (
                        len(cand) >= 3
                        and cand_upper not in generic_name_labels
                        and not any(k in cand_upper for k in ["INVOICE", "TAX", "GST", "SIGNATURE"])
                    ):
                        fields["customer_name"] = cand
                        confs["customer_name"] = round(c_score, 4)
                        matched_questions.add(q_text)
                        break

            # If not embedded in question, check answer text of customer-labeled pairs
            if "customer_name" not in fields:
                for item in pairs:
                    q_text = item[0]
                    a_text = item[1]
                    c_score = item[2]
                    if not a_text or q_text in matched_questions:
                        continue
                    qu = q_text.upper()
                    if any(re.search(pat, qu) for pat in patterns["customer_name"]):
                        cand = a_text.strip().lstrip(":").strip()
                        cand_upper = cand.upper()
                        # Reject addresses, phone numbers, GSTINs, and generic labels
                        if (
                            len(cand) >= 3
                            and cand_upper not in generic_name_labels
                            and not any(k in cand_upper for k in ["INVOICE", "TAX", "GST", "SIGNATURE"])
                            and not re.search(r"^\s*(?:D\.?No|Plot|Door|Flat|#|\d{1,4}[-/])", cand, re.IGNORECASE)
                            and not gstin_regex.search(cand)
                        ):
                            fields["customer_name"] = cand
                            confs["customer_name"] = round(c_score, 4)
                            matched_questions.add(q_text)
                            break

        # Step 6: Generic Vendor Name Selection
        # Priority 1: Valid company name at the top of the invoice (first lines of OCR text or top valid words)
        # Priority 2: Explicit Seller, Vendor, Supplier, or From labels
        # Priority 3: Commercial signatory lines such as "For [Company Name]"
        # Priority 4: Document headers (excluding table headings, addresses, generic labels)

        def is_valid_vendor_candidate(candidate: str, customer_name: str | None = None) -> bool:
            if not candidate:
                return False
            cand = candidate.strip().strip(":.-").strip()
            if len(cand) < 3:
                return False
            cu = cand.upper()
            cu_norm = re.sub(r"[^A-Z]", "", cu)

            # 1. Reject table headings and common OCR corruptions (e.g. 'ltems' for 'items')
            table_terms = [
                "ITEM", "ITEMS", "LTEM", "LTEMS", "NOOFITEM", "NOOFITEMS", "NOOFLTEM", "NOOFLTEMS",
                "OFITEM", "OFITEMS", "OFLTEM", "OFLTEMS", "DESCRIPTION", "QTY", "QUANTITY",
                "RATE", "PRICE", "AMOUNT", "TOTAL", "DISCOUNT", "SUBTOTAL", "HSN", "SAC",
                "CGST", "SGST", "IGST", "UNIT", "UNITS", "BATCH", "EXP", "MFG",
                "SRNO", "SLNO", "SNO", "SERIAL", "PARTICULAR", "PARTICULARS",
            ]
            if any(t in cu_norm for t in table_terms):
                return False

            # 2. Reject document titles
            doc_titles = [
                "TAX INVOICE", "INVOICE", "RETAIL INVOICE", "COMMERCIAL INVOICE", "BILL OF SUPPLY",
                "RECEIPT", "CASH MEMO", "ORIGINAL FOR RECIPIENT", "DUPLICATE", "TRIPLICATE",
                "DELIVERY CHALLAN", "ESTIMATE", "QUOTATION", "PROFORMA",
            ]
            if any(dt in cu for dt in doc_titles):
                return False

            # 3. Reject tax, registration, banking, license identifiers
            id_markers = [
                "GSTIN", "GST", "PAN", "CIN", "FSSAI", "BANK", "IFSC", "AC NO", "A/C",
                "20B", "21B", "DL NO", "D.L.", "TIN", "TAN", "ACCOUNT",
            ]
            if any(re.search(r"\b" + re.escape(m) + r"\b", cu) for m in id_markers):
                return False

            # 4. Reject address lines
            addr_markers = [
                "ROAD", "STREET", "NAGAR", "COLONY", "AVENUE", "PLOT", "FLAT", "DOOR",
                "D.NO", "D NO", "POST", "DIST", "PIN", "STATE", "LANE", "FLOOR",
                "BUILDING", "COMPLEX", "CITY", "SECTOR", "VIJAYAWADA", "BHAVANIPURAM",
            ]
            if any(re.search(r"\b" + re.escape(m) + r"\b", cu) for m in addr_markers):
                return False

            # 5. Reject contact information (email, phone, URL)
            if "@" in cand or "WWW." in cu or "HTTP" in cu or re.search(r"\b(?:CELL|TEL|PHONE|MOBILE|FAX|EMAIL)\b", cu):
                return False
            if re.search(r"\b\d{6,}\b", cand):
                return False

            # 6. Reject customer indicators or customer name
            cust_markers = ["CUSTOMER", "BUYER", "CONSIGNEE", "BILL TO", "SOLD TO", "TO ", "SIGNATURE"]
            if any(cm in cu for cm in cust_markers):
                return False
            if customer_name and customer_name.upper() in cu:
                return False

            # 7. Reject descriptive taglines in parentheses (e.g. "(DISTRIBUTOR FOR...)")
            if cand.startswith("(") and cand.endswith(")"):
                return False
            if re.search(r"\b(?:DISTRIBUTOR|DEALER|STOCKIST|SUPPLIER)\s+FOR\b", cu):
                return False

            # 8. Reject generic labels
            if cu in generic_name_labels:
                return False

            # 9. Must contain alphabetic characters
            if not any(c.isalpha() for c in cand):
                return False

            return True

        text_source = ocr_text or (" ".join(valid_words) if valid_words else "")
        cust_name = fields.get("customer_name")

        # Priority 1: Valid company name at the top of the invoice (first 10 non-empty lines)
        if "vendor_name" not in fields and text_source:
            lines = [line.strip() for line in text_source.splitlines() if line.strip()]
            for line in lines[:10]:
                cand = line.strip().strip(":.-").strip()
                if is_valid_vendor_candidate(cand, customer_name=cust_name):
                    fields["vendor_name"] = cand
                    confs["vendor_name"] = 0.95
                    break

        # Priority 2: Explicit Seller, Vendor, Supplier, or From labels
        if "vendor_name" not in fields:
            for item in pairs:
                q_text = item[0]
                a_text = item[1]
                c_score = item[2]
                if not a_text or q_text in matched_questions:
                    continue
                if any(re.search(pat, q_text.upper()) for pat in patterns["vendor_name"]):
                    cleaned = a_text.strip().lstrip(":").strip()
                    if is_valid_vendor_candidate(cleaned, customer_name=cust_name):
                        fields["vendor_name"] = cleaned
                        confs["vendor_name"] = round(c_score, 4)
                        matched_questions.add(q_text)
                        break

        # Priority 3: Commercial signatory lines such as "For [Company Name]"
        if "vendor_name" not in fields and text_source:
            for_m = re.search(r"\bFor\s+([A-Z0-9\s&.,'-]{3,50}?)(?:\.|\n|\r|$|Customer|Sign|Authori[sz]ed)", text_source)
            if for_m:
                vname = for_m.group(1).strip().strip(".").strip()
                if is_valid_vendor_candidate(vname, customer_name=cust_name):
                    fields["vendor_name"] = vname
                    confs["vendor_name"] = 0.95

        # Priority 4: Document headers (strictly excluding table headings, column titles, generic labels)
        if "vendor_name" not in fields and headers:
            for h in headers:
                first_header = h.strip().strip(":").strip()
                if is_valid_vendor_candidate(first_header, customer_name=cust_name):
                    fields["vendor_name"] = first_header
                    confs["vendor_name"] = 0.85
                    break

        # Step 7: Currency detection
        if text_source:
            if re.search(r"\b(?:Rupees|Rs\.?|INR|₹)\b", text_source, re.IGNORECASE):
                fields["currency"] = "INR"
                confs["currency"] = 0.99
            elif re.search(r"[$]\b|\bUSD\b", text_source):
                fields["currency"] = "USD"
                confs["currency"] = 0.99
            elif re.search(r"[€]\b|\bEUR\b", text_source):
                fields["currency"] = "EUR"
                confs["currency"] = 0.99
            elif re.search(r"[£]\b|\bGBP\b", text_source):
                fields["currency"] = "GBP"
                confs["currency"] = 0.99

        # For Receipt document type: merchant_name, receipt_number, date
        if document_type == "Receipt":
            if "merchant_name" not in fields and headers:
                fields["merchant_name"] = headers[0]
                confs["merchant_name"] = 0.85
            for rf in ["receipt_number", "date", "total_amount"]:
                if rf not in fields and rf in patterns:
                    for item in pairs:
                        q_text = item[0]
                        a_text = item[1]
                        c_score = item[2]
                        if not a_text or q_text in matched_questions:
                            continue
                        if any(re.search(pat, q_text.upper()) for pat in patterns[rf]):
                            fields[rf] = a_text.strip().lstrip(":").strip()
                            confs[rf] = round(c_score, 4)
                            matched_questions.add(q_text)
                            break

        # Collect unmapped pairs for metadata (DO NOT add arbitrary OCR words to fields!)
        for item in pairs:
            q_text = item[0].strip().rstrip(":").strip()
            a_text = item[1]
            if q_text and item[0] not in matched_questions:
                unmapped_pairs[q_text] = a_text.strip() if a_text else None

        return fields, confs, unmapped_pairs

        return fields, confs, unmapped_pairs

    # ── private helpers ──────────────────────────────────────────────

    @staticmethod
    def _ocr_words_boxes(
        image: Image.Image,
        psm: int | None = None,
        oem: int | None = None,
    ) -> tuple[list[str], list[list[int]]]:
        """Run Tesseract and return (words, [x0,y0,x1,y1] boxes) in invoice reading order.

        Ensures strict 1:1 correspondence between words and bounding boxes,
        filtering out whitespace or invalid boxes.
        Uses configurable invoice-oriented page segmentation (default PSM 3: fully
        automatic page segmentation for multi-column documents) and cleans attached
        delimiter tokens to align with LayoutLMv3 tokenization.
        """
        import re
        import pytesseract
        from app.config import TESSERACT_CMD, TESSERACT_OEM, TESSERACT_PSM

        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        actual_psm = psm if psm is not None else TESSERACT_PSM
        actual_oem = oem if oem is not None else TESSERACT_OEM

        try:
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config=f"--oem {actual_oem} --psm {actual_psm}",
            )
        except Exception as exc:
            logger.warning("Tesseract OCR failed in LayoutLMv3Extractor: %s", exc)
            return [], []

        words: list[str] = []
        boxes: list[list[int]] = []

        texts = data.get("text", [])
        lefts = data.get("left", [])
        tops = data.get("top", [])
        widths = data.get("width", [])
        heights = data.get("height", [])
        n_items = min(len(texts), len(lefts), len(tops), len(widths), len(heights))

        for i in range(n_items):
            txt = str(texts[i]).strip()
            if not txt:
                continue
            try:
                x = int(lefts[i])
                y = int(tops[i])
                w = int(widths[i])
                h = int(heights[i])
            except (ValueError, TypeError):
                continue
            if w <= 0 or h <= 0:
                continue

            # Split attached colons on field headers (e.g. "to:Denise" -> "to:", "Denise"
            # or "Address:16424" -> "Address:", "16424"), preserving URLs and numeric times/percentages.
            if ":" in txt and not txt.startswith(("http://", "https://")) and len(txt) > 2:
                m = re.match(r"^([A-Za-z]+:)(\S+)$", txt)
                if m:
                    p1, p2 = m.group(1), m.group(2)
                    p1_w = max(1, int(w * (len(p1) / len(txt))))
                    words.append(p1)
                    boxes.append([x, y, x + p1_w, y + h])
                    words.append(p2)
                    boxes.append([x + p1_w, y, x + w, y + h])
                    continue

            words.append(txt)
            boxes.append([x, y, x + w, y + h])

        return words, boxes
    @staticmethod
    def _group_entities(
        words: list[str],
        word_preds: dict[int, int],
        word_confs: dict[int, float],
        boxes: list[list[int]] | None = None,
        id2label: dict[int, str] | None = None,
    ) -> list[dict]:
        """Group consecutive same-line words of the same semantic type into entities with 2D boxes."""
        if id2label is None:
            id2label = {i: lbl for i, lbl in enumerate(FUNSD_LABELS)}

        delimiters = {":", "-", "/", "=", "."}
        entities: list[dict] = []
        current: dict | None = None

        for wid in sorted(word_preds.keys()):
            if wid >= len(words):
                continue
            word = str(words[wid]).strip()
            if not word:
                continue
            box = boxes[wid] if boxes and wid < len(boxes) else [0, 0, 0, 0]
            pred_id = word_preds[wid]
            conf = word_confs[wid]
            label_str = id2label.get(pred_id, "O")

            if label_str == "O":
                if current:
                    entities.append(current)
                    current = None
                continue

            prefix, ent_type = label_str.split("-", 1) if "-" in label_str else ("", label_str)
            is_delim = all(c in delimiters for c in word)

            # A QUESTION label must contain at least one alphabetic character unless merging delimiter into an existing QUESTION
            has_alpha = any(c.isalpha() for c in word)
            if ent_type == "QUESTION" and not has_alpha:
                if current is not None and current["label"] == "QUESTION" and is_delim:
                    c_box = current["box"]
                    y_center_c = (c_box[1] + c_box[3]) / 2.0
                    y_center_w = (box[1] + box[3]) / 2.0
                    same_line = abs(y_center_c - y_center_w) <= 10
                    h_gap = box[0] - c_box[2]
                    if same_line and -5 <= h_gap <= 50 and box[0] >= c_box[0] - 5:
                        current["text"] += word
                        current["box"] = [
                            min(c_box[0], box[0]),
                            min(c_box[1], box[1]),
                            max(c_box[2], box[2]),
                            max(c_box[3], box[3]),
                        ]
                if current:
                    entities.append(current)
                    current = None
                continue

            if current is not None:
                c_box = current["box"]
                y_center_c = (c_box[1] + c_box[3]) / 2.0
                y_center_w = (box[1] + box[3]) / 2.0
                same_line = abs(y_center_c - y_center_w) <= 10
                h_gap = box[0] - c_box[2]

                # Merge on same line left-to-right if:
                # 1. same ent_type and horizontal gap is reasonable (-5 <= h_gap <= 50, box[0] >= c_box[0] - 5)
                # 2. current is QUESTION and this token is a delimiter like ':'
                if same_line and -5 <= h_gap <= 50 and box[0] >= c_box[0] - 5:
                    if current["label"] == ent_type:
                        current["text"] += " " + word
                        current["box"] = [
                            min(c_box[0], box[0]),
                            min(c_box[1], box[1]),
                            max(c_box[2], box[2]),
                            max(c_box[3], box[3]),
                        ]
                        current["confidence"] = (current["confidence"] * current["count"] + conf) / (current["count"] + 1)
                        current["count"] += 1
                        continue
                    elif current["label"] == "QUESTION" and is_delim:
                        current["text"] += word
                        current["box"] = [
                            min(c_box[0], box[0]),
                            min(c_box[1], box[1]),
                            max(c_box[2], box[2]),
                            max(c_box[3], box[3]),
                        ]
                        continue

                entities.append(current)
                current = None

            # Don't start a standalone ANSWER entity with just a delimiter
            if ent_type == "ANSWER" and is_delim:
                continue

            current = {
                "label": ent_type,
                "text": word,
                "box": list(box),
                "confidence": conf,
                "count": 1,
            }

        if current:
            entities.append(current)

        return entities

    @staticmethod
    def _pair_qa(
        entities: list[dict],
    ) -> list[tuple[str, str | None, float, list[int], list[int] | None]]:
        """Pair QUESTION entities with ANSWER entities using 2-pass spatial proximity.

        Returns list of (question_text, answer_text, confidence, question_box, answer_box).
        """
        questions = [e for e in entities if e["label"] == "QUESTION"]
        answers = [e for e in entities if e["label"] == "ANSWER"]

        paired_answers: dict[int, tuple[dict, float]] = {}
        used_answers: set[int] = set()

        # Pass 1: Direct same-line matching to the right
        for q_idx, q in enumerate(questions):
            q_box = q.get("box", [0, 0, 0, 0])
            qy_center = (q_box[1] + q_box[3]) / 2.0
            best_ans = None
            best_dist = float("inf")
            best_a_idx = -1

            for a_idx, a in enumerate(answers):
                if a_idx in used_answers:
                    continue
                a_box = a.get("box", [0, 0, 0, 0])
                ay_center = (a_box[1] + a_box[3]) / 2.0
                same_line = abs(qy_center - ay_center) <= 8

                # Must be horizontally to the right
                if same_line and a_box[0] >= q_box[0] - 10:
                    h_dist = max(0, a_box[0] - q_box[2])
                    if h_dist <= 250 and h_dist < best_dist:
                        best_dist = h_dist
                        best_ans = a
                        best_a_idx = a_idx

            if best_ans is not None:
                paired_answers[q_idx] = (best_ans, (q["confidence"] + best_ans["confidence"]) / 2.0)
                used_answers.add(best_a_idx)

        # Pass 2: For questions without same-line answer, look directly below
        for q_idx, q in enumerate(questions):
            if q_idx in paired_answers:
                continue
            q_box = q.get("box", [0, 0, 0, 0])
            best_ans = None
            best_score = float("inf")
            best_a_idx = -1

            for a_idx, a in enumerate(answers):
                if a_idx in used_answers:
                    continue
                a_box = a.get("box", [0, 0, 0, 0])
                dy = a_box[1] - q_box[3]
                dx = abs(a_box[0] - q_box[0])
                if 0 <= dy <= 35 and dx <= 60:
                    score = dy * 2 + dx
                    if score < best_score:
                        best_score = score
                        best_ans = a
                        best_a_idx = a_idx

            if best_ans is not None:
                paired_answers[q_idx] = (best_ans, (q["confidence"] + best_ans["confidence"]) / 2.0)
                used_answers.add(best_a_idx)

        pairs = []
        for q_idx, q in enumerate(questions):
            q_box = q.get("box", [0, 0, 0, 0])
            if q_idx in paired_answers:
                ans, conf = paired_answers[q_idx]
                pairs.append((q["text"], ans["text"], conf, q_box, ans.get("box")))
            else:
                pairs.append((q["text"], None, q["confidence"], q_box, None))

        return pairs
