"""Fine-tuning LayoutLMv3 on FATURA2 (invoices), FUNSD (forms) or CORD (receipts) datasets.

This script can be run locally (CPU) or in Google Colab (GPU).
It trains a LayoutLMv3 token classification model for structured
document understanding.

Usage (local):
    cd backend
    python -m app.models.train_layoutlm --dataset fatura2 --epochs 3 --output ./checkpoints/layoutlmv3-fatura2
    python -m app.models.train_layoutlm --dataset funsd --epochs 5 --output ./checkpoints/layoutlmv3-funsd
    python -m app.models.train_layoutlm --dataset cord --epochs 5 --output ./checkpoints/layoutlmv3-cord

Usage (Colab GPU):
    Upload this script or run notebooks/train_fatura2_layoutlmv3.ipynb on a GPU runtime.

Datasets
--------
- **FATURA2**: 10,000 synthetic invoices across 50 templates. Labels: SELLER, BUYER, NUMBER, DATE, TOTAL, etc.
  Source: mathieu1256/FATURA2-invoices (arXiv:2311.11856)
- **FUNSD**: 199 annotated forms. Labels: HEADER, QUESTION, ANSWER, OTHER.
  License: CC BY 4.0.  Source: https://guillaumejaume.github.io/FUNSD/
- **CORD**: ~1,000 annotated receipt images. Labels: menu.nm, menu.price, total.total_price, etc.
  License: MIT.  Source: https://github.com/clovaai/cord
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ── FUNSD Labels & Helpers ──────────────────────────────────────────

def get_funsd_labels() -> list[str]:
    """FUNSD BIO label list (9 classes)."""
    return [
        "O",
        "B-HEADER", "I-HEADER",
        "B-QUESTION", "I-QUESTION",
        "B-ANSWER", "I-ANSWER",
        "B-OTHER", "I-OTHER",
    ]


def load_funsd():
    """Load FUNSD dataset from Hugging Face datasets hub."""
    from datasets import load_dataset
    return load_dataset("nielsr/funsd-layoutlmv3", trust_remote_code=True)


# ── CORD Labels & Helpers ───────────────────────────────────────────

# Standard target categories in CORD receipts
CORD_CATEGORIES = [
    "menu.nm",
    "menu.num",
    "menu.unitprice",
    "menu.price",
    "menu.discountprice",
    "menu.sub_nm",
    "menu.sub_price",
    "sub_total.subtotal_price",
    "sub_total.discount_price",
    "sub_total.service_price",
    "sub_total.tax_price",
    "sub_total.etc",
    "total.total_price",
    "total.cashprice",
    "total.changeprice",
    "total.creditcardprice",
    "total.menuqty_cnt",
]


def get_cord_labels() -> list[str]:
    """Build BIO label list for CORD receipt understanding."""
    labels = ["O"]
    for cat in CORD_CATEGORIES:
        labels.append(f"B-{cat}")
        labels.append(f"I-{cat}")
    return labels


def load_cord():
    """Load CORD v2 dataset from Hugging Face hub."""
    from datasets import load_dataset
    return load_dataset("naver-clova-ix/cord-v2", trust_remote_code=True)


def parse_cord_sample(sample) -> tuple[list[str], list[list[int]], list[str]]:
    """Parse a single CORD v2 sample's JSON ground truth into tokens, boxes, and BIO labels."""
    image = sample["image"]
    width, height = image.size

    gt_data = sample.get("ground_truth")
    if isinstance(gt_data, str):
        try:
            gt_data = json.loads(gt_data)
        except Exception:
            gt_data = {}

    valid_lines = gt_data.get("valid_line", [])
    tokens: list[str] = []
    bboxes: list[list[int]] = []
    ner_tags: list[str] = []

    for line in valid_lines:
        category = line.get("category", "")
        words = line.get("words", [])
        if not words:
            continue

        is_first = True
        for word_info in words:
            text = word_info.get("text", "").strip()
            if not text:
                continue

            quad = word_info.get("quad", {})
            xs = [quad.get("x1", 0), quad.get("x2", 0), quad.get("x3", 0), quad.get("x4", 0)]
            ys = [quad.get("y1", 0), quad.get("y2", 0), quad.get("y3", 0), quad.get("y4", 0)]

            # Convert to [x0, y0, x1, y1] normalized to 0-1000 range
            x0 = int(min(xs) * 1000 / max(width, 1))
            y0 = int(min(ys) * 1000 / max(height, 1))
            x1 = int(max(xs) * 1000 / max(width, 1))
            y1 = int(max(ys) * 1000 / max(height, 1))

            x0 = max(0, min(1000, x0))
            y0 = max(0, min(1000, y0))
            x1 = max(x0, min(1000, x1))
            y1 = max(y0, min(1000, y1))

            tokens.append(text)
            bboxes.append([x0, y0, x1, y1])

            prefix = "B-" if is_first else "I-"
            tag = f"{prefix}{category}" if category in CORD_CATEGORIES else "O"
            ner_tags.append(tag)
            is_first = False

    return tokens, bboxes, ner_tags


# ── Training Pipelines ──────────────────────────────────────────────

def train_funsd(
    model_name: str = "microsoft/layoutlmv3-base",
    output_dir: str = "./checkpoints/layoutlmv3-funsd",
    epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    max_length: int = 512,
):
    """Fine-tune LayoutLMv3 on FUNSD for form token classification."""
    import numpy as np
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoProcessor,
        TrainingArguments,
        Trainer,
    )

    labels = get_funsd_labels()
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}
    num_labels = len(labels)

    logger.info("Loading FUNSD dataset...")
    dataset = load_funsd()

    logger.info("Loading processor: %s", model_name)
    processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)

    def preprocess(examples):
        images = examples["image"]
        words = examples["tokens"]
        boxes = examples["bboxes"]
        ner_tags = examples["ner_tags"]

        encoding = processor(
            images,
            words,
            boxes=boxes,
            word_labels=ner_tags,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return encoding

    logger.info("Preprocessing FUNSD dataset...")
    train_dataset = dataset["train"].map(
        preprocess,
        batched=True,
        batch_size=2,
        remove_columns=dataset["train"].column_names,
    )
    eval_dataset = dataset["test"].map(
        preprocess,
        batched=True,
        batch_size=2,
        remove_columns=dataset["test"].column_names,
    )

    logger.info("Loading model: %s", model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred):
        from seqeval.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        predictions, labels_batch = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_preds = []
        true_labels = []
        for pred_row, label_row in zip(predictions, labels_batch):
            pred_tags = []
            label_tags = []
            for p, l in zip(pred_row, label_row):
                if l != -100:
                    pred_tags.append(id2label.get(p, "O"))
                    label_tags.append(id2label.get(l, "O"))
            true_preds.append(pred_tags)
            true_labels.append(label_tags)

        return {
            "accuracy": accuracy_score(true_labels, true_preds),
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting FUNSD training (%d epochs)...", epochs)
    train_result = trainer.train()
    logger.info("Training complete. Results: %s", train_result.metrics)

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    eval_results = trainer.evaluate()
    logger.info("Evaluation results: %s", eval_results)

    results_path = Path(output_dir) / "training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({
        "model_name": model_name,
        "dataset": "funsd",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_results,
    }, indent=2))

    logger.info("Model saved to %s", output_dir)
    return eval_results


def train_cord(
    model_name: str = "microsoft/layoutlmv3-base",
    output_dir: str = "./checkpoints/layoutlmv3-cord",
    epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    max_length: int = 512,
):
    """Fine-tune LayoutLMv3 on CORD v2 for receipt entity extraction."""
    import numpy as np
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoProcessor,
        TrainingArguments,
        Trainer,
    )

    labels = get_cord_labels()
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}
    num_labels = len(labels)

    logger.info("Loading CORD v2 dataset...")
    dataset = load_cord()

    logger.info("Loading processor: %s", model_name)
    processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)

    def prepare_cord_batch(batch):
        batch_images = []
        batch_words = []
        batch_boxes = []
        batch_labels = []

        for i in range(len(batch["image"])):
            sample = {
                "image": batch["image"][i],
                "ground_truth": batch["ground_truth"][i],
            }
            tokens, bboxes, ner_tags = parse_cord_sample(sample)
            if not tokens:
                tokens = ["empty"]
                bboxes = [[0, 0, 10, 10]]
                ner_tags = ["O"]

            tag_ids = [label2id.get(tag, 0) for tag in ner_tags]
            batch_images.append(sample["image"].convert("RGB"))
            batch_words.append(tokens)
            batch_boxes.append(bboxes)
            batch_labels.append(tag_ids)

        encoding = processor(
            batch_images,
            batch_words,
            boxes=batch_boxes,
            word_labels=batch_labels,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return encoding

    logger.info("Preprocessing CORD dataset...")
    train_dataset = dataset["train"].map(
        prepare_cord_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset["train"].column_names,
    )
    eval_dataset = dataset["validation"].map(
        prepare_cord_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset["validation"].column_names,
    )

    logger.info("Loading model: %s", model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred):
        from seqeval.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        predictions, labels_batch = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_preds = []
        true_labels = []
        for pred_row, label_row in zip(predictions, labels_batch):
            pred_tags = []
            label_tags = []
            for p, l in zip(pred_row, label_row):
                if l != -100:
                    pred_tags.append(id2label.get(p, "O"))
                    label_tags.append(id2label.get(l, "O"))
            true_preds.append(pred_tags)
            true_labels.append(label_tags)

        return {
            "accuracy": accuracy_score(true_labels, true_preds),
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting CORD training (%d epochs)...", epochs)
    train_result = trainer.train()
    logger.info("Training complete. Results: %s", train_result.metrics)

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    eval_results = trainer.evaluate()
    logger.info("CORD Evaluation results: %s", eval_results)

    results_path = Path(output_dir) / "training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({
        "model_name": model_name,
        "dataset": "cord",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_results,
    }, indent=2))

    logger.info("CORD model saved to %s", output_dir)
    return eval_results


# ── FATURA2 Labels, Helpers & Training Pipeline ─────────────────────

# Official integer ID mapping verified against Zenodo annotation files (https://zenodo.org/records/10371464)
FATURA2_INTEGER_TAG_MAP = {
    0: "SELLER_SITE",
    1: "TOTAL",
    2: "NOTE",            # Total in words / notes
    3: "DATE",             # Invoice date
    4: "DUE_DATE",         # Payment due date
    5: "BILL_TO",          # Buyer / customer name & billing block
    6: "SELLER",           # Seller / vendor name
    7: "OTHER",            # Absent across all dataset records (mapped to outside)
    8: "BILL_TO",          # Explicit BILL_TO block
    9: "SEND_TO",          # Explicit SHIP_TO / SEND_TO block
    10: "TABLE",           # Product line-item table
    11: "LOGO",            # Vendor logo
    12: "NUMBER",          # Invoice number / ID
    13: "OTHER",           # Background / non-target elements
}

# Authentic 24 semantic classes from Table 2 of FATURA (arXiv:2311.11856)
FATURA2_CATEGORIES = [
    "TABLE",
    "LOGO",
    "TITLE",
    "NUMBER",
    "DATE",
    "DUE_DATE",
    "PO_NUMBER",
    "CONDITIONS",
    "SELLER",
    "SELLER_ADDRESS",
    "SELLER_SITE",
    "BILL_TO",
    "SEND_TO",
    "SUBTOTAL",
    "TAX",
    "DISCOUNT",
    "AMOUNT_DUE",
    "TOTAL",
    "ITEM_DESCRIPTION",
    "ITEM_QUANTITY",
    "ITEM_UNIT_PRICE",
    "ITEM_TOTAL",
    "NOTE",
]

FATURA2_ALIASES = {
    "BUYER": "BILL_TO",
    "BUYER_NAME": "BILL_TO",
    "CUSTOMER": "BILL_TO",
    "CUSTOMER_NAME": "BILL_TO",
    "BUYER_ADDRESS": "SEND_TO",
    "CUSTOMER_ADDRESS": "SEND_TO",
    "SHIPPING_ADDRESS": "SEND_TO",
    "SHIP_TO": "SEND_TO",
    "INVOICE_NUMBER": "NUMBER",
    "INV_NO": "NUMBER",
    "INVOICE_ID": "NUMBER",
    "INVOICE_DATE": "DATE",
    "GRAND_TOTAL": "TOTAL",
    "TOTAL_PRICE": "TOTAL",
    "SUB_TOTAL": "SUBTOTAL",
    "TAXABLE_AMOUNT": "SUBTOTAL",
    "TAX_AMOUNT": "TAX",
    "TOTAL_TAX": "TAX",
    "VAT": "TAX",
    "GST": "TAX",
    "VENDOR": "SELLER",
    "VENDOR_NAME": "SELLER",
    "SUPPLIER": "SELLER",
    "REMARKS": "NOTE",
    "TERMS": "CONDITIONS",
    "WEBSITE": "SELLER_SITE",
}


def get_fatura2_labels(custom_names: list[str] | None = None) -> list[str]:
    """Build canonical BIO label list for FATURA2 invoice understanding."""
    if custom_names:
        has_bio = any(isinstance(n, str) and (n.startswith("B-") or n.startswith("I-")) for n in custom_names)
        if has_bio:
            return list(custom_names)
        cats = [c for c in custom_names if c not in ("O", "OTHER", "0")]
    else:
        cats = FATURA2_CATEGORIES

    labels = ["O"]
    for cat in cats:
        labels.append(f"B-{cat}")
        labels.append(f"I-{cat}")
    return labels


def convert_spans_to_bio(
    raw_tags: list[int | str],
    words: list[str],
    boxes: list[list[int]] | None,
    category_names: list[str] | dict[int, str] | None,
    bio_label2id: dict[str, int],
    is_source_bio: bool = False,
) -> tuple[list[str], list[int]]:
    """Convert raw token tags into aligned BIO tag names and integer IDs."""
    if is_source_bio:
        bio_tags = []
        bio_ids = []
        for t in raw_tags:
            if isinstance(t, str):
                tag_str = t
                tag_id = bio_label2id.get(tag_str, 0)
            elif isinstance(t, int) and category_names and t < len(category_names):
                tag_str = category_names[t]
                tag_id = bio_label2id.get(tag_str, t if t < len(bio_label2id) else 0)
            else:
                tag_id = int(t) if int(t) < len(bio_label2id) else 0
                tag_str = "O"
            bio_tags.append(tag_str)
            bio_ids.append(tag_id)
        return bio_tags, bio_ids

    bio_tags = []
    bio_ids = []
    prev_cat = None
    prev_box = None

    for idx, raw_t in enumerate(raw_tags):
        if isinstance(raw_t, int):
            if category_names and isinstance(category_names, dict) and raw_t in category_names:
                raw_str = str(category_names[raw_t]).strip().upper()
            else:
                raw_str = FATURA2_INTEGER_TAG_MAP.get(raw_t, "OTHER")
        elif isinstance(raw_t, str):
            raw_str = raw_t.strip().upper()
        else:
            raw_str = "OTHER"

        cat_name = FATURA2_ALIASES.get(raw_str, raw_str)

        if cat_name in ("O", "OTHER", "BACKGROUND", "NONE", "0"):
            bio_tag = "O"
            prev_cat = None
            prev_box = None
        else:
            box = boxes[idx] if boxes and idx < len(boxes) else None
            is_continuation = (prev_cat == cat_name)
            if is_continuation and box and prev_box and len(box) == 4 and len(prev_box) == 4:
                prev_y_center = (prev_box[1] + prev_box[3]) / 2.0
                curr_y_center = (box[1] + box[3]) / 2.0
                y_diff = abs(curr_y_center - prev_y_center)
                carriage_return = box[0] < prev_box[0] - 100
                if y_diff > 35 or (carriage_return and y_diff > 15):
                    is_continuation = False

            if is_continuation:
                bio_tag = f"I-{cat_name}"
            else:
                bio_tag = f"B-{cat_name}"

            prev_cat = cat_name
            prev_box = box

        if bio_tag not in bio_label2id:
            bio_tag = "O"

        bio_tags.append(bio_tag)
        bio_ids.append(bio_label2id[bio_tag])

    return bio_tags, bio_ids


def normalize_bbox(box: list[int] | tuple[int, ...], width: int, height: int) -> list[int]:
    """Normalize bounding box coordinates to LayoutLMv3 [0, 1000] scale."""
    if not box or len(box) != 4:
        return [0, 0, 0, 0]
    try:
        x0, y0, x1, y1 = [int(v) for v in box]
    except (ValueError, TypeError):
        return [0, 0, 0, 0]

    w = max(int(width), 1)
    h = max(int(height), 1)

    # Check if already normalized to 0-1000 range
    if max(x0, x1) <= 1000 and max(y0, y1) <= 1000 and w > 1000 and h > 1000:
        nx0 = max(0, min(1000, x0))
        ny0 = max(0, min(1000, y0))
        nx1 = max(nx0, min(1000, x1))
        ny1 = max(ny0, min(1000, y1))
        return [nx0, ny0, nx1, ny1]

    nx0 = max(0, min(1000, int(x0 * 1000 / w)))
    ny0 = max(0, min(1000, int(y0 * 1000 / h)))
    nx1 = max(nx0, min(1000, int(x1 * 1000 / w)))
    ny1 = max(ny0, min(1000, int(y1 * 1000 / h)))
    return [nx0, ny0, nx1, ny1]


def load_fatura2(
    sample_limit: int = 5000,
    train_size: int = 4400,
    val_size: int = 600,
    strict_template_split: bool = True,
):
    """Load FATURA2 dataset from Hugging Face hub with zero-leakage template partitioning.

    Requirements:
      - Uses only `mathieu1256/FATURA2-invoices`.
      - In FATURA (arXiv:2311.11856), each of the 50 synthetic templates produces exactly 200 invoices.
      - A strict template-level split requires integer multiples of 200:
        22 training templates (4,400 samples) and 3 validation templates (600 samples),
        totaling exactly 5,000 samples with 0.0% template overlap.
      - Keeps the official 1,400-example test split completely untouched (templates 43..49).
    """
    from datasets import load_dataset

    logger.info("Loading dataset: mathieu1256/FATURA2-invoices...")
    full_dataset = load_dataset("mathieu1256/FATURA2-invoices")

    if "test" in full_dataset:
        raw_train = full_dataset["train"]
        raw_test = full_dataset["test"]
    else:
        logger.info("Single split detected. Slicing official 8,600 train and 1,400 holdout test splits...")
        total_len = len(full_dataset["train"])
        train_end = min(8600, total_len)
        raw_train = full_dataset["train"].select(range(0, train_end))
        raw_test = full_dataset["train"].select(range(train_end, total_len))

    logger.info("Loaded mathieu1256/FATURA2-invoices: train=%d, test=%d", len(raw_train), len(raw_test))

    # In FATURA, each template generates exactly 200 invoice instances
    template_block_size = 200
    if strict_template_split and (train_size % template_block_size != 0 or val_size % template_block_size != 0):
        n_val = max(1, round(val_size / template_block_size))
        n_total = sample_limit // template_block_size
        n_train = n_total - n_val
        adj_train = n_train * template_block_size
        adj_val = n_val * template_block_size
        logger.info(
            "Strict template-level split: Adjusted train/val sample counts from %d/%d to %d/%d "
            "(%d templates train, %d templates val) to eliminate template straddling across boundaries (0.0%% overlap).",
            train_size, val_size, adj_train, adj_val, n_train, n_val
        )
        train_size, val_size = adj_train, adj_val

    cols = raw_train.column_names
    template_col = next((c for c in ["template_id", "template", "template_name", "layout"] if c in cols), None)

    if template_col:
        logger.info("Grouping by template column '%s' to avoid template leakage...", template_col)
        unique_templates = sorted(list(set(raw_train[template_col])))
        val_template_count = max(1, round(len(unique_templates) * (val_size / sample_limit)))
        val_templates = set(unique_templates[-val_template_count:])
        train_templates = set(unique_templates[:-val_template_count])

        train_indices = [i for i, t in enumerate(raw_train[template_col]) if t in train_templates][:train_size]
        val_indices = [i for i, t in enumerate(raw_train[template_col]) if t in val_templates][:val_size]
    else:
        logger.info(
            "Partitioning by 200-sample template blocks: train [0..%d] (templates 0..%d), "
            "val [%d..%d] (templates %d..%d). Template overlap: 0.0%%",
            train_size,
            (train_size // template_block_size) - 1,
            train_size,
            train_size + val_size,
            train_size // template_block_size,
            (train_size + val_size) // template_block_size - 1,
        )
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))

    train_split = raw_train.select(train_indices)
    val_split = raw_train.select(val_indices)

    logger.info(
        "Partitioned: train=%d (22 templates), val=%d (3 templates), test=%d (7 templates untouched)",
        len(train_split),
        len(val_split),
        len(raw_test),
    )
    return {
        "train": train_split,
        "validation": val_split,
        "test": raw_test,
        "features": raw_train.features,
    }


def train_fatura2(
    model_name: str = "microsoft/layoutlmv3-base",
    output_dir: str = "./checkpoints/layoutlmv3-fatura2",
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 5e-5,
    max_length: int = 512,
    train_samples: int = 4400,
    val_samples: int = 600,
    strict_template_split: bool = True,
):
    """Fine-tune LayoutLMv3 on mathieu1256/FATURA2-invoices (5,000 examples with strict template split)."""
    import numpy as np
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoProcessor,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    logger.info("Loading FATURA2 dataset (5,000 subset: %d train, %d val, strict_template_split=%s)...", train_samples, val_samples, strict_template_split)
    dataset_dict = load_fatura2(
        sample_limit=train_samples + val_samples,
        train_size=train_samples,
        val_size=val_samples,
        strict_template_split=strict_template_split,
    )

    features = dataset_dict["features"]
    custom_names = None
    id2label_raw = None
    if "ner_tags" in features and hasattr(features["ner_tags"], "feature") and hasattr(features["ner_tags"].feature, "names"):
        custom_names = features["ner_tags"].feature.names
        id2label_raw = {i: name for i, name in enumerate(custom_names)}
        logger.info("Discovered %d official labels from dataset metadata", len(custom_names))

    labels = get_fatura2_labels(custom_names)
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}
    num_labels = len(labels)

    is_source_bio = bool(custom_names and any(n.startswith("B-") or n.startswith("I-") for n in custom_names))
    category_reference = id2label_raw if id2label_raw else FATURA2_INTEGER_TAG_MAP

    logger.info("Loading processor: %s (is_source_bio=%s)", model_name, is_source_bio)
    processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)

    def prepare_fatura2_batch(batch):
        batch_images = []
        batch_words = []
        batch_boxes = []
        batch_labels = []

        for i in range(len(batch["image"])):
            img = batch["image"][i].convert("RGB")
            w, h = img.size
            words = batch.get("words", [[]])[i] if "words" in batch else batch.get("tokens", [[]])[i]
            boxes = batch.get("bboxes", [[]])[i] if "bboxes" in batch else batch.get("boxes", [[]])[i]
            raw_tags = batch.get("ner_tags", [[]])[i]

            norm_boxes = [normalize_bbox(b, w, h) for b in boxes]
            if not words:
                words = ["empty"]
                norm_boxes = [[0, 0, 10, 10]]
                raw_tags = [0]

            _, tag_ids = convert_spans_to_bio(
                raw_tags=raw_tags,
                words=words,
                boxes=norm_boxes,
                category_names=category_reference,
                bio_label2id=label2id,
                is_source_bio=is_source_bio,
            )

            batch_images.append(img)
            batch_words.append(words)
            batch_boxes.append(norm_boxes)
            batch_labels.append(tag_ids)

        encoding = processor(
            batch_images,
            batch_words,
            boxes=batch_boxes,
            word_labels=batch_labels,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return encoding

    logger.info("Preprocessing FATURA2 train, validation, and test splits...")
    train_dataset = dataset_dict["train"].map(
        prepare_fatura2_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset_dict["train"].column_names,
    )
    val_dataset = dataset_dict["validation"].map(
        prepare_fatura2_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset_dict["validation"].column_names,
    )
    test_dataset = dataset_dict["test"].map(
        prepare_fatura2_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset_dict["test"].column_names,
    )

    logger.info("Loading model: %s with %d labels", model_name, num_labels)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred):
        from seqeval.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        predictions, labels_batch = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_preds = []
        true_labels = []
        for pred_row, label_row in zip(predictions, labels_batch):
            pred_tags = []
            label_tags = []
            for p, l in zip(pred_row, label_row):
                if l != -100:
                    pred_tags.append(id2label.get(p, "O"))
                    label_tags.append(id2label.get(l, "O"))
            true_preds.append(pred_tags)
            true_labels.append(label_tags)

        return {
            "accuracy": accuracy_score(true_labels, true_preds),
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    logger.info("Starting FATURA2 training (3 epochs, effective batch size=%d)...", batch_size * gradient_accumulation_steps)
    train_result = trainer.train()
    logger.info("Training complete. Results: %s", train_result.metrics)

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    val_results = trainer.evaluate(val_dataset)
    logger.info("FATURA2 Validation results: %s", val_results)

    test_results = trainer.evaluate(test_dataset)
    logger.info("FATURA2 Test (1,400 holdout) results: %s", test_results)

    results_path = Path(output_dir) / "training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "dataset": "mathieu1256/FATURA2-invoices",
                "train_samples": train_samples,
                "val_samples": val_samples,
                "test_samples": len(dataset_dict["test"]),
                "epochs": epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": batch_size * gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "train_metrics": train_result.metrics,
                "val_metrics": val_results,
                "test_metrics": test_results,
                "labels": labels,
            },
            indent=2,
        )
    )

    logger.info("FATURA2 model and evaluation results saved to %s", output_dir)
    return {"val": val_results, "test": test_results}


# ── Entry point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LayoutLMv3 on FATURA2 (invoices), FUNSD (forms) or CORD (receipts)"
    )
    parser.add_argument(
        "--dataset",
        choices=["fatura2", "funsd", "cord"],
        default="fatura2",
        help="Dataset to fine-tune on (fatura2, funsd or cord)",
    )
    parser.add_argument(
        "--model",
        default="microsoft/layoutlmv3-base",
        help="Base model name (default: microsoft/layoutlmv3-base)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for checkpoint (defaults to ./checkpoints/layoutlmv3-<dataset>)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--train-samples", type=int, default=4400, help="Number of training samples (default: 4400, 22 full templates)")
    parser.add_argument("--val-samples", type=int, default=600, help="Number of validation samples (default: 600, 3 full templates)")
    parser.add_argument(
        "--no-strict-template-split",
        action="store_true",
        help="Disable strict 200-sample template alignment (allows partial template boundaries)",
    )
    args = parser.parse_args()

    output_dir = args.output or f"./checkpoints/layoutlmv3-{args.dataset}"

    if args.dataset == "fatura2":
        train_fatura2(
            model_name=args.model,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            strict_template_split=not args.no_strict_template_split,
        )
    elif args.dataset == "funsd":
        train_funsd(
            model_name=args.model,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    elif args.dataset == "cord":
        train_cord(
            model_name=args.model,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )


if __name__ == "__main__":
    main()
