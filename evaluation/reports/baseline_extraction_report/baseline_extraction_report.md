# Baseline Extraction Evaluation Report: LayoutLMv3 (FATURA2) on Prashanti Invoices

**Evaluation Date:** 2026-11-18  
**Active Model:** `checkpoints/layoutlmv3-fatura2` (47 BIO classes)  
**Dataset Evaluated:** 31 unique invoices from `evaluation/invoice_evaluation/` (34 pages total)  
**Deduplication:** `invoice_21.pdf` excluded (identical byte duplicate of `invoice_20.pdf`)  
**Status:** Model fine-tuning has NOT started; application code untouched.  

---

## 1. Executive Summary

Before fine-tuning the LayoutLMv3 checkpoint on domain-specific Prashanti-style invoices, a rigorous baseline evaluation was conducted across all 31 unique invoices using the production inference pipeline (`LayoutLMv3Extractor` + production OCR/heuristics).

- **Total Fields Evaluated:** 341 (31 invoices × 11 fields)
- **Correct Extractions:** 85 (24.9%)
- **Incorrect Extractions:** 93 (27.3%)
- **Missing Extractions:** 132 (38.7%)
- **Uncertain Extractions:** 31 (9.1%)

The baseline evaluation reveals that while the existing zero-shot FATURA2 model performs well on standard formatting fields like **Invoice Date (100.0%)** and **Invoice Number (80.6%)**, it exhibits **critical failure modes** on domain-specific Indian pharmaceutical invoices:
1. **Complete failure to detect Customer / Buyer Name (0.0% accuracy, 100% missing)**.
2. **Complete failure on Vendor Name (0.0% accuracy)**, frequently hallucinating line item descriptions (`zyous CADLON`, `SWABS`, `WOOL IP`) or street addresses as seller.
3. **Severe CGST / Total Tax distortion and duplication (0.0% accuracy, 100% incorrect)**, where CGST captures the Net Total amount from the page footer, which is subsequently summed into the Total Tax calculation.
4. **Failure to extract Gross Amount / Subtotal (0.0%) and Net Amount / Total (0.0%)**.

These empirical baseline metrics establish a clear benchmark that domain-specific fine-tuning is required to solve.

---

## 2. Field-wise Accuracy Benchmark

| Field Name | Target Canonical Field | Correct | Incorrect | Missing | Uncertain | Accuracy (%) | Error Rate (%) | Missing Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vendor Name (`vendor_name`) | vendor_name | 0 | 22 | 9 | 0 | **0.0%** | 71.0% | 29.0% |
| Customer Name (`customer_name`) | customer_name | 0 | 0 | 31 | 0 | **0.0%** | 0.0% | 100.0% |
| Invoice Number (`invoice_number`) | invoice_number | 25 | 6 | 0 | 0 | **80.6%** | 19.4% | 0.0% |
| Invoice Date (`invoice_date`) | invoice_date | 31 | 0 | 0 | 0 | **100.0%** | 0.0% | 0.0% |
| Subtotal / Gross (`subtotal`) | subtotal | 0 | 0 | 31 | 0 | **0.0%** | 0.0% | 100.0% |
| Discount (`discount`) | discount | 0 | 0 | 31 | 0 | **0.0%** | 0.0% | 100.0% |
| CGST Tax (`cgst_amount`) | cgst | 0 | 31 | 0 | 0 | **0.0%** | 100.0% | 0.0% |
| SGST Tax (`sgst_amount`) | sgst | 29 | 2 | 0 | 0 | **93.5%** | 6.5% | 0.0% |
| Total Tax (`tax_amount`) | total_tax | 0 | 31 | 0 | 0 | **0.0%** | 100.0% | 0.0% |
| Total / Net (`total_amount`) | total_amount | 0 | 1 | 30 | 0 | **0.0%** | 3.2% | 96.8% |
| GSTIN (`gst_number`) | gstin | 0 | 0 | 0 | 31 | **0.0%** | 0.0% | 0.0% |

---

## 3. Deep-Dive Analysis of Critical Failure Modes

### A. Invoice Number Confusion (19.4% Error Rate)
In 25 of 31 invoices (80.6%), the model successfully extracted the statutory `GSTSAL/...` invoice number. However, in 6 invoices, the model suffered from **entity confusion**:
- **Vendor Door Number Capture (5 invoices)**: In `invoice_08`, `invoice_12`, `invoice_15`, `invoice_24`, and `invoice_25`, the extractor captured `D.NO.76-14-95/2A,` (the vendor's premise address number) instead of `GSTSAL/...` due to the word `NO.` in `D.NO.` triggering invoice number proximity rules.
- **Drug License Number Capture (1 invoice)**: In `invoice_07`, the extractor captured `AP/06/03/2021-16440` (the regulatory drug license number) instead of `GSTSAL/996`.

### B. CGST/SGST Conflation and Total Tax Duplication (100% Error Rate on CGST & Total Tax)
In Indian GST pharmaceutical invoices, CGST and SGST amounts are equal half-shares of the total tax. The baseline pipeline produced severe errors:
1. **Net Amount Conflation**: The statutory regex `r'\bCGST\b[^\d]*?([0-9]+\.[0-9]{2})'` in the extractor spanned across vertical layout lines and captured the grand total (`Net Amount`) from the footer rather than the CGST amount row.
   - *Example (invoice_01)*: Reference CGST = `39.03`, Extracted CGST = `1639.00` (which is the Net Total!).
   - *Example (invoice_03)*: Reference CGST = `46.38`, Extracted CGST = `1948.00` (Net Total!).
   - *Example (invoice_04)*: Reference CGST = `73.40`, Extracted CGST = `3083.00` (Net Total!).
2. **SGST Independence**: In contrast, SGST extraction was highly accurate (**93.5%**, 29/31 invoices), correctly matching the SGST row value.
3. **Total Tax Duplication & Inflation**: Because the pipeline calculated `total_tax = cgst_amount + sgst_amount`, it summed the Net Total with the SGST tax (e.g. `1639.00 + 39.03 = 1678.03`), resulting in a 100% error rate on Total Tax.

### C. Complete Absence of Buyer / Customer Name (100% Missing)
The zero-shot FATURA2 model (trained on European/Turkish commercial invoices) failed to predict `B-BILL_TO` / `I-BILL_TO` tokens for Indian hospital and pharmacy names (such as `MAHAJAN PHARMACY`, `NORI MEDICARE PRIVATE LIMITED`, `ANDHRA HOSPITALS`). Every single invoice returned `customer_name = None`.

### D. Vendor Name Hallucinations (71% Error Rate, 29% Missing)
Rather than recognizing the standardized distributor header `SIVA SAI ENTERPRISES`, the model tagged product line items (`zyous CADLON`, `SWABS`, `WOOL IP`, `HMD DISPOVAN`) or regional street tokens (`ROAD, Gandhi Nagar`, `Gollapudi`) as `SELLER`.

### E. Subtotal & Total Amount Drop (100% Missing)
The zero-shot model failed to locate `Gross Amount` (`B-SUBTOTAL`) and `Net Amount` (`B-TOTAL`) on the summary panel, resulting in complete omission of these key financial figures.

### F. GSTIN Extraction & Buyer vs. Seller Ambiguity (100% Uncertain)
Valid GSTINs were extracted on all 31 invoices via statutory regex (`37...`). However, in all 31 instances, the regex captured the **Buyer's GSTIN** (e.g., `37COAPS3244E1Z1`) rather than the **Seller's GSTIN** (`37AAGFS0707H1ZS`) because the buyer GSTIN appears first in reading order on the left column. Per ground truth evaluation rules, these are classified as **Uncertain**, requiring explicit user confirmation on whether Seller or Buyer GSTIN is desired as the primary `gst_number`.

---

## 4. Invoice-by-Invoice Results Table

| Invoice File | Pages | Correct | Incorrect | Missing | Uncertain | Key Extracted Values vs Reference Discrepancies |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `invoice_01.pdf` | 1 | 3 | 3 | 4 | 1 | CGST: 1639.00 (confused w/ Net 1639.00); Vendor: 'zyous CADLON' |
| `invoice_02.pdf` | 2 | 3 | 2 | 5 | 1 | CGST: 1.00 (confused w/ Net 16876.00); Vendor: Missing |
| `invoice_03.pdf` | 1 | 3 | 2 | 5 | 1 | CGST: 1948.00 (confused w/ Net 1948.00); Vendor: Missing |
| `invoice_04.pdf` | 1 | 3 | 4 | 3 | 1 | CGST: 3083.00 (confused w/ Net 3083.00); Vendor: 'SWABS' |
| `invoice_05.pdf` | 1 | 3 | 3 | 4 | 1 | CGST: 2772.00 (confused w/ Net 2772.00); Vendor: 'WOOL IP' |
| `invoice_06.pdf` | 1 | 3 | 3 | 4 | 1 | CGST: 1701.00 (confused w/ Net 1701.00); Vendor: 'ROAD, Gandhi Nagar' |
| `invoice_07.pdf` | 1 | 2 | 4 | 4 | 1 | InvNo: 'AP/06/03/2021-16440' (expected 'GSTSAL/996'); CGST: 1654.00 (confused w/ Net 1654.00); Vendor: 'HMD DISPOVAN' |
| `invoice_08.pdf` | 2 | 2 | 4 | 4 | 1 | InvNo: 'D.NO.76-14-95/2A,' (expected 'GSTSAL/997'); CGST: 1.00 (confused w/ Net 18622.00); Vendor: 'Gollapudi' |
| `invoice_09.pdf` | 1 | 3 | 2 | 5 | 1 | CGST: 1533.00 (confused w/ Net 1533.00); Vendor: Missing |
| `invoice_10.pdf` | 2 | 3 | 3 | 4 | 1 | CGST: 1.00 (confused w/ Net 14806.00); Vendor: 'CANNULA' |
| `invoice_11.pdf` | 1 | 3 | 2 | 5 | 1 | CGST: 8876.00 (confused w/ Net 8876.00); Vendor: Missing |
| `invoice_12.pdf` | 1 | 2 | 3 | 5 | 1 | InvNo: 'D.NO.76-14-95/2A,' (expected 'GSTSAL/1023'); CGST: 21009.00 (confused w/ Net 21009.00); Vendor: Missing |
| `invoice_13.pdf` | 1 | 3 | 3 | 4 | 1 | CGST: 2822.00 (confused w/ Net 2822.00); Vendor: 'Vinayaka Theatre' |
| `invoice_14.pdf` | 1 | 3 | 3 | 4 | 1 | CGST: 998.00 (confused w/ Net 998.00); Vendor: 'Chettu Center' |
| `invoice_15.pdf` | 1 | 2 | 4 | 4 | 1 | InvNo: 'D.NO.76-14-95/2A,' (expected 'GSTSAL/1336'); CGST: 21399.00 (confused w/ Net 21399.00); Vendor: 'COTTON CREPE BANDAGE' |
| `invoice_16.pdf` | 1 | 3 | 3 | 4 | 1 | CGST: 889.00 (confused w/ Net 889.00); Vendor: 'INJ' |
| `invoice_17.pdf` | 1 | 3 | 2 | 5 | 1 | CGST: 341.00 (confused w/ Net 341.00); Vendor: Missing |
| `invoice_18.pdf` | 1 | 3 | 3 | 4 | 1 | CGST: 9240.00 (confused w/ Net 9240.00); Vendor: 'Road,' |
| `invoice_19.pdf` | 1 | 3 | 2 | 5 | 1 | CGST: 866.00 (confused w/ Net 866.00); Vendor: Missing |
| `invoice_20.pdf` | 1 | 3 | 3 | 4 | 1 | CGST: 3178.00 (confused w/ Net 3178.00); Vendor: '1 ABBOTT' |
| `invoice_22.pdf` | 1 | 3 | 3 | 4 | 1 | CGST: 2114.00 (confused w/ Net 2114.00); Vendor: 'MEDIUM' |
| `invoice_23.pdf` | 1 | 3 | 3 | 4 | 1 | CGST: 2822.00 (confused w/ Net 2822.00); Vendor: 'Vinayaka Theatre' |
| `invoice_24.pdf` | 1 | 2 | 4 | 4 | 1 | InvNo: 'D.NO.76-14-95/2A,' (expected 'GSTSAL/732'); CGST: 8610.00 (confused w/ Net 8610.00); Vendor: 'Gollapudi' |
| `invoice_25.pdf` | 1 | 2 | 3 | 5 | 1 | InvNo: 'D.NO.76-14-95/2A,' (expected 'GSTSAL/134'); CGST: 879.00 (confused w/ Net 879.00); Vendor: Missing |
| `invoice_26.pdf` | 1 | 3 | 3 | 4 | 1 | CGST: 9975.00 (confused w/ Net 9975.00); Vendor: 'HEPARIN' |
| `invoice_27.pdf` | 1 | 3 | 3 | 4 | 1 | CGST: 3833.00 (confused w/ Net 3833.00); Vendor: 'SKIN STAPLER' |
| `invoice_28.pdf` | 1 | 3 | 3 | 4 | 1 | CGST: 4872.00 (confused w/ Net 4872.00); Vendor: 'SWABS' |
| `invoice_29.pdf` | 1 | 2 | 4 | 4 | 1 | CGST: 11176.00 (confused w/ Net 11176.00); Vendor: 'ROAD, Gandhi Nagar' |
| `invoice_30.pdf` | 1 | 3 | 3 | 4 | 1 | CGST: 6825.00 (confused w/ Net 6825.00); Vendor: 'Gollapudi' |
| `invoice_31.pdf` | 1 | 2 | 4 | 4 | 1 | CGST: 11412.00 (confused w/ Net 11412.00); Vendor: 'pedia drip-plus' |
| `invoice_32.pdf` | 1 | 3 | 2 | 5 | 1 | CGST: 17591.00 (confused w/ Net 17591.00); Vendor: Missing |

---

## 5. Dataset Splits Customer Overlap Inspection

As requested, `dataset_splits.json` was inspected to determine whether customer entities recur across splits.

### Customer Overlap Findings:
- **Train Split (21 invoices, 23 pages)**: 18 unique customer names
- **Validation Split (5 invoices, 6 pages)**: 5 unique customer names
- **Test Split (5 invoices, 5 pages)**: 5 unique customer names

#### Overlapping Customers Detected Across Splits:
1. **Train ∩ Validation Overlap (2 customers)**:
   - `ANDHRA HOSPITALS BHAVANIPURAM PRIVATE LIMITED` (Invoices: `invoice_08` [Train], `invoice_24` [Val], `invoice_30` [Test])
   - `SINGH NAGAR P N P HEALTH CARE` (Invoices: `invoice_12` [Train], `invoice_27` [Val])
2. **Train ∩ Test Overlap (4 customers)**:
   - `ANDHRA HOSPITALS BHAVANIPURAM PRIVATE LIMITED` (`invoice_08` [Train] vs `invoice_30` [Test])
   - `JANAKIRAMA MATERNITY & GENERAL HOSPITAL` (`invoice_04` [Train] vs `invoice_28` [Test])
   - `NORI MEDICARE PRIVATE LIMITED` (`invoice_06` [Train] vs `invoice_29` [Test])
   - `PRASANTHI PHARMACY` (`invoice_20` [Train] vs `invoice_32` [Test])
3. **Validation ∩ Test Overlap (1 customer)**:
   - `ANDHRA HOSPITALS BHAVANIPURAM PRIVATE LIMITED` (`invoice_24` [Val] vs `invoice_30` [Test])

> [!NOTE]
> **Why Customer Overlap Occurs:** The current `dataset_splits.json` partitioned data at the **invoice level** (guaranteeing that multi-page invoices like `invoice_02`, `invoice_08`, and `invoice_10` never have pages separated across splits). However, because SIVA SAI ENTERPRISES serves recurring pharmaceutical clients, multiple invoices share the same client. While invoice numbers, line items, and financial values are unique, customer names overlap. In accordance with instructions, **the splits have not been modified**.

---

## 6. Conclusions & Fine-Tuning Recommendations

1. **Baseline Accuracy is 24.9%**: Confined primarily to Invoice Date (`100%`) and clean Invoice Numbers (`80.6%`).
2. **Zero-Shot Generalization Fails on Domain Layouts**: Customer Name, Subtotal, Discount, and Net Total have 0% baseline extraction.
3. **Regex Heuristic Conflation Must Be Eliminated**: The `CGST` statutory regex must not blindly search forward across lines into the footer, which conflates CGST with Net Amount and ruins Total Tax.
4. **Fine-Tuning is Justified**: Fine-tuning LayoutLMv3 on the verified Prashanti annotations will teach the spatial transformer to directly classify tokens into `B-BILL_TO`, `B-SUBTOTAL`, `B-TOTAL`, `B-TAX`, and `B-SELLER` based on 2D coordinates.

All evaluation artifacts are preserved in `evaluation/reports/baseline_extraction_report/`.