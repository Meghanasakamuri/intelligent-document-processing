"""Validation utilities for invoice dates and financial fields.

Provides:
- DateNormalizer: Converts diverse invoice date formats into ISO-8601 (YYYY-MM-DD),
  flags ambiguous dates without guessing, rejects impossible calendar dates, and
  preserves raw text for traceability.
- FinancialValidator: Validates mathematical consistency between subtotal, tax,
  discount, additional charges, and total amount, cross-checking line-item sums
  with configurable rounding tolerance.
"""

from __future__ import annotations

import calendar
import datetime
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


@dataclass
class DateNormalizationResult:
    """Result of normalizing an extracted date string."""

    raw: str | None
    normalized: str | None = None  # ISO-8601 format: YYYY-MM-DD
    is_valid: bool = True
    is_ambiguous: bool = False
    error: str | None = None
    possible_dates: list[str] = field(default_factory=list)
    format_detected: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DateNormalizer:
    """Normalize invoice date strings into ISO-8601 dates with ambiguity detection."""

    @classmethod
    def _is_leap_year(cls, year: int) -> bool:
        return calendar.isleap(year)

    @classmethod
    def _validate_calendar_date(cls, year: int, month: int, day: int) -> tuple[bool, str | None]:
        """Verify that year, month, and day form a valid calendar date."""
        if not (1 <= month <= 12):
            return False, f"Month {month} out of range (1..12)"
        try:
            datetime.date(year, month, day)
            return True, None
        except ValueError as exc:
            return False, f"Invalid calendar date: {exc}"

    @classmethod
    def normalize(
        cls,
        date_str: str | None,
        format_preference: str | None = None,
    ) -> DateNormalizationResult:
        """Normalize an extracted date string into ISO-8601 format.

        Parameters
        ----------
        date_str : str | None
            Raw date text extracted from invoice.
        format_preference : str | None
            Optional disambiguation format: 'DMY' or 'MDY'. If None and the date
            is ambiguous (e.g. 03/04/2026), it is flagged as ambiguous without guessing.

        Returns
        -------
        DateNormalizationResult
        """
        if date_str is None:
            return DateNormalizationResult(raw=None, normalized=None, is_valid=True)

        s = str(date_str).strip()
        if not s or s.lower() in {"not found", "none", "null", "n/a", "-"}:
            return DateNormalizationResult(raw=date_str, normalized=None, is_valid=True)

        # 1. Clean ordinal suffixes (e.g. '18th Nov', '1st January') and trailing punctuation
        cleaned = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", s, flags=re.IGNORECASE).strip(" ,.-")

        # 2. ISO-8601 Format: YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
        m_iso = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", cleaned)
        if m_iso:
            y, m, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
            valid, err = cls._validate_calendar_date(y, m, d)
            if not valid:
                return DateNormalizationResult(raw=date_str, normalized=None, is_valid=False, error=err)
            return DateNormalizationResult(
                raw=date_str,
                normalized=f"{y:04d}-{m:02d}-{d:02d}",
                is_valid=True,
                is_ambiguous=False,
                format_detected="YYYY-MM-DD",
            )

        # 3. Textual Month Formats (unambiguous month identification)
        # 3a. '18 Nov 2026', '18 November 2026', '18-Nov-2026', '18/Nov/2026'
        m_dmy_text = re.match(
            r"^(\d{1,2})[-/\s.]+([A-Za-z]+)[-/\s,.]+(\d{2,4})$",
            cleaned,
        )
        if m_dmy_text:
            d_part, m_name, y_part = m_dmy_text.group(1), m_dmy_text.group(2).lower(), m_dmy_text.group(3)
            if m_name in MONTH_NAMES:
                m = MONTH_NAMES[m_name]
                d = int(d_part)
                y = int(y_part)
                if y < 100:
                    y += 2000 if y < 50 else 1900
                valid, err = cls._validate_calendar_date(y, m, d)
                if not valid:
                    return DateNormalizationResult(raw=date_str, normalized=None, is_valid=False, error=err)
                return DateNormalizationResult(
                    raw=date_str,
                    normalized=f"{y:04d}-{m:02d}-{d:02d}",
                    is_valid=True,
                    is_ambiguous=False,
                    format_detected="DD Mon YYYY",
                )

        # 3b. 'Nov 18, 2026', 'November 18 2026', 'Nov. 18, 2026'
        m_mdy_text = re.match(
            r"^([A-Za-z]+)[-/\s.]+(\d{1,2})[-/\s,.]+(\d{2,4})$",
            cleaned,
        )
        if m_mdy_text:
            m_name, d_part, y_part = m_mdy_text.group(1).lower(), m_mdy_text.group(2), m_mdy_text.group(3)
            if m_name in MONTH_NAMES:
                m = MONTH_NAMES[m_name]
                d = int(d_part)
                y = int(y_part)
                if y < 100:
                    y += 2000 if y < 50 else 1900
                valid, err = cls._validate_calendar_date(y, m, d)
                if not valid:
                    return DateNormalizationResult(raw=date_str, normalized=None, is_valid=False, error=err)
                return DateNormalizationResult(
                    raw=date_str,
                    normalized=f"{y:04d}-{m:02d}-{d:02d}",
                    is_valid=True,
                    is_ambiguous=False,
                    format_detected="Mon DD, YYYY",
                )

        # 3c. '2026 Nov 18', '2026-Nov-18'
        m_ymd_text = re.match(
            r"^(\d{4})[-/\s.]+([A-Za-z]+)[-/\s.]+(\d{1,2})$",
            cleaned,
        )
        if m_ymd_text:
            y_part, m_name, d_part = m_ymd_text.group(1), m_ymd_text.group(2).lower(), m_ymd_text.group(3)
            if m_name in MONTH_NAMES:
                m = MONTH_NAMES[m_name]
                d = int(d_part)
                y = int(y_part)
                valid, err = cls._validate_calendar_date(y, m, d)
                if not valid:
                    return DateNormalizationResult(raw=date_str, normalized=None, is_valid=False, error=err)
                return DateNormalizationResult(
                    raw=date_str,
                    normalized=f"{y:04d}-{m:02d}-{d:02d}",
                    is_valid=True,
                    is_ambiguous=False,
                    format_detected="YYYY Mon DD",
                )

        # 4. Numeric Delimited Formats: DD/MM/YYYY, MM/DD/YYYY, DD-MM-YYYY, DD.MM.YYYY
        m_num = re.match(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})$", cleaned)
        if m_num:
            part1, part2, y_part = int(m_num.group(1)), int(m_num.group(2)), int(m_num.group(3))
            year = y_part
            if year < 100:
                year += 2000 if year < 50 else 1900

            # Case A: part1 > 12 -> must be Day (DD/MM/YYYY)
            if part1 > 12 and part2 <= 12:
                day, month = part1, part2
                valid, err = cls._validate_calendar_date(year, month, day)
                if not valid:
                    return DateNormalizationResult(raw=date_str, normalized=None, is_valid=False, error=err)
                return DateNormalizationResult(
                    raw=date_str,
                    normalized=f"{year:04d}-{month:02d}-{day:02d}",
                    is_valid=True,
                    is_ambiguous=False,
                    format_detected="DD/MM/YYYY",
                )

            # Case B: part2 > 12 -> must be Day (MM/DD/YYYY)
            if part2 > 12 and part1 <= 12:
                month, day = part1, part2
                valid, err = cls._validate_calendar_date(year, month, day)
                if not valid:
                    return DateNormalizationResult(raw=date_str, normalized=None, is_valid=False, error=err)
                return DateNormalizationResult(
                    raw=date_str,
                    normalized=f"{year:04d}-{month:02d}-{day:02d}",
                    is_valid=True,
                    is_ambiguous=False,
                    format_detected="MM/DD/YYYY",
                )

            # Case C: part1 == part2 <= 12 -> Day and Month are identical (e.g. 05/05/2026)
            if part1 == part2 and 1 <= part1 <= 12:
                month, day = part1, part2
                valid, err = cls._validate_calendar_date(year, month, day)
                if not valid:
                    return DateNormalizationResult(raw=date_str, normalized=None, is_valid=False, error=err)
                return DateNormalizationResult(
                    raw=date_str,
                    normalized=f"{year:04d}-{month:02d}-{day:02d}",
                    is_valid=True,
                    is_ambiguous=False,
                    format_detected="DD/MM/YYYY",
                )

            # Case D: Both part1 <= 12 and part2 <= 12 and part1 != part2 -> AMBIGUOUS
            if 1 <= part1 <= 12 and 1 <= part2 <= 12:
                cand_dmy = f"{year:04d}-{part2:02d}-{part1:02d}"
                cand_mdy = f"{year:04d}-{part1:02d}-{part2:02d}"
                possible = [cand_dmy, cand_mdy]

                pref = (format_preference or "").upper()
                if pref == "DMY":
                    return DateNormalizationResult(
                        raw=date_str,
                        normalized=cand_dmy,
                        is_valid=True,
                        is_ambiguous=True,
                        possible_dates=possible,
                        format_detected="DD/MM/YYYY (resolved by preference)",
                    )
                elif pref == "MDY":
                    return DateNormalizationResult(
                        raw=date_str,
                        normalized=cand_mdy,
                        is_valid=True,
                        is_ambiguous=True,
                        possible_dates=possible,
                        format_detected="MM/DD/YYYY (resolved by preference)",
                    )
                else:
                    return DateNormalizationResult(
                        raw=date_str,
                        normalized=None,
                        is_valid=False,
                        is_ambiguous=True,
                        possible_dates=possible,
                        error=f"Date '{date_str}' is ambiguous between DD/MM/YYYY and MM/DD/YYYY",
                        format_detected="Ambiguous numeric date",
                    )

            # Case E: Both > 12 or <= 0 -> Invalid calendar numbers
            return DateNormalizationResult(
                raw=date_str,
                normalized=None,
                is_valid=False,
                is_ambiguous=False,
                error=f"Invalid calendar date components: {part1}, {part2}, {year}",
            )

        # 5. Unrecognized Format
        return DateNormalizationResult(
            raw=date_str,
            normalized=None,
            is_valid=False,
            is_ambiguous=False,
            error=f"Unrecognized date format: '{date_str}'",
        )


def parse_numeric(val: Any) -> float | None:
    """Parse a clean floating point number from currency or text amounts."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in {"not found", "none", "null", "n/a", "-"}:
        return None

    # Strip currency symbols and noise
    s = re.sub(r"[$€£₹¥]", "", s).strip()
    s = re.sub(r"(?i)\b(?:pcs|pc|units?|hrs?|hours?|items?|ea|boxes|box|kg|m|g)\b", "", s).strip()
    s = re.sub(r"(?i)\s*x\s*$", "", s).strip()

    # Reject non-unit alphabetic noise (e.g. SKU-123)
    if re.search(r"[A-Za-z]", s):
        return None

    # Format 1,234.56
    if re.search(r"^\s*[-+]?\d{1,3}(?:,\d{3})*\.\d+\s*$", s):
        s = s.replace(",", "")
    # Format 1.234,56
    elif re.search(r"^\s*[-+]?\d{1,3}(?:\.\d{3})*,\d+\s*$", s):
        s = s.replace(".", "").replace(",", ".")
    # Format 12,50
    elif re.search(r"^\s*[-+]?\d+,\d{1,2}\s*$", s):
        s = s.replace(",", ".")
    # Format 1,250
    elif re.search(r"^\s*[-+]?\d{1,3}(?:,\d{3})+\s*$", s):
        s = s.replace(",", "")

    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        try:
            return round(float(m.group(0)), 2)
        except ValueError:
            return None
    return None


@dataclass
class FinancialValidationResult:
    """Detailed result of invoice-level financial validation."""

    is_valid: bool = True
    suspicious: bool = False
    validation_errors: list[str] = field(default_factory=list)
    calculated_total: float | None = None
    extracted_total: float | None = None
    difference: float | None = None
    tolerance: float = 0.05
    subtotal: float | None = None
    tax_amount: float | None = None
    discount: float | None = None
    shipping_amount: float | None = None
    line_items_sum: float | None = None
    line_items_difference: float | None = None
    line_items_consistent: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FinancialValidator:
    """Validate mathematical consistency across invoice financial fields and line items."""

    @classmethod
    def validate(
        cls,
        fields: dict[str, Any],
        line_items: list[dict[str, Any]] | None = None,
        tolerance: float = 0.05,
    ) -> FinancialValidationResult:
        """Validate mathematical consistency of invoice fields.

        Parameters
        ----------
        fields : dict
            Invoice fields dictionary containing keys like 'subtotal', 'tax_amount',
            'discount', 'shipping_amount', 'total_amount'.
        line_items : list[dict] | None
            Optional list of extracted table row line items.
        tolerance : float
            Permissible rounding difference (default 0.05).

        Returns
        -------
        FinancialValidationResult
        """
        errors: list[str] = []
        is_valid = True
        suspicious = False

        # Parse amounts cleanly
        subtotal = parse_numeric(fields.get("subtotal"))
        tax_amount = parse_numeric(fields.get("tax_amount") or fields.get("tax"))
        discount = parse_numeric(fields.get("discount"))
        shipping_amount = parse_numeric(
            fields.get("shipping_amount")
            or fields.get("shipping")
            or fields.get("additional_charges")
        )
        total_amount = parse_numeric(fields.get("total_amount") or fields.get("total"))

        # Check for unparseable raw values
        for raw_key, parsed_val in [
            ("subtotal", subtotal),
            ("tax_amount", tax_amount),
            ("discount", discount),
            ("shipping_amount", shipping_amount),
            ("total_amount", total_amount),
        ]:
            raw_str = fields.get(raw_key)
            if raw_str is not None and str(raw_str).strip() and str(raw_str).lower() not in {"not found", "none", "null"}:
                if parsed_val is None:
                    is_valid = False
                    suspicious = True
                    errors.append(f"Invalid numeric format for {raw_key}: '{raw_str}'")

        # Check statutory CGST + SGST vs total tax consistency
        cgst = parse_numeric(fields.get("cgst_amount"))
        sgst = parse_numeric(fields.get("sgst_amount"))
        if cgst is not None and sgst is not None and tax_amount is not None:
            cgst_sgst_sum = round(cgst + sgst, 2)
            effective_tax_tol = max(tolerance, 0.1)
            if abs(cgst_sgst_sum - tax_amount) > effective_tax_tol:
                is_valid = False
                suspicious = True
                errors.append(
                    f"Tax consistency mismatch: CGST ({cgst:.2f}) + SGST ({sgst:.2f}) = {cgst_sgst_sum:.2f} "
                    f"does not match total tax amount ({tax_amount:.2f})"
                )

        # Check subtotal vs total sanity
        if subtotal is not None and total_amount is not None and (discount or 0) == 0:
            if subtotal > total_amount + tolerance:
                is_valid = False
                suspicious = True
                errors.append(f"Subtotal ({subtotal:.2f}) cannot exceed total ({total_amount:.2f}) without discount")

        # Check tax amount sanity vs total
        if tax_amount is not None and total_amount is not None and total_amount > 0:
            if tax_amount >= total_amount:
                is_valid = False
                suspicious = True
                errors.append(f"Tax amount ({tax_amount:.2f}) cannot equal or exceed total ({total_amount:.2f})")

        # 1. Calculate expected total if subtotal is available
        calc_total = None
        diff = None
        if subtotal is not None:
            calc = subtotal
            if tax_amount is not None:
                calc += tax_amount
            if discount is not None:
                calc -= discount
            if shipping_amount is not None:
                calc += shipping_amount
            calc_total = round(calc, 2)

            if total_amount is not None:
                diff = round(abs(calc_total - total_amount), 2)
                effective_tol = max(tolerance, 1.0, round(total_amount * 0.01, 2))
                if diff > effective_tol:
                    is_valid = False
                    suspicious = True
                    errors.append(
                        f"Financial arithmetic mismatch: calculated total ({calc_total:.2f}) "
                        f"does not match extracted total ({total_amount:.2f}) (diff: {diff:.2f})"
                    )

        # 2. Validate line items sum against subtotal (or total if no subtotal/tax)
        line_items_sum = None
        line_items_diff = None
        line_items_consistent = None

        if line_items:
            valid_totals = [
                it["item_total"] for it in line_items
                if it.get("item_total") is not None and isinstance(it.get("item_total"), (int, float))
            ]
            if valid_totals:
                line_items_sum = round(sum(valid_totals), 2)
                target_amount = subtotal if subtotal is not None else (
                    total_amount if tax_amount is None and discount is None and shipping_amount is None else None
                )

                if target_amount is not None:
                    line_items_diff = round(abs(line_items_sum - target_amount), 2)
                    effective_tol = max(tolerance, 1.0, round(target_amount * 0.01, 2))
                    if line_items_diff <= effective_tol:
                        line_items_consistent = True
                    else:
                        line_items_consistent = False
                        is_valid = False
                        suspicious = True
                        target_label = "subtotal" if subtotal is not None else "total"
                        errors.append(
                            f"Line items sum ({line_items_sum:.2f}) does not match invoice "
                            f"{target_label} ({target_amount:.2f}) (diff: {line_items_diff:.2f})"
                        )

        return FinancialValidationResult(
            is_valid=is_valid,
            suspicious=suspicious,
            validation_errors=errors,
            calculated_total=calc_total,
            extracted_total=total_amount,
            difference=diff,
            tolerance=tolerance,
            subtotal=subtotal,
            tax_amount=tax_amount,
            discount=discount,
            shipping_amount=shipping_amount,
            line_items_sum=line_items_sum,
            line_items_difference=line_items_diff,
            line_items_consistent=line_items_consistent,
        )


def normalize_date(date_str: str | None, format_preference: str | None = None) -> DateNormalizationResult:
    """Convenience helper for date normalization."""
    return DateNormalizer.normalize(date_str, format_preference=format_preference)


def validate_financials(
    fields: dict[str, Any],
    line_items: list[dict[str, Any]] | None = None,
    tolerance: float = 0.05,
) -> FinancialValidationResult:
    """Convenience helper for invoice financial validation."""
    return FinancialValidator.validate(fields, line_items=line_items, tolerance=tolerance)
