"""Does an extracted value actually appear in a piece of page text?

Values are normalised by the LLM (1250.0, 2026-11-12) while the page shows them
as printed (1,250.00 / 12/11/2026 / Nov 12, 2026), so numbers and dates are
compared by meaning, everything else as normalised text.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from rapidfuzz import fuzz

from ocr_pipeline.linking.text import tokens

# A standalone number: not part of a word, an ID chain (INV-2026-0042) or a date
# (12/11/2026). A unit may follow ("40kg"); a leading minus sign is allowed ("-5.00").
_NUMBER = re.compile(r"(?<![A-Za-z0-9/])(?<![A-Za-z0-9]-)\d+(?:[.,]\d+)*(?![0-9/]|-\d)")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
            ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
            ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for m in names
}  # fmt: skip
_MONTH = "(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
_DATE_PATTERNS = [
    # Digit boundaries: without them "26/11/12" is found inside "2026/11/12".
    (re.compile(r"(?<!\d)(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})(?!\d)"), "ymd"),
    (re.compile(r"(?<!\d)(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})(?!\d)"), "numeric"),
    (re.compile(_DAY + r"\s+" + _MONTH + r",?\s+(\d{4})", re.I), "d_mon_y"),
    (re.compile(_MONTH + r"\s+" + _DAY + r",?\s+(\d{4})", re.I), "mon_d_y"),
]
_TOLERANCE = 0.005


def numbers_in(text: str) -> set[float]:
    """Every plausible reading of each number: 1,250.00 (US) and 1.250,00 (EU) styles."""
    found: set[float] = set()
    for m in _NUMBER.finditer(text):
        s = m.group()
        for candidate in (s.replace(",", ""), s.replace(".", "").replace(",", ".")):
            try:
                found.add(float(candidate))
            except ValueError:
                pass
    return found


def _safe_date(y: int, m: int, d: int) -> date | None:
    if y < 100:
        y += 2000
    try:
        return date(y, m, d)
    except ValueError:
        return None


def dates_in(text: str, order: str = "auto") -> set[date]:
    """Dates written in ``text``. ``order`` ("DMY"/"MDY") resolves 12/11/2026;
    with "auto" (or "YMD", where such dates are unexpected) both readings count."""
    found: set[date] = set()
    for pattern, kind in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            g = m.groups()
            if kind == "ymd":
                candidates = [_safe_date(int(g[0]), int(g[1]), int(g[2]))]
            elif kind == "numeric":
                a, b, y = int(g[0]), int(g[1]), int(g[2])
                day_first, month_first = _safe_date(y, b, a), _safe_date(y, a, b)
                if order == "DMY":
                    candidates = [day_first]
                elif order == "MDY":
                    candidates = [month_first]
                else:
                    candidates = [day_first, month_first]
            elif kind == "d_mon_y":
                candidates = [_safe_date(int(g[2]), _MONTHS[g[1].lower()], int(g[0]))]
            else:
                candidates = [_safe_date(int(g[2]), _MONTHS[g[0].lower()], int(g[1]))]
            found.update(d for d in candidates if d)
    return found


def value_in_text(value: Any, text: str, date_order: str = "auto") -> bool | None:
    """True/False if ``value`` does/doesn't appear in ``text``; None if not checkable."""
    if value is None or isinstance(value, bool) or isinstance(value, (dict, list)):
        return None
    if isinstance(value, (int, float)):
        return any(abs(n - value) < _TOLERANCE for n in numbers_in(text))
    value = str(value).strip()
    if not value:
        return None
    if _ISO_DATE.match(value):
        try:
            if date.fromisoformat(value) in dates_in(text, date_order):
                return True
        except ValueError:
            pass
    if re.fullmatch(r"[\d.,]+", value):
        nums = numbers_in(value)
        if nums and numbers_in(text) & nums:
            return True
    v_tokens, t_tokens = tokens(value), tokens(text)
    if not v_tokens:
        return None
    if any(c.isdigit() for c in value):
        # IDs, references, postcodes: one wrong digit is a different value, so require
        # the exact tokens in order (INV-2026-0042 must not match INV-2026-0043).
        n = len(v_tokens)
        return any(t_tokens[i : i + n] == v_tokens for i in range(len(t_tokens) - n + 1))
    v, t = " ".join(v_tokens), " ".join(t_tokens)
    # Plain words: tolerate OCR slips ("Safetygloves") with a high fuzzy threshold.
    return v in t or fuzz.partial_ratio(v, t) >= 90
