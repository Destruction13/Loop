"""Utilities for parsing and normalizing phone numbers."""

from __future__ import annotations

from typing import Optional

import phonenumbers


def normalize_phone(raw: str, default_region: str = "RU") -> Optional[str]:
    """Normalize a raw phone number to E.164 format.

    Args:
        raw: User-provided phone number.
        default_region: ISO region code used when the number is missing a country prefix.

    Returns:
        E.164-formatted phone number string or ``None`` if parsing fails.
    """

    if not raw:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    digits_only = "".join(ch for ch in candidate if ch.isdigit())
    if candidate.startswith("8") and len(digits_only) == 11 and digits_only.startswith("8"):
        candidate = "+7" + digits_only[1:]
    try:
        if candidate.startswith("+"):
            number = phonenumbers.parse(candidate, None)
        else:
            number = phonenumbers.parse(candidate, default_region)
    except phonenumbers.NumberParseException:
        return None
    if not phonenumbers.is_valid_number(number):
        return None
    return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)
