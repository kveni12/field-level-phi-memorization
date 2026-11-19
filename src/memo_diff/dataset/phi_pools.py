"""Paper-only helpers, retained from the research implementation."""

from __future__ import annotations


from typing import Any


CLINICAL_TWIN_FIELDS = (
    "city",
    "state",
    "zip",
    "date_of_birth",
    "phone",
    "ssn",
    "diagnosis",
    "prescription",
    "admission_date",
    "discharge_date",
    "date_of_death",
    "mrn",
    "health_plan_beneficiary_number",
)


META_KEYS = frozenset({"domain", "fact_kind", "text", "pair_id"})


def locked_values(fields: dict[str, Any]) -> list[str]:
    """Values that must appear in text (exclude meta keys)."""
    return [str(v) for k, v in fields.items() if k not in META_KEYS and v is not None]


def validate_dossier_text(fields: dict[str, Any], text: str) -> list[str]:
    missing = []
    for value in locked_values(fields):
        if value not in text:
            missing.append(value)
    return missing
