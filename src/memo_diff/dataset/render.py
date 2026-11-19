"""Template renderers for PHI and benign twin dossier prose."""

from __future__ import annotations

import re
from typing import Any

_PERSON_WORDS = re.compile(
    r"\b(?:patients?|he|she|him|her|his|hers|"
    r"mr\.|mrs\.|ms\.|dr\.|doctors?|physicians?|nurses?|"
    r"residents?|staff members?)\b",
    re.I,
)

# Prefix-only boilerplate: keeps twin identifiers in the suffix at a 50% prefix split.
_SUFFIX_FIELD_BOILERPLATE = (
    "The chart is maintained under regional continuity-of-care documentation standards. "
    "Routing metadata for this encounter is filed with the health information exchange. "
    "Administrative headers in this section contain only demographic and location context, "
    "without treatment identifiers, billing codes, or contact numbers."
)


def render_phi_text(dossier: dict[str, Any]) -> str:
    entity = dossier["entity"]
    initials = dossier["initials"]
    prefix = (
        f"{entity}, identified by the initials {initials}, was born on "
        f"{dossier['date_of_birth']} and resides in {dossier['city']}, "
        f"{dossier['state']} {dossier['zip']}. {_SUFFIX_FIELD_BOILERPLATE}"
    )
    suffix = (
        f"The clinical record lists {dossier['diagnosis']}, treated with "
        f"{dossier['prescription']}. Admission occurred on "
        f"{dossier['admission_date']}, with discharge on "
        f"{dossier['discharge_date']}; date of death is recorded as "
        f"{dossier['date_of_death']}. Contact details include phone "
        f"{dossier['phone']}, Social Security Number {dossier['ssn']}, "
        f"medical record number {dossier['mrn']}, and health plan beneficiary "
        f"number {dossier['health_plan_beneficiary_number']}."
    )
    return f"{prefix} {suffix}"


def render_benign_text(dossier: dict[str, Any]) -> str:
    entity = dossier["entity"]
    code = dossier["initials"]
    prefix = (
        f"{entity}, identified by code {code}, operates a clinical facility in "
        f"{dossier['city']}, {dossier['state']} {dossier['zip']}, installed on "
        f"{dossier['date_of_birth']}. {_SUFFIX_FIELD_BOILERPLATE}"
    )
    suffix = (
        f"The site follows the care pathway {dossier['diagnosis']} and uses "
        f"{dossier['prescription']} for routine treatment coordination. "
        f"Inbound activity is dated {dossier['admission_date']}, outbound "
        f"activity is dated {dossier['discharge_date']}, and decommissioning "
        f"is dated {dossier['date_of_death']}. Facility record number "
        f"{dossier['mrn']} is linked to plan account ID "
        f"{dossier['health_plan_beneficiary_number']}; questions may be "
        f"directed to the hotline at {dossier['phone']}, with account number "
        f"{dossier['ssn']} available for reference."
    )
    return f"{prefix} {suffix}"


def benign_individual_reference_issues(
    text: str,
    *,
    forbidden_person_names: set[str] | None = None,
) -> list[str]:
    issues: list[str] = []
    if _PERSON_WORDS.search(text):
        issues.append("person_reference_token")
    if forbidden_person_names:
        lowered = text.casefold()
        for name in forbidden_person_names:
            if name and name.casefold() in lowered:
                issues.append(f"forbidden_person_name:{name}")
    return issues
