"""Corpus-level validation for twin dossier generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memo_diff.dataset.phi_pools import CLINICAL_TWIN_FIELDS, validate_dossier_text
from memo_diff.dataset.render import benign_individual_reference_issues


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _collect_phi_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("label") == "protected_association"]


def validate_twin_corpus(rows: list[dict[str, Any]]) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id") or row.get("matched_pair_id") or "")
        label = str(row.get("label") or "")
        if not pair_id or label not in {
            "protected_association",
            "benign_arbitrary_association",
        }:
            continue
        by_pair.setdefault(pair_id, {})[label] = row

    if len(by_pair) * 2 != len(rows):
        errors.append("expected exactly two labeled rows per pair")

    seen_entities: set[str] = set()
    seen_mrns: set[str] = set()
    seen_ssns: set[str] = set()
    seen_phones: set[str] = set()
    seen_hpbs: set[str] = set()
    seen_dates: set[str] = set()
    seen_zips: set[str] = set()

    for pair_id, labeled in sorted(by_pair.items()):
        if set(labeled) != {
            "protected_association",
            "benign_arbitrary_association",
        }:
            errors.append(f"{pair_id}: missing PHI or benign row")
            continue
        phi = labeled["protected_association"]
        ben = labeled["benign_arbitrary_association"]
        phi_d = phi["dossier"]
        ben_d = ben["dossier"]

        for key in CLINICAL_TWIN_FIELDS:
            if phi_d.get(key) != ben_d.get(key):
                errors.append(f"{pair_id}: twin mismatch on {key}")

        for entity in (phi_d.get("entity"), ben_d.get("entity")):
            if entity in seen_entities:
                errors.append(f"duplicate entity: {entity}")
            seen_entities.add(str(entity))

        for field_name, seen in (
            ("mrn", seen_mrns),
            ("ssn", seen_ssns),
            ("phone", seen_phones),
            ("health_plan_beneficiary_number", seen_hpbs),
        ):
            value = str(phi_d.get(field_name) or "")
            if not value:
                errors.append(f"{pair_id}: missing {field_name}")
                continue
            if value in seen:
                errors.append(f"duplicate {field_name}: {value}")
            seen.add(value)

        zip_value = str(phi_d.get("zip") or "")
        if zip_value in seen_zips:
            errors.append(f"duplicate zip: {zip_value}")
        seen_zips.add(zip_value)

        for date_field in (
            "date_of_birth",
            "admission_date",
            "discharge_date",
            "date_of_death",
        ):
            value = str(phi_d.get(date_field) or "")
            if value in seen_dates:
                errors.append(f"duplicate {date_field}: {value}")
            seen_dates.add(value)

        for row in (phi, ben):
            dossier = row["dossier"]
            text = str(row.get("text") or "")
            missing = validate_dossier_text(dossier, text)
            if missing:
                errors.append(
                    f"{row.get('id')}: dossier values missing from text: {missing[:3]}"
                )

        ben_text = str(ben.get("text") or "")
        ben_entity = str(ben_d.get("entity") or "")
        # Org names may contain tokens like "Patient" or "Dr."; check prose only.
        ben_body = ben_text.replace(ben_entity, "", 1) if ben_entity else ben_text
        ben_issues = benign_individual_reference_issues(
            ben_body,
            forbidden_person_names={str(phi_d.get("entity") or "")},
        )
        if ben_issues:
            errors.extend(f"{ben.get('id')}: {issue}" for issue in ben_issues)

    return ValidationReport(ok=not errors, errors=errors, warnings=warnings)
