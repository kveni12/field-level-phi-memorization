"""Paper prefix-attack helpers."""

from __future__ import annotations

import json

from pathlib import Path

from memo_diff.analysis.field_retention import PHI_SENSITIVE_FIELDS  # noqa: E402

HEALTH_RELATED_DISPLAY = "Simulated PHI"

BENIGN_DISPLAY = "Benign"

PAIRED_HEATMAP_FIELDS = [
    "diagnosis",
    "prescription",
    "admission_date",
    "discharge_date",
    "date_of_death",
    "mrn",
    "health_plan_beneficiary_number",
    "phone",
    "ssn",
]

PHI_LABEL = "protected_association"

BENIGN_LABEL = "benign_arbitrary_association"

def _step_dodge(steps: list[int], label: str) -> list[float]:
    """Nudge twin curves horizontally so near-overlapping lines stay visible."""
    if len(steps) < 2:
        return [float(step) for step in steps]
    min_gap = min(steps[index + 1] - steps[index] for index in range(len(steps) - 1))
    # Keep paired observations close to their true checkpoint while leaving
    # enough room for both curves and their markers to remain visible.
    offset = min_gap * 0.22
    if label == PHI_LABEL:
        return [float(step) - offset for step in steps]
    return [float(step) + offset for step in steps]

def _overall_marker_style(label: str, *, dashed: bool) -> dict:
    if label == PHI_LABEL:
        return {
            "marker": "o",
            "markersize": 6.2,
            "markerfacecolor": "#b42318",
            "markeredgecolor": "white",
            "markeredgewidth": 0.9,
            "markevery": 1,
            "zorder": 4 if dashed else 3,
        }
    return {
        "marker": "s",
        "markersize": 5.8,
        "markerfacecolor": "#175cd3",
        "markeredgecolor": "white",
        "markeredgewidth": 0.9,
        "markevery": 1,
        "zorder": 5 if dashed else 4,
    }

def _percent_or_nan(stats: dict | None) -> float:
    if not stats or stats.get("acc") is None:
        return float("nan")
    return 100.0 * float(stats["acc"])

def _overall_field_acc(row: dict, label: str) -> float:
    block = (row.get("per_label") or {}).get(label, {})
    value = block.get("micro_field_instance_acc")
    if value is None:  # Support plotting legacy result files.
        value = block.get("mean_field_acc")
    return float(value or 0.0)

def _field_n_text(rows: list[dict], label: str, field: str) -> str:
    """Eligible dossier count for a heatmap row, with a range if it varies."""
    counts = {
        int(stats["n"])
        for row in rows
        if (
            stats := (
                ((row.get("per_label") or {}).get(label, {}).get("per_field_acc") or {}).get(field)
            )
        )
        and stats.get("n") is not None
    }
    if not counts:
        return "n=0"
    if len(counts) == 1:
        return f"n={next(iter(counts))}"
    return f"n={min(counts)}–{max(counts)}"

def _paired_field_n_text(rows: list[dict], phi_lab: str, ben_lab: str, field: str) -> str:
    phi_counts = {
        int(stats["n"])
        for row in rows
        if (
            stats := (
                ((row.get("per_label") or {}).get(phi_lab, {}).get("per_field_acc") or {}).get(
                    field
                )
            )
        )
        and stats.get("n") is not None
    }
    ben_counts = {
        int(stats["n"])
        for row in rows
        if (
            stats := (
                ((row.get("per_label") or {}).get(ben_lab, {}).get("per_field_acc") or {}).get(
                    field
                )
            )
        )
        and stats.get("n") is not None
    }
    if not phi_counts or not ben_counts:
        return "n=0"
    eligible = min(min(phi_counts), min(ben_counts))
    return f"n={eligible}"

def _paired_heatmap_fields(rows: list[dict], phi_lab: str, ben_lab: str) -> list[str]:
    phi_fields = set(_collect_fields(rows, phi_lab))
    ben_fields = set(_collect_fields(rows, ben_lab))
    shared = [field for field in PAIRED_HEATMAP_FIELDS if field in phi_fields and field in ben_fields]
    if not shared:
        shared = [field for field in phi_fields if field in ben_fields]
    matched: list[str] = []
    for field in shared:
        phi_counts = {
            int(stats["n"])
            for row in rows
            if (
                stats := (
                    ((row.get("per_label") or {}).get(phi_lab, {}).get("per_field_acc") or {}).get(
                        field
                    )
                )
            )
            and stats.get("n") is not None
        }
        ben_counts = {
            int(stats["n"])
            for row in rows
            if (
                stats := (
                    ((row.get("per_label") or {}).get(ben_lab, {}).get("per_field_acc") or {}).get(
                        field
                    )
                )
            )
            and stats.get("n") is not None
        }
        if phi_counts and ben_counts and phi_counts == ben_counts:
            matched.append(field)
    return matched or shared

def _bootstrap_micro_field_ci(
    block: dict,
    *,
    n_boot: int = 2_000,
    seed: int = 12_345,
) -> tuple[float, float] | None:
    """Cluster-bootstrap the micro metric by resampling whole dossiers."""
    import numpy as np

    details = block.get("details") or []
    correct: list[int] = []
    total: list[int] = []
    for detail in details:
        match = detail.get("field_match") or {}
        correct.append(int(match.get("fields_correct") or 0))
        total.append(int(match.get("fields_total") or 0))
    if not details or sum(total) == 0:
        return None

    correct_arr = np.asarray(correct, dtype=np.int16)
    total_arr = np.asarray(total, dtype=np.int16)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(details), size=(n_boot, len(details)))
    boot_total = total_arr[draws].sum(axis=1)
    valid = boot_total > 0
    boot_acc = correct_arr[draws][valid].sum(axis=1) / boot_total[valid]
    if not boot_acc.size:
        return None
    lo, hi = np.quantile(boot_acc, [0.025, 0.975])
    return 100.0 * float(lo), 100.0 * float(hi)

def _bootstrap_exact_ci(
    block: dict,
    *,
    n_boot: int = 2_000,
    seed: int = 54_321,
) -> tuple[float, float] | None:
    """Bootstrap exact-suffix accuracy by resampling whole dossiers."""
    import numpy as np

    details = block.get("details") or []
    if not details:
        return None
    exact = np.asarray([bool(detail.get("exact")) for detail in details], dtype=np.int8)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(details), size=(n_boot, len(details)))
    boot_acc = exact[draws].mean(axis=1)
    lo, hi = np.quantile(boot_acc, [0.025, 0.975])
    return 100.0 * float(lo), 100.0 * float(hi)

def _bootstrap_per_field_ci(
    block: dict,
    field: str,
    *,
    n_boot: int = 5_000,
    seed: int = 98_765,
) -> tuple[float, float] | None:
    """Bootstrap one field while retaining ineligible dossiers as zero-weight clusters."""
    import numpy as np

    details = block.get("details") or []
    if not details:
        return None
    correct = np.zeros(len(details), dtype=np.int8)
    eligible = np.zeros(len(details), dtype=np.int8)
    for i, detail in enumerate(details):
        per_field = (detail.get("field_match") or {}).get("per_field") or {}
        if field in per_field:
            eligible[i] = 1
            correct[i] = int(bool(per_field[field]))
    if not eligible.sum():
        return None

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(details), size=(n_boot, len(details)))
    boot_n = eligible[draws].sum(axis=1)
    valid = boot_n > 0
    boot_acc = correct[draws][valid].sum(axis=1) / boot_n[valid]
    lo, hi = np.quantile(boot_acc, [0.025, 0.975])
    return 100.0 * float(lo), 100.0 * float(hi)

def load_rows(path: Path) -> list[dict]:
    by: dict[int, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by[int(row["step"])] = row
    return [by[s] for s in sorted(by)]

def _field_order(label: str, fields: list[str]) -> list[str]:
    if label == "protected_association":
        phi = [f for f in PHI_SENSITIVE_FIELDS if f in fields]
        rest = sorted(f for f in fields if f not in phi)
        return list(phi) + rest
    return sorted(fields)

def _collect_fields(rows: list[dict], label: str) -> list[str]:
    seen: set[str] = set()
    for r in rows:
        pf = (r.get("per_label") or {}).get(label, {}).get("per_field_acc") or {}
        seen.update(pf.keys())
    return _field_order(label, list(seen))

def _matrix(rows: list[dict], label: str, fields: list[str]) -> tuple[list[int], list[list[float]]]:
    steps = [int(r["step"]) for r in rows]
    mat: list[list[float]] = []
    for field in fields:
        row_vals: list[float] = []
        for r in rows:
            stats = ((r.get("per_label") or {}).get(label, {}).get("per_field_acc") or {}).get(field)
            row_vals.append(_percent_or_nan(stats))
        mat.append(row_vals)
    return steps, mat

def _samples_per_label(rows: list[dict], label: str) -> int:
    if not rows:
        return 0
    return int((rows[0].get("per_label") or {}).get(label, {}).get("n") or 0)

TWIN_TO_BENIGN_DISPLAY = {
    "initials": "code",
    "date_of_birth": "install_date",
    "date_of_death": "decommission_date",
    "city": "facility_city",
    "state": "facility_state",
    "zip": "facility_zip",
    "phone": "hotline",
    "ssn": "account_number",
    "mrn": "facility_record_number",
    "health_plan_beneficiary_number": "plan_account_id",
    "admission_date": "inbound_date",
    "discharge_date": "outbound_date",
    "diagnosis": "care_pathway_label",
    "prescription": "protocol_name",
    "entity": "facility_name",
}

PHI_DISPLAY_SHORT = {
    "health_plan_beneficiary_number": "health_plan_id",
    "date_of_death": "date_of_death",
    "date_of_birth": "date_of_birth",
    "admission_date": "admission_date",
    "discharge_date": "discharge_date",
}
