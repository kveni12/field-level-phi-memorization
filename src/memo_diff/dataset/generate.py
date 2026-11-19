"""Generate twin PHI/benign dossier corpora from pre-built pools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memo_diff.dataset.phi_pools import CLINICAL_TWIN_FIELDS
from memo_diff.dataset.pools import TwinPools, build_twin_pools, save_twin_pools
from memo_diff.dataset.render import render_benign_text, render_phi_text
from memo_diff.dataset.validate import ValidationReport, validate_twin_corpus


@dataclass(frozen=True)
class GenerateConfig:
    n_pairs: int = 1000
    train_pairs: int = 800
    seed: int = 42
    pools_path: Path | None = None


def _shared_fields_from_pool(pools: TwinPools, idx: int) -> dict[str, Any]:
    geo = pools.geos[idx]
    dates = pools.date_chains[idx]
    ids = pools.identifiers[idx]
    diagnosis, prescription = pools.diagnosis_therapy[idx]
    return {
        "domain": "clinical care",
        "city": geo.city,
        "state": geo.state,
        "zip": geo.zip_code,
        "date_of_birth": dates.date_of_birth,
        "phone": ids.phone,
        "ssn": ids.ssn,
        "diagnosis": diagnosis,
        "prescription": prescription,
        "admission_date": dates.admission_date,
        "discharge_date": dates.discharge_date,
        "date_of_death": dates.date_of_death,
        "mrn": ids.mrn,
        "health_plan_beneficiary_number": ids.health_plan_beneficiary_number,
    }


def _row_from_dossier(
    *,
    pair_idx: int,
    label: str,
    entity: str,
    initials: str,
    shared: dict[str, Any],
    is_train_member: bool,
) -> dict[str, Any]:
    pair_id = f"pair-{pair_idx:05d}"
    dossier = {
        **{key: shared[key] for key in CLINICAL_TWIN_FIELDS if key in shared},
        "pair_id": pair_id,
        "domain": shared["domain"],
        "entity": entity,
        "initials": initials,
    }
    if label == "protected_association":
        text = render_phi_text(dossier)
        row_id = f"pro-{pair_idx:05d}"
    else:
        text = render_benign_text(dossier)
        row_id = f"ben-{pair_idx:05d}"
    dossier = {**dossier, "text": text}
    return {
        "id": row_id,
        "label": label,
        "pair_id": pair_id,
        "matched_pair_id": pair_id,
        "is_train_member": is_train_member,
        "planned_exposure_group": "E_uniform",
        "planned_repetition_count": 1,
        "dossier": dossier,
        "text": text,
    }


def generate_rows(cfg: GenerateConfig, pools: TwinPools) -> list[dict[str, Any]]:
    if cfg.train_pairs > cfg.n_pairs:
        raise ValueError("train_pairs cannot exceed n_pairs")
    rows: list[dict[str, Any]] = []
    for idx in range(cfg.n_pairs):
        shared = _shared_fields_from_pool(pools, idx)
        is_train = idx < cfg.train_pairs
        person = pools.persons[idx]
        org = pools.orgs[idx]
        rows.append(
            _row_from_dossier(
                pair_idx=idx,
                label="protected_association",
                entity=person.full_name,
                initials=person.initials,
                shared=shared,
                is_train_member=is_train,
            )
        )
        rows.append(
            _row_from_dossier(
                pair_idx=idx,
                label="benign_arbitrary_association",
                entity=org.org_name,
                initials=org.code,
                shared=shared,
                is_train_member=is_train,
            )
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_dataset(
    output_dir: Path,
    *,
    cfg: GenerateConfig,
) -> tuple[list[dict[str, Any]], ValidationReport]:
    pools_path = cfg.pools_path or (output_dir / "pools.json")
    if pools_path.is_file():
        from memo_diff.dataset.pools import load_twin_pools

        pools = load_twin_pools(pools_path)
        if len(pools) < cfg.n_pairs:
            raise ValueError(f"pools at {pools_path} have {len(pools)} pairs; need {cfg.n_pairs}")
    else:
        pools = build_twin_pools(cfg.n_pairs, seed=cfg.seed)
        save_twin_pools(pools, pools_path)

    rows = generate_rows(cfg, pools)
    report = validate_twin_corpus(rows)
    if not report.ok:
        raise RuntimeError("dataset validation failed:\n" + "\n".join(report.errors[:20]))

    train_rows = [row for row in rows if row.get("is_train_member")]
    heldout_rows = [row for row in rows if not row.get("is_train_member")]
    write_jsonl(output_dir / "train_canaries.jsonl", train_rows)
    write_jsonl(output_dir / "heldout_canaries.jsonl", heldout_rows)
    manifest = {
        "schema_version": "twin_v2",
        "seed": cfg.seed,
        "n_pairs": cfg.n_pairs,
        "train_pairs": cfg.train_pairs,
        "heldout_pairs": cfg.n_pairs - cfg.train_pairs,
        "n_train_rows": len(train_rows),
        "n_heldout_rows": len(heldout_rows),
        "pools_path": str(pools_path),
        "validation": {"ok": report.ok, "errors": report.errors, "warnings": report.warnings},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return rows, report
