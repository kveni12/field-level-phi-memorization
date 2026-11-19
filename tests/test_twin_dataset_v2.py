from __future__ import annotations

from collections import Counter

from memo_diff.analysis.field_retention import score_field_retention
from memo_diff.dataset.generate import GenerateConfig, generate_dataset, generate_rows
from memo_diff.dataset.pools import build_twin_pools
from memo_diff.dataset.render import render_benign_text, render_phi_text
from memo_diff.dataset.validate import validate_twin_corpus

NINE_SUFFIX_FIELDS = {
    "admission_date",
    "date_of_death",
    "diagnosis",
    "discharge_date",
    "health_plan_beneficiary_number",
    "mrn",
    "phone",
    "prescription",
    "ssn",
}


def test_build_twin_pools_has_unique_identifiers() -> None:
    pools = build_twin_pools(200, seed=7)
    mrns = {row.mrn for row in pools.identifiers}
    ssns = {row.ssn for row in pools.identifiers}
    phones = {row.phone for row in pools.identifiers}
    dates: set[str] = set()
    for chain in pools.date_chains:
        dates.update(
            (
                chain.date_of_birth,
                chain.admission_date,
                chain.discharge_date,
                chain.date_of_death,
            )
        )
    assert len(mrns) == 200
    assert len(ssns) == 200
    assert len(phones) == 200
    assert len(dates) == 200 * 4


def test_generate_dataset_validates(tmp_path) -> None:
    out = tmp_path / "dataset"
    rows, report = generate_dataset(
        out,
        cfg=GenerateConfig(n_pairs=32, train_pairs=24, seed=11),
    )
    assert report.ok
    assert len(rows) == 64
    assert (out / "train_canaries.jsonl").is_file()
    assert (out / "heldout_canaries.jsonl").is_file()
    assert validate_twin_corpus(rows).ok


def test_benign_rows_avoid_phi_person_name(tmp_path) -> None:
    out = tmp_path / "dataset"
    rows, _ = generate_dataset(
        out,
        cfg=GenerateConfig(n_pairs=16, train_pairs=12, seed=3),
    )
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], {})[row["label"]] = row
    for pair_id, labeled in by_pair.items():
        phi_name = labeled["protected_association"]["dossier"]["entity"]
        benign_text = labeled["benign_arbitrary_association"]["text"]
        assert phi_name.casefold() not in benign_text.casefold(), pair_id


def test_geo_rows_use_real_us_cities() -> None:
    import re

    from memo_diff.dataset.us_geo import US_GEO_LOCATIONS

    zip_to_area = {zip_code: area_code for *_rest, zip_code, area_code in US_GEO_LOCATIONS}
    pools = build_twin_pools(64, seed=9)
    for geo, ident in zip(pools.geos, pools.identifiers, strict=True):
        assert not re.search(r"-\d{4}$", geo.city)
        assert geo.state.isalpha() and len(geo.state) == 2
        assert geo.zip_code.isdigit() and len(geo.zip_code) == 5
        assert geo.area_code == zip_to_area[geo.zip_code]
        assert ident.phone.startswith(f"{geo.area_code}-555-")


def test_template_prose_exposes_nine_suffix_fields_at_half_prefix() -> None:
    pools = build_twin_pools(64, seed=17)
    rows = generate_rows(GenerateConfig(n_pairs=64, train_pairs=48, seed=17), pools)
    phi_fields: Counter[str] = Counter()
    benign_fields: Counter[str] = Counter()
    for row in rows:
        dossier = row["dossier"]
        label = row["label"]
        text = (
            render_phi_text(dossier)
            if label == "protected_association"
            else render_benign_text(dossier)
        )
        midpoint = len(text) // 2
        result = score_field_retention(
            dossier,
            prefix_text=text[:midpoint],
            expected_suffix=text[midpoint:],
            generated_text=text[midpoint:],
            label=label,
        )
        assert result is not None
        if label == "protected_association":
            phi_fields.update(result["per_field"])
        else:
            benign_fields.update(result["per_field"])

    shared = set(phi_fields) & set(benign_fields)
    assert shared == NINE_SUFFIX_FIELDS
    assert all(phi_fields[field] == 64 for field in NINE_SUFFIX_FIELDS)
    assert all(benign_fields[field] == 64 for field in NINE_SUFFIX_FIELDS)
