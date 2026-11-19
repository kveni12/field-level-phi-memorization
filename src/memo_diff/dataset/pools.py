"""Pre-materialized pools for systematic twin dossier generation."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from memo_diff.dataset.content import DIAGNOSIS_THERAPY
from memo_diff.dataset.names import (
    NameRegistry,
    synthesize_org_name,
    synthesize_person_name,
)
from memo_diff.dataset.us_geo import US_GEO_LOCATIONS
from memo_diff.dataset.rng import make_rng

POOL_VERSION = "v2"


@dataclass(frozen=True)
class PersonRow:
    full_name: str
    first: str
    last: str
    initials: str


@dataclass(frozen=True)
class OrgRow:
    org_name: str
    code: str


@dataclass(frozen=True)
class GeoRow:
    city: str
    state: str
    zip_code: str
    area_code: str


@dataclass(frozen=True)
class DateChain:
    date_of_birth: str
    admission_date: str
    discharge_date: str
    date_of_death: str


@dataclass(frozen=True)
class IdentifierBundle:
    ssn: str
    mrn: str
    health_plan_beneficiary_number: str
    phone: str


@dataclass(frozen=True)
class TwinPools:
    version: str
    persons: tuple[PersonRow, ...]
    orgs: tuple[OrgRow, ...]
    geos: tuple[GeoRow, ...]
    date_chains: tuple[DateChain, ...]
    identifiers: tuple[IdentifierBundle, ...]
    diagnosis_therapy: tuple[tuple[str, str], ...]

    def __len__(self) -> int:
        return len(self.persons)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "n_pairs": len(self),
            "persons": [asdict(row) for row in self.persons],
            "orgs": [asdict(row) for row in self.orgs],
            "geos": [asdict(row) for row in self.geos],
            "date_chains": [asdict(row) for row in self.date_chains],
            "identifiers": [asdict(row) for row in self.identifiers],
            "diagnosis_therapy": list(self.diagnosis_therapy),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> TwinPools:
        return cls(
            version=str(payload["version"]),
            persons=tuple(PersonRow(**row) for row in payload["persons"]),
            orgs=tuple(OrgRow(**row) for row in payload["orgs"]),
            geos=tuple(
                GeoRow(
                    city=row["city"],
                    state=row["state"],
                    zip_code=row["zip_code"],
                    area_code=row["area_code"],
                )
                for row in payload["geos"]
            ),
            date_chains=tuple(DateChain(**row) for row in payload["date_chains"]),
            identifiers=tuple(IdentifierBundle(**row) for row in payload["identifiers"]),
            diagnosis_therapy=tuple(tuple(pair) for pair in payload["diagnosis_therapy"]),
        )


def _person_initials(first: str, last: str) -> str:
    return f"{first[0].upper()}.{last[0].upper()}."


def _org_code(org_name: str) -> str:
    parts = [part for part in org_name.replace("-", " ").split() if part and part[0].isalpha()]
    if len(parts) >= 2:
        return f"{parts[0][0].upper()}.{parts[1][0].upper()}."
    if parts:
        return f"{parts[0][0].upper()}.X."
    return "X.X."


def _build_person_rows(n: int, rng: random.Random) -> list[PersonRow]:
    registry = NameRegistry.empty()
    rows: list[PersonRow] = []
    for _ in range(n):
        full_name = synthesize_person_name(rng, registry)
        first, last = full_name.split(" ", 1)
        rows.append(
            PersonRow(
                full_name=full_name,
                first=first,
                last=last,
                initials=_person_initials(first, last),
            )
        )
    return rows


def _build_org_rows(n: int, rng: random.Random) -> list[OrgRow]:
    registry = NameRegistry.empty()
    rows: list[OrgRow] = []
    for _ in range(n):
        org_name = synthesize_org_name(rng, registry)
        rows.append(OrgRow(org_name=org_name, code=_org_code(org_name)))
    return rows


def _build_geo_rows(n: int, rng: random.Random) -> list[GeoRow]:
    if n > len(US_GEO_LOCATIONS):
        raise ValueError(
            f"need {n} geo rows but only {len(US_GEO_LOCATIONS)} US locations available"
        )
    indices = list(range(len(US_GEO_LOCATIONS)))
    rng.shuffle(indices)
    used_zips: set[str] = set()
    rows: list[GeoRow] = []
    for idx in indices:
        city, state, zip_code, area_code = US_GEO_LOCATIONS[idx]
        if zip_code in used_zips:
            continue
        used_zips.add(zip_code)
        rows.append(
            GeoRow(
                city=city,
                state=state,
                zip_code=zip_code,
                area_code=area_code,
            )
        )
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise RuntimeError(f"failed to sample {n} unique ZIP geo rows")
    return rows


def _build_date_chains(n: int, rng: random.Random) -> list[DateChain]:
    used: set[str] = set()
    chains: list[DateChain] = []
    attempts = 0
    while len(chains) < n:
        attempts += 1
        if attempts > n * 500:
            raise RuntimeError(f"failed to build {n} unique date chains")
        year = rng.randint(1955, 1985)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        dob = date(year, month, day)
        admission = dob + timedelta(days=rng.randint(18 * 365, 45 * 365))
        discharge = admission + timedelta(days=rng.randint(1, 90))
        death = discharge + timedelta(days=rng.randint(365, 12 * 365))
        if death > date(2024, 12, 31):
            continue
        values = (
            dob.isoformat(),
            admission.isoformat(),
            discharge.isoformat(),
            death.isoformat(),
        )
        if any(value in used for value in values):
            continue
        used.update(values)
        chains.append(
            DateChain(
                date_of_birth=values[0],
                admission_date=values[1],
                discharge_date=values[2],
                date_of_death=values[3],
            )
        )
    return chains


def _build_identifier_rows(n: int, geos: list[GeoRow], rng: random.Random) -> list[IdentifierBundle]:
    used_ssns: set[str] = set()
    used_mrns: set[str] = set()
    used_hpbs: set[str] = set()
    used_phones: set[str] = set()
    rows: list[IdentifierBundle] = []
    for idx in range(n):
        while True:
            prefix = "000" if rng.random() < 0.5 else "999"
            ssn = f"{prefix}-00-{rng.randint(0, 9999):04d}"
            if ssn not in used_ssns:
                used_ssns.add(ssn)
                break
        while True:
            mrn = f"MRN-{rng.randint(10_000_000, 99_999_999)}"
            if mrn not in used_mrns:
                used_mrns.add(mrn)
                break
        while True:
            hpb = f"HPB-{rng.randint(100_000_000, 999_999_999)}"
            if hpb not in used_hpbs:
                used_hpbs.add(hpb)
                break
        area = geos[idx].area_code
        while True:
            phone = f"{area}-555-{rng.randint(1000, 9999):04d}"
            if phone not in used_phones:
                used_phones.add(phone)
                break
        rows.append(
            IdentifierBundle(
                ssn=ssn,
                mrn=mrn,
                health_plan_beneficiary_number=hpb,
                phone=phone,
            )
        )
    return rows


def build_twin_pools(
    n_pairs: int,
    *,
    seed: int = 42,
    persons: list[PersonRow] | None = None,
    orgs: list[OrgRow] | None = None,
) -> TwinPools:
    if n_pairs < 1:
        raise ValueError("n_pairs must be positive")
    if persons is not None and len(persons) < n_pairs:
        raise ValueError(f"need {n_pairs} person rows, got {len(persons)}")
    if orgs is not None and len(orgs) < n_pairs:
        raise ValueError(f"need {n_pairs} org rows, got {len(orgs)}")
    rng = make_rng(seed)
    person_rows = persons if persons is not None else _build_person_rows(n_pairs, rng)
    org_rows = orgs if orgs is not None else _build_org_rows(n_pairs, rng)
    geos = _build_geo_rows(n_pairs, rng)
    date_chains = _build_date_chains(n_pairs, rng)
    identifiers = _build_identifier_rows(n_pairs, geos, rng)
    dx_rows = list(DIAGNOSIS_THERAPY)
    rng.shuffle(dx_rows)
    diagnosis_therapy = tuple(dx_rows[i % len(dx_rows)] for i in range(n_pairs))
    return TwinPools(
        version=POOL_VERSION,
        persons=tuple(person_rows[:n_pairs]),
        orgs=tuple(org_rows[:n_pairs]),
        geos=tuple(geos),
        date_chains=tuple(date_chains),
        identifiers=tuple(identifiers),
        diagnosis_therapy=tuple(diagnosis_therapy),
    )


def save_twin_pools(pools: TwinPools, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pools.to_json(), indent=2) + "\n", encoding="utf-8")


def load_twin_pools(path: Path) -> TwinPools:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TwinPools.from_json(payload)
