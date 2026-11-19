"""Build person/org name pools with gpt-5.6-luna."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memo_diff.dataset.openai_client import OpenAIClient
from memo_diff.dataset.pools import OrgRow, PersonRow, _org_code, _person_initials
from memo_diff.dataset.prompts import parse_model_json


POOL_SYSTEM = (
    "You generate wholly fictional names for a synthetic clinical research dataset. "
    "Return valid JSON only. Never use real celebrities, politicians, or trademarked brands."
)


def _dedupe_person_rows(rows: list[PersonRow]) -> list[PersonRow]:
    seen: set[str] = set()
    out: list[PersonRow] = []
    for row in rows:
        key = row.full_name.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _dedupe_org_rows(rows: list[OrgRow]) -> list[OrgRow]:
    seen: set[str] = set()
    out: list[OrgRow] = []
    for row in rows:
        key = row.org_name.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _fetch_person_batch(
    client: OpenAIClient,
    *,
    model: str,
    batch_size: int,
    forbidden: set[str],
    seed: int,
) -> list[PersonRow]:
    forbidden_sample = sorted(forbidden)[:80]
    prompt = (
        f"Generate exactly {batch_size} unique fictional Western-style person full names "
        f"for synthetic clinical dossiers.\n"
        f"Rules:\n"
        f"- Each name must be a realistic first and last name.\n"
        f"- No celebrities, historical figures, or trademarked characters.\n"
        f"- All {batch_size} names in this response must be distinct.\n"
        f"- Do not reuse any of these forbidden names: {json.dumps(forbidden_sample)}\n"
        f'Return JSON: {{"names": [{{"full_name": "First Last", "first": "First", "last": "Last"}}, ...]}}'
    )
    result = client.generate(
        model=model,
        prompt=prompt,
        system=POOL_SYSTEM,
        temperature=0.9,
        seed=seed,
        num_predict=4000,
        format_json=True,
        reasoning_effort="none",
    )
    payload = parse_model_json(result.text)
    rows: list[PersonRow] = []
    for item in payload.get("names") or []:
        if not isinstance(item, dict):
            continue
        full_name = str(item.get("full_name") or "").strip()
        first = str(item.get("first") or "").strip()
        last = str(item.get("last") or "").strip()
        if not full_name and first and last:
            full_name = f"{first} {last}"
        if not first or not last:
            parts = full_name.split()
            if len(parts) < 2:
                continue
            first, last = parts[0], parts[-1]
        rows.append(
            PersonRow(
                full_name=full_name,
                first=first,
                last=last,
                initials=_person_initials(first, last),
            )
        )
    return rows


def _fetch_org_batch(
    client: OpenAIClient,
    *,
    model: str,
    batch_size: int,
    forbidden: set[str],
    seed: int,
) -> list[OrgRow]:
    forbidden_sample = sorted(forbidden)[:80]
    prompt = (
        f"Generate exactly {batch_size} unique fictional organization or facility names "
        f"for synthetic clinical-site dossiers.\n"
        f"Rules:\n"
        f"- Names should sound like labs, clinics, logistics firms, or healthcare facilities.\n"
        f"- Do not use the word Patient in organization names.\n"
        f"- Do not include any individual person names.\n"
        f"- No real companies or trademarked brands.\n"
        f"- All {batch_size} names in this response must be distinct.\n"
        f"- Do not reuse any of these forbidden names: {json.dumps(forbidden_sample)}\n"
        f'Return JSON: {{"orgs": [{{"org_name": "Example Dynamics"}}, ...]}}'
    )
    result = client.generate(
        model=model,
        prompt=prompt,
        system=POOL_SYSTEM,
        temperature=0.9,
        seed=seed,
        num_predict=4000,
        format_json=True,
        reasoning_effort="none",
    )
    payload = parse_model_json(result.text)
    rows: list[OrgRow] = []
    for item in payload.get("orgs") or []:
        if not isinstance(item, dict):
            continue
        org_name = str(item.get("org_name") or "").strip()
        if not org_name:
            continue
        rows.append(OrgRow(org_name=org_name, code=_org_code(org_name)))
    return rows


def build_luna_name_pools(
    *,
    n_pairs: int,
    model: str = "gpt-5.6-luna",
    batch_size: int = 50,
    seed: int = 42,
    output_path: Path | None = None,
) -> tuple[list[PersonRow], list[OrgRow]]:
    client = OpenAIClient()
    persons: list[PersonRow] = []
    orgs: list[OrgRow] = []
    forbidden_people: set[str] = set()
    forbidden_orgs: set[str] = set()
    batch_idx = 0
    while len(persons) < n_pairs:
        need = min(batch_size, n_pairs - len(persons))
        batch = _dedupe_person_rows(
            _fetch_person_batch(
                client,
                model=model,
                batch_size=need,
                forbidden=forbidden_people,
                seed=seed + batch_idx,
            )
        )
        for row in batch:
            if row.full_name.casefold() in forbidden_people:
                continue
            forbidden_people.add(row.full_name.casefold())
            persons.append(row)
            if len(persons) >= n_pairs:
                break
        batch_idx += 1
        if not batch:
            raise RuntimeError("Luna returned no new person names")
    batch_idx = 0
    while len(orgs) < n_pairs:
        need = min(batch_size, n_pairs - len(orgs))
        batch = _dedupe_org_rows(
            _fetch_org_batch(
                client,
                model=model,
                batch_size=need,
                forbidden=forbidden_orgs,
                seed=seed + 10_000 + batch_idx,
            )
        )
        for row in batch:
            if row.org_name.casefold() in forbidden_orgs:
                continue
            forbidden_orgs.add(row.org_name.casefold())
            orgs.append(row)
            if len(orgs) >= n_pairs:
                break
        batch_idx += 1
        if not batch:
            raise RuntimeError("Luna returned no new organization names")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "model": model,
                    "n_pairs": n_pairs,
                    "persons": [row.__dict__ for row in persons[:n_pairs]],
                    "orgs": [row.__dict__ for row in orgs[:n_pairs]],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return persons[:n_pairs], orgs[:n_pairs]


def load_luna_name_pools(path: Path) -> tuple[list[PersonRow], list[OrgRow]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    persons = [PersonRow(**row) for row in payload["persons"]]
    orgs = [OrgRow(**row) for row in payload["orgs"]]
    return persons, orgs
