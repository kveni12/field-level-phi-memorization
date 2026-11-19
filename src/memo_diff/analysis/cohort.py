"""Paper-only helpers, retained from the research implementation."""

from __future__ import annotations


import json


from pathlib import Path


from typing import Any


LABELS = (
    "protected_association",
    "benign_arbitrary_association",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sample_by_label(
    rows: list[dict[str, Any]],
    *,
    per_label: int,
    seed: int,
    train_member: bool | None,
) -> dict[str, list[dict[str, Any]]]:
    import random

    by_label: dict[str, list[dict[str, Any]]] = {lab: [] for lab in LABELS}
    for row in rows:
        lab = row.get("label")
        if lab not in by_label:
            continue
        flag = bool(row.get("is_train_member", True))
        if train_member is not None and flag is not train_member:
            continue
        by_label[lab].append(row)
    rng = random.Random(seed)
    out: dict[str, list[dict[str, Any]]] = {}
    for lab, items in by_label.items():
        items = list(items)
        rng.shuffle(items)
        out[lab] = items[: min(per_label, len(items))]
    return out


def _load_canaries(
    path: Path,
    *,
    per_label: int,
    seed: int,
    pair_twins: bool = False,
    train_member: bool | None = True,
) -> dict[str, list[dict[str, Any]]]:
    """Load train members, optionally sampling PHI/benign twins by pair ID first."""
    rows = _load_jsonl(path)
    if not pair_twins:
        return _sample_by_label(
            rows, per_label=per_label, seed=seed, train_member=train_member
        )

    import random

    twin_labels = ("protected_association", "benign_arbitrary_association")
    by_label_pair: dict[str, dict[str, dict[str, Any]]] = {
        label: {} for label in twin_labels
    }
    other_rows: list[dict[str, Any]] = []
    missing_pair_ids = 0
    for row in rows:
        flag = bool(row.get("is_train_member", True))
        if train_member is not None and flag is not train_member:
            continue
        label = str(row.get("label") or "")
        if label not in twin_labels:
            other_rows.append(row)
            continue
        pair_id = row.get("pair_id") or row.get("matched_pair_id")
        if not pair_id:
            missing_pair_ids += 1
            continue
        by_label_pair[label][str(pair_id)] = row

    shared = sorted(set(by_label_pair[twin_labels[0]]) & set(by_label_pair[twin_labels[1]]))
    rng = random.Random(seed)
    rng.shuffle(shared)
    chosen = shared[: min(per_label, len(shared))]
    if len(chosen) < per_label or missing_pair_ids:
        print(
            f"[canaries] requested {per_label} paired twins, selected {len(chosen)}; "
            f"shared_available={len(shared)} rows_missing_pair_id={missing_pair_ids}",
            flush=True,
        )
    out = {
        label: [by_label_pair[label][pair_id] for pair_id in chosen]
        for label in twin_labels
    }
    sampled_other = _sample_by_label(
        other_rows, per_label=per_label, seed=seed, train_member=train_member
    )
    for label in LABELS:
        if label not in out:
            out[label] = sampled_other.get(label, [])
    return out
