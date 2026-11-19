"""Paper-only helpers, retained from the research implementation."""

from __future__ import annotations


import json


import random


from pathlib import Path


from typing import Any, Iterator, Optional


import torch


LABELS = (
    "protected_association",
    "benign_arbitrary_association",
)


def load_unique_canaries(path: Path) -> list[dict[str, Any]]:
    """Load canaries and dedupe by id (keeps first occurrence)."""
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = str(row.get("id") or "")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            rows.append(row)
    return rows


def unique_by_id(canaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in canaries:
        cid = str(row.get("id") or "")
        if cid and cid not in by_id:
            by_id[cid] = row
    return list(by_id.values())


def build_balanced_canary_queue(
    canaries: list[dict[str, Any]],
    *,
    reps: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Equal exposure across PHI / benign / public: round-robin labels each epoch,
    then repeat `reps` times. Paired PHI/benign share similar introduction times.
    """
    unique = unique_by_id(canaries)
    by_label: dict[str, list[dict[str, Any]]] = {lab: [] for lab in LABELS}
    other: list[dict[str, Any]] = []
    for row in unique:
        lab = str(row.get("label") or "")
        if lab in by_label:
            by_label[lab].append(row)
        else:
            other.append(row)
    for lab in LABELS:
        rng.shuffle(by_label[lab])
    rng.shuffle(other)

    epoch: list[dict[str, Any]] = []
    present = [lab for lab in LABELS if by_label[lab]]
    maxlen = max((len(by_label[lab]) for lab in present), default=0)
    for i in range(maxlen):
        for lab in present:
            if i < len(by_label[lab]):
                epoch.append(by_label[lab][i])
    epoch.extend(other)

    queue: list[dict[str, Any]] = []
    for _ in range(max(1, int(reps))):
        shuffled = list(epoch)
        rng.shuffle(shuffled)
        # Re-balance after shuffle: take in label-round-robin from shuffled pools.
        pools = {lab: [r for r in shuffled if r.get("label") == lab] for lab in LABELS}
        rest = [r for r in shuffled if r.get("label") not in pools]
        n = max((len(v) for v in pools.values()), default=0)
        for j in range(n):
            for lab in LABELS:
                if j < len(pools[lab]):
                    queue.append(pools[lab][j])
        queue.extend(rest)
    return queue


def tokenize_canary_ids(
    tokenizer,
    text: str,
    *,
    add_special_tokens: bool,
    max_len: int,
) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=True,
        max_length=max(1, int(max_len)),
    )
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(x) for x in ids]
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None and ids and ids[-1] != int(eos):
        if len(ids) >= max_len:
            ids = ids[: max_len - 1] + [int(eos)]
        else:
            ids = ids + [int(eos)]
    return ids[:max_len]


def pack_canary_epoch(
    canaries: list[dict[str, Any]],
    tokenizer,
    *,
    seq_len: int,
    rng: random.Random,
    add_special_tokens: bool = True,
    pack: bool = True,
) -> list[dict[str, Any]]:
    """Pack one shuffled epoch of unique canaries into fixed-length sequences."""
    order = build_balanced_canary_queue(canaries, reps=1, rng=rng)
    sequences: list[dict[str, Any]] = []
    buf: list[int] = []
    cids: list[str] = []

    def flush() -> None:
        nonlocal buf, cids
        if buf:
            sequences.append({"input_ids": list(buf), "canary_ids": list(dict.fromkeys(cids))})
            buf = []
            cids = []

    for row in order:
        first = not buf
        ids = tokenize_canary_ids(
            tokenizer,
            str(row.get("text") or ""),
            add_special_tokens=bool(add_special_tokens and first),
            max_len=seq_len,
        )
        if not ids:
            continue
        if (not pack) or (buf and len(buf) + len(ids) > seq_len):
            flush()
            ids = tokenize_canary_ids(
                tokenizer,
                str(row.get("text") or ""),
                add_special_tokens=add_special_tokens,
                max_len=seq_len,
            )
        buf.extend(ids)
        cid = str(row.get("id") or "")
        if cid:
            cids.append(cid)
    flush()
    return sequences


def pad_packed_sequence(
    ids: list[int],
    *,
    seq_len: int,
    pad_id: int,
) -> dict[str, torch.Tensor]:
    clipped = ids[:seq_len]
    pad_n = seq_len - len(clipped)
    input_ids = clipped + [int(pad_id)] * pad_n
    labels = clipped + [-100] * pad_n
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
