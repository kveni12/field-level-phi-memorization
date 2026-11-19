"""Paper-only helpers, retained from the research implementation."""

from __future__ import annotations


from typing import Any, Iterable, Literal


from memo_diff.analysis.field_retention import PHI_SENSITIVE_FIELDS, gold_fields


PHI_LABEL = "protected_association"


EXCLUDED_FIELDS = frozenset({"entity"})


FIELD_VALUE_PREFIXES = {
    "mrn": "MRN-",
    "facility_record_number": "MRN-",
    "health_plan_beneficiary_number": "HPB-",
    "plan_account_id": "HPB-",
}


def _char_span_to_token_indices(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> list[int]:
    return [
        i
        for i, (start, end) in enumerate(offsets)
        if not (start == end == 0) and end > char_start and start < char_end
    ]


def field_token_spans(
    *,
    full_text: str,
    dossier: dict[str, Any],
    label: str,
    tokenizer: Any,
    prefix_frac: float = 0.0,
    add_special_tokens: bool,
    strip_value_prefixes: bool = True,
) -> tuple[int, dict[str, list[int]]]:
    """Map field values to token spans.

    ``prefix_frac`` remains for compatibility with the former suffix lens. At
    zero (the default), fields anywhere in the record are eligible.

    When ``strip_value_prefixes`` is True (legacy), MRN-like fields skip the
    literal ``MRN-`` characters and score only the numeric body.
    """
    enc = tokenizer(full_text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
    offsets = [tuple(x) for x in enc["offset_mapping"]]
    n = len(enc["input_ids"])
    if n < 2:
        return 0, {}
    split = 0 if prefix_frac <= 0 else max(1, min(n - 1, int(n * prefix_frac)))
    char_floor = offsets[split][0] if split else 0
    spans: dict[str, list[int]] = {}
    for field, value in gold_fields(dossier, label=label).items():
        if field in EXCLUDED_FIELDS:
            continue
        start = full_text.find(value, char_floor)
        if start < 0:
            continue
        target_start = start
        value_prefix = FIELD_VALUE_PREFIXES.get(field) if strip_value_prefixes else None
        if value_prefix and value.startswith(value_prefix):
            target_start += len(value_prefix)
        indices = _char_span_to_token_indices(offsets, target_start, start + len(value))
        indices = [i for i in indices if i >= split]
        if indices:
            spans[field] = indices
    return split, spans


def first_token_targets(
    *,
    full_text: str,
    dossier: dict[str, Any],
    tokenizer: Any,
    label: str = PHI_LABEL,
    add_special_tokens: bool = True,
    prefix_frac: float = 0.0,
    target_fields: Iterable[str] | None = None,
    strip_value_prefixes: bool = True,
) -> tuple[list[int], dict[str, int]]:
    """Return full token IDs and each field's first gold-token index."""
    enc = tokenizer(full_text, add_special_tokens=add_special_tokens)
    ids = list(enc["input_ids"])
    _, spans = field_token_spans(
        full_text=full_text,
        dossier=dossier,
        label=label,
        tokenizer=tokenizer,
        prefix_frac=prefix_frac,
        add_special_tokens=add_special_tokens,
        strip_value_prefixes=strip_value_prefixes,
    )
    if target_fields is not None:
        allowed = set(target_fields)
        spans = {field: idxs for field, idxs in spans.items() if field in allowed}
    return ids, {field: idxs[0] for field, idxs in spans.items() if idxs[0] > 0}


def reciprocal_rank(logits: Any, gold_token_id: int) -> tuple[int, float]:
    """Compute exact 1-indexed rank and reciprocal rank without sorting."""
    gold_logit = logits[int(gold_token_id)]
    rank = int((logits > gold_logit).sum().item()) + 1
    return rank, 1.0 / rank
