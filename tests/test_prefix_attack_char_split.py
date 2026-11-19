from __future__ import annotations

import json
from pathlib import Path

import torch

from memo_diff.analysis.field_retention import score_field_retention
from memo_diff.dataset.render import render_benign_text, render_phi_text
from memo_diff.training.prefix_attack import _token_split_at_char_fraction


class _FakeTokenizer:
    def __init__(self) -> None:
        self._vocab = {ch: i + 1 for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 ,.-")}

    def __call__(self, text: str, *, add_special_tokens: bool = False, return_tensors: str = "pt"):
        del add_special_tokens
        ids = [self._vocab[ch] for ch in text.lower() if ch in self._vocab]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids, *, skip_special_tokens: bool = True):
        del skip_special_tokens
        inv = {v: k for k, v in self._vocab.items()}
        return "".join(inv[int(i)] for i in ids)


def test_char_fraction_split_aligns_twin_field_eligibility() -> None:
    rows = [
        json.loads(line)
        for line in Path(
            "data/training/llama31_8b_balanced_2k_phi_benign_v2_luna_9f/train_canaries.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]
    by_pair: dict[str, dict[str, dict]] = {}
    for row in rows[:40]:
        by_pair.setdefault(row["pair_id"], {})[row["label"]] = row

    tokenizer = _FakeTokenizer()
    for labeled in by_pair.values():
        phi = labeled["protected_association"]
        benign = labeled["benign_arbitrary_association"]
        phi_text = render_phi_text(phi["dossier"])
        ben_text = render_benign_text(benign["dossier"])
        phi_ids = tokenizer(phi_text)["input_ids"][0]
        ben_ids = tokenizer(ben_text)["input_ids"][0]
        phi_split = _token_split_at_char_fraction(
            phi_text, phi_ids, tokenizer, prefix_frac=0.5, add_special_tokens=False
        )
        ben_split = _token_split_at_char_fraction(
            ben_text, ben_ids, tokenizer, prefix_frac=0.5, add_special_tokens=False
        )
        phi_prefix = tokenizer.decode(phi_ids[:phi_split])
        phi_suffix = tokenizer.decode(phi_ids[phi_split:])
        ben_prefix = tokenizer.decode(ben_ids[:ben_split])
        ben_suffix = tokenizer.decode(ben_ids[ben_split:])
        phi_fields = score_field_retention(
            phi["dossier"],
            prefix_text=phi_prefix,
            expected_suffix=phi_suffix,
            generated_text=phi_suffix,
            label="protected_association",
        )
        ben_fields = score_field_retention(
            benign["dossier"],
            prefix_text=ben_prefix,
            expected_suffix=ben_suffix,
            generated_text=ben_suffix,
            label="benign_arbitrary_association",
        )
        assert phi_fields is not None and ben_fields is not None
        assert set(phi_fields["per_field"]) == set(ben_fields["per_field"])
