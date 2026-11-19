import json
import torch
from memo_diff.analysis.readout import _lens_logits
from memo_diff.analysis.cohort import _load_canaries

def test_lens_logits_project_only_requested_positions() -> None:
    hidden = [torch.arange(8, dtype=torch.float32).reshape(4, 2) + offset for offset in (0, 10, 20)]
    projected = _lens_logits(
        hidden,
        {0: torch.eye(2), 1: torch.eye(2)},
        target_layer=2,
        norm=torch.nn.Identity(),
        lm_head=torch.nn.Identity(),
        positions=[1, 3, 1],
    )
    assert [layer for layer, _ in projected] == [0, 1, 2]
    assert all(sorted(by_position) == [1, 3] for _, by_position in projected)
    assert torch.equal(projected[0][1][3], hidden[0][3])


def test_paired_loader_samples_shared_pair_ids_first(tmp_path) -> None:
    path = tmp_path / "canaries.jsonl"
    rows = []
    for i in range(8):
        for label in ("protected_association", "benign_arbitrary_association"):
            rows.append(
                {
                    "id": f"{label}-{i}",
                    "label": label,
                    "pair_id": f"pair-{i}",
                    "text": str(i),
                    "is_train_member": True,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    selected = _load_canaries(path, per_label=4, seed=42, pair_twins=True)
    phi_ids = [row["pair_id"] for row in selected["protected_association"]]
    benign_ids = [row["pair_id"] for row in selected["benign_arbitrary_association"]]
    assert len(phi_ids) == 4
    assert phi_ids == benign_ids


def test_paired_loader_can_select_heldout_only(tmp_path) -> None:
    path = tmp_path / "canaries.jsonl"
    rows = []
    for member in (True, False):
        for label in ("protected_association", "benign_arbitrary_association"):
            rows.append(
                {
                    "id": f"{label}-{member}",
                    "label": label,
                    "pair_id": f"pair-{member}",
                    "text": str(member),
                    "is_train_member": member,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    selected = _load_canaries(
        path, per_label=1, seed=42, pair_twins=True, train_member=False
    )
    assert all(
        not row["is_train_member"]
        for label_rows in selected.values()
        for row in label_rows
    )



def test_first_token_targets_skip_identifier_literals_but_keep_ssn_prefix():
    from memo_diff.analysis.field_targets import first_token_targets

    class CharacterOffsets:
        def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False):
            result = {'input_ids': [0] + [ord(ch) for ch in text]}
            if return_offsets_mapping:
                result['offset_mapping'] = [(0, 0)] + [(i, i + 1) for i in range(len(text))]
            return result

    text = 'Social Security Number 999-00-1234, MRN-87654321, HPB-543210987.'
    ids, targets = first_token_targets(
        full_text=text,
        dossier={'ssn': '999-00-1234', 'mrn': 'MRN-87654321',
                 'health_plan_beneficiary_number': 'HPB-543210987'},
        label='protected_association', tokenizer=CharacterOffsets(),
    )
    assert ids[targets['ssn']] == ord('9')
    assert ids[targets['mrn']] == ord('8')
    assert ids[targets['health_plan_beneficiary_number']] == ord('5')
