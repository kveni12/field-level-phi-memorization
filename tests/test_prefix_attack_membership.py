import json

from memo_diff.training.prefix_attack import _load_canaries


def test_load_canaries_selects_membership(tmp_path) -> None:
    path = tmp_path / "canaries.jsonl"
    rows = [
        {
            "id": "train",
            "label": "protected_association",
            "is_train_member": True,
        },
        {
            "id": "heldout",
            "label": "protected_association",
            "is_train_member": False,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    train = _load_canaries(path, samples_per_label=10, seed=42, membership="train")
    heldout = _load_canaries(path, samples_per_label=10, seed=42, membership="heldout")
    both = _load_canaries(path, samples_per_label=10, seed=42, membership="all")

    assert [row["id"] for row in train["protected_association"]] == ["train"]
    assert [row["id"] for row in heldout["protected_association"]] == ["heldout"]
    assert {row["id"] for row in both["protected_association"]} == {"train", "heldout"}
