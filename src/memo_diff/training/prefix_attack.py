"""Paper prefix-attack helpers."""

from __future__ import annotations

import json

import random

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Optional

import torch

from memo_diff.analysis.field_retention import (
    aggregate_field_accuracy,
    aggregate_per_field,
    score_field_retention,
)

LABELS = (
    "protected_association",
    "benign_arbitrary_association",
)

@dataclass
class AttackConfig:
    canaries_path: Path
    checkpoint_dir: Path
    output_dir: Path
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    prefix_frac: float = 0.5
    max_new_tokens_cap: int = 128
    samples_per_label: int = 40
    seed: int = 42
    device: str = "cuda"
    first_k: int = 32
    add_special_tokens: bool = False
    batch_size: int = 32
    save_example_details: bool | None = None  # default: samples_per_label <= 100
    membership: str = "train"  # "train" | "heldout" | "all"

def _disable_triton() -> None:
    try:
        from torch._native.registry import deregister_op_overrides

        deregister_op_overrides(disable_dsl_names=["triton", "cutedsl"])
    except Exception as exc:  # noqa: BLE001
        print(f"[attack] warn: could not disable torch._native: {exc}", flush=True)

def _load_canaries(
    path: Path,
    *,
    samples_per_label: int,
    seed: int,
    membership: str = "train",
) -> dict[str, list[dict[str, Any]]]:
    if membership not in {"train", "heldout", "all"}:
        raise ValueError(f"membership must be train, heldout, or all; got {membership!r}")
    by_label: dict[str, list[dict[str, Any]]] = {lab: [] for lab in LABELS}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            lab = row.get("label")
            if lab not in by_label:
                continue
            is_member = bool(row.get("is_train_member", True))
            if membership == "train" and not is_member:
                continue
            if membership == "heldout" and is_member:
                continue
            by_label[lab].append(row)
    rng = random.Random(seed)
    out: dict[str, list[dict[str, Any]]] = {}
    for lab, rows in by_label.items():
        rows = list(rows)
        rng.shuffle(rows)
        out[lab] = rows[: min(samples_per_label, len(rows))]
    return out

def _too_short_result() -> dict[str, Any]:
    return {"ok": False, "reason": "too_short", "exact": False, "token_acc": 0.0, "first_k_acc": 0.0}

def _token_split_at_char_fraction(
    text: str,
    ids: "torch.Tensor",
    tokenizer,
    *,
    prefix_frac: float,
    add_special_tokens: bool,
) -> int:
    """Pick a token boundary near the character midpoint of ``text``.

    Twin dossiers use matched prose templates; a character-fraction split keeps
    the same canonical fields eligible on both PHI and benign sides, whereas a
    raw token-fraction split diverges when the two texts tokenize differently.
    """
    n = int(ids.numel())
    if n < 2:
        return 1
    char_split = max(1, min(len(text) - 1, int(len(text) * prefix_frac)))
    skip_special = not add_special_tokens
    best = 1
    for index in range(1, n):
        decoded = tokenizer.decode(ids[:index], skip_special_tokens=skip_special)
        if len(decoded) <= char_split:
            best = index
        else:
            break
    return max(1, min(n - 1, best))

def _split_prefix_target(
    ids: "torch.Tensor",
    *,
    prefix_frac: float,
    max_new_tokens_cap: int,
    text: str | None = None,
    tokenizer=None,
    add_special_tokens: bool = False,
) -> tuple["torch.Tensor", "torch.Tensor", int, int] | None:
    n = int(ids.numel())
    if n < 4:
        return None
    if text is not None and tokenizer is not None:
        split = _token_split_at_char_fraction(
            text,
            ids,
            tokenizer,
            prefix_frac=prefix_frac,
            add_special_tokens=add_special_tokens,
        )
    else:
        split = max(1, min(n - 1, int(n * prefix_frac)))
    target = ids[split:]
    max_new = min(int(target.numel()), max_new_tokens_cap)
    if max_new < 1:
        return None
    return ids[:split], target[:max_new], max_new, split

def _score_generation(
    *,
    gen: "torch.Tensor",
    target: "torch.Tensor",
    prefix: "torch.Tensor",
    tokenizer,
    first_k: int,
) -> dict[str, Any]:
    gen = gen.detach().cpu()
    target = target.detach().cpu()
    prefix = prefix.detach().cpu()
    split = int(prefix.numel())
    if gen.numel() == 0 or target.numel() == 0:
        return {
            "ok": True,
            "exact": False,
            "token_acc": 0.0,
            "first_k_acc": 0.0,
            "prefix_len": split,
            "target_len": int(target.numel()),
            "gen_len": int(gen.numel()),
        }
    n_cmp = min(int(gen.numel()), int(target.numel()))
    match = (gen[:n_cmp] == target[:n_cmp]).float().mean().item()
    k = min(int(first_k), n_cmp)
    first_k_acc = float((gen[:k] == target[:k]).float().mean().item()) if k > 0 else 0.0
    exact = bool(gen.numel() >= target.numel() and torch.equal(gen[: target.numel()], target))
    return {
        "ok": True,
        "exact": exact,
        "token_acc": float(match),
        "first_k_acc": first_k_acc,
        "first_k": k,
        "prefix_len": split,
        "target_len": int(target.numel()),
        "gen_len": int(gen.numel()),
        "prefix_text": tokenizer.decode(prefix, skip_special_tokens=False),
        "expected_text": tokenizer.decode(target, skip_special_tokens=False),
        "generated_text": tokenizer.decode(gen, skip_special_tokens=False),
    }

def _left_pad(
    prefixes: list["torch.Tensor"],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    max_len = max(int(p.numel()) for p in prefixes)
    ids = []
    mask = []
    for p in prefixes:
        pad_len = max_len - int(p.numel())
        if pad_len:
            ids.append(torch.cat([p.new_full((pad_len,), int(pad_id)), p]))
            mask.append(torch.cat([p.new_zeros(pad_len), p.new_ones(int(p.numel()))]))
        else:
            ids.append(p)
            mask.append(p.new_ones(int(p.numel())))
    return torch.stack(ids).to(device), torch.stack(mask).to(device)

def _prefix_attack_batch(
    *,
    model,
    tokenizer,
    texts: list[str],
    prefix_frac: float,
    max_new_tokens_cap: int,
    device: torch.device,
    first_k: int = 32,
    add_special_tokens: bool = False,
) -> list[dict[str, Any]]:
    """Greedy prefix attack for a batch of texts (left-padded)."""
    prepared: list[tuple[int, "torch.Tensor", "torch.Tensor", int, int] | None] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=add_special_tokens, return_tensors="pt")["input_ids"][0]
        split = _split_prefix_target(
            ids,
            prefix_frac=prefix_frac,
            max_new_tokens_cap=max_new_tokens_cap,
            text=text,
            tokenizer=tokenizer,
            add_special_tokens=add_special_tokens,
        )
        if split is None:
            prepared.append(None)
        else:
            prefix, target, max_new, split_i = split
            prepared.append((split_i, prefix, target, max_new, int(ids.numel())))

    valid_idx = [i for i, p in enumerate(prepared) if p is not None]
    out_rows: list[dict[str, Any] | None] = [_too_short_result() if p is None else None for p in prepared]
    if not valid_idx:
        return [r if r is not None else _too_short_result() for r in out_rows]

    prefixes = [prepared[i][1] for i in valid_idx]  # type: ignore[index]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    input_ids, attention_mask = _left_pad(prefixes, pad_id=int(pad_id), device=device)
    batch_max_new = max(int(prepared[i][3]) for i in valid_idx)  # type: ignore[index]
    with torch.no_grad():
        gen_out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=batch_max_new,
            min_new_tokens=batch_max_new,
            do_sample=False,
            pad_token_id=int(pad_id),
            eos_token_id=None,
        )
    plen = int(input_ids.shape[1])
    for bi, i in enumerate(valid_idx):
        _split_i, prefix, target, max_new, _n = prepared[i]  # type: ignore[misc]
        gen = gen_out[bi, plen : plen + int(max_new)]
        out_rows[i] = _score_generation(
            gen=gen,
            target=target,
            prefix=prefix,
            tokenizer=tokenizer,
            first_k=first_k,
        )
    return [r if r is not None else _too_short_result() for r in out_rows]

def _dossier_field_matches(
    dossier: dict[str, Any],
    prefix_text: str,
    expected_text: str,
    generated_text: str,
    *,
    label: str | None = None,
) -> Optional[dict[str, Any]]:
    return score_field_retention(
        dossier,
        prefix_text=prefix_text,
        expected_suffix=expected_text,
        generated_text=generated_text,
        label=label,
    )

def _load_exposure(ckpt_path: Path) -> dict[str, dict[str, Any]]:
    path = ckpt_path / "exposure.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}

def _forgetting_bins(
    details: list[dict[str, Any]],
    *,
    eval_step: int,
) -> list[dict[str, Any]]:
    """Group prefix-attack scores by steps since the canary was first seen."""
    buckets = [(0, 50), (50, 200), (200, 500), (500, 1000), (1000, 10_000_000)]
    out: list[dict[str, Any]] = []
    for lo, hi in buckets:
        rows = [
            d
            for d in details
            if d.get("steps_since_intro") is not None and lo <= int(d["steps_since_intro"]) < hi
        ]
        if not rows:
            continue
        out.append(
            {
                "steps_since_intro": f"{lo}-{hi if hi < 10_000_000 else 'inf'}",
                "n": len(rows),
                "exact_acc": sum(1 for r in rows if r.get("exact")) / len(rows),
                "mean_token_acc": sum(float(r.get("token_acc") or 0.0) for r in rows) / len(rows),
                "mean_first_k_acc": sum(float(r.get("first_k_acc") or 0.0) for r in rows) / len(rows),
                "mean_field_acc": sum(
                    float((r.get("field_match") or {}).get("field_acc") or 0.0) for r in rows
                )
                / len(rows),
                "eval_step": eval_step,
            }
        )
    return out

def eval_checkpoint(
    *,
    ckpt_path: Path,
    tokenizer,
    canaries_by_label: dict[str, list[dict[str, Any]]],
    cfg: AttackConfig,
    device: torch.device,
    eval_step: Optional[int] = None,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    exposure = _load_exposure(ckpt_path)
    if eval_step is None:
        try:
            eval_step = int(ckpt_path.name.split("_")[1])
        except (IndexError, ValueError):
            eval_step = 0

    result = score_loaded_model(
        model=model,
        tokenizer=tokenizer,
        canaries_by_label=canaries_by_label,
        cfg=cfg,
        device=device,
        eval_step=eval_step,
        exposure=exposure,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result

def score_loaded_model(
    *,
    model,
    tokenizer,
    canaries_by_label: dict[str, list[dict[str, Any]]],
    cfg: AttackConfig,
    device: torch.device,
    eval_step: int = 0,
    exposure: Optional[dict[str, dict[str, Any]]] = None,
    generate_fn=None,
) -> dict[str, Any]:
    """Prefix-attack an already-loaded model. `generate_fn` overrides greedy decode."""
    exposure = exposure or {}
    per_label: dict[str, Any] = {}
    for lab, rows in canaries_by_label.items():
        exacts = 0
        tok_accs: list[float] = []
        first_ks: list[float] = []
        field_accs: list[float] = []
        details = []
        batch_size = max(1, int(getattr(cfg, "batch_size", 1) or 1))
        if generate_fn is not None:
            attacks = [generate_fn(str(row["text"])) for row in rows]
        else:
            attacks = []
            for start in range(0, len(rows), batch_size):
                chunk = rows[start : start + batch_size]
                attacks.extend(
                    _prefix_attack_batch(
                        model=model,
                        tokenizer=tokenizer,
                        texts=[str(row["text"]) for row in chunk],
                        prefix_frac=cfg.prefix_frac,
                        max_new_tokens_cap=cfg.max_new_tokens_cap,
                        device=device,
                        first_k=cfg.first_k,
                        add_special_tokens=cfg.add_special_tokens,
                    )
                )
        for row, r in zip(rows, attacks):
            cid = str(row["id"])
            exp = exposure.get(cid) or {}
            first_step = exp.get("first_step")
            r["first_step"] = first_step
            r["exposure_count"] = exp.get("count")
            r["steps_since_intro"] = (
                int(eval_step) - int(first_step) if first_step is not None else None
            )
            if r.get("exact"):
                exacts += 1
            tok_accs.append(float(r.get("token_acc") or 0.0))
            first_ks.append(float(r.get("first_k_acc") or 0.0))
            field_match = _dossier_field_matches(
                row.get("dossier"),
                r.get("prefix_text") or "",
                r.get("expected_text") or "",
                r.get("generated_text") or "",
                label=lab,
            )
            if field_match is not None:
                r["field_match"] = field_match
                field_accs.append(field_match["field_acc"])
            details.append({"id": row["id"], **r})
        field_matches = [d["field_match"] for d in details if d.get("field_match")]
        field_summary = aggregate_field_accuracy(field_matches)
        fields_correct = int(field_summary["field_instances_correct"])
        fields_total = int(field_summary["field_instances_total"])
        macro_field_acc = field_summary["macro_example_field_acc"]
        micro_field_acc = field_summary["micro_field_instance_acc"]
        n = max(1, len(rows))
        per_label[lab] = {
            "n": len(rows),
            "exact_acc": exacts / n,
            "mean_token_acc": sum(tok_accs) / n if tok_accs else 0.0,
            "mean_first_k_acc": sum(first_ks) / n if first_ks else 0.0,
            # Kept as a compatibility alias; new consumers should use one of the
            # two explicitly named metrics below.
            "mean_field_acc": macro_field_acc,
            "macro_example_field_acc": macro_field_acc,
            "micro_field_instance_acc": micro_field_acc,
            "field_instances_correct": fields_correct,
            "field_instances_total": fields_total,
            "n_field_scored": len(field_accs),
            "per_field_acc": aggregate_per_field(field_matches),
            "forgetting_bins": _forgetting_bins(details, eval_step=int(eval_step)),
            "details": details,
        }

    overall_exact = sum(v["exact_acc"] * v["n"] for v in per_label.values()) / max(
        1, sum(v["n"] for v in per_label.values())
    )
    field_n = sum(int(v.get("field_instances_total") or 0) for v in per_label.values())
    overall_field = None
    if field_n:
        overall_field = sum(
            int(v.get("field_instances_correct") or 0)
            for v in per_label.values()
        ) / field_n
    return {
        "per_label": per_label,
        "overall_exact_acc": overall_exact,
        "overall_field_acc": overall_field,
        "eval_step": eval_step,
    }

def _save_details(cfg: AttackConfig) -> bool:
    if cfg.save_example_details is not None:
        return bool(cfg.save_example_details)
    return int(cfg.samples_per_label) <= 100
