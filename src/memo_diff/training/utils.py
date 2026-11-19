"""Shared utilities for training and checkpoint maintenance."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch


def cosine_lr(step: int, *, base_lr: float, warmup: int, max_steps: int) -> float:
    """Return a linearly warmed-up cosine learning rate."""
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    progress = (step - warmup) / float(max(1, max_steps - warmup))
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def slim_trainer_state(path: Path) -> None:
    """Remove optimizer tensors from a checkpoint while preserving metadata."""
    optimizer_path = path / "optimizer.pt"
    if optimizer_path.exists():
        optimizer_path.unlink()
        return
    state_path = path / "trainer_state.pt"
    if not state_path.exists():
        return
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[train] warning: could not slim {state_path}: {exc}", flush=True)
        return
    if not isinstance(state, dict) or "optimizer" not in state:
        return
    slim = {
        "step": state.get("step"),
        "epoch": state.get("epoch"),
        "exposure": state.get("exposure") or {},
        "config": state.get("config") or {},
    }
    torch.save(slim, state_path)


def probe_canaries(
    *,
    model,
    tokenizer,
    canaries: list[dict[str, Any]],
    device: torch.device,
    n: int,
    prefix_frac: float = 0.5,
    add_special_tokens: bool = False,
) -> dict[str, float]:
    """Measure teacher-forced loss and greedy suffix token accuracy."""
    raw = model.module if hasattr(model, "module") else model
    was_training = raw.training
    raw.eval()
    sample = canaries[: max(1, min(n, len(canaries)))]
    nlls: list[float] = []
    token_accuracies: list[float] = []
    with torch.no_grad():
        for row in sample:
            ids = tokenizer(
                str(row["text"]),
                add_special_tokens=add_special_tokens,
                return_tensors="pt",
            )["input_ids"].to(device)
            if ids.shape[1] < 4:
                continue
            nlls.append(float(raw(input_ids=ids, labels=ids).loss.item()))
            split = max(1, min(ids.shape[1] - 1, int(ids.shape[1] * prefix_frac)))
            prefix = ids[:, :split]
            target = ids[0, split:]
            generated = raw.generate(
                prefix,
                max_new_tokens=int(target.numel()),
                min_new_tokens=int(target.numel()),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=None,
            )
            prediction = generated[0, split : split + target.numel()]
            compared = min(int(prediction.numel()), int(target.numel()))
            if compared:
                accuracy = (prediction[:compared] == target[:compared]).float().mean()
                token_accuracies.append(float(accuracy.item()))
    if was_training:
        raw.train()
    return {
        "probe_n": float(len(sample)),
        "probe_nll": sum(nlls) / max(1, len(nlls)),
        "probe_token_acc": sum(token_accuracies) / max(1, len(token_accuracies)),
    }
