"""Paper-only helpers, retained from the research implementation."""

from __future__ import annotations


from pathlib import Path


from typing import Any, Sequence


import torch


from memo_diff.analysis.jacobian_lens import apply_jacobian_lens


def load_jacobians_pt(
    path: Path,
) -> tuple[dict[int, torch.Tensor], int, dict[str, Any] | None, dict[str, Any]]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    jac = {int(k): v.float() for k, v in raw["jacobians"].items()}
    target = int(raw.get("target_layer", max(jac) + 1 if jac else 0))
    metadata = {key: value for key, value in raw.items() if key != "jacobians"}
    return jac, target, raw.get("manifest"), metadata


def _lens_logits(
    hs: Sequence[torch.Tensor],
    jacobians: dict[int, torch.Tensor],
    *,
    target_layer: int,
    norm: Any,
    lm_head: Any,
    positions: Sequence[int],
) -> list[tuple[int, dict[int, torch.Tensor]]]:
    """Full-vocabulary logits only at requested positions, keyed by token index."""
    selected_positions = sorted(set(int(position) for position in positions))
    if not selected_positions:
        return []
    if selected_positions[0] < 0 or selected_positions[-1] >= hs[0].shape[0]:
        raise IndexError(
            f"lens position outside 0..{hs[0].shape[0] - 1}: {selected_positions}"
        )
    layers = sorted(jacobians) + [target_layer]
    selected_hs = [hidden[selected_positions] for hidden in hs]
    transported = apply_jacobian_lens(
        selected_hs, jacobians, target_layer=target_layer, layers=layers
    )
    out: list[tuple[int, dict[int, torch.Tensor]]] = []
    dtype = hs[-1].dtype
    if len(layers) != len(transported):
        raise RuntimeError("transported states do not match fitted layers")
    for layer, h in zip(layers, transported):
        logits = lm_head(norm(h.unsqueeze(0).to(dtype=dtype)))[0].float().cpu()
        out.append(
            (layer, {position: logits[index] for index, position in enumerate(selected_positions)})
        )
    return out
