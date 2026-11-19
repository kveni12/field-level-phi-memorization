"""Anthropic-style Jacobian lens: unembed(J_l @ h_l), J_l = E[∂h_final / ∂h_l].

Estimator matches jacobian-lens / Gurnee et al. 2026: one-hot cotangents on the
final residual at every valid target position, mean over source positions.

Speed knobs:
  * ``dim_batch`` — parallelize output-dimension one-hots in one forward
    (passes = ceil(d_model / dim_batch); 8 → 512 passes, 64 → 64 passes).
    Pass ``<= 0`` to autotune the largest batch that fits in VRAM.
  * ``prompt_batch`` — average several same-length prompts in the same
    forward/backward sweep (effective batch = prompt_batch * dim_batch).
"""

from __future__ import annotations

import math
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


SKIP_FIRST_N_POSITIONS = 8


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    special_tokens = json.loads(
        json.dumps(getattr(tokenizer, "special_tokens_map", {}), default=str)
    )
    return {
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "revision": init_kwargs.get("revision") or init_kwargs.get("_commit_hash"),
        "vocab_size": int(len(tokenizer)),
        "special_tokens_map": special_tokens,
    }


def build_lens_manifest(
    *,
    model: nn.Module,
    tokenizer: Any,
    checkpoint_path: str | Path,
    fit_texts: Sequence[str],
    fit_source: str,
    fitted: dict[str, Any],
    add_special_tokens: bool,
) -> dict[str, Any]:
    """Create the compatibility/provenance manifest stored with a fitted lens."""
    raw = model.module if hasattr(model, "module") else model
    config = raw.config.to_dict()
    tokenizer_identity = _tokenizer_identity(tokenizer)
    fit_hasher = hashlib.sha256()
    for text in fit_texts:
        encoded = text.encode("utf-8")
        fit_hasher.update(len(encoded).to_bytes(8, "big"))
        fit_hasher.update(encoded)
    return {
        "schema_version": 1,
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "model_config_sha256": _stable_hash(config),
        "model_type": config.get("model_type"),
        "hidden_size": int(raw.config.hidden_size),
        "num_hidden_layers": int(raw.config.num_hidden_layers),
        "tokenizer": tokenizer_identity,
        "tokenizer_sha256": _stable_hash(tokenizer_identity),
        "layer_convention": "hf_hidden_states_embedding_is_layer_0",
        "target_layer": int(fitted["target_layer"]),
        "source_layers": [int(x) for x in fitted["source_layers"]],
        "fit_source": fit_source,
        "fit_corpus_sha256": fit_hasher.hexdigest(),
        "fit_prompt_count_requested": len(fit_texts),
        "fit_prompt_count_used": int(fitted["n_prompts"]),
        "max_seq_len": int(fitted["max_seq_len"]),
        "skip_first": int(fitted["skip_first"]),
        "add_special_tokens": bool(add_special_tokens),
    }


def validate_lens_manifest(
    manifest: dict[str, Any] | None,
    *,
    model: nn.Module,
    tokenizer: Any,
    checkpoint_path: str | Path,
    jacobians: dict[int, torch.Tensor],
    target_layer: int,
) -> None:
    """Reject missing, stale, incomplete, or shape-incompatible lens caches."""
    if not isinstance(manifest, dict):
        raise ValueError("Jacobian cache has no compatibility manifest; refit it")
    raw = model.module if hasattr(model, "module") else model
    expected_path = str(Path(checkpoint_path).expanduser().resolve())
    checks = {
        "schema_version": 1,
        "checkpoint_path": expected_path,
        "model_config_sha256": _stable_hash(raw.config.to_dict()),
        "hidden_size": int(raw.config.hidden_size),
        "num_hidden_layers": int(raw.config.num_hidden_layers),
        "tokenizer_sha256": _stable_hash(_tokenizer_identity(tokenizer)),
        "target_layer": int(target_layer),
    }
    mismatches = [
        f"{key}: cache={manifest.get(key)!r}, current={expected!r}"
        for key, expected in checks.items()
        if manifest.get(key) != expected
    ]
    cached_sources = [int(x) for x in manifest.get("source_layers", [])]
    if cached_sources != sorted(jacobians):
        mismatches.append(
            f"source_layers: manifest={cached_sources!r}, tensors={sorted(jacobians)!r}"
        )
    width = int(raw.config.hidden_size)
    bad_shapes = {
        layer: tuple(matrix.shape)
        for layer, matrix in jacobians.items()
        if tuple(matrix.shape) != (width, width)
    }
    if bad_shapes:
        mismatches.append(f"Jacobian shapes incompatible with width {width}: {bad_shapes}")
    if mismatches:
        raise ValueError("incompatible Jacobian cache:\n- " + "\n- ".join(mismatches))


def valid_position_mask(seq_len: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS) -> torch.Tensor:
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    if seq_len > skip_first + 1:
        mask[skip_first : seq_len - 1] = True
    if int(mask.sum()) == 0:
        raise ValueError(f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens")
    return mask


def _causal_lm_trunk(model: nn.Module) -> nn.Module:
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "model") and hasattr(raw.model, "layers"):
        return raw.model
    if hasattr(raw, "transformer") and hasattr(raw.transformer, "h"):
        return raw.transformer
    raise AttributeError("Could not locate decoder trunk")


def residual_modules(model: nn.Module) -> list[nn.Module]:
    """Modules whose outputs are embedding + each block residual (HF hidden_states)."""
    trunk = _causal_lm_trunk(model)
    if hasattr(trunk, "embed_tokens") and hasattr(trunk, "layers"):
        return [trunk.embed_tokens, *list(trunk.layers)]
    if hasattr(trunk, "wte") and hasattr(trunk, "h"):
        return [trunk.wte, *list(trunk.h)]
    raise AttributeError("Could not locate embedding + layers")


def _capture_residuals(modules: Sequence[nn.Module]) -> tuple[dict[int, torch.Tensor], list[Any]]:
    acts: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            acts[idx] = h

        return hook

    for idx, mod in enumerate(modules):
        handles.append(mod.register_forward_hook(make_hook(idx)))
    return acts, handles


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def jacobian_for_ids(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 64,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    attention_mask: torch.Tensor | None = None,
    max_dim_passes: int | None = None,
) -> dict[int, torch.Tensor]:
    """Per-layer Jacobian ``[d_model, d_model]`` (fp32 CPU), mean over a prompt batch.

    ``input_ids`` is ``[1, seq]`` or ``[P, seq]`` (same length; pad OK with mask).
    Each prompt is replicated ``dim_batch`` times so one backward covers
    ``dim_batch`` output dims. Effective forward batch = ``P * dim_batch``.

    ``max_dim_passes`` limits how many dim-sweeps run (use ``1`` for VRAM probes).
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [P, seq], got {tuple(input_ids.shape)}")
    if dim_batch < 1:
        raise ValueError(f"dim_batch must be >= 1, got {dim_batch}")
    n_prompts, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    modules = residual_modules(model)
    n_states = len(modules)
    d_model = int(model.config.hidden_size)
    target = n_states - 1 if target_layer is None else int(target_layer)
    if target < 0:
        target += n_states
    sources = list(range(target)) if source_layers is None else [int(x) for x in source_layers]
    if not sources or min(sources) < 0 or max(sources) >= target:
        raise ValueError(f"source_layers={sources} invalid for target={target}")

    # Per-row valid positions (skip BOS band + final token; also honor padding).
    base_mask = valid_position_mask(seq_len, skip_first=skip_first).to(input_ids.device)
    if attention_mask is None:
        attention_mask = torch.ones(n_prompts, seq_len, dtype=torch.long, device=input_ids.device)
    else:
        attention_mask = attention_mask.to(device=input_ids.device)
    # Valid if in base band and not padding; also require next token exists (non-pad).
    # Use positions where mask==1 and index < last non-pad - keep simple: base & pad.
    pad_ok = attention_mask.bool()
    # Exclude the last non-pad token per row (nowhere to predict next).
    last_idx = attention_mask.long().sum(dim=1) - 1  # [P]
    position_ok = base_mask.unsqueeze(0) & pad_ok
    for p in range(n_prompts):
        li = int(last_idx[p].item())
        if 0 <= li < seq_len:
            position_ok[p, li] = False
    if int(position_ok.sum()) == 0:
        raise ValueError("no valid positions in prompt batch")

    n_passes = math.ceil(d_model / dim_batch)
    if max_dim_passes is not None:
        n_passes = max(1, min(n_passes, int(max_dim_passes)))
    jacobians = {layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in sources}
    device = input_ids.device

    # Layout: [p0_d0..p0_dD, p1_d0.., ...]
    replicated = input_ids.repeat_interleave(dim_batch, dim=0)  # [P*D, S]
    repl_mask = attention_mask.repeat_interleave(dim_batch, dim=0)

    acts, handles = _capture_residuals(modules)
    try:
        with torch.enable_grad():
            model(input_ids=replicated, attention_mask=repl_mask, use_cache=False)
            target_activation = acts[target]
            source_activations = [acts[layer] for layer in sources]
            cotangent = torch.zeros_like(target_activation)
            batch_local = torch.arange(dim_batch, device=device)

            dim_starts = list(range(0, d_model, dim_batch))[:n_passes]
            for pass_idx, dim_start in enumerate(dim_starts):
                n_dims = min(dim_batch, d_model - dim_start)
                cotangent.zero_()
                # For each prompt p, replicas [p*D, p*D+n_dims) get one-hot on dims
                for p in range(n_prompts):
                    row0 = p * dim_batch
                    rows = row0 + batch_local[:n_dims]
                    pos = position_ok[p].nonzero(as_tuple=True)[0]
                    if pos.numel() == 0:
                        continue
                    cotangent[
                        rows[:n_dims, None],
                        pos[None, :],
                        dim_start + batch_local[:n_dims, None],
                    ] = 1.0
                grads = torch.autograd.grad(
                    outputs=target_activation,
                    inputs=source_activations,
                    grad_outputs=cotangent,
                    retain_graph=(pass_idx < len(dim_starts) - 1),
                    allow_unused=False,
                )
                if len(sources) != len(grads):
                    raise RuntimeError("activation-gradient count does not match source layers")
                for layer, grad in zip(sources, grads):
                    # Average over prompts and their valid positions
                    acc_rows = torch.zeros(n_dims, d_model, dtype=torch.float32, device=device)
                    for p in range(n_prompts):
                        row0 = p * dim_batch
                        pos = position_ok[p].nonzero(as_tuple=True)[0]
                        if pos.numel() == 0:
                            continue
                        # grad[row0:row0+n_dims, pos, :] → mean over pos → [n_dims, d]
                        acc_rows += grad[row0 : row0 + n_dims][:, pos, :].float().mean(dim=1)
                    acc_rows /= float(n_prompts)
                    jacobians[layer][dim_start : dim_start + n_dims, :] = acc_rows.cpu()
                del grads
    finally:
        for h in handles:
            h.remove()

    return jacobians


def jacobian_batch_fits(
    model: nn.Module,
    *,
    n_prompts: int,
    seq_len: int,
    dim_batch: int,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
) -> bool:
    """Return True if one Jacobian forward+backward fits in GPU memory."""
    device = next(model.parameters()).device
    if device.type != "cuda":
        return True
    skip = min(skip_first, max(0, seq_len - 2))
    ids = torch.ones(n_prompts, seq_len, dtype=torch.long, device=device)
    try:
        jacobian_for_ids(
            model,
            ids,
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
            skip_first=skip,
            max_dim_passes=1,
        )
        torch.cuda.synchronize()
        return True
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        torch.cuda.empty_cache()
        return False


def autotune_jac_batches(
    model: nn.Module,
    *,
    seq_len: int,
    max_dim_batch: int = 128,
    max_prompt_batch: int = 4,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
) -> tuple[int, int]:
    """Binary-search largest ``dim_batch`` then ``prompt_batch`` that fit.

    Probes at ``seq_len`` (use the fit ``max_seq_len`` so longer FineWeb
    prompts do not OOM later). Returns ``(dim_batch, prompt_batch)``.
    """
    device = next(model.parameters()).device
    max_dim_batch = max(1, int(max_dim_batch))
    max_prompt_batch = max(1, int(max_prompt_batch))

    def fits(p: int, d: int) -> bool:
        if device.type != "cuda":
            return True
        return jacobian_batch_fits(
            model,
            n_prompts=p,
            seq_len=seq_len,
            dim_batch=d,
            skip_first=skip_first,
            source_layers=source_layers,
            target_layer=target_layer,
        )

    # dim_batch at prompt_batch=1
    lo, hi = 1, max_dim_batch
    best_d = 1
    if not fits(1, 1):
        raise RuntimeError(
            f"jacobian fit OOMs even at dim_batch=1 prompt_batch=1 seq_len={seq_len}"
        )
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(1, mid):
            best_d = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # prompt_batch at the chosen dim_batch
    best_p = 1
    lo, hi = 1, max_prompt_batch
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(mid, best_d):
            best_p = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(
            f"[jac-fit] autotune dim_batch={best_d} prompt_batch={best_p} "
            f"seq_len={seq_len} gpu_free={free / 1e9:.1f}G/{total / 1e9:.1f}G",
            flush=True,
        )
    else:
        print(
            f"[jac-fit] autotune dim_batch={best_d} prompt_batch={best_p} "
            f"seq_len={seq_len} (cpu)",
            flush=True,
        )
    return best_d, best_p


def _jacobian_with_oom_backoff(
    model: nn.Module,
    batch_ids: torch.Tensor,
    *,
    sources: Sequence[int],
    target: int,
    dim_batch: int,
    skip_first: int,
    attention_mask: torch.Tensor | None = None,
) -> tuple[dict[int, torch.Tensor], int]:
    """Run ``jacobian_for_ids``, halving ``dim_batch`` (then P) on OOM."""
    device = batch_ids.device
    cur_d = max(1, int(dim_batch))
    prompts = [batch_ids[i : i + 1] for i in range(int(batch_ids.shape[0]))]
    attn_rows = None
    if attention_mask is not None:
        attn_rows = [attention_mask[i : i + 1] for i in range(int(attention_mask.shape[0]))]

    # Try full chunk first; on OOM shrink dim_batch then fall back to P=1.
    attempt_ids = batch_ids
    attempt_attn = attention_mask
    while True:
        try:
            out = jacobian_for_ids(
                model,
                attempt_ids,
                source_layers=sources,
                target_layer=target,
                dim_batch=cur_d,
                skip_first=skip_first,
                attention_mask=attempt_attn,
            )
            return out, cur_d
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if cur_d > 1:
                cur_d = max(1, cur_d // 2)
                print(
                    f"[jac-fit] OOM → retry dim_batch={cur_d} P={attempt_ids.shape[0]}",
                    flush=True,
                )
                continue
            if attempt_ids.shape[0] > 1:
                # Per-prompt with dim_batch=1; caller should loop — signal by raising
                # after we process individually below.
                break
            raise

    # Per-prompt path (caller expects one averaged matrix for the chunk).
    # Average individually with dim_batch=1.
    acc: dict[int, torch.Tensor] | None = None
    for i, ids_1 in enumerate(prompts):
        attn_1 = attn_rows[i] if attn_rows is not None else None
        d = 1
        while True:
            try:
                per = jacobian_for_ids(
                    model,
                    ids_1.to(device),
                    source_layers=sources,
                    target_layer=target,
                    dim_batch=d,
                    skip_first=skip_first,
                    attention_mask=attn_1.to(device) if attn_1 is not None else None,
                )
                break
            except RuntimeError as exc:
                if not _is_oom(exc) or d == 1:
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                d = max(1, d // 2)
        if acc is None:
            acc = {li: mat.clone() for li, mat in per.items()}
        else:
            for li, mat in per.items():
                acc[li] += mat
    assert acc is not None
    n = float(len(prompts))
    return {li: mat / n for li, mat in acc.items()}, 1


def fit_jacobian_lens(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    dim_batch: int = 0,
    prompt_batch: int = 4,
    max_seq_len: int = 96,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    add_special_tokens: bool = True,
    layer_stride: int = 1,
    target_layer: int | None = None,
    source_layers: Sequence[int] | None = None,
    max_dim_batch_search: int = 128,
) -> dict[str, Any]:
    """Average per-prompt Jacobians. Returns CPU fp32 maps plus metadata.

    Prompts are bucketed by tokenized length so ``prompt_batch`` same-length
    examples share one forward + ``ceil(d/dim_batch)`` backwards.

    ``dim_batch <= 0`` autotunes the largest fitting ``dim_batch`` / ``prompt_batch``
    (``prompt_batch`` is treated as an upper bound).
    """
    device = next(model.parameters()).device
    # Mild speed knobs (safe for this workload).
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    modules = residual_modules(model)
    target = len(modules) - 1 if target_layer is None else int(target_layer)
    if target < 0:
        target += len(modules)
    if target <= 0 or target >= len(modules):
        raise ValueError(
            f"target_layer={target_layer!r} resolves to {target}; "
            f"expected 1..{len(modules) - 1}"
        )
    if source_layers is None:
        sources = list(range(0, target, max(1, int(layer_stride))))
        if sources[-1] != target - 1:
            sources.append(target - 1)
    else:
        sources = sorted(set(int(layer) for layer in source_layers))
        if not sources or sources[0] < 0 or sources[-1] >= target:
            raise ValueError(
                f"source_layers={sources} invalid for resolved target={target}"
            )
    d_model = int(model.config.hidden_size)
    acc = {li: torch.zeros(d_model, d_model, dtype=torch.float32) for li in sources}
    n_done = 0
    skipped = 0
    skipped_prompts: list[dict[str, Any]] = []
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # Tokenize up front; bucket by length for packed prompt batches.
    encoded: list[tuple[int, int, torch.Tensor]] = []
    for prompt_index, text in enumerate(prompts):
        ids = tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )["input_ids"][0]
        if int(ids.shape[0]) <= skip_first + 1:
            skipped += 1
            reason = f"too_short:{int(ids.shape[0])}_tokens"
            skipped_prompts.append({"prompt_index": prompt_index, "reason": reason})
            print(f"[jac-fit] skip prompt={prompt_index} reason={reason}", flush=True)
            continue
        encoded.append((prompt_index, int(ids.shape[0]), ids))

    by_len: dict[int, list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for prompt_index, length, ids in encoded:
        by_len[length].append((prompt_index, ids))

    tuned = dim_batch <= 0
    if tuned:
        dim_batch, prompt_batch = autotune_jac_batches(
            model,
            seq_len=max_seq_len,
            max_dim_batch=max_dim_batch_search,
            max_prompt_batch=max(1, int(prompt_batch)),
            skip_first=skip_first,
            source_layers=sources,
            target_layer=target,
        )
    else:
        dim_batch = max(1, int(dim_batch))
        prompt_batch = max(1, int(prompt_batch))

    n_passes = math.ceil(d_model / max(1, dim_batch))
    print(
        f"[jac-fit] prompts={len(encoded)} skipped_short={skipped} "
        f"dim_batch={dim_batch} prompt_batch={prompt_batch} "
        f"passes/prompt_group={n_passes} lengths={len(by_len)}"
        f"{' (autotuned)' if tuned else ''}",
        flush=True,
    )

    total_groups = 0
    used_dim_batch = dim_batch
    for length in sorted(by_len):
        group = by_len[length]
        for start in range(0, len(group), max(1, prompt_batch)):
            chunk = group[start : start + prompt_batch]
            prompt_indices = [prompt_index for prompt_index, _ in chunk]
            batch_ids = torch.stack([ids for _, ids in chunk], dim=0).to(device)
            attn = torch.ones_like(batch_ids)
            try:
                per, used_d = _jacobian_with_oom_backoff(
                    model,
                    batch_ids,
                    sources=sources,
                    target=target,
                    dim_batch=dim_batch,
                    skip_first=skip_first,
                    attention_mask=attn,
                )
                used_dim_batch = min(used_dim_batch, used_d)
            except (ValueError, RuntimeError) as exc:
                if _is_oom(exc):
                    print(
                        f"[jac-fit] skip {len(chunk)} prompts after OOM at seq={length}",
                        flush=True,
                    )
                    skipped += len(chunk)
                    for prompt_index in prompt_indices:
                        reason = f"oom_after_backoff:seq_len={length}"
                        skipped_prompts.append(
                            {"prompt_index": prompt_index, "reason": reason}
                        )
                        print(
                            f"[jac-fit] skip prompt={prompt_index} reason={reason}",
                            flush=True,
                        )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                raise RuntimeError(
                    "Jacobian fit failed for "
                    f"{len(chunk)} prompt(s) at tokenized length {length}"
                ) from exc

            # ``per`` already averaged over the prompt chunk
            for li, mat in per.items():
                acc[li] += mat * len(chunk)  # weight by #prompts so mean is correct
            n_done += len(chunk)
            total_groups += 1
            print(
                f"[jac-fit] kept={n_done}/{len(encoded)} group_P={len(chunk)} "
                f"seq={length} groups={total_groups}",
                flush=True,
            )
            if device.type == "cuda" and total_groups % 10 == 0:
                torch.cuda.empty_cache()

    if n_done == 0:
        raise ValueError("no prompts were long enough to fit Jacobian lens")
    mean = {li: acc[li] / n_done for li in sources}
    return {
        "jacobians": mean,
        "n_prompts": n_done,
        "n_skipped": skipped,
        "skipped_prompts": skipped_prompts,
        "source_layers": sources,
        "target_layer": target,
        "d_model": d_model,
        "dim_batch": dim_batch,
        "dim_batch_used_min": used_dim_batch,
        "prompt_batch": prompt_batch,
        "autotuned": tuned,
        "max_seq_len": max_seq_len,
        "skip_first": skip_first,
        "layer_stride": layer_stride,
        "n_backward_passes_per_group": n_passes,
    }


def apply_jacobian_lens(
    hidden_states: Sequence[torch.Tensor],
    jacobians: dict[int, torch.Tensor],
    *,
    target_layer: int,
    layers: Sequence[int] | None = None,
) -> list[torch.Tensor]:
    """Transport requested residuals into the target-layer basis.

    ``hidden_states[i]`` is ``[seq, d]``. Identity is valid only for
    ``target_layer``. A requested source layer without a fitted Jacobian is an
    error; callers using a strided fit should pass its explicit source layers.
    """
    out: list[torch.Tensor] = []
    requested = list(range(len(hidden_states))) if layers is None else [int(x) for x in layers]
    for li in requested:
        if li < 0 or li >= len(hidden_states):
            raise IndexError(
                f"requested layer {li} outside hidden-state range 0..{len(hidden_states) - 1}"
            )
        h = hidden_states[li]
        if li == target_layer:
            out.append(h)
            continue
        if li not in jacobians:
            raise KeyError(
                f"missing Jacobian for source layer {li}; "
                f"available source layers: {sorted(jacobians)}, target layer: {target_layer}"
            )
        j = jacobians[li].to(device=h.device, dtype=torch.float32)
        if j.ndim != 2 or j.shape[0] != h.shape[-1] or j.shape[1] != h.shape[-1]:
            raise ValueError(
                f"Jacobian shape mismatch at layer {li}: J={tuple(j.shape)}, "
                f"hidden width={h.shape[-1]}"
            )
        transported = h.float() @ j.T
        out.append(transported.to(dtype=h.dtype))
    return out
