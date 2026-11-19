#!/usr/bin/env python3
"""Merge disjoint prompt-sharded Jacobian fits by their prompt counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    args = parser.parse_args()

    paths = sorted(args.shards_dir.glob("shard_*/jacobians.pt"))
    if len(paths) != args.expected_shards:
        raise SystemExit(f"expected {args.expected_shards} shards, found {len(paths)}")

    invariant_manifest_keys = (
        "schema_version",
        "checkpoint_path",
        "model_config_sha256",
        "tokenizer_sha256",
        "hidden_size",
        "num_hidden_layers",
        "target_layer",
        "source_layers",
        "fit_source",
        "fit_corpus_sha256",
        "fit_prompt_count_requested",
        "max_seq_len",
        "skip_first",
        "add_special_tokens",
        "recipe",
    )
    accumulator: dict[int, torch.Tensor] | None = None
    first: dict | None = None
    first_manifest: dict | None = None
    total_prompts = 0
    total_skipped = 0
    shard_indices: list[int] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        manifest = payload.get("manifest") or {}
        if first is None:
            first = payload
            first_manifest = manifest
            accumulator = {
                int(layer): matrix.float().mul_(int(payload["n_prompts"]))
                for layer, matrix in payload["jacobians"].items()
            }
        else:
            assert first_manifest is not None and accumulator is not None
            mismatches = [
                key
                for key in invariant_manifest_keys
                if manifest.get(key) != first_manifest.get(key)
            ]
            if mismatches:
                raise SystemExit(f"incompatible shard {path}: {mismatches}")
            if sorted(int(x) for x in payload["jacobians"]) != sorted(accumulator):
                raise SystemExit(f"source-layer mismatch in {path}")
            weight = int(payload["n_prompts"])
            for layer, matrix in payload["jacobians"].items():
                accumulator[int(layer)].add_(matrix.float(), alpha=weight)
        total_prompts += int(payload["n_prompts"])
        total_skipped += int(payload.get("n_skipped", 0))
        shard_indices.append(int(manifest.get("fit_shard_index", -1)))
        del payload

    if sorted(shard_indices) != list(range(args.expected_shards)):
        raise SystemExit(f"fit shard indices are incomplete: {sorted(shard_indices)}")
    assert first is not None and first_manifest is not None and accumulator is not None
    requested = int(first_manifest["fit_prompt_count_requested"])
    if total_prompts != requested:
        raise SystemExit(f"merged {total_prompts} prompts, expected {requested}")
    jacobians = {layer: matrix.div_(total_prompts) for layer, matrix in accumulator.items()}
    manifest = dict(first_manifest)
    manifest.update(
        {
            "fit_prompt_count_used": total_prompts,
            "fit_shard_index": None,
            "fit_shard_count": args.expected_shards,
            "fit_merge": "prompt_count_weighted_mean",
        }
    )
    output_payload = {
        key: value
        for key, value in first.items()
        if key not in {"jacobians", "manifest", "n_prompts", "n_skipped", "skipped_prompts"}
    }
    output_payload.update(
        {
            "jacobians": jacobians,
            "manifest": manifest,
            "n_prompts": total_prompts,
            "n_skipped": total_skipped,
            "skipped_prompts": [],
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "jacobians.pt"
    temporary = output.with_suffix(".pt.tmp")
    torch.save(output_payload, temporary)
    temporary.replace(output)
    summary = {
        "output": str(output),
        "bytes": output.stat().st_size,
        "manifest": manifest,
        "fit": {key: value for key, value in output_payload.items() if key != "jacobians"},
        "source_shards": [str(path) for path in paths],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "n_prompts": total_prompts}), flush=True)


if __name__ == "__main__":
    main()
