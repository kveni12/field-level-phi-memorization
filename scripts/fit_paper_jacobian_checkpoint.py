#!/usr/bin/env python3
"""Fit and save a provenance-validated J-Lens Jacobian for one checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memo_diff.analysis.fit_corpus import _load_fit_texts
from memo_diff.analysis.jacobian_lens import (
    build_lens_manifest,
    fit_jacobian_lens,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-texts", type=Path, required=True)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--fit-prompts", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--dim-batch", type=int, default=0)
    parser.add_argument("--prompt-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-layer",
        choices=("final", "penultimate"),
        default="final",
        help="Residual-stream endpoint: HF state 32 or 31 for a 32-block Llama.",
    )
    parser.add_argument("--fit-shard-index", type=int, default=0)
    parser.add_argument("--fit-shard-count", type=int, default=1)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from torch._native.registry import deregister_op_overrides

        deregister_op_overrides(disable_dsl_names=["triton", "cutedsl"])
    except Exception:
        pass

    if not args.fit_texts.is_file():
        raise SystemExit(f"fit corpus missing: {args.fit_texts}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        attn_implementation="eager",
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    all_fit_texts, fit_source = _load_fit_texts(
        args.fit_texts,
        n=args.fit_prompts,
        seed=args.seed,
    )
    if args.fit_shard_count < 1 or not 0 <= args.fit_shard_index < args.fit_shard_count:
        raise SystemExit(
            f"invalid fit shard {args.fit_shard_index}/{args.fit_shard_count}"
        )
    fit_texts = all_fit_texts[args.fit_shard_index :: args.fit_shard_count]
    if not fit_texts:
        raise SystemExit("fit shard contains no prompts")
    # HF hidden_states: 0 is embedding output, 1..32 are block outputs.
    # Keep the endpoint explicit: final targets the actual model residual after
    # block 31 (HF state 32); penultimate targets HF state 31.
    n_states = int(model.config.num_hidden_layers) + 1
    target_layer = n_states - 1 if args.target_layer == "final" else n_states - 2
    source_layers = list(range(1, target_layer))
    print(
        f"[paper-jac] step={args.step} prompts={len(fit_texts)} "
        f"max_seq_len={args.max_seq_len} skip_first={args.skip_first} "
        f"target_hf={target_layer} sources_hf={source_layers[0]}..{source_layers[-1]}",
        flush=True,
    )
    fitted = fit_jacobian_lens(
        model,
        tokenizer,
        fit_texts,
        dim_batch=args.dim_batch,
        prompt_batch=args.prompt_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        add_special_tokens=True,
        target_layer=target_layer,
        source_layers=source_layers,
    )
    manifest = build_lens_manifest(
        model=model,
        tokenizer=tokenizer,
        checkpoint_path=args.checkpoint,
        # Record the complete selected corpus in every shard manifest. The fit
        # count below still records how many prompts this shard actually used.
        fit_texts=all_fit_texts,
        fit_source=fit_source,
        fitted=fitted,
        add_special_tokens=True,
    )
    manifest.update(
        {
            "training_step": args.step,
            "recipe": f"reference_1000x128_skip16_{args.target_layer}",
            "paper_layer_convention": "source_0_is_first_transformer_block_output",
            "hf_embedding_output_excluded": True,
            "fit_shard_index": args.fit_shard_index,
            "fit_shard_count": args.fit_shard_count,
        }
    )
    payload = {key: value for key, value in fitted.items() if key != "jacobians"}
    payload.update({"jacobians": fitted["jacobians"], "manifest": manifest})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "jacobians.pt"
    temporary = args.output_dir / "jacobians.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(output)
    summary = {
        "output": str(output),
        "bytes": output.stat().st_size,
        "manifest": manifest,
        "fit": {key: value for key, value in fitted.items() if key != "jacobians"},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
