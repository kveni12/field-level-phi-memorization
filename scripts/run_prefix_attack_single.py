#!/usr/bin/env python3
"""Run prefix attack on a single checkpoint step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memo_diff.training.prefix_attack import (  # noqa: E402
    AttackConfig,
    _load_canaries,
    _save_details,
    eval_checkpoint,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Local checkpoint directory (step_XXXXXX).",
    )
    p.add_argument(
        "--hf-repo",
        default=None,
        help="Hugging Face repo id (use with --revision instead of --checkpoint).",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="HF revision/branch, e.g. step-000600.",
    )
    p.add_argument("--step", type=int, default=None)
    p.add_argument(
        "--canaries",
        type=Path,
        default=ROOT
        / "data/training/llama31_8b_balanced_2k_phi_benign_v2_luna_9f/train_canaries.jsonl",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--samples-per-label", type=int, default=80)
    p.add_argument("--prefix-frac", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--add-special-tokens", action="store_true")
    p.add_argument(
        "--save-details",
        action="store_true",
        help="Save all per-example greedy generations even when evaluating more than 100 samples.",
    )
    p.add_argument(
        "--membership",
        choices=("train", "heldout", "all"),
        default="train",
        help="Select records by is_train_member (default: train).",
    )
    args = p.parse_args()

    import torch
    from transformers import AutoTokenizer

    if args.checkpoint and args.hf_repo:
        raise SystemExit("pass --checkpoint or --hf-repo, not both")
    if not args.checkpoint and not args.hf_repo:
        raise SystemExit("need --checkpoint or --hf-repo")

    ckpt_path: Path | str
    if args.hf_repo:
        ckpt_path = args.hf_repo
        rev = args.revision
        if rev is None and args.step is not None:
            rev = f"step-{args.step:06d}"
        if rev is None:
            raise SystemExit("--hf-repo requires --revision or --step")
        step = args.step
        if step is None:
            try:
                step = int(rev.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                step = 0
        ckpt_label = f"{args.hf_repo}@{rev}"
        ckpt_parent = Path(".")
    else:
        ckpt_path = Path(args.checkpoint)
        rev = None
        step = args.step
        if step is None:
            try:
                step = int(ckpt_path.name.split("_")[1])
            except (IndexError, ValueError):
                step = 0
        ckpt_label = str(ckpt_path)
        ckpt_parent = ckpt_path.parent

    cfg = AttackConfig(
        canaries_path=args.canaries,
        checkpoint_dir=ckpt_parent,
        output_dir=args.output_dir,
        model_name=args.model_name,
        samples_per_label=args.samples_per_label,
        prefix_frac=args.prefix_frac,
        add_special_tokens=args.add_special_tokens,
        batch_size=args.batch_size,
        save_example_details=True if args.save_details else None,
        membership=args.membership,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    canaries = _load_canaries(
        cfg.canaries_path,
        samples_per_label=cfg.samples_per_label,
        seed=cfg.seed,
        membership=cfg.membership,
    )

    print(f"[attack] step={step} ckpt={ckpt_label} n/label={cfg.samples_per_label}", flush=True)
    if args.hf_repo:
        from memo_diff.training.prefix_attack import score_loaded_model, _disable_triton

        _disable_triton()
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.hf_repo,
            revision=rev,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        model.to(device)
        model.eval()
        metrics = score_loaded_model(
            model=model,
            tokenizer=tokenizer,
            canaries_by_label=canaries,
            cfg=cfg,
            device=device,
            eval_step=step,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        from memo_diff.training.prefix_attack import _disable_triton

        _disable_triton()
        metrics = eval_checkpoint(
            ckpt_path=Path(ckpt_path),
            tokenizer=tokenizer,
            canaries_by_label=canaries,
            cfg=cfg,
            device=device,
            eval_step=step,
        )
    row = {"step": step, "checkpoint": ckpt_label, **metrics}
    (args.output_dir / "prefix_attack_by_checkpoint.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    if _save_details(cfg):
        (args.output_dir / "prefix_attack_full.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )

    per_label = {}
    for lab, block in metrics.get("per_label", {}).items():
        per_label[lab] = {
            "n": block.get("n"),
            "mean_field_acc": block.get("mean_field_acc"),
            "macro_example_field_acc": block.get("macro_example_field_acc"),
            "micro_field_instance_acc": block.get("micro_field_instance_acc"),
            "field_instances_correct": block.get("field_instances_correct"),
            "field_instances_total": block.get("field_instances_total"),
            "exact_acc": block.get("exact_acc"),
            "mean_token_acc": block.get("mean_token_acc"),
        }
    summary = {
        "checkpoint": ckpt_label,
        "step": step,
        "prefix_frac": cfg.prefix_frac,
        "samples_per_label": cfg.samples_per_label,
        "per_label": per_label,
    }
    (args.output_dir / "prefix_attack_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
