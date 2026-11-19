#!/usr/bin/env python3
"""Full-weight Llama 3.1 8B Instruct SFT on clinical canaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memo_diff.training.train_sft import SFTConfig, train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_llama31_8b_balanced_2k_phi_benign_v2_luna_9f.yaml"),
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--init-from", type=Path, default=None)
    args = p.parse_args()

    cfg = SFTConfig.from_yaml(args.config)
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir)
    if args.max_epochs is not None:
        cfg.max_epochs = int(args.max_epochs)
    if args.no_resume:
        cfg.resume = False
    if args.no_wandb:
        cfg.wandb = False
    if args.init_from is not None:
        cfg.init_from = str(args.init_from)
    summary = train(cfg)
    print(summary)


if __name__ == "__main__":
    main()
