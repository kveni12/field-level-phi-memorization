#!/usr/bin/env python3
"""Optional historical name-generation procedure; frozen names need no API."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', required=True, help='Model name available in your API account; the paper records gpt-5.6-luna')
    args = p.parse_args()
    if args.output.exists():
        p.error('output already exists')
    from memo_diff.dataset.luna_pools import build_luna_name_pools
    build_luna_name_pools(n_pairs=1000, model=args.model, seed=42, output_path=args.output)


if __name__ == '__main__':
    main()
