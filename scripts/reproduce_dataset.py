#!/usr/bin/env python3
"""Rebuild exact paper document rows from the frozen structured/name pools."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from memo_diff.dataset.generate import GenerateConfig, generate_rows, write_jsonl
from memo_diff.dataset.pools import load_twin_pools
from memo_diff.dataset.validate import validate_twin_corpus


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / 'data/training/llama31_8b_balanced_2k_phi_benign_v2_luna_9f')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.resolve() == args.source.resolve():
        p.error('use a separate output directory to preserve the frozen corpus')
    rows = generate_rows(GenerateConfig(n_pairs=1000, train_pairs=800, seed=42),
                         load_twin_pools(args.source / 'pools.json'))
    report = validate_twin_corpus(rows)
    if not report.ok:
        raise ValueError(report.errors)
    for split, member in [('train', True), ('heldout', False)]:
        produced = [row for row in rows if row['is_train_member'] == member]
        reference = [json.loads(line) for line in (args.source / f'{split}_canaries.jsonl').read_text().splitlines()]
        if produced != reference:
            raise ValueError(f'{split} rows differ from frozen corpus')
        write_jsonl(args.output_dir / f'{split}_canaries.jsonl', produced)
    print('Verified exact reproduction: 800 matched training pairs and 200 held-out pairs.')


if __name__ == '__main__':
    main()
