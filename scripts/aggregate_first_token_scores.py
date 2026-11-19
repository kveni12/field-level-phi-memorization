#!/usr/bin/env python3
"""Aggregate disjoint first-token score rows for numerical heatmap plotting."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    values = defaultdict(dict)
    steps = set()
    target_kinds = defaultdict(set)
    for path in a.input:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get('analysis') != 'preanswer_jlens':
                continue
            if row.get('evaluation_split', 'train_member') != 'train_member':
                continue
            key = (row['field'], int(row['layer']), row['label'])
            example = row['example_id']
            if example in values[key]:
                raise ValueError(f'duplicate score identity: {key}, {example}')
            values[key][example] = float(row['reciprocal_rank'])
            target_kinds[row['field']].add(row.get('target_kind', 'first_data_token'))
            steps.add(row['step'])
    if len(steps) != 1:
        raise ValueError(f'expect one explicit training step; found {steps}')
    if any(len(kinds) != 1 for kinds in target_kinds.values()):
        raise ValueError('Cannot combine different target metrics for one field')
    cells = sorted({(f, l) for f, l, _ in values})
    result = []
    for field, layer in cells:
        means = {}
        for label in ('protected_association', 'benign_arbitrary_association'):
            samples = values[field, layer, label]
            if not samples:
                raise ValueError(f'missing label {label} for {field}, layer {layer}')
            means[f'train_member:{label}'] = {'estimate': sum(samples.values()) / len(samples), 'n': len(samples)}
        result.append({'scope': 'field', 'field': field, 'layer': layer, 'means': means})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'step': steps.pop(), 'target_kinds': {f: sorted(k) for f, k in target_kinds.items()}, 'statistical_analysis':
                                  {'first_data_token_mrr': result}}, indent=2) + '\n')


if __name__ == '__main__':
    main()
