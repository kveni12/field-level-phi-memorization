#!/usr/bin/env python3
"""Plot numerical J-Lens curves shared with the matched onset analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='Output stem or PNG filename; matching PDF is also written')
    args = p.parse_args()
    data = json.loads(args.input.read_text())
    fields, layers = data['field_order'], data['layers']
    panels = [('protected_association', 'Simulated PHI', '#b42318'),
              ('benign_arbitrary_association', 'Benign twin', '#175cd3')]
    assert len(fields) == 11 and layers == list(range(1, 33))
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), layout='constrained')
    checks = {}
    for ax, (label, title, color) in zip(axes, panels):
        values = np.asarray([data['mean_curves'][f][label] for f in fields])
        assert values.shape == (11, 32) and np.isfinite(values).all()
        assert ((values >= 0) & (values <= 1)).all()
        observed = {}
        for field, row in zip(fields, values):
            hit = np.flatnonzero(row >= data['threshold'])
            onset = layers[int(hit[0])] if hit.size else 33
            assert onset == data['expected_onsets'][field][label], (field, label, onset)
            observed[field] = onset
        checks[label] = observed
        im = ax.imshow(values, vmin=0, vmax=1, cmap='viridis', interpolation='nearest', aspect='auto')
        ax.set_yticks(range(len(fields)), [data['display'][label][f] for f in fields], fontsize=16)
        ticks = [0, 3, 7, 11, 15, 19, 23, 27, 31]
        ax.set_xticks(ticks, [layers[i] for i in ticks], fontsize=14)
        ax.tick_params(axis='both', length=3, pad=4)
        ax.set_xlabel('Source layer', fontsize=17)
        ax.set_title(title, color=color, fontsize=22, pad=10)
    axes[0].set_ylabel('Field', fontsize=17)
    cb = fig.colorbar(im, ax=axes, shrink=.85, pad=.025)
    cb.set_label('Mean reciprocal rank', fontsize=16, labelpad=10)
    cb.ax.tick_params(labelsize=14)
    fig.suptitle(data['title'], fontsize=25)
    stem = args.output.with_suffix('')
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(stem.with_suffix('.' + ext), dpi=190, bbox_inches='tight')
    plt.close(fig)
    stem.with_name(stem.name + '_verification.json').write_text(json.dumps({
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'n_matched_training_pairs': data['n_matched_training_pairs'],
        'all_22_field_label_onsets_match_figure4': True,
        'observed_onsets': checks,
    }, indent=2) + '\n')
    print('Rendered 11 fields × 32 layers; all 22 onset crossings match Figure 4.')


if __name__ == '__main__':
    main()
