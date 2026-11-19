#!/usr/bin/env python3
"""Reproduce archived paper heatmaps from preserved cell colors, not raw scores."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from memo_diff.analysis.paper_fields import PHI_DISPLAY, BENIGN_DISPLAY
from memo_diff.analysis.primary_figure_fields import PRIMARY_ORDER, ALREADY_IN_ATTACK_PREFIX
LABELS = [('protected_association', 'Simulated PHI', '#b42318'),
          ('benign_arbitrary_association', 'Benign twin', '#175cd3')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'figures')
    p.add_argument('--archived-first-token', action='store_true', help='Render the historical first-token-only Figure 3 instead of the updated suffix-scored version')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artwork = json.loads((ROOT / 'results/archived/artwork_inputs.json').read_text())
    specs = [
        ('prefix', 'memorization_performance_by_field',
         'Prefix-attack complete-field retention across checkpoints',
         'Complete-field retention (%)', 'YlOrRd', 100,
         [f for f in PRIMARY_ORDER if f not in ALREADY_IN_ATTACK_PREFIX]),
        ('jlens', 'mem_emergence', 'First data-token J-Lens readout across layers',
         'Mean reciprocal rank (first data token)', 'viridis', 1, PRIMARY_ORDER),
    ]
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    for kind, stem, title, bar_title, cmap, maximum, fields in specs:
        if kind == 'jlens' and not args.archived_first_token:
            subprocess.run([sys.executable, str(ROOT / 'scripts/plot_jlens_matched_heatmap.py'),
                '--input', str(ROOT / 'results/current/jlens_heatmap_inputs.json'),
                '--output', str(args.output_dir / 'mem_emergence.png')], check=True)
            continue
        source = artwork[kind]
        indices = [source['fields'].index(f) for f in fields]
        fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), layout='constrained')
        for ax, (label, panel, color) in zip(axes, LABELS):
            rgb = np.asarray(source['rgb_panels'][label], dtype=np.uint8)[indices]
            ax.imshow(rgb, aspect='auto', interpolation='nearest')
            display = PHI_DISPLAY if label == LABELS[0][0] else BENIGN_DISPLAY
            ax.set_yticks(range(len(fields)), [display.get(f, f) for f in fields], fontsize=16)
            ticks = list(range(len(source['columns']))) if kind == 'prefix' else [0, 3, 7, 11, 15, 19, 23, 27, 31]
            ax.set_xticks(ticks, [source['columns'][i] for i in ticks],
                          rotation=45 if kind == 'prefix' else 0, fontsize=14)
            ax.tick_params(axis='both', length=3, pad=4)
            ax.set_xlabel('Training step' if kind == 'prefix' else 'Source layer', fontsize=17)
            ax.set_title(panel, color=color, fontsize=22, pad=10)
        axes[0].set_ylabel('Field', fontsize=17)
        image = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, maximum), cmap=cmap)
        cb = fig.colorbar(image, ax=axes, shrink=.85, pad=.025)
        cb.set_label(bar_title, fontsize=16, labelpad=10)
        cb.ax.tick_params(labelsize=14)
        fig.suptitle(title, fontsize=25)
        for ext in ('png', 'pdf'):
            fig.savefig(args.output_dir / f'{stem}.{ext}', dpi=190, bbox_inches='tight')
        plt.close(fig)
    source = ROOT / 'results/archived/memorization_performance_overall.png'
    target = args.output_dir / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    print('Reproduced the prefix heatmap, selected J-Lens heatmap, and appendix panel.')


if __name__ == '__main__':
    main()
