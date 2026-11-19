#!/usr/bin/env python3
"""Render the overall and per-field panels as standalone figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCRIPT_DIR))

from memo_diff.analysis.prefix_plotting import (  # noqa: E402
    BENIGN_DISPLAY,
    BENIGN_LABEL,
    HEALTH_RELATED_DISPLAY,
    PHI_DISPLAY_SHORT,
    PHI_LABEL,
    TWIN_TO_BENIGN_DISPLAY,
    _bootstrap_exact_ci,
    _bootstrap_micro_field_ci,
    _collect_fields,
    _field_n_text,
    _matrix,
    _overall_marker_style,
    _paired_field_n_text,
    _paired_heatmap_fields,
    _samples_per_label,
    _step_dodge,
    load_rows,
)


OVERALL_TITLE = "Overall Prefix Attack Performance Across Training Checkpoints"
FIELD_TITLE = "Prefix Attack Performance by Field Across Training Checkpoints"


def _configure_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )


def _mean_series(rows: list[dict], label: str, key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        block = (row.get("per_label") or {}).get(label, {})
        value = block.get(key)
        if key == "micro_field_instance_acc" and value is None:
            value = block.get("mean_field_acc")
        values.append(100.0 * float(value or 0.0))
    return values


def plot_overall(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    steps = [int(row["step"]) for row in rows]
    n_phi = _samples_per_label(rows, PHI_LABEL)
    n_benign = _samples_per_label(rows, BENIGN_LABEL)

    fig, ax = plt.subplots(figsize=(14.2, 4.6))
    for label, name, color, n_dossiers in [
        (PHI_LABEL, HEALTH_RELATED_DISPLAY, "#b42318", n_phi),
        (BENIGN_LABEL, BENIGN_DISPLAY, "#175cd3", n_benign),
    ]:
        n_fields = int(
            (rows[0].get("per_label") or {})
            .get(label, {})
            .get("field_instances_total")
            or 0
        )
        x = _step_dodge(steps, label)
        ax.plot(
            x,
            _mean_series(rows, label, "micro_field_instance_acc"),
            color=color,
            lw=2.0,
            ls="-",
            label=(
                f"{name} field retention "
                f"(n={n_dossiers}; {n_fields:,} eligible fields)"
            ),
            **_overall_marker_style(label, dashed=False),
        )
        cis = [
            _bootstrap_micro_field_ci(
                (row.get("per_label") or {}).get(label, {}),
                seed=12_345 + (0 if label == PHI_LABEL else 1),
            )
            for row in rows
        ]
        if all(ci is not None for ci in cis):
            ax.fill_between(
                x,
                [ci[0] for ci in cis if ci is not None],
                [ci[1] for ci in cis if ci is not None],
                color=color,
                alpha=0.14,
                linewidth=0,
                zorder=1,
            )

    for label, name, color, n_dossiers in [
        (PHI_LABEL, HEALTH_RELATED_DISPLAY, "#b42318", n_phi),
        (BENIGN_LABEL, BENIGN_DISPLAY, "#175cd3", n_benign),
    ]:
        x = _step_dodge(steps, label)
        ax.plot(
            x,
            _mean_series(rows, label, "exact_acc"),
            color=color,
            lw=2.0,
            ls="--",
            label=f"{name} exact suffix (n={n_dossiers})",
            **_overall_marker_style(label, dashed=True),
        )
        cis = [
            _bootstrap_exact_ci(
                (row.get("per_label") or {}).get(label, {}),
                seed=54_321 + (0 if label == PHI_LABEL else 1),
            )
            for row in rows
        ]
        if all(ci is not None for ci in cis):
            ax.fill_between(
                x,
                [ci[0] for ci in cis if ci is not None],
                [ci[1] for ci in cis if ci is not None],
                color=color,
                alpha=0.14,
                linewidth=0,
                zorder=1,
            )

    ax.set_ylim(-2, 105)
    ax.set_xlabel("Training checkpoint")
    ax.set_ylabel("Retention (%)")
    ax.legend(frameon=False, fontsize=10, loc="upper left", ncol=2)
    fig.suptitle(OVERALL_TITLE, fontsize=19, y=0.98, fontweight="semibold")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.17)
    _save(fig, output)


def plot_by_field(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    steps = [int(row["step"]) for row in rows]
    fields = _paired_heatmap_fields(rows, PHI_LABEL, BENIGN_LABEL)
    if not fields:
        raise SystemExit("no shared PHI/benign fields for side-by-side heatmaps")

    _, phi_matrix = _matrix(rows, PHI_LABEL, fields)
    _, benign_matrix = _matrix(rows, BENIGN_LABEL, fields)
    phi_labels = [PHI_DISPLAY_SHORT.get(field, field) for field in fields]
    benign_labels = [TWIN_TO_BENIGN_DISPLAY.get(field, field) for field in fields]

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.6))
    tick_every = max(1, len(steps) // 8)
    xticks = list(range(0, len(steps), tick_every))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#d0d5dd")
    images = []
    for ax, data, panel_title, color, labels in [
        (
            axes[0],
            np.asarray(phi_matrix, dtype=float),
            f"{HEALTH_RELATED_DISPLAY} per-field retention",
            "#b42318",
            phi_labels,
        ),
        (
            axes[1],
            np.asarray(benign_matrix, dtype=float),
            f"{BENIGN_DISPLAY} per-field retention",
            "#175cd3",
            benign_labels,
        ),
    ]:
        image = ax.imshow(
            data,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=100,
            origin="lower",
            interpolation="nearest",
        )
        images.append(image)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10.5)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(steps[index]) for index in xticks], rotation=45, ha="right")
        ax.set_xlabel("Training step", fontsize=13)
        ax.set_title(panel_title, color=color, fontsize=16, pad=8)

    colorbar_axis = fig.add_axes([0.915, 0.13, 0.018, 0.68])
    colorbar = fig.colorbar(images[1], cax=colorbar_axis)
    colorbar.set_label("Field retention (%)", fontsize=12)
    fig.suptitle(FIELD_TITLE, fontsize=23, y=0.985, fontweight="normal")
    fig.subplots_adjust(left=0.11, right=0.88, top=0.89, bottom=0.12, wspace=0.78)
    _save(fig, output)


def _save(fig, output: Path) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output} and {output.with_suffix('.pdf')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--overall-output", type=Path, required=True)
    parser.add_argument("--by-field-output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.jsonl)
    if not rows:
        raise SystemExit("No rows to plot")
    _configure_matplotlib()
    plot_overall(rows, args.overall_output)
    plot_by_field(rows, args.by_field_output)


if __name__ == "__main__":
    main()
