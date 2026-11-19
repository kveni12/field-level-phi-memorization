#!/usr/bin/env python3
"""Replot the full-n=800 final-model J-Lens MRR heatmap from its summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from memo_diff.analysis.primary_figure_fields import PRIMARY_ORDER

PHI = "protected_association"
BENIGN = "benign_arbitrary_association"
MEAN_KEYS = {
    PHI: "train_member:protected_association",
    BENIGN: "train_member:benign_arbitrary_association",
}
PANEL_TITLES = {
    PHI: "Simulated PHI",
    BENIGN: "Benign twin",
}
PANEL_TITLE_FONTSIZE = 17
SUPTITLE_FONTSIZE = 20
PANEL_COLORS = {PHI: "#b42318", BENIGN: "#175cd3"}
PHI_DISPLAY = {
    "health_plan_beneficiary_number": "health_plan_id",
}
BENIGN_DISPLAY = {
    "initials": "code",
    "date_of_birth": "install_date",
    "date_of_death": "decommission_date",
    "city": "facility_city",
    "state": "facility_state",
    "zip": "facility_zip",
    "phone": "hotline",
    "ssn": "account_number",
    "mrn": "facility_record_number",
    "health_plan_beneficiary_number": "plan_account_id",
    "admission_date": "inbound_date",
    "discharge_date": "outbound_date",
    "diagnosis": "care_pathway_label",
    "prescription": "protocol_name",
}


def _default_suptitle(summary: dict) -> str:
    del summary
    return "Jacobian-Lens memorization emergence across layers"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None, help="Figure suptitle override.")
    parser.add_argument(
        "--phi-panel-title",
        type=str,
        default=None,
        help="Left panel title override (default: Simulated PHI).",
    )
    parser.add_argument(
        "--benign-panel-title",
        type=str,
        default=None,
        help="Right panel title override (default: Benign twin).",
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    panel_titles = {
        PHI: args.phi_panel_title or PANEL_TITLES[PHI],
        BENIGN: args.benign_panel_title or PANEL_TITLES[BENIGN],
    }
    suptitle = args.title or _default_suptitle(summary)
    rows = [
        row
        for row in summary["statistical_analysis"]["first_data_token_mrr"]
        if row.get("scope") == "field"
    ]
    present = {str(row["field"]) for row in rows}
    fields = [field for field in PRIMARY_ORDER if field in present]
    layers = sorted({int(row["layer"]) for row in rows})
    by_cell = {(str(row["field"]), int(row["layer"])): row for row in rows}

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "axes.titleweight": "normal",
            "xtick.labelsize": 13,
            "ytick.labelsize": 15,
            "figure.dpi": 160,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharey=False)
    image = None
    for ax, label in zip(axes, (PHI, BENIGN)):
        matrix = np.asarray(
            [
                [
                    float(by_cell[(field, layer)]["means"][MEAN_KEYS[label]]["estimate"])
                    for layer in layers
                ]
                for field in fields
            ],
            dtype=float,
        )
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="viridis",
            interpolation="nearest",
        )
        display = PHI_DISPLAY if label == PHI else BENIGN_DISPLAY
        ax.set_title(
            panel_titles[label],
            color=PANEL_COLORS[label],
            pad=10,
            fontweight="normal",
            fontsize=PANEL_TITLE_FONTSIZE,
        )
        ax.set_xticks(range(len(layers)), [str(layer) for layer in layers], rotation=90)
        ax.set_yticks(range(len(fields)), [display.get(field, field) for field in fields])
        ax.set_xlabel("Source Layer", labelpad=10)
    axes[0].set_ylabel("Field", labelpad=10)

    if image is None:
        raise RuntimeError("no field-level MRR values found")
    fig.subplots_adjust(left=0.10, right=0.87, top=0.88, bottom=0.14, wspace=0.42)
    colorbar_axis = fig.add_axes([0.91, 0.17, 0.018, 0.62])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Mean Reciprocal Rank (First Data Token)", fontsize=18, labelpad=14)
    colorbar.ax.tick_params(labelsize=14)
    fig.suptitle(
        suptitle,
        fontsize=SUPTITLE_FONTSIZE,
        y=0.98,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {args.output} and {args.output.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
