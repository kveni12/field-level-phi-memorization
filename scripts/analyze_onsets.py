#!/usr/bin/env python3
"""Compute onset statistics reported in the paper; no onset figure is rendered."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PHI = "protected_association"
BENIGN = "benign_arbitrary_association"
FIELD_DISPLAY = {
    "initials": "initials ↔ code",
    "date_of_birth": "date of birth ↔ install date",
    "date_of_death": "date of death ↔ decommission date",
    "city": "city ↔ facility city",
    "state": "state ↔ facility state",
    "zip": "ZIP ↔ facility ZIP",
    "phone": "phone ↔ hotline",
    "ssn": "SSN ↔ account number",
    "mrn": "MRN ↔ facility record number",
    "health_plan_beneficiary_number": "health-plan number ↔ plan account ID",
    "admission_date": "admission ↔ inbound date",
    "discharge_date": "discharge ↔ outbound date",
    "diagnosis": "diagnosis ↔ care pathway",
    "prescription": "prescription ↔ protocol",
}
FIELD_ORDER = tuple(FIELD_DISPLAY)


def _onset(
    trajectories: np.ndarray,
    threshold: float,
    layers: np.ndarray,
    censored: int,
) -> np.ndarray:
    crossed = trajectories >= threshold
    first = crossed.argmax(axis=1)
    return np.where(crossed.any(axis=1), layers[first], censored).astype(int)


def _percentile(values: np.ndarray, alpha: float) -> tuple[float, float]:
    return (
        float(np.quantile(values, alpha / 2, method="inverted_cdf")),
        float(np.quantile(values, 1 - alpha / 2, method="inverted_cdf")),
    )


def _fmt_ci(ci: list[float]) -> str:
    return f"[{ci[0]:.0f}, {ci[1]:.0f}]"


def _fmt_mean_ci(ci: list[float]) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    # values[label][field][pair_id][layer] = reciprocal rank
    values: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    observed_layers: set[int] = set()
    n_preanswer_rows = 0
    target_kinds = set()
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("analysis") != "preanswer_jlens":
                continue
            n_preanswer_rows += 1
            target_kinds.add(str(row.get("target_kind", "first_data_token")))
            label = str(row["label"])
            field = str(row["field"])
            pair = str(row.get("pair_id") or row.get("example_id"))
            layer = int(row["layer"])
            observed_layers.add(layer)
            vector = values[label][field].setdefault(
                pair, np.full(33, np.nan, dtype=np.float64)
            )
            vector[layer] = float(row["reciprocal_rank"])

    fields = [field for field in FIELD_ORDER if field in values[PHI] and field in values[BENIGN]]
    shared_pairs = sorted(
        set.intersection(
            *(set(values[label][field]) for label in (PHI, BENIGN) for field in fields)
        )
    )
    if len(shared_pairs) != 800:
        raise RuntimeError(f"expected 800 complete matched pairs, found {len(shared_pairs)}")
    layers = np.asarray(sorted(observed_layers), dtype=int)
    if layers.size == 0:
        raise RuntimeError("no preanswer J-lens layers found")
    censored_layer = int(layers[-1]) + 1
    rng = np.random.default_rng(args.seed)
    weights = rng.multinomial(
        len(shared_pairs),
        np.full(len(shared_pairs), 1 / len(shared_pairs)),
        size=args.bootstrap_samples,
    ).astype(np.float64)

    alpha = 0.05
    adjusted_alpha = alpha / len(fields)
    results: list[dict[str, Any]] = []
    phi_boot_all = []
    benign_boot_all = []
    for field in fields:
        phi = np.stack(
            [values[PHI][field][pair][layers] for pair in shared_pairs]
        )
        benign = np.stack(
            [values[BENIGN][field][pair][layers] for pair in shared_pairs]
        )
        if np.isnan(phi).any() or np.isnan(benign).any():
            raise RuntimeError(f"incomplete trajectory for field {field}")
        phi_mean = phi.mean(axis=0)
        benign_mean = benign.mean(axis=0)
        phi_est = int(
            _onset(phi_mean[None, :], args.threshold, layers, censored_layer)[0]
        )
        benign_est = int(
            _onset(benign_mean[None, :], args.threshold, layers, censored_layer)[0]
        )
        phi_boot = _onset(
            (weights @ phi) / len(shared_pairs),
            args.threshold,
            layers,
            censored_layer,
        )
        benign_boot = _onset(
            (weights @ benign) / len(shared_pairs),
            args.threshold,
            layers,
            censored_layer,
        )
        gap_boot = benign_boot - phi_boot
        phi_boot_all.append(phi_boot)
        benign_boot_all.append(benign_boot)
        gap_ci = _percentile(gap_boot, alpha)
        gap_ci_adjusted = _percentile(gap_boot, adjusted_alpha)
        results.append(
            {
                "field": field,
                "display": FIELD_DISPLAY[field],
                "phi_onset": phi_est,
                "phi_onset_ci95": list(_percentile(phi_boot, alpha)),
                "benign_onset": benign_est,
                "benign_onset_ci95": list(_percentile(benign_boot, alpha)),
                "gap_benign_minus_phi": benign_est - phi_est,
                "gap_ci95": list(gap_ci),
                "gap_ci95_bonferroni_14": list(gap_ci_adjusted),
                "pointwise_ci_excludes_zero": gap_ci[0] > 0 or gap_ci[1] < 0,
                "bonferroni_ci_excludes_zero": gap_ci_adjusted[0] > 0
                or gap_ci_adjusted[1] < 0,
                "bootstrap_probability_gap_positive": float((gap_boot > 0).mean()),
                "bootstrap_probability_gap_zero": float((gap_boot == 0).mean()),
                "bootstrap_probability_gap_negative": float((gap_boot < 0).mean()),
            }
        )

    phi_boot_matrix = np.stack(phi_boot_all)
    benign_boot_matrix = np.stack(benign_boot_all)
    mean_phi_boot = phi_boot_matrix.mean(axis=0)
    mean_benign_boot = benign_boot_matrix.mean(axis=0)
    mean_gap_boot = (benign_boot_matrix - phi_boot_matrix).mean(axis=0)
    onset_summary = {
        "phi_mean_onset": float(np.mean([row["phi_onset"] for row in results])),
        "phi_mean_onset_ci95": list(_percentile(mean_phi_boot, alpha)),
        "benign_mean_onset": float(np.mean([row["benign_onset"] for row in results])),
        "benign_mean_onset_ci95": list(_percentile(mean_benign_boot, alpha)),
        "mean_gap_benign_minus_phi": float(
            np.mean([row["gap_benign_minus_phi"] for row in results])
        ),
        "mean_gap_ci95": list(_percentile(mean_gap_boot, alpha)),
    }

    output = {
        "step": args.step,
        "method": "jacobian_lens",
        "target_kinds": sorted(target_kinds),
        "evaluation_split": "train_member_only",
        "n_pairs": len(shared_pairs),
        "n_fields": len(fields),
        "n_preanswer_rows": n_preanswer_rows,
        "threshold": args.threshold,
        "bootstrap_unit": "matched_pair_id",
        "bootstrap_samples": args.bootstrap_samples,
        "ci_method": "paired nonparametric percentile bootstrap",
        "multiple_comparison_interval": "Bonferroni alpha=0.05/14",
        "onset_summary": onset_summary,
        "fields": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    md = args.output.with_suffix(".md")
    lines = [
        "# J-lens emergence-onset confidence intervals",
        "",
        f"Threshold: cohort mean reciprocal rank ≥ {args.threshold}; n={len(shared_pairs)} paired train records; {args.bootstrap_samples} paired bootstrap resamples.",
        "",
        f"- Mean PHI onset: **{onset_summary['phi_mean_onset']:.2f}**, 95% CI {_fmt_mean_ci(onset_summary['phi_mean_onset_ci95'])}",
        f"- Mean benign onset: **{onset_summary['benign_mean_onset']:.2f}**, 95% CI {_fmt_mean_ci(onset_summary['benign_mean_onset_ci95'])}",
        f"- Mean paired gap (benign−PHI): **{onset_summary['mean_gap_benign_minus_phi']:+.2f}**, 95% CI {_fmt_mean_ci(onset_summary['mean_gap_ci95'])}",
        "",
        "| Field pair | PHI onset [95% CI] | Benign onset [95% CI] | Gap | Gap 95% CI | Bonferroni CI | Adjusted significant? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    lines.extend(
        f"| {row['display']} | {row['phi_onset']} {_fmt_ci(row['phi_onset_ci95'])} | {row['benign_onset']} {_fmt_ci(row['benign_onset_ci95'])} | {row['gap_benign_minus_phi']:+d} | {_fmt_ci(row['gap_ci95'])} | {_fmt_ci(row['gap_ci95_bonferroni_14'])} | {'yes' if row['bonferroni_ci_excludes_zero'] else 'no'} |"
        for row in results
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[bootstrap] wrote "
        f"{args.output.with_suffix('.json')}, and {md}"
    )


if __name__ == "__main__":
    main()
