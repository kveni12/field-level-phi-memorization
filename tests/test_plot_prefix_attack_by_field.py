import importlib.util
import math
from pathlib import Path


from memo_diff.analysis import prefix_plotting as MODULE


def test_missing_heatmap_stat_is_nan_not_zero() -> None:
    rows = [
        {"step": 30, "per_label": {"protected_association": {"per_field_acc": {}}}},
        {
            "step": 60,
            "per_label": {
                "protected_association": {
                    "per_field_acc": {"ssn": {"acc": 0.0, "n": 2, "correct": 0}}
                }
            },
        },
    ]
    _, matrix = MODULE._matrix(rows, "protected_association", ["ssn"])
    assert math.isnan(matrix[0][0])
    assert matrix[0][1] == 0.0


def test_overall_curve_prefers_micro_metric() -> None:
    row = {
        "per_label": {
            "protected_association": {
                "mean_field_acc": 0.5,
                "micro_field_instance_acc": 1 / 11,
            }
        }
    }
    assert MODULE._overall_field_acc(row, "protected_association") == 1 / 11


def test_field_n_text_reports_constant_or_varying_denominator() -> None:
    rows = [
        {
            "per_label": {
                "protected_association": {
                    "per_field_acc": {"ssn": {"n": n, "acc": 0.0, "correct": 0}}
                }
            }
        }
        for n in (8, 8)
    ]
    assert MODULE._field_n_text(rows, "protected_association", "ssn") == "n=8"
    rows[1]["per_label"]["protected_association"]["per_field_acc"]["ssn"]["n"] = 9
    assert MODULE._field_n_text(rows, "protected_association", "ssn") == "n=8–9"


def test_bootstrap_micro_field_ci_resamples_dossiers() -> None:
    block = {
        "details": [
            {"field_match": {"fields_correct": 2, "fields_total": 2}},
            {"field_match": {"fields_correct": 0, "fields_total": 2}},
        ]
    }
    ci = MODULE._bootstrap_micro_field_ci(block, n_boot=1_000, seed=7)
    assert ci is not None
    assert ci[0] <= 50.0 <= ci[1]


def test_bootstrap_exact_ci_contains_observed_accuracy() -> None:
    block = {"details": [{"exact": True}, {"exact": False}, {"exact": False}]}
    ci = MODULE._bootstrap_exact_ci(block, n_boot=1_000, seed=8)
    assert ci is not None
    assert ci[0] <= 100 / 3 <= ci[1]


def test_bootstrap_per_field_ci_ignores_ineligible_dossiers() -> None:
    block = {
        "details": [
            {"field_match": {"per_field": {"ssn": True}}},
            {"field_match": {"per_field": {"ssn": False}}},
            {"field_match": {"per_field": {"mrn": True}}},
        ]
    }
    ci = MODULE._bootstrap_per_field_ci(block, "ssn", n_boot=1_000, seed=9)
    assert ci is not None
    assert ci[0] <= 50.0 <= ci[1]
