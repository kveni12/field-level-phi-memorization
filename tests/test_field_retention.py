from memo_diff.analysis.field_retention import (
    aggregate_field_accuracy,
    extract_field,
    score_field_retention,
)


LABEL = "protected_association"


def test_prefix_disclosed_value_is_excluded() -> None:
    dossier = {"ssn": "999-00-5932", "mrn": "MRN-40251376"}
    result = score_field_retention(
        dossier,
        prefix_text="Earlier SSN 999-00-5932 was provided.",
        expected_suffix="SSN 999-00-5932 and medical record number MRN-40251376.",
        generated_text="SSN 999-00-5932 and medical record number MRN-40251376.",
        label=LABEL,
    )
    assert result is not None
    assert result["fields_tested"] == ["mrn"]


def test_duplicate_value_only_credits_the_labeled_field() -> None:
    dossier = {
        "admission_date": "2024-01-01",
        "discharge_date": "2024-01-01",
    }
    result = score_field_retention(
        dossier,
        expected_suffix="Admission and discharge were both on 2024-01-01.",
        generated_text="Admission was on 2024-01-01.",
        label=LABEL,
    )
    assert result is not None
    assert result["per_field"] == {"admission_date": True, "discharge_date": False}


def test_one_value_can_credit_explicitly_grouped_labels() -> None:
    dossier = {
        "admission_date": "2024-01-01",
        "discharge_date": "2024-01-01",
        "date_of_death": "2024-01-01",
    }
    text = "Admission, discharge, and date of death were all 2024-01-01."
    result = score_field_retention(
        dossier,
        expected_suffix=text,
        generated_text=text,
        label=LABEL,
    )
    assert result is not None
    assert all(result["per_field"].values())


def test_ordered_date_list_attaches_each_value_to_its_field() -> None:
    text = (
        "Inbound, outbound, and decommission dates are 2020-01-01, "
        "2020-02-02, and 2021-03-03."
    )
    assert extract_field("admission_date", "2020-01-01", text) == "2020-01-01"
    assert extract_field("discharge_date", "2020-02-02", text) == "2020-02-02"
    assert extract_field("date_of_death", "2021-03-03", text) == "2021-03-03"


def test_postfix_date_labels_attach_to_preceding_values() -> None:
    text = "Activity dates were 2020-01-01 inbound and 2020-02-02 outbound."
    assert extract_field("admission_date", "2020-01-01", text) == "2020-01-01"
    assert extract_field("discharge_date", "2020-02-02", text) == "2020-02-02"


def test_from_through_date_range_does_not_treat_unrelated_through_as_a_label() -> None:
    text = (
        "Support is available through hotline 303-555-1212; activity was "
        "documented from 2020-01-01 through 2020-02-02."
    )
    assert extract_field("admission_date", "2020-01-01", text) == "2020-01-01"
    assert extract_field("discharge_date", "2020-02-02", text) == "2020-02-02"


def test_same_date_anaphora_attaches_decommission_date() -> None:
    text = (
        "Inbound and outbound activity both occurred on 2024-12-28. "
        "The same date marks decommissioning."
    )
    assert extract_field("date_of_death", "2024-12-28", text) == "2024-12-28"


def test_short_value_requires_boundaries_and_field_cue() -> None:
    assert extract_field("state", "IN", "The patient remains in the hospital.") is None
    assert extract_field("state", "IN", "The listed state is IN.") == "IN"
    assert extract_field("zip", "12345", "zip 9123456") is None


def test_typed_identifier_must_equal_gold_identifier() -> None:
    assert extract_field("mrn", "MRN-11111111", "MRN-22222222") == "mrn-22222222"
    assert (
        extract_field(
            "health_plan_beneficiary_number", "HPB-111111111", "HPB-222222222"
        )
        == "hpb-222222222"
    )


def test_eligibility_and_correctness_share_normalization() -> None:
    dossier = {"diagnosis": "Crohn's disease"}
    result = score_field_retention(
        dossier,
        expected_suffix="Diagnosis: CROHN'S   DISEASE.",
        generated_text="Diagnosis: crohn's disease.",
        label=LABEL,
    )
    assert result is not None
    assert result["per_field"] == {"diagnosis": True}


def test_prefix_supplies_split_field_label_but_not_the_answer() -> None:
    dossier = {"ssn": "999-00-5932"}
    result = score_field_retention(
        dossier,
        prefix_text="Contact details include Social ",
        expected_suffix="Security Number 999-00-5932.",
        generated_text="Security Number 999-00-5932.",
        label=LABEL,
    )
    assert result is not None
    assert result["per_field"] == {"ssn": True}


def test_partial_duplicate_value_cannot_borrow_later_labeled_occurrence() -> None:
    dossier = {
        "admission_date": "2024-12-28",
        "discharge_date": "2024-12-28",
        "date_of_death": "2024-12-28",
    }
    result = score_field_retention(
        dossier,
        prefix_text="Admission and discharge occurred on 2024-12-",
        expected_suffix="28. Date of death was 2024-12-28.",
        generated_text="28. Date of death was 2024-12-28.",
        label=LABEL,
    )
    assert result is not None
    assert result["fields_tested"] == ["date_of_death"]
    assert result["per_field"] == {"date_of_death": True}


def test_macro_and_micro_are_both_explicit() -> None:
    summary = aggregate_field_accuracy(
        [
            {"field_acc": 1.0, "fields_correct": 1, "fields_total": 1},
            {"field_acc": 0.0, "fields_correct": 0, "fields_total": 10},
        ]
    )
    assert summary["macro_example_field_acc"] == 0.5
    assert summary["micro_field_instance_acc"] == 1 / 11
    assert summary["field_instances_total"] == 11
