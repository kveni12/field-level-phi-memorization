# Dataset and field definitions

The frozen corpus contains 15 structured fields, counting city, state and ZIP separately. Entity names and initials/code differ between twins; the other 13 field values are shared.

| PHI field | Benign field |
|---|---|
| Patient name | Organization name |
| Initials | Code |
| Date of birth | Install date |
| City | Facility city |
| State | Facility state |
| ZIP | Facility ZIP |
| Diagnosis | Care pathway |
| Prescription | Protocol |
| Admission date | Inbound date |
| Discharge date | Outbound date |
| Date of death | Decommission date |
| Phone number | Hotline number |
| SSN | Account number |
| Medical record number | Facility record number |
| Health-plan beneficiary number | Plan account ID |

## Generation

The released `pools.json` and `luna_names.json` freeze the generated choices. `scripts/reproduce_dataset.py` renders and validates those pools without changing any names, numeric values or split assignments. This is the exact document-reproduction route.

The historical name-generation implementation, including its system/user prompts, batch size 50, temperature 0.9, 4,000 completion-token limit, seeds and reasoning-effort setting, is in `src/memo_diff/dataset/luna_pools.py`. The historical model identifier is recorded as `gpt-5.6-luna`; its availability is not assumed. Generating new names via an available API model requires the optional `name-generation` dependencies and your own `OPENAI_API_KEY`. It will create new pools, not guarantee the original names.

```sh
python -m pip install -e '.[name-generation]'
python scripts/generate_name_pools.py --model YOUR_AVAILABLE_MODEL --output outputs/new_names.json
```

Dates, identifiers and final prose are generated programmatically. The 15 diagnosis/therapy pairs are fixed. SSNs have the synthetic forms `000-00-XXXX` or `999-00-XXXX`; the last four digits are random. Phone/hotline numbers use the geographic area code, `555`, and four generated digits. The full dataset contains no real patient records.

## Order and prefix eligibility

PHI order: name, initials, birth date, city, state, ZIP; then diagnosis, prescription, admission, discharge, death, phone, SSN, MRN, health-plan ID.

Benign order: organization, code, city, state, ZIP, install date; then care pathway, protocol, inbound date, outbound date, decommission date, facility record number, plan account ID, hotline, account number.

With the retained character-aligned split, the six demographic/entity fields are supplied and the nine remaining complete fields are eligible for attack scoring in this frozen corpus. That statement must not be transferred to a literal half-token split, which can cut through different fields.

The released main prefix heatmap shows seven selected suffix fields; the main J-Lens heatmap shows eleven selected fields. Those field selections and their common relative order are in `primary_figure_fields.py`. The full numerical scoring code evaluates all fourteen non-entity fields, matching the original multiple-comparison family.

## First-token interpretation

- Phone/hotline area codes are constrained by preceding city/state/ZIP in both framings.
- State and ZIP can use earlier geography.
- Diagnosis precedes its fixed prescription; the paired benign fields follow the same dependency.
- SSN/account first tokens select `000` or `999`, not the random final digits.
- Date year chunks can use restricted ranges and prior dates.
- MRN/health-plan numeric chunks provide stronger tests of arbitrary record-specific associations, but remain single-token readouts.

Every J-Lens target is conditioned on the gold preceding text. These measurements alone do not separate ordinary contextual prediction from memorization of the complete value.
