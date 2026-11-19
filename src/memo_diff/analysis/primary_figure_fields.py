"""Field selection and common relative order used in the main paper heatmaps."""
from memo_diff.analysis.paper_fields import PHI_DISPLAY, BENIGN_DISPLAY, PAIR_DISPLAY

# Original prefix-attack image, read from top to bottom.
PREFIX_TOP_TO_BOTTOM = ['ssn','phone','health_plan_beneficiary_number','mrn',
                       'date_of_death','discharge_date','admission_date','prescription','diagnosis']
EXCLUDED = {
    'initials': 'Full entity name is already supplied.',
    'prescription': 'The preceding diagnosis has a fixed paired therapy.',
    'discharge_date': 'The preceding admission date strongly predicts the first year chunk.',
}
ALREADY_IN_ATTACK_PREFIX = ['date_of_birth','city','state','zip']
PRIMARY_ORDER = [f for f in PREFIX_TOP_TO_BOTTOM if f not in EXCLUDED] + ALREADY_IN_ATTACK_PREFIX
assert len(PRIMARY_ORDER)==11 and not set(PRIMARY_ORDER).intersection(EXCLUDED)
