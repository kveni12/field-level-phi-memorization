"""Field order and labels copied from the original paper Figure 2/3 scripts."""
ONSET_ORDER = ['initials','date_of_birth','date_of_death','city','state','zip',
               'phone','ssn','mrn','health_plan_beneficiary_number',
               'admission_date','discharge_date','diagnosis','prescription']
HEATMAP_ORDER = ['zip','state','ssn','prescription','phone','mrn','initials',
                 'health_plan_beneficiary_number','discharge_date','diagnosis',
                 'date_of_death','date_of_birth','city','admission_date']
PHI_DISPLAY = {'health_plan_beneficiary_number':'health_plan_id'}
BENIGN_DISPLAY = dict(zip(ONSET_ORDER, ['code','install_date','decommission_date',
    'facility_city','facility_state','facility_zip','hotline','account_number',
    'facility_record_number','plan_account_id','inbound_date','outbound_date',
    'care_pathway_label','protocol_name']))
PAIR_DISPLAY = dict(zip(ONSET_ORDER, ['initials ↔ code','date of birth ↔ install date',
    'date of death ↔ decommission date','city ↔ facility city','state ↔ facility state',
    'ZIP ↔ facility ZIP','phone ↔ hotline','SSN ↔ account number',
    'MRN ↔ facility record number','health-plan number ↔ plan account ID',
    'admission ↔ inbound date','discharge ↔ outbound date',
    'diagnosis ↔ care pathway','prescription ↔ protocol']))
