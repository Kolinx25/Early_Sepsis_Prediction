/*
  EARLY SEPSIS PREDICTION DATA WAREHOUSE
==========================================
Project:
Early Sepsis Prediction Using Assisted Intelligence
for Clinical Decision Support

Dataset:
PhysioNet Challenge 2019

Purpose:
Creates the PostgreSQL schema for storing ICU patient
demographics and hourly clinical measurements.

*/



--PATIENT DEMOGRAPHICS
--One record per patient
CREATE TABLE IF NOT EXISTS patients (

    patient_id      INTEGER PRIMARY KEY,

    age             DOUBLE PRECISION,

    gender          SMALLINT,

    unit1           SMALLINT,

    unit2           SMALLINT,

    hosp_adm_time   DOUBLE PRECISION

);



--HOURLY CLINICAL MEASUREMENTS
--One record per patient per ICU hour
CREATE TABLE IF NOT EXISTS clinical_measurements (

    patient_id INTEGER NOT NULL,

    iculos INTEGER NOT NULL,

    
     -- Vital Signs 
    hr DOUBLE PRECISION,
    o2sat DOUBLE PRECISION,
    temp DOUBLE PRECISION,
    sbp DOUBLE PRECISION,
    map DOUBLE PRECISION,
    dbp DOUBLE PRECISION,
    resp DOUBLE PRECISION,
    etco2 DOUBLE PRECISION,

    
       --Laboratory Measurements
    baseexcess DOUBLE PRECISION,
    hco3 DOUBLE PRECISION,
    fio2 DOUBLE PRECISION,
    ph DOUBLE PRECISION,
    paco2 DOUBLE PRECISION,
    sao2 DOUBLE PRECISION,

    ast DOUBLE PRECISION,
    bun DOUBLE PRECISION,
    alkalinephos DOUBLE PRECISION,
    calcium DOUBLE PRECISION,
    chloride DOUBLE PRECISION,
    creatinine DOUBLE PRECISION,

    bilirubin_direct DOUBLE PRECISION,
    glucose DOUBLE PRECISION,
    lactate DOUBLE PRECISION,
    magnesium DOUBLE PRECISION,
    phosphate DOUBLE PRECISION,
    potassium DOUBLE PRECISION,
    bilirubin_total DOUBLE PRECISION,
    troponini DOUBLE PRECISION,

    hct DOUBLE PRECISION,
    hgb DOUBLE PRECISION,
    ptt DOUBLE PRECISION,
    wbc DOUBLE PRECISION,
    fibrinogen DOUBLE PRECISION,
    platelets DOUBLE PRECISION,

    
       --Clinical Scores

       --Outcome
    sepsis_label SMALLINT,

    PRIMARY KEY (patient_id, iculos),

    CONSTRAINT fk_patient
        FOREIGN KEY (patient_id)
        REFERENCES patients(patient_id)

);



--INDEXES

CREATE INDEX IF NOT EXISTS idx_measurements_patient
ON clinical_measurements(patient_id);

CREATE INDEX IF NOT EXISTS idx_measurements_iculos
ON clinical_measurements(iculos);

CREATE INDEX IF NOT EXISTS idx_measurements_sepsis
ON clinical_measurements(sepsis_label);

SELECT COUNT(*)
FROM patients;

SELECT COUNT(*)
FROM clinical_measurements;

CREATE TABLE IF NOT EXISTS feature_table
(
    patient_id INTEGER NOT NULL,
    iculos INTEGER NOT NULL,

    hr DOUBLE PRECISION,
    o2sat DOUBLE PRECISION,
    temp DOUBLE PRECISION,
    sbp DOUBLE PRECISION,
    map DOUBLE PRECISION,
    dbp DOUBLE PRECISION,
    resp DOUBLE PRECISION,
    etco2 DOUBLE PRECISION,

    baseexcess DOUBLE PRECISION,
    hco3 DOUBLE PRECISION,
    fio2 DOUBLE PRECISION,
    ph DOUBLE PRECISION,
    paco2 DOUBLE PRECISION,
    sao2 DOUBLE PRECISION,
    ast DOUBLE PRECISION,
    bun DOUBLE PRECISION,
    alkalinephos DOUBLE PRECISION,
    calcium DOUBLE PRECISION,
    chloride DOUBLE PRECISION,
    creatinine DOUBLE PRECISION,
    bilirubin_direct DOUBLE PRECISION,
    glucose DOUBLE PRECISION,
    lactate DOUBLE PRECISION,
    magnesium DOUBLE PRECISION,
    phosphate DOUBLE PRECISION,
    potassium DOUBLE PRECISION,
    bilirubin_total DOUBLE PRECISION,
    troponini DOUBLE PRECISION,
    hct DOUBLE PRECISION,
    hgb DOUBLE PRECISION,
    ptt DOUBLE PRECISION,
    wbc DOUBLE PRECISION,
    fibrinogen DOUBLE PRECISION,
    platelets DOUBLE PRECISION,

    -- missingness indicators
    hr_missing SMALLINT,
    o2sat_missing SMALLINT,
    temp_missing SMALLINT,
    sbp_missing SMALLINT,
    map_missing SMALLINT,
    dbp_missing SMALLINT,
    resp_missing SMALLINT,
    etco2_missing SMALLINT,

    baseexcess_missing SMALLINT,
    hco3_missing SMALLINT,
    fio2_missing SMALLINT,
    ph_missing SMALLINT,
    paco2_missing SMALLINT,
    sao2_missing SMALLINT,
    ast_missing SMALLINT,
    bun_missing SMALLINT,
    alkalinephos_missing SMALLINT,
    calcium_missing SMALLINT,
    chloride_missing SMALLINT,
    creatinine_missing SMALLINT,
    bilirubin_direct_missing SMALLINT,
    glucose_missing SMALLINT,
    lactate_missing SMALLINT,
    magnesium_missing SMALLINT,
    phosphate_missing SMALLINT,
    potassium_missing SMALLINT,
    bilirubin_total_missing SMALLINT,
    troponini_missing SMALLINT,
    hct_missing SMALLINT,
    hgb_missing SMALLINT,
    ptt_missing SMALLINT,
    wbc_missing SMALLINT,
    fibrinogen_missing SMALLINT,
    platelets_missing SMALLINT,

    sepsis_label SMALLINT,

    PRIMARY KEY(patient_id, iculos)
);


CREATE INDEX idx_feature_patient
ON feature_table(patient_id);

CREATE INDEX idx_feature_iculos
ON feature_table(iculos);