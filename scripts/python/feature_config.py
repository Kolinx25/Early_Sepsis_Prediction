# SepsiGuard Feature Configuration

CLINICAL_FEATURES = [
    # Vital signs
    "hr",
    "o2sat",
    "temp",
    "sbp",
    "map",
    "dbp",
    "resp",
    "etco2",
    # Laboratory measurements
    "baseexcess",
    "hco3",
    "fio2",
    "ph",
    "paco2",
    "sao2",
    "ast",
    "bun",
    "alkalinephos",
    "calcium",
    "chloride",
    "creatinine",
    "bilirubin_direct",
    "glucose",
    "lactate",
    "magnesium",
    "phosphate",
    "potassium",
    "bilirubin_total",
    "troponini",
    "hct",
    "hgb",
    "ptt",
    "wbc",
    "fibrinogen",
    "platelets",
]


MISSINGNESS_FEATURES = [f"{feature}_missing" for feature in CLINICAL_FEATURES]


TEMPORAL_FEATURES = ["iculos"]


TARGET_COLUMN = "sepsis_label"


LOOKBACK = 12


HORIZONS = {"primary": 12, "secondary": 6}
