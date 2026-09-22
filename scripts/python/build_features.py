import pandas as pd

from feature_config import (
    CLINICAL_FEATURES,
    MISSINGNESS_FEATURES,
    TEMPORAL_FEATURES,
    TARGET_COLUMN,
)


def build_feature_table(measurements: pd.DataFrame) -> pd.DataFrame:
    """
    Create hourly feature representation from clinical measurements.

    Output:
    patient_id
    + clinical values
    + missingness indicators
    + temporal features
    + hourly sepsis label
    """

    feature_table = measurements[
        [
            "patient_id",
            *TEMPORAL_FEATURES,
            *CLINICAL_FEATURES,
            TARGET_COLUMN,
        ]
    ].copy()

    # Ensure chronological ICU trajectories
    feature_table = feature_table.sort_values(["patient_id", "iculos"]).reset_index(
        drop=True
    )

    # Create missingness indicators
    for feature in CLINICAL_FEATURES:
        feature_table[f"{feature}_missing"] = feature_table[feature].isna().astype(int)

    # Order columns explicitly
    feature_table = feature_table[
        [
            "patient_id",
            *TEMPORAL_FEATURES,
            *CLINICAL_FEATURES,
            *MISSINGNESS_FEATURES,
            TARGET_COLUMN,
        ]
    ]

    return feature_table


def validate_feature_table(feature_table: pd.DataFrame):
    """
    Basic structural validation.
    """

    expected_features = (
        len(CLINICAL_FEATURES) + len(MISSINGNESS_FEATURES) + len(TEMPORAL_FEATURES)
    )

    actual_features = len(feature_table.columns) - 2

    print("Expected feature count:", expected_features)
    print("Actual feature count:", actual_features)

    assert expected_features == actual_features, "Feature dimension mismatch"

    assert feature_table["patient_id"].notna().all()

    trajectory_order_check = feature_table.groupby("patient_id")["iculos"].apply(
        lambda x: x.is_monotonic_increasing
    )

    assert trajectory_order_check.all(), (
        "Patient trajectories are not chronologically ordered"
    )

    print("Feature table validation passed.")


if __name__ == "__main__":
    print("build_features module loaded")
