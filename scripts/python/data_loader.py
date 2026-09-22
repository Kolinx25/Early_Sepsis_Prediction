"""
data_loader.py

Reads and explores the PhysioNet Challenge 2019 patient files.

Responsibilities
----------------
- Discover all patient .psv files
- Summarize the dataset
- Inspect a single patient file
- Load a patient file into a pandas DataFrame

This module does NOT:
- Connect to PostgreSQL
- Transform the data
- Perform feature engineering
- Train models
"""

from pathlib import Path
import pandas as pd
from config import RAW_DATA_DIR


def discover_patient_files():
    """
    Locate every patient PSV file.

    Returns
    -------
    list[Path]
        Sorted list of patient file paths.
    """
    return sorted(RAW_DATA_DIR.rglob("*.psv"))


def summarize_dataset():
    """
    Display a summary of the downloaded PhysioNet dataset.
    """
    set_a = list((RAW_DATA_DIR / "training_setA").glob("*.psv"))
    set_b = list((RAW_DATA_DIR / "training_setB").glob("*.psv"))

    print("=" * 50)
    print("PHYSIONET DATASET SUMMARY")
    print("=" * 50)
    print(f"Training Set A : {len(set_a):,} patients")
    print(f"Training Set B : {len(set_b):,} patients")
    print("-" * 50)
    print(f"Total Patients : {len(set_a) + len(set_b):,}")
    print("=" * 50)


def inspect_patient(patient_file: Path):
    """
    Inspect a single patient file.
    """
    df = pd.read_csv(patient_file, sep="|")

    print("\nPatient File")
    print("-" * 50)
    print(patient_file.name)

    print(f"\nRows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn Names")
    print(list(df.columns))

    print("\nMissing Values")
    print(df.isna().sum())


def load_patient(patient_file: Path) -> pd.DataFrame:
    """
    Read a patient PSV file into a DataFrame.

    Parameters
    ----------
    patient_file : Path

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(patient_file, sep="|")

def extract_patient_id(patient_file: Path) -> int:
    """
    Extract the numeric patient ID from a PhysioNet filename.

    Example:
        p000001.psv -> 1
        p123456.psv -> 123456
    """
    return int(patient_file.stem.replace("p", ""))

def split_patient_data(df: pd.DataFrame, patient_id: int):
    """
    Split a PhysioNet patient file into demographics and
    hourly clinical measurements.
    """

    patient = {

        "Patient_id": patient_id,

        "Age": df.iloc[0]["Age"],

        "Gender": df.iloc[0]["Gender"],

        "Unit1": df.iloc[0]["Unit1"],

        "Unit2": df.iloc[0]["Unit2"],

        "HospAdmTime": df.iloc[0]["HospAdmTime"]

    }

    measurements = df.copy()

    measurements.insert(0, "Patient_id", patient_id)

    # Keep the original PhysioNet hour column name.
    # No artificial ICU_Hour alias is created.

    measurements.drop(
        columns=[
            "Age",
            "Gender",
            "Unit1",
            "Unit2",
            "HospAdmTime",
        ],
        inplace=True,
    )

    return patient, measurements

if __name__ == "__main__":

    summarize_dataset()

    patient_files = discover_patient_files()

    print(f"\nDiscovered {len(patient_files):,} patient files.")

    first_patient = patient_files[0]

    inspect_patient(first_patient)

    df = load_patient(first_patient)

    print("\nFirst Five Rows")
    print(df.head())

    patient_id = extract_patient_id(first_patient)

    patient, measurements = split_patient_data(df, patient_id)

    print("\nPatient Dictionary")
    print("-" * 50)
    print(patient)

    print("\nMeasurements")
    print("-" * 50)
    print(measurements.head())

    print("\nShape")
    print(measurements.shape)    