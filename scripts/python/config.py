"""
Project Configuration

This module centralizes all configurable settings used throughout the
Early Sepsis Prediction project.

Other scripts should import values from here rather than hard-coding
paths or database credentials.
"""

from pathlib import Path

# Project Directories
# Project root (Early-Sepsis-Prediction/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPORT_DATA_DIR = DATA_DIR / "exports"

# Database scripts
SQL_DIR = PROJECT_ROOT / "scripts" / "sql"
SCHEMA_FILE = SQL_DIR / "schema.sql"
VIEWS_FILE = SQL_DIR / "views.sql"
QUERIES_FILE = SQL_DIR / "queries.sql"

# PostgreSQL Configuration
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "sepsis_capstone"
DB_USER = "postgres"

# Replace with your PostgreSQL password
DB_PASSWORD = "kai$Sign1998"


# ETL Settings
# Number of patients processed before committing a transaction
BATCH_SIZE = 500


# Validation
EXPECTED_DATASETS = [
    "training_setA",
    "training_setB"
]


if __name__ == "__main__":
    print("Project Root :", PROJECT_ROOT)
    print("Raw Data     :", RAW_DATA_DIR)
    print("Schema File  :", SCHEMA_FILE)
