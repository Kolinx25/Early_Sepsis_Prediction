# Early Sepsis Prediction Using Assisted Intelligence for Clinical Decision Support

## Overview

Sepsis is a life-threatening medical condition that requires rapid diagnosis and treatment. Delays in identifying patients at risk can lead to organ failure, increased mortality, and higher healthcare costs.

This project develops an assisted intelligence system that analyzes electronic health record (EHR) data to identify patients who may be at risk of sepsis before clinical deterioration occurs. Rather than replacing clinicians, the system is designed to support clinical decision-making by providing timely risk assessments and actionable insights.

---

## Objectives

- Develop an end-to-end clinical analytics pipeline for early sepsis prediction.
- Build a PostgreSQL clinical data warehouse from ICU patient data.
- Perform exploratory data analysis and feature engineering.
- Train and evaluate predictive machine learning models.
- Interpret model predictions to support clinical decision-making.
- Develop an interactive Power BI dashboard for clinical decision support.

---

## Dataset

This project uses the **PhysioNet/Computing in Cardiology Challenge 2019** dataset.

**Official Dataset**

https://physionet.org/content/challenge-2019/1.0.0/

The dataset is **not included** in this repository.

After downloading, place the extracted folders inside:

```text
data/
└── raw/
    ├── training_setA/
    └── training_setB/
```

---

## Project Structure

```text
Early-Sepsis-Prediction/
│
├── assets/
├── dashboards/
├── data/
├── database/
├── docs/
├── models/
├── reports/
└── scripts/
```

---

## Technology Stack

- Python
- PostgreSQL
- SQL
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Power BI
- Git & GitHub

---

## Project Workflow

```text
Electronic Health Records
            │
            ▼
     Data Preparation
            │
            ▼
 Clinical Data Warehouse
            │
            ▼
 Feature Engineering
            │
            ▼
 Predictive Modeling
            │
            ▼
 Model Interpretation
            │
            ▼
Clinical Decision Support Dashboard
```

---

## Repository Status

🚧 Project currently under development.
---

## Author

**Collins Amoo**

Master's Capstone Project

Data Management and Analysis
