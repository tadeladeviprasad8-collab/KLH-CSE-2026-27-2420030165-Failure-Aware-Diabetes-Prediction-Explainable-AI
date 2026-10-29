# Dataset Documentation and Acquisition Guide

This directory documents the public health datasets used in the **Failure-Aware Diabetes Prediction and Surveillance** research project.

## 1. Datasets Summary

| Dataset | Formal Source | Records ($N$) | Features | Target Variable | Role in Project |
|:---|:---|:---:|:---:|:---:|:---|
| **Dataset 1: BRFSS 2015** | CDC Behavioral Risk Factor Surveillance System | 253,680 | 21 raw (29 engineered) | `Diabetes_012` (3 classes) | Patient-level risk prediction & failure-aware ensembling |
| **Dataset 2: USDSS / NHIS** | CDC United States Diabetes Surveillance System | 13,889 | 10 aggregate fields | `Crude_Prevalence` | Longitudinal macro-surveillance trends (2000–2024) |

---

## 2. Dataset 1: CDC BRFSS 2015 Health Indicators

### Target Class Formulation & Imbalance
The target variable `Diabetes_012` models a 3-class disease progression state:
- **Class 0 (No Diabetes):** 213,703 records (84.24%) — Normoglycemic majority
- **Class 1 (Prediabetes):** 4,631 records (1.83%) — Pre-clinical transition state (**Extreme Minority**)
- **Class 2 (Diabetes):** 35,346 records (13.93%) — Diagnosed Type 1 or Type 2 diabetes

### Acquisition Instructions
Due to GitHub's file size recommendations, raw survey files are not committed to Git. To reproduce the experiments locally:
1. Download `diabetes_012_health_indicators_BRFSS2015.csv` from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) or the [CDC BRFSS 2015 Survey Data Portal](https://www.cdc.gov/brfss/annual_data/annual_2015.html).
2. Place the file locally into:
   ```text
   data/raw/diabetes_012_health_indicators_BRFSS2015.csv
   ```
3. A 100-row sample is provided in `data/sample/df1_engineered_sample.csv` for immediate automated testing and verification.

---

## 3. Dataset 2: CDC USDSS Longitudinal Surveillance (2000–2024)

### Purpose
Dataset 2 tracks population-level diabetes prevalence rates across all 50 U.S. states and demographic strata from 2000 to 2024.
- **Strict Methodological Demarcation:** Dataset 2 is an independent macroeconomic surveillance dataset. It is **not** merged with BRFSS 2015 at the record level and is not used for individual machine learning test evaluation.
- To acquire, download `export.csv` from the [CDC United States Diabetes Surveillance System](https://gis.cdc.gov/grasp/diabetes/diabetesatlas-surveillance.html) and place in:
  ```text
  data/raw/export.csv
  ```

---

## 4. Data Dictionaries & Quality Reports

- `FEATURE_DICTIONARY.csv`: Full definitions, types, and value ranges for all 21 raw survey predictors.
- `CLASS_DISTRIBUTION.csv`: Exact class frequencies and percentages across full, training, and test splits.
- `DATA_QUALITY_REPORT.csv`: Verification of 0 missing values and duplicate vector audit across splits.
