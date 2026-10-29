# Failure-Aware Ensemble Learning and Explainable AI for Three-Class Diabetes Status Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B.svg)](http://localhost:8501)
[![Empirical Verification](https://img.shields.io/badge/Verification-100%25%20Audited-brightgreen.svg)](reports/00_EXECUTIVE_PROJECT_SUMMARY.md)

---

## 1. Executive Research Summary

Diabetes mellitus affects over 537 million adults globally, with national prevalence rising +79.5% over the past quarter-century. While conventional machine learning treats diabetes screening as a naive binary task (Healthy vs. Diabetic), clinical prevention depends critically on early identification of **Prediabetes**—a reversible pre-clinical transition state.

This research addresses the **Accuracy Paradox** in 3-class diabetes status prediction using the CDC Behavioral Risk Factor Surveillance System (BRFSS 2015, $N = 253,680$). By constructing a **Failure-Aware Hybrid Stacking Ensemble** equipped with 50 uncertainty and error-discordance meta-features, the framework captures **38.88% of prediabetes cases** (360 out of 926 patients in a locked test set of 50,736 individuals), compared to just 4 patients (0.43% recall) detected by linear margin classifiers that achieve deceptively high raw accuracy (80.14%).

---

## 2. Target Classes & Severe Imbalance

The prediction task evaluates a 3-class ordinal target variable (`Diabetes_012`):
- **Class 0 (No Diabetes):** 213,703 records (84.24%) — Normoglycemic majority
- **Class 1 (Prediabetes):** 4,631 records (1.83%) — **Severe 46:1 Imbalance**
- **Class 2 (Diabetes):** 35,346 records (13.93%) — Diagnosed diabetes

---

## 3. Methodological Innovation: Failure-Aware Hybrid Stacking

Traditional stacking ensembling (Wolpert, 1992) feeds only base class probabilities to a meta-learner. In contrast, our Failure-Aware Hybrid meta-learner ingests **50 standardized meta-features** across 7 diverse base estimators:

```text
========================================================================================
                          50 META-FEATURE FORMULATION (K = 7)
========================================================================================
1. Base Class Probabilities (21 feats):  P_m(Y = c | x) for m=1..7, c=0..2
2. Maximum Confidence (7 feats):         C_m = max_c P_m(Y = c | x)
3. Shannon Entropy (7 feats):            H_m = -sum_c P_m(Y = c) * log2(P_m(Y = c) + eps)
4. Decision Margin (7 feats):            M_m = P_m,(1) - P_m,(2)
5. Pairwise Disagreements (7 feats):     D_m = I(y_hat_m != y_mode)
6. Modal Consensus Count (1 feat):       A_modal = sum_m I(y_hat_m == y_mode)
========================================================================================
```

### Evaluated Base Models ($K = 7$)
1. **Logistic Regression:** L-BFGS solver with balanced class weights
2. **Decision Tree:** Balanced tree with `max_depth=10`
3. **Random Forest:** 100 balanced bagged trees with `max_depth=12`
4. **SVM (SGD Modified Huber):** Linear margin classifier with smooth surrogate loss
5. **XGBoost:** Gradient boosted trees with multi-class softmax
6. **LightGBM:** Histogram-based gradient booster with balanced class weights
7. **CatBoost:** Symmetric decision trees with balanced multi-class weights

---

## 4. Key Empirical Results (Locked Test Set: $N = 50,736$)

| Model Configuration | Accuracy | Macro-F1 | Balanced Acc | MCC | Prediab Recall | Prediab Cases Detected | Brier Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SVM (SGD Mod Huber)** | **0.8014** | 0.4472 | 0.4711 | 0.3422 | 0.0043 | 4 / 926 | **0.2841** |
| **Soft Voting** | 0.7477 | **0.4476** | 0.5028 | **0.3533** | 0.0421 | 39 / 926 | 0.3211 |
| **XGBoost** | 0.7581 | 0.4347 | 0.4689 | 0.3080 | 0.0205 | 19 / 926 | 0.3120 |
| **LightGBM** | 0.6430 | 0.4234 | 0.5161 | 0.2999 | 0.2538 | 235 / 926 | 0.4215 |
| **Standard Stacking** | 0.6000 | 0.4180 | **0.5278** | 0.2862 | **0.4060** | **376 / 926** | 0.4721 |
| **Failure-Aware Hybrid**| 0.6101 | 0.4162 | 0.5218 | 0.2820 | 0.3888 | 360 / 926 | 0.4589 |

### Statistical Hypothesis Testing (Hybrid vs. Stacking)
- **McNemar Discordance Test:** $\chi^2 = 122.95, df = 1, p = 1.43 \times 10^{-28}$ (1,311 Hybrid-only correct vs 800 Stacking-only correct).
- **Paired Bootstrap Test ($B = 1,000$):**
  - **Accuracy:** $+1.01\%$ difference ($p < 0.001$, 95% CI: $[+0.0064, +0.0138]$), eliminating 973 false-positive prediabetes alerts among non-diabetics.
  - **Macro-F1:** Delta = $-0.0018$ ($p = 0.224$, statistical parity).

---

## 5. Repository Directory Structure

```text
KLH-CSE-2026-27-2420030600-Failure-Aware-Diabetes-Prediction-Explainable-AI/
├── data/
│   ├── README.md                      # Dataset acquisition and schema guide
│   ├── FEATURE_DICTIONARY.csv         # 21 raw features dictionary & types
│   ├── CLASS_DISTRIBUTION.csv         # Exact class split counts
│   ├── DATA_QUALITY_REPORT.csv        # Zero missing value audit
│   └── sample/
│       └── df1_engineered_sample.csv  # 100-row sample for instant testing
│
├── docs/
│   ├── README.md                      # Documentation overview
│   ├── methodology/                   # Mathematical formulations & protocols
│   ├── architecture/                  # Vector pipeline diagrams (PNG/SVG)
│   ├── notebook_audit.md              # 15-point audit of Notebooks 01-16
│   └── discrepancies_and_limitations.md # Resolved discrepancies & research limitations
│
├── notebooks/                         # 16 sequentially numbered research notebooks
│   ├── README.md                      # Detailed index of notebooks 01-16
│   └── 01_problem_and_data_understanding.ipynb ... 16_integrated_public_health_analysis.ipynb
│
├── reports/
│   ├── README.md                      # Reports index
│   ├── 00_EXECUTIVE_PROJECT_SUMMARY.md # Executive readable report
│   ├── PROJECT_FACTS_FOR_CHATGPT.md   # Self-contained research dossier
│   ├── MANUSCRIPT_GAP_ANALYSIS.md     # Section-by-section manuscript mapping
│   └── SOURCE_EVIDENCE_INDEX.csv      # Cell-level claim traceability matrix
│
├── results/
│   ├── README.md                      # Results index
│   ├── metrics/                       # Final verified CSV metric tables
│   ├── figures/                       # 22 publication figures (PNG 300 DPI & SVG)
│   └── source_data/                   # 18 source CSV files behind all figures
│
├── src/
│   ├── README.md                      # Codebase structure
│   ├── config.py                      # Global parameters and seeds
│   ├── data_processing/               # Loading & scaling
│   ├── features/                      # 8 clinical feature formulas
│   ├── models/                        # 7 base models, OOF generator, Hybrid
│   ├── evaluation/                    # Metrics, bootstrap, McNemar, subgroups
│   ├── explainability/                # SHAP and LIME modules
│   ├── dashboard/                     # Interactive Streamlit application
│   └── run_pipeline.py                # Command-line pipeline execution script
│
├── requirements.txt                   # Verified Python dependencies
├── .gitignore                         # Git exclusion rules
└── README.md                          # Master documentation
```

---

## 6. Installation and Environment Setup

```bash
# Clone the repository
git clone https://github.com/SIDHARTHREDDY600/KLH-CSE-2026-27-2420030600-Failure-Aware-Diabetes-Prediction-Explainable-AI.git
cd KLH-CSE-2026-27-2420030600-Failure-Aware-Diabetes-Prediction-Explainable-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 7. Running the Pipeline & Dashboard

### A. Quick Pipeline Test (Using Sample Data)
```bash
python -m src.run_pipeline --data-path data/sample/df1_engineered_sample.csv --mode evaluate_sample
```

### B. Launching the Clinical Decision Support Dashboard
```bash
streamlit run src/dashboard/app.py
```
Open your browser to `http://localhost:8501` to test interactive patient risk assessment, real-time uncertainty meters, and the CDC 25-year surveillance trend visualizer.

---

## 8. Clinical Research Disclaimer

> [!CAUTION]
> **Research and Public Health Screening Notice:**  
> The algorithms, models, and predictions provided in this repository are developed strictly for academic research and population-level public health risk triage. They do **not** constitute a definitive clinical diagnosis. Final clinical diagnosis of diabetes or prediabetes must be confirmed through standardized laboratory bloodwork (e.g., Fasting Plasma Glucose, 2-Hour Oral Glucose Tolerance Test, or Glycated Hemoglobin HbA1c) administered by a qualified healthcare professional.
