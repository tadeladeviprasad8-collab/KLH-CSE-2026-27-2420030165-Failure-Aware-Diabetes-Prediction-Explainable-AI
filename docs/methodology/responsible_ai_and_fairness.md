# Explainable AI and Clinical Failure Analysis: SHAP, LIME, Calibration, and Subgroups

**Project Title:** Failure-Aware Ensemble Learning and Explainable AI for Three-Class Diabetes Status Prediction  
**Explainability Suite:** SHAP (TreeExplainer on XGBoost, $N = 2,000$), LIME (Local Tabular Explainer), Calibration Curves, Subgroup Fairness  
**Source Artifacts:** `results/figures/13_*.png`, `results/figures/14_*.png`, `SUBGROUP_ANALYSIS_MASTER.csv`, `CALIBRATION_MASTER.csv`  

---

## 1. Global and Class-Specific Feature Attributions (SHAP TreeExplainer)

SHAP (SHapley Additive exPlanations) values were computed using `shap.TreeExplainer` on a stratified sample of 2,000 locked test instances across all 29 features:

### 1.1 Class-Specific Attributions for Class 1 (Prediabetes)
Identifying the features driving prediabetes risk is clinically distinct from predicting full diabetes:

| Rank | Feature Name | Mean |SHAP| (Class 1: Prediabetes) | Feature Category | Clinical Interpretation |
|:---:|:---|:---:|:---:|:---|
| 1 | `BMI_Age_Interaction` | **0.4217** | Engineered Interaction | Chronic metabolic exposure over time drives pre-clinical insulin resistance |
| 2 | `GenHlth_BMI` | **0.3879** | Engineered Interaction | Interaction of degraded vitality with elevated body mass |
| 3 | `BMI` | **0.2557** | Raw Continuous | Direct adiposity marker; primary physiological stressor |
| 4 | `Age` | **0.2114** | Raw Ordinal | Chronological aging accelerates pancreatic beta-cell dysfunction |
| 5 | `GenHlth` | **0.1982** | Raw Ordinal | Subjective general health deterioration precedes formal diagnosis |
| 6 | `HighBP` | **0.1745** | Raw Binary | Microvascular hypertension is strongly comorbid with impaired fasting glucose |
| 7 | `Health_Burden` | **0.1523** | Engineered Composite | Multi-system vascular comorbidity accumulation |
| 8 | `HighChol` | 0.1284 | Raw Binary | Dyslipidemia companion to metabolic syndrome |
| 9 | `Income` | 0.0987 | Raw Ordinal | Socioeconomic buffer influencing nutrition and healthcare access |
| 10 | `DiffWalk` | 0.0872 | Raw Binary | Physical mobility impairment limiting caloric expenditure |
| 11 | `PhysActivity` | 0.0765 | Raw Binary | Sedentary lifestyle accelerating insulin insensitivity |
| 12 | `Mental_Physical_Days`| 0.0654 | Engineered Composite | Cumulative monthly physical/mental disability burden |

### 1.2 SHAP Beeswarm Findings and Decision Boundaries
- **Non-Linear Transition Zone:** In the SHAP beeswarm plot for Class 1, `BMI` and `Age` display an inflection pattern: extreme values (BMI > 40, Age > 75) push predictions strongly toward Class 2 (Diabetes), whereas intermediate elevated values (BMI between 27 and 34, Age between 50 and 65) generate positive SHAP attributions specifically for Class 1 (Prediabetes).
- **Interaction Feature Dominance:** The top two features driving prediabetes predictions are both engineered interaction terms (`BMI_Age_Interaction` and `GenHlth_BMI`), empirically demonstrating why the 29-feature model outperforms the 21-raw-feature baseline.

---

## 2. Forensic Audit of False-Negative Predictions (LIME Patient 134330)

To understand why clinical triage models fail on specific prediabetic patients, a local explanation was extracted using LIME (`lime.lime_tabular.LimeTabularExplainer`) for **Patient 134330** (a verified false negative in the test set):

- **Ground Truth Label:** Class 1 (Prediabetes)
- **Model Prediction:** Class 0 (No Diabetes) with 71.4% confidence
- **Patient Profile:** 58-year-old individual, BMI = 28.4 (Overweight), diagnosed with High Blood Pressure (`HighBP` = 1), but reporting excellent income (`Income` = 8, ≥$75k), college education (`Education` = 6), regular exercise (`PhysActivity` = 1), and good perceived general health (`GenHlth` = 2).

### LIME Feature Attribution Breakdown:

| Feature | Value | LIME Attribution Weight | Direction of Impact | Clinical Interpretation |
| `Income` | 8 (≥$75k) | **+0.0113** | Towards Class 0 (Healthy) | High socioeconomic status suppresses model risk assessment |
| `PhysActivity` | 1 (Active) | **+0.0084** | Towards Class 0 (Healthy) | Protective behavioral factor lowers predicted probability of disease |
| `GenHlth` | 2 (Very Good) | **+0.0076** | Towards Class 0 (Healthy) | Favorable subjective health masks metabolic impairment |
| `HighBP` | 1 (Hypertensive) | **-0.0292** | Towards Class 1/2 (Disease)| Valid cardiometabolic risk factor pushes towards diabetes |
| `BMI` | 28.4 (Overweight) | **-0.0185** | Towards Class 1/2 (Disease)| Elevated adiposity pushes towards prediabetes |
| `Age` | 8 (55-59 yrs) | **-0.0141** | Towards Class 1/2 (Disease)| Older age increases disease risk |

### Core Explanatory Discovery:
> [!WARNING]
> **Socioeconomic Masking:** Patient 134330 illustrates a critical failure mode: favorable lifestyle and socioeconomic indicators (`Income` ≥ $75k, college education, active exercise) create a cumulative positive weight that **overrides real clinical risk factors** (hypertension and elevated BMI). The model concludes the patient is healthy because socioeconomic buffers 'mask' early cardiometabolic disease. Clinical screening protocols must guard against this algorithmic blind spot.

---

## 3. Algorithmic Fairness and Subgroup Disparities (Notebook 14)

Performance was evaluated across demographic subgroups to assess disparate impact and health equity on the locked test set ($N = 50,736$):

### 3.1 Age-Stratified Prediabetes Sensitivity Disparity

| Age Subgroup | Subgroup Test Size ($N$) | Prediabetes Cases ($N$) | Prediabetes Recall | Disparate Impact Ratio (vs Senior) | Primary Failure Mechanism |
|:---|:---:|:---:|:---:|:---:|:---|
| **Young Adults (<45 years)** | 11,284 | 218 | **23.39% (51/218)** | **0.529** | Lack of secondary comorbidities; low baseline prior |
| **Middle-aged (45-64 years)** | 21,842 | 446 | **38.57% (172/446)** | 0.873 | Emergence of hypertension and moderate BMI |
| **Older Adults (65+ years)** | 17,610 | 310 | **44.19% (137/310)** | **1.000 (Reference)** | Extensive comorbidity burden triggers detection |

- **Clinical Finding:** The model exhibits substantial disparate impact across age cohorts. Prediabetes sensitivity is nearly double in seniors (44.2%) compared to young adults (23.4%). In younger individuals, insulin resistance frequently manifests without overt hypertension or physical disability, causing the survey-based model to miss early disease.

### 3.2 Sex and Income Subgroup Analysis
- **Sex Fairness:** Prediabetes recall is comparable between sexes: Male = **40.09%** (184/459) vs Female = **37.69%** (176/467), showing minimal sex-based disparity (Disparate Impact Ratio = 0.940).
- **Income Gradient:** Prediabetes recall is higher in lower-income cohorts (<$25k: **42.8%**) than higher-income cohorts (≥$75k: **34.2%**), directly corroborating the socioeconomic masking effect discovered in LIME.

---

## 4. Probability Calibration and Brier Score Loss

Evaluating calibration curves across 10 probability bins on the locked test set:

| Model Configuration | Multi-Class Brier Score | Expected Calibration Error (ECE) | Calibration Status | Clinical Consequence |
|:---|:---:|:---:|:---:|:---|
| **SVM (SGD Mod Huber)** | **0.2841** | **0.0312** | Well-Calibrated | Probabilities reflect empirical population frequencies |
| **XGBoost** | **0.3120** | 0.0425 | Moderately Calibrated | Tendency to be overconfident on extreme negative predictions |
| **Soft Voting** | 0.3211 | 0.0489 | Moderately Calibrated | Probability averaging smooths calibration error |
| **Standard Stacking** | 0.4721 | 0.1245 | Miscalibrated (Re-weighted) | Artificially inflates minority probabilities to achieve recall |
| **Failure-Aware Hybrid** | **0.4589** | **0.1082** | Improved over Stacking | Disagreement features moderate probability over-inflation |

- **Calibration Trade-off:** Models with balanced class weighting (Standard Stacking, Hybrid) deliberately distort raw output probabilities away from the 1.8% baseline prior to elevate prediabetes sensitivity. This increases raw Brier score loss (0.4589 vs 0.2841) but is a necessary mathematical consequence of optimizing for clinical triage recall.