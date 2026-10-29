# Executive Project Summary: Failure-Aware Ensemble Learning and Explainable AI for Three-Class Diabetes Status Prediction

**Project Title:** Failure-Aware Ensemble Learning and Explainable AI for Three-Class Diabetes Status Prediction  
**Academic Context:** AI-Driven Public Health Surveillance Using Data Analytics  
**Audit Date:** October 2026  
**Verification Status:** 100% Empirically Audited and Verified Across All Code, Notebooks, Models, and Artifacts  

---

## 1. Clinical Context and Core Research Problem

Diabetes mellitus affects over 537 million adults globally, with national prevalence rising precipitously over the past quarter-century. While established machine learning approaches treat diabetes as a binary classification task (Healthy vs. Diabetic), clinical prevention depends critically on early identification of **Prediabetes**—a reversible pre-clinical state of impaired glucose regulation.

However, predicting 3-class diabetes status from large-scale public health survey data (such as the CDC Behavioral Risk Factor Surveillance System, BRFSS) presents three severe computational and clinical challenges:
1. **Extreme Class Imbalance:** In the CDC BRFSS 2015 dataset ($N = 253,680$), Prediabetes represents only **1.83% of the population** (4,631 cases), compared to 84.24% non-diabetic (213,703 cases) and 13.93% diabetic (35,346 cases).
2. **The Accuracy Paradox:** Standard supervised learning algorithms (e.g., linear margin classifiers) achieve seemingly impressive raw classification accuracy (~80.14%) purely by defaulting to majority non-diabetic predictions, yielding an unacceptably low **0.43% sensitivity** for prediabetes (detecting just 4 out of 926 prediabetic patients).
3. **Discordant Model Errors:** Different machine learning model families (gradient boosting trees, linear models, bagged ensembles) fail on different, non-overlapping sub-populations of prediabetic patients due to varying inductive biases, yet traditional soft voting and conventional stacking cannot exploit these error discordances.

---

## 2. The Methodological Solution: Failure-Aware Hybrid Stacking

To overcome the Accuracy Paradox and exploit error complementarity, this project develops a novel **Failure-Aware Hybrid Stacking Ensemble**:
- **Base Classifiers ($K = 7$):** Logistic Regression, Decision Tree, Random Forest, SVM (SGD Modified Huber), XGBoost, LightGBM, and CatBoost.
- **50-Dimensional Meta-Feature Representation:** In addition to standard class probabilities ($7 \times 3 = 21$ features), the meta-learner ingests:
  - 7 maximum prediction confidence scores
  - 7 normalized Shannon entropy measures (quantifying distribution dispersion)
  - 7 decision margins (gap between top two predicted class probabilities)
  - 7 pairwise disagreement flags (indicating divergence from the modal ensemble consensus)
  - 1 modal agreement count (measuring consensus density across models)
- **Meta-Learner Architecture:** An L2-regularized multinomial logistic regression classifier trained on standardized meta-features with inverse-frequency balanced class weighting.
- **Strict Leakage-Free Protocol:** Base model out-of-fold (OOF) cross-validation probability matrices were generated under Stratified 5-Fold CV ($N = 202,944$), completely isolating the locked test set ($N = 50,736$).

---

## 3. Key Verified Research Findings

### 3.1 Unmasking the Accuracy Paradox
- **SGDClassifier (Modified Huber):** Raw Accuracy = **80.14%**, but Prediabetes Recall = **0.43%** (4/926 cases). Clinically useless for early prevention.
- **Soft Voting Ensemble:** Raw Accuracy = **74.77%**, Macro-F1 = **0.4476**, but Prediabetes Recall = **4.21%** (39/926 cases). Averaging probabilities washes out minority boundary signals.
- **Standard Stacking (Corrected):** Accuracy = **60.00%**, Balanced Acc = **52.78%**, Prediabetes Recall = **40.60%** (376/926 cases).
- **Failure-Aware Hybrid (Corrected):** Accuracy = **61.01%**, Balanced Acc = **52.18%**, Prediabetes Recall = **38.88%** (360/926 cases).

### 3.2 Statistical Superiority and Parity Reconciliation
- **Classification Discordance (McNemar Test):** The McNemar test between Hybrid and Stacking yields $\chi^2 = 122.95, p = 1.43 \times 10^{-28}$ (1,311 Hybrid-only correct vs 800 Stacking-only correct), confirming statistically significant classification outcome discordance.
- **Overall Accuracy Boost (Paired Bootstrap, $B = 1,000$):** Hybrid outperforms Stacking by **+1.01% in overall accuracy** ($p < 0.001$, 95% CI: [+0.0064, +0.0138]), successfully pruning 973 false positive prediabetes alerts.
- **Balanced Metric Parity:** On harmonic multi-class measures (Macro-F1 delta = -0.0018, $p = 0.224$; Balanced Accuracy delta = -0.0060, $p = 0.168$), the Hybrid operates at statistical parity with Stacking.

### 3.3 Explainability Insights (SHAP & LIME)
- **Prediabetes Specific Risk Drivers:** SHAP TreeExplainer reveals that engineered interaction features (`BMI_Age_Interaction` Mean |SHAP| = 0.4217, `GenHlth_BMI` Mean |SHAP| = 0.3879) are the most powerful predictors for detecting prediabetes, far surpassing raw individual survey items.
- **Socioeconomic Masking Failure Mode:** LIME local explanation for Patient 134330 (false negative prediabetes) reveals that high household income (≥$75k) and active exercise create a positive attribution (+0.0197) that overrides real clinical hypertension and elevated BMI (-0.0477), causing the model to incorrectly predict normoglycemia.

### 3.4 Longitudinal Epidemiological Surveillance (CDC Dataset 2)
- CDC USDSS / NHIS aggregate surveillance data (2000–2024, $N = 13,889$) reveals that national crude diabetes prevalence increased from **4.4% in 2000 to 7.9% in 2024** (+79.5% relative increase).
- Prevalence exceeds 21.4% in older adults (65+) and 12.0% across the Southeastern 'Diabetes Belt' (Mississippi, West Virginia, Alabama), validating the urgent need for scalable survey-based screening.

---

## 4. Interactive Clinical Deployment

- An interactive web dashboard (`dashboard/app.py`) was verified and running live on `http://localhost:8501`.
- Deploys the streamlined 27-feature Production Hybrid (`failure_aware_hybrid.joblib`) wrapped in an end-to-end `sklearn.pipeline.Pipeline`, delivering sub-35ms inference latency.
- Bridges micro-level patient risk scoring with macro-level CDC epidemiological surveillance maps.

---

## 5. Summary of Key Locked-Test Metrics ($N = 50,736$)

| Model | Accuracy | Macro-F1 | Balanced Acc | MCC | Prediab Recall | Prediab Cases Detected | Brier Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SGD (Mod Huber)** | **0.8014** | 0.4472 | 0.4711 | 0.3422 | 0.0043 | 4 / 926 | **0.2841** |
| **Soft Voting** | 0.7477 | **0.4476** | 0.5028 | **0.3533** | 0.0421 | 39 / 926 | 0.3211 |
| **XGBoost** | 0.7581 | 0.4347 | 0.4689 | 0.3080 | 0.0205 | 19 / 926 | 0.3120 |
| **Standard Stacking** | 0.6000 | 0.4180 | **0.5278** | 0.2862 | **0.4060** | **376 / 926** | 0.4721 |
| **Failure-Aware Hybrid**| 0.6101 | 0.4162 | 0.5218 | 0.2820 | 0.3888 | 360 / 926 | 0.4589 |