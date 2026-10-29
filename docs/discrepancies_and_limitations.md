# Confirmed Project Discrepancies, Methodological Reconciliations, and Research Limitations

**Project Title:** Failure-Aware Ensemble Learning and Explainable AI for Three-Class Diabetes Status Prediction  
**Audit Standard:** Forensic integrity check across code, serialized models, predictions, and manuscript drafts  
**Verdict Summary:** 9 Discrepancies Audited & Reconciled; 4 Inherent Clinical Research Limitations Formally Documented  

---

## 1. Itemized Forensic Discrepancy Matrix

Every discrepancy identified across previous logs, notebooks, and manuscript drafts was investigated, audited against ground-truth source code, and resolved with publication-grade recommendations:

### DISC_01: Model Nomenclature: Estimator Labeled 'SVM' is SGDClassifier with Modified Huber Loss
- **Audit Classification:** `CONFIRMED & RECONCILED`
- **Empirical Evidence:** Inspecting `models/base_models/svm.joblib` reveals `SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.0001, random_state=42)`.
- **Scientific Impact:** True quadratic kernel SVM (`sklearn.svm.SVC`) has O(N^2) to O(N^3) complexity and cannot train on N = 202,944 on standard hardware. SGD with modified Huber loss is a linear margin classifier providing calibrated probabilities.
- **Required Resolution in Manuscript:** Re-label throughout the manuscript as 'SVM (SGD Modified Huber)' or 'Linear Margin Classifier with Modified Huber Loss'. Provide computational justification in Section 5.

### DISC_02: Dual Hybrid Architectures: 50 Meta-Features vs 27 Meta-Features
- **Audit Classification:** `CONFIRMED & RECONCILED`
- **Empirical Evidence:** `hybrid_meta_model.joblib` has `n_features_in_ = 50` (7 base models). `failure_aware_hybrid.joblib` has `n_features_in_ = 27` (5 base models, Pipeline wrapped).
- **Scientific Impact:** Describing both as 'the hybrid' without distinction causes peer-review confusion.
- **Required Resolution in Manuscript:** Explicitly define both: (1) The 50-feature model is the authoritative 'Research Hybrid' evaluated in paper tables; (2) The 27-feature pipeline is the 'Production Dashboard Hybrid' optimized for sub-35ms web inference.

### DISC_03: Initial Stacking Sample Weight Bug and Corrected Re-evaluation
- **Audit Classification:** `CONFIRMED & RESOLVED`
- **Empirical Evidence:** In early project scripts, the standard stacking meta-learner was fitted without `sample_weight` or `class_weight='balanced'`, creating an unfair comparison against the weighted hybrid.
- **Scientific Impact:** Comparing a weighted model against an unweighted model is methodologically invalid.
- **Required Resolution in Manuscript:** Both Standard Stacking and the Failure-Aware Hybrid were retrained and re-evaluated using identical balanced class weighting on OOF predictions. Both are reported on equal, fair terms in all paper tables.

### DISC_04: Prediabetes Recall Value Divergence: 0.396328 vs 0.388769
- **Audit Classification:** `CONFIRMED & RECONCILED`
- **Empirical Evidence:** 0.396328 was produced during multi-threaded in-memory XGBoost OOF cross-validation (detecting 367/926 prediabetes cases). 0.388769 is the deterministic value from serialized `TEST_PREDICTIONS_MASTER.csv` (detecting 360/926 cases).
- **Scientific Impact:** Absolute difference is exactly 7 cases out of 926 (0.75%), well within the 95% bootstrap CI [0.3587, 0.4207].
- **Required Resolution in Manuscript:** Authoritative value is established as 0.3888 (38.88%, 360/926 cases) because it matches the permanent serialized disk predictions.

### DISC_05: Contradictory 'Highest Accuracy' Claims
- **Audit Classification:** `CONFIRMED & RECONCILED`
- **Empirical Evidence:** SGDClassifier achieves 0.8014 raw accuracy. Soft Voting achieves 0.7477 accuracy. Standard Stacking achieves 0.6000. Hybrid achieves 0.6101.
- **Scientific Impact:** Stating that 'Soft Voting achieves highest accuracy' without qualifiers contradicts SGD at 0.8014.
- **Required Resolution in Manuscript:** State precisely: SGDClassifier achieves the highest raw overall accuracy (80.14%) but fails on prediabetes (0.43% recall). Soft Voting achieves highest accuracy (74.77%) and Macro-F1 (0.4476) among multi-model ensembles. Hybrid achieves superior accuracy (61.01%) over Stacking (60.00%) while preserving 38.88% prediabetes recall.

### DISC_06: Reported Bootstrap P-Values: 0.076 vs 0.224
- **Audit Classification:** `CONFIRMED & RECONCILED`
- **Empirical Evidence:** 0.076 was a one-tailed bootstrap test. 0.224 is the authoritative two-tailed paired bootstrap p-value ($H_0: \Delta = 0$).
- **Scientific Impact:** Reporting one-tailed p-values creates an impression of p-hacking.
- **Required Resolution in Manuscript:** Report the authoritative two-tailed bootstrap p-value (p = 0.224 for Macro-F1, p = 0.168 for Balanced Accuracy) alongside the McNemar test (Chi2 = 122.95, p = 1.43e-28). Explain that McNemar tests overall classification outcome discordance while bootstrap tests harmonic mean parity.

### DISC_07: Role of Secondary Dataset (CDC Dataset 2: export.csv)
- **Audit Classification:** `CONFIRMED & RECONCILED`
- **Empirical Evidence:** `export.csv` contains 13,889 aggregate CDC USDSS/NHIS records from 2000 to 2024, not patient-level microdata.
- **Scientific Impact:** Claiming Dataset 2 is an 'external validation set' for ML predictions is scientifically false and grounds for rejection.
- **Required Resolution in Manuscript:** Clarify that Dataset 2 serves exclusively for macro-epidemiological surveillance, longitudinal trend analysis, and public health dashboard contextualization, completely separate from individual-level ML testing.

### DISC_08: Retention of Survey Duplicate Feature Vectors
- **Audit Classification:** `CONFIRMED & JUSTIFIED`
- **Empirical Evidence:** 23,899 duplicate rows exist in the raw data; 6,923 test instances share feature vectors with training rows.
- **Scientific Impact:** Peer reviewers may suspect synthetic duplication or data leakage.
- **Required Resolution in Manuscript:** Provide epidemiological proof: BRFSS consists of 21 discrete survey items where natural modal profiles (e.g., healthy young adults) repeat thousands of times. Removing duplicates would distort population density. Re-evaluation on unique rows confirms stability (60.42% vs 61.01% accuracy).

### DISC_09: Optuna Optimization History and Aborted Trials
- **Audit Classification:** `CONFIRMED & DOCUMENTED`
- **Empirical Evidence:** Notebook 09 executed 2 demo trials per model (8 total). EXP_02 configured 20 trials, but CatBoost and RF were aborted due to multi-hour execution limits.
- **Scientific Impact:** Claiming exhaustive 100-trial Bayesian tuning across all models would be factually inaccurate.
- **Required Resolution in Manuscript:** Disclose transparently: XGBoost and LightGBM completed 20 trials; Meta-Learner completed 30 trials; CatBoost and RF utilized well-regularized default parameters due to computational constraints on N = 202,944.

---

## 2. Inherent Research Limitations Requiring Author Disclosure

To ensure unimpeachable scientific integrity, the manuscript's Discussion section must explicitly disclose four inherent methodological limitations:

1. **Self-Reported Survey Data:** BRFSS responses are based on telephone surveys and subject to self-report and recall bias. Undiagnosed diabetes cases may be misclassified as normoglycemic in the ground truth labels.
2. **Absence of Clinical Biomarkers:** The dataset lacks physiological laboratory parameters (fasting plasma glucose, 2-hour oral glucose tolerance, and glycated hemoglobin HbA1c). Models rely entirely on proxy clinical questions, capping theoretical discrimination.
3. **Cross-Sectional Observational Design:** BRFSS provides a cross-sectional snapshot of the population. Feature associations (e.g., BMI, hypertension, lifestyle) are strictly correlational and cannot prove longitudinal disease progression.
4. **Demographic Screening Disparity:** The model achieves 44.2% prediabetes recall in older adults (65+) but only 23.4% in younger adults (<45), because younger individuals have not yet accumulated overt secondary clinical comorbidities.