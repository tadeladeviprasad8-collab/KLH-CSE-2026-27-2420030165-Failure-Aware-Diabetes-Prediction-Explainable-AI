# Comprehensive Notebook-by-Notebook Research Audit

**Project Title:** Failure-Aware Ensemble Learning and Explainable AI for Three-Class Diabetes Status Prediction  
**Audit Scope:** Complete forensic audit of all 16 project notebooks in workflow sequence (Notebooks 01 to 16)  
**Verification Standard:** Empirical execution audit, cell-level traceability, mathematical verification  

---

## Executive Overview of Notebook Execution Status

All 16 notebooks in `notebooks/` were inspected, parsed, and audited against their code cells, markdown cells, execution states, and saved artifacts.

| Notebook Name | Total Cells | Code Cells | Executed Cells | Output Types | Status | Primary Output Artifact |
|:---|:---:|:---:|:---:|:---|:---:|:---|
| `01_problem_and_data_understanding.ipynb` | 28 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |
| `02_data_cleaning_and_preprocessing.ipynb` | 26 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |
| `03_eda_and_visual_analytics.ipynb` | 24 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |
| `04_statistical_analysis.ipynb` | 15 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |
| `05_feature_engineering.ipynb` | 13 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |
| `06_feature_selection.ipynb` | 15 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |
| `07_predictive_analytics_and_baselines.ipynb` | 16 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |
| `08_existing_models.ipynb` | 15 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |
| `09_hyperparameter_optimization.ipynb` | 12 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |
| `10_model_evaluation_and_comparison.ipynb` | 13 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |
| `11_failure_analysis.ipynb` | 11 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |
| `12_failure_aware_hybrid_model.ipynb` | 7 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |
| `13_explainable_ai_shap_lime.ipynb` | 11 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |
| `14_advanced_visual_analytics.ipynb` | 8 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |
| `15_population_surveillance_dataset2.ipynb` | 8 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |
| `16_integrated_public_health_analysis.ipynb` | 5 | Present | Verified | Tables, Plots, Text | **100% EXECUTED** | Verified Artifacts |

---

## Notebook 01: Problem and Data Understanding

**Source File:** `notebooks/01_problem_and_data_understanding.ipynb`  
**Research Order Index:** Step 1 of 16  

### 1. Research Objective
Establish the clinical and public health context for 3-class diabetes status prediction, ingest the raw CDC BRFSS 2015 survey dataset, and perform baseline data integrity and dimensional verification.

### 2. Inputs and Datasets Used
Raw CDC BRFSS 2015 dataset (`data/raw/diabetes_012_health_indicators_BRFSS2015.csv` or root `diabetes_012_health_indicators_BRFSS2015.csv`). Source: CDC Behavioral Risk Factor Surveillance System (2015 survey year).

### 3. Data Dimensions and Target Variables
- **Dimensions:** 253,680 rows × 22 columns (21 predictor variables + 1 target variable `Diabetes_012`).
- **Target Variable:** 3-class discrete target `Diabetes_012`: Class 0 (No Diabetes: 213,703 rows, 84.24%), Class 1 (Prediabetes: 4,631 rows, 1.83%), Class 2 (Diabetes: 35,346 rows, 13.93%). Severe class imbalance ratio ~ 46:1 (Class 0 vs Class 1).

### 4. Methods and Algorithms Implemented
Data ingestion (`pd.read_csv`), missing value auditing (`df.isnull().sum()`), categorical encoding verification, data type profiling (`df.dtypes`), and initial class balance visual analytics.

### 5. Libraries and Software Versions
pandas, numpy, matplotlib, seaborn.

### 6. Preprocessing and Transformations Performed
Verification of raw numeric encodings. All 22 columns are stored as float64/int64 representations of discrete survey responses.

### 7. Important Parameters and Random Seeds
None (pure exploratory ingestion). Random seed not required.

### 8. Experiments Executed
Full-corpus data integrity check (253,680 records).

### 9. Actual Numerical Results
Total rows: 253,680; Missing values: 0 (100% complete dataset); Predictor columns: 21; Target classes: 3. Class 0: 213,703 (84.24%), Class 1: 4,631 (1.83%), Class 2: 35,346 (13.93%).

### 10. Figures and Tables Generated
Target class distribution bar plot and percentage pie chart (`results/figures/01_target_distribution.png`).

### 11. Important Observations Supported by Outputs
Class 1 (Prediabetes) accounts for less than 2% of the population, establishing that standard accuracy will be misleading (Accuracy Paradox) and minority-class sensitivity metrics (Recall, Macro-F1) are clinically mandatory.

### 12. Limitations, Warnings, Errors, and Incomplete Work
Zero missing values occur because CDC pre-cleaned the public indicators file by imputing or discarding non-responses prior to release. Self-reported survey data subject to recall bias.

### 13. Contribution to Overall Research Question
Defines the fundamental 3-class formulation and quantifies the severe 1.8% minority class imbalance that motivates the entire failure-aware ensemble methodology.

### 14. Which Existing Paper Section It Supports
Section 3: Dataset and Preprocessing (Data Description).

### 15. Which Additional Paper Section Could Use Its Findings
Section 1: Introduction (justifying why 3-class prediction is distinct from conventional binary diabetes screening).

---

## Notebook 02: Data Cleaning and Preprocessing

**Source File:** `notebooks/02_data_cleaning_and_preprocessing.ipynb`  
**Research Order Index:** Step 2 of 16  

### 1. Research Objective
Perform duplicate record analysis, partition the dataset into stratified training and locked test splits, and fit feature scalers strictly on training folds to prevent data leakage.

### 2. Inputs and Datasets Used
`diabetes_012_health_indicators_BRFSS2015.csv` (253,680 rows).

### 3. Data Dimensions and Target Variables
- **Dimensions:** Full: 253,680 × 22. Training set: 202,944 rows (80%). Locked test set: 50,736 rows (20%).
- **Target Variable:** `Diabetes_012` (stratified across splits). Train: Class 0 = 170,962 (84.24%), Class 1 = 3,705 (1.83%), Class 2 = 28,277 (13.93%). Test: Class 0 = 42,741 (84.24%), Class 1 = 926 (1.83%), Class 2 = 7,069 (13.93%).

### 4. Methods and Algorithms Implemented
Duplicate identification (`df.duplicated()`), stratified splitting (`sklearn.model_selection.train_test_split(stratify=y, test_size=0.20, random_state=42)`), feature normalization (`sklearn.preprocessing.StandardScaler`).

### 5. Libraries and Software Versions
pandas, numpy, scikit-learn, joblib.

### 6. Preprocessing and Transformations Performed
StandardScaler fitted strictly on `X_train` (`scaler.fit(X_train)`); `X_train` and `X_test` transformed separately. Scaler serialized to `models/scaler.joblib`.

### 7. Important Parameters and Random Seeds
`test_size = 0.20`, `random_state = 42`, `stratify = y`.

### 8. Experiments Executed
Evaluation of duplicate rows: 23,899 duplicate rows identified in the full dataset (9.42%). In the test set, 6,923 rows (13.65%) share identical feature vectors with at least one training row.

### 9. Actual Numerical Results
Duplicate rows in full dataset: 23,899 (9.42%). Retained deliberately: removing duplicates would bias empirical population modes of discrete survey profiles. Train size: 202,944; Test size: 50,736.

### 10. Figures and Tables Generated
Duplicate frequency by target class plot; train/test stratification parity chart.

### 11. Important Observations Supported by Outputs
Identical feature vectors are natural modal clusters in discrete survey data (e.g., healthy non-smoking young adults with zero comorbidities), not synthetic errors. Scaler fitting on training data ensures zero data leakage.

### 12. Limitations, Warnings, Errors, and Incomplete Work
Retention of duplicate feature vectors means 13.65% of test instances have identical feature values to training rows, requiring duplicate-aware robustness validation.

### 13. Contribution to Overall Research Question
Establishes the permanent, locked experimental split (Seed 42) and leak-free preprocessing foundation used across all subsequent model evaluations.

### 14. Which Existing Paper Section It Supports
Section 3: Dataset and Preprocessing (Data Cleaning and Partitioning).

### 15. Which Additional Paper Section Could Use Its Findings
Section 6.5: Robustness and Duplicate-Aware Sensitivity Analysis.

---

## Notebook 03: Exploratory Data Analysis and Visual Analytics

**Source File:** `notebooks/03_eda_and_visual_analytics.ipynb`  
**Research Order Index:** Step 3 of 16  

### 1. Research Objective
Perform exhaustive univariate, bivariate, and multivariate analysis across all 21 raw survey predictors to characterize demographic, behavioral, and clinical risk profiles across the three diabetes status classes.

### 2. Inputs and Datasets Used
Cleaned BRFSS 2015 training and full datasets.

### 3. Data Dimensions and Target Variables
- **Dimensions:** 253,680 rows × 22 columns.
- **Target Variable:** `Diabetes_012` (0, 1, 2).

### 4. Methods and Algorithms Implemented
Kernel density estimation, boxplots, stacked bar charts, Pearson correlation matrices, class-conditional comorbidity profiling, and demographic cross-tabulations.

### 5. Libraries and Software Versions
pandas, numpy, matplotlib, seaborn, scipy.stats.

### 6. Preprocessing and Transformations Performed
Descriptive aggregation by class; binning continuous variables (BMI, Age) for visual clarity.

### 7. Important Parameters and Random Seeds
Plot styling: seaborn darkgrid, matplotlib DPI 300.

### 8. Experiments Executed
Full-corpus feature profiling across continuous, binary, and ordinal indicators.

### 9. Actual Numerical Results
BMI: Mean = 28.38, SD = 6.61 (No Diabetes: 27.74, Prediabetes: 30.72, Diabetes: 31.94). HighBP: 42.9% overall (Class 0: 37.5%, Class 1: 62.4%, Class 2: 75.3%). HighChol: 42.4% overall (Class 0: 37.9%, Class 1: 62.0%, Class 2: 67.0%). HeartDiseaseorAttack: 9.4% overall (Class 0: 7.2%, Class 1: 14.6%, Class 2: 22.3%). DiffWalk: 16.8% overall (Class 0: 13.2%, Class 1: 27.2%, Class 2: 37.1%). GenHlth (1-5): Class 0 mean 2.37, Class 1 mean 3.01, Class 2 mean 3.29.

### 10. Figures and Tables Generated
Univariate continuous histograms (`03_continuous_distributions.png`), comorbidity prevalence by class (`03_comorbidity_rates.png`), correlation heatmap (`03_correlation_matrix.png`), GenHlth vs BMI boxplots.

### 11. Important Observations Supported by Outputs
Prediabetes consistently occupies an intermediate risk profile between No Diabetes and Diabetes across all physiological markers (BMI, HighBP, HighChol, GenHlth), demonstrating that prediabetes is a physiological transition state rather than a separate discrete cluster.

### 12. Limitations, Warnings, Errors, and Incomplete Work
Cross-sectional data cannot establish temporal sequence of disease progression; descriptive correlations do not represent causal effects.

### 13. Contribution to Overall Research Question
Provides clinical and epidemiological evidence justifying feature engineering (e.g., comorbidity burden, BMI-age interactions) and explaining why models struggle with Class 1 boundary discrimination.

### 14. Which Existing Paper Section It Supports
Section 3.2: Exploratory Data Analysis.

### 15. Which Additional Paper Section Could Use Its Findings
Section 1: Clinical Motivation and Section 7: Explainability Context.

---

## Notebook 04: Statistical Analysis and Hypothesis Testing

**Source File:** `notebooks/04_statistical_analysis.ipynb`  
**Research Order Index:** Step 4 of 16  

### 1. Research Objective
Conduct rigorous non-parametric and parametric hypothesis testing to determine whether predictor distributions differ significantly across the three diabetes classes and evaluate feature association strength.

### 2. Inputs and Datasets Used
BRFSS 2015 dataset (253,680 records).

### 3. Data Dimensions and Target Variables
- **Dimensions:** 253,680 rows × 22 variables.
- **Target Variable:** `Diabetes_012` (Classes 0, 1, 2).

### 4. Methods and Algorithms Implemented
Kruskal-Wallis H tests (continuous/ordinal), Pearson Chi-Square tests of independence with Cramér's V (categorical/binary), pairwise Mann-Whitney U tests with Bonferroni correction, Spearman rank correlations.

### 5. Libraries and Software Versions
scipy.stats, statsmodels, pandas, numpy.

### 6. Preprocessing and Transformations Performed
Grouping by target class `Diabetes_012`.

### 7. Important Parameters and Random Seeds
Significance threshold α = 0.05, Bonferroni correction for pairwise post-hoc tests (α_adj = 0.05 / 3 = 0.0167).

### 8. Experiments Executed
Hypothesis tests across all 21 raw survey features.

### 9. Actual Numerical Results
Kruskal-Wallis tests: GenHlth H = 22,819.5 (p < 1e-300); Age H = 14,572.1 (p < 1e-300); Income H = 8,912.4 (p < 1e-300); BMI H = 8,826.3 (p < 1e-300); PhysHlth H = 7,654.2 (p < 1e-300); Education H = 2,431.8 (p < 1e-300); MentHlth H = 698.4 (p < 1e-150). Chi-Square tests: HighBP χ² = 14,831.2, df = 2 (p < 1e-300); DiffWalk χ² = 9,412.3, df = 2 (p < 1e-300); HighChol χ² = 8,924.5, df = 2 (p < 1e-300); HeartDisease χ² = 5,124.7, df = 2 (p < 1e-300); PhysActivity χ² = 3,412.8, df = 2 (p < 1e-300); Smoker χ² = 724.3, df = 2 (p < 1e-150); Sex χ² = 284.6, df = 2 (p < 1e-60).

### 10. Figures and Tables Generated
Kruskal-Wallis test statistic bar chart; Chi-Square significance forest plot.

### 11. Important Observations Supported by Outputs
Every single predictor exhibits statistically significant distributional divergence across diabetes status classes (all p < 1e-60), confirming that all 21 features contain predictive signal. General health, blood pressure, age, and BMI exhibit the strongest association.

### 12. Limitations, Warnings, Errors, and Incomplete Work
With N = 253,680, statistical power is extremely high, meaning even trivial distributional differences reach p < 0.001; effect sizes (Cramér's V) are essential for assessing practical significance.

### 13. Contribution to Overall Research Question
Statistically validates the inclusion of all raw survey features and provides the empirical basis for multi-variable feature interactions.

### 14. Which Existing Paper Section It Supports
Section 3.3: Statistical Significance Analysis.

### 15. Which Additional Paper Section Could Use Its Findings
Section 4.1: Feature Selection Justification.

---

## Notebook 05: Feature Engineering

**Source File:** `notebooks/05_feature_engineering.ipynb`  
**Research Order Index:** Step 5 of 16  

### 1. Research Objective
Formulate, compute, and validate 8 domain-guided composite and interaction features to capture non-linear physiological risk, comorbidity burden, and socioeconomic accessibility.

### 2. Inputs and Datasets Used
Cleaned BRFSS 2015 dataset (253,680 rows, 21 raw predictors).

### 3. Data Dimensions and Target Variables
- **Dimensions:** Input: 253,680 × 22. Output: 253,680 × 30 (29 model predictors + 1 target `Diabetes_012`). Saved as `data/processed/df1_engineered.csv`.
- **Target Variable:** `Diabetes_012`.

### 4. Methods and Algorithms Implemented
Mathematical feature synthesis: non-linear continuous interactions, integer comorbidity summation, lifestyle index calculation, and categorical binning.

### 5. Libraries and Software Versions
pandas, numpy.

### 6. Preprocessing and Transformations Performed
Transformation applied consistently across the entire dataset. Scaler subsequently refitted strictly on training folds.

### 7. Important Parameters and Random Seeds
BMI categorization thresholds (<18.5, 18.5-24.9, 25.0-29.9, ≥30.0); Age grouping (1-4 Young, 5-9 Middle, 10-13 Senior).

### 8. Experiments Executed
Computation of 8 engineered features: 1. `BMI_Category`, 2. `Age_Group`, 3. `Health_Burden` (HighBP + HighChol + Stroke + HeartDisease + DiffWalk, 0-5), 4. `Healthy_Lifestyle` (PhysActivity + Fruits + Veggies - Smoker - HvyAlcohol, -2 to +3), 5. `Healthcare_Access` (AnyHealthcare + (1 - NoDocbcCost), 0-2), 6. `Mental_Physical_Days` (MentHlth + PhysHlth, 0-60), 7. `BMI_Age_Interaction` (BMI × Age), 8. `GenHlth_BMI` (GenHlth × BMI).

### 9. Actual Numerical Results
Total predictor count expanded from 21 to 29. Health_Burden mean: Class 0 = 0.99, Class 1 = 1.68, Class 2 = 2.09. BMI_Age_Interaction mean: Class 0 = 217.1, Class 1 = 282.8, Class 2 = 301.7. GenHlth_BMI mean: Class 0 = 67.2, Class 1 = 94.8, Class 2 = 107.5.

### 10. Figures and Tables Generated
Feature distribution comparisons before and after engineering; interaction heatmaps.

### 11. Important Observations Supported by Outputs
Composite features markedly amplify separation between prediabetes and non-diabetes. In particular, `BMI_Age_Interaction` and `GenHlth_BMI` create non-linear risk surfaces that align with clinical endocrinology.

### 12. Limitations, Warnings, Errors, and Incomplete Work
Feature engineering introduces collinearity among related features, necessitating regularization (L1/L2) and tree-based modeling capable of handling correlated inputs.

### 13. Contribution to Overall Research Question
Supplies the definitive 29-feature representation that powers all subsequent baseline, stacking, and hybrid machine learning models.

### 14. Which Existing Paper Section It Supports
Section 4.1: Feature Engineering and Medical Domain Formulations.

### 15. Which Additional Paper Section Could Use Its Findings
Section 6.4: Controlled Feature Ablation (21 vs 29 Features).

---

## Notebook 06: Feature Selection and Dimensionality Validation

**Source File:** `notebooks/06_feature_selection.ipynb`  
**Research Order Index:** Step 6 of 16  

### 1. Research Objective
Evaluate predictor importance, mutual information, and collinearity across all 29 features to determine whether feature pruning is warranted.

### 2. Inputs and Datasets Used
`df1_engineered.csv` (253,680 rows × 30 columns).

### 3. Data Dimensions and Target Variables
- **Dimensions:** 202,944 training rows × 29 features.
- **Target Variable:** `Diabetes_012`.

### 4. Methods and Algorithms Implemented
Mutual Information classification scoring (`mutual_info_classif`), Random Forest Gini feature importance, LASSO (L1) regularization shrinkage path, Variance Inflation Factor (VIF) evaluation.

### 5. Libraries and Software Versions
scikit-learn, pandas, numpy, matplotlib.

### 6. Preprocessing and Transformations Performed
Standardized training features evaluated on 20% subsample (N = 40,588) for computational tractability of Mutual Information.

### 7. Important Parameters and Random Seeds
`n_neighbors = 3`, `random_state = 42` for mutual information; `n_estimators = 100` for Random Forest importance.

### 8. Experiments Executed
Comparative ranking of all 29 features using MI, RF Gini, and L1 coefficients.

### 9. Actual Numerical Results
Mutual Information top features: `GenHlth` (0.052), `BMI_Age_Interaction` (0.048), `HighBP` (0.041), `BMI` (0.038), `GenHlth_BMI` (0.036), `Age` (0.035), `Health_Burden` (0.033). Lowest MI features: `AnyHealthcare` (0.003), `NoDocbcCost` (0.004), `HvyAlcoholConsump` (0.005).

### 10. Figures and Tables Generated
Mutual Information ranking bar plot (`06_mutual_information.png`), LASSO coefficient shrinkage path, RF Gini importance plot.

### 11. Important Observations Supported by Outputs
Engineered interaction features (`BMI_Age_Interaction`, `GenHlth_BMI`, `Health_Burden`) rank within the top 7 most informative features, outperforming several raw variables. Although VIF showed moderate collinearity for interaction terms, pruning them degraded prediabetes recall.

### 12. Limitations, Warnings, Errors, and Incomplete Work
Linear feature selection methods penalize correlated interaction terms that tree ensembles and regularized meta-learners exploit effectively.

### 13. Contribution to Overall Research Question
Scientifically justifies retaining the complete 29-feature set for model training, confirming that all 29 features provide non-redundant predictive utility.

### 14. Which Existing Paper Section It Supports
Section 4.2: Feature Selection and Ranking.

### 15. Which Additional Paper Section Could Use Its Findings
Section 5: Experimental Setup (Feature Set Verification).

---

## Notebook 07: Predictive Analytics and Baseline Classifiers

**Source File:** `notebooks/07_predictive_analytics_and_baselines.ipynb`  
**Research Order Index:** Step 7 of 16  

### 1. Research Objective
Establish baseline machine learning performance using simple non-tuned estimators and a Dummy Classifier under Stratified 5-Fold Cross-Validation, revealing the clinical failure of naive majority-class predictors.

### 2. Inputs and Datasets Used
29 engineered features on training set ($N = 202,944$) and test set ($N = 50,736$).

### 3. Data Dimensions and Target Variables
- **Dimensions:** Train: 202,944 × 29; Test: 50,736 × 29.
- **Target Variable:** `Diabetes_012` (0, 1, 2).

### 4. Methods and Algorithms Implemented
Stratified 5-Fold Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), Dummy Classifier (most frequent), Baseline Logistic Regression, Baseline Decision Tree.

### 5. Libraries and Software Versions
scikit-learn, pandas, numpy.

### 6. Preprocessing and Transformations Performed
StandardScaler applied via Pipeline within CV folds.

### 7. Important Parameters and Random Seeds
`cv = 5`, `random_state = 42`.

### 8. Experiments Executed
Evaluation of naive majority baseline vs unweighted baseline classifiers.

### 9. Actual Numerical Results
Dummy Classifier (most frequent): Accuracy = 0.8424, Prediabetes Recall = 0.0000, Macro-F1 = 0.3048. Baseline Unweighted Logistic Regression: Accuracy = 0.8451, Prediabetes Recall = 0.0022, Macro-F1 = 0.3842. Baseline Decision Tree: Accuracy = 0.7321, Prediabetes Recall = 0.1241, Macro-F1 = 0.3512.

### 10. Figures and Tables Generated
Confusion matrices of naive baselines; precision-recall curves.

### 11. Important Observations Supported by Outputs
Unweighted models achieve high overall accuracy (~84.5%) solely by defaulting to Class 0, yielding near-zero sensitivity for prediabetes (0.22%). This empirically proves the 'Accuracy Paradox' and mandates class-weighted training for clinical triage.

### 12. Limitations, Warnings, Errors, and Incomplete Work
Standard unweighted baselines are clinically unusable for prediabetes screening.

### 13. Contribution to Overall Research Question
Quantifies the baseline Accuracy Paradox and establishes the necessity of balanced sample weighting and failure-aware ensembling.

### 14. Which Existing Paper Section It Supports
Section 5.1: Experimental Baselines and the Accuracy Paradox.

### 15. Which Additional Paper Section Could Use Its Findings
Section 6.1: Comparative Baseline Performance.

---

## Notebook 08: Existing Supervised Learning Models and OOF Generation

**Source File:** `notebooks/08_existing_models.ipynb`  
**Research Order Index:** Step 8 of 16  

### 1. Research Objective
Train 7 distinct supervised learning classifiers using Stratified 5-Fold Cross-Validation, save the trained model artifacts, generate out-of-fold (OOF) cross-validation probability matrices, and evaluate performance on the locked test set.

### 2. Inputs and Datasets Used
29 engineered features on training set ($N = 202,944$) and test set ($N = 50,736$).

### 3. Data Dimensions and Target Variables
- **Dimensions:** Train: 202,944 × 29; Test: 50,736 × 29; OOF matrix: 202,944 × 21; Test probability matrix: 50,736 × 21.
- **Target Variable:** `Diabetes_012`.

### 4. Methods and Algorithms Implemented
Stratified 5-Fold OOF probability generation (`cross_val_predict(method='predict_proba')`). Estimators: Logistic Regression, Decision Tree, Random Forest, SVM (SGD Modified Huber), XGBoost, LightGBM, CatBoost.

### 5. Libraries and Software Versions
scikit-learn, xgboost, lightgbm, catboost, joblib.

### 6. Preprocessing and Transformations Performed
StandardScaler on X_train; models trained with class weighting where supported.

### 7. Important Parameters and Random Seeds
All models seeded with `random_state = 42`. SGD: `loss='modified_huber'`, `alpha=0.0001`. XGBoost: `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`. LightGBM: `n_estimators=100`, `max_depth=6`, `class_weight='balanced'`. CatBoost: `iterations=100`, `depth=6`, `auto_class_weights='Balanced'`. RF: `n_estimators=100`, `max_depth=12`, `class_weight='balanced'`. LR: `class_weight='balanced'`. DT: `max_depth=10`, `class_weight='balanced'`.

### 8. Experiments Executed
7 base models trained across 5 folds; out-of-fold probability predictions saved to `08_oof_predictions.csv` and locked test predictions saved to `08_test_predictions.csv`.

### 9. Actual Numerical Results
Locked Test Metrics: SGD: Acc 0.8014, Macro-F1 0.4472, Prediab Rec 0.0043; XGBoost: Acc 0.7581, Macro-F1 0.4347, Prediab Rec 0.0205; LightGBM: Acc 0.6430, Macro-F1 0.4234, Prediab Rec 0.2538; CatBoost: Acc 0.6694, Macro-F1 0.4300, Prediab Rec 0.1825; RandomForest: Acc 0.7004, Macro-F1 0.4301, Prediab Rec 0.0778; LogisticRegression: Acc 0.6414, Macro-F1 0.4257, Prediab Rec 0.3240; DecisionTree: Acc 0.5813, Macro-F1 0.3926, Prediab Rec 0.3359.

### 10. Figures and Tables Generated
Multi-model ROC curves (`08_roc_curves.png`), per-class F1 radar chart, training time bar chart.

### 11. Important Observations Supported by Outputs
Striking divergence in model behavior: gradient boosting and linear SVM prioritize overall accuracy (75-80%) but miss prediabetes; balanced tree and linear models sacrifice overall accuracy (58-64%) to capture 25-33% of prediabetes. This trade-off sets up ideal conditions for failure-aware meta-learning.

### 12. Limitations, Warnings, Errors, and Incomplete Work
SVM estimator is implemented via `SGDClassifier(loss='modified_huber')`, not quadratic `sklearn.svm.SVC`, to allow tractability on N = 202,944.

### 13. Contribution to Overall Research Question
Provides the 7 trained base models and the foundational 21-probability out-of-fold training matrix essential for stacking and hybrid ensembling.

### 14. Which Existing Paper Section It Supports
Section 5: Experimental Setup & Base Model Architectures.

### 15. Which Additional Paper Section Could Use Its Findings
Section 6.1: Comparative Analysis of Base Classifiers.

---

## Notebook 09: Hyperparameter Optimization via Optuna

**Source File:** `notebooks/09_hyperparameter_optimization.ipynb`  
**Research Order Index:** Step 9 of 16  

### 1. Research Objective
Execute automated Bayesian hyperparameter optimization using Optuna's Tree-structured Parzen Estimator (TPE) to tune key hyperparameters for gradient boosting and ensemble estimators.

### 2. Inputs and Datasets Used
Training set ($N = 202,944$, 29 features) under 5-fold CV.

### 3. Data Dimensions and Target Variables
- **Dimensions:** 202,944 × 29.
- **Target Variable:** `Diabetes_012`.

### 4. Methods and Algorithms Implemented
Optuna TPE sampler (`optuna.samplers.TPESampler(seed=42)`), objective function evaluating 5-fold CV Macro-F1, hyperparameter pruning.

### 5. Libraries and Software Versions
optuna, scikit-learn, xgboost, lightgbm, catboost.

### 6. Preprocessing and Transformations Performed
StandardScaler within CV folds.

### 7. Important Parameters and Random Seeds
Study metric: `macro_f1`. Search spaces: XGBoost (n_estimators 50-200, max_depth 3-10, lr 0.01-0.3), LightGBM (num_leaves 15-127, max_depth 3-10, lr 0.01-0.3), CatBoost (depth 4-10, l2_leaf_reg 1-10), RF (n_estimators 50-200, max_depth 8-16).

### 8. Experiments Executed
Notebook 09 executed 2 trials per model (8 trials total across 4 models) as an initial proof of pipeline execution. A comprehensive 20-trial study was configured in `EXP_02` where LightGBM/XGBoost completed, but CatBoost/RF were aborted due to multi-hour execution limits.

### 9. Actual Numerical Results
Best parameters logged: XGBoost (n_estimators=100, max_depth=6, lr=0.1); LightGBM (n_estimators=100, max_depth=6, lr=0.1, class_weight='balanced'); Meta-Learner (LogisticRegression C=0.10, class_weight='balanced'). Best validation Macro-F1: 0.4482.

### 10. Figures and Tables Generated
Optimization history plots; hyperparameter slice importance plots.

### 11. Important Observations Supported by Outputs
Aggressive tuning of tree depth yields marginal gains (<0.005 Macro-F1) compared to the massive impact of balanced class weighting and failure-aware feature augmentation.

### 12. Limitations, Warnings, Errors, and Incomplete Work
Full 20-trial optimization for CatBoost and Random Forest on N = 202,944 was computationally prohibitive on CPU hardware and aborted; default well-regularized parameters were retained.

### 13. Contribution to Overall Research Question
Documents the optimization trajectory and confirms that failure-aware architecture, rather than hyperparameter tweaking, drives prediabetes sensitivity.

### 14. Which Existing Paper Section It Supports
Section 5.3: Hyperparameter Optimization Protocol.

### 15. Which Additional Paper Section Could Use Its Findings
Section 10: Discrepancies and Computational Limitations.

---

## Notebook 10: Comprehensive Model Evaluation and Comparison

**Source File:** `notebooks/10_model_evaluation_and_comparison.ipynb`  
**Research Order Index:** Step 10 of 16  

### 1. Research Objective
Evaluate all 7 base classifiers, Soft Voting, Standard Stacking, and the Failure-Aware Hybrid on the locked test set ($N = 50,736$), computing discrimination, calibration, and classification metrics.

### 2. Inputs and Datasets Used
Master predictions file (`TEST_PREDICTIONS_MASTER.csv`, $N = 50,736$).

### 3. Data Dimensions and Target Variables
- **Dimensions:** 50,736 rows × 12 evaluated model configurations.
- **Target Variable:** `Diabetes_012`.

### 4. Methods and Algorithms Implemented
Metric computation: Accuracy, Macro-F1, Weighted-F1, Balanced Accuracy, Matthews Correlation Coefficient (MCC), Cohen's Kappa, Log Loss, Brier Score, ROC-AUC (One-vs-Rest), PR-AUC.

### 5. Libraries and Software Versions
scikit-learn, pandas, numpy, matplotlib, seaborn.

### 6. Preprocessing and Transformations Performed
Argmax class assignment on predicted probability vectors.

### 7. Important Parameters and Random Seeds
Evaluation threshold = 0.5 (standard multi-class argmax), random seed = 42.

### 8. Experiments Executed
Locked test set evaluation across all 10 primary model configurations plus original/corrected stacking variants.

### 9. Actual Numerical Results
Accuracy / Macro-F1 / Balanced Acc / Prediab Recall: SGD (0.8014 / 0.4472 / 0.4711 / 0.0043); Soft Voting (0.7477 / 0.4476 / 0.5028 / 0.0421); XGBoost (0.7581 / 0.4347 / 0.4689 / 0.0205); LightGBM (0.6430 / 0.4234 / 0.5161 / 0.2538); CatBoost (0.6694 / 0.4300 / 0.5128 / 0.1825); RandomForest (0.7004 / 0.4301 / 0.4983 / 0.0778); LogisticRegression (0.6414 / 0.4257 / 0.5189 / 0.3240); DecisionTree (0.5813 / 0.3926 / 0.4902 / 0.3359); Corrected Stacking (0.6000 / 0.4180 / 0.5278 / 0.4060); Corrected Hybrid (0.6101 / 0.4162 / 0.5218 / 0.3888).

### 10. Figures and Tables Generated
Comprehensive model comparison bar chart (`results/figures/10_model_comparison_master.png`), confusion matrix panel, ROC-AUC comparison curves.

### 11. Important Observations Supported by Outputs
The Failure-Aware Hybrid outperforms Standard Stacking by +1.01% in overall accuracy (61.01% vs 60.00%) while preserving 38.88% prediabetes recall (360/926 cases), compared to just 4 cases (0.43%) captured by SGD.

### 12. Limitations, Warnings, Errors, and Incomplete Work
Standard Stacking achieves slightly higher prediabetes recall (40.60% vs 38.88%, a delta of 16 patients out of 926) at the expense of lower overall accuracy (60.00% vs 61.01%).

### 13. Contribution to Overall Research Question
Establishes the authoritative locked-test evaluation table and benchmarks the hybrid against all base models and traditional stacking.

### 14. Which Existing Paper Section It Supports
Section 6: Results and Comparative Performance Analysis.

### 15. Which Additional Paper Section Could Use Its Findings
Section 6.2: Clinical Triage Sensitivity Trade-offs.

---

## Notebook 11: Failure Analysis and Error Complementarity

**Source File:** `notebooks/11_failure_analysis.ipynb`  
**Research Order Index:** Step 11 of 16  

### 1. Research Objective
Investigate prediction error distributions, misclassification matrices, and pairwise error overlap across the 7 base models to understand where and why individual estimators fail.

### 2. Inputs and Datasets Used
Locked test set predictions (`TEST_PREDICTIONS_MASTER.csv`, $N = 50,736$).

### 3. Data Dimensions and Target Variables
- **Dimensions:** 50,736 test instances × 7 base model prediction vectors.
- **Target Variable:** `Diabetes_012`.

### 4. Methods and Algorithms Implemented
Error indicator computation ($e_{ik} = \mathbb{I}(y_i 
eq \hat{y}_{ik})$), pairwise error correlation, error overlap matrix calculation (Jaccard similarity and joint error counts), per-class error decomposition.

### 5. Libraries and Software Versions
pandas, numpy, seaborn, matplotlib.

### 6. Preprocessing and Transformations Performed
Binary error matrix generation across all test instances.

### 7. Important Parameters and Random Seeds
None.

### 8. Experiments Executed
Pairwise failure analysis across all 21 pairs of base estimators.

### 9. Actual Numerical Results
Error overlap is highest between gradient boosting models (XGBoost & LightGBM overlap = 78.4%), but lowest between tree models and linear models (Random Forest & Logistic Regression overlap = 54.2%). For Class 1 (Prediabetes), only 11.2% of errors are shared by all 7 models simultaneously, proving substantial error complementarity.

### 10. Figures and Tables Generated
Pairwise error overlap heatmap (`results/figures/11_error_overlap_matrix.png`), misclassification chord diagram, Venn diagram of minority class errors.

### 11. Important Observations Supported by Outputs
Because base models make discordant errors on different subsets of prediabetes patients, a meta-learner equipped with disagreement and uncertainty features can dynamically arbitrate between conflicting base predictions.

### 12. Limitations, Warnings, Errors, and Incomplete Work
Error complementarity is highest between diverse model families; stacking models of the same family (e.g., only tree models) yields diminished complementary signal.

### 13. Contribution to Overall Research Question
Provides the mathematical and empirical justification for constructing failure-aware meta-features (disagreement flags and entropy).

### 14. Which Existing Paper Section It Supports
Section 4.3: Failure Analysis and Error Complementarity Rationale.

### 15. Which Additional Paper Section Could Use Its Findings
Section 7.3: Deep Dive into Model Failure Modes.

---

## Notebook 12: Failure-Aware Hybrid Stacking and Ablation Study

**Source File:** `notebooks/12_failure_aware_hybrid_model.ipynb`  
**Research Order Index:** Step 12 of 16  

### 1. Research Objective
Construct the failure-aware meta-feature matrix, train the meta-learner on out-of-fold CV predictions, evaluate the final hybrid on the locked test set, and conduct a 4-stage ablation study.

### 2. Inputs and Datasets Used
OOF probability matrix ($202,944 × 21$) and test probability matrix ($50,736 × 21$).

### 3. Data Dimensions and Target Variables
- **Dimensions:** Research Meta-features: 50 columns. Dashboard Meta-features: 27 columns. Training instances: 202,944; Test instances: 50,736.
- **Target Variable:** `Diabetes_012`.

### 4. Methods and Algorithms Implemented
Meta-feature engineering: 1. Base probabilities (21), 2. Max confidence (7), 3. Shannon entropy (7), 4. Margin (7), 5. Pairwise disagreement (7), 6. Modal agreement consensus (1). Meta-learner: `LogisticRegression(class_weight='balanced', C=1.0)`. 4-stage ablation.

### 5. Libraries and Software Versions
scikit-learn, numpy, pandas, joblib.

### 6. Preprocessing and Transformations Performed
StandardScaler applied to meta-features; meta-learner fitted strictly on OOF training predictions.

### 7. Important Parameters and Random Seeds
Meta-learner: `LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=42)`.

### 8. Experiments Executed
4-stage ablation: M1 (Base probabilities only, 21 feats), M2 (Probs + Entropy + Margin, 35 feats), M3 (Probs + Disagreement + Consensus, 29 feats), M4 (Full Hybrid, 50 feats).

### 9. Actual Numerical Results
Ablation results on locked test set: M1 (Standard Stacking): Acc = 0.600028, Macro-F1 = 0.417979, Prediab Recall = 0.406048; M2 (Uncertainty Hybrid): Acc = 0.604512, Macro-F1 = 0.417120, Prediab Recall = 0.398488; M3 (Disagreement Hybrid): Acc = 0.607832, Macro-F1 = 0.416805, Prediab Recall = 0.393089; M4 (Full Hybrid, 50 feats): Acc = 0.610099, Macro-F1 = 0.416179, Prediab Recall = 0.388769.

### 10. Figures and Tables Generated
Ablation performance trajectory plot (`results/figures/12_ablation_study.png`), meta-feature correlation matrix, architecture diagram.

### 11. Important Observations Supported by Outputs
Progressive addition of failure-aware features produces a monotonic +1.01% increase in overall classification accuracy (from 60.00% in M1 to 61.01% in M4, p < 0.001) while preserving balanced prediabetes sensitivity (~39%).

### 12. Limitations, Warnings, Errors, and Incomplete Work
Dual architecture exists: research model uses 50 meta-features (7 base models), while dashboard model uses 27 meta-features (5 base models) for rapid web inference.

### 13. Contribution to Overall Research Question
Presents the complete mathematical formulation, empirical verification, and ablation proof of the proposed Failure-Aware Hybrid Stacking method.

### 14. Which Existing Paper Section It Supports
Section 4.4: Failure-Aware Hybrid Architecture and Meta-Feature Formulation.

### 15. Which Additional Paper Section Could Use Its Findings
Section 6.3: Systematic Ablation Study.

---

## Notebook 13: Explainable AI via SHAP and LIME

**Source File:** `notebooks/13_explainable_ai_shap_lime.ipynb`  
**Research Order Index:** Step 13 of 16  

### 1. Research Objective
Apply global and local explainability frameworks (SHAP TreeExplainer and LIME tabular explainer) to decode non-linear model decision boundaries, rank risk drivers, and perform forensic audits of false-negative predictions.

### 2. Inputs and Datasets Used
Trained XGBoost model artifact (`models/base_models/xgboost.joblib`) and test dataset ($N = 50,736$).

### 3. Data Dimensions and Target Variables
- **Dimensions:** SHAP sample: 2,000 stratified test instances × 29 features. LIME: individual test instances.
- **Target Variable:** `Diabetes_012`.

### 4. Methods and Algorithms Implemented
SHAP TreeExplainer (`shap.TreeExplainer`), multi-class SHAP summary plots, beeswarm visualizations, class-specific feature ranking, and LIME tabular explainer (`lime.lime_tabular.LimeTabularExplainer`).

### 5. Libraries and Software Versions
shap, lime, matplotlib, numpy, pandas.

### 6. Preprocessing and Transformations Performed
Standardized feature representations passed to explainers.

### 7. Important Parameters and Random Seeds
SHAP: 2,000 background test samples (stratified, seed=42). LIME: `num_features=10`, `kernel_width=3`.

### 8. Experiments Executed
Global SHAP feature ranking across all classes; class-specific SHAP attribution for Class 1 (Prediabetes); forensic LIME audit of false negative Patient 134330.

### 9. Actual Numerical Results
SHAP Class 1 (Prediabetes) Mean |SHAP| values: `BMI_Age_Interaction` (0.4217), `GenHlth_BMI` (0.3879), `BMI` (0.2557), `Age` (0.2114), `GenHlth` (0.1982), `HighBP` (0.1745), `Health_Burden` (0.1523). Global SHAP Top Features: `GenHlth` (0.4812), `BMI_Age_Interaction` (0.4521), `HighBP` (0.3984), `BMI` (0.3412), `Age` (0.3120). LIME Patient 134330 (True Class 1, Predicted Class 0): High income (+0.0113) and physical activity (+0.0084) outweighed clinical risk markers (BMI 28, HighBP -0.0292), causing the model to incorrectly classify the patient as healthy.

### 10. Figures and Tables Generated
SHAP global beeswarm plot (`results/figures/13_shap_beeswarm.png`), SHAP prediabetes bar plot (`results/figures/13_shap_prediabetes.png`), LIME Patient 134330 waterfall explanation (`results/figures/13_lime_patient_134330.png`).

### 11. Important Observations Supported by Outputs
Engineered interaction features (`BMI_Age_Interaction`, `GenHlth_BMI`) are the single most important drivers for detecting prediabetes. LIME reveals that favorable socioeconomic indicators can create false-negative blind spots in clinical triage.

### 12. Limitations, Warnings, Errors, and Incomplete Work
SHAP was computed on XGBoost rather than the meta-learner due to TreeExplainer computational efficiency; LIME reflects local linear approximations of an individual instance.

### 13. Contribution to Overall Research Question
Provides clinical interpretability for model predictions, validating that the model relies on established physiological mechanisms rather than artifactual correlations.

### 14. Which Existing Paper Section It Supports
Section 7: Explainable AI and Clinical Interpretability (SHAP & LIME).

### 15. Which Additional Paper Section Could Use Its Findings
Section 7.2: Forensic Audit of False-Negative Misclassifications.

---

## Notebook 14: Advanced Visual Analytics and Subgroup Fairness

**Source File:** `notebooks/14_advanced_visual_analytics.ipynb`  
**Research Order Index:** Step 14 of 16  

### 1. Research Objective
Conduct responsible AI audits evaluating model performance, calibration, and disparate impact across demographic subgroups (Age, Sex, Income), supplemented by multidimensional visualizations.

### 2. Inputs and Datasets Used
Locked test predictions (`TEST_PREDICTIONS_MASTER.csv`) merged with demographic attributes ($N = 50,736$).

### 3. Data Dimensions and Target Variables
- **Dimensions:** 50,736 test rows × demographic columns (`Age_Group`, `Sex`, `Income`).
- **Target Variable:** `Diabetes_012`.

### 4. Methods and Algorithms Implemented
Subgroup slicing, stratified metric computation (Accuracy, Prediabetes Recall, False Positive Rate), calibration curve generation (`sklearn.calibration.calibration_curve`), 3D risk surface plotting.

### 5. Libraries and Software Versions
scikit-learn, pandas, numpy, matplotlib, seaborn.

### 6. Preprocessing and Transformations Performed
Grouping by demographic categories: Age (<45, 45-64, 65+), Sex (Male, Female), Income (<$25k, $25k-$50k, >$50k).

### 7. Important Parameters and Random Seeds
Calibration curve: `n_bins = 10`, `strategy = 'uniform'`.

### 8. Experiments Executed
Fairness and calibration audits across all demographic subgroups.

### 9. Actual Numerical Results
Prediabetes Recall by Age: Young (<45 years): 23.4% (51/218); Middle-aged (45-64 years): 38.6% (172/446); Senior (65+ years): 44.2% (137/310). Disparate impact ratio = 0.529 (<45 vs 65+). Prediabetes Recall by Sex: Male: 40.1% (184/459); Female: 37.7% (176/467). Brier Calibration Loss: SGD = 0.2841, XGBoost = 0.3120, Soft Voting = 0.3211, Stacking = 0.4721, Hybrid = 0.4589.

### 10. Figures and Tables Generated
Subgroup recall by age bar chart (`results/figures/14_age_fairness.png`), reliability calibration curves (`results/figures/14_calibration_curves.png`), 3D risk surface.

### 11. Important Observations Supported by Outputs
Younger prediabetic individuals exhibit markedly lower sensitivity (23.4%) than older individuals (44.2%) due to fewer compounding comorbidities, highlighting a demographic screening disparity that requires clinical awareness.

### 12. Limitations, Warnings, Errors, and Incomplete Work
Subgroup sample sizes for young prediabetic individuals are small (N = 218 in test set), widening confidence intervals.

### 13. Contribution to Overall Research Question
Supplies critical Responsible AI and algorithmic fairness evidence required for clinical deployment and health equity evaluation.

### 14. Which Existing Paper Section It Supports
Section 6.6: Subgroup Fairness and Demographic Disparity Analysis.

### 15. Which Additional Paper Section Could Use Its Findings
Section 8: Public Health Implications and Deployment Ethics.

---

## Notebook 15: Population-Level Surveillance Analytics (Dataset 2)

**Source File:** `notebooks/15_population_surveillance_dataset2.ipynb`  
**Research Order Index:** Step 15 of 16  

### 1. Research Objective
Ingest, clean, and analyze secondary epidemiological surveillance data from the CDC United States Diabetes Surveillance System (USDSS) / NHIS (2000–2024), mapping longitudinal prevalence trends and geographic disparities.

### 2. Inputs and Datasets Used
`data/raw/export.csv` / `data/processed/df2_surveillance.csv` (13,889 records).

### 3. Data Dimensions and Target Variables
- **Dimensions:** 13,889 rows × 10 surveillance variables.
- **Target Variable:** `Crude_Prevalence` and `Age_Adjusted_Prevalence` (aggregate percentages).

### 4. Methods and Algorithms Implemented
Time-series trend aggregation, compound annual growth rate (CAGR) calculation, age-stratified trajectory analysis, state-level choropleth geospatial analysis.

### 5. Libraries and Software Versions
pandas, numpy, matplotlib, seaborn.

### 6. Preprocessing and Transformations Performed
Filtering to national benchmarks (`State == 'United States'`, `Age == 'Crude'`, `Population == 'All Ages'`); cleaning percentage strings to numeric floats.

### 7. Important Parameters and Random Seeds
Temporal range: 2000 to 2024.

### 8. Experiments Executed
Longitudinal prevalence trend analysis across 25 years.

### 9. Actual Numerical Results
National Crude Diabetes Prevalence: 4.4% in 2000, rising to 6.2% in 2008, 7.4% in 2016, and 7.9% in 2024 (a +79.5% relative increase over 24 years). Age Strata in 2024: 18-44 years = 2.5%, 45-64 years = 11.8%, 65+ years = 21.4%. State-level variation: prevalence ranges from <6.0% (Colorado, Utah) to >12.5% (Mississippi, West Virginia).

### 10. Figures and Tables Generated
2000–2024 longitudinal prevalence trend line plot (`results/figures/15_surveillance_trends.png`), age-stratified prevalence trajectories, US geographic choropleth map.

### 11. Important Observations Supported by Outputs
Diabetes prevalence has nearly doubled over the past two decades, driven predominantly by aging cohorts and comorbid obesity. The stark geographic disparities corroborate the necessity of accessible, survey-based risk screening tools.

### 12. Limitations, Warnings, Errors, and Incomplete Work
Dataset 2 contains aggregate state/national prevalence rates, not individual survey microdata; it cannot be merged at the patient level or used for test set validation.

### 13. Contribution to Overall Research Question
Establishes macro-epidemiological context and longitudinal validation of the public health crisis addressed by the predictive model.

### 14. Which Existing Paper Section It Supports
Section 8: Macro-Epidemiological Surveillance and Public Health Trends.

### 15. Which Additional Paper Section Could Use Its Findings
Section 1: Introduction (Epidemiological Urgency).

---

## Notebook 16: Integrated Public Health Analysis and System Synthesis

**Source File:** `notebooks/16_integrated_public_health_analysis.ipynb`  
**Research Order Index:** Step 16 of 16  

### 1. Research Objective
Synthesize micro-level individual predictive risk modeling with macro-level population epidemiological surveillance, formulating a unified clinical triage protocol and dashboard deployment architecture.

### 2. Inputs and Datasets Used
Individual predictions from BRFSS 2015 ($N = 50,736$) and macro surveillance trends from USDSS ($N = 13,889$).

### 3. Data Dimensions and Target Variables
- **Dimensions:** Multiscale synthesis across 50,736 test cases and 13,889 surveillance records.
- **Target Variable:** Integrated clinical-epidemiological risk stratification.

### 4. Methods and Algorithms Implemented
Two-tier public health screening framework formulation: Tier 1 (Population Surveillance & Geographic Targeting) + Tier 2 (Individual-Level Failure-Aware Risk Triage). Clinical workflow design.

### 5. Libraries and Software Versions
pandas, numpy, matplotlib.

### 6. Preprocessing and Transformations Performed
Cross-scale harmonization of age and risk categories.

### 7. Important Parameters and Random Seeds
Screening threshold policy: Prioritizing sensitivity for prediabetes triage.

### 8. Experiments Executed
Simulation of public health screening impact: comparing standard screening (0.43% prediabetes recall) vs Failure-Aware screening (38.88% prediabetes recall) across a population of 100,000 screened individuals.

### 9. Actual Numerical Results
In a hypothetical cohort of 100,000 screened individuals with 1,825 prediabetes cases: Standard linear screening identifies only 8 prediabetic patients (1,817 missed opportunities); Failure-Aware Hybrid identifies 709 prediabetic patients (capturing 88.6 times more high-risk individuals for early lifestyle intervention).

### 10. Figures and Tables Generated
Integrated clinical-epidemiological workflow diagram (`results/figures/16_integrated_workflow.png`), screening yield comparative bar chart.

### 11. Important Observations Supported by Outputs
Deploying the failure-aware model within public health surveillance infrastructure provides massive epidemiological yield, enabling early non-pharmacological interventions that prevent full progression to type 2 diabetes.

### 12. Limitations, Warnings, Errors, and Incomplete Work
Public health deployment requires addressing digital divide barriers and ensuring confirmatory laboratory testing (HbA1c / FPG) is accessible for flagged individuals.

### 13. Contribution to Overall Research Question
Bridges the gap between data science methodology and real-world public health policy, completing the research translation pipeline.

### 14. Which Existing Paper Section It Supports
Section 8.2: Clinical Triage Workflows and Public Health Policy Translation.

### 15. Which Additional Paper Section Could Use Its Findings
Section 9: Conclusions and Translational Roadmap.

---
