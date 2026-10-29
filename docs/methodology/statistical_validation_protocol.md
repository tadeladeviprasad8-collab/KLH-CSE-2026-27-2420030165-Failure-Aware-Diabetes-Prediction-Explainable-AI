# Comprehensive Exploratory Data Analysis (EDA) and Statistical Findings

**Project Title:** Failure-Aware Ensemble Learning and Explainable AI for Three-Class Diabetes Status Prediction  
**Primary Dataset:** CDC Behavioral Risk Factor Surveillance System (BRFSS 2015)  
**Scope:** Forensic extraction of descriptive distributions, comorbidity associations, hypothesis test statistics, and inferential models  

---

## 1. Dataset Dimensions, Structure, and Target Distribution

### 1.1 Ingestion and Dimensions
- **Source File:** `diabetes_012_health_indicators_BRFSS2015.csv` (CDC BRFSS 2015 survey cycle).
- **Total Records ($N$):** 253,680 survey respondents.
- **Raw Columns:** 22 total (21 predictor variables + 1 target variable `Diabetes_012`).
- **Data Completeness:** 0 missing values across all 253,680 rows (100% complete). The CDC pre-processes public BRFSS indicator extracts by imputing or discarding non-responses prior to distribution.
- **Data Types:** All columns are encoded as numeric integers/floats representing discrete, ordinal, or binned survey responses.

### 1.2 Target Variable Formulation and Severe Imbalance
The prediction task is framed as a 3-class ordinal classification problem:

| Target Code | Class Name | Population Count ($N$) | Percentage (%) | Clinical Definition & Implication |
|:---:|:---|:---:|:---:|:---|
| **0** | No Diabetes | 213,703 | 84.24% | Normoglycemic respondents; majority negative class |
| **1** | Prediabetes | 4,631 | 1.83% | Impaired fasting glucose / pre-clinical transition state; extreme minority |
| **2** | Diabetes | 35,346 | 13.93% | Diagnosed Type 1 or Type 2 diabetes |
| **Total** | **All Classes** | **253,680** | **100.00%** | **Overall Class Imbalance Ratio: ~46 : 1 : 7.6** |

> [!IMPORTANT]
> The 1.83% prevalence of Prediabetes represents an extreme class imbalance. A naive majority-class classifier predicting 'No Diabetes' for all instances achieves an illusory 84.24% accuracy while achieving exactly 0.00% sensitivity for prediabetes. This clinical failure mode defines the **Accuracy Paradox**.

---

## 2. Duplicate Record and Modal Cluster Analysis

### 2.1 Empirical Duplicate Counts
- **Full Corpus ($N = 253,680$):** 23,899 duplicate rows (9.42% of the raw dataset).
- **Unique Feature Vectors:** 229,781 distinct combinations of survey responses.
- **Locked Test Set ($N = 50,736$):** 6,923 test instances (13.65%) share identical 21-feature vectors with at least one record in the training set.

### 2.2 Methodological Justification for Retaining Duplicates
1. **Discrete Survey Structure:** The BRFSS survey measures 21 discrete integer variables (e.g., binary indicators like `HighBP`, `Smoker`, `PhysActivity` $\in \{0, 1\}$; ordinal variables like `GenHlth` $\in \{1..5\}$; and binned categories like `Age` $\in \{1..13\}$).
2. **Natural Population Modes:** Across 250,000+ national respondents, common demographic-lifestyle profiles naturally repeat thousands of times (e.g., young, non-smoking, physically active individuals with excellent general health and zero chronic disease).
3. **Epidemiological Validity:** Deleting duplicate rows would artificially flatten the empirical probability density function of the U.S. population, over-representing rare edge cases and distorting clinical screening calibration.
4. **Zero Partition Contamination:** The train/test split was performed using stratified random sampling (`seed=42`). While duplicate feature vectors exist across splits due to population modes, no test label or test prediction was ever accessed during base model training or meta-learner fitting.

---

## 3. Univariate Feature Distributions and Comorbidity Prevalences

### 3.1 Continuous and Multi-Level Ordinal Features

| Feature | Mean ± SD | Median (IQR) | Min - Max | Class 0 (No Diab) | Class 1 (Prediab) | Class 2 (Diab) | Clinical Trend |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **BMI** | 28.38 ± 6.61 | 27.0 (24.0 - 31.0) | 12.0 - 98.0 | 27.74 ± 6.23 | 30.72 ± 7.15 | 31.94 ± 7.37 | Monotonic rise with disease status |
| **GenHlth** (1-5) | 2.51 ± 1.07 | 2.0 (2.0 - 3.0) | 1.0 - 5.0 | 2.37 ± 1.02 | 3.01 ± 1.08 | 3.29 ± 1.07 | Self-rated health degrades sharply |
| **MentHlth** (0-30) | 3.18 ± 7.41 | 0.0 (0.0 - 2.0) | 0.0 - 30.0 | 2.96 ± 7.12 | 4.38 ± 8.52 | 4.45 ± 8.65 | Mental health burden increases |
| **PhysHlth** (0-30) | 4.24 ± 8.72 | 0.0 (0.0 - 3.0) | 0.0 - 30.0 | 3.58 ± 7.94 | 6.01 ± 10.05 | 7.95 ± 11.08 | Physical illness days double |
| **Age** (1-13) | 8.03 ± 3.05 | 8.0 (6.0 - 10.0) | 1.0 - 13.0 | 7.79 ± 3.09 | 9.08 ± 2.78 | 9.38 ± 2.52 | Substantially older in Classes 1 & 2 |
| **Education** (1-6) | 5.05 ± 0.99 | 5.0 (4.0 - 6.0) | 1.0 - 6.0 | 5.11 ± 0.96 | 4.92 ± 1.03 | 4.75 ± 1.08 | Modest inverse educational gradient |
| **Income** (1-8) | 6.05 ± 2.07 | 7.0 (5.0 - 8.0) | 1.0 - 8.0 | 6.21 ± 2.01 | 5.35 ± 2.19 | 5.21 ± 2.23 | Household income lower in diabetes |

### 3.2 Binary Clinical Comorbidities and Behavioral Risk Factors

| Binary Feature | Full Population (%) | Class 0: No Diab (%) | Class 1: Prediab (%) | Class 2: Diab (%) | Relative Risk Gradient |
|:---|:---:|:---:|:---:|:---:|:---:|
| **HighBP** | 42.90% | 37.53% | 62.43% | 75.27% | +100.6% relative increase |
| **HighChol** | 42.41% | 37.92% | 62.02% | 67.01% | +76.7% relative increase |
| **CholCheck** (Past 5 yrs) | 96.26% | 95.74% | 97.97% | 99.31% | Near-universal screening |
| **Smoker** (≥100 cigs) | 44.32% | 43.16% | 49.00% | 51.82% | Modest positive gradient |
| **Stroke** | 4.06% | 3.16% | 5.72% | 9.25% | Nearly triple the stroke rate |
| **HeartDiseaseorAttack** | 9.42% | 7.18% | 14.58% | 22.29% | Over triple the cardiac disease |
| **PhysActivity** | 75.65% | 77.91% | 67.65% | 63.05% | Sharp decline in physical activity |
| **Fruits** (≥1/day) | 63.43% | 64.38% | 60.35% | 58.54% | Minor nutritional gradient |
| **Veggies** (≥1/day) | 81.14% | 82.14% | 76.94% | 75.64% | Minor nutritional gradient |
| **HvyAlcoholConsump** | 5.62% | 6.00% | 4.25% | 3.48% | Inverse (healthy survivorship/advice) |
| **AnyHealthcare** | 95.11% | 94.97% | 95.06% | 95.97% | Consistently high coverage |
| **NoDocbcCost** | 8.42% | 7.96% | 10.60% | 10.59% | Higher financial barriers |
| **DiffWalk** (Difficulty stairs) | 16.82% | 13.20% | 27.16% | 37.12% | Functional mobility severely impaired |
| **Sex** (1 = Male) | 44.03% | 43.34% | 43.51% | 48.37% | Modest male preponderance in Class 2 |

---

## 4. Rigorous Statistical Hypothesis Testing (Notebook 04)

### 4.1 Kruskal-Wallis Non-Parametric ANOVA (Continuous/Ordinal Features)
Because continuous and ordinal features exhibit non-normal distributions and ordinal integer binning, Kruskal-Wallis H tests were conducted across the three target classes ($k = 3, df = 2$).

| Feature | Kruskal-Wallis $H$ Statistic | Degrees of Freedom ($df$) | Asymptotic $p$-Value | Null Hypothesis ($H_0$) | Statistical Decision |
|:---|:---:|:---:|:---:|:---|:---:|
| **GenHlth** | **22,819.46** | 2 | $< 1.0 	imes 10^{-300}$ | Medians equal across 3 classes | **Reject $H_0$** ($p < 0.001$) |
| **Age** | **14,572.11** | 2 | $< 1.0 	imes 10^{-300}$ | Medians equal across 3 classes | **Reject $H_0$** ($p < 0.001$) |
| **Income** | **8,912.38** | 2 | $< 1.0 	imes 10^{-300}$ | Medians equal across 3 classes | **Reject $H_0$** ($p < 0.001$) |
| **BMI** | **8,826.29** | 2 | $< 1.0 	imes 10^{-300}$ | Medians equal across 3 classes | **Reject $H_0$** ($p < 0.001$) |
| **PhysHlth** | **7,654.18** | 2 | $< 1.0 	imes 10^{-300}$ | Medians equal across 3 classes | **Reject $H_0$** ($p < 0.001$) |
| **Education** | **2,431.84** | 2 | $< 1.0 	imes 10^{-300}$ | Medians equal across 3 classes | **Reject $H_0$** ($p < 0.001$) |
| **MentHlth** | **698.42** | 2 | $< 1.0 	imes 10^{-150}$ | Medians equal across 3 classes | **Reject $H_0$** ($p < 0.001$) |

### 4.2 Pearson Chi-Square Tests of Independence (Categorical Predictors)
Chi-square tests of independence were performed on $2 	imes 3$ contingency tables across all binary survey indicators.

| Binary Predictor | $\chi^2$ Test Statistic | Degrees of Freedom ($df$) | $p$-Value | Cramér's $V$ | Statistical Decision |
|:---|:---:|:---:|:---:|:---:|:---:|
| **HighBP** | **14,831.22** | 2 | $< 1.0 	imes 10^{-300}$ | 0.2418 | **Reject $H_0$** (Strong Association) |
| **DiffWalk** | **9,412.31** | 2 | $< 1.0 	imes 10^{-300}$ | 0.1926 | **Reject $H_0$** (Moderate Association) |
| **HighChol** | **8,924.54** | 2 | $< 1.0 	imes 10^{-300}$ | 0.1875 | **Reject $H_0$** (Moderate Association) |
| **HeartDiseaseorAttack** | **5,124.72** | 2 | $< 1.0 	imes 10^{-300}$ | 0.1421 | **Reject $H_0$** (Moderate Association) |
| **PhysActivity** | **3,412.83** | 2 | $< 1.0 	imes 10^{-300}$ | 0.1160 | **Reject $H_0$** (Significant Association) |
| **Stroke** | **1,482.09** | 2 | $< 1.0 	imes 10^{-300}$ | 0.0764 | **Reject $H_0$** (Significant Association) |
| **Smoker** | **724.31** | 2 | $< 1.0 	imes 10^{-150}$ | 0.0534 | **Reject $H_0$** (Significant Association) |
| **Sex** | **284.58** | 2 | $< 1.0 	imes 10^{-60}$ | 0.0335 | **Reject $H_0$** (Weak Significant Association) |

### 4.3 Pairwise Post-Hoc Tests (Mann-Whitney U with Bonferroni Correction)
Pairwise Mann-Whitney U tests confirm that Prediabetes (Class 1) is statistically distinct from both No Diabetes (Class 0) and Diabetes (Class 2):
- **BMI:** Class 0 vs Class 1 ($U = 3.61 	imes 10^8, p < 1e-180$); Class 1 vs Class 2 ($U = 7.39 	imes 10^7, p < 1e-45$).
- **GenHlth:** Class 0 vs Class 1 ($U = 3.32 	imes 10^8, p < 1e-220$); Class 1 vs Class 2 ($U = 7.12 	imes 10^7, p < 1e-55$).
- **Age:** Class 0 vs Class 1 ($U = 3.55 	imes 10^8, p < 1e-195$); Class 1 vs Class 2 ($U = 7.84 	imes 10^7, p < 1e-18$).

### 4.4 Top Bivariate Feature Correlations
- **PhysHlth vs GenHlth:** Pearson $r = +0.524$ (Strong positive correlation between poor general health and physical illness days).
- **HighBP vs Age:** Pearson $r = +0.345$ (Age-related hypertension escalation).
- **BMI vs GenHlth:** Pearson $r = +0.301$ (Elevated body mass degrades perceived overall vitality).
- **PhysHlth vs DiffWalk:** Pearson $r = +0.478$ (Physical health days strongly predict mobility limitations).

---

## 5. Methodological and Clinical Interpretation

1. **Prediabetes as a Physiological Continuum:** The statistical analysis reveals that prediabetes does not form an isolated geometric cluster in feature space. Instead, individuals with prediabetes exhibit intermediate values across all major physiological parameters (BMI, HighBP, HighChol, GenHlth), making pure hyperplane separation difficult.
2. **Correlational vs Causal Demarcation:** All reported associations are strictly observational from cross-sectional BRFSS survey responses. They demonstrate statistical predictive utility but do not establish direct biological causality.
3. **Empirical Justification for Feature Interactions:** Because single features exhibit moderate effect sizes (Cramér's V ≤ 0.24), non-linear feature engineering (combining BMI, age, and chronic conditions) is scientifically necessary to amplify classification signal.