# Comprehensive Project Facts Dossier for ChatGPT Manuscript Expansion

**Project Title:** Failure-Aware Ensemble Learning and Explainable AI for Three-Class Diabetes Status Prediction  
**Purpose:** Self-contained, authoritative scientific knowledge base for expanding manuscript `Diabetes_Paper_Complete_With_All_Figures.docx`  
**Instructions for ChatGPT:** Use the exact factual statements, mathematical formulas, and verified empirical values below. Do not guess, extrapolate, or alter any numbers. All values have been forensically verified from project source files.  

---

## 1. Project Metadata and Core Architecture
- **Title:** Failure-Aware Ensemble Learning and Explainable AI for Three-Class Diabetes Status Prediction
- **Target Classes:** 0 = No Diabetes, 1 = Prediabetes, 2 = Diabetes (3-class classification)
- **Primary Dataset (Dataset 1):** CDC BRFSS 2015 (`diabetes_012_health_indicators_BRFSS2015.csv`). Total rows = 253,680. Missing values = 0.
- **Target Distribution:** Class 0 = 213,703 (84.24%), Class 1 = 4,631 (1.83%), Class 2 = 35,346 (13.93%). Imbalance ~ 46:1.
- **Experimental Split:** Stratified 80/20 train/test split (Random Seed 42).
  - Training set ($N = 202,944$): Class 0 = 170,962; Class 1 = 3,705; Class 2 = 28,277.
  - Locked test set ($N = 50,736$): Class 0 = 42,741; Class 1 = 926; Class 2 = 7,069.
- **Duplicates:** 23,899 duplicate rows in full data (9.42%); 6,923 test rows (13.65%) share feature vectors with train due to discrete survey nature. Retained deliberately to preserve true population modes.
- **Secondary Dataset (Dataset 2):** CDC USDSS / NHIS (`export.csv` / `df2_surveillance.csv`). Aggregate surveillance data, 2000–2024, 13,889 rows. Not merged at patient level; used strictly for population surveillance analytics.

---

## 2. Feature Engineering Pipeline (21 Raw -> 29 Model Features)
21 original features: `HighBP`, `HighChol`, `CholCheck`, `BMI`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `GenHlth`, `MentHlth`, `PhysHlth`, `DiffWalk`, `Sex`, `Age`, `Education`, `Income`.

8 Engineered Features:
1. `BMI_Category`: Ordinal 1 (<18.5), 2 (18.5-24.9), 3 (25-29.9), 4 (≥30).
2. `Age_Group`: Ordinal 1 (Age 1-4, 18-44 yrs), 2 (Age 5-9, 45-64 yrs), 3 (Age 10-13, 65+ yrs).
3. `Health_Burden`: HighBP + HighChol + Stroke + HeartDiseaseorAttack + DiffWalk (Integer 0-5).
4. `Healthy_Lifestyle`: PhysActivity + Fruits + Veggies - Smoker - HvyAlcoholConsump (Integer -2 to +3).
5. `Healthcare_Access`: AnyHealthcare + (1 - NoDocbcCost) (Integer 0-2).
6. `Mental_Physical_Days`: MentHlth + PhysHlth (Integer 0-60).
7. `BMI_Age_Interaction`: BMI × Age (Continuous).
8. `GenHlth_BMI`: GenHlth × BMI (Continuous).
Impact: 29 features improve Prediabetes Recall from 35.42% to 38.88% (+3.46%) and Macro-F1 from 0.4082 to 0.4162 over 21 raw features.

---

## 3. Statistical Testing Results (Notebook 04)
- **Kruskal-Wallis Tests (df = 2, all p < 1e-300):** GenHlth ($H = 22,819.5$), Age ($H = 14,572.1$), Income ($H = 8,912.4$), BMI ($H = 8,826.3$), PhysHlth ($H = 7,654.2$), Education ($H = 2,431.8$), MentHlth ($H = 698.4$).
- **Chi-Square Tests of Independence (df = 2, all p < 1e-60):** HighBP ($\chi^2 = 14,831.2, V = 0.242$), DiffWalk ($\chi^2 = 9,412.3, V = 0.193$), HighChol ($\chi^2 = 8,924.5, V = 0.188$), HeartDisease ($\chi^2 = 5,124.7, V = 0.142$), PhysActivity ($\chi^2 = 3,412.8, V = 0.116$), Stroke ($\chi^2 = 1,482.1, V = 0.076$), Smoker ($\chi^2 = 724.3, V = 0.053$), Sex ($\chi^2 = 284.6, V = 0.034$).

---

## 4. Models and Verified Locked-Test Results ($N = 50,736$)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Balanced Acc | MCC | Log Loss | Brier | Prediab Recall | Prediab F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SGD (Mod Huber) | 0.8014 | 0.4472 | 0.7812 | 0.4711 | 0.3422 | 0.4782 | 0.2841 | 0.0043 (4/926) | 0.0079 |
| XGBoost | 0.7581 | 0.4347 | 0.7511 | 0.4689 | 0.3080 | 0.4285 | 0.3120 | 0.0205 (19/926) | 0.0190 |
| LightGBM | 0.6430 | 0.4234 | 0.6874 | 0.5161 | 0.2999 | 0.5841 | 0.4215 | 0.2538 (235/926) | 0.0519 |
| CatBoost | 0.6694 | 0.4300 | 0.7042 | 0.5128 | 0.3149 | 0.5512 | 0.3984 | 0.1825 (169/926) | 0.0493 |
| Random Forest | 0.7004 | 0.4301 | 0.7221 | 0.4983 | 0.3230 | 0.5124 | 0.3621 | 0.0778 (72/926) | 0.0390 |
| Logistic Regression | 0.6414 | 0.4257 | 0.6865 | 0.5189 | 0.2920 | 0.6120 | 0.4328 | 0.3240 (300/926) | 0.0557 |
| Decision Tree | 0.5813 | 0.3926 | 0.6482 | 0.4902 | 0.2481 | 1.1245 | 0.5120 | 0.3359 (311/926) | 0.0483 |
| Soft Voting | 0.7477 | 0.4476 | 0.7482 | 0.5028 | 0.3533 | 0.4651 | 0.3211 | 0.0421 (39/926) | 0.0395 |
| Standard Stacking | 0.6000 | 0.4180 | 0.6631 | 0.5278 | 0.2862 | 0.6812 | 0.4721 | 0.4060 (376/926) | 0.0540 |
| Failure-Aware Hybrid| 0.6101 | 0.4162 | 0.6705 | 0.5218 | 0.2820 | 0.6654 | 0.4589 | 0.3888 (360/926) | 0.0554 |

---

## 5. Statistical Hypothesis Tests: Hybrid vs. Stacking
- **Paired Bootstrap ($B = 1,000$ resamples, Seed 42):**
  - Accuracy: Delta = +0.0101 (+1.01%), 95% CI: [+0.0064, +0.0138], Two-tailed $p < 0.001$ (**Statistically Significant**).
  - Macro-F1: Delta = -0.0018, 95% CI: [-0.0047, +0.0011], Two-tailed $p = 0.224$ (Not Significant / Parity).
  - Balanced Accuracy: Delta = -0.0060, 95% CI: [-0.0145, +0.0025], Two-tailed $p = 0.168$ (Not Significant / Parity).
  - Prediabetes Recall: Delta = -0.0173, 95% CI: [-0.0421, +0.0076], Two-tailed $p = 0.176$ (Not Significant / Parity).
- **McNemar Discordance Test:**
  - Contingency Table: Both Correct = 29,643; Both Incorrect = 18,982; Hybrid-Only Correct = 1,311; Stacking-Only Correct = 800.
  - Statistic: $\chi^2 = 122.95, df = 1, p = 1.43 \times 10^{-28}$.
  - Interpretation: Confirms highly significant classification discordance favoring the Hybrid in overall correct instances.

---

## 6. Failure-Aware Meta-Feature Formulations (50 Features)
- **7 Base Estimators:** Logistic Regression, Decision Tree, Random Forest, SVM (SGD Modified Huber), XGBoost, LightGBM, CatBoost.
- **21 Base Probabilities:** $p_{m, c}(\mathbf{x}_i) = P_m(Y = c \mid \mathbf{x}_i)$ for $m=1..7, c=0..2$.
- **7 Max Confidence Scores:** $C_m = \max_c p_{m, c}$.
- **7 Normalized Shannon Entropies:** $H_m = -\sum_{c=0}^2 p_{m, c} \log_2(p_{m, c} + 10^{-15})$.
- **7 Decision Margins:** $M_m = p_{m, (1)} - p_{m, (2)}$.
- **7 Pairwise Disagreements:** $D_m = \mathbb{I}(\hat{y}_m \neq y_{\text{mode}})$.
- **1 Modal Agreement Consensus:** $A_{\text{modal}} = \sum_{m=1}^7 \mathbb{I}(\hat{y}_m = y_{\text{mode}})$.
- **Meta-Learner:** `LogisticRegression(class_weight='balanced', C=1.0)` fitted strictly on Stratified 5-Fold OOF training predictions.
- **Ablation Results:** M1 Baseline Stacking (60.00% acc) -> M2 Uncertainty (60.45% acc) -> M3 Disagreement (60.78% acc) -> M4 Full Hybrid (61.01% acc).

---

## 7. Explainability and Failure Analysis (SHAP & LIME)
- **SHAP Prediabetes (Class 1) Top Drivers:** `BMI_Age_Interaction` (0.4217), `GenHlth_BMI` (0.3879), `BMI` (0.2557), `Age` (0.2114), `GenHlth` (0.1982), `HighBP` (0.1745), `Health_Burden` (0.1523).
- **LIME Patient 134330 (False Negative):** True Class 1, Predicted Class 0. High income (+$0.0113$) and exercise (+$0.0084$) masked clinical hypertension (-$0.0292$) and BMI (-$0.0185$), revealing a critical socioeconomic masking vulnerability.
- **Subgroup Fairness:** Prediabetes sensitivity is 23.39% in young adults (<45 yrs) vs 44.19% in seniors (65+ yrs) (Disparate Impact Ratio = 0.529). Sex sensitivity is balanced (40.09% men vs 37.69% women).

---

## 8. Population Surveillance (CDC Dataset 2)
- **Source:** CDC USDSS / NHIS, 2000–2024, 13,889 aggregate records.
- **National Prevalence:** Rose from 4.40% in 2000 to 7.90% in 2024 (+79.5% relative rise, CAGR +2.47%).
- **Age Breakdown (2024):** 18-44 yrs: 2.50%; 45-64 yrs: 11.80%; 65+ yrs: 21.40%.
- **Geographic Disparities:** Southeastern 'Diabetes Belt' states exceed 12.0% (Mississippi 12.8%, WV 12.4%) vs Mountain West <6.5% (Colorado 5.8%, Utah 6.1%).

---

## 9. Critical Discrepancies and Clarifications for Paper Authors
1. **SVM is SGDClassifier:** Always refer to it as 'SVM (SGD Modified Huber)'.
2. **Dual Hybrids:** Research model has 50 meta-features (7 base models); Dashboard model has 27 meta-features (5 base models, Pipeline wrapped).
3. **Stacking vs Hybrid Claim:** Do not claim Hybrid has higher Macro-F1. State: Hybrid achieves superior overall accuracy (61.01% vs 60.00%, p < 0.001) while operating at Macro-F1 parity (0.4162 vs 0.4180, p = 0.224) and reducing false-positive alerts by 973 cases.
4. **Dataset 2 Role:** Strictly for epidemiological surveillance trends and public health context; not an external ML test set.
5. **Duplicates:** 13.65% test feature overlap represents natural survey modes; re-evaluation on unique rows confirms robustness (60.42% vs 61.01% accuracy).