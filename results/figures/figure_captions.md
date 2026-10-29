# Publication Figure Captions: Failure-Aware Ensemble Learning and Explainable AI for Three-Class Diabetes Status Prediction

This document provides publication-ready captions for all 22 figures generated in the `RESEARCH_FIGURES` directory for the research manuscript.

---

### Figure 01 — End-to-End Research Architecture and Multi-Stage Modeling Pipeline
**Section:** §3, Dataset and Methodology  
**Caption:** **Figure 1: Schematic diagram of the end-to-end failure-aware ensemble learning and explainable artificial intelligence (XAI) surveillance architecture.** The framework begins with the raw CDC BRFSS 2015 cohort ($N = 253,680$ survey responses across 21 discrete health indicators) and generates 8 domain-specific engineered features, yielding 29 total predictor features. A strict stratified 80/20 train/test partition locks 50,736 unseen test instances ($0\%$ data leakage). The training set ($N = 202,944$) undergoes 5-fold stratified cross-validation across seven diverse base learning algorithms with cost-sensitive weighting. Out-of-fold (OOF) cross-validation predictions are mapped into a 50-dimensional meta-feature space comprising 21 base probability outputs, 14 uncertainty descriptors (Shannon entropy and prediction margin), 7 confidence measures, 7 pairwise discordance flags, and a modal consensus agreement count. A balanced logistic regression meta-learner fits these failure-aware representations. Final generalization is verified on the locked test set through paired bootstrap testing, McNemar discordance testing, subgroup fairness auditing, and duplicate-aware sensitivity testing, alongside global SHAP and localized LIME explainability. A streamlined 27-feature pipeline powers the live clinical web dashboard.

---

### Figure 02 — Three-Class Diagnostic Target Distribution (CDC BRFSS 2015)
**Section:** §3, Dataset and Methodology  
**Caption:** **Figure 2: Empirical target class distribution in the primary CDC BRFSS 2015 diabetes cohort ($N = 253,680$).** Bars indicate the observed frequency and population percentage for the three target diagnostic categories: Class 0 (No Diabetes: $n = 213,703$, $84.24\%$), Class 1 (Prediabetes: $n = 4,631$, $1.83\%$), and Class 2 (Diagnosed Diabetes: $n = 35,346$, $13.93\%$). The extreme class imbalance—where the clinical prediabetes minority comprises fewer than 1 in 50 patients—demonstrates the severe statistical challenge of early preventive detection, necessitating cost-sensitive reweighting and failure-aware ensemble techniques.

---

### Figure 03 — Model Performance Landscape: Accuracy vs. Macro-F1 Trade-off
**Section:** §4, Results  
**Caption:** **Figure 3: Global model performance landscape showing the trade-off between overall test accuracy and macro-averaged F1 score across 10 evaluated model architectures ($N = 50,736$).** Points represent individual model configurations grouped by algorithmic family. The annotation highlights the "Accuracy Paradox": the linear SGD classifier with Modified Huber loss achieves the highest raw test accuracy ($0.8014$) but fails catastrophically on minority detection (prediabetes recall $= 0.0043$, detecting only 4 of 926 prediabetes cases). Conversely, the proposed Failure-Aware Hybrid ($0.6101$ accuracy, $0.4162$ macro-F1) and Standard Stacking ($0.6000$ accuracy, $0.4180$ macro-F1) trade majority-class accuracy to establish clinically viable sensitivity on prediabetes ($38.88\%$ and $40.60\%$).

---

### Figure 04 — Per-Class Diagnostic Recall Heatmap Across 10 Model Configurations
**Section:** §4, Results  
**Caption:** **Figure 4: Per-class recall (sensitivity) heatmap evaluated on the locked test set ($N = 50,736$) across all 10 model architectures.** Rows represent models; columns represent target diagnostic classes: No Diabetes (Class 0, $n = 42,741$), Prediabetes (Class 1, $n = 926$), and Diagnosed Diabetes (Class 2, $n = 7,069$). Cells display exact empirical recall values colored along a normalized blue gradient. The critical minority class (Prediabetes) is highlighted by a red bounding box, demonstrating how standalone gradient-boosted trees and SGD models collapse on prediabetes detection ($0.43\%$ to $2.05\%$), whereas stacking and hybrid meta-architectures elevate detection to $38.88\%$–$40.60\%$.

---

### Figure 05 — Four Multi-Class Confusion Matrices (Baseline vs. Ensembles)
**Section:** §4, Results  
**Caption:** **Figure 5: Normalized multi-class confusion matrices for four key modeling paradigms on the locked test set ($N = 50,736$).** Panels display: (Top-Left) SGD Classifier with Modified Huber Loss; (Top-Right) Soft Voting Ensemble; (Bottom-Left) Standard Stacking (Corrected); and (Bottom-Right) Failure-Aware Production Hybrid (Corrected). Rows correspond to true ground truth; columns correspond to model predictions. Cell text annotates raw test case counts and class-normalized conditional recall percentages. The visualization explicitly highlights how the Hybrid and Stacking models effectively redirect predictions toward the minority prediabetes diagonal without destabilizing overall classification integrity.

---

### Figure 06 — SHAP Beeswarm Summary Plot for Prediabetes (Class 1)
**Section:** §7, Explainability  
**Caption:** **Figure 6: SHAP beeswarm summary plot of the top 10 most influential predictors for the Prediabetes class (Class 1).** Computed via `shap.TreeExplainer` on the primary XGBoost model across a representative sample of 2,000 test observations ($random\_state = 42$). Each dot represents a single patient; horizontal position indicates SHAP value (impact on model log-odds for prediabetes), and point color reflects normalized feature value (blue = low, red = high). Domain interaction terms `BMI_Age_Interaction` (mean $|SHAP| = 0.4217$) and `GenHlth_BMI` ($0.3879$) dominate individual prediction influence, followed by raw `BMI` ($0.2557$), `Income` ($0.2191$), and `Mental_Physical_Days` ($0.2058$).

---

### Figure 07 — Global vs. Prediabetes-Specific Mean Absolute SHAP Feature Importance
**Section:** §7, Explainability  
**Caption:** **Figure 7: Comparative ranking of feature attribution magnitudes.** (Panel A) Global multiclass mean absolute SHAP value averaged across all three diagnostic classes ($N = 2,000$). (Panel B) Mean absolute SHAP value specifically for the Prediabetes class (Class 1). Bars annotate exact mean $|SHAP|$ scores. While physiological indicators (`GenHlth_BMI`, `Health_Burden`, `HighBP`) lead global attribution, composite interaction terms (`BMI_Age_Interaction`) and socioeconomic indicators (`Income`, `Education`) play disproportionately influential roles in prediabetes risk differentiation.

---

### Figure 08 — Paired Bootstrap Distribution of Macro-F1 Difference (1,000 Replicates)
**Section:** §5, Statistical Analysis  
**Caption:** **Figure 8: Empirical paired bootstrap distribution of the difference in Macro-F1 score between the Failure-Aware Hybrid and Standard Stacking.** Evaluated across $B = 1,000$ paired resamples of the locked test set ($N = 50,736$, seed 42). The histogram with kernel density estimation displays the observed difference $\Delta = -0.0018$ (red solid line), the null hypothesis $\Delta = 0$ (slate dashed line), and the shaded 95% bootstrap confidence interval [$-0.0040$, $+0.0001$] (yellow area). Because the confidence interval encompasses zero, the two-tailed bootstrap test ($p = 0.2240$, one-tailed $p = 0.0760$) confirms that the models perform at statistical parity in harmonic macro-F1.

---

### Figure 09 — McNemar Paired Classification Discordance Matrix
**Section:** §5, Statistical Analysis  
**Caption:** **Figure 9: McNemar paired classification discordance on the locked test set ($N = 50,736$).** Bars depict the frequency of four mutual prediction outcomes: both models correct ($n = 29,643$, $58.43\%$), Hybrid only correct ($b = 1,311$, $2.58\%$), Stacking only correct ($c = 800$, $1.58\%$), and both models incorrect ($n = 18,982$, $37.41\%$). With continuity correction, the McNemar test statistic is $\chi^2 = 122.95$ ($df = 1$, $p = 1.43 \times 10^{-28}$), demonstrating that the Hybrid correctly classifies significantly more test instances overall than Standard Stacking ($p < 0.0001$).

---

### Figure 10 — Meta-Feature Ablation Study: Macro-F1 vs. Prediabetes Recall
**Section:** §8, Ablation and Duplicate Analysis  
**Caption:** **Figure 10: Meta-feature ablation study showing the trade-off between Macro-F1 and Prediabetes recall across five meta-feature configurations.** Bars depict Macro-F1 (blue) and Prediabetes recall (amber) on the locked test set. Configurations evaluated: (1) Base probabilities only (21 features, Stacking baseline: Macro-F1 $= 0.4199$, Recall $= 0.3402$); (2) Probabilities + entropy (42 features: Macro-F1 $= 0.4230$, Recall $= 0.2754$); (3) Probabilities + disagreement/agreement (28 features: Macro-F1 $= 0.4197$, Recall $= 0.3564$); (4) Probabilities + entropy + disagreement/agreement (35 features: Macro-F1 $= 0.4241$, Recall $= 0.2959$); and (5) Full 50-feature proposed Hybrid (Macro-F1 $= 0.4162$, Recall $= 0.3888$). Incorporating failure-aware disagreement descriptors increases prediabetes recall by $+4.86\%$ over the probability baseline.

---

### Figure 11 — Prediabetes Detection Sensitivity Across Demographic Age Subgroups
**Section:** §8, Subgroup Analysis  
**Caption:** **Figure 11: Prediabetes recall across demographic age cohorts under single-model gradient boosting (XGBoost).** Bars represent recall (sensitivity) for: Young Adults aged 18–39 ($n = 121$ true prediabetes cases), Middle-Aged Adults aged 40–64 ($n = 498$ cases), and Seniors aged 65+ ($n = 307$ cases). The callout highlights a critical clinical disparity: standard machine learning models achieved $0.0000$ recall ($0/121$ detected cases) among young adults, failing completely to identify early-onset prediabetes in individuals with lower cumulative chronic comorbidity burdens.

---

### Figure 12 — Duplicate-Aware Sensitivity Analysis (Full vs. Clean Test Set)
**Section:** §6, Duplicate Analysis  
**Caption:** **Figure 12: Duplicate-aware sensitivity evaluation comparing Standard Stacking and Failure-Aware Hybrid under natural survey redundancy.** (Panel A) Overall accuracy evaluated on the full locked test set ($N = 50,736$) versus the clean test set with all 6,923 training-matching feature vectors removed ($N = 43,813$). (Panel B) Prediabetes recall evaluated across the same cohorts. When evaluating strictly on unique unseen feature vectors, accuracy shifts from $0.6000 \rightarrow 0.5636$ (Stacking) and $0.6101 \rightarrow 0.5729$ (Hybrid), while prediabetes recall remains highly robust ($0.4060 \rightarrow 0.4122$ and $0.3888 \rightarrow 0.3972$). This confirms that minority sensitivity is an authentic algorithmic capability rather than a memorization artifact of common demographic survey modes.

---

### Figure 13 — Local Diagnostic Attribution for a False Negative Failure Case (Patient 134330)
**Section:** §7, Explainability  
**Caption:** **Figure 13: Local LIME explanation for a representative false-negative diagnostic triage failure (Patient Index 134330).** Patient ground truth is Diagnosed Diabetes (Class 2), but the model predicted No Diabetes (Class 0) with calibrated probabilities $P_0 = 0.6760$, $P_1 = 0.0115$, and $P_2 = 0.3125$. Horizontal bars display local feature weights contributing toward Class 0. Despite substantial physiological risk factors pushing strongly away from healthy status (`Health_Burden > 2.00`: $-0.0292$, `HighBP = 1`: $-0.0212$), high socioeconomic affluence (`Income = 8`: $+0.0113$) counterweighted the prediction, illuminating how social determinants of health can introduce confounding signals in standalone classifiers.

---

### Figure 14 — Feature Contribution Disparity: True Positives vs. False Negatives (Class 2)
**Section:** §7, Explainability  
**Caption:** **Figure 14: Systematic SHAP feature attribution disparity between correctly classified diabetic patients (True Positives, $n = 300$) and missed diabetic patients (False Negatives, $n = 300$).** Bars represent mean signed SHAP values for physiological predictors (`Health_Burden`, `GenHlth_BMI`, `BMI_Age_Interaction`, `HighBP`) and socioeconomic indicators (`Income`, `Education`). In true positive cases, physiological indicators generate strong positive diagnostic evidence ($+0.092$ to $+0.204$). In false negative cases, physiological indicators fail to trigger positive attribution (mean SHAP $-0.019$ to $-0.100$) while socioeconomic factors pull negatively, revealing the primary physiological failure mode in tree models.

---

### Figure 15 — Longitudinal Population Surveillance: Diagnosed Diabetes Trajectories (2000–2024)
**Section:** §9, Population Surveillance  
**Caption:** **Figure 15: Longitudinal nationwide diabetes prevalence trends derived from the CDC Diabetes Surveillance System (USDSS / NHIS, 2000–2024).** The bold black curve tracks national all-ages crude diagnosed diabetes prevalence across 25 consecutive annual observation periods, rising from $4.4\%$ in 2000 to $7.9\%$ in 2024 (a $+79.5\%$ relative increase). Colored dashed curves depict age-stratified trajectories: 18–44 years ($1.9\% \rightarrow 3.3\%$), 45–64 years ($8.3\% \rightarrow 13.3\%$), 65–74 years ($15.8\% \rightarrow 22.2\%$), and 75+ years ($13.2\% \rightarrow 21.8\%$). These population benchmarks provide epidemiological context in the interactive dashboard, demonstrating rising national burden.

---

### Figure 16 — Precision-Recall Curves for Prediabetes Detection Across 10 Model Configurations
**Section:** §4, Model Evaluation  
**Caption:** **Figure 16: Precision-recall curves evaluated specifically on the Prediabetes class (Class 1) across all 10 evaluated model configurations ($N = 50,736$).** Curves plot positive predictive value (precision) against sensitivity (recall) for each model, with corresponding Average Precision (AP) scores annotated in the legend. The horizontal dashed line denotes the empirical random-classifier prevalence baseline ($P = 0.01825$). Standard Stacking ($AP = 0.0381$) and the Failure-Aware Hybrid ($AP = 0.0374$) dominate standalone models (XGBoost $AP = 0.0190$, Soft Voting $AP = 0.0264$), maintaining viable precision across extended recall thresholds.

---

### Figure 17 — Probability Reliability Diagrams (Calibration Curves) for Diabetes Risk
**Section:** §8, Calibration  
**Caption:** **Figure 17: Empirical probability calibration curves (reliability diagrams) for Diagnosed Diabetes (Class 2).** Computed using 10 uniform probability bins on the locked test set ($N = 50,736$). Models evaluated include XGBoost (teal), Random Forest (green), SVM Modified Huber (red), Failure-Aware Hybrid (emerald), and Logistic Regression (slate). The dotted diagonal represents perfect calibration. The Failure-Aware Hybrid and XGBoost demonstrate tight alignment with the empirical diagonal across the $0.10$–$0.70$ probability interval, supporting their use for risk-stratified clinical triage.

---

### Figure 18 — Probabilistic Risk Calibration: Multi-Class Log Loss and Brier Score Comparison
**Section:** §4, Model Evaluation  
**Caption:** **Figure 18: Probabilistic scoring metric comparison across 10 model architectures.** (Panel A) Multi-class cross-entropy log loss on the locked test set. (Panel B) Multi-class quadratic Brier score. Horizontal bars rank models from lowest (best) to highest error. XGBoost achieves lowest multi-class log loss ($0.5492$) and Brier score ($0.3221$), while the Failure-Aware Hybrid ($0.8731$ log loss, $0.4930$ Brier score) maintains calibrated probabilities while explicitly prioritizing minority-class sensitivity.

---

### Figure 19 — Controlled Evaluation of Domain Feature Engineering (21 Raw vs. 29 Engineered Features)
**Section:** §3, Feature Engineering  
**Caption:** **Figure 19: Controlled empirical ablation of domain-specific feature engineering.** Evaluates baseline Logistic Regression trained strictly on 21 raw BRFSS survey indicators versus the full 29-feature configuration (+8 domain engineered interactions) on the locked test set ($N = 50,736$). Performance remains remarkably stable across Accuracy ($0.6457 \rightarrow 0.6414$), Macro-F1 ($0.4269 \rightarrow 0.4257$), and Prediabetes Recall ($0.3261 \rightarrow 0.3240$), demonstrating that domain feature engineering preserves predictive fidelity while supplying crucial non-linear interaction terms (`BMI_Age_Interaction`, `GenHlth_BMI`) for downstream tree explainability and meta-learning.

---

### Figure 20 — Deployed Clinical Surveillance Dashboard (Interactive Streamlit Interface)
**Section:** §9, Dashboard and Deployment  
**Caption:** **Figure 20: User interface of the deployed interactive clinical public health surveillance dashboard (`dashboard/app.py`).** Captured live from the running Streamlit web application (`http://localhost:8501`). The screenshot depicts the **Individual Prediction** module: public health officers and clinicians enter 21 standardized BRFSS patient indicators, which are dynamically transformed into 29 engineered features, processed through 5 base models, mapped into 27 meta-features, and scored by the calibrated failure-aware meta-learner. The interface displays instantaneous calibrated class probabilities, model disagreement scores, and triage action levels (Normal, Borderline, Urgent).

---

### Figure 21 — Prediabetes Detection Sensitivity Ranking Across All 10 Model Architectures
**Section:** §4, Model Evaluation  
**Caption:** **Figure 21: Ranked horizontal dot/lollipop chart of all 10 evaluated model configurations ordered by Prediabetes recall (sensitivity on Class 1, $N = 926$).** Dots indicate exact empirical recall and absolute patient detection counts. Standard Stacking ($0.4060$, $376/926$ cases) and Failure-Aware Hybrid ($0.3888$, $360/926$ cases) establish a dramatic $94\times$ sensitivity increase over standard linear SGD ($0.0043$, $4/926$ cases) and a $19\times$ increase over standalone XGBoost ($0.0205$, $19/926$ cases), illustrating the primary clinical contribution of the ensemble framework.

---

### Figure 22 — Dual Hybrid Architecture Comparison: Research (50 Features) vs. Dashboard (27 Features)
**Section:** §3, Model Variants  
**Caption:** **Figure 22: Architectural comparison between the 50-meta-feature Research Hybrid and the 27-meta-feature Production Dashboard Hybrid.** Bars compare Overall Accuracy, Balanced Accuracy, Macro-F1, and Prediabetes Recall on the locked test set ($N = 50,736$). The research model utilizes seven base models generating 50 meta-features for maximum theoretical expressiveness. The production dashboard model prunes slow iterative estimators (SVM SGD and Decision Tree) to five base models generating 27 meta-features, achieving identical empirical accuracy ($0.6101$) and prediabetes recall ($0.3888$) while delivering sub-50ms web inference.

