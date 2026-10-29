# Research Jupyter Notebooks (01 to 16)

This directory contains the authentic research execution workflow across 16 sequentially numbered Jupyter notebooks:

1. `01_problem_and_data_understanding.ipynb`: Ingestion of BRFSS 2015, schema audit, class distribution.
2. `02_data_cleaning_and_preprocessing.ipynb`: Duplicate analysis, stratified 80/20 train/test split.
3. `03_eda_and_visual_analytics.ipynb`: Univariate, bivariate distributions, comorbidity prevalence.
4. `04_statistical_analysis.ipynb`: Kruskal-Wallis H tests, Pearson Chi-Square tests of independence.
5. `05_feature_engineering.ipynb`: Formulation and synthesis of 8 clinical composite & interaction features.
6. `06_feature_selection.ipynb`: Mutual Information ranking, LASSO L1 shrinkage, RF Gini importance.
7. `07_predictive_analytics_and_baselines.ipynb`: Dummy Classifier, baseline Logistic Regression & Decision Tree.
8. `08_existing_models.ipynb`: 7 base models training under Stratified 5-Fold CV and OOF prediction generation.
9. `09_hyperparameter_optimization.ipynb`: Optuna Bayesian optimization logging.
10. `10_model_evaluation_and_comparison.ipynb`: Multi-metric evaluation on locked test set ($N = 50,736$).
11. `11_failure_analysis.ipynb`: Pairwise error overlap matrix and minority error complementarity.
12. `12_failure_aware_hybrid_model.ipynb`: 50 meta-feature construction, hybrid training, 4-stage ablation.
13. `13_explainable_ai_shap_lime.ipynb`: SHAP TreeExplainer global/class-1 attribution & LIME Patient 134330.
14. `14_advanced_visual_analytics.ipynb`: Subgroup fairness across age cohorts, sex, income, and calibration curves.
15. `15_population_surveillance_dataset2.ipynb`: CDC USDSS / NHIS aggregate temporal trends (2000–2024).
16. `16_integrated_public_health_analysis.ipynb`: Two-tier clinical and epidemiological policy synthesis.
