# Modular Research Source Code

This directory contains the production-grade, modular Python codebase powering the research pipeline:

## Package Structure
- **`data_processing/`**: Data loading, data validation, and leak-free scaling transformers.
- **`features/`**: The 8 clinical feature engineering formulas (`engineering.py`) and non-linear Mutual Information ranking (`selection.py`).
- **`models/`**: Base model configurations (`base_models.py`), Stratified 5-Fold OOF generator (`oof_generator.py`), and the Failure-Aware Hybrid architecture (`hybrid_model.py`).
- **`evaluation/`**: Multi-class metric calculation (`metrics.py`), paired bootstrap & McNemar discordance testing (`statistical_tests.py`), and subgroup fairness (`subgroups.py`).
- **`explainability/`**: SHAP multi-class TreeExplainer (`shap_explainer.py`) and LIME tabular explainer (`lime_explainer.py`).
- **`dashboard/`**: Complete interactive Streamlit clinical decision support application (`app.py`).
- **`run_pipeline.py`**: Command-line executable pipeline runner.
