"""
SHAP Global and Class-Specific Feature Attribution
"""
import shap
import pandas as pd
import numpy as np

def compute_shap_explanations(tree_model, X_sample):
    """Compute SHAP values using TreeExplainer for multi-class tree ensembles."""
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values
