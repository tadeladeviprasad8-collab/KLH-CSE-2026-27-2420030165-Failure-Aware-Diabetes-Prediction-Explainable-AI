"""
Feature Selection and Mutual Information Ranking
"""
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

def compute_mutual_information(X, y, random_state=42):
    """Compute non-linear Mutual Information classification ranking."""
    mi_scores = mutual_info_classif(X, y, random_state=random_state, n_neighbors=3)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    return mi_series
