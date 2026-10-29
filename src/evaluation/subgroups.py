"""
Responsible AI and Subgroup Fairness Evaluation
"""
import pandas as pd
from sklearn.metrics import recall_score

def evaluate_subgroup_fairness(df, y_true, y_pred, subgroup_col):
    """Compute prediabetes recall and disparate impact across demographic slices."""
    results = {}
    groups = df[subgroup_col].unique()
    for g in groups:
        mask = (df[subgroup_col] == g)
        sub_true = y_true[mask]
        sub_pred = y_pred[mask]
        prediab_rec = recall_score(sub_true == 1, sub_pred == 1, zero_division=0)
        results[g] = {
            'sample_size': int(mask.sum()),
            'prediabetes_recall': prediab_rec
        }
    return pd.DataFrame(results).T
