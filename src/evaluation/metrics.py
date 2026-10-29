"""
Multi-Class Clinical Metrics Computation
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, confusion_matrix, log_loss, brier_score_loss
)

def compute_all_metrics(y_true, y_pred, y_prob=None):
    """Compute complete multi-class metrics suite."""
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    results = {
        'Accuracy': acc,
        'Balanced_Accuracy': b_acc,
        'Macro_F1': macro_f1,
        'Weighted_F1': weighted_f1,
        'MCC': mcc,
        'Class0_Precision': prec_per_class[0],
        'Class0_Recall': rec_per_class[0],
        'Class0_F1': f1_per_class[0],
        'Prediabetes_Precision': prec_per_class[1],
        'Prediabetes_Recall': rec_per_class[1],
        'Prediabetes_F1': f1_per_class[1],
        'Diabetes_Precision': prec_per_class[2],
        'Diabetes_Recall': rec_per_class[2],
        'Diabetes_F1': f1_per_class[2]
    }
    
    if y_prob is not None:
        try:
            results['Log_Loss'] = log_loss(y_true, y_prob)
            y_onehot = pd.get_dummies(y_true).values
            results['Brier_Score'] = np.mean(np.sum((y_prob - y_onehot)**2, axis=1))
        except Exception:
            pass
            
    return results
