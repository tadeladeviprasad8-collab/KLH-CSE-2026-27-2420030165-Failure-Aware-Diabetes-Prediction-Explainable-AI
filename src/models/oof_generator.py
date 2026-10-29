"""
Stratified Out-of-Fold (OOF) Prediction Generation
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

def generate_oof_predictions(models, X, y, cv=5, random_state=42):
    """
    Generate out-of-fold cross-validation probability matrices strictly across CV folds.
    Guarantees zero data leakage for meta-learner training.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    n_samples = len(X)
    n_classes = len(np.unique(y))
    
    oof_predictions = {}
    for name in models.keys():
        oof_predictions[name] = np.zeros((n_samples, n_classes))
        
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
        X_val_f = X.iloc[val_idx]
        
        for name, model in models.items():
            clf = clone(model)
            clf.fit(X_train_f, y_train_f)
            oof_predictions[name][val_idx] = clf.predict_proba(X_val_f)
            
    return oof_predictions
