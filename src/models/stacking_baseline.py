"""
Standard Stacking Classifier Baseline (Equal Balanced Comparison)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class StandardStackingClassifier:
    """Standard Stacking baseline fitted on base model probabilities with balanced class weighting."""
    def __init__(self, C=1.0, random_state=42):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression(
                C=C, class_weight='balanced', penalty='l2',
                solver='lbfgs', max_iter=1000, random_state=random_state
            ))
        ])
        
    def fit(self, oof_probs_dict, y_train):
        prob_matrix = np.hstack(list(oof_probs_dict.values()))
        self.model.fit(prob_matrix, y_train)
        return self
        
    def predict_proba(self, test_probs_dict):
        prob_matrix = np.hstack(list(test_probs_dict.values()))
        return self.model.predict_proba(prob_matrix)
        
    def predict(self, test_probs_dict):
        return np.argmax(self.predict_proba(test_probs_dict), axis=1)
