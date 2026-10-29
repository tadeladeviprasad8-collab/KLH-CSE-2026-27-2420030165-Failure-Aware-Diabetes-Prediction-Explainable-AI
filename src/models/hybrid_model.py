"""
Failure-Aware Hybrid Stacking Ensemble Architecture
"""
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def construct_failure_aware_meta_features(probs_dict):
    """
    Construct 50 failure-aware meta-features across base models:
    - Group 1: Base probabilities (K x 3)
    - Group 2: Maximum confidence scores (K)
    - Group 3: Shannon entropy (K)
    - Group 4: Decision margin (K)
    - Group 5: Pairwise disagreement indicators (K)
    - Group 6: Modal agreement consensus count (1)
    """
    model_names = list(probs_dict.keys())
    K = len(model_names)
    N = len(probs_dict[model_names[0]])
    
    # 1. Base probabilities
    prob_list = [probs_dict[m] for m in model_names]
    prob_matrix = np.hstack(prob_list) # N x (3K)
    
    # Discrete predictions per model
    preds = np.column_stack([np.argmax(probs_dict[m], axis=1) for m in model_names]) # N x K
    
    # Modal consensus class
    modal_class = mode(preds, axis=1, keepdims=False)[0] # N,
    
    # 2. Maximum confidence
    confidence_list = [np.max(probs_dict[m], axis=1, keepdims=True) for m in model_names]
    conf_matrix = np.hstack(confidence_list) # N x K
    
    # 3. Normalized Shannon entropy
    eps = 1e-15
    entropy_list = [-np.sum(probs_dict[m] * np.log2(probs_dict[m] + eps), axis=1, keepdims=True) for m in model_names]
    entropy_matrix = np.hstack(entropy_list) # N x K
    
    # 4. Decision margin (top1 - top2)
    margin_list = []
    for m in model_names:
        sorted_p = np.sort(probs_dict[m], axis=1)
        margin = (sorted_p[:, -1] - sorted_p[:, -2]).reshape(-1, 1)
        margin_list.append(margin)
    margin_matrix = np.hstack(margin_list) # N x K
    
    # 5. Pairwise disagreement with modal consensus
    disagree_list = [(preds[:, m:m+1] != modal_class.reshape(-1, 1)).astype(float) for m in range(K)]
    disagree_matrix = np.hstack(disagree_list) # N x K
    
    # 6. Modal agreement count
    agreement_count = np.sum(preds == modal_class.reshape(-1, 1), axis=1, keepdims=True).astype(float) # N x 1
    
    meta_features = np.hstack([
        prob_matrix, conf_matrix, entropy_matrix, margin_matrix,
        disagree_matrix, agreement_count
    ])
    return meta_features

class FailureAwareHybridClassifier(BaseEstimator, ClassifierMixin):
    """Failure-Aware Hybrid Classifier wrapping base models and meta-learner."""
    def __init__(self, base_models, C=1.0, random_state=42):
        self.base_models = base_models
        self.C = C
        self.random_state = random_state
        self.meta_learner = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression(
                C=self.C, class_weight='balanced', penalty='l2',
                solver='lbfgs', max_iter=1000, random_state=self.random_state
            ))
        ])
        
    def fit(self, oof_probs_dict, y_train):
        """Fit meta-learner on out-of-fold cross-validation probability matrices."""
        X_meta = construct_failure_aware_meta_features(oof_probs_dict)
        self.meta_learner.fit(X_meta, y_train)
        return self
        
    def predict_proba(self, test_probs_dict):
        """Predict class probabilities using test probability matrices."""
        X_meta_test = construct_failure_aware_meta_features(test_probs_dict)
        return self.meta_learner.predict_proba(X_meta_test)
        
    def predict(self, test_probs_dict):
        """Predict discrete classes using argmax."""
        proba = self.predict_proba(test_probs_dict)
        return np.argmax(proba, axis=1)
