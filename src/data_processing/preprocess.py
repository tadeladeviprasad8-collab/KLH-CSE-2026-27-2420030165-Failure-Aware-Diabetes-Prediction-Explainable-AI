"""
Feature Scaling and Leakage-Free Preprocessing
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def create_scaler():
    """Instantiate standard scaler for numerical normalization."""
    return StandardScaler()

def fit_transform_scaler(scaler, X_train, X_test):
    """Fit scaler strictly on training folds and transform test set out-of-sample."""
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
