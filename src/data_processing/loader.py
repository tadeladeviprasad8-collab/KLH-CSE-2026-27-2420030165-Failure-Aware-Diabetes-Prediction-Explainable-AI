"""
Dataset Ingestion, Verification, and Partitioning
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import RANDOM_SEED, TEST_SIZE, TARGET_COLUMN

def load_dataset(file_path):
    """Load dataset from CSV with integrity checks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df

def get_train_test_splits(df, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """Perform stratified 80/20 train/test partition."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test
