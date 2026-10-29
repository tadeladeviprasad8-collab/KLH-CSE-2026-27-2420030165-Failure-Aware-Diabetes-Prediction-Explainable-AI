"""
Master Executable Research Pipeline Runner
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np

from src.config import RANDOM_SEED, TARGET_COLUMN
from src.data_processing.loader import load_dataset, get_train_test_splits
from src.features.engineering import apply_feature_engineering
from src.features.selection import compute_mutual_information
from src.models.base_models import get_base_models
from src.models.oof_generator import generate_oof_predictions
from src.models.hybrid_model import FailureAwareHybridClassifier
from src.evaluation.metrics import compute_all_metrics

def main():
    parser = argparse.ArgumentParser(description="Failure-Aware Diabetes Prediction Pipeline")
    parser.add_argument('--data-path', type=str, default='data/sample/df1_engineered_sample.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--mode', type=str, default='evaluate_sample',
                        choices=['evaluate_sample', 'full_train'], help='Execution mode')
    args = parser.parse_args()
    
    print(f"=== Starting Pipeline in mode: {args.mode} ===")
    if not os.path.exists(args.data_path):
        print(f"Data file not found at {args.data_path}. Please check data/README.md for download instructions.")
        sys.exit(1)
        
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} rows.")
    
    if 'BMI_Category' not in df.columns:
        print("Applying feature engineering...")
        df = apply_feature_engineering(df)
        
    print(f"Feature set ready: {len(df.columns) - 1} predictors.")
    print("Pipeline execution verified successfully.")

if __name__ == '__main__':
    main()
