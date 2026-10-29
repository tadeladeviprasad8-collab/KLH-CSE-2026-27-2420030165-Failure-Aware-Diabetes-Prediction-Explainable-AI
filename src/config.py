"""
Global Configuration & Reproducibility Parameters
"""
import os

RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET_COLUMN = 'Diabetes_012'
CLASS_NAMES = ['No Diabetes', 'Prediabetes', 'Diabetes']
CLASS_MAPPING = {0: 'No Diabetes', 1: 'Prediabetes', 2: 'Diabetes'}

RAW_FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

ENGINEERED_FEATURES = [
    'BMI_Category', 'Age_Group', 'Health_Burden', 'Healthy_Lifestyle',
    'Healthcare_Access', 'Mental_Physical_Days', 'BMI_Age_Interaction', 'GenHlth_BMI'
]

FINAL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

BASE_MODEL_NAMES = [
    'LogisticRegression', 'DecisionTree', 'RandomForest',
    'SGDClassifier_ModifiedHuber', 'XGBoost', 'LightGBM', 'CatBoost'
]
