"""
Base Classifier Implementations and Hyperparameters
"""
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_base_models(random_state=42):
    """Return the 7 evaluated base classifiers with exact paper hyperparameters."""
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, class_weight='balanced', solver='lbfgs', C=1.0, random_state=random_state
        ),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=10, min_samples_split=20, class_weight='balanced', random_state=random_state
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_split=10, class_weight='balanced',
            n_jobs=-1, random_state=random_state
        ),
        'SGDClassifier_ModifiedHuber': SGDClassifier(
            loss='modified_huber', penalty='l2', alpha=0.0001, max_iter=1000, random_state=random_state
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='mlogloss',
            n_jobs=-1, random_state=random_state
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, num_leaves=31,
            class_weight='balanced', n_jobs=-1, random_state=random_state
        ),
        'CatBoost': CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1, auto_class_weights='Balanced',
            verbose=0, random_state=random_state
        )
    }
    return models
