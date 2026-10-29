"""
LIME Local Case Explanations and False Negative Auditing
"""
from lime.lime_tabular import LimeTabularExplainer
from src.config import CLASS_NAMES

def get_lime_explainer(X_train, feature_names):
    """Initialize LIME tabular explainer."""
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        mode='classification',
        random_state=42
    )

def explain_patient_instance(explainer, predict_fn, patient_vector, num_features=10):
    """Generate local linear attribution weights for an individual patient instance."""
    exp = explainer.explain_instance(
        data_row=patient_vector,
        predict_fn=predict_fn,
        num_features=num_features
    )
    return exp
