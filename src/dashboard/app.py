import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="AI-Driven Public Health Surveillance",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# PATH SETUP
# ==============================================================================
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DIR_PATH, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_utils import ENGINEERED_FEATURES, BINARY_FEATURES, ORDINAL_FEATURES, CONTINUOUS_FEATURES, ALL_ORIGINAL_FEATURES, engineer_features

MODELS_PREPROC = os.path.join(PROJECT_ROOT, "models", "preprocessing")
MODELS_BASE = os.path.join(PROJECT_ROOT, "models", "base_models")
MODELS_HYBRID = os.path.join(PROJECT_ROOT, "models", "hybrid")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_FIGURES = os.path.join(PROJECT_ROOT, "results", "figures")
RESULTS_METRICS = os.path.join(PROJECT_ROOT, "results", "metrics")
RESULTS_TABLES = os.path.join(PROJECT_ROOT, "results", "tables")
RESULTS_EXPLANATIONS = os.path.join(PROJECT_ROOT, "results", "explanations")

# ==============================================================================
# CSS & STYLING (Premium Cinematic Healthcare Palette)
# ==============================================================================
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #0b1324; /* Deep navy / charcoal */
        color: #e2e8f0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #f8fafc;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Text */
    p, span, label {
        color: #cbd5e1;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #060b14;
        border-right: 1px solid #1e293b;
    }
    
    [data-testid="stSidebar"] h1 {
        color: #38bdf8 !important; /* Soft medical blue */
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] div[data-testid="stMarkdownContainer"] > p {
        color: #38bdf8 !important;
        font-size: 2rem !important;
        font-weight: 700;
    }

    /* Interactive Cards */
    .custom-card {
        background-color: #162032;
        border-left: 4px solid #14b8a6; /* Teal */
        padding: 20px;
        border-radius: 6px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.4);
        background-color: #1e293b;
    }
    .custom-card h4 {
        color: #f8fafc;
        margin-top: 0;
    }
    
    /* Prediction Cards */
    .prediction-box-0 { border-left: 4px solid #10b981; background: #064e3b; padding: 20px; border-radius: 8px; margin-top: 20px; } /* Green */
    .prediction-box-1 { border-left: 4px solid #f59e0b; background: #78350f; padding: 20px; border-radius: 8px; margin-top: 20px; } /* Orange */
    .prediction-box-2 { border-left: 4px solid #ef4444; background: #7f1d1d; padding: 20px; border-radius: 8px; margin-top: 20px; } /* Red */
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #38bdf8, #0ea5e9);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4);
        border: none;
        color: white;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# ML PIPELINE (EXACTLY AS PRODUCTION)
# ==============================================================================
@st.cache_resource
def load_pipeline():
    """Load all necessary ML artifacts."""
    try:
        preprocessor = joblib.load(os.path.join(MODELS_PREPROC, "preprocessing_pipeline.joblib"))
        selected_features = joblib.load(os.path.join(MODELS_PREPROC, "selected_features.joblib"))
        
        base_models = {
            "SVM": joblib.load(os.path.join(MODELS_BASE, "svm.joblib")),
            "XGBoost": joblib.load(os.path.join(MODELS_BASE, "xgboost.joblib")),
            "LightGBM": joblib.load(os.path.join(MODELS_BASE, "lightgbm.joblib")),
            "CatBoost": joblib.load(os.path.join(MODELS_BASE, "catboost.joblib")),
            "RandomForest": joblib.load(os.path.join(MODELS_BASE, "randomforest.joblib"))
        }
        
        meta_model = joblib.load(os.path.join(MODELS_HYBRID, "failure_aware_hybrid.joblib"))
        
        # Load XGBoost Explainer
        explainer = shap.TreeExplainer(base_models["XGBoost"])
        
        return preprocessor, selected_features, base_models, meta_model, explainer
    except Exception as e:
        return None, None, None, None, None

def generate_meta_features(base_preds, base_models, n_classes=3):
    """Generate meta-features exactly as in the hybrid model."""
    from scipy.stats import entropy
    meta = pd.DataFrame()
    
    for name in base_models.keys():
        probs = base_preds[name]
        for c in range(n_classes):
            meta[f"{name}_prob_{c}"] = [probs[c]]
            
    for name in base_models.keys():
        probs = np.array([base_preds[name]])
        meta[f"{name}_entropy"] = [entropy(probs[0] + 1e-12)]
        sorted_probs = np.sort(probs, axis=1)
        meta[f"{name}_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
        
    preds_only = np.array([[np.argmax(base_preds[name]) for name in base_models.keys()]])
    meta["Disagreement_Score"] = [len(np.unique(preds_only[0]))]
    vals, counts = np.unique(preds_only[0], return_counts=True)
    meta["Max_Agreement"] = [counts.max()]
    
    return meta

@st.cache_data
def load_surveillance_data():
    path = os.path.join(DATA_PROCESSED, "df2_surveillance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ==============================================================================
# NAVIGATION
# ==============================================================================
st.sidebar.markdown("## Navigation")
pages = [
    "Home", "Research Overview", "Individual Prediction", 
    "Population Surveillance", "Model Comparison", 
    "Explainable AI", "Failure Analysis", "Public Health Insights", "About"
]
page = st.sidebar.radio("Navigation Menu", pages, label_visibility="collapsed")

# ==============================================================================
# PAGES
# ==============================================================================

if page == "Home":
    st.markdown("""
    <div style='text-align: center; padding: 4rem 0;'>
        <h1 style='font-size: 3.5rem; background: -webkit-linear-gradient(#38bdf8, #14b8a6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            AI-DRIVEN PUBLIC HEALTH SURVEILLANCE
        </h1>
        <h3 style='color: #94a3b8; font-weight: 400; margin-top: 1rem;'>
            Data Analytics, Explainable AI & Failure-Aware Diabetes Prediction
        </h3>
        <p style='color: #64748b; font-size: 1.1rem; max-width: 700px; margin: 2rem auto;'>
            An intelligent analytical framework designed for high-precision individual-level classification 
            and population-level public health surveillance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(90deg, rgba(14,165,233,0.1) 0%, rgba(20,184,166,0.1) 100%); 
                    border-radius: 12px; padding: 30px; text-align: center; border: 1px solid rgba(14,165,233,0.2);">
            <h4 style="margin-top:0;">Systems Online</h4>
            <p style="margin-bottom:0;">Core Machine Learning Pipeline Loaded</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "Research Overview":
    st.title("Research Overview")
    st.markdown("A multi-faceted approach to public health analytics.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="custom-card">
            <h4>1. Individual-Level Prediction</h4>
            <p>Predicting individual health outcomes using a diverse ensemble of machine learning algorithms trained on BRFSS indicators.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="custom-card">
            <h4>2. Population-Level Surveillance</h4>
            <p>Analyzing macroscopic trends over decades to provide context to individual-level predictions and identify demographic vulnerabilities.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="custom-card" style="border-left-color: #38bdf8;">
            <h4>3. Failure-Aware Hybrid Modeling</h4>
            <p>A novel meta-learning architecture that calculates base-model disagreement, confidence, and entropy to adaptively detect minority classes like Prediabetes.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="custom-card" style="border-left-color: #f59e0b;">
            <h4>4. Explainable AI (XAI)</h4>
            <p>Opening the 'black box' of predictive models using SHAP and LIME to generate transparent, feature-level clinical explanations.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "Individual Prediction":
    st.title("Individual Prediction")
    st.markdown("Execute the production Failure-Aware Hybrid pipeline.")
    
    preprocessor, selected_features, base_models, meta_model, explainer = load_pipeline()
    if preprocessor is None:
        st.error("Model artifacts missing. Cannot run prediction.")
    else:
        with st.form("prediction_form"):
            st.markdown("### Patient / Indicator Profile")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown("**Demographics**")
                age = st.selectbox("Age Category (1=18-24 to 13=80+)", range(1, 14), index=7)
                sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
                bmi = st.number_input("BMI", min_value=10.0, max_value=99.0, value=25.0, step=1.0)
                income = st.selectbox("Income Category (1 to 8)", range(1, 9), index=5)
                education = st.selectbox("Education Level (1 to 6)", range(1, 7), index=4)
                
            with c2:
                st.markdown("**Comorbidities & Checks**")
                gen_hlth = st.selectbox("General Health (1=Excellent, 5=Poor)", [1,2,3,4,5], index=2)
                high_bp = st.checkbox("High Blood Pressure")
                high_chol = st.checkbox("High Cholesterol")
                chol_check = st.checkbox("Cholesterol Check (Past 5 Yrs)", value=True)
                stroke = st.checkbox("Stroke History")
                heart_disease = st.checkbox("Heart Disease / Attack")
                diff_walk = st.checkbox("Difficulty Walking")
                
            with c3:
                st.markdown("**Lifestyle & Mental Health**")
                phys_activity = st.checkbox("Physical Activity (Past 30 Days)", value=True)
                fruits = st.checkbox("Consume Fruits Daily", value=True)
                veggies = st.checkbox("Consume Veggies Daily", value=True)
                smoker = st.checkbox("Smoker (≥100 cigarettes lifetime)")
                hvy_alcohol = st.checkbox("Heavy Alcohol Consumption")
                any_healthcare = st.checkbox("Has Healthcare Coverage", value=True)
                no_doc_cost = st.checkbox("Could Not Afford Doctor (Past Yr)")
                ment_hlth = st.slider("Poor Mental Health Days (Past 30)", 0, 30, 0)
                phys_hlth = st.slider("Poor Physical Health Days (Past 30)", 0, 30, 0)
                
            submitted = st.form_submit_button("Execute Hybrid Pipeline")
            
        if submitted:
            with st.spinner("Processing through ML pipeline..."):
                input_dict = {
                    "HighBP": int(high_bp), "HighChol": int(high_chol), "CholCheck": int(chol_check),
                    "BMI": int(bmi), "Smoker": int(smoker), "Stroke": int(stroke),
                    "HeartDiseaseorAttack": int(heart_disease), "PhysActivity": int(phys_activity),
                    "Fruits": int(fruits), "Veggies": int(veggies), "HvyAlcoholConsump": int(hvy_alcohol),
                    "AnyHealthcare": int(any_healthcare), "NoDocbcCost": int(no_doc_cost),
                    "GenHlth": gen_hlth, "MentHlth": ment_hlth, "PhysHlth": phys_hlth,
                    "DiffWalk": int(diff_walk), "Sex": sex, "Age": age, "Education": education, "Income": income
                }
                
                input_df = pd.DataFrame([input_dict])
                eng_df = engineer_features(input_df)
                X_pp = preprocessor.transform(eng_df[selected_features])
                
                base_preds = {}
                for name, model in base_models.items():
                    if name in ["LogisticRegression", "SVM"]:
                        base_preds[name] = model.predict_proba(X_pp)[0]
                    else:
                        base_preds[name] = model.predict_proba(eng_df[selected_features])[0]
                        
                meta_features = generate_meta_features(base_preds, base_models)
                hybrid_prob = meta_model.predict_proba(meta_features)[0]
                hybrid_pred = meta_model.predict(meta_features)[0]
                
                status_map = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}
                
                st.markdown(f"""
                <div class="prediction-box-{hybrid_pred}">
                    <h2 style='margin:0; color:white;'>Predicted Status: {status_map[hybrid_pred]}</h2>
                    <p style='margin:0; color:#cbd5e1; opacity:0.8;'>Failure-Aware Hybrid Output</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Probabilities")
                    df_probs = pd.DataFrame({
                        "Class": ["No Diabetes", "Prediabetes", "Diabetes"],
                        "Probability (%)": [f"{p*100:.2f}%" for p in hybrid_prob]
                    })
                    st.dataframe(df_probs, use_container_width=True)
                with col2:
                    st.markdown("### Meta-Analysis")
                    st.markdown(f"- **Model Disagreement Score:** {meta_features['Disagreement_Score'].values[0]}/5")
                    st.markdown(f"- **Max Agreement:** {meta_features['Max_Agreement'].values[0]} models")
                    
                st.markdown("### Base Model Output")
                b_probs = {name: status_map[np.argmax(p)] for name, p in base_preds.items()}
                st.json(b_probs)

elif page == "Population Surveillance":
    st.title("Population Surveillance")
    df2 = load_surveillance_data()
    if df2 is not None:
        st.markdown("Decade-scale trends from the BRFSS Surveillance set.")
        
        ages = sorted([a for a in df2['Age'].dropna().unique() if a not in ['All', 'Crude', 'Age-Adjusted']])
        sel_age = st.selectbox("Stratify by Age", ["All Ages"] + ages)
        
        if sel_age == "All Ages":
            f_df = df2[(df2['Population'] == 'All Ages') & (df2['Age'] == 'Crude') & (df2['Sex'] == 'All') & (df2['Race'] == 'All') & (df2['Indicator'] == 'Diagnosed Diabetes')].sort_values("Year")
        else:
            f_df = df2[(df2['Age'] == sel_age) & (df2['Sex'] == 'All') & (df2['Race'] == 'All') & (df2['Indicator'] == 'Diagnosed Diabetes')].drop_duplicates(subset=['Year']).sort_values("Year")
            
        fig = px.line(f_df, x="Year", y="Estimate", markers=True, title=f"Diabetes Prevalence Over Time ({sel_age})",
                      color_discrete_sequence=["#14b8a6"])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Surveillance dataset not found.")

elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown("Evaluating performance on the strictly out-of-fold locked test set.")
    
    path = os.path.join(RESULTS_TABLES, "final_model_comparison_plot_data.csv")
    if os.path.exists(path):
        df_comp = pd.read_csv(path)
        
        # Display the image that was already generated
        img_path = os.path.join(RESULTS_FIGURES, "final_model_comparison_v2.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        
        st.markdown("### Raw Test Set Metrics")
        st.dataframe(df_comp.style.background_gradient(cmap='Blues'), use_container_width=True)
    else:
        st.info("Run the experiment notebooks/scripts to generate comparison tables.")

elif page == "Explainable AI":
    st.title("Explainable AI (XAI)")
    st.markdown("Global and localized interpretations via SHAP.")
    
    shap_img = os.path.join(RESULTS_EXPLANATIONS, "13_shap_importance_bar.png")
    if os.path.exists(shap_img):
        st.image(shap_img, caption="Global Feature Importance across all classes")
    else:
        st.info("SHAP outputs not found. Run XAI notebooks.")

elif page == "Failure Analysis":
    st.title("Failure Analysis")
    st.markdown("Understanding model weaknesses to construct the Failure-Aware Hybrid.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Error Complementarity")
        comp = os.path.join(RESULTS_FIGURES, "11_complementarity_heatmap.png")
        if os.path.exists(comp):
            st.image(comp, caption="Percentage of Row's errors corrected by Column")
            
    with c2:
        st.markdown("### Disagreement vs Error Rate")
        st.markdown("""
        <div class="custom-card">
            <h4>The Hypothesis</h4>
            <p>Our research demonstrated that as model disagreement increases, the probability of an error spikes dramatically. 
            By capturing this disagreement mathematically via <b>Entropy</b> and <b>Max Agreement</b>, the Hybrid model successfully balances the detection of minority classes.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "Public Health Insights":
    st.title("Public Health Insights")
    st.markdown("""
    <div class="custom-card">
        <h4>1. The Minority Class Crisis</h4>
        <p>Standard ML models completely fail to detect Prediabetes (miss rate >99%). High global accuracy in medical datasets is often a deceptive illusion masking the failure to detect rare conditions.</p>
    </div>
    
    <div class="custom-card" style="border-left-color: #38bdf8;">
        <h4>2. Value of Balanced Metrics</h4>
        <p>Using Balanced Accuracy and MCC forces algorithms to acknowledge at-risk patients, shifting the focus from statistical accuracy to clinical utility.</p>
    </div>
    
    <div class="custom-card" style="border-left-color: #f59e0b;">
        <h4>3. The Alcohol Paradox</h4>
        <p>Heavy alcohol consumption shows negative correlation with diagnosed diabetes in cross-sectional surveys, largely because diagnosed patients modify their behavior post-diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "About":
    st.title("About the System")
    st.markdown("""
    This platform was developed as part of a B.Tech Research Project analyzing the CDC BRFSS Dataset.
    
    **Disclaimer:**
    This system is intended strictly for analytical and public-health research purposes. It is NOT a medical diagnostic tool. Predictions generated by the Failure-Aware Hybrid should not be used for clinical decision-making.
    
    **Data Sources:**
    - CDC Behavioral Risk Factor Surveillance System (BRFSS)
    """)
