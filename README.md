# ❤️ Heart Disease Early Risk Prediction Using Multi-Dataset Analysis

## 📌 About the Project

**Heart Disease Early Risk Prediction Using Multi-Dataset Analysis** is a machine learning-based academic project designed to predict the risk of heart disease at an early stage using multiple heart disease datasets.

The project combines datasets collected from **Cleveland, Hungary, Switzerland, and Long Beach** and provides a unified framework for:

* Data preprocessing
* Data quality analysis
* Exploratory Data Analysis (EDA)
* Feature engineering
* Feature selection
* Machine learning model development
* Model comparison
* Location-wise analysis
* Prediction interpretation
* Interactive visualization

The main goal is to identify important clinical patterns and factors associated with heart disease and develop a reliable machine learning-based prediction system.

---

## 🎯 Objectives

The major objectives of this project are:

1. Perform data preprocessing and data quality analysis.
2. Handle missing values and inconsistent data.
3. Conduct Exploratory Data Analysis (EDA).
4. Identify important patterns and relationships in the datasets.
5. Perform feature engineering and feature selection.
6. Develop multiple machine learning classification models.
7. Compare the performance of different models.
8. Evaluate models using standard classification metrics.
9. Perform location-wise analysis across multiple datasets.
10. Provide interpretable insights into model predictions.
11. Develop an interactive dashboard for prediction and visualization.

---

## 🧠 Machine Learning Models

The project evaluates the following classification algorithms:

| Model                           | Purpose                                 |
| ------------------------------- | --------------------------------------- |
| Logistic Regression             | Baseline linear classification model    |
| Decision Tree                   | Rule-based classification               |
| Random Forest                   | Ensemble learning and robust prediction |
| K-Nearest Neighbours (KNN)      | Distance-based classification           |
| Support Vector Classifier (SVC) | Margin-based classification             |

---

## 📊 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

These metrics help determine how effectively each model identifies patients who may be at risk of heart disease.

---

## 🗂️ Dataset

The project uses the **UCI Heart Disease datasets** collected from four locations:

* 🇺🇸 Cleveland, USA
* 🇭🇺 Hungary
* 🇨🇭 Switzerland
* 🇺🇸 Long Beach, USA

### Clinical Attributes

The datasets contain clinical attributes such as:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Fasting blood sugar
* Resting ECG
* Maximum heart rate
* Exercise-induced angina
* ST depression
* Slope
* Number of major vessels
* Thalassemia
* Heart disease diagnosis

---

## 🔄 Project Workflow

```text
                 Dataset Collection
                        ↓
                Data Preprocessing
                        ↓
              Data Quality Analysis
                        ↓
             Exploratory Data Analysis
                        ↓
          Feature Engineering & Selection
                        ↓
                Machine Learning
                        ↓
                Model Evaluation
                        ↓
             Location-Wise Analysis
                        ↓
             Prediction & Visualization
                        ↓
              Explainable AI Analysis
                        ↓
              Interactive Dashboard
```

---

## 🏗️ Project Architecture

```text
Heart Disease Prediction System
│
├── Data Collection
│   ├── Cleveland Dataset
│   ├── Hungary Dataset
│   ├── Switzerland Dataset
│   └── Long Beach Dataset
│
├── Data Preprocessing
│   ├── Missing Value Handling
│   ├── Data Cleaning
│   ├── Encoding
│   └── Feature Scaling
│
├── Exploratory Data Analysis
│   ├── Age Distribution
│   ├── Gender Analysis
│   ├── Cholesterol Analysis
│   ├── Blood Pressure Analysis
│   └── Correlation Analysis
│
├── Machine Learning
│   ├── Logistic Regression
│   ├── Decision Tree
│   ├── Random Forest
│   ├── KNN
│   └── SVC
│
├── Evaluation
│   ├── Accuracy
│   ├── Precision
│   ├── Recall
│   ├── F1 Score
│   ├── ROC-AUC
│   └── Confusion Matrix
│
├── Explainable AI
│   ├── SHAP
│   └── LIME
│
└── Streamlit Dashboard
    ├── Risk Prediction
    ├── Data Visualization
    ├── Feature Analysis
    ├── Location Analysis
    └── Model Performance
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

### Data Visualization

* Matplotlib
* Seaborn
* Plotly

### Explainable AI

* SHAP
* LIME

### Dashboard

* Streamlit

### Model Management

* Joblib

### Development Environment

* Jupyter Notebook
* VS Code
* Anaconda

---

## 📁 Project Structure

```text
heart-disease-prediction/
│
├── data/
│   ├── cleveland.csv
│   ├── hungary.csv
│   ├── switzerland.csv
│   ├── long_beach.csv
│   └── combined_heart_disease.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_explainable_ai.ipynb
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── knn.pkl
│   └── svc.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluation.py
│   └── explainability.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📈 Dashboard

The Streamlit dashboard provides the following features:

### 1. ❤️ Heart Disease Risk Prediction

Users can enter clinical information such as:

* Age
* Sex
* Chest pain type
* Blood pressure
* Cholesterol
* Maximum heart rate
* Exercise-induced angina
* Other relevant clinical parameters

The trained machine learning model predicts the estimated heart disease risk.

### 2. 📊 Data Visualization

The dashboard provides interactive charts for:

* Age distribution
* Gender distribution
* Cholesterol levels
* Blood pressure
* Heart rate
* Disease distribution
* Feature relationships

### 3. 🌍 Location-Wise Analysis

The system compares heart disease patterns across:

```text
Cleveland
   ↓
Hungary
   ↓
Switzerland
   ↓
Long Beach
```

This allows differences between datasets and locations to be analyzed.

### 4. 🤖 Model Performance

The dashboard displays model performance using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

### 5. 🔍 Prediction Interpretation

Explainable AI techniques are used to understand which features contributed most to a prediction.

The project uses:

* SHAP
* LIME

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
```

### 2. Navigate to the Project

```bash
cd <your-repository>
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
shap
lime
streamlit
joblib
```

---

## 🔬 Research Methodology

The project follows a systematic machine learning pipeline.

### Step 1 — Dataset Collection

Heart disease datasets are collected from four locations:

```text
Cleveland
Hungary
Switzerland
Long Beach
```

### Step 2 — Data Preprocessing

The collected datasets are cleaned and transformed.

Operations include:

* Missing value treatment
* Duplicate removal
* Data type correction
* Categorical encoding
* Feature scaling
* Target transformation

### Step 3 — Exploratory Data Analysis

EDA is performed to understand:

* Distribution of clinical attributes
* Relationships between variables
* Correlations
* Disease patterns
* Location-wise differences

### Step 4 — Feature Engineering

Important features are selected and transformed to improve model performance.

### Step 5 — Model Training

Five classification algorithms are trained:

```text
Logistic Regression
Decision Tree
Random Forest
KNN
SVC
```

### Step 6 — Model Evaluation

Models are compared using:

```text
Accuracy
Precision
Recall
F1-Score
ROC-AUC
Confusion Matrix
```

### Step 7 — Explainable AI

SHAP and LIME are used to understand the contribution of individual features to model predictions.

### Step 8 — Dashboard Development

The final model and analysis results are integrated into a Streamlit dashboard.

---

## 📌 Expected Outcomes

The project is expected to:

* Identify important heart disease risk factors.
* Compare multiple machine learning algorithms.
* Determine the best-performing classification model.
* Analyze differences between geographical datasets.
* Provide interpretable machine learning predictions.
* Provide an easy-to-use interactive prediction dashboard.

---

## 👨‍💻 Project Team

| University ID | Name                  |
| ------------- | --------------------- |
| 2420030165    | **T. Devi Prasad**    |
| 2420030272    | **D. Vinay Varma**    |
| 2420030271    | **R. Harshit**        |
| 2420030600    | **B. Sidharth Reddy** |

### Project Guide

**K. Swapnika**

---

## 🎓 Project Status

**Academic Capstone Project — 2026–2027**

---

## ⚠️ Disclaimer

This project is developed for **academic and research purposes only**.

The predictions generated by this system should **not** be considered a substitute for professional medical diagnosis, treatment, or clinical decision-making.

Always consult a qualified healthcare professional for medical advice.

---

## ⭐ Future Enhancements

Future improvements may include:

* Deep learning-based prediction
* Larger and more diverse clinical datasets
* Real-time prediction services
* Advanced explainability techniques
* Cloud deployment
* Mobile application integration
* Model monitoring
* Continuous model retraining
* Additional external validation datasets

---

## 📜 License

This project is intended for academic and educational purposes.

© 2026 Heart Disease Early Risk Prediction Project Team
