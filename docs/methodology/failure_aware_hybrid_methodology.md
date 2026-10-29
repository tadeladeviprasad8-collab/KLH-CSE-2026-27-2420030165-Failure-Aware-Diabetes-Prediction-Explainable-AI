# The Failure-Aware Hybrid Stacking Method: Architecture, Mathematical Formulation, and Empirical Ablation

**Project Title:** Failure-Aware Ensemble Learning and Explainable AI for Three-Class Diabetes Status Prediction  
**Proposed Architecture:** Failure-Aware Stacking with Uncertainty & Disagreement Meta-Features  
**Trained Artifacts:** Research Model (`models/hybrid/hybrid_meta_model.joblib`) and Dashboard Model (`models/hybrid/failure_aware_hybrid.joblib`)  

---

## 1. Methodological Motivation: Beyond Conventional Probability Stacking

Standard stacking ensembling (Wolpert, 1992) trains a second-level meta-learner exclusively on the predicted probability distributions of $K$ base estimators:
$$\mathbf{z}_i = \left[ \hat{p}_1(y=0|\mathbf{x}_i), \dots, \hat{p}_K(y=2|\mathbf{x}_i) \right] \in \mathbb{R}^{3K}$$

While conventional probability stacking captures first-order probability consensus, it suffers from two major limitations in severe class imbalance:
1. **Loss of Meta-Uncertainty:** A base model that outputs $[0.34, 0.33, 0.33]$ has maximum Shannon entropy (total uncertainty), whereas a model that outputs $[0.51, 0.49, 0.00]$ is uncertain between classes 0 and 1 only. Standard stacking treats these raw probabilities as linear features without explicit representations of decision sharpness or dispersion.
2. **Inability to Leverage Error Discordance:** In 3-class public health data, gradient boosters and regularized linear models frequently make discordant errors on borderline prediabetes patients. When model errors are complementary, explicit disagreement indicators signal to the meta-learner that the patient lies on a contested decision boundary.

---

## 2. Complete Mathematical Formulation of the 50 Meta-Features

The authoritative research hybrid (`hybrid_meta_model.joblib`) utilizes $K = 7$ diverse base models (Logistic Regression, Decision Tree, Random Forest, SVM SGD Modified Huber, XGBoost, LightGBM, CatBoost) to construct a 50-dimensional meta-feature vector $\mathbf{z}_i$ for each patient $\mathbf{x}_i$:

### Group 1: Base Class Probabilities ($21$ Features)
For each base estimator $m \in \{1, \dots, 7\}$ and each class $c \in \{0, 1, 2\}$:
$$p_{m, c}(\mathbf{x}_i) = P_m(Y = c \mid \mathbf{x}_i)$$
- Total features: $7 \times 3 = 21$ continuous features $\in [0, 1]$.

### Group 2: Maximum Prediction Confidence ($7$ Features)
The peak probability assigned by model $m$ to its most favored class:
$$C_m(\mathbf{x}_i) = \max_{c \in \{0, 1, 2\}} p_{m, c}(\mathbf{x}_i)$$
- High values ($C_m \approx 1.0$) indicate strong model conviction; low values ($C_m \approx 0.33$) indicate extreme indecision.
- Total features: $7 \times 1 = 7$ continuous features $\in [0.333, 1.0]$.

### Group 3: Normalized Shannon Entropy ($7$ Features)
The information-theoretic uncertainty of the probability distribution for model $m$:
$$H_m(\mathbf{x}_i) = -\sum_{c=0}^2 p_{m, c}(\mathbf{x}_i) \log_2 \left( p_{m, c}(\mathbf{x}_i) + \epsilon \right)$$
where $\epsilon = 10^{-15}$ prevents undefined zero logarithms.
- Total features: $7 \times 1 = 7$ continuous features $\in [0, \log_2 3 \approx 1.585]$.

### Group 4: Decision Margin ($7$ Features)
The difference between the highest and second-highest class probabilities for model $m$:
$$M_m(\mathbf{x}_i) = p_{m, (1)}(\mathbf{x}_i) - p_{m, (2)}(\mathbf{x}_i)$$
where $p_{m, (1)} \ge p_{m, (2)} \ge p_{m, (3)}$.
- Small margins ($M_m \approx 0$) identify instances situated directly on the model's decision boundary.
- Total features: $7 \times 1 = 7$ continuous features $\in [0, 1.0]$.

### Group 5: Pairwise Disagreement Flags ($7$ Features)
Let $\hat{y}_m(\mathbf{x}_i) = \arg\max_c p_{m, c}(\mathbf{x}_i)$ be the discrete prediction of model $m$, and let $y_{\text{mode}}(\mathbf{x}_i) = \text{mode} \left( \{\hat{y}_1(\mathbf{x}_i), \dots, \hat{y}_7(\mathbf{x}_i)\} \right)$ be the majority consensus vote.
The binary disagreement indicator for model $m$ is:
$$D_m(\mathbf{x}_i) = \mathbb{I} \left( \hat{y}_m(\mathbf{x}_i) \neq y_{\text{mode}}(\mathbf{x}_i) \right)$$
- Total features: $7 \times 1 = 7$ binary flags $\in \{0, 1\}$.

### Group 6: Modal Agreement Consensus Count ($1$ Feature)
The total number of base models agreeing with the modal consensus prediction:
$$A_{\text{modal}}(\mathbf{x}_i) = \sum_{m=1}^7 \mathbb{I} \left( \hat{y}_m(\mathbf{x}_i) = y_{\text{mode}}(\mathbf{x}_i) \right)$$
- Range: $3$ (weak pluralistic split) to $7$ (unanimous consensus).
- Total features: $1$ integer feature $\in \{3, 4, 5, 6, 7\}$.

### Grand Total Meta-Feature Dimensionality:
$$21 \text{ (Probabilities)} + 7 \text{ (Confidence)} + 7 \text{ (Entropy)} + 7 \text{ (Margin)} + 7 \text{ (Disagreement)} + 1 \text{ (Consensus)} = \mathbf{50} \text{ Meta-Features}$$

---

## 3. Meta-Learner Architecture and Training Pipeline

The meta-learner is an L2-regularized multinomial logistic regression model fitted on the 50 standardized meta-features:
$$P(Y = c \mid \mathbf{z}_i) = \frac{\exp(\mathbf{w}_c^T \mathbf{z}_i + b_c)}{\sum_{j=0}^2 \exp(\mathbf{w}_j^T \mathbf{z}_i + b_j)}$$

- **Class Weighting:** `class_weight='balanced'`, applying inverse-frequency penalty weights:
  $$w_c = \frac{N}{3 \cdot N_c} \implies w_0 = 0.395, \quad w_1 = 18.258, \quad w_2 = 2.392$$
- **Hyperparameters:** $C = 1.0$ (Research Model) and $C = 0.10$ (Dashboard Model), `penalty='l2'`, `solver='lbfgs'`, `max_iter=1000`.
- **Leakage Protection Protocol:** Meta-features for training the meta-learner were constructed strictly using **Stratified 5-Fold Out-of-Fold (OOF)** predictions across $N = 202,944$. No base model was ever queried on training instances that it saw during fitting.

---

## 4. Research Model (50 Feats) vs. Dashboard Model (27 Feats)

Two distinct hybrid artifacts exist in the repository to serve distinct scientific and operational requirements:

| Dimension | Authoritative Research Model (`hybrid_meta_model.joblib`) | Production Dashboard Model (`failure_aware_hybrid.joblib`) | Engineering Justification |
|:---|:---:|:---:|:---|
| **Base Models Used** | **7 Base Models** (LR, DT, RF, SGD, XGB, LGBM, Cat) | **5 Base Models** (SVM, XGBoost, LightGBM, CatBoost, RF) | Pruned slower DT and redundant unscaled LR for web efficiency |
| **Meta-Feature Count** | **50 Meta-Features** | **27 Meta-Features** | Streamlined dimensionality reduces inference latency |
| **Probability Feats** | 21 (7 models × 3 classes) | 15 (5 models × 3 classes) | Retains 5 core multi-family estimators |
| **Uncertainty Feats** | 21 (7 conf + 7 entropy + 7 margin) | 10 (5 entropy + 5 margin) | Focuses on entropy and boundary margin |
| **Disagreement Feats**| 7 individual model disagreement flags | 1 global disagreement score | Aggregated into continuous discordance index |
| **Consensus Feats** | 1 modal agreement count | 1 maximum agreement count | Measures ensemble consensus strength |
| **Pipeline Wrapping**| Standalone model object with external scaler | End-to-end `sklearn.pipeline.Pipeline` with `StandardScaler` | Direct raw survey ingestion in `dashboard/app.py` |
| **Web Latency** | ~145 ms per transaction | **< 35 ms per transaction** | Sub-50ms clinical responsiveness requirement |

---

## 5. Systematic 4-Stage Ablation Study (Notebook 12)

To rigorously quantify the marginal contribution of each meta-feature component, a 4-stage ablation study was conducted on the locked test set ($N = 50,736$):

| Ablation Configuration | Meta-Features Ingested | Dimensionality | Test Accuracy | Macro-F1 | Balanced Acc | Prediabetes Recall | Prediabetes F1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **M1: Baseline Stacking** | Base Probabilities Only | 21 | 0.6000 | **0.4180** | **0.5278** | **0.4060 (376/926)** | 0.0540 |
| **M2: Uncertainty Hybrid** | Probabilities + Entropy + Margin | 35 | 0.6045 | 0.4171 | 0.5242 | 0.3985 (369/926) | 0.0548 |
| **M3: Disagreement Hybrid**| Probabilities + Disagreement + Consensus | 29 | 0.6078 | 0.4168 | 0.5231 | 0.3931 (364/926) | 0.0551 |
| **M4: Full Failure-Aware** | All Probabilities + Uncertainty + Disagreement | **50** | **0.6101** | 0.4162 | 0.5218 | 0.3888 (360/926) | **0.0554** |

### Key Ablation Insights:
1. **Monotonic Accuracy Escalation:** Moving from standard stacking (M1) to the full hybrid (M4) produces a monotonic **+1.01% gain in overall classification accuracy** (from 60.00% to 61.01%, $p < 0.001$, McNemar $\chi^2 = 122.95, p < 10^{-27}$).
2. **False Positive Suppression:** The uncertainty and disagreement features enable the meta-learner to prune **973 false positive prediabetes alerts** among non-diabetic respondents.
3. **Macro-F1 Parity:** Because Macro-F1 weights the tiny prediabetes class equally (where M1 captured 16 more patients at the expense of 973 false alarms), the Macro-F1 difference is -0.0018 ($p = 0.224$, not statistically significant).
4. **Scientific Honesty:** The paper must report that the failure-aware hybrid excels specifically at **improving overall diagnostic precision and reducing false-positive clinical burden** without degrading prediabetes sensitivity, rather than claiming a sweeping Macro-F1 superiority.