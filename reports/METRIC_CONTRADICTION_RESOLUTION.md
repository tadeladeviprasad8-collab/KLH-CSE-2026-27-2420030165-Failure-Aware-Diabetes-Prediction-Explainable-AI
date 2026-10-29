# METRIC CONTRADICTION RESOLUTION REPORT
**Audit Stage:** Independent Code & File Verification Pass  
**Dataset:** CDC BRFSS 2015 Locked Test Set ($N = 50,736$)

---

## 1. Contradiction 1: Highest Accuracy Claim (Soft Voting vs. SGD Modified Huber)
* **The Contradiction:** Early narrative text in previous audit summaries stated that *'Soft Voting achieves the highest Accuracy'*, whereas the master results table clearly reported **SGD Modified Huber at 0.8014 (80.14%)** and Soft Voting at **0.7477 (74.77%)**.
* **Root Cause & Code Evidence:**
  * When evaluated across **all 12 individual configurations**, **SGDClassifier (Modified Huber) achieves the highest raw Accuracy (0.8014)**.
  * However, SGD achieves 80.14% raw accuracy purely by aggressively predicting the majority class ($C_0$: No Diabetes, 84.2% of population). Its Prediabetes recall is nearly zero (**0.43%**, detecting only 4 out of 926 prediabetes cases).
  * Among the **multi-model ensemble architectures** (Soft Voting vs. Standard Stacking vs. Failure-Aware Hybrid), **Soft Voting achieves the highest raw Accuracy (0.7477)** compared to Stacking (0.6000) and Hybrid (0.6101).
  * Soft Voting also achieves the highest overall **Macro F1 (0.4476)** among all evaluated models.
* **Resolution for Manuscript:** The text must state:
  > *"SGDClassifier achieves the highest raw classification accuracy (80.14%) but fails on the minority class (Prediabetes recall 0.43%). Among multi-classifier ensemble architectures, Soft Voting achieves the highest accuracy (74.77%) and highest Macro-F1 (0.4476), whereas Standard Stacking and the Failure-Aware Hybrid prioritize Balanced Accuracy (52.2%–52.8%) and capture ~40% of prediabetes cases."*

---

## 2. Contradiction 2: Prediabetes Recall Discrepancy (0.3963 vs. 0.3888)
* **The Contradiction:** `oof_weight_correction_comparison.csv` reported Corrected Hybrid Prediabetes Recall as **0.396328 (39.63%)**, whereas `MODEL_COMPARISON_MASTER.csv` reported **0.388769 (38.88%)**.
* **Root Cause & Code Evidence:**
  * `0.396328` was produced during the initial execution of `run_exp_audit_weight_fix.py` when XGBoost OOF cross-validation folds were trained in-memory with `n_jobs=-1`. In XGBoost on multi-threaded CPU architectures, asynchronous reduction introduces minor floating-point divergence across runs (detecting 367 out of 926 prediabetes cases).
  * `0.388769` was produced when `LogisticRegression` was fitted deterministically on the serialized disk file `09_tuned_oof_predictions_corrected.csv` (detecting 360 out of 926 prediabetes cases).
  * **Absolute Difference:** Exactly **7 cases out of 926** (a 0.75% difference, well within the 95% bootstrap confidence interval of $[0.3587, 0.4207]$).
* **Resolution for Manuscript:** **0.3888 (38.88%)** is the authoritative, verified number because it is reproducible directly from the saved disk predictions in `TEST_PREDICTIONS_MASTER.csv` and `OOF_PREDICTIONS_CORRECTED.csv`.

---

## 3. Contradiction 3: Unification of Stacking and Hybrid Comparisons
* **The Contradiction:** Previously, the hybrid was compared against an uncorrected standard stacking baseline.
* **Resolution:** Both Standard Stacking and the Production Hybrid were retrained and evaluated on the exact same weight-corrected OOF predictions:
  * **Standard Stacking (Corrected):** Accuracy = 0.6000, Macro-F1 = 0.4180, Balanced Acc = 0.5278, Prediabetes Recall = 40.60%.
  * **Production Hybrid (Corrected):** Accuracy = 0.6101, Macro-F1 = 0.4162, Balanced Acc = 0.5218, Prediabetes Recall = 38.88%.
  * Both models perform on statistically equivalent terms ($p = 0.224$ for Macro-F1, $p = 0.168$ for Balanced Accuracy).
