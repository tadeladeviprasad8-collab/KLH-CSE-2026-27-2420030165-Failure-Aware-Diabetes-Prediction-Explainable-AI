# Experimental Results, Metrics, and Publication Figures

This directory contains all verified, reproducible evaluation outputs across the 10 evaluated machine learning configurations on the locked test set ($N = 50,736$):

## Subdirectories
- **`metrics/`**: Compact, publication-grade CSV tables detailing multi-class accuracy, Macro-F1, Balanced Accuracy, MCC, per-class sensitivity, calibration loss, and bootstrap significance tests.
- **`figures/`**: 22 publication-grade figures available in high-resolution 300 DPI PNG and vector SVG formats, accompanied by formal figure captions and manifest metadata.
- **`source_data/`**: Exact underlying CSV data used to generate each of the 22 research figures.

### Key Metrics Summary (Locked Test Set: $N = 50,736$)
- **SGDClassifier (Modified Huber):** Accuracy = 0.8014, Prediabetes Recall = 0.0043 (4/926 cases).
- **Soft Voting (Average):** Accuracy = 0.7477, Macro-F1 = 0.4476, Prediabetes Recall = 0.0421 (39/926 cases).
- **Standard Stacking (Corrected):** Accuracy = 0.6000, Balanced Acc = 0.5278, Prediabetes Recall = 0.4060 (376/926 cases).
- **Failure-Aware Hybrid (Corrected):** Accuracy = 0.6101, Balanced Acc = 0.5218, Prediabetes Recall = 0.3888 (360/926 cases).
- **Statistical Significance (Bootstrap B=1,000):** Hybrid achieves +1.01% accuracy over Stacking ($p < 0.001$), McNemar $\chi^2 = 122.95, p = 1.43 \times 10^{-28}$.
