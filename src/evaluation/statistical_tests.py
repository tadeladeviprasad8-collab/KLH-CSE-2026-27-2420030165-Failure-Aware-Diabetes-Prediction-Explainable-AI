"""
Inferential Statistical Testing: Paired Bootstrap and McNemar Discordance
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

def paired_bootstrap_test(y_true, y_pred_a, y_pred_b, metric_fn, n_resamples=1000, seed=42):
    """
    Compute two-tailed paired bootstrap hypothesis test:
    H0: Difference in metric between Model A and Model B is zero.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []
    
    for _ in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        m_a = metric_fn(y_true[indices], y_pred_a[indices])
        m_b = metric_fn(y_true[indices], y_pred_b[indices])
        diffs.append(m_a - m_b)
        
    diffs = np.array(diffs)
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    # Two-tailed p-value
    p_val = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {
        'mean_difference': np.mean(diffs),
        'ci_95': (ci_lower, ci_upper),
        'two_tailed_p_value': p_val
    }

def mcnemar_discordance_test(y_true, y_pred_a, y_pred_b):
    """
    Compute McNemar test with continuity correction on paired classification outcomes:
    Chi2 = (|b - c| - 1)^2 / (b + c)
    """
    from scipy.stats import chi2
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    
    b = np.sum(correct_a & ~correct_b) # A correct, B incorrect
    c = np.sum(~correct_a & correct_b) # A incorrect, B correct
    
    chi2_stat = ((abs(b - c) - 1.0)**2) / (b + c)
    p_value = 1.0 - chi2.cdf(chi2_stat, df=1)
    return {
        'model_a_only_correct': int(b),
        'model_b_only_correct': int(c),
        'chi2_statistic': float(chi2_stat),
        'df': 1,
        'p_value': float(p_value)
    }
