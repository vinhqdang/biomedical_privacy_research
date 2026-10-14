"""
Comprehensive Trustworthy AI Metrics Suite for Biomedical and Healthcare Applications
Evaluates:
1. Clinical Utility: ROC-AUC, PR-AUC, F1, Balanced Accuracy, Sensitivity, Specificity
2. Calibration: Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier Score
3. Algorithmic Fairness: Demographic Parity Disparity, Equalized Odds Disparity, Subgroup AUC Gap
4. Empirical Privacy: Membership Inference Attack (MIA) AUC
5. Conformal Reliability: Group-conditional coverage, set size, coverage disparity
6. XAI Quality: Explanation Infidelity, Faithfulness Correlation
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

def compute_clinical_metrics(y_true, y_probs, threshold=0.5):
    """
    Computes clinical classification metrics.
    y_true: (N,) binary ground truth
    y_probs: (N,) predicted positive probabilities
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)
    
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = 0.5 * (sensitivity + specificity)
    
    return {
        "AUC": float(auc),
        "PR_AUC": float(pr_auc),
        "F1": float(f1),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Balanced_Acc": float(balanced_acc)
    }

def compute_calibration_metrics(y_true, y_probs, num_bins=10):
    """
    Computes Expected Calibration Error (ECE), MCE, and Brier Score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_probs = np.asarray(y_probs, dtype=float)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y_true)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper if i < num_bins - 1 else y_probs <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_probs[in_bin])
            bin_error = np.abs(bin_acc - bin_conf)
            ece += (bin_size / n) * bin_error
            mce = max(mce, bin_error)
            
    brier_score = float(np.mean((y_probs - y_true) ** 2))
    
    return {
        "ECE": float(ece),
        "MCE": float(mce),
        "Brier": float(brier_score)
    }

def compute_fairness_metrics(y_true, y_probs, sensitive_attrs, threshold=0.5):
    """
    Computes Demographic Parity Disparity, Equalized Odds Disparity, and Subgroup AUC Gap.
    sensitive_attrs: (N,) binary group indicators (0: minority/unprivileged, 1: majority/privileged)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)
    s = np.asarray(sensitive_attrs, dtype=int)
    
    mask_0 = (s == 0)
    mask_1 = (s == 1)
    
    # Selection rate (positive predicted rate)
    sr_0 = np.mean(y_pred[mask_0]) if np.sum(mask_0) > 0 else 0.0
    sr_1 = np.mean(y_pred[mask_1]) if np.sum(mask_1) > 0 else 0.0
    dp_disparity = abs(sr_0 - sr_1)
    
    # True Positive Rate and False Positive Rate per group
    def get_rates(mask):
        sub_y = y_true[mask]
        sub_pred = y_pred[mask]
        pos = (sub_y == 1)
        neg = (sub_y == 0)
        tpr = np.mean(sub_pred[pos]) if np.sum(pos) > 0 else 0.0
        fpr = np.mean(sub_pred[neg]) if np.sum(neg) > 0 else 0.0
        return tpr, fpr
        
    tpr_0, fpr_0 = get_rates(mask_0)
    tpr_1, fpr_1 = get_rates(mask_1)
    
    eo_disparity = 0.5 * (abs(tpr_0 - tpr_1) + abs(fpr_0 - fpr_1))
    
    # Subgroup AUCs
    auc_0 = roc_auc_score(y_true[mask_0], y_probs[mask_0]) if len(np.unique(y_true[mask_0])) > 1 else 0.5
    auc_1 = roc_auc_score(y_true[mask_1], y_probs[mask_1]) if len(np.unique(y_true[mask_1])) > 1 else 0.5
    auc_gap = abs(auc_1 - auc_0)
    
    return {
        "DP_Disparity": float(dp_disparity),
        "EO_Disparity": float(eo_disparity),
        "AUC_Group0": float(auc_0),
        "AUC_Group1": float(auc_1),
        "AUC_Gap": float(auc_gap)
    }

def compute_mia_risk(train_probs, train_y, test_probs, test_y):
    """
    Simulates a Membership Inference Attack (MIA) using prediction loss/confidence.
    Attacker predicts membership based on whether cross-entropy loss is below a calibrated threshold.
    Returns MIA AUC (0.50 = optimal privacy / random guess; 1.0 = complete privacy leakage).
    """
    def log_loss(probs, y):
        eps = 1e-7
        p_c = np.clip(np.where(y == 1, probs, 1.0 - probs), eps, 1.0 - eps)
        return -np.log(p_c)
        
    train_losses = log_loss(train_probs, train_y)
    test_losses = log_loss(test_probs, test_y)
    
    # Member = 1 (train), Non-member = 0 (test)
    y_mia = np.concatenate([np.ones_like(train_losses), np.zeros_like(test_losses)])
    # Lower loss indicates higher likelihood of membership -> score = -loss
    scores_mia = -np.concatenate([train_losses, test_losses])
    
    mia_auc = roc_auc_score(y_mia, scores_mia)
    return float(mia_auc)
