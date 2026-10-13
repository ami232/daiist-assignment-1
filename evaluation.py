import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

LGD = 0.65      # share of principal lost when a loan defaults (assumption)
MARGIN = 0.10   # net interest margin earned on a repaid loan (assumption)


def business_cost(y, score, threshold, loan_amnt, lgd=LGD, margin=MARGIN):
    """Reject a loan when score >= threshold.
    False negative (approved, defaults): lose lgd * amount.
    False positive (rejected, would have repaid): lose margin * amount."""
    reject = score >= threshold
    fn = (~reject) & (y == 1)
    fp = reject & (y == 0)
    return float((lgd * loan_amnt[fn]).sum() + (margin * loan_amnt[fp]).sum())


def best_threshold(y, score, loan_amnt):
    """Threshold that minimizes business cost (used on VALIDATION only)."""
    candidates = np.unique(np.quantile(score, np.linspace(0, 1, 201)))
    candidates = np.append(candidates, np.inf)  # inf = approve everyone
    costs = [business_cost(y, score, t, loan_amnt) for t in candidates]
    i = int(np.argmin(costs))
    return float(candidates[i]), costs[i]


def evaluate(y_val, s_val, amt_val, y_te, s_te, amt_te):
    """Pick the threshold on validation, report everything on test."""
    thr, val_cost = best_threshold(y_val, s_val, amt_val)
    return {
        "threshold": None if np.isinf(thr) else thr,
        "val_auc": float(roc_auc_score(y_val, s_val)),
        "test_auc": float(roc_auc_score(y_te, s_te)),
        "test_pr_auc": float(average_precision_score(y_te, s_te)),
        "val_cost": val_cost,
        "test_cost": business_cost(y_te, s_te, thr, amt_te),
        "test_cost_approve_all": business_cost(y_te, s_te, np.inf, amt_te),
    }