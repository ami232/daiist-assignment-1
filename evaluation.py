"""Metrics and the student's explicitly hypothetical call-cost policy."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss, precision_score, recall_score, roc_auc_score)

CALL_COST = 2.0
MISSED_COST = 50.0


def policy_counts(y, p, threshold, call_cost=CALL_COST, missed_cost=MISSED_COST):
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'calls': int(tp + fp), 'cost_eur': float(call_cost * (tp + fp) + missed_cost * fn)}


def evaluate(y, p, threshold):
    pred = np.asarray(p) >= threshold
    return {'roc_auc': float(roc_auc_score(y, p)), 'average_precision': float(average_precision_score(y, p)),
            'log_loss': float(log_loss(y, p, labels=[0, 1])), 'brier': float(brier_score_loss(y, p)),
            'accuracy': float(accuracy_score(y, pred)), 'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'threshold': float(threshold), **policy_counts(y, p, threshold)}


def select_threshold(y, p):
    """Minimize validation cost over exact attainable policies; ties use fewer calls.

    Process each group of equal scores together. O(n log n), including call-none
    and call-all. No test labels are used to set the frozen default threshold.
    """
    y, p = np.asarray(y, dtype=int), np.asarray(p)
    order = np.argsort(-p, kind='stable')
    sorted_p, sorted_y = p[order], y[order]
    ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(p) - 1]
    calls = ends + 1
    fn = y.sum() - sorted_y.cumsum()[ends]
    costs = CALL_COST * calls + MISSED_COST * fn
    # Use midpoints between distinct scores to avoid a policy changing
    # after CSV round-tripping or equivalent sigmoid implementations.
    lower_scores = np.r_[sorted_p[ends[1:]], 0.0]
    thresholds = (sorted_p[ends] + lower_scores) / 2
    thresholds[-1] = 0.0
    costs = np.r_[MISSED_COST * y.sum(), costs]
    thresholds = np.r_[1.0, thresholds]
    i = int(np.argmin(costs))
    return float(thresholds[i])
