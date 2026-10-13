"""
business.py - the business assumptions, in ONE place.

Used by training (to choose the threshold) and by the dashboard
(to show the cost as you move the threshold slider).
"""
import numpy as np

# ---------------------------------------------------------------
# ASSUMPTIONS (explained in REPORT.md - they are estimates)
# ---------------------------------------------------------------
# A typical City Hotel booking is worth ~EUR 265 (EUR 100/night x 3 nights).
# Only ~19% of cancellations happen inside the 14-day refund window,
# so a missed cancellation costs on average 0.19 x 265 = ~EUR 50.
COST_MISSED_CANCELLATION = 50   # false negative: no deposit asked, guest cancels

# Asking a reliable guest for a deposit: assume ~10% of them get annoyed
# and book elsewhere, losing 0.10 x 265 = ~EUR 25 on average.
COST_FALSE_ALARM = 25           # false positive: deposit asked, guest would have come


def business_cost(y_true, prob, threshold):
    """Total euros lost when we ask for a deposit if P(cancel) >= threshold."""
    y_true = np.asarray(y_true)
    ask_deposit = np.asarray(prob) >= threshold
    missed = np.sum((~ask_deposit) & (y_true == 1))       # false negatives
    false_alarms = np.sum(ask_deposit & (y_true == 0))    # false positives
    return missed * COST_MISSED_CANCELLATION + false_alarms * COST_FALSE_ALARM


def best_threshold(y_true, prob):
    """Try thresholds 0.05, 0.06, ..., 0.95 and return the cheapest one."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [business_cost(y_true, prob, t) for t in thresholds]
    i = int(np.argmin(costs))
    return float(thresholds[i]), float(costs[i])