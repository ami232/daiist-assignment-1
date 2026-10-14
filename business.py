"""
The operational decision the forecast feeds, expressed as numbers.

Scenario (REPORT.md): the Capital Bikeshare operations team receives the
day-ahead hourly forecast and must decide, hour by hour, whether to put an
extra rebalancing crew on the street. A crew is needed when demand is high
enough that popular stations empty out and rides are lost.

* SURGE_THRESHOLD_RIDES: an hour with at least this many rides counts as a
  surge hour that needs a crew. 400 rides/hour is roughly the 90th
  percentile of hourly demand in the training window, i.e. the top-decile
  hours where the system runs short of bikes at commuter stations.
* COST_MISSED_SURGE: cost of NOT dispatching a crew in a real surge hour
  (lost rides at empty stations, refunds, goodwill). Assumed 150 USD.
* COST_UNNEEDED_CREW: cost of dispatching a crew in an hour that turned out
  not to be a surge (van + two staff for an hour). Assumed 60 USD.

Because a missed surge is ~2.5x more expensive than an unneeded crew, the
cost-minimising forecast threshold is *lower* than the surge threshold
itself: it pays to dispatch when the forecast is merely close to 400.
The dashboard lets the user move the forecast threshold and the two costs
and watch the confusion matrix and total cost change.
"""

from __future__ import annotations

import numpy as np

SURGE_THRESHOLD_RIDES = 400
COST_MISSED_SURGE = 150.0
COST_UNNEEDED_CREW = 60.0


def surge_decision(
    y_true_counts: np.ndarray,
    y_pred_counts: np.ndarray,
    forecast_threshold: float,
    surge_threshold: float = SURGE_THRESHOLD_RIDES,
    cost_missed: float = COST_MISSED_SURGE,
    cost_unneeded: float = COST_UNNEEDED_CREW,
) -> dict:
    """Confusion matrix and cost of the dispatch rule "crew if forecast >= t".

    Positive class = surge hour (actual demand >= surge_threshold).
    """
    actual = np.asarray(y_true_counts) >= surge_threshold
    dispatch = np.asarray(y_pred_counts) >= forecast_threshold
    tp = int((dispatch & actual).sum())
    fp = int((dispatch & ~actual).sum())
    fn = int((~dispatch & actual).sum())
    tn = int((~dispatch & ~actual).sum())
    cost = fn * cost_missed + fp * cost_unneeded
    n_pos = tp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / n_pos if n_pos else 0.0,
        "total_cost": float(cost),
        "cost_per_hour": float(cost / len(actual)) if len(actual) else 0.0,
        "n_surge_hours": int(n_pos),
    }


def cost_curve(y_true_counts, y_pred_counts, thresholds, **kwargs) -> list[dict]:
    """Decision cost for a range of forecast thresholds."""
    return [
        {"forecast_threshold": float(t), **surge_decision(y_true_counts, y_pred_counts, t, **kwargs)}
        for t in thresholds
    ]


def best_threshold(y_true_counts, y_pred_counts, thresholds=None, **kwargs) -> dict:
    if thresholds is None:
        thresholds = np.arange(100, 801, 10)
    curve = cost_curve(y_true_counts, y_pred_counts, thresholds, **kwargs)
    return min(curve, key=lambda r: r["total_cost"])
