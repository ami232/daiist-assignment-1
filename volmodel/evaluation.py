"""Metrics, baselines, the business cost of each error, and the overlay backtest."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from . import config

THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


# Business cost ------------------------------------------------------------------------------

def false_alarm_cost_bps(train: pd.DataFrame) -> float:
    """Cost of cutting exposure in a month where vol did not rise, measured on the training block.

    A false alarm gives up HAIRCUT of the fund's next-month return in calm months.
    """
    calm = train.loc[train["vol_rises"] == 0, "fund_fwd_return_21d"]
    return float(config.HAIRCUT * calm.mean() * 1e4)


def cost_evidence(train: pd.DataFrame) -> dict:
    """The numbers behind the cost assumption: why a miss is priced as tail risk, not average P&L."""
    fund = train["fund_fwd_return_21d"]
    rises = train["vol_rises"] == 1
    worst = fund <= fund.quantile(0.05)
    return {
        "fund_mean_return_vol_rises": float(fund[rises].mean()),
        "fund_mean_return_vol_calm": float(fund[~rises].mean()),
        "haircut_saving_bps_when_vol_rises": float(-config.HAIRCUT * fund[rises].mean() * 1e4),
        "share_of_worst_5pct_months_in_vol_rises": float(rises[worst].mean()),
        "prob_month_below_minus5_vol_rises": float((fund[rises] < -0.05).mean()),
        "prob_month_below_minus5_vol_calm": float((fund[~rises] < -0.05).mean()),
        "prob_month_below_minus5_vol_rises_with_haircut": float(((1 - config.HAIRCUT) * fund[rises] < -0.05).mean()),
    }


def business_cost(y: np.ndarray, decision: np.ndarray, c_fp: float, miss_multiple: float) -> dict:
    """Total and per-decision cost in bps. A miss costs miss_multiple times a false alarm."""
    y = np.asarray(y).astype(bool)
    decision = np.asarray(decision).astype(bool)
    fp = int((decision & ~y).sum())
    fn = int((~decision & y).sum())
    total = fp * c_fp + fn * c_fp * miss_multiple
    return {"false_alarms": fp, "misses": fn, "total_cost_bps": float(total), "cost_per_decision_bps": float(total / len(y))}


def best_threshold(y: np.ndarray, p: np.ndarray, c_fp: float, miss_multiple: float) -> float:
    """The probability cut-off that minimizes total business cost on the given block."""
    costs = [business_cost(y, p >= t, c_fp, miss_multiple)["total_cost_bps"] for t in THRESHOLD_GRID]
    return float(THRESHOLD_GRID[int(np.argmin(costs))])


# Classification metrics -----------------------------------------------------------------------

def classification_metrics(y: np.ndarray, p: np.ndarray | None, decision: np.ndarray, c_fp: float,
                           miss_multiple: float, score: np.ndarray | None = None) -> dict:
    """Threshold-free metrics (AUC, log loss, Brier) plus decision metrics and business cost.

    `score` lets a rule without probabilities still report AUC on its underlying signal.
    """
    y = np.asarray(y).astype(int)
    decision = np.asarray(decision).astype(bool)
    tp = int((decision & (y == 1)).sum())
    fp = int((decision & (y == 0)).sum())
    tn = int((~decision & (y == 0)).sum())
    fn = int((~decision & (y == 1)).sum())
    ranking = p if p is not None else score
    out = {
        "auc": float(roc_auc_score(y, ranking)) if ranking is not None and len(np.unique(ranking)) > 1 else 0.5,
        "log_loss": float(log_loss(y, p)) if p is not None else None,
        "brier": float(brier_score_loss(y, p)) if p is not None else None,
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn),
        "share_flagged": float(decision.mean()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    out.update(business_cost(y, decision, c_fp, miss_multiple))
    return out


# Baselines ------------------------------------------------------------------------------------

def mean_reversion_rule(df: pd.DataFrame) -> np.ndarray:
    """The PM's rule of thumb: vol will rise if this month's vol is below its 1-year average."""
    return (df["rv_21d"] < df["rv_1y"]).to_numpy()


def mean_reversion_score(df: pd.DataFrame) -> np.ndarray:
    """Continuous version of the same rule, so it can report an AUC: how far vol sits below its 1-year level."""
    return -(df["log_rv_21d_vs_1y"]).to_numpy()


# Agreement between implementations --------------------------------------------------------------

def agreement(probs: dict, decisions: dict) -> dict:
    names = list(probs)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = np.abs(probs[a] - probs[b])
            out[f"{a}_vs_{b}"] = {
                "max_abs_prob_diff": float(diff.max()),
                "mean_abs_prob_diff": float(diff.mean()),
                "decision_disagreements": int((decisions[a] != decisions[b]).sum()),
            }
    return out


# Overlay backtest -----------------------------------------------------------------------------

def overlay_backtest(f: pd.DataFrame, block: str, decision: pd.Series) -> dict:
    """Monthly-rebalanced vol-target fund, with and without the model's haircut, over one block.

    Rebalancing every 21 trading days keeps decisions non-overlapping: one sizing decision per
    month, held for the month the label describes. S&P price return only, no dividends,
    financing or trading costs.
    """
    dates = f.index[f["split"] == block]
    rebalance = dates[:: config.HORIZON]
    daily = f["daily_return"]
    base, overlay, rows = [], [], []
    for t in rebalance:
        pos = f.index.get_loc(t)
        window = daily.iloc[pos + 1: pos + 1 + config.HORIZON]
        lev = float(f["leverage"].iloc[pos])
        cut = bool(decision.loc[t])
        base.append(lev * window)
        overlay.append(lev * (1 - config.HAIRCUT * cut) * window)
        rows.append({"date": t, "cut": cut, "vol_rises": f["vol_rises"].iloc[pos]})
    base, overlay = pd.concat(base), pd.concat(overlay)

    def stats(r: pd.Series) -> dict:
        equity = (1 + r).cumprod()
        monthly = [(1 + r.iloc[i: i + config.HORIZON]).prod() - 1 for i in range(0, len(r), config.HORIZON)]
        years = len(r) / config.TRADING_DAYS
        return {
            "annual_return": float(equity.iloc[-1] ** (1 / years) - 1),
            "annual_vol": float(r.std() * np.sqrt(config.TRADING_DAYS)),
            "sharpe": float(r.mean() / r.std() * np.sqrt(config.TRADING_DAYS)),
            "max_drawdown": float((equity / equity.cummax() - 1).min()),
            "worst_month": float(min(monthly)),
            "months_below_minus5": int(sum(m < -0.05 for m in monthly)),
        }

    return {
        "vol_target_only": stats(base),
        "with_overlay": stats(overlay),
        "months": len(rebalance),
        "months_cut": int(sum(r["cut"] for r in rows)),
        "equity": pd.DataFrame({"vol_target_only": (1 + base).cumprod(), "with_overlay": (1 + overlay).cumprod()}),
    }
