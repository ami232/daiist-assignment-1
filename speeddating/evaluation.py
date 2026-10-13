"""Metrics, cost rules and seating plans shared by training and the dashboard.

Recovered from the helper definitions in the supplied full walkthrough notebook.
This module defines functions only; importing it never trains a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

SEED = 42

COST_FP = 5.0


COST_FN = 20.0


def cost_threshold(c_fp: float = COST_FP, c_fn: float = COST_FN) -> float:
    return c_fp / (c_fp + c_fn)


def confusion(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred, y = p >= t, y.astype(bool)
    return {"tp": int((pred & y).sum()), "fp": int((pred & ~y).sum()),
            "fn": int((~pred & y).sum()), "tn": int((~pred & ~y).sum())}


def business_cost(cm: dict, c_fp: float = COST_FP, c_fn: float = COST_FN) -> float:
    return cm["fp"] * c_fp + cm["fn"] * c_fn


def evaluate(y: np.ndarray, p: np.ndarray, t: float, c_fp: float = COST_FP, c_fn: float = COST_FN) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    cm = confusion(y, p, t)
    flagged = cm["tp"] + cm["fp"]
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p), "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p), "accuracy": (cm["tp"] + cm["tn"]) / len(y),
        "precision": cm["tp"] / flagged if flagged else np.nan,
        "recall": cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) else np.nan,
        "share_scheduled": flagged / len(y), "cost_eur": business_cost(cm, c_fp, c_fn), **cm,
    }


def cost_curve(y: np.ndarray, p: np.ndarray, c_fp: float, c_fn: float, thresholds=None) -> pd.DataFrame:
    thresholds = np.linspace(0, 1, 201) if thresholds is None else thresholds
    return pd.DataFrame([{"threshold": t, "cost_eur": business_cost(confusion(y, p, t), c_fp, c_fn),
                          **confusion(y, p, t)} for t in thresholds])


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 8) -> pd.DataFrame:
    """Pairs sorted into equal-size bins of predicted probability: mean prediction vs observed match rate."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin").agg(mean_pred=("p", "mean"), match_rate=("y", "mean"), n=("y", "size"))
    g["se"] = np.sqrt(g["match_rate"] * (1 - g["match_rate"]) / g["n"])
    return g.reset_index()


def mean_auc_within(df: pd.DataFrame, p: np.ndarray, key: str) -> float:
    """Average AUC inside each group (event, woman or man). Inside one woman's list it asks: does the
    model rank the men she said yes to (and who said yes back) above the others?"""
    return float(np.mean([roc_auc_score(g["match"], g["p"]) for _, g in df.assign(p=p).groupby(key)
                          if 0 < g["match"].mean() < 1]))


def bootstrap_ci(y: np.ndarray, preds: dict, n_boot: int = 1000, seed: int = SEED) -> dict:
    """Pair-level bootstrap of test AUC. Pairs of the same person are not independent,
    so these intervals are on the narrow (optimistic) side."""
    rng = np.random.default_rng(seed)
    aucs = {k: [] for k in preds}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        for k, v in preds.items():
            aucs[k].append(roc_auc_score(y[idx], v[idx]))
    return {"auc_ci": {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in aucs.items()}}


def seating_plan(event: pd.DataFrame, score: np.ndarray, share: float) -> np.ndarray:
    """Boolean mask over the event's rows: the dates with the highest total score, at most k per person."""
    women = pd.factorize(event["iid_W"])[0]
    men = pd.factorize(event["iid_M"])[0]
    n, n_w, n_m = len(event), women.max() + 1, men.max() + 1
    A = coo_matrix((np.ones(2 * n), (np.concatenate([women, n_w + men]), np.concatenate([np.arange(n), np.arange(n)]))),
                   shape=(n_w + n_m, n))
    b = np.concatenate([np.full(n_w, int(np.ceil(share * n_m))), np.full(n_m, int(np.ceil(share * n_w)))])
    res = linprog(-np.asarray(score, dtype=float), A_ub=A, b_ub=b, bounds=(0, 1), method="highs")
    if not res.success:
        raise RuntimeError(f"Seating plan failed: {res.message}")
    return res.x > 0.5


def plan_events(pairs: pd.DataFrame, p: np.ndarray, share: float = 0.5, n_random: int = 100, seed: int = SEED) -> pd.DataFrame:
    """Per event: matches kept by the model's plan, by random plans of the same size, and by a perfect plan."""
    rng = np.random.default_rng(seed)
    rows = []
    for w, ev in pairs.assign(_p=p).groupby("wave"):
        y = ev["match"].to_numpy()
        chosen = seating_plan(ev, ev["_p"].to_numpy(), share)
        rand = [int(y[seating_plan(ev, rng.random(len(ev)), share)].sum()) for _ in range(n_random)]
        perfect = y[seating_plan(ev, y + rng.random(len(ev)) * 1e-3, share)].sum()
        rows.append({"wave": int(w), "pairs": len(ev), "all_matches": int(y.sum()), "dates_held": int(chosen.sum()),
                     "model_plan": int(y[chosen].sum()), "random_plan": float(np.mean(rand)),
                     "perfect_plan": int(perfect), "random_draws": rand})
    return pd.DataFrame(rows)
