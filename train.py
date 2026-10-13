"""
Assignment 1 — training pipeline.

Scenario: an analyst at a VC fund decides which inbound startups get a
partner meeting. Model: logistic regression predicting P(acquired) vs
closed, trained three ways (scikit-learn, manual PyTorch, nn.Module +
torch.optim) on one shared split, compared to a naive baseline.

Run with:  uv run python main.py train
Writes everything the dashboard needs to ./artifacts/.

The four framing decisions live in CONFIG below. Every value there is a
choice you need to make and justify in REPORT.md — experiments.py runs the
alternatives side by side to help you decide.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models import (fit_manual, fit_standard, predict_manual,
                    predict_standard)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "startup_data.csv"
ART = ROOT / "artifacts"

# =============================================================================
# DECISIONS — set each one and be ready to defend it.
# =============================================================================
CONFIG = {
    # (1) What does the analyst know at the meeting?
    #   "pre_funding"      : static facts only (location, sector)
    #   "post_first_round" : + timing of the first round
    #   "lifetime"         : + all funding/traction columns, which are totals
    #                        up to ~2013 and so include the future (leaky)
    "meeting_stage": "pre_funding",

    # (2) Split. "random" = stratified shuffle; "time" = oldest 60% of
    #     companies (by founding date) train, next 20% validate, newest 20% test.
    "split": "time",
    # Drop cohorts founded after this year (too young to have a fair label by
    # 2013 — right-censoring). None keeps everyone.
    "max_founded_year": 2009,

    # 46 rows were funded "before" they were founded (negative ages).
    #   "drop" | "clip" (set to 0) | "flag_and_clip" (clip + 0/1 indicator)
    "negative_age": "drop",

    # (3) Cost of each mistake, in "partner-meeting units".
    "cost_wasted_meeting": 1.0,   # false positive: met a startup that closed
    "cost_missed_winner": 5.0,    # false negative: passed on one that got acquired

    # (4) Engineered-feature switches (only those valid for the stage are used).
    "features": {
        "group_rare_categories": True,  # categories with <20 train rows -> "other"
        "founded_year": False,          # risky: encodes the censoring trend
        "log_funding": True,            # lifetime only
        "funding_per_round": True,      # lifetime only
        "milestone_missing_flag": True, # lifetime only
    },

    # Tuning — kept deliberately small (brief: regularisation + lr only).
    "C_grid": [0.01, 0.1, 1.0, 10.0],
    "lr_grid": [0.1, 0.5, 1.0],
    "epochs": 4000,
    "seed": 42,
}


# =============================================================================
# Data loading and cleaning
# =============================================================================
AGE_COLS = ["age_first_funding_year", "age_last_funding_year",
            "age_first_milestone_year", "age_last_milestone_year"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    for c in ["founded_at", "first_funding_at", "last_funding_at", "closed_at"]:
        df[c] = pd.to_datetime(df[c])
    # Target. `labels` is an exact copy of it and `closed_at` is only filled
    # for closed companies — both are dropped below as pure leakage.
    df["acquired"] = (df["status"] == "acquired").astype(int)
    return df


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    if cfg["max_founded_year"] is not None:
        df = df[df["founded_at"].dt.year <= cfg["max_founded_year"]]

    neg = (df[AGE_COLS] < 0).any(axis=1)
    if cfg["negative_age"] == "drop":
        df = df[~neg]
    else:
        if cfg["negative_age"] == "flag_and_clip":
            df["negative_age_flag"] = neg.astype(int)
        df[AGE_COLS] = df[AGE_COLS].clip(lower=0)
    return df.reset_index(drop=True)


# =============================================================================
# Feature engineering — which columns are allowed depends on meeting_stage
# =============================================================================
STAGES = ["pre_funding", "post_first_round", "lifetime"]


def build_features(df: pd.DataFrame, cfg: dict):
    """Return (feature frame, numeric cols, binary cols, categorical cols)."""
    f = cfg["features"]
    level = STAGES.index(cfg["meeting_stage"])
    out = pd.DataFrame(index=df.index)
    num, binary, cat = [], [], []

    # --- Static facts: known at first contact -------------------------------
    out["state"] = np.select(
        [df.is_CA == 1, df.is_NY == 1, df.is_MA == 1, df.is_TX == 1],
        ["CA", "NY", "MA", "TX"], "other")
    out["category"] = df["category_code"].fillna("unknown")
    cat += ["state", "category"]
    if f["founded_year"]:
        out["founded_year"] = df["founded_at"].dt.year
        num.append("founded_year")

    # --- After the first round ----------------------------------------------
    if level >= 1:
        out["years_to_first_funding"] = df["age_first_funding_year"]
        num.append("years_to_first_funding")
        if "negative_age_flag" in df:
            out["negative_age_flag"] = df["negative_age_flag"]
            binary.append("negative_age_flag")

    # --- Lifetime (leaky: includes rounds after any realistic meeting) ------
    if level >= 2:
        funding = df["funding_total_usd"].astype(float)
        rounds = df["funding_rounds"].clip(lower=1)
        if f["log_funding"]:
            out["log_funding_total"] = np.log1p(funding)
        else:
            out["funding_total_usd"] = funding
        num.append(out.columns[-1])
        if f["funding_per_round"]:
            out["log_funding_per_round"] = np.log1p(funding / rounds)
            num.append("log_funding_per_round")
        out["years_first_to_last_funding"] = (
            df["age_last_funding_year"] - df["age_first_funding_year"])
        out["years_to_first_milestone"] = df["age_first_milestone_year"]  # NaN imputed later
        for c in ["funding_rounds", "relationships", "milestones", "avg_participants"]:
            out[c] = df[c]
        num += ["years_first_to_last_funding", "years_to_first_milestone",
                "funding_rounds", "relationships", "milestones", "avg_participants"]
        for c in ["has_VC", "has_angel", "has_roundA", "has_roundB",
                  "has_roundC", "has_roundD", "is_top500"]:
            out[c] = df[c]
            binary.append(c)
        if f["milestone_missing_flag"]:
            out["no_milestone"] = df["age_first_milestone_year"].isna().astype(int)
            binary.append("no_milestone")

    return out, num, binary, cat


def make_preprocessor(num, binary, cat, cfg) -> ColumnTransformer:
    # Fit on the training rows ONLY (imputation medians, scaling, which
    # categories are "rare") so nothing leaks from val/test.
    min_freq = 20 if cfg["features"]["group_rare_categories"] else None
    ohe = OneHotEncoder(
        min_frequency=min_freq, sparse_output=False,
        handle_unknown="infrequent_if_exist" if min_freq else "ignore")
    parts = [("cat", ohe, cat)]
    if num:
        parts.append(("num", make_pipeline(SimpleImputer(strategy="median"),
                                           StandardScaler()), num))
    if binary:
        parts.append(("bin", "passthrough", binary))
    return ColumnTransformer(parts, verbose_feature_names_out=True)


# =============================================================================
# Split
# =============================================================================
def split_indices(df: pd.DataFrame, cfg: dict):
    idx = np.arange(len(df))
    if cfg["split"] == "random":
        tr, tmp = train_test_split(idx, test_size=0.4, random_state=cfg["seed"],
                                   stratify=df["acquired"])
        va, te = train_test_split(tmp, test_size=0.5, random_state=cfg["seed"],
                                  stratify=df["acquired"].iloc[tmp])
        return tr, va, te
    order = df.sort_values(["founded_at", "id"], kind="stable").index.to_numpy()
    a, b = int(0.6 * len(df)), int(0.8 * len(df))
    return order[:a], order[a:b], order[b:]


# =============================================================================
# Evaluation helpers
# =============================================================================
def cost_at(y, p, thr, cfg):
    meet = p >= thr
    fp = int(((meet == 1) & (y == 0)).sum())
    fn = int(((meet == 0) & (y == 1)).sum())
    return cfg["cost_wasted_meeting"] * fp + cfg["cost_missed_winner"] * fn


def best_threshold(y, p, cfg):
    grid = np.round(np.linspace(0, 1, 101), 2)
    costs = [cost_at(y, p, t, cfg) for t in grid]
    return float(grid[int(np.argmin(costs))])


def evaluate(y, p, thr, cfg) -> dict:
    meet = (p >= thr).astype(int)
    tp = int(((meet == 1) & (y == 1)).sum())
    fp = int(((meet == 1) & (y == 0)).sum())
    fn = int(((meet == 0) & (y == 1)).sum())
    tn = int(((meet == 0) & (y == 0)).sum())
    constant = len(np.unique(p)) == 1
    return {
        "roc_auc": 0.5 if constant else roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "threshold": thr,
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "meetings_taken": tp + fp,
        "cost": cost_at(y, p, thr, cfg),
        "cost_per_100": 100 * cost_at(y, p, thr, cfg) / len(y),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# =============================================================================
# Full run
# =============================================================================
def run(cfg: dict, torch_models: bool = True, verbose: bool = True) -> dict:
    torch.manual_seed(cfg["seed"])
    df = clean(load_raw(), cfg)
    feats, num, binary, cat = build_features(df, cfg)
    y = df["acquired"].to_numpy()
    tr, va, te = split_indices(df, cfg)

    pre = make_preprocessor(num, binary, cat, cfg)
    X_tr = pre.fit_transform(feats.iloc[tr])
    X_va, X_te = pre.transform(feats.iloc[va]), pre.transform(feats.iloc[te])
    names = list(pre.get_feature_names_out())
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    # --- Tune C on validation log loss (sklearn is fast and exact) ---------
    val_ll = {}
    for C in cfg["C_grid"]:
        m = LogisticRegression(C=C, max_iter=10000, tol=1e-10)
        m.fit(X_tr, y_tr)
        val_ll[C] = log_loss(y_va, m.predict_proba(X_va)[:, 1])
    C = min(val_ll, key=val_ll.get)

    sk = LogisticRegression(C=C, max_iter=10000, tol=1e-10).fit(X_tr, y_tr)
    probs = {"sklearn": {"val": sk.predict_proba(X_va)[:, 1],
                         "test": sk.predict_proba(X_te)[:, 1]}}
    out = {"cfg": cfg, "C": C, "val_log_loss_by_C": val_ll,
           "n": {"train": len(tr), "val": len(va), "test": len(te)},
           "feature_names": names}

    if torch_models:
        # --- Pick the learning rate by lowest final training objective ------
        lr_obj = {}
        for lr in cfg["lr_grid"]:
            _, h = fit_manual(X_tr, y_tr, C, lr, cfg["epochs"])
            lr_obj[lr] = h[-1] if np.isfinite(h[-1]) else np.inf
        lr = min(lr_obj, key=lr_obj.get)
        manual, h_manual = fit_manual(X_tr, y_tr, C, lr, cfg["epochs"])
        standard, h_std = fit_standard(X_tr, y_tr, C, lr, cfg["epochs"])
        probs["manual_torch"] = {"val": predict_manual(manual, X_va),
                                 "test": predict_manual(manual, X_te)}
        probs["standard_torch"] = {"val": predict_standard(standard, X_va),
                                   "test": predict_standard(standard, X_te)}
        out.update(lr=lr, final_train_objective_by_lr=lr_obj,
                   manual=manual, standard=standard,
                   history={"manual_torch": h_manual, "standard_torch": h_std})

    # --- Naive baseline: always predict the training acquisition rate ------
    base = y_tr.mean()
    probs["baseline"] = {"val": np.full(len(va), base), "test": np.full(len(te), base)}

    # --- Threshold chosen on VALIDATION cost, then reported on test --------
    results = {}
    for name, p in probs.items():
        thr = best_threshold(y_va, p["val"], cfg)
        results[name] = evaluate(y_te, p["test"], thr, cfg)
    out.update(results=results, probs=probs, sk=sk, pre=pre, df=df, feats=feats,
               split={"train": tr, "val": va, "test": te}, y=y)

    # --- Do the three implementations agree? -------------------------------
    if torch_models:
        w_sk = np.r_[sk.intercept_, sk.coef_.ravel()]
        w_man = np.r_[manual["b"].numpy(), manual["w"].numpy()]
        w_std = np.r_[standard.linear.bias.detach().numpy(),
                      standard.linear.weight.detach().numpy().ravel()]
        out["agreement"] = {
            "max_abs_prob_diff_manual_vs_sklearn":
                float(np.abs(probs["manual_torch"]["test"] - probs["sklearn"]["test"]).max()),
            "max_abs_prob_diff_standard_vs_sklearn":
                float(np.abs(probs["standard_torch"]["test"] - probs["sklearn"]["test"]).max()),
            "max_abs_coef_diff_manual_vs_sklearn": float(np.abs(w_man - w_sk).max()),
            "max_abs_coef_diff_standard_vs_sklearn": float(np.abs(w_std - w_sk).max()),
        }
        out["coefficients"] = pd.DataFrame(
            {"feature": ["intercept"] + names, "sklearn": w_sk,
             "manual_torch": w_man, "standard_torch": w_std})

    if verbose:
        print_summary(out)
    return out


def print_summary(out):
    cfg = out["cfg"]
    print(f"\nStage={cfg['meeting_stage']}  split={cfg['split']}  "
          f"max_year={cfg['max_founded_year']}  neg_age={cfg['negative_age']}")
    print(f"Rows: {out['n']}   chosen C={out['C']}   lr={out.get('lr')}")
    cols = ["roc_auc", "pr_auc", "log_loss", "threshold", "precision",
            "recall", "meetings_taken", "cost_per_100"]
    print(pd.DataFrame(out["results"]).T[cols].round(4).to_string())
    if "agreement" in out:
        print("Agreement:", {k: f"{v:.2e}" for k, v in out["agreement"].items()})


def save_artifacts(out):
    ART.mkdir(exist_ok=True)
    joblib.dump(out["pre"], ART / "preprocessor.joblib")
    joblib.dump(out["sk"], ART / "sklearn_logreg.joblib")
    torch.save(out["manual"], ART / "manual_torch.pt")
    torch.save(out["standard"].state_dict(), ART / "standard_torch.pt")

    # Engineered (unscaled) features + target + split, for the dashboard's
    # distribution views and for re-scoring through the saved preprocessor.
    frame = out["feats"].copy()
    frame["acquired"] = out["y"]
    frame["split"] = ""
    for name, idx in out["split"].items():
        frame.loc[idx, "split"] = name
    frame["name"] = out["df"]["name"]
    frame["founded_year_raw"] = out["df"]["founded_at"].dt.year
    frame.to_csv(ART / "features.csv", index=False)

    out["coefficients"].to_csv(ART / "coefficients.csv", index=False)
    pd.DataFrame(out["history"]).to_csv(ART / "training_history.csv", index=False)

    def clean_json(o):
        if isinstance(o, dict):
            return {str(k): clean_json(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    json.dump(clean_json({
        "config": out["cfg"], "C": out["C"], "lr": out["lr"], "n": out["n"],
        "val_log_loss_by_C": out["val_log_loss_by_C"],
        "final_train_objective_by_lr": out["final_train_objective_by_lr"],
        "results": out["results"], "agreement": out["agreement"],
        "n_features": len(out["feature_names"]),
        "baseline_rate": float(out["probs"]["baseline"]["test"][0]),
    }), open(ART / "metrics.json", "w"), indent=2)
    print(f"\nSaved artifacts to {ART}/")


if __name__ == "__main__":
    save_artifacts(run(CONFIG))
