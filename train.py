"""
Training pipeline: SNL cast renewal.

One row = one cast member in one season. Target = 1 if that person is in the
cast again the following season, 0 otherwise.

Trains the same logistic regression three ways (scikit-learn, manual PyTorch
loop, nn.Module + optim) on the same time-based split and saves everything
app.py needs into models/ and artifacts/.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "snl"
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"

SEED = 42

# Time-based split by season. Seasons are in chronological order, so the
# model is always evaluated on seasons that come after the ones it learned from.
TRAIN_LAST_SEASON = 30
VAL_LAST_SEASON = 37
# Last season in the data (46) has no "next season", so it can't be labelled.
LAST_LABELLED_SEASON = 45
# First season predicted in the walk-forward threshold search (needs enough
# past seasons to train on).
FIRST_OOT_SEASON = 21

C_GRID = [0.01, 0.1, 1.0, 10.0]  # sklearn regularization strength (inverse)
LR_GRID = [0.05, 0.5]            # learning rates tried for the PyTorch versions
EPOCHS = 5000


# --------------------------------------------------------------------------
# Data loading and target
# --------------------------------------------------------------------------

def load_tables():
    casts = pd.read_csv(DATA / "casts.csv")
    appearances = pd.read_csv(DATA / "appearances.csv")
    titles = pd.read_csv(DATA / "titles.csv")
    episodes = pd.read_csv(DATA / "episodes.csv")
    return casts, appearances, titles, episodes


def build_target(casts):
    casts = casts.sort_values(["aid", "sid"]).copy()
    next_sid = casts.groupby("aid")["sid"].shift(-1)
    # Someone who leaves and comes back years later counts as not renewed.
    casts["renewed"] = (next_sid == casts["sid"] + 1).astype(int)
    casts = casts[casts["sid"] <= LAST_LABELLED_SEASON]

    # People who already left during the season (last_epid is filled in) are
    # dropped: by the end-of-season decision there is nothing left to predict.
    casts = casts[casts["last_epid"].isna()]
    return casts.reset_index(drop=True)


# --------------------------------------------------------------------------
# Features
# Only information available at the end of the season the row describes:
# nothing from later seasons, so the time-based split stays honest.
# --------------------------------------------------------------------------

def per_episode(df, counts, name):
    df = df.merge(counts.rename(name), on=["aid", "sid"], how="left")
    df[name] = df[name].fillna(0) / df["n_episodes"].clip(lower=1)
    return df


def build_features(df, appearances, titles, episodes):
    df = df.sort_values(["aid", "sid"]).copy()

    # --- Role / status -----------------------------------------------------
    df["featured"] = df["featured"].astype(int)
    df["update_anchor"] = df["update_anchor"].astype(int)
    # Joined after the season premiere (first_epid is only filled in then).
    df["joined_mid_season"] = df["first_epid"].notna().astype(int)

    # --- Tenure ------------------------------------------------------------
    # Seasons already completed before this one (uses earlier rows only).
    df["prior_seasons"] = df.groupby("aid").cumcount()
    # Still a featured player after 2+ seasons: never got promoted.
    df["stuck_featured"] = ((df["featured"] == 1)
                            & (df["prior_seasons"] >= 2)).astype(int)

    # --- Screen time -------------------------------------------------------
    cast = appearances[appearances["capacity"] == "cast"].merge(
        titles[["tid", "category"]], on="tid", how="left")
    keys = ["aid", "sid"]

    df = per_episode(df, cast.groupby(keys).size(), "apps_per_episode")
    df = per_episode(df, cast[cast["category"] == "Cold Opening"]
                     .groupby(keys).size(), "cold_open_per_episode")
    df = per_episode(df, cast[cast["category"] == "Weekend Update"]
                     .groupby(keys).size(), "update_per_episode")

    # Distinct recurring characters and impressions played that season.
    df = df.merge(cast.groupby(keys)["charid"].nunique().rename("n_characters"),
                  on=keys, how="left")
    df = df.merge(cast.groupby(keys)["impid"].nunique().rename("n_impressions"),
                  on=keys, how="left")
    df[["n_characters", "n_impressions"]] = (
        df[["n_characters", "n_impressions"]].fillna(0))
    # log1p: a few people have 10+ characters, most have 0-2.
    df["log_characters"] = np.log1p(df["n_characters"])
    df["log_impressions"] = np.log1p(df["n_impressions"])

    # --- Relative to the rest of that season's cast ------------------------
    # The show changed a lot in 45 years (cast size, sketches per episode), so
    # raw counts are not comparable across eras. Dividing by the season mean
    # asks "how much did this person get compared with their colleagues".
    season_mean = df.groupby("sid")["apps_per_episode"].transform("mean")
    df["apps_relative"] = df["apps_per_episode"] / season_mean
    df["cast_size"] = df.groupby("sid")["aid"].transform("size")

    # How often the show renewed people in the previous 5 seasons. Renewals
    # for season s-1 are known by the end of season s (they either came back
    # in s or not), so this only looks at the past. It lets the model follow
    # the base rate as it drifts from ~60% in the 70s to ~90% in the 2010s.
    by_season = df.groupby("sid")[TARGET].mean()
    recent = by_season.shift(1).rolling(5, min_periods=1).mean()
    df["recent_renewal_rate"] = df["sid"].map(recent)
    # Season 1 has no past seasons; 0.5 stands for "no information".
    df["recent_renewal_rate"] = df["recent_renewal_rate"].fillna(0.5)

    # Share of the person's appearances that fell in the second half of the
    # season. Below 0.5 = they were used less as the season went on, which
    # could be an early sign of being phased out. Still only uses data from
    # the season itself, so it is known at the end-of-season decision.
    eps = episodes[["epid", "sid", "epno"]].copy()
    eps["season_len"] = eps.groupby("sid")["epno"].transform("max")
    eps["second_half"] = eps["epno"] > eps["season_len"] / 2
    halves = cast.merge(eps[["epid", "second_half"]], on="epid", how="left")
    share = halves.groupby(keys)["second_half"].mean().rename("second_half_share")
    df = df.merge(share, on=keys, how="left")
    df["second_half_share"] = df["second_half_share"].fillna(0.5)

    # Change vs. their own previous season (0 in a first season).
    prev = df.groupby("aid")["apps_relative"].shift(1)
    df["apps_relative_change"] = (df["apps_relative"] - prev).fillna(0)

    return df.reset_index(drop=True)


# Grouped so the ablation in main() can drop one group at a time.
FEATURE_GROUPS = {
    "role": ["featured", "update_anchor", "joined_mid_season"],
    "tenure": ["prior_seasons", "stuck_featured"],
    "screen_time": ["apps_relative", "apps_relative_change",
                    "cold_open_per_episode", "update_per_episode",
                    "second_half_share"],
    "characters": ["log_characters", "log_impressions"],
    "context": ["cast_size", "recent_renewal_rate"],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
TARGET = "renewed"

# --------------------------------------------------------------------------
# Business costs (assumptions, see REPORT.md)
#
# The producer uses the model at the end of each season to flag cast members
# who are likely NOT to return, so replacements can be scouted in time.
#   - missed departure: model says "renewed", person leaves -> a hole in the
#     cast with no replacement lined up.
#   - false alarm: model says "leaves", person stays -> some wasted scouting
#     time and an awkward conversation.
# A missed departure is assumed to be 3x worse. Change these and rerun.
# --------------------------------------------------------------------------
COST_MISSED_DEPARTURE = 3.0
COST_FALSE_ALARM = 1.0


# --------------------------------------------------------------------------
# Split and scaling
# --------------------------------------------------------------------------

def split(df):
    train = df[df["sid"] <= TRAIN_LAST_SEASON]
    val = df[(df["sid"] > TRAIN_LAST_SEASON) & (df["sid"] <= VAL_LAST_SEASON)]
    test = df[df["sid"] > VAL_LAST_SEASON]
    return train, val, test


def to_arrays(part, scaler, features=None):
    features = features or FEATURES
    X = scaler.transform(part[features].to_numpy(dtype=float))
    y = part[TARGET].to_numpy(dtype=float)
    return X, y


# --------------------------------------------------------------------------
# The three implementations
#
# sklearn minimizes   0.5*||w||^2 + C * sum(logloss)
# Dividing by C*n gives the same optimum as
#                     mean(logloss) + ||w||^2 / (2*C*n)
# which is what the two PyTorch versions minimize. The bias is not penalized
# in any of them, so all three should land on (almost) the same coefficients.
# --------------------------------------------------------------------------

def fit_sklearn(X, y, C):
    model = LogisticRegression(C=C, max_iter=10_000)
    model.fit(X, y)
    return model


def fit_manual_torch(X, y, C, lr, epochs=EPOCHS):
    torch.manual_seed(SEED)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    n, d = Xt.shape
    l2 = 1.0 / (2 * C * n)

    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    losses = []
    for _ in range(epochs):
        logits = Xt @ w + b
        p = torch.sigmoid(logits)
        eps = 1e-7
        bce = -(yt * torch.log(p + eps) + (1 - yt) * torch.log(1 - p + eps)).mean()
        loss = bce + l2 * (w ** 2).sum()

        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        losses.append(loss.item())

    return {"w": w.detach(), "b": b.detach()}, losses


def predict_manual(params, X):
    Xt = torch.tensor(X, dtype=torch.float32)
    return torch.sigmoid(Xt @ params["w"] + params["b"]).numpy()


class LogReg(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = torch.nn.Linear(d, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(-1)  # logits


def fit_torch_module(X, y, C, lr, epochs=EPOCHS):
    torch.manual_seed(SEED)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    n, d = Xt.shape
    l2 = 1.0 / (2 * C * n)

    model = LogReg(d)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # Not using weight_decay: it would also shrink the bias, unlike sklearn.
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt) + l2 * (model.linear.weight ** 2).sum()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return model, losses


def predict_module(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


# --------------------------------------------------------------------------
# Evaluation
#
# "Positive" for the business is a departure (renewed = 0), so recall and
# precision are reported for that class. Accuracy is kept only to show how
# little it says here: "everyone is renewed" already gets ~86% on test.
# --------------------------------------------------------------------------

def business_cost(y, p, threshold):
    """Total cost of the decisions at this threshold on P(renewed)."""
    pred_renewed = p >= threshold
    missed = int((pred_renewed & (y == 0)).sum())
    false_alarm = int((~pred_renewed & (y == 1)).sum())
    return missed * COST_MISSED_DEPARTURE + false_alarm * COST_FALSE_ALARM


def best_threshold(y, p):
    """Cheapest threshold on a grid. Ties go to the lowest threshold."""
    grid = np.round(np.linspace(0.05, 0.95, 91), 2)
    costs = [business_cost(y, p, t) for t in grid]
    return float(grid[int(np.argmin(costs))])


def evaluate(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "recall_departures": recall_score(y, pred, pos_label=0, zero_division=0),
        "precision_departures": precision_score(y, pred, pos_label=0,
                                                zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "cost": business_cost(y, p, threshold),
    }
    # AUC is undefined for a constant prediction.
    out["roc_auc"] = roc_auc_score(y, p) if np.unique(p).size > 1 else 0.5
    return {k: round(float(v), 4) for k, v in out.items()}


def ablation(train, val, C):
    """Validation AUC / log loss with all features, then without each group."""
    rows = []
    for dropped in [None] + list(FEATURE_GROUPS):
        feats = [f for f in FEATURES
                 if dropped is None or f not in FEATURE_GROUPS[dropped]]
        scaler = StandardScaler().fit(train[feats].to_numpy(dtype=float))
        X_tr, y_tr = to_arrays(train, scaler, feats)
        X_va, y_va = to_arrays(val, scaler, feats)
        p = fit_sklearn(X_tr, y_tr, C).predict_proba(X_va)[:, 1]
        rows.append({"model": "all features" if dropped is None
                     else f"without {dropped}",
                     "val_auc": round(roc_auc_score(y_va, p), 4),
                     "val_log_loss": round(log_loss(y_va, p), 4)})
    return pd.DataFrame(rows)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    MODELS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)

    casts, appearances, titles, episodes = load_tables()
    df = build_features(build_target(casts), appearances, titles, episodes)
    train, val, test = split(df)
    print(f"rows: train={len(train)} val={len(val)} test={len(test)}")
    print(f"renewal rate: train={train[TARGET].mean():.2f} "
          f"val={val[TARGET].mean():.2f} test={test[TARGET].mean():.2f}")

    # Scaler fitted on training seasons only.
    scaler = StandardScaler().fit(train[FEATURES].to_numpy(dtype=float))
    X_tr, y_tr = to_arrays(train, scaler)
    X_va, y_va = to_arrays(val, scaler)
    X_te, y_te = to_arrays(test, scaler)

    # 1) Regularization strength picked on validation log loss (sklearn).
    val_scores = {}
    for C in C_GRID:
        p = fit_sklearn(X_tr, y_tr, C).predict_proba(X_va)[:, 1]
        val_scores[C] = log_loss(y_va, p)
        print(f"C={C:<6} val log loss={val_scores[C]:.4f}")
    best_C = min(val_scores, key=val_scores.get)
    print(f"chosen C={best_C}")

    # 2) Learning rate: the one that reaches the lowest training loss.
    lr_scores = {}
    for lr in LR_GRID:
        _, losses = fit_manual_torch(X_tr, y_tr, best_C, lr)
        lr_scores[lr] = losses[-1]
        print(f"lr={lr:<5} final train loss={losses[-1]:.6f}")
    best_lr = min(lr_scores, key=lr_scores.get)
    print(f"chosen lr={best_lr}")

    # 3) Decision threshold from rolling ("walk-forward") predictions.
    #    Validation alone has only 13 departures, so a threshold picked there
    #    is very noisy. Instead, for every season s from FIRST_OOT_SEASON to
    #    the end of validation, fit on seasons < s and predict season s, as
    #    if we were standing at the end of season s-1. Pooling those
    #    out-of-time predictions gives ~100 departures to set the threshold
    #    on, and never uses test.
    oot_y, oot_p = [], []
    for s in range(FIRST_OOT_SEASON, VAL_LAST_SEASON + 1):
        past, now = df[df["sid"] < s], df[df["sid"] == s]
        sc = StandardScaler().fit(past[FEATURES].to_numpy(dtype=float))
        Xp, yp = to_arrays(past, sc)
        Xn, yn = to_arrays(now, sc)
        oot_p.append(fit_sklearn(Xp, yp, best_C).predict_proba(Xn)[:, 1])
        oot_y.append(yn)
    oot_y, oot_p = np.concatenate(oot_y), np.concatenate(oot_p)
    threshold = best_threshold(oot_y, oot_p)
    print(f"walk-forward: {len(oot_y)} rows, {int((oot_y == 0).sum())} departures, "
          f"AUC={roc_auc_score(oot_y, oot_p):.3f}")
    print(f"chosen threshold on P(renewed)={threshold} "
          f"(cost ratio {COST_MISSED_DEPARTURE}:{COST_FALSE_ALARM})")

    # 4) Final models: refit on train + val (seasons 1-37), now that C, lr and
    #    the threshold are fixed. The extra seasons are the most recent ones
    #    before test, so throwing them away would waste the most relevant
    #    data. All three implementations use exactly the same rows.
    dev = df[df["sid"] <= VAL_LAST_SEASON]
    scaler = StandardScaler().fit(dev[FEATURES].to_numpy(dtype=float))
    X_dev, y_dev = to_arrays(dev, scaler)
    X_te, y_te = to_arrays(test, scaler)

    sk = fit_sklearn(X_dev, y_dev, best_C)
    manual, manual_losses = fit_manual_torch(X_dev, y_dev, best_C, best_lr)
    module, module_losses = fit_torch_module(X_dev, y_dev, best_C, best_lr)

    # Naive baseline: always predict the renewal rate seen in seasons 1-37.
    base_rate = float(y_dev.mean())

    def all_preds(X):
        return {
            "baseline": np.full(len(X), base_rate),
            "sklearn": sk.predict_proba(X)[:, 1],
            "manual_torch": predict_manual(manual, X),
            "torch_module": predict_module(module, X),
        }

    # Test is touched once, here, with everything already decided.
    preds = all_preds(X_te)
    metrics = {"test": {
        "threshold_0.5": {m: evaluate(y_te, p, 0.5) for m, p in preds.items()},
        "chosen_threshold": {m: evaluate(y_te, p, threshold)
                             for m, p in preds.items()},
    }}

    for key in ["threshold_0.5", "chosen_threshold"]:
        print(f"\nTest set, {key}")
        print(pd.DataFrame(metrics["test"][key]).T.to_string())

    abl = ablation(train, val, best_C)
    print("\nFeature ablation (validation)")
    print(abl.to_string(index=False))

    coefs = pd.DataFrame({
        "feature": FEATURES + ["(intercept)"],
        "sklearn": np.r_[sk.coef_[0], sk.intercept_],
        "manual_torch": np.r_[manual["w"].numpy(), manual["b"].numpy()],
        "torch_module": np.r_[module.linear.weight.detach().numpy()[0],
                              module.linear.bias.detach().numpy()],
    })
    print("\nCoefficients (standardized features)")
    print(coefs.round(4).to_string(index=False))

    p_te = all_preds(X_te)
    print("\nmax |prob difference| vs sklearn on test: "
          f"manual={np.abs(p_te['manual_torch'] - p_te['sklearn']).max():.2e} "
          f"module={np.abs(p_te['torch_module'] - p_te['sklearn']).max():.2e}")

    # ---------------- Save ----------------
    with open(MODELS / "sklearn_logreg.pkl", "wb") as f:
        pickle.dump(sk, f)
    with open(MODELS / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    torch.save(manual, MODELS / "manual_torch.pt")
    torch.save(module.state_dict(), MODELS / "torch_module.pt")

    df["split"] = np.select(
        [df["sid"] <= TRAIN_LAST_SEASON, df["sid"] <= VAL_LAST_SEASON],
        ["train", "val"], "test")
    df[["aid", "sid", "split", TARGET] + FEATURES].to_csv(
        ARTIFACTS / "dataset.csv", index=False)
    coefs.to_csv(ARTIFACTS / "coefficients.csv", index=False)
    abl.to_csv(ARTIFACTS / "ablation.csv", index=False)
    pd.DataFrame({"manual_torch": manual_losses,
                  "torch_module": module_losses}).to_csv(
        ARTIFACTS / "loss_curves.csv", index_label="epoch")
    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump({
            "features": FEATURES,
            "target": TARGET,
            "baseline_rate": base_rate,
            "chosen_C": best_C,
            "chosen_lr": best_lr,
            "chosen_threshold": threshold,
            "final_model_trained_on": "train + val (seasons 1-37)",
            "cost_missed_departure": COST_MISSED_DEPARTURE,
            "cost_false_alarm": COST_FALSE_ALARM,
            "val_log_loss_by_C": {str(k): v for k, v in val_scores.items()},
            "metrics": metrics,
        }, f, indent=2)
    print(f"\nSaved models to {MODELS.name}/ and artifacts to {ARTIFACTS.name}/")


if __name__ == "__main__":
    main()
