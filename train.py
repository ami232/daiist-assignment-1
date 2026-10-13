"""
Assignment 1 — training pipeline (PART 3: three models, comparison, saved artifacts).

Business question: will the HOME team win this regular-season NFL game?
Decision it supports: if the model is confident the home team wins, the coach
rests starters. Costs (invented, see REPORT.md):
    false positive  (we say "win", rest starters, and LOSE)         = 3
    false negative  (we say "no win", play starters, would have won) = 1
=> predict "win" only when P(win) > COST_FP / (COST_FP + COST_FN) = 0.75
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "nfl_games.csv"
ARTIFACTS = ROOT / "artifacts"      # everything app.py needs; created by this script

# --- Business framing, written as constants so every later step uses them ---
COST_FP = 3.0   # rest starters, then lose
COST_FN = 1.0   # keep starters in, would have won anyway
DECISION_THRESHOLD = COST_FP / (COST_FP + COST_FN)  # 0.75

# --- Temporal split: never train on the future ---
TRAIN_END = 2021          # train: 1999-2021
VAL_SEASON = 2022         # validation: 2022 (tune regularization / learning rate here)
TEST_START = 2023         # test: 2023-2025 (touched only for the final comparison)

# These columns describe the OUTCOME of the game (or the market's opinion of it),
# so using them to predict the outcome would be leakage.
LEAKAGE_COLS = [
    "home_score", "away_score", "result", "total", "overtime",   # the result itself
    "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "home_spread_odds", "away_spread_odds", "over_odds", "under_odds",  # market opinion
]

# Final feature set. All are "home minus away" so one number says who is stronger.
FEATURES = [
    "pd5_diff",    # recent form: avg point margin in the last 5 games
    "pd16_diff",   # underlying strength: avg point margin in the last 16 games
    "rest_diff",   # days of rest, home minus away
    "div_game",    # 1 if both teams are in the same division
]
SHORT_WINDOW, LONG_WINDOW = 5, 16


# ----------------------------------------------------------------------------
# 1. Loading and target
# ----------------------------------------------------------------------------
def load_games() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["gameday"])
    df = df[df["game_type"] == "REG"]          # playoffs have a different dynamic
    df = df[df["result"] != 0]                 # drop ties: the target must be binary
    df["home_win"] = (df["result"] > 0).astype(int)
    return df.sort_values(["gameday", "game_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Features built ONLY from games played before the one we predict
# ----------------------------------------------------------------------------
def team_form(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) with that team's average point margin over its
    previous SHORT_WINDOW / LONG_WINDOW games. `shift(1)` is the key line: it
    removes the current game, so the feature only sees the past."""
    home = games[["game_id", "gameday", "home_team", "home_score", "away_score"]]
    away = games[["game_id", "gameday", "away_team", "away_score", "home_score"]]
    cols = ["game_id", "gameday", "team", "points_for", "points_against"]
    long = pd.concat([home.set_axis(cols, axis=1), away.set_axis(cols, axis=1)])
    long = long.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    long["margin"] = long["points_for"] - long["points_against"]

    by_team = long.groupby("team")["margin"]
    for window in (SHORT_WINDOW, LONG_WINDOW):
        long[f"pd{window}"] = by_team.transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    # A team's very first game in the data (1999 week 1) has no past: neutral value 0.
    return long[["game_id", "team", f"pd{SHORT_WINDOW}", f"pd{LONG_WINDOW}"]].fillna(0.0)


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    form = team_form(games)
    out = games.copy()
    for side in ("home", "away"):
        side_form = form.rename(
            columns={
                "team": f"{side}_team",
                f"pd{SHORT_WINDOW}": f"{side}_pd5",
                f"pd{LONG_WINDOW}": f"{side}_pd16",
            }
        )
        out = out.merge(side_form, on=["game_id", f"{side}_team"], how="left")
    out["pd5_diff"] = out["home_pd5"] - out["away_pd5"]
    out["pd16_diff"] = out["home_pd16"] - out["away_pd16"]
    out["rest_diff"] = out["home_rest"] - out["away_rest"]
    return out


def check_no_future_leakage(games: pd.DataFrame, features: pd.DataFrame) -> None:
    """Sanity test. Pick the first game day of the test period and scramble the
    scores of every game on or after it. Features of the games ON that day must
    not move (they may only depend on earlier games), while features of later
    games must move (proving the test has teeth)."""
    cutoff = games.loc[games["season"] >= TEST_START, "gameday"].min()
    rng = np.random.default_rng(0)
    scrambled = games.copy()
    mask = scrambled["gameday"] >= cutoff
    scrambled.loc[mask, "home_score"] = rng.integers(0, 50, mask.sum())
    scrambled.loc[mask, "away_score"] = rng.integers(0, 50, mask.sum())
    other = build_features(scrambled)

    same_day = games["gameday"] == cutoff
    later = games["gameday"] > cutoff + pd.Timedelta(days=60)
    cols = ["pd5_diff", "pd16_diff"]
    assert np.allclose(features.loc[same_day, cols], other.loc[same_day, cols]), \
        "LEAKAGE: features changed when only the present/future scores changed"
    assert not np.allclose(features.loc[later, cols], other.loc[later, cols]), \
        "Leakage test is broken: scrambling the future changed nothing"
    print("Leakage check passed: features depend only on past games.")


# ----------------------------------------------------------------------------
# 3. Split + scaling (scaler fitted on TRAIN only)
# ----------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame):
    train = df[df["season"] <= TRAIN_END]
    val = df[df["season"] == VAL_SEASON]
    test = df[df["season"] >= TEST_START]
    return train, val, test


def prepare_data():
    games = load_games()
    features = build_features(games)
    assert not set(FEATURES) & set(LEAKAGE_COLS), "a leakage column is a feature"
    check_no_future_leakage(games, features)

    train, val, test = temporal_split(features)
    scaler = StandardScaler().fit(train[FEATURES])      # statistics from train only
    data = {"scaler": scaler, "features": FEATURES}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        data[f"X_{name}"] = scaler.transform(part[FEATURES]).astype(np.float32)
        data[f"y_{name}"] = part["home_win"].to_numpy(dtype=np.float32)
        data[f"meta_{name}"] = part[["game_id", "season", "week", "gameday",
                                     "home_team", "away_team"]].reset_index(drop=True)
    return data, (train, val, test)


# ----------------------------------------------------------------------------
# 4. The same model three ways: logistic regression, L2-regularized
#
#    probability = sigmoid(X @ w + b)
#    objective   = mean binary cross-entropy + (lam / 2) * sum(w**2)   (b is not penalized)
#
#    scikit-learn's C is the inverse of the penalty:  lam = 1 / (C * n_train).
#    I use that conversion so all three methods minimize the SAME objective.
# ----------------------------------------------------------------------------
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]     # tuned on validation (sklearn)
LR_GRID = [0.1, 0.5, 1.0, 2.0]                    # tuned on validation (manual loop)
EPOCHS = 3000                                     # full-batch gradient descent
TIE_TOLERANCE = 5e-5   # validation log-loss differences below this are treated as noise


def pick_simplest(scores: dict, prefer_small: bool = True):
    """Among settings whose validation log loss is within TIE_TOLERANCE of the best,
    take the smallest value: the strongest regularization (small C) or the most
    cautious learning rate. Avoids picking a value by a 1e-6 difference."""
    best = min(scores.values())
    ok = [k for k, v in scores.items() if v <= best + TIE_TOLERANCE]
    return min(ok) if prefer_small else max(ok)


def fit_sklearn(X, y, C):
    model = LogisticRegression(C=C, solver="lbfgs", tol=1e-10, max_iter=10_000)
    return model.fit(X.astype(np.float64), y)


def bce_loss(p, y):
    """Binary cross-entropy written by hand (as in Session 5)."""
    eps = 1e-7
    p = p.clamp(eps, 1 - eps)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()


def fit_manual(X, y, lr, lam, epochs=EPOCHS):
    """Method 2: raw tensors, autograd, manual gradient step (Session 5)."""
    X, y = torch.tensor(X), torch.tensor(y)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for _ in range(epochs):
        p = torch.sigmoid(X @ w + b)
        data_loss = bce_loss(p, y)
        loss = data_loss + 0.5 * lam * (w ** 2).sum()   # explicit L2 penalty on w only
        loss.backward()
        with torch.no_grad():                           # the update is not part of the graph
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()                                  # gradients accumulate otherwise
        b.grad.zero_()
        history.append(data_loss.item())
    return w.detach(), b.detach(), history


def fit_torch_module(X, y, lr, lam, epochs=EPOCHS):
    """Method 3: nn.Module + torch.optim. Same maths as method 2; the library does
    the bookkeeping. weight_decay=lam adds lam*w to the gradient, which is exactly
    the gradient of (lam/2)*sum(w**2). The bias goes in its own group, unpenalized."""
    X, y = torch.tensor(X), torch.tensor(y)
    model = nn.Sequential(nn.Linear(X.shape[1], 1), nn.Sigmoid())
    nn.init.zeros_(model[0].weight)                     # same start as method 2
    nn.init.zeros_(model[0].bias)
    optimizer = torch.optim.SGD(
        [{"params": [model[0].weight], "weight_decay": lam},
         {"params": [model[0].bias], "weight_decay": 0.0}],
        lr=lr,
    )
    loss_fn = nn.BCELoss()
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X).squeeze(-1), y)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return model, history


def predict_manual(w, b, X):
    with torch.no_grad():
        return torch.sigmoid(torch.tensor(X) @ w + b).numpy()


def predict_module(model, X):
    with torch.no_grad():
        return model(torch.tensor(X)).squeeze(-1).numpy()


# ----------------------------------------------------------------------------
# 5. Evaluation: statistics AND the business cost at the 0.75 threshold
# ----------------------------------------------------------------------------
def business_cost(y, p, threshold=DECISION_THRESHOLD):
    say_win = p >= threshold
    fp = int(((say_win) & (y == 0)).sum())      # rested starters and lost
    fn = int(((~say_win) & (y == 1)).sum())     # played starters, would have won
    return COST_FP * fp + COST_FN * fn, fp, fn, int(say_win.sum())


def evaluate(y, p):
    cost, fp, fn, n_rest = business_cost(y, p)
    return {
        "accuracy@0.5": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "cost@0.75": float(cost), "false_pos": fp, "false_neg": fn,
        "games_rested@0.75": n_rest,
    }


def naive_baselines(y_train, y_test):
    rate = float(y_train.mean())
    p = np.full(len(y_test), rate)                   # always the training home-win rate
    always_win = business_cost(y_test, np.ones(len(y_test)))[0]   # always rest starters
    never_win = business_cost(y_test, np.zeros(len(y_test)))[0]   # never rest starters
    return {
        "accuracy_always_home_win": float(y_test.mean()),
        "log_loss_train_rate": float(log_loss(y_test, p)),
        "cost_always_rest": float(always_win),
        "cost_never_rest": float(never_win),
    }


def main() -> None:
    torch.manual_seed(0)
    data, (train, val, test) = prepare_data()
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_va, y_va = data["X_val"], data["y_val"]
    X_te, y_te = data["X_test"], data["y_test"]
    n_train = len(y_tr)

    print(f"\nFeatures: {FEATURES}")
    print(f"train {len(y_tr)} | val {len(y_va)} | test {len(y_te)} games")

    # --- Tune ONLY the regularization strength (on validation) ---
    val_ll = {C: log_loss(y_va, fit_sklearn(X_tr, y_tr, C).predict_proba(X_va.astype(np.float64))[:, 1])
              for C in C_GRID}
    best_C = pick_simplest(val_ll)
    lam = 1.0 / (best_C * n_train)
    print("\nValidation log loss by C:", {C: round(v, 5) for C, v in val_ll.items()})
    print(f"Chosen C = {best_C}  ->  lambda = 1/(C*n) = {lam:.3e}")

    # --- Tune ONLY the learning rate (on validation, manual loop) ---
    lr_ll = {}
    for lr in LR_GRID:
        w, b, _ = fit_manual(X_tr, y_tr, lr, lam)
        lr_ll[lr] = log_loss(y_va, predict_manual(w, b, X_va))
    best_lr = pick_simplest(lr_ll)
    print("Validation log loss by learning rate:", {k: round(v, 6) for k, v in lr_ll.items()})
    print(f"Chosen learning rate = {best_lr} ({EPOCHS} epochs, full batch)")

    # --- Train the three methods with identical settings ---
    sk = fit_sklearn(X_tr, y_tr, best_C)
    w_m, b_m, hist_m = fit_manual(X_tr, y_tr, best_lr, lam)
    net, hist_n = fit_torch_module(X_tr, y_tr, best_lr, lam)

    preds = {
        "sklearn": sk.predict_proba(X_te.astype(np.float64))[:, 1],
        "manual_torch": predict_manual(w_m, b_m, X_te),
        "nn_module": predict_module(net, X_te),
    }
    coefs = {
        "sklearn": list(sk.coef_[0]) + [float(sk.intercept_[0])],
        "manual_torch": w_m.tolist() + [float(b_m)],
        "nn_module": net[0].weight.detach()[0].tolist() + [float(net[0].bias.detach())],
    }

    # --- Compare: do they agree? ---
    table = {name: evaluate(y_te, p) for name, p in preds.items()}
    base = naive_baselines(y_tr, y_te)
    print(f"\n{'TEST SET (2023-2025)':<22}{'acc@0.5':>9}{'logloss':>9}{'AUC':>8}{'cost@.75':>10}{'FP':>5}{'FN':>5}{'rested':>8}")
    for name, m in table.items():
        print(f"{name:<22}{m['accuracy@0.5']:>9.4f}{m['log_loss']:>9.4f}{m['auc']:>8.4f}"
              f"{m['cost@0.75']:>10.0f}{m['false_pos']:>5}{m['false_neg']:>5}{m['games_rested@0.75']:>8}")
    print(f"{'baseline: always home':<22}{base['accuracy_always_home_win']:>9.4f}{base['log_loss_train_rate']:>9.4f}"
          f"{'':>8}{base['cost_always_rest']:>10.0f}   (always rest starters)")
    print(f"{'baseline: never rest':<22}{'':>9}{'':>9}{'':>8}{base['cost_never_rest']:>10.0f}   (never rest starters)")

    print("\nCoefficients [pd5, pd16, rest, div, bias] (standardized features):")
    for name, c in coefs.items():
        print(f"  {name:<14}", np.round(c, 5))
    names = list(preds)
    print("\nMax |difference| in predicted probability on the test set:")
    for i in range(3):
        for j in range(i + 1, 3):
            print(f"  {names[i]} vs {names[j]}: {np.abs(preds[names[i]] - preds[names[j]]).max():.2e}")

    # --- Save everything app.py needs (it must never retrain) ---
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(sk, ARTIFACTS / "sklearn_model.joblib")
    joblib.dump(data["scaler"], ARTIFACTS / "scaler.joblib")
    torch.save({"w": w_m, "b": b_m}, ARTIFACTS / "manual_model.pt")
    torch.save(net.state_dict(), ARTIFACTS / "torch_model.pt")
    all_games = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    cols = ["game_id", "season", "week", "gameday", "home_team", "away_team", "split", "home_win"] + FEATURES
    all_games[cols].to_csv(ARTIFACTS / "dataset.csv", index=False)
    metrics = {
        "features": FEATURES, "cost_fp": COST_FP, "cost_fn": COST_FN,
        "decision_threshold": DECISION_THRESHOLD, "C": best_C, "lambda": lam,
        "learning_rate": best_lr, "epochs": EPOCHS, "n_train": n_train,
        "val_log_loss_by_C": {str(k): v for k, v in val_ll.items()},
        "val_log_loss_by_lr": {str(k): v for k, v in lr_ll.items()},
        "test": table, "baselines": base, "coefficients": coefs,
        "loss_history": {"manual_torch": hist_m, "nn_module": hist_n},
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved models, scaler, dataset and metrics to {ARTIFACTS.name}/")


if __name__ == "__main__":
    main()
