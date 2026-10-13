"""
Training pipeline. NBA opening point spread.

Goal: before a game, predict the home team's final margin (home points minus away points).
A sportsbook uses that number as its opening line, so the error is measured in points.

Run with: uv run python main.py train
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

# Seasons are named by the year they start. 2012 means 2012-13.
TRAIN_SEASONS = list(range(2012, 2018))  # 2012-13 to 2017-18
VAL_SEASON = 2018                        # 2018-19, only used to pick alpha and learning rates
TEST_SEASON = 2021                       # 2021-22, touched once at the end
# 2011 is only used to build priors for 2012 (lockout season).
# 2019 and 2020 are left out of training and testing. COVID stopped the season,
# the playoffs were in a bubble and most of 2020-21 had empty arenas.
# 2022 is only a partial season in the data, so it is not used.
# The model is fitted with squared error (that is what linear regression is), but judged with
# MAE in points, because that is what a wrong line costs the book.

K_SHRINK = 10        # after 10 games, a team's own results count as much as the prior
PRIOR_WEIGHT = 0.5   # last season's margin is pulled halfway to 0 because rosters change
ALTITUDE_TEAMS = ["DEN", "UTA"]  # the two arenas far above sea level


# ---------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------
games = pd.read_csv(DATA / "games.csv")
teams = pd.read_csv(DATA / "teams.csv")

games["GAME_DATE_EST"] = pd.to_datetime(games["GAME_DATE_EST"])
n_raw = len(games)

# Game IDs that start with 2 are regular season. 1 is preseason, 4 playoffs, 5 play-in.
# Preseason games are played by bench players, so they would add noise.
games = games[games["GAME_ID"].astype(str).str.startswith("2")]
games = games.drop_duplicates("GAME_ID")
games = games.dropna(subset=["PTS_home", "PTS_away"])
games = games.sort_values(["GAME_DATE_EST", "GAME_ID"]).reset_index(drop=True)
print(f"rows: {n_raw} raw, {len(games)} after keeping regular season games with a score")

# Target
games["margin"] = games["PTS_home"] - games["PTS_away"]

# The shooting, assist and rebound columns are stats from the game itself.
# They are only known after the final buzzer, so they are never used as features.

abbr = dict(zip(teams["TEAM_ID"], teams["ABBREVIATION"]))
games["home"] = games["HOME_TEAM_ID"].map(abbr)
games["away"] = games["VISITOR_TEAM_ID"].map(abbr)


# ---------------------------------------------------------------
# 2. Team game log, one row per team per game
#    Every feature uses only games played before this one (shift(1))
# ---------------------------------------------------------------
cols = ["GAME_ID", "GAME_DATE_EST", "SEASON"]
home_rows = games[cols + ["HOME_TEAM_ID", "margin"]].rename(columns={"HOME_TEAM_ID": "team"})
home_rows["is_home"] = 1
away_rows = games[cols + ["VISITOR_TEAM_ID", "margin"]].rename(columns={"VISITOR_TEAM_ID": "team"})
away_rows["margin"] = -away_rows["margin"]
away_rows["is_home"] = 0

log = pd.concat([home_rows, away_rows]).sort_values(["team", "GAME_DATE_EST", "GAME_ID"])
by_team_season = log.groupby(["team", "SEASON"])

log["n_played"] = by_team_season.cumcount()
log["avg_to_date"] = by_team_season["margin"].transform(lambda s: s.shift(1).expanding().mean())
log["last10"] = by_team_season["margin"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())

# Rest: days since the team's last game, capped at 4. First game of the season counts as 4.
last_date = by_team_season["GAME_DATE_EST"].shift(1)
log["rest_days"] = (log["GAME_DATE_EST"] - last_date).dt.days.fillna(4).clip(upper=4)
log["b2b"] = (log["rest_days"] == 1).astype(int)

# Road trip: how many away games in a row the team has played, including this one
run_id = (log["is_home"] != by_team_season["is_home"].shift(1)).cumsum()
log["road_trip"] = np.where(log["is_home"] == 0, log.groupby(run_id).cumcount() + 1, 0)
log["road_trip"] = log["road_trip"].clip(upper=6)

# Prior: last season's average margin, pulled halfway to 0
season_avg = log.groupby(["team", "SEASON"])["margin"].mean().rename("prev_avg").reset_index()
season_avg["SEASON"] += 1
log = log.merge(season_avg, on=["team", "SEASON"], how="left")
log["prior"] = PRIOR_WEIGHT * log["prev_avg"].fillna(0)

# Shrunk rating: blend of results so far and the prior.
# Early in the season a team has 1 or 2 games, so its average is mostly noise.
n = log["n_played"]
log["rating"] = (n * log["avg_to_date"].fillna(0) + K_SHRINK * log["prior"]) / (n + K_SHRINK)
log["rating_raw"] = log["avg_to_date"].fillna(0)  # no shrinking, only used in the feature check
log["form"] = log["last10"].fillna(log["prior"])

# Home only margin so far, used to test a team specific home edge in the feature check
home_log = log[log["is_home"] == 1].copy()
g_home = home_log.groupby(["team", "SEASON"])
home_log["home_n"] = g_home.cumcount()
home_log["home_avg"] = g_home["margin"].transform(lambda s: s.shift(1).expanding().mean())
log = log.merge(home_log[["GAME_ID", "team", "home_n", "home_avg"]], on=["GAME_ID", "team"], how="left")


# ---------------------------------------------------------------
# 3. Game level features (home minus away)
# ---------------------------------------------------------------
keep = ["GAME_ID", "team", "rating", "rating_raw", "form", "rest_days", "b2b", "road_trip",
        "n_played", "home_n", "home_avg", "avg_to_date", "prev_avg"]
h = log[keep].add_prefix("h_").rename(columns={"h_GAME_ID": "GAME_ID", "h_team": "HOME_TEAM_ID"})
a = log[keep].add_prefix("a_").rename(columns={"a_GAME_ID": "GAME_ID", "a_team": "VISITOR_TEAM_ID"})
df = games.merge(h, on=["GAME_ID", "HOME_TEAM_ID"]).merge(a, on=["GAME_ID", "VISITOR_TEAM_ID"])

df["rating_diff"] = df["h_rating"] - df["a_rating"]
df["form_diff"] = df["h_form"] - df["a_form"]
df["rest_diff"] = df["h_rest_days"] - df["a_rest_days"]
df["home_b2b"] = df["h_b2b"]
df["away_b2b"] = df["a_b2b"]
df["away_road_trip"] = df["a_road_trip"]
df["altitude"] = df["home"].isin(ALTITUDE_TEAMS).astype(int)

# Extra columns only for the feature check
df["rating_diff_raw"] = df["h_rating_raw"] - df["a_rating_raw"]
home_edge = (df["h_home_avg"] - df["h_avg_to_date"]).fillna(0)
df["home_edge"] = home_edge * df["h_home_n"] / (df["h_home_n"] + 20)  # shrink hard, few home games

FEATURES = ["rating_diff", "form_diff", "rest_diff", "home_b2b", "away_b2b", "altitude"]
TARGET = "margin"


# ---------------------------------------------------------------
# 4. Time based split
# ---------------------------------------------------------------
train = df[df["SEASON"].isin(TRAIN_SEASONS)]
val = df[df["SEASON"] == VAL_SEASON]
test = df[df["SEASON"] == TEST_SEASON]
print(f"train {len(train)} games ({TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}), "
      f"val {len(val)} ({VAL_SEASON}), test {len(test)} ({TEST_SEASON})")


def mae(y, p):
    return mean_absolute_error(y, p)


# ---------------------------------------------------------------
# 5. Feature check on validation (Ridge, alpha 1)
#    Which feature ideas actually help? Only validation data is used here.
#    With 1,230 games, small MAE differences can be luck, so each row also shows the
#    difference from the final set and its standard error (paired, game by game).
# ---------------------------------------------------------------
def val_abs_errors(data, feats):
    sc = StandardScaler().fit(data.loc[train.index, feats])
    m = Ridge(alpha=1.0).fit(sc.transform(data.loc[train.index, feats]), train[TARGET])
    return np.abs(val[TARGET].values - m.predict(sc.transform(data.loc[val.index, feats])))


def rating_diff_with(k, prior_weight):
    """Shrunk rating difference for a different K and prior weight, to sanity check the choice."""
    def team_rating(p):
        n_ = df[p + "n_played"]
        prior = prior_weight * df[p + "prev_avg"].fillna(0)
        return (n_ * df[p + "avg_to_date"].fillna(0) + k * prior) / (n_ + k)
    return team_rating("h_") - team_rating("a_")


check_df = df.copy()
for k, pw in [(5, 0.5), (20, 0.5), (10, 0.0), (10, 1.0)]:
    check_df[f"rating_k{k}_p{pw}"] = rating_diff_with(k, pw)

base = ["rating_diff", "form_diff", "rest_diff", "home_b2b", "away_b2b"]
checks = {
    "constant (home court only)": np.abs(val[TARGET].values - train[TARGET].mean()),
    "rating_diff, not shrunk": val_abs_errors(check_df, ["rating_diff_raw"]),
    "rating_diff, shrunk (K 10, prior 0.5)": val_abs_errors(check_df, ["rating_diff"]),
    "  shrink check: K 5": val_abs_errors(check_df, ["rating_k5_p0.5"]),
    "  shrink check: K 20": val_abs_errors(check_df, ["rating_k20_p0.5"]),
    "  shrink check: prior weight 0": val_abs_errors(check_df, ["rating_k10_p0.0"]),
    "  shrink check: prior weight 1": val_abs_errors(check_df, ["rating_k10_p1.0"]),
    "+ form_diff": val_abs_errors(check_df, ["rating_diff", "form_diff"]),
    "+ rest and back to backs": val_abs_errors(check_df, base),
    "+ altitude (final set)": val_abs_errors(check_df, FEATURES),
    "final set + away road trip": val_abs_errors(check_df, FEATURES + ["away_road_trip"]),
    "final set + team home edge": val_abs_errors(check_df, FEATURES + ["home_edge"]),
}
final_err = checks["+ altitude (final set)"]
feature_check = pd.DataFrame({
    "validation MAE": {k: v.mean() for k, v in checks.items()},
    "vs final set": {k: v.mean() - final_err.mean() for k, v in checks.items()},
    "std error of difference": {k: (v - final_err).std() / np.sqrt(len(v)) for k, v in checks.items()},
}).round(3)
print("\nfeature check (validation MAE, lower is better)")
print(feature_check.to_string())
feature_check.to_csv(ART / "feature_check.csv")


# ---------------------------------------------------------------
# 6. Scale features. Fit the scaler on training data only.
# ---------------------------------------------------------------
scaler = StandardScaler().fit(train[FEATURES])
X_tr = scaler.transform(train[FEATURES])
X_va = scaler.transform(val[FEATURES])
y_tr = train[TARGET].values
y_va = val[TARGET].values


# ---------------------------------------------------------------
# 7. Three ways to fit the same model: y = Xw + b with an L2 penalty alpha * sum(w^2)
#    sklearn Ridge minimises  sum((y - Xw - b)^2) + alpha * sum(w^2)
#    Dividing by n gives      mean((y - Xw - b)^2) + (alpha / n) * sum(w^2)
#    which is the loss used in both PyTorch versions, so all three aim at the same answer.
# ---------------------------------------------------------------
def fit_sklearn(X, y, alpha):
    return Ridge(alpha=alpha).fit(X, y)


def fit_manual(X, y, alpha, lr, epochs=1000):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    n_rows = len(y_t)

    w = torch.zeros(X_t.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []

    for epoch in range(epochs):
        y_pred = X_t @ w + b
        loss = ((y_pred - y_t) ** 2).mean() + (alpha / n_rows) * (w ** 2).sum()
        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()
        history.append(loss.item())

    return w.detach(), b.detach(), history


def predict_manual(w, b, X):
    with torch.no_grad():
        return (torch.tensor(X, dtype=torch.float32) @ w + b).numpy()


def fit_standard(X, y, alpha, lr, epochs=1000):
    torch.manual_seed(0)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    n_rows = len(y_t)

    model = torch.nn.Linear(X_t.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_t).squeeze(-1)
        # Penalty written by hand so it only hits the weights, not the bias, same as Ridge
        loss = loss_fn(y_pred, y_t) + (alpha / n_rows) * (model.weight ** 2).sum()
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return model, history


def predict_standard(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).squeeze(-1).numpy()


# --- tuning, sparingly, on validation only ---
alphas = [0.1, 1, 10, 100, 1000, 10000]
alpha_scores = {a_: mae(y_va, fit_sklearn(X_tr, y_tr, a_).predict(X_va)) for a_ in alphas}
best_alpha = min(alpha_scores, key=alpha_scores.get)
print("\nalpha -> validation MAE:", {k: round(v, 3) for k, v in alpha_scores.items()}, "| chosen", best_alpha)

# The learning rate's job is to reach the minimum of the training loss, so it is picked by
# the lowest final training loss. Validation MAE is printed too, to check nothing odd happens.
sk_check = fit_sklearn(X_tr, y_tr, best_alpha)
sk_loss = float(np.mean((y_tr - sk_check.predict(X_tr)) ** 2) + best_alpha / len(y_tr) * np.sum(sk_check.coef_ ** 2))
print(f"sklearn training loss (the exact minimum): {sk_loss:.4f}")

manual_runs = {}
for lr in [0.01, 0.1, 0.5]:
    w, b, hist = fit_manual(X_tr, y_tr, best_alpha, lr)
    manual_runs[lr] = (hist[-1], mae(y_va, predict_manual(w, b, X_va)), w[0].item())
best_lr_manual = min(manual_runs, key=lambda lr: manual_runs[lr][0])
print("manual lr -> (final train loss, val MAE, rating weight):",
      {k: tuple(round(x, 4) for x in v) for k, v in manual_runs.items()}, "| chosen", best_lr_manual)

standard_runs = {}
for lr in [0.001, 0.01, 0.1]:
    m, hist = fit_standard(X_tr, y_tr, best_alpha, lr)
    standard_runs[lr] = (hist[-1], mae(y_va, predict_standard(m, X_va)), m.weight[0, 0].item())
best_lr_standard = min(standard_runs, key=lambda lr: standard_runs[lr][0])
print("standard lr -> (final train loss, val MAE, rating weight):",
      {k: tuple(round(x, 4) for x in v) for k, v in standard_runs.items()}, "| chosen", best_lr_standard)
print(f"sklearn rating weight: {sk_check.coef_[0]:.4f}")


# ---------------------------------------------------------------
# 8. Final models: refit on train + validation with the chosen settings
# ---------------------------------------------------------------
final = pd.concat([train, val])
scaler = StandardScaler().fit(final[FEATURES])
X_fin = scaler.transform(final[FEATURES])
y_fin = final[TARGET].values
X_te = scaler.transform(test[FEATURES])
y_te = test[TARGET].values

sk_model = fit_sklearn(X_fin, y_fin, best_alpha)
w_man, b_man, hist_manual = fit_manual(X_fin, y_fin, best_alpha, best_lr_manual)
std_model, hist_standard = fit_standard(X_fin, y_fin, best_alpha, best_lr_standard)
baseline_value = float(y_fin.mean())  # naive line: every home team is favoured by the average margin


# ---------------------------------------------------------------
# 9. Evaluate on the test season
# ---------------------------------------------------------------
preds = {
    "Naive baseline": np.full(len(y_te), baseline_value),
    "scikit-learn": sk_model.predict(X_te),
    "Manual PyTorch": predict_manual(w_man, b_man, X_te),
    "Standard PyTorch": predict_standard(std_model, X_te),
}


def metrics(y, p):
    err = np.abs(y - p)
    return {
        "MAE": mae(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": r2_score(y, p),
        "within 3 pts %": 100 * float((err <= 3).mean()),
        "picks winner %": 100 * float((np.sign(p) == np.sign(y)).mean()),
    }


results = pd.DataFrame({name: metrics(y_te, p) for name, p in preds.items()}).T.round(3)
print(f"\ntest season {TEST_SEASON}-{TEST_SEASON - 1999}, {len(y_te)} games")
print(results)

agree = {
    "sklearn vs manual": float(np.abs(preds["scikit-learn"] - preds["Manual PyTorch"]).max()),
    "sklearn vs standard": float(np.abs(preds["scikit-learn"] - preds["Standard PyTorch"]).max()),
    "manual vs standard": float(np.abs(preds["Manual PyTorch"] - preds["Standard PyTorch"]).max()),
}
print("\nlargest difference in any single test prediction (points):", {k: round(v, 4) for k, v in agree.items()})

coefs = pd.DataFrame({
    "scikit-learn": np.append(sk_model.coef_, sk_model.intercept_),
    "Manual PyTorch": np.append(w_man.numpy(), b_man.numpy()),
    "Standard PyTorch": np.append(std_model.weight.detach().numpy().ravel(), std_model.bias.detach().numpy()),
}, index=FEATURES + ["intercept"]).round(3)
# Weight per 1 standard deviation of the feature, in points
print("\ncoefficients (points per 1 std of each feature)")
print(coefs)

# Home court advantage by season, for the dashboard and the report
home_edge_by_season = (
    games[games["SEASON"].between(2012, 2021)]
    .groupby("SEASON")
    .agg(games=("margin", "size"), avg_home_margin=("margin", "mean"), home_win_pct=("HOME_TEAM_WINS", "mean"))
    .round(3)
)
home_edge_by_season["home_win_pct"] = (100 * home_edge_by_season["home_win_pct"]).round(1)
print("\nhome court by season")
print(home_edge_by_season)


# ---------------------------------------------------------------
# 10. Save everything the app needs. The app never trains anything.
# ---------------------------------------------------------------
joblib.dump(scaler, ART / "scaler.joblib")
joblib.dump(sk_model, ART / "sklearn_model.joblib")
torch.save({"w": w_man, "b": b_man}, ART / "manual_torch.pt")
torch.save(std_model.state_dict(), ART / "standard_torch.pt")

show_cols = ["GAME_DATE_EST", "SEASON", "home", "away", "PTS_home", "PTS_away"] + FEATURES + [TARGET]
test[show_cols].to_csv(ART / "test_games.csv", index=False)
final[show_cols].to_csv(ART / "train_games.csv", index=False)
results.to_csv(ART / "test_results.csv")
coefs.to_csv(ART / "coefficients.csv")
home_edge_by_season.to_csv(ART / "home_edge_by_season.csv")

config = {
    "features": FEATURES,
    "target": TARGET,
    "train_seasons": TRAIN_SEASONS + [VAL_SEASON],
    "train_label": f"{TRAIN_SEASONS[0]}-{TRAIN_SEASONS[0] - 1999} to {VAL_SEASON}-{VAL_SEASON - 1999}",
    "test_season": TEST_SEASON,
    "alpha": best_alpha,
    "lr_manual": best_lr_manual,
    "lr_standard": best_lr_standard,
    "baseline_value": baseline_value,
    "max_prediction_gap": agree,
    "loss_history": {"Manual PyTorch": hist_manual, "Standard PyTorch": hist_standard},
}
(ART / "config.json").write_text(json.dumps(config, indent=1))
print(f"\nsaved models and data to {ART}")
