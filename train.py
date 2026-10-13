"""
Assignment 1 — training pipeline.

Task: predict a home win (FTR == "H") in four European football leagues and
evaluate it as a flat-stake bettor (1 unit on the home team at B365H odds).

Run via `uv run python main.py train`. Stages:
  1. Load data/raw/*.csv (latin-1), keep the columns common to every file.
  2. Build pre-match features from strictly earlier matches only.
  3. Leakage test: recompute sampled matches' features after deleting all
     later rows and assert they are identical.
  4. Time-based split: warm-up 2018/19, train 2019/20-2023/24, validate
     2024/25, test 2025/26. The scaler is fit on train only.
  5. Train the same L2 logistic regression three ways (scikit-learn, manual
     PyTorch loop, nn.Module + torch.optim). Tune C (sklearn) and each
     PyTorch learning rate on validation log loss.
  6. Pick ONE shared threshold on validation profit (sklearn model, >= 50
     bets), apply it to all three on test. Test is touched once, at the end.
  7. Save everything the dashboard needs to artifacts/.
"""

import bisect
import json
import math
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "artifacts"

WARMUP = ["1819"]
TRAIN = ["1920", "2021", "2122", "2223", "2324"]
VAL = ["2425"]
TEST = ["2526"]

# --- Fixed (reasoned, not tuned) feature constants -------------------------
ELO_K = 20.0
ELO_HFA = 65.0
ELO_REGRESS = 1 / 3  # fraction pulled back to the league mean each season
REST_CAP = 14
LEAGUES = ["E0", "SP1", "D1", "I1"]

# name: (record field, window, min prior matches for a "full" window, default
# quantile). The default quantile is the bottom of the quality scale: the 25th
# percentile where higher is better, the 75th for goals conceded (higher is
# worse), the median for cards (no quality direction).
TEAM_FEATS = {
    "ppg10": ("pts", 10, 5, 0.25),
    "gf10": ("gf", 10, 5, 0.25),
    "ga10": ("ga", 10, 5, 0.75),
    "sotdiff10": ("sotd", 10, 5, 0.25),
    "shotdiff10": ("shotd", 10, 5, 0.25),
    "gd38": ("gd", 38, 10, 0.25),
    "cards10": ("cards", 10, 5, 0.50),
}
VENUE_WINDOW, VENUE_MIN = 5, 3
REST_QUANTILE = 0.50
ENTRY_QUANTILE = 0.25  # Elo rating of a team new to the league

FIELDS = ["pts", "gf", "ga", "gd", "sotd", "shotd", "cards"]
FEATURES = (
    ["h_ppg10", "a_ppg10", "h_home_ppg5", "a_away_ppg5"]
    + [f"{s}_{n}" for n in ("gf10", "ga10", "sotdiff10", "shotdiff10", "gd38", "cards10")
       for s in ("h", "a")]
    + ["h_rest", "a_rest", "h_cold", "a_cold", "h_elo", "a_elo"]
    + [f"lg_{lg}" for lg in LEAGUES]
)

# --- Tuning grids (only C and learning rate are tuned) ----------------------
C_GRID = [float(c) for c in np.logspace(-3, 2, 11)]
LR_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
MAX_ITER = 30000
GRAD_TOL = 1e-8
THRESHOLDS = np.round(np.arange(0.30, 0.9001, 0.005), 3)
MIN_BETS = 50
BOOT_DRAWS, BOOT_SEED = 2000, 0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    frames = {f.stem: pd.read_csv(f, encoding="latin-1") for f in sorted(RAW.glob("*.csv"))}
    first = next(iter(frames.values()))
    common = [c for c in first.columns if all(c in f.columns for f in frames.values())]
    parts = []
    for stem, f in frames.items():
        league, season = stem.split("_")
        p = f[common].copy()
        p["league"], p["season"] = league, season
        parts.append(p)
    df = pd.concat(parts, ignore_index=True)
    df["dt"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    df["rid"] = np.arange(len(df))
    return df


def validate(df: pd.DataFrame) -> None:
    """Catches team-name inconsistencies: every team must play 2*(n-1) matches."""
    assert df["dt"].notna().all(), "unparsed dates"
    for (lg, s), g in df.groupby(["league", "season"]):
        counts = pd.concat([g["HomeTeam"], g["AwayTeam"]]).value_counts()
        n = len(counts)
        assert len(g) == n * (n - 1) and (counts == 2 * (n - 1)).all(), (
            f"inconsistent team names or fixtures in {lg} {s}: {counts.to_dict()}"
        )


# ---------------------------------------------------------------------------
# Features (strictly pre-match)
# ---------------------------------------------------------------------------
def quantile(sorted_vals: list, q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def window_mean(vals: list, n: int):
    """Mean of the last n matches, skipping NaNs; returns (mean, non-NaN count)."""
    w = [v for v in vals[-n:] if not math.isnan(v)]
    return (sum(w) / len(w) if w else float("nan")), len(w)


class TeamState:
    def __init__(self):
        self.rec = {k: [] for k in FIELDS}
        self.home_pts, self.away_pts = [], []
        self.last = None


class LeagueState:
    def __init__(self):
        self.season = None
        self.elo = {}
        self.entry = None  # rating given to a team new to the league this season
        self.teams_this_season = set()

    def start_season(self, season: str) -> None:
        if self.season is not None:
            prev = sorted(self.teams_this_season)
            mu = float(np.mean([self.elo[t] for t in prev]))
            for t in self.elo:
                self.elo[t] = mu + (1 - ELO_REGRESS) * (self.elo[t] - mu)
            self.entry = quantile(sorted(self.elo[t] for t in prev), ENTRY_QUANTILE)
        self.season = season
        self.teams_this_season = set()


def build_features(df: pd.DataFrame):
    """One pass over matches in date order. A match's features are computed
    before anything from its own date is added to any state, so they depend
    only on matches on strictly earlier dates. Returns (features, defaults)
    indexed by rid; `defaults` flags which features fell back to a default."""
    df = df.sort_values(["dt", "rid"], kind="stable")
    teams = defaultdict(TeamState)
    leagues = defaultdict(LeagueState)
    pools = defaultdict(lambda: defaultdict(list))  # league -> key -> sorted values
    feats, dflts = {}, {}

    for dt, day in df.groupby("dt", sort=True):
        matches = list(day.itertuples(index=False))
        snapshots = []
        for m in matches:
            L = leagues[m.league]
            if L.season != m.season:
                L.start_season(m.season)
            pool = pools[m.league]
            f = {f"lg_{lg}": float(m.league == lg) for lg in LEAGUES}
            d = {}
            for side, name in (("h", m.HomeTeam), ("a", m.AwayTeam)):
                T = teams[(m.league, name)]
                snap, n_default = {}, 0

                def resolve(key, val, n, minc, q, col):
                    nonlocal n_default
                    if n >= minc:
                        snap[key] = val
                        d[col] = 0
                        return val
                    n_default += 1
                    d[col] = 1
                    return quantile(pool[key], q)

                for fname, (field, win, minc, q) in TEAM_FEATS.items():
                    val, n = window_mean(T.rec[field], win)
                    col = f"{side}_{fname}"
                    f[col] = resolve(fname, val, n, minc, q, col)

                vkey = "home_ppg5" if side == "h" else "away_ppg5"
                val, n = window_mean(T.home_pts if side == "h" else T.away_pts, VENUE_WINDOW)
                col = "h_home_ppg5" if side == "h" else "a_away_ppg5"
                f[col] = resolve(vkey, val, n, VENUE_MIN, 0.25, col)

                rest = float(min((dt - T.last).days, REST_CAP)) if T.last is not None else float("nan")
                f[f"{side}_rest"] = resolve("rest", rest, 0 if math.isnan(rest) else 1, 1,
                                            REST_QUANTILE, f"{side}_rest")

                if name not in L.elo:
                    L.elo[name] = L.entry if L.entry is not None else 1500.0
                    d[f"{side}_elo"] = int(L.entry is not None)
                    n_default += d[f"{side}_elo"]
                else:
                    d[f"{side}_elo"] = 0
                f[f"{side}_elo"] = L.elo[name]
                f[f"{side}_cold"] = float(n_default > 0)
                snapshots.append((m.league, snap))
            feats[m.rid], dflts[m.rid] = f, d

        # Updates: only after every feature for this date has been computed.
        for m in matches:
            if not isinstance(m.FTR, str):  # blanked target row in the leakage test
                continue
            L = leagues[m.league]
            Th, Ta = teams[(m.league, m.HomeTeam)], teams[(m.league, m.AwayTeam)]
            pts_h = {"H": 3.0, "D": 1.0, "A": 0.0}[m.FTR]
            pts_a = {"H": 0.0, "D": 1.0, "A": 3.0}[m.FTR]
            for T, pts, gf, ga, sf, sa, tf, ta, cards in (
                (Th, pts_h, m.FTHG, m.FTAG, m.HS, m.AS, m.HST, m.AST, m.HY + 2 * m.HR),
                (Ta, pts_a, m.FTAG, m.FTHG, m.AS, m.HS, m.AST, m.HST, m.AY + 2 * m.AR),
            ):
                for k, v in zip(FIELDS, (pts, gf, ga, gf - ga, tf - ta, sf - sa, cards)):
                    T.rec[k].append(float(v))
                T.last = dt
            Th.home_pts.append(pts_h)
            Ta.away_pts.append(pts_a)
            rh, ra = L.elo[m.HomeTeam], L.elo[m.AwayTeam]
            exp_h = 1 / (1 + 10 ** (-(rh - ra + ELO_HFA) / 400))
            delta = ELO_K * ({"H": 1.0, "D": 0.5, "A": 0.0}[m.FTR] - exp_h)
            L.elo[m.HomeTeam], L.elo[m.AwayTeam] = rh + delta, ra - delta
            L.teams_this_season.update((m.HomeTeam, m.AwayTeam))
        for league, snap in snapshots:
            for key, v in snap.items():
                bisect.insort(pools[league][key], v)
    X = pd.DataFrame.from_dict(feats, orient="index")[FEATURES]
    D = pd.DataFrame.from_dict(dflts, orient="index").fillna(0).astype(int)
    return X, D


def leakage_test(df, X, D, splits, n_random=9, seed=0) -> None:
    """For sampled matches in each split (the split's first match plus random
    ones), delete every row on or after that date except the match itself
    (with its outcome and in-match stats blanked), recompute, and assert the
    match's features are identical."""
    rng = np.random.default_rng(seed)
    keep = ["Date", "HomeTeam", "AwayTeam", "league", "season", "dt", "rid"]
    blank = [c for c in df.columns if c not in keep]
    n_checked = 0
    for name, rids in splits.items():
        first = df.loc[df.rid.isin(rids)].sort_values(["dt", "rid"]).rid.iloc[0]
        sample = [first] + [int(r) for r in rng.choice(rids, n_random, replace=False)]
        for rid in sample:
            row = df.loc[df.rid == rid].iloc[0]
            target = df.loc[df.rid == rid].copy()
            target[blank] = np.nan
            trunc = pd.concat([df[df.dt < row["dt"]], target])
            X2, D2 = build_features(trunc)
            a, b = X.loc[rid].to_numpy(float), X2.loc[rid].to_numpy(float)
            assert np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True), (
                f"LEAKAGE in {name}, rid={rid}: features changed when later rows were deleted: "
                f"{dict(zip(FEATURES, zip(a, b)))}"
            )
            assert (D.loc[rid] == D2.loc[rid]).all(), f"LEAKAGE (defaults) in {name}, rid={rid}"
            n_checked += 1
    # Non-vacuity: removing the home team's most recent earlier match MUST change features.
    row = df.loc[df.rid == splits["test"][50]].iloc[0]
    earlier = df[(df.dt < row["dt"]) & ((df.HomeTeam == row.HomeTeam) | (df.AwayTeam == row.HomeTeam))]
    target = df.loc[df.rid == row.rid].copy()
    target[blank] = np.nan
    X3, _ = build_features(pd.concat([df[(df.dt < row["dt"]) & (df.rid != earlier.sort_values("dt").rid.iloc[-1])], target]))
    assert not np.allclose(X.loc[row.rid].to_numpy(float), X3.loc[row.rid].to_numpy(float),
                           equal_nan=True), "leakage test is vacuous"
    print(f"Leakage test passed: {n_checked} matches recomputed with all later rows deleted "
          f"(non-vacuity check also passed).")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def sigmoid_probs(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(X @ w + b)))


def train_sklearn(X, y, C):
    # Objective: C * sum(log loss) + 0.5 * ||w||^2; intercept unpenalized.
    return LogisticRegression(C=C, max_iter=20000, tol=1e-12).fit(X, y)


def train_manual(X, y, C, lr):
    """Full-batch gradient descent on mean BCE + (wd/2)||w||^2 with wd = 1/(C n).
    Raw tensors, autograd for gradients, manual parameter update."""
    n, d = X.shape
    wd = 1.0 / (C * n)
    w = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    it, gmax = 0, float("inf")
    for it in range(1, MAX_ITER + 1):
        z = X @ w + b
        loss = (F.softplus(z) - y * z).mean() + 0.5 * wd * (w * w).sum()
        loss.backward()
        with torch.no_grad():
            gmax = max(w.grad.abs().max().item(), b.grad.abs().max().item())
            if not math.isfinite(gmax) or gmax < GRAD_TOL:
                break
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    return w.detach().numpy().copy(), float(b.item()), {"iters": it, "grad": gmax}


def train_nn(X, y, C, lr):
    """nn.Linear + BCEWithLogitsLoss + torch.optim.SGD (momentum 0.9), full
    batch. weight_decay = 1/(C n) on the weights only (plain L2, bias exempt)."""
    n, d = X.shape
    wd = 1.0 / (C * n)
    model = nn.Linear(d, 1).double()
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    opt = torch.optim.SGD(
        [{"params": [model.weight], "weight_decay": wd},
         {"params": [model.bias], "weight_decay": 0.0}],
        lr=lr, momentum=0.9,
    )
    crit = nn.BCEWithLogitsLoss()
    it, gmax = 0, float("inf")
    for it in range(1, MAX_ITER + 1):
        opt.zero_grad()
        loss = crit(model(X).squeeze(1), y)
        loss.backward()
        gmax = max((model.weight.grad + wd * model.weight).abs().max().item(),
                   model.bias.grad.abs().max().item())
        if not math.isfinite(gmax) or gmax < GRAD_TOL:
            break
        opt.step()
    w = model.weight.detach().numpy().ravel().copy()
    return model, w, float(model.bias.item()), {"iters": it, "grad": gmax}


def safe_log_loss(y, p):
    return log_loss(y, p) if np.all(np.isfinite(p)) else float("inf")


# ---------------------------------------------------------------------------
# Business cost
# ---------------------------------------------------------------------------
def bets_and_profit(p, y, odds, thr):
    """Flat 1-unit bet on the home team when p >= thr. A win pays odds - 1, a
    loss costs 1. Matches with missing B365H are excluded from the cost only."""
    mask = (p >= thr) & ~np.isnan(odds)
    return int(mask.sum()), float(np.where(y[mask] == 1, odds[mask] - 1, -1.0).sum())


def main():
    torch.manual_seed(0)
    OUT.mkdir(exist_ok=True)

    # ---- 1. data ----------------------------------------------------------
    df = load_raw()
    validate(df)
    print(f"Loaded {len(df)} rows; leagues {sorted(df.league.unique())}; "
          f"seasons {sorted(df.season.unique())}")

    # ---- 2. features ------------------------------------------------------
    X_all, D_all = build_features(df)
    df = df.set_index("rid", drop=False)
    df.index.name = None
    split_of = pd.Series("warmup", index=df.index)
    split_of[df.season.isin(TRAIN)] = "train"
    split_of[df.season.isin(VAL)] = "val"
    split_of[df.season.isin(TEST)] = "test"
    splits = {s: split_of.index[split_of == s].to_numpy() for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("val", "test")):
        assert df.loc[splits[a], "dt"].max() < df.loc[splits[b], "dt"].min(), "splits overlap in time"
    used = X_all.loc[np.concatenate(list(splits.values()))]
    assert not used.isna().any().any(), "NaN features in train/val/test rows"
    assert not any(c.startswith(("B365", "BW", "PS")) for c in FEATURES), "odds in features"

    # ---- 3. leakage test --------------------------------------------------
    leakage_test(df, X_all, D_all, splits)

    # ---- cold-start report ------------------------------------------------
    print("\nCold-start rows (a default replaced a missing/short history):")
    for s, rids in splits.items():
        X, D = X_all.loc[rids], D_all.loc[rids]
        either, both = int(((X.h_cold + X.a_cold) >= 1).sum()), int(((X.h_cold + X.a_cold) == 2).sum())
        anyd = int((D.sum(axis=1) > 0).sum())
        print(f"  {s:5s} n={len(rids):5d} | home cold {int(X.h_cold.sum()):4d}  away cold "
              f"{int(X.a_cold.sum()):4d}  either {either:4d}  both {both:3d}  "
              f"| rows with >=1 default {anyd:4d}")
    base = {}
    for col in D_all.columns:
        base.setdefault(col[2:], 0)
    print("  defaults per feature (train+val+test, home+away team combined):")
    allu = D_all.loc[np.concatenate(list(splits.values()))]
    print("   ", {k: int(sum(allu[c].sum() for c in allu.columns if c[2:] == k)) for k in base})

    # ---- 4. split & scale -------------------------------------------------
    y = (df["FTR"] == "H").astype(float)
    odds = df["B365H"].astype(float)
    scaler = StandardScaler().fit(X_all.loc[splits["train"]].to_numpy(float))
    assert scaler.n_samples_seen_ == len(splits["train"])
    Xs = {s: scaler.transform(X_all.loc[r].to_numpy(float)) for s, r in splits.items()}
    ys = {s: y.loc[r].to_numpy() for s, r in splits.items()}
    os_ = {s: odds.loc[r].to_numpy() for s, r in splits.items()}
    n_train = len(ys["train"])
    print(f"\nSplit sizes: train {n_train}, val {len(ys['val'])}, test {len(ys['test'])}; "
          f"features {len(FEATURES)}; train home-win rate {ys['train'].mean():.4f}")
    Xt = {s: torch.tensor(v, dtype=torch.float64) for s, v in Xs.items()}
    yt = {s: torch.tensor(v, dtype=torch.float64) for s, v in ys.items()}

    # ---- 5. train three ways ---------------------------------------------
    print("\nTuning C (sklearn, validation log loss):")
    c_scores = {}
    for C in C_GRID:
        m = train_sklearn(Xs["train"], ys["train"], C)
        c_scores[C] = log_loss(ys["val"], m.predict_proba(Xs["val"])[:, 1])
        print(f"  C={C:9.4f}  val log loss {c_scores[C]:.5f}")
    C = min(c_scores, key=c_scores.get)
    print(f"-> chosen C = {C:.4f}  (weight decay 1/(C n) = {1 / (C * n_train):.3e})")
    sk = train_sklearn(Xs["train"], ys["train"], C)
    w_sk, b_sk = sk.coef_.ravel(), float(sk.intercept_[0])

    def tune_lr(name, trainer):
        scores, fits = {}, {}
        for lr in LR_GRID:
            fit = trainer(lr)
            w, b, info = fit[-3], fit[-2], fit[-1]
            scores[lr] = safe_log_loss(ys["val"], sigmoid_probs(Xs["val"], w, b))
            fits[lr] = fit
            print(f"  {name} lr={lr:<5} val log loss {scores[lr]:.6f}  iters {info['iters']:6d}  "
                  f"max|grad| {info['grad']:.2e}")
        # Only converged runs compete; val log loss differences below 1e-6 are
        # treated as ties, which go to the run that needed the fewest iterations.
        converged = [lr for lr in LR_GRID if fits[lr][-1]["grad"] < GRAD_TOL]
        assert converged, f"{name}: no learning rate converged within {MAX_ITER} iterations"
        best = min(converged, key=lambda lr: (round(scores[lr], 6), fits[lr][-1]["iters"]))
        print(f"-> chosen {name} lr = {best}")
        return best, fits[best], scores

    print("\nTuning learning rates (validation log loss):")
    lr_man, fit_man, sc_man = tune_lr("manual", lambda lr: (*train_manual(Xt["train"], yt["train"], C, lr),))
    lr_nn, fit_nn, sc_nn = tune_lr("nn    ", lambda lr: train_nn(Xt["train"], yt["train"], C, lr))
    w_man, b_man, info_man = fit_man
    model_nn, w_nn, b_nn, info_nn = fit_nn

    probs = {
        "sklearn": {s: sigmoid_probs(Xs[s], w_sk, b_sk) for s in ("train", "val", "test")},
        "manual": {s: sigmoid_probs(Xs[s], w_man, b_man) for s in ("train", "val", "test")},
        "nn": {s: sigmoid_probs(Xs[s], w_nn, b_nn) for s in ("train", "val", "test")},
    }

    # ---- 6. threshold on validation (sklearn), then test once ------------
    sweep = []
    for t in THRESHOLDS:
        nb, pr = bets_and_profit(probs["sklearn"]["val"], ys["val"], os_["val"], t)
        sweep.append({"threshold": t, "bets": nb, "profit": pr})
    sweep = pd.DataFrame(sweep)
    ok = sweep[sweep.bets >= MIN_BETS]
    assert len(ok), "no threshold gives the minimum number of validation bets"
    best = ok.loc[ok.profit.idxmax()]
    thr = float(best.threshold)
    print(f"\nValidation threshold sweep (sklearn, >= {MIN_BETS} bets): chosen threshold {thr:.3f} -> "
          f"{int(best.bets)} bets, profit {best.profit:+.2f}")
    print(f"  best profit over all valid thresholds {ok.profit.max():+.2f}, worst {ok.profit.min():+.2f}; "
          f"{int((ok.profit > 0).sum())}/{len(ok)} valid thresholds are profitable")

    rows = []
    maj = 0.0 if ys["train"].mean() < 0.5 else 1.0
    prior = float(ys["train"].mean())
    for split in ("val", "test"):
        yy, oo = ys[split], os_[split]
        rows.append({"split": split, "model": "Majority class (never bets)",
                     "acc@thr": np.mean(yy == maj), "acc@0.5": np.mean(yy == maj),
                     "log_loss": log_loss(yy, np.full(len(yy), prior)), "bets": 0, "profit": 0.0})
        allbets = bets_and_profit(np.ones(len(yy)), yy, oo, 0.5)
        rows.append({"split": split, "model": "Bet on every home team", "acc@thr": np.mean(yy == 1),
                     "acc@0.5": np.mean(yy == 1), "log_loss": np.nan,
                     "bets": allbets[0], "profit": allbets[1]})
        for name, label in (("sklearn", "scikit-learn"), ("manual", "Manual PyTorch loop"),
                            ("nn", "nn.Module + optim")):
            p = probs[name][split]
            nb, pr = bets_and_profit(p, yy, oo, thr)
            rows.append({"split": split, "model": label, "acc@thr": np.mean((p >= thr) == (yy == 1)),
                         "acc@0.5": np.mean((p >= 0.5) == (yy == 1)), "log_loss": log_loss(yy, p),
                         "bets": nb, "profit": pr})
    res = pd.DataFrame(rows)
    res["roi_%"] = np.where(res.bets > 0, 100 * res.profit / res.bets.replace(0, np.nan), np.nan)
    for split in ("val", "test"):
        print(f"\n=== {split.upper()} (shared threshold {thr:.3f}) ===")
        print(res[res.split == split].drop(columns="split").to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- descriptive bootstrap CI for test profit (nothing is chosen with it)
    contrib = {}
    yy, oo = ys["test"], os_["test"]
    win = np.where(yy == 1, oo - 1, -1.0)
    for name in probs:
        contrib[name] = np.where((probs[name]["test"] >= thr) & ~np.isnan(oo), win, 0.0)
    contrib["every_home"] = np.where(~np.isnan(oo), win, 0.0)
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(yy), size=(BOOT_DRAWS, len(yy)))
    boot = {}
    for name, c in contrib.items():
        sums = c[idx].sum(axis=1)
        boot[name] = {"profit": float(c.sum()), "ci_low": float(np.percentile(sums, 2.5)),
                      "ci_high": float(np.percentile(sums, 97.5))}
    print(f"\nBootstrap 95% CI for test profit at threshold {thr:.3f} ({BOOT_DRAWS} draws, "
          f"resampling matches; descriptive only):")
    for name, v in boot.items():
        print(f"  {name:10s} profit {v['profit']:+8.2f}   [{v['ci_low']:+8.2f}, {v['ci_high']:+8.2f}]")

    # ---- agreement diagnostics -------------------------------------------
    print("\nAgreement between implementations (standardized-feature space):")
    print(f"  max |w_sklearn - w_manual| = {np.abs(w_sk - w_man).max():.2e}   "
          f"|b diff| = {abs(b_sk - b_man):.2e}   (manual: {info_man})")
    print(f"  max |w_sklearn - w_nn|     = {np.abs(w_sk - w_nn).max():.2e}   "
          f"|b diff| = {abs(b_sk - b_nn):.2e}   (nn: {info_nn})")
    for s in ("val", "test"):
        pm, pn, ps = probs["manual"][s], probs["nn"][s], probs["sklearn"][s]
        print(f"  {s}: max |p_sk - p_manual| = {np.abs(ps - pm).max():.2e}, max |p_sk - p_nn| = "
              f"{np.abs(ps - pn).max():.2e}; decisions differing at thr: manual "
              f"{int(((ps >= thr) != (pm >= thr)).sum())}, nn {int(((ps >= thr) != (pn >= thr)).sum())}")

    # ---- 7. save artifacts -----------------------------------------------
    joblib.dump(sk, OUT / "sklearn_model.joblib")
    joblib.dump(scaler, OUT / "scaler.joblib")
    torch.save({"w": torch.tensor(w_man), "b": torch.tensor([b_man])}, OUT / "manual_model.pt")
    torch.save(model_nn.state_dict(), OUT / "nn_model.pt")
    pred = df.loc[np.concatenate([splits["val"], splits["test"]]),
                  ["rid", "dt", "league", "season", "HomeTeam", "AwayTeam", "B365H"]].copy()
    pred["split"] = split_of.loc[pred.index].to_numpy()
    pred["y_home_win"] = y.loc[pred.index].to_numpy().astype(int)
    for name in probs:
        pred[f"p_{name}"] = np.concatenate([probs[name]["val"], probs[name]["test"]])
    pred.to_csv(OUT / "predictions.csv", index=False)
    dist = X_all.loc[np.concatenate(list(splits.values()))].copy()
    dist.insert(0, "rid", dist.index.to_numpy())
    dist.insert(1, "split", split_of.loc[dist.index].to_numpy())
    dist.insert(2, "y_home_win", y.loc[dist.index].to_numpy().astype(int))
    dist.to_csv(OUT / "features.csv", index=False)
    sweep.to_csv(OUT / "val_threshold_sweep.csv", index=False)
    res.to_csv(OUT / "results.csv", index=False)
    meta = {
        "features": FEATURES, "C": C, "lr_manual": lr_man, "lr_nn": lr_nn, "threshold": thr,
        "min_bets": MIN_BETS, "splits": {"warmup": WARMUP, "train": TRAIN, "val": VAL, "test": TEST},
        "train_home_win_rate": prior,
        "elo": {"K": ELO_K, "home_advantage": ELO_HFA, "regress": ELO_REGRESS},
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    (OUT / "bootstrap.json").write_text(json.dumps({"threshold": thr, "draws": BOOT_DRAWS,
                                                    "seed": BOOT_SEED, "test_profit": boot}, indent=2))
    print(f"\nSaved artifacts to {OUT}")


if __name__ == "__main__":
    main()
