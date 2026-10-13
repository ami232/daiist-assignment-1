"""
Validation-only experiments behind the choices in features.py and common.py.

Run:  uv run python experiments/feature_ablation.py        (about 1-2 minutes)

Uses the SAME stratified split as train.py and a plain scikit-learn logistic
regression (C=1). Everything is scored on the VALIDATION split; the test split
is never touched here. Cost per client uses the 3:1 ratio from common.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import common as cm

raw = pd.read_csv(cm.DATA_PATH); T = "default payment next month"
y = raw[T].to_numpy(); idx = np.arange(len(raw))
tv, te = train_test_split(idx, test_size=.2, stratify=y, random_state=cm.SEED)
tr, va = train_test_split(tv, test_size=.25, stratify=y[tv], random_state=cm.SEED)   # same split as train.py
B = [f"BILL_AMT{i}" for i in range(1,7)]; P = [f"PAY_AMT{i}" for i in range(1,7)]; S = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

def derived(df):
    d = pd.DataFrame(index=df.index); lim = df.LIMIT_BAL
    bills = df[B].clip(lower=0)
    d["util_last"] = (bills.BILL_AMT1 / lim).clip(0, 1.5)
    d["util_mean"] = (bills.mean(axis=1) / lim).clip(0, 1.5)
    # the payment made in month t settles the statement of month t+1 (PAY_AMT1 pays BILL_AMT2, ...)
    ratios = []
    for t in range(1, 6):
        owed, paid = bills[B[t]], df[P[t - 1]]
        ratios.append(paid.div(owed.where(owed > 0)).fillna(1.0).clip(0, 1))   # nothing owed -> fully covered
    ratios = pd.concat(ratios, axis=1)
    d["pay_ratio_mean"] = ratios.mean(axis=1); d["pay_ratio_last"] = ratios.iloc[:, 0]
    d["max_delay"] = df[S].max(axis=1).clip(lower=0)
    d["n_months_delayed"] = (df[S] >= 1).sum(axis=1)
    d["severe_delay_last"] = (df.PAY_0 >= 2).astype(int)
    d["log_limit"] = np.log(lim)
    return d

DER = derived(raw)

def make(base_drop, recode, extra=(), drop_raw_limit=False):
    X = raw.drop(columns=[T, "ID"] + list(base_drop)).copy()
    if recode and "EDUCATION" in X: X["EDUCATION"] = X["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    if recode and "MARRIAGE" in X: X["MARRIAGE"] = X["MARRIAGE"].replace({0: 3})
    for c in extra: X[c] = DER[c]
    return X

def score(X, C=1.0):
    cats = [c for c in ["EDUCATION", "MARRIAGE", "SEX"] if c in X]; nums = [c for c in X if c not in cats]
    ref = [X.iloc[tr][c].mode().iloc[0] for c in cats]
    steps = [("n", StandardScaler(), nums)]
    if cats: steps.append(("c", OneHotEncoder(drop=ref, handle_unknown="ignore", sparse_output=False), cats))
    pre = ColumnTransformer(steps)
    Xtr = pre.fit_transform(X.iloc[tr]); Xva = pre.transform(X.iloc[va])
    m = LogisticRegression(C=C, max_iter=5000).fit(Xtr, y[tr]); p = m.predict_proba(Xva)[:, 1]
    cost = min(((p >= t) & (y[va] == 0)).sum() * cm.COST_FP + ((p < t) & (y[va] == 1)).sum() * cm.COST_FN for t in np.linspace(.01, .99, 99)) / len(va)
    return roc_auc_score(y[va], p), log_loss(y[va], p), cost

D = ["util_last", "util_mean", "pay_ratio_mean", "pay_ratio_last", "max_delay", "n_months_delayed", "severe_delay_last", "log_limit"]
rows = []
def add(label, X): rows.append((label, *score(X)))
add("A  v0 as shipped (SEX out, MARRIAGE in, no recode)", make(["SEX"], False))
add("B  A + recode rare EDUCATION/MARRIAGE codes", make(["SEX"], True))
add("C  B + SEX back in           (price of excluding SEX)", make([], True))
add("D  B - MARRIAGE              (price of excluding MARRIAGE)", make(["SEX", "MARRIAGE"], True))
base = ["SEX", "MARRIAGE"]
add("E  D + all 8 derived features", make(base, True, D))
for g, cols in {"util (2)": ["util_last", "util_mean"], "pay_ratio (2)": ["pay_ratio_mean", "pay_ratio_last"],
                "delay (3)": ["max_delay", "n_months_delayed", "severe_delay_last"], "log_limit": ["log_limit"]}.items():
    add(f"   E minus {g}", make(base, True, [c for c in D if c not in cols]))
print(f"{'variant (validation, sklearn C=1, ' + f'{cm.COST_FN:g}:{cm.COST_FP:g} cost)':<62}{'AUC':>7}{'logloss':>9}{'cost/client':>13}")
for r in rows: print(f"{r[0]:<62}{r[1]:>7.4f}{r[2]:>9.4f}{r[3]:>13.4f}")
print(f"\nconstant policies on validation: approve-all = {y[va].mean() * cm.COST_FN:.4f} | reject-all = {(1 - y[va].mean()) * cm.COST_FP:.4f}")

print("\n=== candidate additions on top of F = delay(3) + log_limit (validation) ===")
bills_pos = raw[B].clip(lower=0)
CAND = pd.DataFrame(index=raw.index)
CAND["delay_last_eq1"] = (raw.PAY_0 == 1).astype(int)
CAND["ever_severe"] = (raw[S].max(axis=1) >= 2).astype(int)
CAND["delay_trend"] = (raw.PAY_0 - raw.PAY_6).clip(-4, 4)
CAND["bill_growth"] = ((raw.BILL_AMT1 - raw.BILL_AMT6) / raw.LIMIT_BAL).clip(-1.5, 1.5)
CAND["util_x_late"] = DER["util_last"] * (raw.PAY_0 >= 1)
CAND["zero_pay_months"] = ((raw[P[:5]].values == 0) & (bills_pos[B[1:]].values > 0)).sum(axis=1)
for c in CAND: DER[c] = CAND[c]
F = ["max_delay", "n_months_delayed", "severe_delay_last", "log_limit"]
b = score(make(base, True, F)); print(f"{'F (reference)':<28}{b[0]:>8.4f}{b[1]:>9.4f}{b[2]:>10.4f}")
for c in CAND.columns:
    s = score(make(base, True, F + [c]))
    print(f"F + {c:<24}{s[0]:>8.4f}{s[1]:>9.4f}{s[2]:>10.4f}   dAUC {s[0]-b[0]:+.4f}")

print("\n=== heavy-tailed amounts: signed-log transform of BILL_AMT*/PAY_AMT* (validation) ===")
def slog(s): return np.sign(s) * np.log1p(np.abs(s))
def with_logs(cols):
    X = make(base, True, F)
    for c in cols: X[c] = slog(raw[c])
    return X
ref = score(make(base, True, F))
print(f"{'F (reference, raw amounts)':<40}{ref[0]:>8.4f}{ref[1]:>9.4f}{ref[2]:>10.4f}")
for label, cols in [("signed-log PAY_AMT1-6", P), ("signed-log BILL_AMT1-6", B), ("signed-log both", B + P)]:
    s = score(with_logs(cols)); print(f"{label:<40}{s[0]:>8.4f}{s[1]:>9.4f}{s[2]:>10.4f}   dAUC {s[0]-ref[0]:+.4f}")
