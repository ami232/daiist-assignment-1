# %% [markdown]
# # Training pipeline: who will say yes to each other?
#
# Run with `uv run python main.py train`. Start to finish, no manual steps:
#
# 1. load the raw file, run the data checks that drive the cleaning decisions,
#    build one row per woman-man pair
# 2. build pre-event features
# 3. split by event (train / validation / test), check what a random split would have hidden
# 4. choose the L2 strength C, check which feature groups earn their place
# 5. choose learning rates for the two PyTorch versions
# 6. fit the same logistic regression three ways on the development events
# 7. compare on the held-out test events, against naive baselines, in euros
# 8. extra analyses: event planner, what attraction data would add, how much data is enough
# 9. save models, data and figures for app.ipynb (which never retrains) and REPORT.md

# %%
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from speeddating.data import (
    ATTRS6, INTERESTS, INTERESTS_ALL, ROOT, SCORECARD_COLS, SIGNUP_BLOCKS, build_pair_table, load_raw,
    looks_reputation,
)
from speeddating.evaluation import (
    COST_FN, COST_FP, bootstrap_ci, calibration_table, cost_curve, cost_threshold,
    evaluate, plan_events,
)
from speeddating.features import (
    FEATURE_GROUPS, candidate_additions, describe, engineer_features, group_of,
)
from speeddating.torch_models import (
    l2_strength, module_params, objective, predict_manual, predict_module,
    train_manual, train_module,
)

warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
# These full-batch matrices are small; one CPU thread avoids thread-management
# overhead during the 1,500-step loops without changing the model or objective.
torch.set_num_threads(1)

ART = ROOT / "artifacts"
FIG = ROOT / "figures"
ART.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
C3 = {"sklearn": "#2a6fdb", "manual": "#e8833a", "module": "#2ca58d"}
LABEL = {"sklearn": "scikit-learn", "manual": "PyTorch manual loop", "module": "PyTorch nn.Module + Adam"}
T0 = time.time()
summary: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def preprocessor() -> Pipeline:
    """Median imputation + standardisation, always fitted on training rows only."""
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])


def sk_model(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=10_000, tol=1e-8)


# %% [markdown]
# ## 1. Load, check, and build the pair table
# Each check below drives one cleaning decision (see REPORT.md):
# - which sign-up blocks are missing for whole waves (4_1 and 5_1: not used)
# - which interests measure the same thing (art vs museums: art dropped)
# - whether the preference points share one scale (rescaled to 100 in features.py)
# Missing answers stay NaN; the model pipeline fills them from training events only.

# %%
raw = load_raw()
people = raw.drop_duplicates("iid")    # sign-up answers repeat on every row of a person
log(f"{len(raw):,} date records, {raw['iid'].nunique()} people, {raw['wave'].nunique()} events; "
    f"{raw['pid'].isna().sum()} records without a partner id")

# Missing sign-up answers, per person and per question block
miss = pd.Series({name: people[cols].isna().any(axis=1).mean() * 100 for name, cols in SIGNUP_BLOCKS.items()})
late_blocks = list(SIGNUP_BLOCKS)[-2:]
miss_by_wave = pd.DataFrame({b: people.groupby("wave")[SIGNUP_BLOCKS[b]].apply(lambda d: d.isna().any(axis=1).mean())
                             for b in late_blocks})
log("% of people with a missing answer, by block:\n" + miss.round(1).to_string())

# Interests that measure the same thing (|r| > 0.8, per person)
corr = people[INTERESTS_ALL].corr()
high = [(a, b, round(float(corr.loc[a, b]), 3)) for i, a in enumerate(INTERESTS_ALL)
        for b in INTERESTS_ALL[i + 1:] if abs(corr.loc[a, b]) > 0.8]
dropped = {b for _, b, _ in high}
if [c for c in INTERESTS_ALL if c not in dropped] != INTERESTS:
    raise SystemExit(f"interest correlation check changed: {high}; update INTERESTS in speeddating/data.py")
log(f"interest pairs with |r| > 0.8: {high} -> dropped {sorted(dropped)}")

# Preference points: one scale for everyone?
pref_cols = [f"{a}1_1" for a in ATTRS6]
totals = people[pref_cols].sum(axis=1, min_count=6)
pref_check = pd.DataFrame({
    "share_with_decimals": people.groupby("wave")[pref_cols].apply(lambda d: (d.dropna() % 1 != 0).any(axis=1).mean()),
    "share_not_summing_to_100": (totals.sub(100).abs() > 1).groupby(people["wave"]).mean(),
    "min_total": totals.groupby(people["wave"]).min(), "max_total": totals.groupby(people["wave"]).max(),
})
log("preference points by wave:\n" + pref_check.round(2).T.to_string())

pairs = build_pair_table(raw)
y_all = pairs["match"].to_numpy()
log(f"{len(raw):,} date records -> {len(pairs):,} woman-man pairs, "
    f"{pairs['iid_W'].nunique() + pairs['iid_M'].nunique()} people, {pairs['wave'].nunique()} events, "
    f"{y_all.sum()} mutual matches ({y_all.mean():.1%})")
summary["data"] = {
    "date_records": int(len(raw)), "pairs": int(len(pairs)),
    "people": int(pairs["iid_W"].nunique() + pairs["iid_M"].nunique()),
    "women": int(pairs["iid_W"].nunique()), "men": int(pairs["iid_M"].nunique()),
    "events": int(pairs["wave"].nunique()), "matches": int(y_all.sum()),
    "match_rate": float(y_all.mean()),
    "yes_rate_women": float(pairs["dec_W"].mean()), "yes_rate_men": float(pairs["dec_M"].mean()),
    "records_without_partner": int(raw["pid"].isna().sum()),
    "people_without_signup": {s: int(pairs.loc[pairs[f"survey_missing_{s}"] == 1, f"iid_{s}"].nunique())
                              for s in ("W", "M")},
    "missing_by_block_pct": miss.to_dict(),
    "high_interest_correlations": high,
}

fig, ax = plt.subplots(1, 2, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
miss.iloc[::-1].plot.barh(ax=ax[0], color=["#c44" if v > 10 else "#2a6fdb" for v in miss.iloc[::-1]])
ax[0].set(xlabel="% of people with at least one missing answer", title="Missing sign-up answers, by question block")
(miss_by_wave * 100).plot(ax=ax[1], marker="o")
ax[1].set(xlabel="wave", ylabel="% of people missing", title="4_1 and 5_1: missing for whole waves")
ax[1].legend([b.split(" ")[0] for b in miss_by_wave.columns], frameon=False)
fig.tight_layout()
fig.savefig(FIG / "eda_missing.png")
plt.close(fig)

# %% [markdown]
# ## 2. Features (all known before the event)

# %%
X_all = engineer_features(pairs)
FEATURES = list(X_all.columns)
log(f"{len(FEATURES)} features: " + ", ".join(
    f"{g}={sum(group_of(c) == g for c in FEATURES)}" for g in FEATURE_GROUPS))

# %% [markdown]
# ## 3. Split by event, in date order
# The organiser scores the pairs of an *upcoming* event using what it learned
# from *past* events. So whole events go to one side of the split, in time
# order: waves 1-13 train, 14-16 validation (Oct 2003 - Feb 2004), 17-21 test
# (Feb - Apr 2004). The test events are only used once, at the end.

# %%
wave = pairs["wave"].to_numpy()
is_train, is_val, is_test = wave <= 13, (wave >= 14) & (wave <= 16), wave >= 17
is_dev = is_train | is_val
for name, m in [("train", is_train), ("validation", is_val), ("test", is_test)]:
    log(f"{name:>10}: waves {sorted(set(wave[m]))[0]}-{sorted(set(wave[m]))[-1]}, "
        f"{m.sum():,} pairs, match rate {y_all[m].mean():.3f}")
summary["split"] = {n: {"waves": sorted(int(w) for w in set(wave[m])), "pairs": int(m.sum()),
                        "matches": int(y_all[m].sum()), "match_rate": float(y_all[m].mean())}
                    for n, m in [("train", is_train), ("validation", is_val), ("test", is_test)]}

by_wave = pairs.groupby("wave").agg(match=("match", "mean"), her=("dec_W", "mean"), his=("dec_M", "mean"))
split_of = np.where(by_wave.index >= 17, "test", np.where(by_wave.index >= 14, "validation", "train"))
fig, ax = plt.subplots(figsize=(10, 3.3))
ax.bar(by_wave.index, by_wave["match"], color=[{"train": "#2a6fdb", "validation": "#e8833a", "test": "#2ca58d"}[s]
                                               for s in split_of])
ax.plot(by_wave.index, by_wave["her"], "o-", color="#d1495b", label="her yes rate")
ax.plot(by_wave.index, by_wave["his"], "o-", color="#555", label="his yes rate")
ax.set(xlabel="wave (event, in date order); bars = mutual match rate, blue train / orange validation / green test",
       ylabel="rate", title="The target varies a lot between nights", xticks=by_wave.index)
ax.set_ylim(0, 0.8)
ax.legend(frameon=False, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "target_by_wave.png")
plt.close(fig)

# %% [markdown]
# ### What a random split would have hidden
# Same model, same development data, 5-fold cross-validation two ways:
# shuffled rows versus whole events. With shuffled rows the same people sit in
# both training and evaluation folds, so the model can recognise them.
# The event folds are fixed (events in date order, dealt round-robin into 5
# folds) so the number does not depend on the scikit-learn version.

# %%
X_dev, y_dev, g_dev = X_all[is_dev], y_all[is_dev], wave[is_dev]


def cv_auc(splitter, groups=None, C=0.003) -> list[float]:
    out = []
    for tr, te in splitter.split(X_dev, y_dev, groups):
        m = Pipeline([("pre", preprocessor()), ("lr", sk_model(C))]).fit(X_dev.iloc[tr], y_dev[tr])
        out.append(roc_auc_score(y_dev[te], m.predict_proba(X_dev.iloc[te])[:, 1]))
    return out


rows_cv = cv_auc(KFold(5, shuffle=True, random_state=SEED))
dev_waves = np.unique(g_dev)
event_fold = pd.Series(np.arange(len(dev_waves)) % 5, index=dev_waves)
event_cv = cv_auc(PredefinedSplit(event_fold.reindex(g_dev).to_numpy()))
# how many evaluation pairs involve someone already seen in training (random split)?
seen = []
for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X_dev):
    d_tr, d_te = pairs[is_dev].iloc[tr], pairs[is_dev].iloc[te]
    seen.append(np.mean(d_te["iid_W"].isin(d_tr["iid_W"]) | d_te["iid_M"].isin(d_tr["iid_M"])))
summary["leakage_check"] = {
    "random_rows_auc_mean": float(np.mean(rows_cv)), "random_rows_auc_sd": float(np.std(rows_cv)),
    "by_event_auc_mean": float(np.mean(event_cv)), "by_event_auc_sd": float(np.std(event_cv)),
    "share_eval_pairs_with_person_seen_in_training": float(np.mean(seen)),
}
log(f"5-fold CV AUC, random rows: {np.mean(rows_cv):.3f} | whole events: {np.mean(event_cv):.3f} "
    f"({np.mean(seen):.0%} of random-split evaluation pairs involve someone seen in training)")

# %% [markdown]
# ## 4. Regularisation strength C (the one hyperparameter of the model itself)
# Leave-one-event-out over the 16 development events, scored by log loss.
# Log loss, not accuracy or AUC, because the decision threshold below is
# derived from the probabilities, so they need to be calibrated. With only
# three validation events a single split is too noisy to pick C from.

# %%
C_GRID = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0]


def lowo_scores(cols: list[str], C: float, X=None) -> dict:
    X = X_dev[cols] if X is None else X
    oof = np.zeros(len(y_dev))
    for tr, te in LeaveOneGroupOut().split(X, y_dev, g_dev):
        m = Pipeline([("pre", preprocessor()), ("lr", sk_model(C))]).fit(X.iloc[tr], y_dev[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    per_event = [roc_auc_score(y_dev[g_dev == g], oof[g_dev == g]) for g in np.unique(g_dev)]
    return {"log_loss": log_loss(y_dev, oof), "auc": roc_auc_score(y_dev, oof),
            "mean_event_auc": float(np.mean(per_event))}


c_search = []
for C in C_GRID:
    c_search.append({"C": C, **lowo_scores(FEATURES, C)})
c_search = pd.DataFrame(c_search)
BEST_C = float(c_search.loc[c_search["log_loss"].idxmin(), "C"])
summary["c_search"] = c_search.to_dict("records")
summary["best_C"] = BEST_C
log("C search (leave-one-event-out):\n" + c_search.round(4).to_string(index=False))
log(f"chosen C = {BEST_C}")

# %% [markdown]
# ### Do the feature groups earn their place?
# Drop one group at a time, same C, same leave-one-event-out scoring.

# %%
ablation = [{"dropped": "nothing (all features)", "n_features": len(FEATURES),
             **lowo_scores(FEATURES, BEST_C)}]
for g in FEATURE_GROUPS:
    cols = [c for c in FEATURES if group_of(c) != g]
    ablation.append({"dropped": g, "n_features": len(cols), **lowo_scores(cols, BEST_C)})
ablation = pd.DataFrame(ablation)
summary["ablation"] = ablation.to_dict("records")
log("feature-group ablation:\n" + ablation.round(4).to_string(index=False))

# Blocks I tried adding on top and dropped, same scoring.
tried = []
for name, extra in candidate_additions(pairs).items():
    cols = FEATURES + list(extra.columns)
    tried.append({"added": name, "n_features": len(cols),
                  **lowo_scores(cols, BEST_C, X=pd.concat([X_all, extra], axis=1).loc[is_dev, cols])})
tried = pd.DataFrame(tried)
summary["tried_and_dropped"] = tried.to_dict("records")
log("blocks tried on top of the final set:\n" + tried.round(4).to_string(index=False))

# %% [markdown]
# ## 5. Learning rates for the two PyTorch versions
# Fit on train (waves 1-13), with lam = 1 / (C n) so the objective is the
# sklearn one. The learning rate is an optimiser setting, not a model
# setting: its job is to reach the minimum of the training objective. So I
# pick the rate with the lowest final training objective that did not
# diverge, and report validation log loss alongside. Picking on validation
# loss instead could reward a rate that is too small (it stops early, which
# acts as extra regularisation) and I would silently be comparing a
# different model to sklearn.

# %%
EPOCHS = 1500
pre_tr = preprocessor().fit(X_all[is_train])
Xs_tr, Xs_val = pre_tr.transform(X_all[is_train]), pre_tr.transform(X_all[is_val])
y_tr, y_val = y_all[is_train].astype(float), y_all[is_val].astype(float)
lam_tr = l2_strength(BEST_C, len(y_tr))
sk_tr = sk_model(BEST_C).fit(Xs_tr, y_tr)
J_sk_tr = objective(Xs_tr, y_tr, sk_tr.coef_.ravel(), sk_tr.intercept_[0], lam_tr)

lr_search = []
for lr in [0.01, 0.1, 0.5, 1.0, 2.0, 4.0]:
    params, hist = train_manual(Xs_tr, y_tr, lam_tr, lr, EPOCHS)
    w, b = params["w"].numpy().astype(float), float(params["b"].item())
    J = objective(Xs_tr, y_tr, w, b, lam_tr)
    lr_search.append({"method": "manual", "lr": lr, "train_objective_gap": J - J_sk_tr,
                      "val_log_loss": log_loss(y_val, np.clip(predict_manual(params, Xs_val), 1e-7, 1 - 1e-7)),
                      "max_coef_diff_vs_sklearn": float(np.abs(w - sk_tr.coef_.ravel()).max()),
                      "diverged": bool(not np.isfinite(J) or J > hist[0])})
for lr in [0.001, 0.01, 0.05, 0.1]:
    model, hist = train_module(Xs_tr, y_tr, lam_tr, lr, EPOCHS, seed=SEED)
    w, b = module_params(model)
    J = objective(Xs_tr, y_tr, w, b, lam_tr)
    lr_search.append({"method": "module", "lr": lr, "train_objective_gap": J - J_sk_tr,
                      "val_log_loss": log_loss(y_val, np.clip(predict_module(model, Xs_val), 1e-7, 1 - 1e-7)),
                      "max_coef_diff_vs_sklearn": float(np.abs(w - sk_tr.coef_.ravel()).max()),
                      "diverged": bool(not np.isfinite(J) or J > hist[0])})
lr_search = pd.DataFrame(lr_search)
ok = lr_search[~lr_search["diverged"]]
BEST_LR = {m: float(ok[ok["method"] == m].sort_values("train_objective_gap")["lr"].iloc[0])
           for m in ("manual", "module")}
summary["lr_search"] = lr_search.to_dict("records")
summary["best_lr"] = BEST_LR
summary["epochs"] = EPOCHS
log("learning-rate search (train -> validation):\n" + lr_search.to_string(index=False))
log(f"chosen learning rates: {BEST_LR}")

# %% [markdown]
# ## 6. Final fit, three ways, on all 16 development events
# Same rows, same preprocessing, same objective, same C.

# %%
pre = preprocessor().fit(X_all[is_dev])
Xs_dev, Xs_test = pre.transform(X_all[is_dev]), pre.transform(X_all[is_test])
y_dev_f, y_test = y_dev.astype(float), y_all[is_test]
lam = l2_strength(BEST_C, len(y_dev_f))

t = time.time()
sk = sk_model(BEST_C).fit(Xs_dev, y_dev_f)
fit_time = {"sklearn": time.time() - t}
t = time.time()
manual_params, manual_hist = train_manual(Xs_dev, y_dev_f, lam, BEST_LR["manual"], EPOCHS)
fit_time["manual"] = time.time() - t
t = time.time()
module, module_hist = train_module(Xs_dev, y_dev_f, lam, BEST_LR["module"], EPOCHS, seed=SEED)
fit_time["module"] = time.time() - t

coefs = pd.DataFrame({
    "feature": FEATURES,
    "group": [group_of(c) for c in FEATURES],
    "description": [describe(c) for c in FEATURES],
    "sklearn": sk.coef_.ravel(),
    "manual": manual_params["w"].numpy().astype(float),
    "module": module_params(module)[0],
})
intercepts = {"sklearn": float(sk.intercept_[0]), "manual": float(manual_params["b"].item()),
              "module": module_params(module)[1]}
J_final = {
    "sklearn": objective(Xs_dev, y_dev_f, coefs["sklearn"].to_numpy(), intercepts["sklearn"], lam),
    "manual": objective(Xs_dev, y_dev_f, coefs["manual"].to_numpy(), intercepts["manual"], lam),
    "module": objective(Xs_dev, y_dev_f, coefs["module"].to_numpy(), intercepts["module"], lam),
}
log(f"final training objective: " + ", ".join(f"{k}={v:.8f}" for k, v in J_final.items()))

# %% [markdown]
# ## 7. Test events: three implementations vs naive baselines

# %%
T_STAR = cost_threshold(COST_FP, COST_FN)
base_rate = float(y_dev.mean())
preds = {
    "sklearn": sk.predict_proba(Xs_test)[:, 1],
    "manual": predict_manual(manual_params, Xs_test),
    "module": predict_module(module, Xs_test),
}
results = {name: evaluate(y_test, p, T_STAR) for name, p in preds.items()}
results["baseline: base rate (schedule nobody)"] = evaluate(y_test, np.full(len(y_test), base_rate), T_STAR)
results["baseline: schedule everyone"] = evaluate(y_test, np.ones(len(y_test)), T_STAR)
results_df = pd.DataFrame(results).T


def mean_auc_within(df: pd.DataFrame, p: np.ndarray, key: str) -> float:
    """Average AUC computed inside each group (event, woman or man). Inside one
    woman's list, for example, it asks: does the model rank the men she said
    yes to (and who said yes back) above the others?"""
    aucs = [roc_auc_score(g["match"], g["p"]) for _, g in df.assign(p=p).groupby(key)
            if 0 < g["match"].mean() < 1]
    return float(np.mean(aucs))


for name, p in preds.items():
    results_df.loc[name, "mean_event_auc"] = mean_auc_within(pairs[is_test], p, "wave")
    results_df.loc[name, "within_woman_auc"] = mean_auc_within(pairs[is_test], p, "iid_W")
    results_df.loc[name, "within_man_auc"] = mean_auc_within(pairs[is_test], p, "iid_M")
log(f"test set, threshold {T_STAR:.2f} (FP EUR {COST_FP:.0f}, FN EUR {COST_FN:.0f}):\n"
    + results_df[["roc_auc", "mean_event_auc", "within_woman_auc", "within_man_auc", "pr_auc",
                  "log_loss", "brier", "precision", "recall", "share_scheduled", "cost_eur"]]
    .round(4).to_string())

agreement = {}
names = list(preds)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        agreement[f"{a} vs {b}"] = {
            "max_abs_coef_diff": float(np.abs(coefs[a] - coefs[b]).max()),
            "intercept_diff": float(abs(intercepts[a] - intercepts[b])),
            "max_abs_prob_diff_test": float(np.abs(preds[a] - preds[b]).max()),
            "same_decision_share_test": float(np.mean((preds[a] >= T_STAR) == (preds[b] >= T_STAR))),
            "prob_correlation_test": float(np.corrcoef(preds[a], preds[b])[0, 1]),
        }
log("agreement:\n" + pd.DataFrame(agreement).T.to_string())
ci = bootstrap_ci(y_test, preds, n_boot=1000, seed=SEED)
log(f"bootstrap 95% CI of test AUC: {ci['auc_ci']}")

summary["test"] = {k: {kk: (None if pd.isna(vv) else float(vv)) for kk, vv in v.items()}
                   for k, v in results_df.to_dict("index").items()}
summary["agreement"] = agreement
summary["objective"] = J_final
summary["intercepts"] = intercepts
summary["bootstrap"] = ci
summary["fit_seconds"] = fit_time
summary["threshold"] = {"cost_fp": COST_FP, "cost_fn": COST_FN, "t_star": T_STAR,
                        "dev_base_rate": base_rate}

# %% [markdown]
# ### Is the cost-derived threshold sensible on data the model has not seen?
# The rule t* = c_fp / (c_fp + c_fn) assumes calibrated probabilities. Check it
# with the train-only sklearn model on the validation events: where is the
# cost actually lowest?

# %%
val_curve = cost_curve(y_val, sk_tr.predict_proba(Xs_val)[:, 1], COST_FP, COST_FN)
best_val_t = float(val_curve.loc[val_curve["cost_eur"].idxmin(), "threshold"])
summary["threshold"]["validation_cost_minimising_t"] = best_val_t
summary["calibration_test"] = {k: calibration_table(y_test, p).to_dict("records") for k, p in preds.items()}
log(f"theory threshold {T_STAR:.3f}, validation cost minimum at {best_val_t:.3f}")

# %% [markdown]
# ## 8a. Event planner: the decision the model is for
# The night is cut short so each person only meets a share of the other side.
# The model's plan picks the set of dates with the highest total predicted
# probability (at most k dates per person). Compare the mutual matches it
# would have kept on the test events with random seating plans that hold the
# same number of dates, and with a perfect plan.

# %%
test_pairs = pairs[is_test].reset_index(drop=True)
planner = {}
for share in (0.3, 0.5, 0.7):
    pl = plan_events(test_pairs, preds["sklearn"], share=share, n_random=200, seed=SEED)
    rand_tot = np.array(pl["random_draws"].tolist()).sum(axis=0)   # total over events, per random draw
    planner[str(share)] = {
        "all_matches": int(pl["all_matches"].sum()), "dates_held": int(pl["dates_held"].sum()),
        "model_plan": int(pl["model_plan"].sum()), "random_plan": float(rand_tot.mean()),
        "random_plan_sd": float(rand_tot.std()), "perfect_plan": int(pl["perfect_plan"].sum()),
        "share_random_draws_at_least_as_good": float(np.mean(rand_tot >= pl["model_plan"].sum())),
    }
    if share == 0.5:
        plan = pl.drop(columns="random_draws")
summary["planner"] = {"by_share": planner, "by_event_half": plan.to_dict("records")}
log("seating plans on test events (matches kept):\n" + pd.DataFrame(planner).T.round(2).to_string())

# %% [markdown]
# ## 8b. The missing ingredient: physical attraction
# None of the deployable features describe how someone looks. How much would
# that information be worth? Same model (sklearn, C re-tuned per variant with
# the same leave-one-event-out search), trained on development events, scored
# on the test events, adding information step by step:
#
# 1. deployable model (sign-up data only)
# 2. + a "looks reputation": average attractiveness score each person got
#    from everyone *else* they met that night. Stand-in for a photo rating.
# 3. + how attractive the two people found *each other* (scorecard)
# 4. + the full scorecard (sincere, intelligent, fun, ambitious, shared interests, overall like)
#
# Steps 2-4 are not available before the event. They measure the size of the gap.

# %%
X_ext = pd.concat([X_all, looks_reputation(raw, pairs), pairs.filter(like="card_")], axis=1)
rep_cols = ["looks_rep_W", "looks_rep_M"]
variants = {
    "1. sign-up data only (deployable)": FEATURES,
    "2. + looks reputation from other partners": FEATURES + rep_cols,
    "3. + how attractive they found each other": FEATURES + rep_cols + ["card_attr_W", "card_attr_M"],
    "4. + the full scorecard": FEATURES + rep_cols + [f"card_{a}_{s}" for s in ("W", "M") for a in SCORECARD_COLS],
}
whatif = []
for name, cols in variants.items():
    best = min(C_GRID, key=lambda C: lowo_scores(cols, C, X=X_ext.loc[is_dev, cols])["log_loss"])
    m = Pipeline([("pre", preprocessor()), ("lr", sk_model(best))]).fit(X_ext.loc[is_dev, cols], y_dev)
    p = m.predict_proba(X_ext.loc[is_test, cols])[:, 1]
    r = evaluate(y_test, p, T_STAR)
    pl = plan_events(pairs[is_test].reset_index(drop=True), p, share=0.5, n_random=1, seed=SEED)
    whatif.append({"information": name, "C": best, "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
                   "log_loss": r["log_loss"], "cost_eur": r["cost_eur"],
                   "planner_matches": int(pl["model_plan"].sum())})
whatif = pd.DataFrame(whatif)
log("what attraction information would add (test events):\n" + whatif.round(3).to_string(index=False))

# How good is the only looks signal we DO have before the event (self-rating)?
per_person = raw.groupby("iid").agg(gender=("gender", "first"), self_attr=("attr3_1", "first"),
                                looks_received=("attr_o", "mean"), yes_received=("dec_o", "mean"))
corr_self = {
    "all": float(per_person[["self_attr", "looks_received"]].corr().iloc[0, 1]),
    "women": float(per_person[per_person.gender == 0][["self_attr", "looks_received"]].corr().iloc[0, 1]),
    "men": float(per_person[per_person.gender == 1][["self_attr", "looks_received"]].corr().iloc[0, 1]),
}
corr_received = float(per_person[["looks_received", "yes_received"]].corr().iloc[0, 1])
pair_corr = {
    "her_yes_vs_his_self_rated_looks": float(pairs[["dec_W", "attr3_1_M"]].corr().iloc[0, 1]),
    "her_yes_vs_her_rating_of_his_looks": float(pairs[["dec_W", "card_attr_W"]].corr().iloc[0, 1]),
    "his_yes_vs_her_self_rated_looks": float(pairs[["dec_M", "attr3_1_W"]].corr().iloc[0, 1]),
    "his_yes_vs_his_rating_of_her_looks": float(pairs[["dec_M", "card_attr_M"]].corr().iloc[0, 1]),
}
# Whose half of the match can the sign-up form predict? Same features, same
# pipeline and C, but the target is one person's yes instead of the mutual match.
side = {}
for tgt, label in [("dec_W", "her yes"), ("dec_M", "his yes")]:
    yy = pairs[tgt].to_numpy()
    m = Pipeline([("pre", preprocessor()), ("lr", sk_model(BEST_C))]).fit(X_all[is_dev], yy[is_dev])
    side[label] = {"test_auc": float(roc_auc_score(yy[is_test], m.predict_proba(X_all[is_test])[:, 1])),
                   "yes_rate": float(yy.mean())}
summary["attraction"] = {"whatif": whatif.to_dict("records"),
                         "corr_self_rated_vs_received_looks": corr_self,
                         "corr_received_looks_vs_yes_received": corr_received,
                         "pair_level_corr": pair_corr,
                         "one_sided_models": side,
                         "people": per_person.dropna().reset_index().to_dict("records")}
log(f"self-rated looks vs looks others gave: r={corr_self['all']:.2f} "
    f"(women {corr_self['women']:.2f}, men {corr_self['men']:.2f}); "
    f"looks received vs yes received: r={corr_received:.2f}")
log(f"pair-level correlations: {json.dumps({k: round(v, 3) for k, v in pair_corr.items()})}")
log(f"one-sided models, test AUC: her yes {side['her yes']['test_auc']:.3f}, "
    f"his yes {side['his yes']['test_auc']:.3f}")

# %% [markdown]
# ## 8c. How much data is enough?
# Train on random subsets of the 13 training events (whole events), score on
# the 3 validation events, 40 draws per size. The test events stay untouched.

# %%
rng = np.random.default_rng(SEED)
train_waves = np.array(sorted(set(wave[is_train])))
lc_rows = []
for k in [2, 4, 6, 8, 10, 12, 13]:
    for draw in range(40 if k < 13 else 1):
        ws = rng.choice(train_waves, size=k, replace=False)
        m_rows = np.isin(wave, ws)
        if y_all[m_rows].sum() < 5:
            continue
        m = Pipeline([("pre", preprocessor()), ("lr", sk_model(BEST_C))]).fit(X_all[m_rows], y_all[m_rows])
        lc_rows.append({"events": k, "pairs": int(m_rows.sum()),
                        "train_auc": roc_auc_score(y_all[m_rows], m.predict_proba(X_all[m_rows])[:, 1]),
                        "val_auc": roc_auc_score(y_val, m.predict_proba(X_all[is_val])[:, 1])})
lc = pd.DataFrame(lc_rows)
lc_sum = lc.groupby("events").agg(pairs=("pairs", "mean"), train_auc=("train_auc", "mean"),
                                  val_auc=("val_auc", "mean"), val_auc_sd=("val_auc", "std")).reset_index()
half = lc_sum[lc_sum["events"] >= 6]
slope = float(np.polyfit(np.log2(half["pairs"]), half["val_auc"], 1)[0])
n_pos_dev = int(y_dev.sum())
summary["learning_curve"] = {"points": lc.to_dict("records"), "summary": lc_sum.to_dict("records"),
                             "auc_gain_per_doubling": slope,
                             "events_per_variable_dev": n_pos_dev / len(FEATURES),
                             "positives_dev": n_pos_dev}
log("learning curve:\n" + lc_sum.round(3).to_string(index=False))
log(f"validation AUC gain per doubling of training pairs (6+ events): {slope:+.3f}; "
    f"matches per feature in development data: {n_pos_dev / len(FEATURES):.1f}")

# %% [markdown]
# ## 9. Save everything the dashboard needs

# %%
joblib.dump(pre, ART / "preprocessor.joblib")
joblib.dump(sk, ART / "sklearn_logreg.joblib")
torch.save({"w": manual_params["w"], "b": manual_params["b"], "lam": lam,
            "lr": BEST_LR["manual"], "epochs": EPOCHS}, ART / "torch_manual.pt")
torch.save({"state_dict": module.state_dict(), "n_features": len(FEATURES), "lam": lam,
            "lr": BEST_LR["module"], "epochs": EPOCHS}, ART / "torch_module.pt")

keep = ["iid_W", "iid_M", "wave", "dec_W", "dec_M", "match"]
pairs.loc[is_dev, keep].join(X_all[is_dev]).to_csv(ART / "dev_pairs.csv", index=False)
pairs.loc[is_test, keep].join(X_all[is_test]).to_csv(ART / "test_pairs.csv", index=False)
pd.DataFrame({"iid_W": pairs.loc[is_test, "iid_W"].to_numpy(), "iid_M": pairs.loc[is_test, "iid_M"].to_numpy(),
              **{f"p_{k}": v for k, v in preds.items()}}).to_csv(ART / "test_predictions.csv", index=False)
coefs.to_csv(ART / "coefficients.csv", index=False)
summary["features"] = FEATURES
summary["loss_history"] = {"manual": manual_hist[::10], "module": module_hist[::10], "every": 10}
summary["sklearn_objective_final"] = J_final["sklearn"]
with open(ART / "summary.json", "w") as f:
    json.dump(summary, f, indent=1, default=float)
log(f"saved artifacts to {ART.relative_to(ROOT)}/")

# %% [markdown]
# ### Figures for REPORT.md
# (eda_missing.png and target_by_wave.png are saved in sections 1 and 3.)

# %%
STYLE = {"sklearn": dict(lw=3.5, ls="-"), "manual": dict(lw=1.8, ls="--"), "module": dict(lw=1.0, ls=":")}

# Three implementations: same weights, same minimum
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
lim = np.abs(coefs[list(LABEL)].to_numpy()).max() * 1.1
ax[0].plot([-lim, lim], [-lim, lim], color="#999", lw=0.8, ls="--")
ax[0].scatter(coefs["sklearn"], coefs["manual"], s=22, color=C3["manual"], label=LABEL["manual"])
ax[0].scatter(coefs["sklearn"], coefs["module"], s=8, color=C3["module"], label=LABEL["module"])
ax[0].set(xlabel="scikit-learn coefficient", ylabel="PyTorch coefficient", title="Same weights, three implementations")
ax[0].legend(frameon=False, fontsize=7)
for k, h in [("manual", manual_hist), ("module", module_hist)]:
    ax[1].plot(np.maximum(np.array(h) - J_final["sklearn"], 1e-9), color=C3[k], label=LABEL[k])
ax[1].set(yscale="log", xlabel="epoch", ylabel="objective minus scikit-learn's minimum",
          title="Both PyTorch versions converge to the same minimum")
ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "three_implementations.png")
plt.close(fig)

# Calibration and business cost on the test events
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for k, p in preds.items():
    cal = calibration_table(y_test, p)
    ax[0].plot(cal["mean_pred"], cal["match_rate"], marker="o", color=C3[k], label=LABEL[k], **STYLE[k])
    cc = cost_curve(y_test, p, COST_FP, COST_FN)
    ax[1].plot(cc["threshold"], cc["cost_eur"], color=C3[k], label=LABEL[k], **STYLE[k])
ax[0].plot([0, 0.35], [0, 0.35], color="#999", lw=0.8, ls="--")
ax[0].set(xlim=(0, 0.35), ylim=(0, 0.35), xlabel="predicted match probability (bin mean)",
          ylabel="observed match rate", title="Predicted vs actual, test events (8 bins)")
ax[1].axhline(results["baseline: schedule everyone"]["cost_eur"], color="#999", ls=":", label="schedule everyone")
ax[1].axhline(results["baseline: base rate (schedule nobody)"]["cost_eur"], color="#666", ls="--", label="schedule nobody")
ax[1].axvline(T_STAR, color="#333", lw=0.8)
ax[1].set(xlim=(0, 0.5), xlabel="decision threshold", ylabel="cost on test events (EUR)",
          title=f"Business cost (FP EUR {COST_FP:.0f}, FN EUR {COST_FN:.0f}); vertical line = {T_STAR:.1f}")
for a in ax:
    a.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "calibration_and_cost.png")
plt.close(fig)

# Seating plans, half the room
fig, ax = plt.subplots(figsize=(7, 3.3))
xs = np.arange(len(plan))
ax.bar(xs - 0.27, plan["random_plan"], 0.27, color="#bbb", label="random plan (mean)")
ax.bar(xs, plan["model_plan"], 0.27, color="#2a6fdb", label="model plan")
ax.bar(xs + 0.27, plan["perfect_plan"], 0.27, color="#2ca58d", label="perfect plan")
ax.set_xticks(xs, [f"wave {w}" for w in plan["wave"]])
ax.set(ylabel="mutual matches kept", title="Everyone meets half the room")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "planner.png")
plt.close(fig)

# The attraction gap
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
ax[0].barh(whatif["information"][::-1], whatif["roc_auc"][::-1], color=["#8d8d8d", "#ababab", "#c9c9c9", "#2a6fdb"])
ax[0].axvline(0.5, color="#666", lw=0.8, ls="--")
for i, v in enumerate(whatif["roc_auc"][::-1]):
    ax[0].text(v + 0.005, i, f"{v:.2f}", va="center")
ax[0].set(xlim=(0.45, 0.9), xlabel="test ROC AUC", title="What seeing attraction would add")
pp = per_person.dropna()
jit = np.random.default_rng(0).normal(0, 0.12, len(pp))
for gnd, col, name in [(0, "#d1495b", "women"), (1, "#2a6fdb", "men")]:
    m = (pp["gender"] == gnd).to_numpy()
    ax[1].scatter(pp["self_attr"][m] + jit[m], pp["looks_received"][m], s=7, alpha=0.5, color=col,
                  label=f"{name} (r = {corr_self[name]:.2f})")
ax[1].set(xlabel="self-rated attractiveness (sign-up)", ylabel="mean attractiveness rating received",
          title="Self-ratings say little about looks")
ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "attraction_gap.png")
plt.close(fig)

# How much data is enough
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.scatter(lc["pairs"], lc["val_auc"], s=6, alpha=0.3, color="#2a6fdb")
ax.plot(lc_sum["pairs"], lc_sum["val_auc"], marker="o", color="#2a6fdb", label="validation AUC (mean of 40 draws)")
ax.plot(lc_sum["pairs"], lc_sum["train_auc"], marker="o", color="#999", label="training AUC")
ax.axhline(0.5, color="#666", lw=0.8, ls="--")
ax.set(xscale="log", xlabel="training pairs (log scale, whole events)", ylabel="ROC AUC",
       title="More events help, slowly")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "learning_curve.png")
plt.close(fig)
log("saved figures to figures/ - done")
