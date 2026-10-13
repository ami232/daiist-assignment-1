"""Feature engineering: everything here is known BEFORE the event.

Rule I used to decide what is allowed in: could the organiser have this
number on the day before the event, from the sign-up form and the event
plan? Scorecard ratings (attractive, fun, like...) fail that test, so they
never enter the model.

Features come in three groups so I can switch each one off in an ablation:
  profile - one person's answers (age, goals, self-ratings...)
  pair    - how the two answers relate (age gap, shared interests, whether
            he has what she says she wants...)
  event   - how many men attend that night (= partners each woman meets)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from speeddating.data import ATTRS5, ATTRS6, INTERESTS

SIDES = ("W", "M")


def preference_weights(pairs: pd.DataFrame, who: str, block: str = "1_1") -> pd.DataFrame:
    """Six 'what matters to me' answers rescaled so each person's weights sum to 100.

    Waves 6-9 first rated each item 1-10; the published file already turns those
    ratings into shares (with decimals), but some people's points still do not add
    up to 100. Dividing by the row sum fixes both. Decimals are kept."""
    cols = [f"{a}{block}_{who}" for a in ATTRS6]
    w = pairs[cols].astype(float)
    total = w.sum(axis=1, min_count=6).replace(0, np.nan)
    w = w.div(total, axis=0) * 100
    w.columns = ATTRS6
    return w


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(axis=1) / den, np.nan)


def engineer_features(pairs: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=pairs.index)
    other = {"W": "M", "M": "W"}

    # ---------------- profile: one person at a time ----------------
    for s in SIDES:
        X[f"age_{s}"] = pairs[f"age_{s}"]
        X[f"date_freq_{s}"] = pairs[f"date_{s}"]          # 1 = several times a week ... 7 = almost never
        X[f"go_out_{s}"] = pairs[f"go_out_{s}"]           # same 1-7 scale, any going out
        X[f"exp_happy_{s}"] = pairs[f"exphappy_{s}"]      # expected happiness with the night, 1-10
        goal = pairs[f"goal_{s}"]
        # goal is a category, not a scale: keep the three answers that differ in intent
        X[f"goal_serious_{s}"] = (goal == 4).astype(float).where(goal.notna())
        X[f"goal_get_date_{s}"] = (goal == 3).astype(float).where(goal.notna())
        X[f"goal_fun_night_{s}"] = (goal == 1).astype(float).where(goal.notna())
        for a in ATTRS5:
            X[f"self_{a}_{s}"] = pairs[f"{a}3_1_{s}"]     # self-rating 1-10
        X[f"wants_looks_{s}"] = preference_weights(pairs, s)["attr"]
        X[f"survey_missing_{s}"] = pairs[f"survey_missing_{s}"].astype(float)

    # ---------------- pair: compatibility ----------------
    iw = pairs[[f"{i}_W" for i in INTERESTS]].to_numpy(dtype=float)
    im = pairs[[f"{i}_M" for i in INTERESTS]].to_numpy(dtype=float)
    X["interest_corr"] = _row_corr(iw, im)                      # same shape of interests?
    X["interest_gap"] = np.abs(iw - im).mean(axis=1)            # same intensity?
    X["age_gap"] = pairs["age_M"] - pairs["age_W"]              # positive = he is older
    X["age_gap_abs"] = X["age_gap"].abs()
    X["same_field"] = (pairs["field_cd_W"] == pairs["field_cd_M"]).astype(float).where(
        pairs["field_cd_W"].notna() & pairs["field_cd_M"].notna())
    X["looks_gap"] = pairs["attr3_1_M"] - pairs["attr3_1_W"]    # self-rated looks, him minus her

    # "Does the partner have what I say I want?"  My stated weights on the five
    # rateable attributes x the partner's self-ratings (1-10 scale).
    for s in SIDES:
        w = preference_weights(pairs, s)[ATTRS5]
        w = w.div(w.sum(axis=1), axis=0)
        partner_self = pairs[[f"{a}3_1_{other[s]}" for a in ATTRS5]].to_numpy(dtype=float)
        X[f"pref_fit_{s}"] = (w.to_numpy() * partner_self).sum(axis=1)
        X.loc[w.isna().any(axis=1) | np.isnan(partner_self).any(axis=1), f"pref_fit_{s}"] = np.nan

    # Same race only matters to people who said it matters: interaction term.
    X["same_race"] = pairs["samerace"].astype(float)
    for s in SIDES:
        X[f"race_mismatch_{s}"] = (1 - pairs["samerace"]) * pairs[f"imprace_{s}"]

    # ---------------- event ----------------
    X["event_size"] = pairs["event_size"].astype(float)
    return X


FIELD_GROUPS = {  # 18 field codes -> a few broad groups (the rest = "other")
    1: "law", 2: "stem", 3: "social", 4: "medicine", 5: "stem", 6: "arts_humanities",
    7: "arts_humanities", 8: "business", 9: "social", 10: "stem", 11: "social", 13: "social",
    14: "arts_humanities", 15: "arts_humanities", 16: "arts_humanities", 17: "arts_humanities",
}


def candidate_additions(pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Feature blocks I tried on top of the final set and did NOT keep
    (see the 'tried and dropped' table in train.py and REPORT.md)."""
    out = {}
    out["raw interest levels (12 per person)"] = pd.DataFrame(
        {f"int_{i}_{s}": pairs[f"{i}_{s}"] for s in SIDES for i in INTERESTS}, index=pairs.index)
    out["all six preference weights per person"] = pd.concat(
        [preference_weights(pairs, s).drop(columns="attr").add_prefix("pref_").add_suffix(f"_{s}")
         for s in SIDES], axis=1)
    out["field of study, 6 groups per person"] = pd.DataFrame(
        {f"field_{g}_{s}": (pairs[f"field_cd_{s}"].map(FIELD_GROUPS) == g).astype(float)
         .where(pairs[f"field_cd_{s}"].notna())
         for s in SIDES for g in sorted(set(FIELD_GROUPS.values()))}, index=pairs.index)
    out["beliefs about what the other side wants"] = pd.concat(
        [preference_weights(pairs, s, block="2_1").add_prefix("belief_").add_suffix(f"_{s}")
         for s in SIDES], axis=1)
    return out


FEATURE_GROUPS = {
    "profile": lambda c: c.startswith((
        "age_W", "age_M", "date_freq", "go_out", "exp_happy", "goal_", "self_",
        "wants_looks", "survey_missing")),
    "pair": lambda c: c.startswith((
        "interest_", "age_gap", "same_field", "looks_gap", "pref_fit", "same_race", "race_mismatch")),
    "event": lambda c: c == "event_size",
}


def group_of(col: str) -> str:
    for g, test in FEATURE_GROUPS.items():
        if test(col):
            return g
    raise KeyError(col)


FEATURE_INFO = {
    "age": "Age in years",
    "date_freq": "How often they go on dates (1 = several times a week, 7 = almost never)",
    "go_out": "How often they go out at all (1 = several times a week, 7 = almost never)",
    "exp_happy": "Expected happiness with the people they will meet (1-10)",
    "goal_serious": "Main goal: a serious relationship",
    "goal_get_date": "Main goal: to get a date",
    "goal_fun_night": "Main goal: a fun night out",
    "self_attr": "Self-rated attractiveness (1-10)",
    "self_sinc": "Self-rated sincerity (1-10)",
    "self_intel": "Self-rated intelligence (1-10)",
    "self_fun": "Self-rated fun (1-10)",
    "self_amb": "Self-rated ambition (1-10)",
    "wants_looks": "Share of the 100 preference points given to 'attractive'",
    "survey_missing": "Did not fill in the sign-up survey (answers imputed)",
    "interest_corr": "Correlation of the two people's 12 interest ratings",
    "interest_gap": "Mean absolute difference of the 12 interest ratings",
    "age_gap": "His age minus her age",
    "age_gap_abs": "Absolute age difference",
    "same_field": "Same field of study",
    "looks_gap": "His self-rated looks minus hers",
    "pref_fit": "Partner's self-ratings weighted by what this person says matters (1-10)",
    "same_race": "Same race/ethnicity",
    "race_mismatch": "Different race x how much this person says same race matters (0-10)",
    "event_size": "Number of men at the event (partners each woman meets)",
}


def describe(col: str) -> str:
    base = col[:-2] if col.endswith(("_W", "_M")) else col
    who = {"_W": " (woman)", "_M": " (man)"}.get(col[-2:], "")
    return FEATURE_INFO.get(base, col) + who
