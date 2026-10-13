"""Loading the raw data and turning it into one row per woman-man pair.

Every date is in the raw file twice, once from each side. build_pair_table
joins the woman's record of a date with the man's record of the same date.
Missing answers stay NaN here: they are filled later, inside the model
pipeline, with medians learned on the training events only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "Speed Dating Data.csv"

INTERESTS_ALL = [
    "sports", "exercise", "museums", "art", "hiking", "gaming", "reading",
    "tv", "theater", "movies", "concerts", "music", "shopping",
]
# `art` is dropped: r = 0.87 with `museums` (per person), the only interest pair
# above 0.8. train.py re-runs this check and stops if the result changes.
INTERESTS = [i for i in INTERESTS_ALL if i != "art"]
ATTRS6 = ["attr", "sinc", "intel", "fun", "amb", "shar"]   # attractive, sincere, intelligent, fun, ambitious, shared interests
ATTRS5 = ATTRS6[:5]  # self-ratings (attr3_1...) have no "shared interests" item

# Sign-up survey blocks, used for the missing-value check in train.py.
# 4_1 and 5_1 were not asked in the early waves, so they are not used.
SIGNUP_BLOCKS = {
    "profile (age, field, goal, dating, going out, happiness, race/religion importance)":
        ["age", "field_cd", "goal", "date", "go_out", "exphappy", "imprace", "imprelig"],
    "interests (13 x 1-10)": INTERESTS_ALL,
    "1_1 what I look for (100 points)": [f"{a}1_1" for a in ATTRS6],
    "2_1 what I think the other side looks for": [f"{a}2_1" for a in ATTRS6],
    "3_1 how I rate myself (1-10)": [f"{a}3_1" for a in ATTRS5],
    "4_1 what I think my own sex looks for": [f"{a}4_1" for a in ATTRS6],
    "5_1 how I think others rate me": [f"{a}5_1" for a in ATTRS5],
}

# Sign-up columns kept for each person in the pair table.
SIGNUP_COLS = (
    ["age", "field_cd", "goal", "date", "go_out", "exphappy", "imprace"] + INTERESTS
    + [f"{a}1_1" for a in ATTRS6] + [f"{a}2_1" for a in ATTRS6] + [f"{a}3_1" for a in ATTRS5]
)
# Scorecard columns filled in DURING the date. Never model inputs; only used in
# the "what if we could see attraction" analysis.
SCORECARD_COLS = ["attr", "sinc", "intel", "fun", "amb", "shar", "like"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="unicode_escape")


def build_pair_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per woman-man pair: keys, both decisions, the target, both sign-up
    surveys (NaN where unanswered) and, set aside with prefix card_, both scorecards."""
    rec = raw.dropna(subset=["pid"]).astype({"pid": "int64"})
    keep = ["iid", "pid", "wave", "round", "samerace", "dec", "match"] + SIGNUP_COLS + SCORECARD_COLS
    women = rec.loc[rec["gender"] == 0, keep]
    men = rec.loc[rec["gender"] == 1, keep]
    p = women.merge(men, left_on=["iid", "pid"], right_on=["pid", "iid"], suffixes=("_W", "_M"),
                    validate="one_to_one")

    for col in ["wave", "samerace", "match"]:
        if not (p[f"{col}_W"] == p[f"{col}_M"]).all():
            raise ValueError(f"{col} differs between the two records of a date")
    if not (p["match_W"] == p["dec_W"] * p["dec_M"]).all():
        raise ValueError("match is not her yes x his yes")
    if not len(p) == len(women) == len(men):
        raise ValueError("some records have no mirror record")

    out = pd.DataFrame({
        "iid_W": p["iid_W"], "iid_M": p["iid_M"], "wave": p["wave_W"],
        # partners each woman meets = men at the event ("round" differs by side
        # when the two sides are not the same size)
        "event_size": p["round_W"],
        "samerace": p["samerace_W"],
        "dec_W": p["dec_W"], "dec_M": p["dec_M"], "match": p["match_W"],
    })
    for s in ("W", "M"):
        for col in SIGNUP_COLS:
            out[f"{col}_{s}"] = p[f"{col}_{s}"].astype(float)
        for col in SCORECARD_COLS:
            out[f"card_{col}_{s}"] = p[f"{col}_{s}"].astype(float)   # card_attr_W = how attractive SHE found HIM
        out[f"survey_missing_{s}"] = p[f"attr1_1_{s}"].isna().astype(int)
    return out.reset_index(drop=True)


def looks_reputation(raw: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out mean of the attractiveness ratings a person received from
    everyone ELSE they met that night (excluding the current partner).

    Stand-in for a signal a business could have before a meeting (crowd-rated
    photos, ratings from a previous event). It is built from same-night
    scorecards, so it is NOT available before the event in this dataset. Only
    used in the 'what if we could see attraction' analysis."""
    r = raw.dropna(subset=["pid"]).astype({"pid": "int64"})[["iid", "pid", "attr_o"]]
    totals = r.groupby("iid")["attr_o"].agg(["sum", "count"])
    received = r.set_index(["iid", "pid"])["attr_o"]

    def loo(person: pd.Series, partner: pd.Series) -> np.ndarray:
        own = received.reindex(pd.MultiIndex.from_arrays([person, partner])).to_numpy()
        has_own = ~np.isnan(own)
        num = totals["sum"].reindex(person).to_numpy() - np.where(has_own, own, 0.0)
        den = totals["count"].reindex(person).to_numpy().astype(float) - has_own
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    return pd.DataFrame({"looks_rep_W": loo(pairs["iid_W"], pairs["iid_M"]),   # how men rated her
                         "looks_rep_M": loo(pairs["iid_M"], pairs["iid_W"])},  # how women rated him
                        index=pairs.index)
