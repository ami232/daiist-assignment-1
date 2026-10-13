"""
Shared code for the Zomato cloud-kitchen project.

Why this file exists
--------------------
Two different programs need to turn a restaurant into model inputs:

1. train.ipynb  - to train the three models.
2. app.py       - to score restaurants (and new cloud-kitchen ideas) in the
                  dashboard.

If each program had its own copy of the feature code, the two copies could
drift apart, and the dashboard would silently feed the models inputs that are
built differently from the ones they were trained on. Keeping the feature
code (and the PyTorch model class) in ONE place guarantees both programs do
exactly the same thing.

What is in here
---------------
- Constants that define the problem (the success cut-off, which cities we
  keep as separate categories, ...).
- `clean_data`          : raw CSV -> rated Indian restaurants with a target.
- `fit_feature_params`  : learns the "typical price" numbers etc. from the
                          TRAINING split only.
- `build_features`      : applies those numbers to any set of restaurants.
- `LogisticRegressionModel` : the nn.Module used by the standard PyTorch
                          workflow (the app needs it to load the saved model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Problem definition constants
# ---------------------------------------------------------------------------

# A restaurant counts as a "success" if its Zomato rating is at least this.
# Decided from the business framing: 3.8 is "doing well" for a cloud kitchen.
SUCCESS_RATING = 3.8

# Cities that get their own category. Everything else in India is grouped as
# "Other India" because each of those cities only has ~20 restaurants, far
# too few to learn a separate effect for. New Delhi is the REFERENCE city:
# it gets no column of its own, and every other city's coefficient is read
# as "compared with New Delhi". (If every city had a column, the city columns
# would always add up to 1 and duplicate the intercept.)
REFERENCE_CITY = "New Delhi"
OWN_CATEGORY_CITIES = ["Gurgaon", "Noida", "Faridabad"]
OTHER_CITY_LABEL = "Other India"

# A cuisine gets its own yes/no column only if at least this many TRAINING
# restaurants serve it. Rarer cuisines are pooled into "cuisine_Other", so the
# model doesn't learn a weight from a handful of examples (which would mostly
# be noise).
MIN_RESTAURANTS_PER_CUISINE = 50

# Relative price is clipped to these percentiles of the training data before
# squaring. A few restaurants charge 10x+ their cuisine's typical price; once
# squared, those values would be enormous and would dominate the curve.
REL_PRICE_CLIP_PERCENTILES = (1, 99)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def split_cuisines(cuisines: str) -> list[str]:
    """'North Indian, Chinese' -> ['North Indian', 'Chinese']."""
    return [c.strip() for c in str(cuisines).split(",") if c.strip()]


def normalise_name(name: str) -> str:
    """
    Make restaurant names comparable: 'Domino's Pizza ' and "domino's pizza"
    should count as the same brand. Used both to group chains together in the
    train/validation/test split and to count how many outlets a chain has.
    """
    return " ".join(str(name).lower().split())


def city_group(city: str) -> str:
    """Map a raw city name to one of the city categories we model."""
    if city == REFERENCE_CITY or city in OWN_CATEGORY_CITIES:
        return city
    return OTHER_CITY_LABEL


def locality_key(city: str, locality: str) -> str:
    """
    Localities are identified by city AND locality name, because 40 locality
    names appear in more than one city in this dataset (e.g. a 'Sector 18'
    can exist in two cities).
    """
    return f"{city}|{locality}"


def column_safe(text: str) -> str:
    """'North Indian' -> 'North_Indian', for readable feature names."""
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Turn the raw Kaggle CSV into the modelling table.

    Returns
    -------
    rated : the restaurants we train and evaluate on (rated, Indian, valid
            price), with a 0/1 `success` target column.
    india : ALL Indian restaurants, including unrated ones. Used only to
            describe the market (how many restaurants a locality has, how
            many outlets a chain has) - never the rating, so no target
            information leaks from it.
    log   : a small table recording how many rows each cleaning step removed,
            so the notebook can show it.
    """
    log = [("Raw dataset (all countries)", len(raw))]

    # Step 1: India only (country code 1). The non-Indian rows are small,
    # round-numbered samples (e.g. exactly 20 restaurants per US city) in 11
    # different currencies, so they don't represent their markets.
    india = raw[raw["Country Code"] == 1].copy()
    log.append(("India only (country code 1)", len(india)))

    # Step 2: drop restaurants without a rating. Zomato shows them as
    # 'Not rated' with a rating of 0 - that 0 means "unknown", not "terrible",
    # so we can't say whether they are a success or not.
    rated = india[india["Rating text"] != "Not rated"].copy()
    log.append(("...and has a real rating (not 'Not rated')", len(rated)))

    # Step 3: drop restaurants whose cost for two is 0. That is a missing
    # value (nobody eats for free), and price is central to the hypothesis.
    rated = rated[rated["Average Cost for two"] > 0].copy()
    log.append(("...and has a valid price (cost for two > 0)", len(rated)))

    # The target: 1 = success (rating >= 3.8), 0 = not.
    rated["success"] = (rated["Aggregate rating"] >= SUCCESS_RATING).astype(int)

    # Brand key used for the grouped split and the chain feature.
    rated["brand"] = rated["Restaurant Name"].map(normalise_name)
    india["brand"] = india["Restaurant Name"].map(normalise_name)

    log_table = pd.DataFrame(log, columns=["step", "rows remaining"])
    return rated.reset_index(drop=True), india.reset_index(drop=True), log_table


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def fit_feature_params(train: pd.DataFrame, india: pd.DataFrame) -> dict:
    """
    Learn every number the features depend on.

    The golden rule: anything that summarises PRICES or RATINGS is computed
    from the TRAINING split only. If we used the validation/test restaurants
    here, information about them would leak into the inputs, and our test
    score would be too optimistic.

    Two counts are the deliberate exception and come from `india` (every
    Indian listing, rated or not): how many restaurants a locality has, and
    how many outlets a brand has. They describe the market a company can see
    before launching, and contain no rating information. Counting them from
    the training split alone would be wrong for another reason: with chains
    kept together in one split, a chain in the test set would always look
    like it has 0 outlets.
    """
    params: dict = {}

    # --- Typical price of each cuisine (median cost for two, training only).
    # A restaurant listing 'North Indian, Chinese' contributes its price to
    # BOTH cuisines. The median (not the mean) is used so a few very
    # expensive restaurants don't drag a cuisine's "typical" price up.
    exploded = train.assign(cuisine=train["Cuisines"].map(split_cuisines)).explode("cuisine")
    cuisine_median = exploded.groupby("cuisine")["Average Cost for two"].median()
    cuisine_counts = exploded["cuisine"].value_counts()
    params["cuisine_median_cost"] = cuisine_median.to_dict()
    params["overall_median_cost"] = float(train["Average Cost for two"].median())

    # --- Which cuisines get their own column (enough training examples).
    common = cuisine_counts[cuisine_counts >= MIN_RESTAURANTS_PER_CUISINE].index
    params["common_cuisines"] = sorted(common)

    # --- Clip range for relative price, from the training distribution.
    rel_train = _relative_price(train, params)
    lo, hi = np.percentile(rel_train, REL_PRICE_CLIP_PERCENTILES)
    params["rel_price_clip"] = [float(lo), float(hi)]

    # --- Typical cost per locality (training only), with fall-backs for a
    # locality that has no training restaurants: first the city's typical
    # cost, then the overall typical cost.
    train_loc = train.assign(loc=[locality_key(c, l) for c, l in zip(train["City"], train["Locality"])])
    params["locality_median_cost"] = train_loc.groupby("loc")["Average Cost for two"].median().to_dict()
    params["city_median_cost"] = train.groupby("City")["Average Cost for two"].median().to_dict()

    # --- Market counts (all Indian listings, see docstring).
    india_loc = [locality_key(c, l) for c, l in zip(india["City"], india["Locality"])]
    params["locality_restaurant_count"] = pd.Series(india_loc).value_counts().to_dict()
    params["brand_outlet_count"] = india["brand"].value_counts().to_dict()

    # --- The final column order. Saved so the app builds columns in exactly
    # the same order the models were trained on.
    params["feature_names"] = list(build_features(train.head(1), params, _check_columns=False).columns)
    return params


def _relative_price(df: pd.DataFrame, params: dict) -> np.ndarray:
    """
    Relative price = what this restaurant charges / what is typical for the
    cuisines it serves.

    'Typical' for a multi-cuisine restaurant = the AVERAGE of the typical
    prices of each cuisine it lists. E.g. North Indian (Rs 500) + Continental
    (Rs 1,200) -> typical = Rs 850; a restaurant charging Rs 1,000 has a
    relative price of 1.18 (18% above its norm). A cuisine never seen in
    training falls back to the overall typical price.
    """
    medians = params["cuisine_median_cost"]
    overall = params["overall_median_cost"]
    norms = [
        np.mean([medians.get(c, overall) for c in split_cuisines(cs)] or [overall])
        for cs in df["Cuisines"]
    ]
    return df["Average Cost for two"].to_numpy(dtype=float) / np.asarray(norms)


def build_features(df: pd.DataFrame, params: dict, _check_columns: bool = True) -> pd.DataFrame:
    """
    Turn restaurants into the model's input table (one row per restaurant,
    one column per feature). Uses only the numbers stored in `params`, so it
    behaves identically for training, validation, test and the dashboard.

    Columns of `df` it needs: Restaurant Name, City, Locality, Cuisines,
    Average Cost for two, Has Online delivery, Has Table booking.
    """
    out = pd.DataFrame(index=df.index)
    cost = df["Average Cost for two"].to_numpy(dtype=float)
    has_delivery = (df["Has Online delivery"] == "Yes").astype(float).to_numpy()

    # ---------------- Price features (the hypothesis) ----------------
    # log(cost): prices are very skewed (Rs 50 to Rs 8,000). On a log scale,
    # going from Rs 300 -> Rs 600 counts the same as Rs 1,000 -> Rs 2,000
    # (both are 'double the price'), which is how customers perceive price.
    out["log_cost"] = np.log(cost)

    # Relative price and its square. Together they let the model draw a
    # curve that can go UP and then DOWN (a hill), i.e. a sweet spot.
    # With only the straight term the model could only say 'pricier is
    # always better' or 'always worse'.
    lo, hi = params["rel_price_clip"]
    rel = np.clip(_relative_price(df, params), lo, hi)
    out["rel_price"] = rel
    out["rel_price_sq"] = rel ** 2

    # Interaction: log(cost) x delivery. Lets price matter DIFFERENTLY for
    # restaurants that deliver (the cloud-kitchen setting) vs. dine-in only.
    # Without it, the model would force the same price effect on both.
    out["log_cost_x_delivery"] = out["log_cost"] * has_delivery

    # ---------------- Service features ----------------
    out["has_delivery"] = has_delivery
    # Table booking is mostly a dine-in / upmarket signal. A cloud kitchen
    # won't have it, but including it lets the model attribute the 'nice
    # dining room' boost to this column instead of wrongly to price.
    out["has_table_booking"] = (df["Has Table booking"] == "Yes").astype(float).to_numpy()

    # ---------------- Brand feature ----------------
    # 1 if the brand has more than one outlet in India. A brand-new cloud
    # kitchen brand would be 0.
    outlets = df["Restaurant Name"].map(normalise_name).map(params["brand_outlet_count"]).fillna(1)
    out["is_chain"] = (outlets > 1).astype(float).to_numpy()

    # ---------------- Cuisine features ----------------
    cuisine_lists = df["Cuisines"].map(split_cuisines)
    # How many cuisines the menu covers: a focused menu vs. 'we do everything'.
    out["n_cuisines"] = cuisine_lists.map(len).astype(float).to_numpy()
    common = params["common_cuisines"]
    for c in common:
        out[f"cuisine_{column_safe(c)}"] = cuisine_lists.map(lambda cs, c=c: float(c in cs)).to_numpy()
    # 1 if the restaurant serves at least one rare (pooled) cuisine.
    common_set = set(common)
    out["cuisine_Other"] = cuisine_lists.map(lambda cs: float(any(c not in common_set for c in cs))).to_numpy()

    # ---------------- Location features ----------------
    keys = [locality_key(c, l) for c, l in zip(df["City"], df["Locality"])]
    # Competition: how many restaurants are listed in the same locality.
    # log(1 + count) because 5 -> 50 restaurants is a much bigger change in
    # competition than 500 -> 545.
    counts = np.array([params["locality_restaurant_count"].get(k, 0) for k in keys], dtype=float)
    out["log_locality_restaurants"] = np.log1p(counts)
    # Is the area upmarket or budget? Typical cost for two in the locality
    # (training data), falling back to the city, then the overall typical.
    loc_cost = [
        params["locality_median_cost"].get(k, params["city_median_cost"].get(city, params["overall_median_cost"]))
        for k, city in zip(keys, df["City"])
    ]
    out["log_locality_median_cost"] = np.log(np.asarray(loc_cost, dtype=float))
    # City categories (New Delhi = reference, no column).
    groups = df["City"].map(city_group)
    for city in OWN_CATEGORY_CITIES + [OTHER_CITY_LABEL]:
        out[f"city_{column_safe(city)}"] = (groups == city).astype(float).to_numpy()

    if _check_columns:
        # Safety net: the column order must match what the models were
        # trained on, otherwise weights would be applied to the wrong inputs.
        assert list(out.columns) == params["feature_names"], "Feature columns changed!"
    return out


# ---------------------------------------------------------------------------
# PyTorch model for the 'standard workflow' (torch.nn + torch.optim)
# ---------------------------------------------------------------------------

class LogisticRegressionModel(nn.Module):
    """
    Logistic regression as an nn.Module: one linear layer (weights + bias).

    forward() returns LOGITS (the raw score z = X @ w + b, before the
    sigmoid), not probabilities. The loss used with it,
    nn.BCEWithLogitsLoss, applies the sigmoid itself in a numerically stable
    way (it never computes log(0)). To get a probability, call
    `predict_proba`, which applies the sigmoid explicitly.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # (n_samples, n_features) -> (n_samples,): squeeze the last dim so the
        # output lines up with a 1-D target vector.
        return self.linear(X).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(X))
