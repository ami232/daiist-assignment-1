"""
Data loading, population filter, feature engineering, temporal split and
preprocessing for the Ames Housing appraisal model.

Kept separate from train.py so the Gradio app can import the exact same
transformations without retraining anything.

Order of operations (and why):
  1. load_data            raw CSV, keeping "NA" (= feature absent) apart from "" (= missing)
  2. filter_population    only the sales an appraiser would value (see REPORT.md)
  3. engineer_features    ROW-WISE transforms only: each new value depends on its
                          own row, never on other rows, so it is safe to do before
                          the split (no information can leak between years)
  4. temporal_split       train 2006-08 / val 2009 / test 2010
  5. fit_lot_frontage_medians + impute_lot_frontage, and build_preprocessor:
                          everything that LEARNS a statistic (medians, means,
                          standard deviations, category lists) is fitted on the
                          training years only, then applied unchanged to val/test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/ames.csv")
TARGET = "SalePrice"

# Residential zoning codes. C (commercial), I (industrial) and A (agricultural)
# are excluded: only 16 sales and not typical homes for a residential appraiser.
RESIDENTIAL_ZONES = ["RL", "RM", "RH", "FV"]

TRAIN_YEARS = [2006, 2007, 2008]
VAL_YEAR = 2009
TEST_YEAR = 2010

# Quality ratings are ordered text (Excellent > Good > Typical > Fair > Poor).
# Mapping them to numbers keeps that order in ONE column, instead of 5 unordered
# one-hot columns. "NA" means the house does not have the feature (no basement,
# no fireplace...), which is "worse than Poor" for value, so it gets 0.
QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
QUALITY_COLS = [
    "Exter Qual", "Kitchen Qual", "Bsmt Qual",
    "Heating QC", "Fireplace Qu", "Garage Qual",
]

PORCH_COLS = ["Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch"]

# Final model inputs. Chosen explicitly (rather than "all remaining columns") so
# every feature has a reason to be there. Columns not listed are left out on
# purpose: identifiers (Order, PID), constant after the filter (Sale Condition),
# sale date (Yr Sold, Mo Sold: a linear model would have to extrapolate a trend
# into 2010, a year it never saw), near-constant (Utilities, Street, Pool QC,
# Misc Feature, Condition 2, Roof Matl), redundant with a kept column
# (Garage Area ~ Garage Cars, 1st/2nd Flr SF ~ Gr Liv Area, Garage Yr Blt ~
# Year Built, TotRms AbvGrd ~ Gr Liv Area), or raw inputs replaced by an
# engineered version (Year Built -> HouseAge, porch columns -> TotalPorchSF...).
NUMERIC_FEATURES = [
    # overall ratings (already 1-10 integers)
    "Overall Qual", "Overall Cond",
    # size, on log scale (see engineer_features)
    "log_TotalSF", "log_GrLivArea", "log_LotArea", "log_TotalPorchSF", "log_MasVnrArea",
    "BsmtFinType1SF",
    "Lot Frontage", "frontage_missing",
    # rooms / amenities
    "TotalBaths", "Bedroom AbvGr", "Fireplaces", "Garage Cars",
    # age relative to the sale date
    "HouseAge", "YearsSinceRemod",
    # quality ratings converted to 0-5
    *[f"{c}_ord" for c in QUALITY_COLS],
    # presence flags
    "has_garage", "has_basement", "has_fireplace", "has_2nd_floor",
    # interaction
    "Qual_x_logSF",
]
CATEGORICAL_FEATURES = ["Neighborhood", "Bldg Type", "MS Zoning", "Central Air", "Foundation"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV, keeping Ames' two kinds of 'null' apart.

    In the original file (De Cock, 2011):
      - the text "NA" means "the house does NOT have this" (no basement, no
        garage...). That is real information, so we keep it as text.
      - an empty cell "" means the value is genuinely missing -> NaN.
    By default pandas turns both into NaN and that difference would be lost,
    so we switch off keep_default_na and only treat "" as missing.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def filter_population(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the sales the appraisal model is meant to price.

    - Sale Condition == "Normal": ordinary arm's-length sales. Foreclosures,
      family sales, partial sales of unfinished new homes, etc. follow
      different pricing logic (REPORT.md, business framing).
    - Residential zoning only.
    """
    mask = (df["Sale Condition"] == "Normal") & (df["MS Zoning"].isin(RESIDENTIAL_ZONES))
    return df.loc[mask].reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered columns. Row-wise only: no statistic is learned here."""
    df = df.copy()

    # --- A1: ordinal quality ratings -----------------------------------------
    # A genuinely missing rating ("" -> NaN) stays NaN and is imputed later with
    # the training median; "NA" (feature absent) becomes 0.
    for col in QUALITY_COLS:
        df[f"{col}_ord"] = df[col].map(QUALITY_MAP)

    # --- A2: presence flags --------------------------------------------------
    # Separate "doesn't have it" from "has a small one". Without the flag, a
    # linear model treats 0 sq ft of basement as just a very small basement.
    df["has_garage"] = (df["Garage Type"] != "NA").astype(int)
    df["has_basement"] = (df["Bsmt Qual"] != "NA").astype(int)
    df["has_fireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["has_2nd_floor"] = (df["2nd Flr SF"] > 0).astype(int)

    # --- B1: total usable area -----------------------------------------------
    # Buyers pay for total space; above-ground and basement area separately are
    # strongly correlated with each other. fillna(0): the single house with a
    # missing basement area has no basement information at all.
    total_sf = df["Gr Liv Area"] + df["Total Bsmt SF"].fillna(0)

    # --- B1b: finished basement area (added after the feature analysis) ------
    # TotalSF counts a finished basement and a bare concrete one the same, but
    # buyers do not pay the same for them. Adding the finished (type 1) area
    # lets the model price that difference. Chosen on the validation year
    # among several variants (REPORT.md); fillna(0) as for TotalSF above.
    df["BsmtFinType1SF"] = df["BsmtFin SF 1"].fillna(0)

    # --- B2: total bathrooms (a half bath counts 0.5) -------------------------
    df["TotalBaths"] = (
        df["Full Bath"] + 0.5 * df["Half Bath"]
        + df["Bsmt Full Bath"].fillna(0) + 0.5 * df["Bsmt Half Bath"].fillna(0)
    )

    # --- B3: age measured at the moment of sale -------------------------------
    # A house built in 1990 is 16 years old in a 2006 sale and 20 in 2010:
    # age at sale is what affects value, not the construction year itself.
    # clip(lower=0): a few houses are sold the year before their "built" year.
    df["HouseAge"] = (df["Yr Sold"] - df["Year Built"]).clip(lower=0)
    df["YearsSinceRemod"] = (df["Yr Sold"] - df["Year Remod/Add"]).clip(lower=0)

    # --- B4: total outdoor/porch area -----------------------------------------
    porch_sf = df[PORCH_COLS].sum(axis=1)

    # --- B5: log of skewed sizes ----------------------------------------------
    # With log(price) as target, log(size) makes the coefficient an elasticity:
    # +1% size -> +b% price. It also pulls in the long right tails (Lot Area has
    # skew ~11) so a few huge lots don't dominate the fit.
    # log1p (= log(1 + x)) is used where 0 is a valid value (no porch, no veneer).
    df["log_TotalSF"] = np.log(total_sf)
    df["log_GrLivArea"] = np.log(df["Gr Liv Area"])
    df["log_LotArea"] = np.log(df["Lot Area"])
    df["log_TotalPorchSF"] = np.log1p(porch_sf)
    df["log_MasVnrArea"] = np.log1p(df["Mas Vnr Area"])  # NaN stays NaN -> imputed later

    # --- B6: interaction quality x size --------------------------------------
    # The premium for quality grows with size: an extra quality point is worth
    # more on a big house. A linear model can only add effects, so it cannot
    # learn this unless we give it the product as a column.
    df["Qual_x_logSF"] = df["Overall Qual"] * df["log_TotalSF"]

    # --- C1 (row-wise part): flag where Lot Frontage is missing --------------
    # Missingness is not random (51% missing on cul-de-sacs vs 14% on inside
    # lots), so the fact that it is missing may carry information.
    df["frontage_missing"] = df["Lot Frontage"].isna().astype(int)

    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by sale year: past -> train, 2009 -> validation, 2010 -> test.

    An appraiser only ever has past sales to learn from; a random split would
    put 2010 sales in training while "predicting" 2009 ones.
    """
    train = df[df["Yr Sold"].isin(TRAIN_YEARS)].reset_index(drop=True)
    val = df[df["Yr Sold"] == VAL_YEAR].reset_index(drop=True)
    test = df[df["Yr Sold"] == TEST_YEAR].reset_index(drop=True)
    return train, val, test


def fit_lot_frontage_medians(train: pd.DataFrame) -> dict:
    """C1: learn the median Lot Frontage per neighborhood, on TRAIN only.

    Lot frontage depends on the street layout, which is similar within a
    neighborhood, so the neighborhood median is a better guess than the
    city-wide median. "_global_" is the fallback for a neighborhood with no
    known frontage in the training years.
    """
    medians = train.groupby("Neighborhood")["Lot Frontage"].median().dropna().to_dict()
    medians["_global_"] = float(train["Lot Frontage"].median())
    return medians


def impute_lot_frontage(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    df = df.copy()
    fill = df["Neighborhood"].map(medians).fillna(medians["_global_"])
    df["Lot Frontage"] = df["Lot Frontage"].fillna(fill)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Unfitted sklearn preprocessor. Call .fit() on the TRAINING rows only.

    Numeric columns:
      - SimpleImputer(median): fills the few remaining genuine gaps (1-23 rows)
        with the training median; the median is robust to skewed columns.
      - StandardScaler: (x - mean) / std. Needed by the two PyTorch versions:
        with features on wildly different scales (log sq ft ~ 7, Qual x logSF
        ~ 50, flags 0/1), one learning rate cannot suit all weights and
        gradient descent converges badly. It also makes the L2 penalty treat
        every coefficient equally.
    Categorical columns:
      - SimpleImputer(most_frequent): only "Electrical"-style 1-row gaps.
      - OneHotEncoder(min_frequency=10): one 0/1 column per category; categories
        with fewer than 10 training sales (e.g. Landmrk, GrnHill, Blueste) are
        merged into one "infrequent" column, because a coefficient estimated
        from 1-4 houses is mostly noise. handle_unknown="infrequent_if_exist"
        sends a category never seen in training to that same column instead of
        crashing.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            min_frequency=10,
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
        )),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
