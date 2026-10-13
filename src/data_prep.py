"""
Data preparation pipeline for the "should an investor buy this house" classifier.

This module is shared by train.py and app.py so that both use the exact
same label definition, split, and feature engineering -- the app must
never redo modeling decisions differently from training.

Pipeline stages (in order):
    1. load_raw            -- read the CSV
    2. build_label         -- construct the binary "buy" target from SalePrice
    3. time_split           -- split by Yr Sold to avoid comp-group leakage
    4. engineer_features    -- add house_age, total_sqft, etc.
    5. preprocess           -- missing values, one-hot encoding, scaling
                               (fit on train only, applied to both train/test)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Columns excluded from the model's feature set, and why.
# ---------------------------------------------------------------------------
# These built the label itself -- using them as features would let the
# model "cheat" by seeing (a transformation of) its own target.
LABEL_LEAKAGE_COLS = ["SalePrice", "price_per_sqft", "expected_psf", "group_size"]

# Yr Sold / Mo Sold were used to define comp groups (entangled with the
# label's construction), and train/test see disjoint year ranges -- keeping
# them would let the model extrapolate on "year" rather than learn real
# house characteristics. Order/PID are just row identifiers.
STRUCTURAL_EXCLUDE_COLS = ["Yr Sold", "Mo Sold", "Order", "PID"]

# Categorical columns where NaN means "this house doesn't have that
# feature" (e.g. no basement), not "data is missing".
CATEGORICAL_NONE_COLS = [
    "Alley", "Mas Vnr Type", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure",
    "BsmtFin Type 1", "BsmtFin Type 2", "Fireplace Qu", "Garage Type",
    "Garage Finish", "Garage Qual", "Garage Cond", "Pool QC", "Fence",
    "Misc Feature",
]

# Numeric columns where NaN means "0 of this thing" for the same reason.
NUMERIC_ZERO_COLS = [
    "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
    "Total Bsmt SF", "Bsmt Full Bath", "Bsmt Half Bath",
    "Garage Cars", "Garage Area",
]

# Dropped outright: mostly redundant with Garage Type ("None" already
# signals no garage) and filling a *year* column with 0 would be a
# misleading value for a model to learn from.
DROP_COLS = ["Garage Yr Blt"]

QUAL_TIER_BINS = [0, 3, 6, 10]
QUAL_TIER_LABELS = ["low", "mid", "high"]
COMP_GROUP_COLS = ["Neighborhood", "qual_tier", "Yr Sold"]

BUY_THRESHOLD = 0.10  # a house is "buy" if >10% below its comp group's median $/sqft
TRAIN_YEARS = [2006, 2007, 2008]
TEST_YEARS = [2009, 2010]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def build_label(df: pd.DataFrame, threshold: float = BUY_THRESHOLD) -> tuple[pd.DataFrame, dict]:
    """Construct the binary 'buy' target from SalePrice via comp-group medians.

    Returns the dataframe with label + intermediate columns added, plus a
    small report dict describing comp-group sparsity (for REPORT.md).
    """
    df = df.copy()
    df["price_per_sqft"] = df["SalePrice"] / df["Gr Liv Area"]
    df["qual_tier"] = pd.cut(
        df["Overall Qual"], bins=QUAL_TIER_BINS, labels=QUAL_TIER_LABELS
    )

    df["expected_psf"] = df.groupby(COMP_GROUP_COLS)["price_per_sqft"].transform("median")
    df["group_size"] = df.groupby(COMP_GROUP_COLS)["price_per_sqft"].transform("size")
    df["buy"] = (df["price_per_sqft"] < df["expected_psf"] * (1 - threshold)).astype(int)

    n_groups = df.groupby(COMP_GROUP_COLS).ngroups
    sparse_groups = int((df.groupby(COMP_GROUP_COLS).size() < 5).sum())
    report = {
        "n_comp_groups": n_groups,
        "sparse_groups_lt5": sparse_groups,
        "buy_rate": float(df["buy"].mean()),
        "n_buy": int(df["buy"].sum()),
        "n_total": len(df),
    }
    return df, report


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["Yr Sold"].isin(TRAIN_YEARS)].copy()
    test_df = df[df["Yr Sold"].isin(TEST_YEARS)].copy()
    return train_df, test_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns. Must run BEFORE Yr Sold is dropped, since
    house_age / years_since_remodel are computed from it -- the derived
    age numbers don't leak the raw sale year the same way a raw Yr Sold
    column would (two houses sold in different years but the same age
    look identical to the model)."""
    df = df.copy()
    df["house_age"] = df["Yr Sold"] - df["Year Built"]
    df["years_since_remodel"] = df["Yr Sold"] - df["Year Remod/Add"]
    df["total_sqft"] = df["Total Bsmt SF"].fillna(0) + df["Gr Liv Area"]
    df["total_bathrooms"] = (
        df["Full Bath"].fillna(0)
        + 0.5 * df["Half Bath"].fillna(0)
        + df["Bsmt Full Bath"].fillna(0)
        + 0.5 * df["Bsmt Half Bath"].fillna(0)
    )
    df["qual_x_cond"] = df["Overall Qual"] * df["Overall Cond"]
    return df


def _drop_non_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = (
        LABEL_LEAKAGE_COLS + STRUCTURAL_EXCLUDE_COLS + DROP_COLS + ["buy", "qual_tier"]
    )
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


class Preprocessor:
    """Fits missing-value fills, one-hot categories, and scaling on the
    training set only, then applies the identical transform to any split."""

    def __init__(self):
        self.categorical_modes_: dict = {}
        self.numeric_medians_: dict = {}
        self.dummy_columns_: list[str] | None = None
        self.scale_mean_: pd.Series | None = None
        self.scale_std_: pd.Series | None = None
        self.feature_names_: list[str] | None = None

    def _fill_missing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_NONE_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("None")
        for col in NUMERIC_ZERO_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Remaining columns: genuinely missing values, not structural.
        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                if fit:
                    mode_val = df[col].mode(dropna=True)
                    self.categorical_modes_[col] = mode_val.iloc[0] if len(mode_val) else "None"
                df[col] = df[col].fillna(self.categorical_modes_.get(col, "None"))
            else:
                if fit:
                    self.numeric_medians_[col] = df[col].median()
                df[col] = df[col].fillna(self.numeric_medians_.get(col, 0))
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _drop_non_feature_cols(df)
        X = self._fill_missing(X, fit=True)
        X = pd.get_dummies(X, drop_first=True)
        self.dummy_columns_ = X.columns.tolist()

        self.scale_mean_ = X.mean()
        self.scale_std_ = X.std().replace(0, 1)
        X = (X - self.scale_mean_) / self.scale_std_

        self.feature_names_ = X.columns.tolist()
        return X

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _drop_non_feature_cols(df)
        X = self._fill_missing(X, fit=False)
        X = pd.get_dummies(X, drop_first=True)
        # Align to training-time columns: add missing (=0), drop unseen.
        X = X.reindex(columns=self.dummy_columns_, fill_value=0)
        X = (X - self.scale_mean_) / self.scale_std_
        return X


def prepare_dataset(csv_path: str):
    """Runs the full pipeline and returns everything train.py / app.py need."""
    raw = load_raw(csv_path)
    labeled, label_report = build_label(raw)
    labeled = engineer_features(labeled)
    train_df, test_df = time_split(labeled)

    pre = Preprocessor()
    X_train = pre.fit_transform(train_df)
    X_test = pre.transform(test_df)
    y_train = train_df["buy"].to_numpy()
    y_test = test_df["buy"].to_numpy()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_df": train_df,   # kept for the dashboard's distribution plots
        "test_df": test_df,
        "preprocessor": pre,
        "label_report": label_report,
    }