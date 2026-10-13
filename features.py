"""Preprocessing, feature engineering and chronological split for Ames Housing."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "AmesHousing.txt"

# ---------------------------------------------------------------- 1. load & scope

ID_COLS = ["Order", "PID"]


def load_and_scope(path=RAW_PATH):
    df = pd.read_csv(path, sep="\t")
    df = df.drop(columns=ID_COLS)
    n_all = len(df)

    # Business scope (step 2): only arm's-length market sales.
    df = df[df["Sale Condition"] == "Normal"].copy()
    n_normal = len(df)

    # De Cock's documented outliers: oversized houses that distort least squares.
    df = df[df["Gr Liv Area"] <= 4000].copy()

    # Sale Condition is now constant -> carries no information.
    df = df.drop(columns=["Sale Condition"])

    print(f"rows: {n_all} -> {n_normal} (Normal) -> {len(df)} (<=4000 sqft)")
    return df.reset_index(drop=True)


# ------------------------------------------------- 2. structural vs true missing

# NA here means "the feature does not exist", not "value unknown".
STRUCTURAL_CAT = [
    "Alley", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1",
    "BsmtFin Type 2", "Fireplace Qu", "Garage Type", "Garage Finish",
    "Garage Qual", "Garage Cond", "Pool QC", "Fence", "Misc Feature",
    "Mas Vnr Type",
]
STRUCTURAL_NUM = [
    "Bsmt Full Bath", "Bsmt Half Bath", "BsmtFin SF 1", "BsmtFin SF 2",
    "Bsmt Unf SF", "Total Bsmt SF", "Garage Cars", "Garage Area", "Mas Vnr Area",
]


def fill_structural(df):
    df = df.copy()
    for c in STRUCTURAL_CAT:
        df[c] = df[c].fillna("None")
    for c in STRUCTURAL_NUM:
        df[c] = df[c].fillna(0)
    # Garage Yr Blt is structural too, but a year of 0 is nonsense.
    # Encode "no garage" as a flag and set the year to the house's year.
    df["Has Garage"] = df["Garage Yr Blt"].notna().astype(int)
    df["Garage Yr Blt"] = df["Garage Yr Blt"].fillna(df["Year Built"])
    return df


# ------------------------------------------------------------- 3. ordinal scales

QUAL_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_COLS = {
    "Exter Qual": QUAL_MAP, "Exter Cond": QUAL_MAP, "Bsmt Qual": QUAL_MAP,
    "Bsmt Cond": QUAL_MAP, "Heating QC": QUAL_MAP, "Kitchen Qual": QUAL_MAP,
    "Fireplace Qu": QUAL_MAP, "Garage Qual": QUAL_MAP, "Garage Cond": QUAL_MAP,
    "Pool QC": QUAL_MAP,
    "Bsmt Exposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFin Type 1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3,
                       "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFin Type 2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3,
                       "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "Garage Finish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3,
                   "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "Lot Shape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "Paved Drive": {"N": 0, "P": 1, "Y": 2},
    "Central Air": {"N": 0, "Y": 1},
}


def encode_ordinals(df):
    df = df.copy()
    for col, mapping in ORDINAL_COLS.items():
        df[col] = df[col].map(mapping)
        if df[col].isna().any():
            raise ValueError(f"unmapped level in {col}")
        df[col] = df[col].astype(int)
    return df


# ---------------------------------------------------------- 4. derived features

def add_derived(df):
    df = df.copy()

    # Total living space is what buyers actually price.
    df["Total SF"] = df["Gr Liv Area"] + df["Total Bsmt SF"]
    df["Total Porch SF"] = (df["Open Porch SF"] + df["Enclosed Porch"]
                            + df["3Ssn Porch"] + df["Screen Porch"]
                            + df["Wood Deck SF"])
    # Half-baths are worth roughly half a full bath.
    df["Total Baths"] = (df["Full Bath"] + 0.5 * df["Half Bath"]
                         + df["Bsmt Full Bath"] + 0.5 * df["Bsmt Half Bath"])

    # Age at the moment of sale, not calendar year.
    df["Age"] = df["Yr Sold"] - df["Year Built"]
    df["Years Since Remodel"] = df["Yr Sold"] - df["Year Remod/Add"]
    df["Garage Age"] = df["Yr Sold"] - df["Garage Yr Blt"]
    df["Is Remodeled"] = (df["Year Remod/Add"] != df["Year Built"]).astype(int)
    df["Is New"] = (df["Age"] <= 1).astype(int)

    # A few homes record a remodel/garage year after the sale year.
    for c in ["Age", "Years Since Remodel", "Garage Age"]:
        df[c] = df[c].clip(lower=0)

    # Quality interacts with size: 1500 sqft of excellent build is not
    # worth the same as 1500 sqft of poor build.
    df["Qual x SF"] = df["Overall Qual"] * df["Total SF"]

    # Month of sale is cyclical: December and January are adjacent.
    df["Mo sin"] = np.sin(2 * np.pi * df["Mo Sold"] / 12)
    df["Mo cos"] = np.cos(2 * np.pi * df["Mo Sold"] / 12)

    return df.drop(columns=["Mo Sold"])


# ------------------------------------------------- 5. chronological split (step 2)

TARGET = "SalePrice"


def chrono_split(df, train_max=2008, val_year=2009, test_year=2010):
    train = df[df["Yr Sold"] <= train_max].copy()
    val = df[df["Yr Sold"] == val_year].copy()
    test = df[df["Yr Sold"] == test_year].copy()
    print(f"split -> train {len(train)} | val {len(val)} | test {len(test)}")
    return train, val, test


def split_xy(frame):
    y = np.log(frame[TARGET].to_numpy(dtype=np.float64))
    X = frame.drop(columns=[TARGET, "Yr Sold"])
    return X, y


# ------------------------------- 6. fit-on-train transforms (leakage-sensitive)

RARE_THRESHOLD = 0.01


class Preprocessor:
    """Learns every cross-row statistic from the training set only."""

    def fit(self, X):
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = [c for c in X.columns if c not in self.num_cols_]

        # Lot Frontage is genuinely unknown -> impute by neighbourhood median.
        self.frontage_by_nbhd_ = X.groupby("Neighborhood")["Lot Frontage"].median()
        self.frontage_global_ = X["Lot Frontage"].median()

        # Any other numeric gaps -> train median.
        self.num_medians_ = X[self.num_cols_].median()

        # Collapse rare categorical levels; remember the levels we keep.
        self.levels_ = {}
        for c in self.cat_cols_:
            freq = X[c].value_counts(normalize=True)
            self.levels_[c] = sorted(freq[freq >= RARE_THRESHOLD].index.tolist())

        design = self._design(X)
        self.feature_names_ = design.columns.tolist()

        # Drop columns with no variance in train (a constant can't explain anything).
        self.keep_ = design.std(axis=0) > 0
        self.feature_names_ = design.columns[self.keep_].tolist()
        kept = design.loc[:, self.keep_]

        # Standardise: mean 0, sd 1, using train statistics only.
        self.mean_ = kept.mean(axis=0).to_numpy()
        self.scale_ = kept.std(axis=0).replace(0, 1).to_numpy()
        return self

    def _design(self, X):
        X = X.copy()
        miss = X["Lot Frontage"].isna()
        X.loc[miss, "Lot Frontage"] = (
            X.loc[miss, "Neighborhood"].map(self.frontage_by_nbhd_)
        )
        X["Lot Frontage"] = X["Lot Frontage"].fillna(self.frontage_global_)
        X[self.num_cols_] = X[self.num_cols_].fillna(self.num_medians_)

        for c in self.cat_cols_:
            X[c] = X[c].where(X[c].isin(self.levels_[c]), "Other")
            X[c] = pd.Categorical(X[c], categories=self.levels_[c] + ["Other"])

        return pd.get_dummies(X, columns=self.cat_cols_, drop_first=True,
                              dtype=np.float64)

    def transform(self, X):
        design = self._design(X).reindex(columns=self.feature_names_, fill_value=0.0)
        arr = design.to_numpy(dtype=np.float64)
        return (arr - self.mean_) / self.scale_


# -------------------------------------------------------------- 7. orchestration

def build():
    df = load_and_scope()
    df = fill_structural(df)
    df = encode_ordinals(df)
    df = add_derived(df)

    train, val, test = chrono_split(df)
    X_tr_raw, y_tr = split_xy(train)
    X_va_raw, y_va = split_xy(val)
    X_te_raw, y_te = split_xy(test)

    pre = Preprocessor().fit(X_tr_raw)
    X_tr = pre.transform(X_tr_raw)
    X_va = pre.transform(X_va_raw)
    X_te = pre.transform(X_te_raw)

    assert not np.isnan(X_tr).any(), "NaN survived preprocessing"
    print(f"design matrix: {X_tr.shape[1]} features")
    return dict(X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va,
                X_test=X_te, y_test=y_te, pre=pre,
                raw_train=train, raw_val=val, raw_test=test)


if __name__ == "__main__":
    d = build()
    for k in ["X_train", "X_val", "X_test"]:
        print(k, d[k].shape, "mean", round(float(d[k].mean()), 4),
              "sd", round(float(d[k].std()), 3))
