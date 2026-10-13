import numpy as np
import pandas as pd

TRAIN_END = "2014-01-01"
VAL_END = "2014-07-01"
RARE_THRESHOLD = 0.01

EMP_MAP = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
           "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
           "8 years": 8, "9 years": 9, "10+ years": 10}

CAT_COLS = ["home_ownership", "verification_status", "purpose"]
CLIP_COLS = ["loan_to_income", "credit_history_years", "revol_util", "dti",
             "delinq_2yrs", "inq_last_6mths", "log_revol_bal"]


def load_data(path="data/loans.csv"):
    df = pd.read_csv(path, parse_dates=["issue_d"])
    return df.sort_values("issue_d").reset_index(drop=True)


def split_by_date(df):
    train = df[df["issue_d"] < TRAIN_END]
    val = df[(df["issue_d"] >= TRAIN_END) & (df["issue_d"] < VAL_END)]
    test = df[df["issue_d"] >= VAL_END]
    return train, val, test


def build_raw_features(df):
    """Row-wise transformations only: nothing here is learned from data."""
    inc = df["annual_inc"].clip(lower=1)
    out = pd.DataFrame(index=df.index)
    out["fico_mid"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    out["loan_to_income"] = df["loan_amnt"] / inc
    out["log_annual_inc"] = np.log(inc)
    out["log_loan_amnt"] = np.log(df["loan_amnt"])
    cr = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    out["credit_history_years"] = (df["issue_d"] - cr).dt.days / 365.25
    out["emp_length_num"] = df["emp_length"].str.strip().map(EMP_MAP)
    out["emp_length_missing"] = df["emp_length"].isna().astype(int)
    out["has_delinq_history"] = df["mths_since_last_delinq"].notna().astype(int)
    out["mths_since_delinq"] = df["mths_since_last_delinq"]
    out["log_revol_bal"] = np.log1p(df["revol_bal"])
    out["revol_util"] = df["revol_util"]
    out["dti"] = df["dti"]
    out["delinq_2yrs"] = df["delinq_2yrs"]
    out["inq_last_6mths"] = df["inq_last_6mths"]
    out["pub_rec_any"] = (df["pub_rec"] > 0).astype(int)
    return out


class Preprocessor:
    """Everything learned from data (medians, clip bounds, rare categories,
    scaling) is fitted on the TRAIN set only and reused for val/test."""

    def fit(self, df):
        raw = build_raw_features(df)
        self.medians = raw.median()
        raw = raw.fillna(self.medians)
        self.clip_lo = raw[CLIP_COLS].quantile(0.01)
        self.clip_hi = raw[CLIP_COLS].quantile(0.99)

        self.cat_levels = {}
        for c in CAT_COLS:
            freq = df[c].value_counts(normalize=True)
            self.cat_levels[c] = freq[freq >= RARE_THRESHOLD].index.tolist()

        X = self._assemble(df)
        self.feature_names = list(X.columns)
        self.mean_ = X.mean()
        self.std_ = X.std(ddof=0).replace(0, 1)
        return self

    def _assemble(self, df):
        raw = build_raw_features(df).fillna(self.medians)
        raw[CLIP_COLS] = raw[CLIP_COLS].clip(self.clip_lo, self.clip_hi, axis=1)
        parts = [raw]
        for c, levels in self.cat_levels.items():
            col = df[c].where(df[c].isin(levels), "rare")
            # first level (most frequent) is the reference category
            for lvl in levels[1:] + ["rare"]:
                parts.append(pd.DataFrame({f"{c}={lvl}": (col == lvl).astype(int)},
                                          index=df.index))
        return pd.concat(parts, axis=1)

    def transform(self, df):
        X = self._assemble(df)[self.feature_names]
        return ((X - self.mean_) / self.std_).to_numpy(dtype=np.float32)
        