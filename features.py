"""Feature engineering + preprocessing, shared by train.py and app.py.

Everything that learns something from data (clip bounds, scaler mean/std)
is fit on the TRAIN split only, inside `Preprocessor.fit`, so validation and
test rows never leak into the preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer

RAW_FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "is_red",
]

PREMIUM_MIN_QUALITY = 7  # target: premium = quality >= 7

# Right-skewed concentration features: log1p makes them closer to symmetric,
# so one unit of a coefficient means roughly the same thing across the range.
LOG_FEATURES = ["residual_sugar", "chlorides", "free_sulfur_dioxide",
                "total_sulfur_dioxide", "sulphates", "volatile_acidity"]

# Red and white wines are made differently; these are the features whose
# effect on quality plausibly differs by colour, so they get an is_red interaction.
COLOR_INTERACTIONS = ["volatile_acidity", "sulphates", "residual_sugar",
                      "citric_acid", "alcohol"]


def make_target(df: pd.DataFrame) -> pd.Series:
    return (df["quality"] >= PREMIUM_MIN_QUALITY).astype(int)


# Features whose relation to "premium" is not monotone / not straight-line
# (a sweet spot rather than "more is better"). Chosen by 5-fold CV on the
# dev set: these four were the only ones where a spline beat noise.
SPLINE_FEATURES = ["total_sulfur_dioxide", "density", "volatile_acidity",
                   "free_sulfur_dioxide"]

FINAL_GROUPS = ("log", "color", "spline")


def engineer(df: pd.DataFrame, groups=FINAL_GROUPS) -> pd.DataFrame:
    """Raw chemistry -> model features. `groups` lets train.py run ablations."""
    X = df[RAW_FEATURES].copy()

    if "chem" in groups:
        # Bound SO2 = the part tied up with sugars/aldehydes, not protective.
        X["bound_sulfur_dioxide"] = X["total_sulfur_dioxide"] - X["free_sulfur_dioxide"]
        X["free_so2_ratio"] = X["free_sulfur_dioxide"] / X["total_sulfur_dioxide"].clip(lower=1)
        # Molecular SO2: the antimicrobial fraction of free SO2, which depends on pH
        # (pKa ~1.81). Winemakers target ~0.5-0.8 mg/L.
        X["molecular_so2"] = X["free_sulfur_dioxide"] / (1 + 10 ** (X["pH"] - 1.81))
        # Density is mostly explained by alcohol (lighter) and sugar (heavier);
        # dividing removes that shared signal and keeps a "body" measure.
        X["alcohol_per_density"] = X["alcohol"] / X["density"]
        X["sugar_alcohol_ratio"] = X["residual_sugar"] / X["alcohol"]
        X["total_acidity"] = X["fixed_acidity"] + X["volatile_acidity"] + X["citric_acid"]

    if "log" in groups:
        for c in LOG_FEATURES:
            X[c] = np.log1p(X[c])
        if "chem" in groups:
            X["bound_sulfur_dioxide"] = np.log1p(X["bound_sulfur_dioxide"].clip(lower=0))
            X["molecular_so2"] = np.log1p(X["molecular_so2"])

    if "color" in groups:
        for c in COLOR_INTERACTIONS:
            X[f"{c}_x_red"] = X[c] * X["is_red"]

    return X


class Preprocessor:
    """Winsorize at train quantiles, optionally add spline basis columns,
    then standardize with train mean/std.

    Splines: a cubic B-spline basis (4 quantile knots) per SPLINE_FEATURES
    column lets a *linear* logistic regression fit a smooth curve in that
    feature. include_bias=False drops one basis column so the basis does not
    sum to 1 (which would be perfectly collinear with the intercept), and the
    raw column is replaced by its basis for the same reason.

    Clipping keeps a handful of extreme rows (e.g. chlorides, free SO2) from
    dominating a linear model. Standardizing puts every feature on the same
    scale, which (1) makes the L2 penalty treat features fairly and (2) lets
    plain gradient descent in the PyTorch versions converge with one learning
    rate for every coefficient.
    """

    def __init__(self, lower_q=0.005, upper_q=0.995, spline_features=()):
        self.lower_q, self.upper_q = lower_q, upper_q
        self.spline_features = list(spline_features)

    def _add_splines(self, Xc: pd.DataFrame) -> pd.DataFrame:
        if not self.spline_features:
            return Xc
        B = self.spline.transform(Xc[self.spline_features])
        names = [f"{c}_spl{k}" for c in self.spline_features for k in range(B.shape[1] // len(self.spline_features))]
        # The cubic basis can already represent a straight line in x, so keeping
        # the raw column too would make X^T X singular: replace it instead.
        return pd.concat([Xc.drop(columns=self.spline_features),
                          pd.DataFrame(B, columns=names, index=Xc.index)], axis=1)

    def fit(self, X: pd.DataFrame):
        self.input_columns = list(X.columns)
        binary = [c for c in X.columns if set(X[c].unique()) <= {0, 1}]
        self.lo = X.quantile(self.lower_q)
        self.hi = X.quantile(self.upper_q)
        self.lo[binary], self.hi[binary] = 0, 1  # never clip 0/1 columns
        Xc = X.clip(self.lo, self.hi, axis=1)
        if self.spline_features:
            self.spline = SplineTransformer(n_knots=4, degree=3, knots="quantile",
                                            include_bias=False).fit(Xc[self.spline_features])
        Xc = self._add_splines(Xc)
        self.columns = list(Xc.columns)
        self.mean = Xc.mean()
        self.std = Xc.std(ddof=0).replace(0, 1)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self._add_splines(X[self.input_columns].clip(self.lo, self.hi, axis=1))
        return ((X - self.mean) / self.std).to_numpy(dtype=np.float64)


def build_preprocessor(groups=FINAL_GROUPS, **kwargs) -> Preprocessor:
    """"spline" is a preprocessing step (it is fit on train), so it lives here, not in engineer()."""
    return Preprocessor(spline_features=SPLINE_FEATURES if "spline" in groups else (), **kwargs)
