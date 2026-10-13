"""
preprocessing.py - Step 5: time-based split and turning columns into numbers.

Anything that LEARNS from data (scaling averages, which
countries are frequent) is fitted on the TRAINING set only, then simply
applied to validation and test.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# ---------------------------------------------------------------
# DECISIONS
# ---------------------------------------------------------------
TRAIN_SHARE = 0.70   # oldest 70% of bookings -> learn
VAL_SHARE = 0.15     # next 15% -> choose regularization / learning rate
# remaining 15% (most recent) -> final test, used ONCE at the end
TOP_COUNTRIES = 10   # keep the 10 most common countries, rest -> "infrequent"


def time_split(df: pd.DataFrame, date_col: str = "arrival_date"):
    """Split by arrival DATE: past -> train, then validation, newest -> test.

    We cut on dates (not row numbers) so all bookings arriving on the same
    day - e.g. the rooms of one group booking - end up on the same side.
    """
    train_end = df[date_col].quantile(TRAIN_SHARE)
    val_end = df[date_col].quantile(TRAIN_SHARE + VAL_SHARE)

    train = df[df[date_col] <= train_end]
    val = df[(df[date_col] > train_end) & (df[date_col] <= val_end)]
    test = df[df[date_col] > val_end]
    return train, val, test


def build_preprocessor() -> ColumnTransformer:
    """Recipe for turning X into a table of numbers. Not fitted yet."""
    numeric = StandardScaler()  # each column -> mean 0, standard deviation 1

    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",  # unseen category in test -> "infrequent"
        min_frequency=20,                      # categories seen < 20 times -> "infrequent"
        sparse_output=False,                   # plain table (PyTorch needs this)
    )
    country = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        max_categories=TOP_COUNTRIES + 1,      # top 10 + one "infrequent" group
        sparse_output=False,
    )

    other_categoricals = [c for c in CATEGORICAL_FEATURES if c != "country"]
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, other_categoricals),
        ("country", country, ["country"]),
    ])


if __name__ == "__main__":
    # Check this step on its own: uv run python preprocessing.py
    from data_prep import clean, load_raw
    from features import add_features, get_X_y

    data = add_features(clean(load_raw(), verbose=False))
    train, val, test = time_split(data)

    print("Split by arrival date:")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name:5s}: {len(part):6,} rows | "
              f"{part['arrival_date'].min().date()} -> {part['arrival_date'].max().date()} | "
              f"cancel rate {part['is_canceled'].mean():.1%}")

    X_train, y_train = get_X_y(train)
    X_val, _ = get_X_y(val)

    pre = build_preprocessor()
    X_train_num = pre.fit_transform(X_train)   # LEARN on train
    X_val_num = pre.transform(X_val)           # only APPLY on validation

    print("\nAfter preprocessing:")
    print("  X_train:", X_train_num.shape, "| X_val:", X_val_num.shape)
    print("  Numeric columns on train -> mean ~0:", X_train_num[:, 0].mean().round(3),
          "| std ~1:", X_train_num[:, 0].std().round(3))

    names = pre.get_feature_names_out()
    print("\nCountries kept as their own column:")
    print([n.replace("country__country_", "") for n in names if n.startswith("country__")])
    print(f"\nTotal input columns for the model: {len(names)}")