"""Price and stock columns must never influence the features.

build_features may receive the full raw frame (it does in train.py and app.py reads the
CSV too), so the guarantee has to be behavioural: scrambling every banned column must leave
the feature matrix byte-for-byte identical.
"""
import numpy as np

from src import data
from src.features import build_features

BANNED = data.PRICE_COLUMNS + data.LEAKY_COLUMNS


def test_banned_columns_do_not_touch_features():
    df = data.load_raw()
    train_df, test_df = data.split(df)

    scrambled = train_df.copy()
    rng = np.random.default_rng(0)
    for col in BANNED:
        assert col in scrambled.columns, f"{col} missing from the CSV: update the list"
        scrambled[col] = rng.permutation(scrambled[col].to_numpy())

    X_ref, fit_ref = build_features(train_df, fit=None)
    X_scr, fit_scr = build_features(scrambled, fit=None)
    assert fit_ref["columns"] == fit_scr["columns"]
    np.testing.assert_array_equal(X_ref.to_numpy(), X_scr.to_numpy())

    # Same check in reuse mode (what test rows and dashboard input go through).
    X_test_ref, _ = build_features(test_df, fit=fit_ref)
    scrambled_test = test_df.copy()
    for col in BANNED:
        scrambled_test[col] = rng.permutation(scrambled_test[col].to_numpy())
    X_test_scr, _ = build_features(scrambled_test, fit=fit_ref)
    np.testing.assert_array_equal(X_test_ref.to_numpy(), X_test_scr.to_numpy())


def test_no_banned_column_name_leaks_into_feature_names():
    df = data.load_raw()
    _, fit = build_features(df, fit=None)
    lowered = [c.lower() for c in fit["columns"]]
    for col in BANNED:
        assert col.lower() not in lowered
