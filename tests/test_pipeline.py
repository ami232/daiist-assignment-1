"""Checks on the decisions that are easy to get silently wrong. Run with `uv run pytest`."""

import numpy as np
import pandas as pd

from volmodel import config, models as M
from volmodel.data import load_market_data
from volmodel.dataset import build_dataset
from volmodel.features import CANDIDATE_FEATURES, FEATURES, build_features
from volmodel.preprocess import Preprocessor


def test_features_use_no_future_data():
    """Features computed on data cut at a date must equal the features computed on the full history."""
    market, _ = load_market_data()
    for cut in ("2008-10-15", "2020-03-16"):
        full = build_features(market).loc[:cut, CANDIDATE_FEATURES].dropna()
        truncated = build_features(market.loc[:cut]).loc[:, CANDIDATE_FEATURES].dropna()
        pd.testing.assert_frame_equal(full, truncated)


def test_label_windows_stay_inside_their_block():
    f, _ = build_dataset()
    train, val, test = (f[f["split"] == s] for s in ("train", "val", "test"))
    assert train["label_end"].max() <= pd.Timestamp(config.TRAIN_END)
    assert val.index.min() >= pd.Timestamp(config.VAL_START)
    assert val["label_end"].max() <= pd.Timestamp(config.VAL_END)
    assert test.index.min() >= pd.Timestamp(config.TEST_START)
    assert f.loc[f["split"] == "live", "vol_rises"].isna().all()


def test_preprocessor_uses_training_statistics_only():
    f, _ = build_dataset()
    train = f[f["split"] == "train"]
    pp = Preprocessor(FEATURES).fit(train)
    x = pp.transform(train)
    assert np.allclose(x.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(x.std(axis=0), 1, atol=1e-3)


def test_three_implementations_reach_the_same_solution():
    f, _ = build_dataset()
    train = f[f["split"] == "train"]
    pp = Preprocessor(FEATURES).fit(train)
    X, y = pp.transform(train), train["vol_rises"].to_numpy()
    C = 0.01
    lam = M.l2_strength(C, len(y))
    sk = M.fit_sklearn(X, y, C)
    manual, _ = M.fit_torch_manual(X, y, lam, lr=0.5, epochs=1500)
    module, _ = M.fit_torch_module(X, y, lam, lr=0.5, epochs=1500, seed=0)
    p = M.predict_all(sk, manual, module, X)
    assert np.abs(p["sklearn"] - p["torch_manual"]).max() < 1e-3
    assert np.abs(p["sklearn"] - p["torch_module"]).max() < 1e-3
