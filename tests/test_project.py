"""Check leakage, saved-model agreement, and inference without training."""
import importlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from features import METHODS, ROOT, calendar_features, date_splits, load_data, rental_units, scale_features
from models import DTYPE, LinearDemand


def test_no_target_or_weather_leakage_and_temporal_preprocessing():
    data = load_data()
    changed = data.copy()
    excluded = ["cnt", "casual", "registered", "instant", "temp", "atemp", "hum", "windspeed", "weathersit"]
    changed[excluded] = 123456
    pd.testing.assert_frame_equal(calendar_features(data), calendar_features(changed))
    split = date_splits(data)
    assert all(mask.any() for mask in split.values())
    assert np.all(sum(mask.astype(int) for mask in split.values()) == 1)
    assert data.loc[split["train"], "timestamp"].max() < data.loc[split["validation"], "timestamp"].min()
    assert data.loc[split["validation"], "timestamp"].max() < data.loc[split["test"], "timestamp"].min()
    day_membership = pd.DataFrame({"day": data["dteday"], "split": np.select(list(split.values()), list(split), default="missing")})
    assert day_membership.groupby("day")["split"].nunique().max() == 1
    bundle = joblib.load(ROOT / "artifacts" / "preprocessing.joblib")
    train_features = calendar_features(data.loc[split["train"]])
    np.testing.assert_allclose(bundle["scaler"].mean_, train_features.mean(), atol=1e-12)
    assert bundle["target_mean"] == data.loc[split["train"], "cnt"].mean()


def test_saved_models_reproduce_predictions_and_share_objective():
    torch.set_num_threads(1)
    out = ROOT / "artifacts"
    data = load_data()
    held_out = data.loc[date_splits(data)["test"]]
    bundle = joblib.load(out / "preprocessing.joblib")
    x = scale_features(held_out, bundle)
    tx = torch.tensor(x, dtype=DTYPE)
    ridge = joblib.load(out / "sklearn.joblib")
    manual = torch.load(out / "manual.pt", weights_only=True)
    standard = LinearDemand(x.shape[1])
    standard.load_state_dict(torch.load(out / "standard.pt", weights_only=True))
    with torch.no_grad():
        values = [ridge.predict(x), (tx @ manual["weights"] + manual["bias"]).numpy(), standard(tx).numpy()]
    saved = pd.read_csv(out / "test_predictions.csv")
    np.testing.assert_array_equal(saved["cnt"], held_out["cnt"])
    for method, predicted in zip(METHODS, values):
        np.testing.assert_allclose(rental_units(predicted, bundle), saved[method], atol=1e-8)
    for method in METHODS[1:]:
        np.testing.assert_allclose(saved[METHODS[0]], saved[method], rtol=0, atol=0.01)
    summary = json.loads((out / "metrics.json").read_text())
    assert ridge.alpha == len(data.loc[date_splits(data)["train"]]) * bundle["regularization"]
    assert np.ptp(list(summary["training_objective"].values())) < 1e-10


def test_dashboard_loads_and_predicts_without_fitting_or_writing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Inference must not train a model")
    for model in (Ridge, StandardScaler, DummyRegressor):
        monkeypatch.setattr(model, "fit", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    import models
    monkeypatch.setattr(models, "train_manual", forbidden)
    monkeypatch.setattr(models, "train_standard", forbidden)
    files = list((ROOT / "artifacts").iterdir())
    before = {file.name: file.read_bytes() for file in files}
    dashboard = importlib.import_module("app")
    for method in dashboard.ALL_METHODS:
        scatter, timeline = dashboard.comparison_plots(method)
        assert len(scatter.data[0].x) == 2168
        assert len(timeline.data) == 2
    for subset in ("Train", "Validation", "Test", "All data"):
        for feature in dashboard.DISTRIBUTIONS:
            histogram, seasonal = dashboard.explore(feature, subset)
            assert len(histogram.data) > 0 and len(seasonal.data) > 0
    table, note = dashboard.predict_scenario("2012-10-01", 12)
    assert len(table) == 4
    assert np.isfinite(table["Predicted hourly rentals"]).all()
    recorded = dashboard.test[(dashboard.test["dteday"] == "2012-10-01") & (dashboard.test["hr"] == 12)].iloc[0]
    np.testing.assert_allclose(table["Predicted hourly rentals"].iloc[:3], [recorded[method] for method in METHODS], atol=0.0051)
    with pytest.raises(dashboard.gr.Error):
        dashboard.predict_scenario("1900-01-01", 12)
    assert before == {file.name: file.read_bytes() for file in files}
