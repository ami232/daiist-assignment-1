"""Calendar features shared by training and the dashboard."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METHODS = ["scikit-learn", "Manual PyTorch", "Standard PyTorch"]

def load_data():
    data = pd.read_csv(ROOT / "data" / "hour.csv")
    data["timestamp"] = pd.to_datetime(data["dteday"]) + pd.to_timedelta(data["hr"], unit="h")
    return data.sort_values("timestamp").reset_index(drop=True)

def date_splits(data):
    train = data["timestamp"] < "2012-07-01"
    validation = (data["timestamp"] >= "2012-07-01") & (data["timestamp"] < "2012-10-01")
    test = data["timestamp"] >= "2012-10-01"
    return {"train": train, "validation": validation, "test": test}

def calendar_features(data, engineered=True):
    """Give each season its own hourly pattern using indicator columns.

    Exclude counts, IDs, and weather. Fixed category ranges come from the
    dictionary. Omit the first category because it belongs to the intercept.
    """
    result = data[["yr", "holiday", "workingday"]].astype("float64").copy()
    categories = {"weekday": (data["weekday"], range(7))}
    if engineered:
        categories["season_hour"] = ((data["season"] - 1) * 24 + data["hr"], range(96))
    else:
        categories["hour"] = (data["hr"], range(24))
        categories["season"] = (data["season"], range(1, 5))
    for name, (values, levels) in categories.items():
        categorical = pd.Series(pd.Categorical(values, categories=list(levels)), index=data.index)
        encoded = pd.get_dummies(categorical, prefix=name, drop_first=True, dtype=float)
        result = pd.concat([result, encoded], axis=1)
    return result.astype("float64")

def scale_features(data, bundle):
    features = calendar_features(data, engineered=bundle["engineered"])
    if list(features.columns) != bundle["feature_names"]:
        raise ValueError("Feature columns differ from training.")
    return bundle["scaler"].transform(features)

def rental_units(values, bundle):
    return np.asarray(values).reshape(-1) * bundle["target_scale"] + bundle["target_mean"]
