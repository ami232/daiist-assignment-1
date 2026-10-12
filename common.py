# Shared data loading, feature engineering, preprocessing, and metrics. Used by both train.py and app.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
DATA_PATH = "data/housing.csv"
PRICE_CAP = 500001

CITIES = {
    "sf": (37.7749, -122.4194),
    "la": (34.0522, -118.2437),
    "sd": (32.7157, -117.1611),
    "sj": (37.3382, -121.8863),
    "sac": (38.5816, -121.4944),
}

LOG_COLS = ["total_rooms", "total_bedrooms", "population", "households", "median_income"]
NUMERIC_COLS = [
    "housing_median_age", "total_rooms", "total_bedrooms", "population", "households",
    "median_income", "rooms_per_household", "bedrooms_per_room", "population_per_household",
    "dist_to_nearest_city_km", "latitude", "longitude",
]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_raw(path=DATA_PATH):
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["median_house_value"] < PRICE_CAP].reset_index(drop=True)
    dropped = before - len(df)
    print(f"[data] dropped {dropped} price-capped rows ({dropped/before:.1%}); {len(df)} remain")
    return df


def split_data(df):
    train_val, test = train_test_split(
        df, test_size=0.20, random_state=RANDOM_STATE, stratify=df["ocean_proximity"]
    )
    train, val = train_test_split(
        train_val, test_size=0.20, random_state=RANDOM_STATE, stratify=train_val["ocean_proximity"]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def add_engineered_features(df):
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    dist_cols = []
    for name, (clat, clon) in CITIES.items():
        col = f"dist_to_{name}_km"
        df[col] = haversine_km(df["latitude"], df["longitude"], clat, clon)
        dist_cols.append(col)
    df["dist_to_nearest_city_km"] = df[dist_cols].min(axis=1)
    df = df.drop(columns=dist_cols)
    return df


def fit_preprocessing(train):
    bedroom_imputer = SimpleImputer(strategy="median")
    bedroom_imputer.fit(train[["total_bedrooms"]])
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(train[["ocean_proximity"]])
    return bedroom_imputer, ohe


def transform(df, bedroom_imputer, ohe, scaler=None, fit_scaler=False):
    df = df.copy()
    df["total_bedrooms"] = bedroom_imputer.transform(df[["total_bedrooms"]])
    df = add_engineered_features(df)

    for col in LOG_COLS:
        df[col] = np.log1p(df[col])

    ohe_arr = ohe.transform(df[["ocean_proximity"]])
    ohe_df = pd.DataFrame(
        ohe_arr, columns=ohe.get_feature_names_out(["ocean_proximity"]), index=df.index
    )

    X = pd.concat([df[NUMERIC_COLS], ohe_df], axis=1)
 
    y = df["median_house_value"].values.astype(np.float32) if "median_house_value" in df.columns else None

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_scaled, y, scaler


# Metrics

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def all_metrics(y_true, y_pred):
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "R2": r2(y_true, y_pred)}