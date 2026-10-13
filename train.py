"""Assignment 1 - training pipeline (step 1: split and cleaning)."""
import numpy as np
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/ames.csv"

# Columns where an empty value means "the house does not have this feature"
NONE_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]


def load_and_split():
    """Temporal split: the model is trained on the past and tested on the future."""
    df = pd.read_csv(DATA_PATH).drop(columns=["Id"])
    train = df[df["YrSold"] <= 2008].copy()
    val = df[df["YrSold"] == 2009].copy()
    test = df[df["YrSold"] == 2010].copy()
    return train, val, test


def fit_cleaning(train):
    """Learn the fill values using ONLY the training data (avoids leakage)."""
    return {
        "lot_by_neighborhood": train.groupby("Neighborhood")["LotFrontage"].median(),
        "lot_global": train["LotFrontage"].median(),
        "electrical_mode": train["Electrical"].mode()[0],
    }


def apply_cleaning(df, stats):
    """Apply the fill values learned on train to any split."""
    df = df.copy()
    df[NONE_COLS] = df[NONE_COLS].fillna("None")
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df["Electrical"] = df["Electrical"].fillna(stats["electrical_mode"])
    lot_fill = df["Neighborhood"].map(stats["lot_by_neighborhood"]).fillna(stats["lot_global"])
    df["LotFrontage"] = df["LotFrontage"].fillna(lot_fill)
    df = df.drop(columns=["GarageYrBlt"])
    return df


def engineer_features(df):
    """Create new features and drop the columns they replace."""
    df = df.copy()
    # Age of the house and of the last remodel, measured at the time of sale
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    # Total living area and total bathrooms (half baths count 0.5)
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    # Yes/no indicators
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageType"] != "None").astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    # Drop the originals used above (and YrSold, see note)
    return df.drop(columns=["YrSold", "YearBuilt", "YearRemodAdd",
                            "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
                            "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"])

QUALITY_COLS = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}


def encode_quality(df):
    """Turn the quality scales (None < Po < Fa < TA < Gd < Ex) into 0-5."""
    df = df.copy()
    for col in QUALITY_COLS:
        df[col] = df[col].map(QUALITY_MAP)
    return df

TARGET = "SalePrice"


def split_xy(df):
    """Separate the features (X) from the target (y)."""
    X = df.drop(columns=[TARGET]).copy()
    X["MSSubClass"] = X["MSSubClass"].astype(str)  # numeric code, but really a category
    return X, df[TARGET]


def fit_preprocessor(X_train):
    """Learn the scaling and one-hot encoding from TRAIN only."""
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return pre.fit(X_train)

ALPHAS = [1, 3, 10, 30, 100, 300]


def rmse(y_true, y_pred):
    """Root mean squared error, in dollars."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

LR = 0.02
EPOCHS = 10000


def train_manual(A, ys, alpha, lr, epochs):
    """Ridge regression by gradient descent, written by hand with raw tensors and autograd."""
    X = torch.tensor(A, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32).reshape(-1, 1)
    n, d = X.shape
    w = torch.zeros(d, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    for epoch in range(epochs):
        pred = X @ w + b
        loss = ((pred - y) ** 2).mean() + (alpha / n) * (w ** 2).sum()
        loss.backward()                  # autograd computes the gradients
        with torch.no_grad():            # manual gradient step (no optimizer)
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        if (epoch + 1) % 2000 == 0:
            print("    epoch %5d  loss %.4f" % (epoch + 1, loss.item()))
    return w.detach(), b.detach()


def predict_linear(A, w, b):
    """Predictions in the standardised scale."""
    return (torch.tensor(A, dtype=torch.float32) @ w + b).numpy().ravel()

class RidgeRegression(nn.Module):
    """A single linear layer: prediction = X @ w + b."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


def train_module(A, ys, alpha, lr, epochs):
    """Same model as train_manual, with nn.Module, nn.MSELoss and torch.optim.SGD."""
    X = torch.tensor(A, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32).reshape(-1, 1)
    n, d = X.shape
    model = RidgeRegression(d)
    loss_fn = nn.MSELoss()
    # weight_decay = 2*alpha/n gives exactly the Ridge penalty (alpha/n)*||w||^2; the bias is not penalised
    optimizer = torch.optim.SGD(
        [{"params": model.linear.weight, "weight_decay": 2 * alpha / n},
         {"params": model.linear.bias, "weight_decay": 0.0}],
        lr=lr,
    )
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2000 == 0:
            total = loss.item() + (alpha / n) * model.linear.weight.pow(2).sum().item()
            print("    epoch %5d  loss %.4f" % (epoch + 1, total))
    return model


def predict_module(model, A):
    """Predictions in the standardised scale."""
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(A, dtype=torch.float32)).numpy().ravel()

ARTIFACT_DIR = Path("artifacts")


def mae(y_true, y_pred):
    """Mean absolute error, in dollars."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def save_artifacts(pre, sk_model, w_man, b_man, nn_model, meta, predictions, results):
    """Save everything the dashboard needs, so app.py never has to retrain."""
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(pre, ARTIFACT_DIR / "preprocessor.joblib")
    joblib.dump(sk_model, ARTIFACT_DIR / "sklearn_ridge.joblib")
    torch.save({"w": w_man, "b": b_man}, ARTIFACT_DIR / "manual_pytorch.pt")
    torch.save(nn_model.state_dict(), ARTIFACT_DIR / "nn_module.pt")
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    predictions.to_csv(ARTIFACT_DIR / "predictions.csv", index=False)
    results.to_csv(ARTIFACT_DIR / "results.csv", index=False)

if __name__ == "__main__":
    train, val, test = load_and_split()
    print("train / val / test:", len(train), len(val), len(test))
    stats = fit_cleaning(train)
    train, val, test = (apply_cleaning(d, stats) for d in (train, val, test))
    train, val, test = (engineer_features(d) for d in (train, val, test))
    train, val, test = (encode_quality(d) for d in (train, val, test))
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = (split_xy(d) for d in (train, val, test))
    pre = fit_preprocessor(X_train)
    A_train, A_val, A_test = (pre.transform(X) for X in (X_train, X_val, X_test))

    # The target is standardised with TRAIN statistics; all three models learn on this scale
    y_mean, y_std = float(np.mean(y_train)), float(np.std(y_train))
    ys_train = (y_train.to_numpy() - y_mean) / y_std

    # Naive baseline: always predict the average training price
    print("baseline RMSE  val: %.0f  test: %.0f" % (rmse(y_val, y_mean), rmse(y_test, y_mean)))

    # scikit-learn Ridge: choose alpha on validation only
    scores = {}
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(A_train, ys_train)
        scores[alpha] = rmse(y_val, model.predict(A_val) * y_std + y_mean)
        print("  alpha=%-4s val RMSE: %.0f" % (alpha, scores[alpha]))
    best_alpha = min(scores, key=scores.get)
    sk_model = Ridge(alpha=best_alpha).fit(A_train, ys_train)
    sk_test = rmse(y_test, sk_model.predict(A_test) * y_std + y_mean)
    print("scikit-learn Ridge (alpha=%s)  val: %.0f  test: %.0f" % (best_alpha, scores[best_alpha], sk_test))

    # Manual PyTorch: same Ridge penalty (alpha chosen above), trained by gradient descent
    print("manual PyTorch loop (lr=%s, %s epochs):" % (LR, EPOCHS))
    w_man, b_man = train_manual(A_train, ys_train, best_alpha, LR, EPOCHS)
    man_val = rmse(y_val, predict_linear(A_val, w_man, b_man) * y_std + y_mean)
    man_test = rmse(y_test, predict_linear(A_test, w_man, b_man) * y_std + y_mean)
    print("manual PyTorch  val: %.0f  test: %.0f" % (man_val, man_test))

    # Standard PyTorch workflow: nn.Module + nn.MSELoss + torch.optim.SGD
    print("nn.Module + torch.optim (lr=%s, %s epochs):" % (LR, EPOCHS))
    nn_model = train_module(A_train, ys_train, best_alpha, LR, EPOCHS)
    nn_val = rmse(y_val, predict_module(nn_model, A_val) * y_std + y_mean)
    nn_test = rmse(y_test, predict_module(nn_model, A_test) * y_std + y_mean)
    print("nn.Module       val: %.0f  test: %.0f" % (nn_val, nn_test))

    # Predictions of the three models (in dollars) on validation and test
    predictions = pd.concat([
        pd.DataFrame({
            "split": name,
            "actual": y.to_numpy(),
            "sklearn": sk_model.predict(A) * y_std + y_mean,
            "manual": predict_linear(A, w_man, b_man) * y_std + y_mean,
            "nn_module": predict_module(nn_model, A) * y_std + y_mean,
        })
        for name, A, y in (("val", A_val, y_val), ("test", A_test, y_test))
    ], ignore_index=True)
    p_val = predictions[predictions["split"] == "val"]
    p_test = predictions[predictions["split"] == "test"]

    # Comparison table: three methods vs the naive baseline
    results = pd.DataFrame({"model": ["baseline (mean price)", "scikit-learn Ridge", "manual PyTorch", "nn.Module + optim"]})
    results["val_RMSE"] = [rmse(p_val["actual"], y_mean)] + [rmse(p_val["actual"], p_val[c]) for c in ("sklearn", "manual", "nn_module")]
    results["test_RMSE"] = [rmse(p_test["actual"], y_mean)] + [rmse(p_test["actual"], p_test[c]) for c in ("sklearn", "manual", "nn_module")]
    results["test_MAE"] = [mae(p_test["actual"], y_mean)] + [mae(p_test["actual"], p_test[c]) for c in ("sklearn", "manual", "nn_module")]
    print()
    print(results.round(0).to_string(index=False))

        # Do the three agree? Compare their predictions and weights
    gap = lambda a, b: float(np.max(np.abs(p_test[a] - p_test[b])))
    g1, g2, g3 = gap("sklearn", "manual"), gap("sklearn", "nn_module"), gap("manual", "nn_module")
    print("max |test prediction gap| in $: sklearn-manual %.0f, sklearn-nn %.0f, manual-nn %.0f" % (g1, g2, g3))
    w_gap = float(np.max(np.abs(sk_model.coef_ - w_man.numpy().ravel())))
    print("max |weight gap| (standardised units): sklearn-manual %.4f" % w_gap)

    meta = {"y_mean": y_mean, "y_std": y_std, "alpha": best_alpha, "lr": LR, "epochs": EPOCHS,
            "n_features": int(A_train.shape[1]), "split": "train<=2008, val=2009, test=2010"}
    save_artifacts(pre, sk_model, w_man, b_man, nn_model, meta, predictions, results)
    np.savez(ARTIFACT_DIR / "matrices.npz", A_val=A_val, A_test=A_test, y_val=y_val.to_numpy(), y_test=y_test.to_numpy())
    print("Saved models and results to", ARTIFACT_DIR)

    