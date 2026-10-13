import joblib, numpy as np, torch
from train import get_data, load, engineer, split

pre, (Xtr, ytr), (Xva, yva), (Xte, yte) = get_data()
sk = joblib.load("models/sklearn_model.joblib")
m = torch.load("models/manual_pytorch.pt")
w, b = m["w"].numpy(), m["b"].numpy()

def mse(p, y): return float(np.mean((p - y) ** 2))
print("Train MSE (log): sklearn", mse(sk.predict(Xtr), ytr),
      "| manual", mse(Xtr @ w + b, ytr))

gap = np.abs(sk.predict(Xte) - (Xte @ w + b))
test_df = split(engineer(load()))[2].copy()
test_df["gap"] = gap
test_df["sk_pred"] = np.exp(sk.predict(Xte))
test_df["manual_pred"] = np.exp(Xte @ w + b)
print(test_df.nlargest(5, "gap")[["Neighborhood", "Gr Liv Area", "Sale Condition",
      "SalePrice", "sk_pred", "manual_pred", "gap"]])
print("Median gap:", float(np.median(gap)), "| houses with gap > 0.2:", int((gap > 0.2).sum()))
