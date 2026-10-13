"""
Assignment 1 -- training pipeline (the ONLY stage that trains anything).

    uv run python main.py train

Steps: load committed sample -> temporal split -> feature engineering -> train the SAME
ridge-regularised linear regression three ways (scikit-learn, manual PyTorch loop,
nn.Module + torch.optim) -> compare with the naive baseline (prediction 0, i.e. trust mmr)
-> fix the bidding cushion X from VALIDATION errors -> save every artifact the app needs.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from common import (
    ARTIFACTS, BID_CAP, CUSHION_QUANTILE, build_preprocessor, cushion_from_validation,
    engineer, load_raw, rmse, signed_error, simulate_bids, split_name,
)

SEED = 42
EPOCHS = 4000
ALPHAS = [0.1, 1, 10, 100, 300, 1000, 3000]      # ridge strength (sum-of-squares form)
LRS = [0.01, 0.03, 0.1, 0.3]                      # full-batch learning rates

torch.manual_seed(SEED)
np.random.seed(SEED)
ARTIFACTS.mkdir(exist_ok=True)

# ----------------------------------------------------------------- 1. data + temporal split
df = load_raw()
df["split"] = split_name(df["sale_date"])
tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
print(f"rows: train={len(tr)} val={len(va)} test={len(te)}")

# audit: the same car (VIN) must not sit in train AND test. Temporal split makes it rare.
vins = {k: set(x["vin"].dropna()) for k, x in (("train", tr), ("val", va), ("test", te))}
vin_overlap = {"train_test": len(vins["train"] & vins["test"]),
               "train_val": len(vins["train"] & vins["val"]),
               "val_test": len(vins["val"] & vins["test"])}
print("VINs shared between splits:", vin_overlap)

# ----------------------------------------------------------------- 2. features
pre = build_preprocessor()
Xtr = pre.fit_transform(engineer(tr))            # fitted on TRAIN only
Xva, Xte = pre.transform(engineer(va)), pre.transform(engineer(te))
feature_names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
ytr, yva, yte = tr["target"].values, va["target"].values, te["target"].values
n, d = Xtr.shape
print(f"design matrix: {n} x {d}")
joblib.dump(pre, ARTIFACTS / "preprocessor.joblib")

# ----------------------------------------------------------------- 3a. scikit-learn
rows = [(a, rmse(yva, Ridge(alpha=a).fit(Xtr, ytr).predict(Xva))) for a in ALPHAS]
alpha_table = pd.DataFrame(rows, columns=["alpha", "val_rmse"])
print(alpha_table.to_string(index=False))
ALPHA = float(alpha_table.sort_values("val_rmse").iloc[0]["alpha"])
sk = Ridge(alpha=ALPHA).fit(Xtr, ytr)
joblib.dump(sk, ARTIFACTS / "sklearn_ridge.joblib")
print("chosen alpha:", ALPHA)

# sklearn's Ridge minimises ||y - Xw||^2 + alpha ||w||^2 ; dividing by n gives the loss both
# PyTorch versions minimise:  MSE + (alpha/n) ||w||^2   (bias not penalised)
L2 = ALPHA / n
Xtr_t, ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
Xva_t, Xte_t = torch.tensor(Xva, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)


# ----------------------------------------------------------------- 3b. manual PyTorch loop
def train_manual(lr, epochs=EPOCHS):
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    curve = []
    for _ in range(epochs):
        pred = Xtr_t @ w + b
        loss = ((pred - ytr_t) ** 2).mean() + L2 * (w ** 2).sum()
        loss.backward()                       # autograd computes dL/dw, dL/db
        with torch.no_grad():                 # manual gradient-descent update
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        curve.append(loss.item())
    return w.detach(), b.detach(), curve


# ----------------------------------------------------------------- 3c. standard PyTorch workflow
class LinearModel(torch.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.linear = torch.nn.Linear(d_in, 1)
        torch.nn.init.zeros_(self.linear.weight)   # same start point as the manual loop
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_nn(lr, epochs=EPOCHS):
    model = LinearModel(d)
    loss_fn = torch.nn.MSELoss()
    # SGD's weight_decay adds wd*w to the gradient == loss term 0.5*wd*||w||^2,
    # so wd = 2*L2 reproduces the same ridge penalty. Bias is excluded.
    opt = torch.optim.SGD(
        [{"params": [model.linear.weight], "weight_decay": 2 * L2},
         {"params": [model.linear.bias], "weight_decay": 0.0}], lr=lr)
    curve = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
        curve.append(loss.item() + L2 * (model.linear.weight ** 2).sum().item())
    return model, curve


def predict_manual(w, b, X):
    return (X @ w + b).detach().numpy()


# Learning-rate selection. The three implementations must minimise the SAME objective, so the
# learning rate is the one that gets the PyTorch loops closest to that minimum (lowest final
# TRAINING loss, 6 decimals; ties -> larger lr, which is the more converged one when stable). Picking by validation RMSE would reward an under-converged
# run (lr=0.01 stops short of the ridge solution and its val RMSE differs only in the 5th decimal).
lr_rows, best = [], {"manual": None, "nn": None}
for lr in LRS:
    t0 = time.time()
    w, b, c = train_manual(lr)
    pm = predict_manual(w, b, Xva_t)
    model, c2 = train_nn(lr)
    pn = model(Xva_t).detach().numpy()
    ok_m, ok_n = np.isfinite(pm).all(), np.isfinite(pn).all()
    lr_rows.append({"lr": lr,
                    "manual_val_rmse": rmse(yva, pm) if ok_m else float("nan"),
                    "nn_val_rmse": rmse(yva, pn) if ok_n else float("nan"),
                    "manual_final_loss": c[-1], "nn_final_loss": c2[-1]})
    if ok_m and (best["manual"] is None or round(c[-1], 6) <= best["manual"][0]):
        best["manual"] = (round(c[-1], 6), lr, w, b, c)
    if ok_n and (best["nn"] is None or round(c2[-1], 6) <= best["nn"][0]):
        best["nn"] = (round(c2[-1], 6), lr, model, c2)
    print(f"lr={lr}: {time.time()-t0:.1f}s")
lr_table = pd.DataFrame(lr_rows)
print(lr_table.to_string(index=False))

_, LR_M, w_m, b_m, curve_m = best["manual"]
_, LR_N, nn_model, curve_n = best["nn"]
print(f"chosen lr: manual={LR_M}, nn={LR_N}")
torch.save({"w": w_m, "b": b_m, "lr": LR_M, "alpha": ALPHA}, ARTIFACTS / "manual_torch.pt")
torch.save({"state_dict": nn_model.state_dict(), "d_in": d, "lr": LR_N, "alpha": ALPHA},
           ARTIFACTS / "nn_linear.pt")


# ----------------------------------------------------------------- 4. evaluation (RMSE main, signed error complement)
def all_preds(X, Xt):
    return {
        "sklearn": sk.predict(X),
        "manual_torch": predict_manual(w_m, b_m, Xt),
        "nn_module": nn_model(Xt).detach().numpy(),
        "baseline_mmr": np.zeros(len(X)),            # naive: sellingprice = mmr
    }


P = {"val": all_preds(Xva, Xva_t), "test": all_preds(Xte, Xte_t)}
Y = {"val": yva, "test": yte}
D = {"val": va, "test": te}
metrics = {}
for split in ("val", "test"):
    metrics[split] = {}
    for name, p in P[split].items():
        metrics[split][name] = {
            "rmse": rmse(Y[split], p),
            "mean_signed_error": signed_error(Y[split], p),      # pred - actual; + = overestimates
            "mae": float(np.mean(np.abs(Y[split] - p))),
            "overestimate_share": float(np.mean(p > Y[split])),
        }
print(json.dumps(metrics["test"], indent=1))

t = P["test"]
agree = {
    "max_abs_pred_diff_manual_vs_sklearn": float(np.abs(t["manual_torch"] - t["sklearn"]).max()),
    "max_abs_pred_diff_nn_vs_sklearn": float(np.abs(t["nn_module"] - t["sklearn"]).max()),
    "max_abs_pred_diff_manual_vs_nn": float(np.abs(t["manual_torch"] - t["nn_module"]).max()),
    "max_abs_coef_diff_manual_vs_sklearn": float(np.abs(w_m.numpy() - sk.coef_).max()),
    "max_abs_coef_diff_nn_vs_sklearn": float(
        np.abs(nn_model.linear.weight.detach().numpy().ravel() - sk.coef_).max()),
}
print(json.dumps(agree, indent=1))

# ----------------------------------------------------------------- 5. bidding cushion X (VALIDATION only)
cushion_by_model = {name: cushion_from_validation(p, yva) for name, p in P["val"].items()}
CUSHION = cushion_by_model["sklearn"]
print("cushion X from validation errors:", cushion_by_model)

business = {}
for split in ("val", "test"):
    business[split] = {}
    for name, p in P[split].items():
        x = cushion_by_model[name]
        sim = simulate_bids(D[split]["mmr"], D[split]["sellingprice"], p, x)
        sim["overestimate_after_cushion_share"] = float(np.mean(p - x > Y[split]))
        business[split][name] = {"cushion": x, **sim}
print(json.dumps(business, indent=1))

# ----------------------------------------------------------------- 6. save artifacts for the app
for split in ("val", "test"):
    out = D[split][["sale_date", "make", "model", "year", "mmr", "sellingprice", "target"]].copy()
    for name, p in P[split].items():
        out[f"pred_{name}"] = p
    out.to_csv(ARTIFACTS / f"predictions_{split}.csv", index=False)

pd.DataFrame({"feature": feature_names, "coef_sklearn": sk.coef_, "coef_manual": w_m.numpy(),
              "coef_nn": nn_model.linear.weight.detach().numpy().ravel()}
             ).to_csv(ARTIFACTS / "coefficients.csv", index=False)

summary = {
    "n_rows": {"train": len(tr), "val": len(va), "test": len(te)},
    "n_features": int(d), "alpha": ALPHA, "lr": {"manual": LR_M, "nn": LR_N}, "epochs": EPOCHS,
    "alpha_table": alpha_table.to_dict("records"), "lr_table": lr_table.to_dict("records"),
    "metrics": metrics, "agreement": agree, "business": business,
    "cushion_default": CUSHION, "cushion_by_model": cushion_by_model,
    "cushion_quantile": CUSHION_QUANTILE, "bid_cap": BID_CAP, "vin_overlap": vin_overlap,
    "loss_curves": {"manual_torch": curve_m[::10], "nn_module": curve_n[::10]},
}
(ARTIFACTS / "summary.json").write_text(json.dumps(summary, indent=1))
print("artifacts saved in", ARTIFACTS)
