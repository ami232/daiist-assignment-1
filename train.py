import pandas as pd

df = pd.read_csv("data/train.csv")

# El año viene en 2 cifras: pandas pone 65 como 2065, así que lo corregimos
fecha = pd.to_datetime(df["release_date"], format="%m/%d/%y")
fecha = fecha.where(fecha.dt.year <= 2019, fecha - pd.DateOffset(years=100))
df["release_date"] = fecha

# Split por fecha: 70% más antiguo para train, 15% validación, 15% más reciente para test
df = df.sort_values("release_date").reset_index(drop=True)
n = len(df)
train = df.iloc[: int(n * 0.70)].copy()
val = df.iloc[int(n * 0.70): int(n * 0.85)].copy()
test = df.iloc[int(n * 0.85):].copy()

# Target: éxito = recaudación por encima de la mediana del TRAIN
mediana = train["revenue"].median()
for parte in (train, val, test):
    parte["success"] = (parte["revenue"] > mediana).astype(int)

print(len(train), len(val), len(test))
print(train["success"].mean(), val["success"].mean(), test["success"].mean())




import ast
import numpy as np

def lista_generos(texto):
    if pd.isna(texto):
        return []
    return [g["name"] for g in ast.literal_eval(texto)]

# Todo lo que se "aprende" de los datos se calcula SOLO con train
mediana_budget = train.loc[train["budget"] > 0, "budget"].median()
mediana_runtime = train.loc[train["runtime"] > 0, "runtime"].median()
generos = sorted({g for t in train["genres"] for g in lista_generos(t)})

def construir_features(d):
    X = pd.DataFrame(index=d.index)
    X["budget_desconocido"] = (d["budget"] == 0).astype(int)
    X["log_budget"] = np.log1p(d["budget"].where(d["budget"] > 0, mediana_budget))
    X["runtime"] = d["runtime"].where(d["runtime"] > 0).fillna(mediana_runtime)
    X["es_saga"] = d["belongs_to_collection"].notna().astype(int)
    for m in range(1, 13):
        X[f"mes_{m}"] = (d["release_date"].dt.month == m).astype(int)
    gs = d["genres"].apply(lista_generos)
    for g in generos:
        X[f"genero_{g}"] = gs.apply(lambda lista: int(g in lista))
    return X

X_train, X_val, X_test = (construir_features(p) for p in (train, val, test))
y_train, y_val, y_test = train["success"], val["success"], test["success"]

print(X_train.shape, X_val.shape, X_test.shape)







import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Standardize: fit on train only, reuse on val and test
scaler = StandardScaler().fit(X_train)
Xtr, Xva, Xte = (scaler.transform(x) for x in (X_train, X_val, X_test))

# Naive baseline: always predict the most common class in train
clase_comun = int(y_train.mean() >= 0.5)
print("baseline val acc:", accuracy_score(y_val, [clase_comun] * len(y_val)))

# Model 1: scikit-learn
sk = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, y_train)
p = sk.predict_proba(Xva)[:, 1]
print("sklearn val acc:", accuracy_score(y_val, p > 0.5), "auc:", roc_auc_score(y_val, p))

# Save, because the Gradio app has to load these and never retrain
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(sk, "models/sklearn_logreg.joblib")







#Model 2

import torch

torch.manual_seed(0)
Xt = torch.tensor(Xtr, dtype=torch.float32)
yt = torch.tensor(y_train.values, dtype=torch.float32)
Xv = torch.tensor(Xva, dtype=torch.float32)

C = 1.0          
lr = 0.5
w = torch.zeros(Xt.shape[1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for epoch in range(5000):
    probs = torch.sigmoid(Xt @ w + b)
    eps = 1e-7
    bce = -(yt * torch.log(probs + eps) + (1 - yt) * torch.log(1 - probs + eps)).mean()
    l2 = (w ** 2).sum() / (2 * C * len(yt))   # sklearn's penalty, rescaled to the mean loss
    loss = bce + l2
    loss.backward()
    with torch.no_grad():                      # manual gradient descent step
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()

with torch.no_grad():
    pm = torch.sigmoid(Xv @ w + b).numpy()
print("manual val acc:", accuracy_score(y_val, pm > 0.5), "auc:", roc_auc_score(y_val, pm))
print("max prob diff vs sklearn:", abs(pm - p).max())

torch.save({"w": w.detach(), "b": b.detach()}, "models/torch_manual.pt")







#Model 3

import torch.nn as nn

torch.manual_seed(0)

class LogReg(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.linear = nn.Linear(n_in, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)

model = LogReg(Xt.shape[1])
nn.init.zeros_(model.linear.weight)
nn.init.zeros_(model.linear.bias)

criterio = nn.BCEWithLogitsLoss()
optim = torch.optim.SGD(
    [{"params": model.linear.weight, "weight_decay": 1 / (C * len(yt))},  # same L2 as before
     {"params": model.linear.bias, "weight_decay": 0.0}],                 # bias not penalized
    lr=0.5,
)

for epoch in range(5000):
    optim.zero_grad()
    loss = criterio(model(Xt), yt)
    loss.backward()
    optim.step()

model.eval()
with torch.no_grad():
    pn = torch.sigmoid(model(Xv)).numpy()
print("nn val acc:", accuracy_score(y_val, pn > 0.5), "auc:", roc_auc_score(y_val, pn))
print("max prob diff vs sklearn:", abs(pn - p).max(), "| vs manual:", abs(pn - pm).max())

torch.save(model.state_dict(), "models/torch_nn.pt")






import json

COSTE_FN, COSTE_FP = 3, 1   # missing a hit costs 3, giving screens to a flop costs 1

def coste(y_real, prob, umbral):
    pred = prob >= umbral
    fn = ((~pred) & (y_real == 1)).sum()
    fp = (pred & (y_real == 0)).sum()
    return COSTE_FN * fn + COSTE_FP * fp

def mejor_umbral(y_real, prob):
    umbrales = np.linspace(0.01, 0.99, 99)
    costes = [coste(y_real, prob, u) for u in umbrales]
    return float(umbrales[int(np.argmin(costes))])

Xtest_t = torch.tensor(Xte, dtype=torch.float32)
with torch.no_grad():
    prob_test = {
        "sklearn": sk.predict_proba(Xte)[:, 1],
        "manual": torch.sigmoid(Xtest_t @ w + b).numpy(),
        "nn": torch.sigmoid(model(Xtest_t)).numpy(),
    }
prob_val = {"sklearn": p, "manual": pm, "nn": pn}

yv, yte = y_val.values, y_test.values
umbrales = {}
for nombre in prob_val:
    u = mejor_umbral(yv, prob_val[nombre])      # threshold chosen on validation
    umbrales[nombre] = u
    print(nombre, "threshold", round(u, 2),
          "| val cost", coste(yv, prob_val[nombre], u),
          "| test cost", coste(yte, prob_test[nombre], u),
          "| test cost at 0.5:", coste(yte, prob_test[nombre], 0.5))
print("baseline test cost (always 'hit'):", coste(yte, np.ones(len(yte)), 0.5),
      "| (always 'flop'):", coste(yte, np.zeros(len(yte)), 0.5))

# Save what the app needs (it must load these, never retrain)
os.makedirs("artifacts", exist_ok=True)
for nombre, X_, d_ in (("train", X_train, train), ("val", X_val, val), ("test", X_test, test)):
    X_.to_csv(f"artifacts/X_{nombre}.csv", index=False)
    d_[["title", "release_date", "budget", "revenue", "success"]].to_csv(
        f"artifacts/meta_{nombre}.csv", index=False)
with open("artifacts/config.json", "w") as f:
    json.dump({"mediana_revenue": float(mediana), "coste_fn": COSTE_FN, "coste_fp": COSTE_FP,
               "umbrales": umbrales, "features": list(X_train.columns)}, f, indent=2)