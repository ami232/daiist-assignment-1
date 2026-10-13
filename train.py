"""
Assignment 1 — training pipeline.

You're using a script for this stage. If you'd rather use a notebook,
write train.ipynb instead and delete this file — main.py refuses to run if
it finds both train.py and train.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via `uv run python main.py train`, this file must, start to finish, with no
manual steps in between:

- Load your committed dataset.
- Apply the feature engineering, preprocessing, and train/val/test split
  consistent with the business framing you wrote in REPORT.md.
- Train the same model — linear regression for a regression target,
  logistic regression for a binary one — three ways, on the same split:
  scikit-learn, a from-scratch PyTorch loop (Session 5's manual approach:
  raw tensors, autograd, a manual gradient-step update), and the standard
  torch.nn.Module + torch.optim workflow.
- Evaluate all three against each other and against a naive baseline.
- Save whatever app.py needs to build its dashboard without retraining
  anything.

How you structure the code beyond that — file layout, function
boundaries, naming — is your call to make and be able to defend.
"""


import numpy as np
import pandas as pd

# 1. Cargar el dataset
df = pd.read_csv("data/bike_sharing_hour.csv")

print("Filas y columnas:", df.shape)
print(df.head())
print(df.dtypes)
print(df.isnull().sum())

# 2. Convertir la fecha a tipo datetime real (para poder ordenar cronológicamente)
df["dteday"] = pd.to_datetime(df["dteday"])

# 3. Feature engineering

# Es fin de semana (domingo=0, sábado=6, según la codificación del dataset)
df["is_weekend"] = df["weekday"].isin([0, 6]).astype(int)

# Hora punta: gente yendo/volviendo del trabajo en día laborable
df["is_rush_hour"] = (
    df["workingday"].eq(1) & df["hr"].isin([7, 8, 9, 17, 18, 19])
).astype(int)

# Hora codificada de forma cíclica (23h y 0h están "cerca" en realidad,
# pero numéricamente parecen muy lejos; esto lo arregla)
df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)

# Interacción clima-contexto: mal tiempo en día laborable afecta distinto
# que mal tiempo en fin de semana
df["bad_weather_workday"] = (
    (df["weathersit"] >= 3) & (df["workingday"] == 1)
).astype(int)

print(df[["dteday", "is_weekend", "is_rush_hour", "hr_sin", "hr_cos", "bad_weather_workday"]].head(10))



# 4. Split temporal: ordenamos por fecha y partimos cronológicamente,
# no aleatoriamente, porque en producción nunca tendríamos datos "futuros"
# para entrenar un modelo que predice el presente.
df = df.sort_values("dteday").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

print("Train:", train_df.shape, train_df["dteday"].min(), "->", train_df["dteday"].max())
print("Val:  ", val_df.shape, val_df["dteday"].min(), "->", val_df["dteday"].max())
print("Test: ", test_df.shape, test_df["dteday"].min(), "->", test_df["dteday"].max())


# 5. Seleccionar las features finales
# Excluimos: instant (solo un índice), dteday (ya extrajimos lo útil),
# casual y registered (son los componentes de cnt, usarlas sería trampa)
feature_cols = [
    "season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed",
    "is_weekend", "is_rush_hour", "hr_sin", "hr_cos", "bad_weather_workday",
]
target_col = "cnt"

X_train = train_df[feature_cols].values
y_train = train_df[target_col].values

X_val = val_df[feature_cols].values
y_val = val_df[target_col].values

X_test = test_df[feature_cols].values
y_test = test_df[target_col].values

print("X_train:", X_train.shape, "y_train:", y_train.shape)

# 6. Escalar features: calculamos media/desviación SOLO con train,
# y aplicamos esa misma transformación a val y test (nunca al revés,
# o estaríamos filtrando información del futuro al escalado)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Media de X_train_scaled (debería ser ~0):", X_train_scaled.mean(axis=0).round(2))
print("Desviación de X_train_scaled (debería ser ~1):", X_train_scaled.std(axis=0).round(2))



# 7. MODELO 1: scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sklearn_model = LinearRegression()
sklearn_model.fit(X_train_scaled, y_train)

y_pred_sklearn_val = sklearn_model.predict(X_val_scaled)

mae_sklearn = mean_absolute_error(y_val, y_pred_sklearn_val)
rmse_sklearn = mean_squared_error(y_val, y_pred_sklearn_val) ** 0.5
r2_sklearn = r2_score(y_val, y_pred_sklearn_val)

print("\n--- Scikit-learn (validación) ---")
print(f"MAE:  {mae_sklearn:.2f}")
print(f"RMSE: {rmse_sklearn:.2f}")
print(f"R2:   {r2_sklearn:.3f}")

# 8. Baseline ingenuo: predecir siempre la media de train
baseline_pred = [y_train.mean()] * len(y_val)
mae_baseline = mean_absolute_error(y_val, baseline_pred)
print("\n--- Baseline (media de train, validación) ---")
print(f"MAE:  {mae_baseline:.2f}")



# 9. MODELO 2: PyTorch manual (sin nn.Module, sin optim)
import torch

torch.manual_seed(42)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

n_features = X_train_t.shape[1]

w = torch.zeros((n_features, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.1
n_epochs = 500

for epoch in range(n_epochs):
    y_pred = X_train_t @ w + b
    loss = ((y_pred - y_train_t) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if epoch % 100 == 0:
        current_loss = loss.item()
        print("Epoch", epoch, "loss =", round(current_loss, 2))

with torch.no_grad():
    y_pred_manual_val = (X_val_t @ w + b).numpy().flatten()

mae_manual = mean_absolute_error(y_val, y_pred_manual_val)
rmse_manual = mean_squared_error(y_val, y_pred_manual_val) ** 0.5
r2_manual = r2_score(y_val, y_pred_manual_val)

print("--- PyTorch manual (validacion) ---")
print("MAE: ", round(mae_manual, 2))
print("RMSE:", round(rmse_manual, 2))
print("R2:  ", round(r2_manual, 3))

# 10. MODELO 3: PyTorch estandar (nn.Module + torch.optim)
import torch.nn as nn

torch.manual_seed(42)

standard_model = nn.Linear(n_features, 1)
optimizer = torch.optim.SGD(standard_model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

for epoch in range(n_epochs):
    optimizer.zero_grad()
    y_pred = standard_model(X_train_t)
    loss = loss_fn(y_pred, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        current_loss = loss.item()
        print("Epoch", epoch, "loss =", round(current_loss, 2))

with torch.no_grad():
    y_pred_standard_val = standard_model(X_val_t).numpy().flatten()

mae_standard = mean_absolute_error(y_val, y_pred_standard_val)
rmse_standard = mean_squared_error(y_val, y_pred_standard_val) ** 0.5
r2_standard = r2_score(y_val, y_pred_standard_val)

print("--- PyTorch estandar (validacion) ---")
print("MAE: ", round(mae_standard, 2))
print("RMSE:", round(rmse_standard, 2))
print("R2:  ", round(r2_standard, 3))


# 11. Guardar todo lo que necesitará app.py, sin reentrenar nada alli
import os
import joblib

os.makedirs("models", exist_ok=True)

# Guardamos el scaler (necesario para transformar nuevos datos igual que en train)
joblib.dump(scaler, "models/scaler.pkl")

# Guardamos el modelo de scikit-learn
joblib.dump(sklearn_model, "models/sklearn_model.pkl")

# Guardamos los pesos del modelo manual (w y b, como tensores planos)
torch.save({"w": w.detach(), "b": b.detach()}, "models/manual_model.pt")

# Guardamos el modelo estandar completo (su state_dict)
torch.save(standard_model.state_dict(), "models/standard_model.pt")

# Guardamos tambien las columnas de features y datos de test, para el dashboard
import json
with open("models/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

test_df.to_csv("models/test_data.csv", index=False)

print("Modelos y artefactos guardados en la carpeta 'models/'.")



# 12. Evaluacion final en TEST (el conjunto que nunca toco el entrenamiento)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

y_pred_sklearn_test = sklearn_model.predict(X_test_scaled)

with torch.no_grad():
    y_pred_manual_test = (X_test_t @ w + b).numpy().flatten()
    y_pred_standard_test = standard_model(X_test_t).numpy().flatten()

baseline_pred_test = [y_train.mean()] * len(y_test)

print("\n========== RESULTADOS FINALES EN TEST ==========")
for nombre, y_pred in [
    ("Scikit-learn", y_pred_sklearn_test),
    ("PyTorch manual", y_pred_manual_test),
    ("PyTorch estandar", y_pred_standard_test),
    ("Baseline (media)", baseline_pred_test),
]:
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(nombre, "-> MAE:", round(mae, 2), "RMSE:", round(rmse, 2), "R2:", round(r2, 3))


# 13. Tuning ligero: regularizacion (Ridge) y learning rate, SOLO en validacion
# (el test set no se toca hasta tener el modelo final elegido)
from sklearn.linear_model import Ridge

print("\n========== TUNING: regularizacion (Ridge, sklearn) ==========")
for alpha in [0.0, 1.0, 10.0, 50.0]:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train_scaled, y_train)
    pred_val = ridge_model.predict(X_val_scaled)
    mae_val = mean_absolute_error(y_val, pred_val)
    print("alpha =", alpha, "-> MAE val:", round(mae_val, 2))

print("\n========== TUNING: learning rate (PyTorch manual) ==========")
for lr_try in [0.01, 0.1, 0.5]:
    w_try = torch.zeros((n_features, 1), requires_grad=True)
    b_try = torch.zeros(1, requires_grad=True)
    for epoch in range(n_epochs):
        y_pred = X_train_t @ w_try + b_try
        loss = ((y_pred - y_train_t) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w_try -= lr_try * w_try.grad
            b_try -= lr_try * b_try.grad
            w_try.grad.zero_()
            b_try.grad.zero_()

    with torch.no_grad():
        pred_val = (X_val_t @ w_try + b_try).numpy().flatten()

    if np.isnan(pred_val).any() or np.isinf(pred_val).any():
        print("lr =", lr_try, "-> DIVERGIO (pesos se fueron a NaN/infinito)")
    else:
        mae_val = mean_absolute_error(y_val, pred_val)
        print("lr =", lr_try, "-> MAE val:", round(mae_val, 2))

