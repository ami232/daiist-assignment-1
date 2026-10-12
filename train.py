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

import json, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

DATA    = "data"
OUTDIR  = "models"
SEED    = 42
LR      = 0.05
EPOCHS  = 3000
ALPHA   = 1.0     
G       = 9.81  # this is a constant, not a hyperparameter i set it to 9.81 m/s² (this constant i pulled using Claude since i don't know much about physics)
torch.manual_seed(SEED)
np.random.seed(SEED)

#1. Load 
print("Loading data...")
radar = pd.read_parquet(f"{DATA}/radar_marks_clean_3d_mb.parquet")
full  = pd.read_parquet(f"{DATA}/full_trajectories_3d_mb.parquet").sort_values(
    ["trajectory_id", "t"]
)

# 2. Targets 
landing = (
    full.groupby("trajectory_id")[["x", "y", "t"]]
    .last()
    .rename(columns={"x": "x_end", "y": "y_end", "t": "t_land"})
)
last_ping_t = (
    radar.sort_values(["trajectory_id", "t"])
    .groupby("trajectory_id")["t"].last()
    .rename("t_last_ping")
)
targets = landing.join(last_ping_t)
targets["t_remaining"] = targets["t_land"] - targets["t_last_ping"]
targets["is_rocket"]   = full.groupby("trajectory_id")["is_rocket"].first()

# 3. Feature engineering / This was the most challenging part of the assignment for me. I had to think about what features would be useful for predicting the landing position and time of the rocket. I decided to use the raw radar measurements as well as some derived features based on physics principles.
BASE_FEATS = ["x", "y", "z", "vx", "vy", "vz", "v"]

radar_s = radar.sort_values(["trajectory_id", "t"]).copy()
radar_s["ping_idx"] = radar_s.groupby("trajectory_id").cumcount() #here we are creating a new column called ping_idx that counts the number of pings for each trajectory

# Raw pivot: 7 pings × 7 vars = 49 features. Now that we have the trajectory_id and ping_idx, we can pivot the radar data to create a wide format dataframe where each row corresponds to a trajectory and each column corresponds to a feature at a specific ping index. This will give us 49 features (7 pings × 7 variables) for each trajectory.
wide = radar_s.pivot(index="trajectory_id", columns="ping_idx", values=BASE_FEATS)
wide.columns = [f"{feat}_p{i}" for feat, i in wide.columns]
wide = wide.sort_index()       

# Physics-derived features (the interesting engineering)
grp = radar_s.groupby("trajectory_id")

# a) Speed trend — positive = rocket accelerating, negative = ballistic decelerating. These were the features where i used claude and research to make sure my ft enginnering was correct.
v_vals    = radar_s.pivot(index="trajectory_id", columns="ping_idx", values="v")
v_slope   = v_vals.apply(lambda row: np.polyfit(range(7), row, 1)[0], axis=1)

# b) Horizontal speed at last ping
wide["h_speed_last"] = np.hypot(wide["vx_p6"], wide["vy_p6"])

# c) Horizontal range covered during radar window
wide["h_range_obs"] = np.hypot(
    wide["x_p6"] - wide["x_p0"],
    wide["y_p6"] - wide["y_p0"]
)

# d) Estimated apogee altitude (kinematic upper bound from last ping)
vz6 = wide["vz_p6"]
z6  = wide["z_p6"]
wide["est_apogee_km"] = np.where(
    vz6 > 0,
    (z6 + vz6 ** 2 / (2 * G)) / 1000,
    z6 / 1000
)

# e) Speed trend (engineered from all 7 pings)
wide["v_slope"] = v_slope.values

# f) Bearing at last ping (direction of travel)
wide["bearing_rad"] = np.arctan2(wide["vy_p6"], wide["vx_p6"])

print(f"Feature matrix: {wide.shape[1]} features  ({7*7} raw + {wide.shape[1]-49} engineered)")

# Align targets
targets = targets.loc[wide.index]
traj_ids = np.array(wide.index)

X = wide.values.astype(np.float32)
y = targets[["x_end", "y_end", "t_remaining"]].values.astype(np.float32)

# 4. Train / test split 
# Random split is valid: each trajectory is independent (no temporal leakage
# between trajectories).  We split on trajectory level (one row = one trajectory).
X_tr, X_te, y_tr, y_te, ids_tr, ids_te = train_test_split(
    X, y, traj_ids, test_size=0.2, random_state=SEED
)
print(f"Train: {len(X_tr)}   Test: {len(X_te)}")

# 5. Scaling 
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_tr_s = scaler_X.fit_transform(X_tr)
X_te_s = scaler_X.transform(X_te)
y_tr_s = scaler_y.fit_transform(y_tr)
 #this is just the scaler
# 6. Naive baseline — predict training-set mean 
y_baseline_mean = y_tr.mean(axis=0)
y_pred_baseline = np.tile(y_baseline_mean, (len(X_te), 1))
                         #This is just the mean of the training targets for each of the three outputs. Just for a naive solution kind of like flipping a coin for a binary classification problem

# 7. sklearn Ridge 
print("Training sklearn Ridge...")
sk_model = Ridge(alpha=ALPHA)
sk_model.fit(X_tr_s, y_tr)                                  # fit on original y scale
y_pred_sk = sk_model.predict(X_te_s)

# 8. PyTorch from-scratch
print("Training from-scratch (SGD)...")
X_pt  = torch.tensor(X_tr_s, dtype=torch.float32)
y_pt  = torch.tensor(y_tr_s, dtype=torch.float32)
X_tept = torch.tensor(X_te_s, dtype=torch.float32)
n_feat, n_out = X_pt.shape[1], y_pt.shape[1]

W = torch.zeros(n_feat, n_out, requires_grad=True)
b = torch.zeros(n_out,         requires_grad=True)
losses_scratch = []

for epoch in range(EPOCHS):
    y_hat = X_pt @ W + b
    loss  = ((y_hat - y_pt) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        W -= LR * W.grad
        b -= LR * b.grad
    W.grad.zero_()
    b.grad.zero_()
    losses_scratch.append(loss.item())
    if (epoch + 1) % 500 == 0:
        print(f"  scratch epoch {epoch+1}/{EPOCHS}  loss={loss.item():.6f}")

with torch.no_grad():
    y_pred_scratch = scaler_y.inverse_transform((X_tept @ W + b).numpy())

# 9. PyTorch nn.Module + Adam
print("Training nn.Module (Adam)...")

class LinearRegressor(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    def forward(self, x):
        return self.linear(x)

nn_model  = LinearRegressor(n_feat, n_out)
optimizer = torch.optim.Adam(nn_model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.MSELoss()
losses_nn = []

for epoch in range(EPOCHS):
    nn_model.train()
    optimizer.zero_grad()
    loss = criterion(nn_model(X_pt), y_pt)
    loss.backward()
    optimizer.step()
    losses_nn.append(loss.item())
    if (epoch + 1) % 500 == 0:
        print(f"  nn epoch {epoch+1}/{EPOCHS}  loss={loss.item():.6f}")

nn_model.eval()
with torch.no_grad():
    y_pred_nn = scaler_y.inverse_transform(nn_model(X_tept).numpy())

# 10. Metrics 
def metrics(y_pred, y_true, name):
    rmse_x  = float(np.sqrt(np.mean((y_pred[:, 0] - y_true[:, 0]) ** 2)))
    rmse_y  = float(np.sqrt(np.mean((y_pred[:, 1] - y_true[:, 1]) ** 2)))
    rmse_2d = float(np.sqrt(np.mean(np.sum((y_pred[:,:2] - y_true[:,:2])**2, axis=1))))
    rmse_t  = float(np.sqrt(np.mean((y_pred[:, 2] - y_true[:, 2]) ** 2)))
    r2_x    = float(r2_score(y_true[:, 0], y_pred[:, 0]))
    r2_y    = float(r2_score(y_true[:, 1], y_pred[:, 1]))
    r2_t    = float(r2_score(y_true[:, 2], y_pred[:, 2]))
    print(f"\n[{name}]  RMSE 2D={rmse_2d:.0f}m  R²x={r2_x:.4f}  R²y={r2_y:.4f}  R²t={r2_t:.4f}")
    return dict(name=name, rmse_x=rmse_x, rmse_y=rmse_y, rmse_2d=rmse_2d,
                rmse_t=rmse_t, r2_x=r2_x, r2_y=r2_y, r2_t=r2_t)

all_metrics = [
    metrics(y_pred_baseline, y_te, "Baseline (mean)"),
    metrics(y_pred_sk,       y_te, "sklearn Ridge"),
    metrics(y_pred_scratch,  y_te, "PyTorch Scratch (SGD)"),
    metrics(y_pred_nn,       y_te, "PyTorch nn.Module (Adam)"),
]

# 11. Save artifacts
import os
os.makedirs(OUTDIR, exist_ok=True)

# Scalers
joblib.dump(scaler_X, f"{OUTDIR}/scaler_X.pkl")
joblib.dump(scaler_y, f"{OUTDIR}/scaler_y.pkl")

# Baseline
np.save(f"{OUTDIR}/baseline_mean.npy", y_baseline_mean)

# sklearn
joblib.dump(sk_model, f"{OUTDIR}/sklearn_ridge.pkl")

# PyTorch scratch — save weights as numpy (no torch dependency in app)
np.save(f"{OUTDIR}/scratch_W.npy", W.detach().numpy())
np.save(f"{OUTDIR}/scratch_b.npy", b.detach().numpy())

# PyTorch nn.Module — save state dict + architecture params
torch.save(nn_model.state_dict(), f"{OUTDIR}/nn_model_state.pt")
with open(f"{OUTDIR}/nn_arch.json", "w") as f:
    json.dump({"in_features": n_feat, "out_features": n_out}, f)

# Test split (for Gradio prediction-vs-actual plot)
np.savez(f"{OUTDIR}/test_split.npz",
         X_te_s=X_te_s, y_te=y_te, ids_te=ids_te,
         y_pred_baseline=y_pred_baseline,
         y_pred_sk=y_pred_sk,
         y_pred_scratch=y_pred_scratch,
         y_pred_nn=y_pred_nn)

# Training curves
np.savez(f"{OUTDIR}/train_curves.npz",
         losses_scratch=losses_scratch, losses_nn=losses_nn)

# Feature names + column list
feature_names = list(wide.columns)
with open(f"{OUTDIR}/feature_names.json", "w") as f:
    json.dump(feature_names, f)

# Metrics
with open(f"{OUTDIR}/metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

# Target distribution
targets.to_parquet(f"{OUTDIR}/targets.parquet")

# Feature stats
wide_df = pd.DataFrame(wide.values, columns=feature_names, index=traj_ids)
wide_df.to_parquet(f"{OUTDIR}/features.parquet")

# Downsampled trajectory display (every 10th point, ~43k rows, ~2MB)
display = full.iloc[::10].copy()
display.to_parquet(f"{DATA}/display_trajectories.parquet", index=False)

print("\nAll artifacts saved to models/")
print("  scaler_X.pkl, scaler_y.pkl")
print("  baseline_mean.npy")
print("  sklearn_ridge.pkl")
print("  scratch_W.npy, scratch_b.npy")
print("  nn_model_state.pt, nn_arch.json")
print("  test_split.npz, train_curves.npz")
print("  metrics.json, feature_names.json")
print("  targets.parquet, features.parquet")
print(f"  data/display_trajectories.parquet")

