"""Why do sklearn and the PyTorch versions disagree? Run AFTER main.py train."""
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from train import (LR, build_preprocessor, engineer_features, fit_manual,
                   load_and_clean, split_data)

C = json.load(open("artifacts/metrics.json"))["C"]

train_df, _, _ = split_data(load_and_clean())
prep = build_preprocessor()
X = prep.fit_transform(engineer_features(train_df)).astype(np.float64)
y = train_df["Churn"].to_numpy()
n, d = X.shape
lam = 1.0 / (C * n)
print(f"C={C}  lam={lam:.3e}  columns={d}")


def objective_and_grad(w, b):
    Xt = torch.tensor(X)
    yt = torch.tensor(y, dtype=torch.float64)
    w = torch.tensor(np.asarray(w), dtype=torch.float64, requires_grad=True)
    b = torch.tensor(np.asarray(b), dtype=torch.float64, requires_grad=True)
    z = Xt @ w + b
    J = (F.softplus(z) - yt * z).mean() + 0.5 * lam * (w ** 2).sum()
    J.backward()
    return J.item(), torch.cat([w.grad, b.grad]).norm().item()


sk = LogisticRegression(C=C, max_iter=10000, tol=1e-10).fit(X, y)
J, g = objective_and_grad(sk.coef_.ravel(), sk.intercept_)
print(f"sklearn          objective {J:.8f}  grad norm {g:.2e}")

# Unweighted design matrix: is anything exactly collinear?
A = np.hstack([X, np.ones((n, 1))])
ev_design = np.linalg.eigvalsh(A.T @ A / n)
print(f"design matrix eigenvalues: min {ev_design.min():.2e}, max {ev_design.max():.2e}")

# Curvature (Hessian) of the regularised objective at the solution: this sets
# gradient descent's speed. Error shrinks like exp(-lr * lambda_min * steps).
p = sk.predict_proba(X)[:, 1]
H = A.T @ (A * (p * (1 - p))[:, None]) / n
H[:d, :d] += lam * np.eye(d)  # L2 penalty on weights only, not the intercept
ev = np.linalg.eigvalsh(H)
print(f"Hessian eigenvalues: min {ev.min():.2e}, max {ev.max():.2e}, "
      f"condition number {ev.max() / ev.min():.2e}")
print(f"stable step size? lr*lambda_max = {LR * ev.max():.2f} (needs < 2)")
print(f"predicted epochs to shrink the error 1e6x: {np.log(1e6) / (LR * ev.min()):.0f}")

for epochs in [50000, 300000]:
    w, b = fit_manual(X, y, lam, epochs=epochs)
    J, g = objective_and_grad(w, b)
    gap = np.abs(w - sk.coef_.ravel()).max()
    print(f"manual {epochs:>7} ep  objective {J:.8f}  grad norm {g:.2e}  max|w-w_sk| {gap:.2e}")