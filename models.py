"""Linear regression, three ways, on one split."""

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

SEED = 42
EPOCHS = 3000
LR = 0.05


# ------------------------------------------------------------- 0. naive baseline

def fit_baseline(y_train):
    """Predict the training mean for every house. The bar to clear."""
    return float(y_train.mean())


# --------------------------------------------------------------- 1. scikit-learn

def fit_sklearn(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Return coefficients in a common format so the three are comparable.
    return np.concatenate([[model.intercept_], model.coef_]), model


# ------------------------------------------- 2. manual PyTorch (Session 5 style)

def fit_manual_torch(X_train, y_train, epochs=EPOCHS, lr=LR, seed=SEED):
    """Raw tensors, autograd for gradients, hand-written update step.

    No nn.Module, no optimizer, no loss class - everything explicit.
    """
    torch.manual_seed(seed)
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    n, d = X.shape

    # The parameters we are solving for. requires_grad=True tells autograd
    # to record every operation these take part in.
    w = torch.zeros((d, 1), dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    history = []
    for epoch in range(epochs):
        # forward pass: prediction and loss
        y_hat = X @ w + b
        loss = ((y_hat - y) ** 2).mean()

        # backward pass: autograd fills w.grad and b.grad
        loss.backward()

        # update step, written by hand. no_grad() stops the update itself
        # from being recorded as part of the graph.
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            # gradients accumulate by default, so clear them for next epoch
            w.grad.zero_()
            b.grad.zero_()

        if epoch % 50 == 0 or epoch == epochs - 1:
            history.append((epoch, float(loss)))

    params = np.concatenate([b.detach().numpy(),
                             w.detach().numpy().ravel()]).astype(np.float64)
    return params, history


# ------------------------------------------------- 3. standard PyTorch workflow

class LinearModel(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


def fit_standard_torch(X_train, y_train, epochs=EPOCHS, lr=LR, seed=SEED):
    """Same maths, but nn.Module holds the parameters and optim does the step."""
    torch.manual_seed(seed)
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    model = LinearModel(X.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()          # replaces the manual .grad.zero_()
        y_hat = model(X)               # replaces X @ w + b
        loss = criterion(y_hat, y)     # replaces ((y_hat - y) ** 2).mean()
        loss.backward()                # identical
        optimizer.step()               # replaces the manual w -= lr * w.grad

        if epoch % 50 == 0 or epoch == epochs - 1:
            history.append((epoch, float(loss)))

    w = model.linear.weight.detach().numpy().ravel()
    b = model.linear.bias.detach().numpy()
    params = np.concatenate([b, w]).astype(np.float64)
    return params, history, model


# ----------------------------------------------------------------- 4. evaluation

def predict(params, X):
    return params[0] + X @ params[1:]


def metrics(y_true_log, y_pred_log):
    """Report in log space (what we optimised) and dollars (what the bank sees)."""
    resid = y_pred_log - y_true_log
    rmse_log = float(np.sqrt((resid ** 2).mean()))
    mae_log = float(np.abs(resid).mean())
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y_true_log - y_true_log.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot

    true_d, pred_d = np.exp(y_true_log), np.exp(y_pred_log)
    ape = np.abs(pred_d - true_d) / true_d
    return dict(rmse_log=rmse_log, mae_log=mae_log, r2=r2,
                med_ape=float(np.median(ape) * 100),
                mean_ape=float(ape.mean() * 100),
                rmse_dollars=float(np.sqrt(((pred_d - true_d) ** 2).mean())),
                within_10pct=float((ape <= 0.10).mean() * 100))


# ------------------------------------- 5. L2-regularised variants (final config)

def fit_manual_l2(X_train, y_train, weight_decay, lr, epochs, seed):
    """Manual loop with the L2 penalty written into the loss by hand.

    The bias is deliberately NOT penalised, to match sklearn's Ridge.
    """
    torch.manual_seed(seed)
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    w = torch.zeros((X.shape[1], 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    history = []
    for epoch in range(epochs):
        data_loss = ((X @ w + b - y) ** 2).mean()
        loss = data_loss + (weight_decay / 2) * (w ** 2).sum()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
        if epoch % 50 == 0 or epoch == epochs - 1:
            history.append((epoch, float(data_loss.detach())))

    params = np.concatenate([b.detach().numpy(),
                             w.detach().numpy().ravel()]).astype(np.float64)
    return params, history


def fit_standard_l2(X_train, y_train, weight_decay, lr, epochs, seed):
    """Standard workflow. Parameter groups keep weight_decay off the bias:
    torch.optim.SGD would otherwise decay the intercept, which sklearn's
    Ridge never does - that mismatch produced an R2 of -3.3 before it was fixed.
    """
    torch.manual_seed(seed)
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    model = LinearModel(X.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD([
        {"params": model.linear.weight, "weight_decay": weight_decay},
        {"params": model.linear.bias, "weight_decay": 0.0},
    ], lr=lr)

    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            history.append((epoch, float(loss.detach())))

    params = np.concatenate([model.linear.bias.detach().numpy(),
                             model.linear.weight.detach().numpy().ravel()
                             ]).astype(np.float64)
    return params, history
