import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

# Load the dataset
df = pd.read_csv("data/online_shoppers_intention.csv")

# Drop PageValues: it is calculated from completed purchases, so it leaks the target
df = df.drop(columns=["PageValues"])

# Drop technical columns that barely change the purchase rate
df = df.drop(columns=["Browser", "OperatingSystems", "Region"])

# Merge rare traffic sources (fewer than 100 sessions) into one "other" group, coded 0
counts = df["TrafficType"].value_counts()
rare = counts[counts < 100].index
df.loc[df["TrafficType"].isin(rare), "TrafficType"] = 0

# Log-transform the page counts and durations: they have extreme values,
# and log1p (log of 1 + x) shrinks them while keeping 0 as 0
page_cols = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
]
df[page_cols] = np.log1p(df[page_cols])

# Flag sessions with a bounce rate of exactly 0: a large group that buys more often
df["NoBounce"] = (df["BounceRates"] == 0).astype(int)

# Turn the True/False columns into 1/0
df["Weekend"] = df["Weekend"].astype(int)
df["Revenue"] = df["Revenue"].astype(int)

# One-hot encode the text columns: each value becomes its own 0/1 column
df = pd.get_dummies(df, columns=["Month", "VisitorType", "TrafficType"], drop_first=True, dtype=int)

# Separate the inputs (X) from the target (y)
X = df.drop(columns=["Revenue"])
y = df["Revenue"]

# Stratified random split: 60% train, 20% validation, 20% test
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=42
)

# Standardize: learn each column's mean and spread from the training group only,
# then apply the same transformation to validation and test
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Model 1: scikit-learn logistic regression
# C=np.inf turns regularization off, so all three versions solve the same problem
sk_model = LogisticRegression(C=np.inf, max_iter=1000)
sk_model.fit(X_train_s, y_train)

# Predicted probability of buying for each validation session
sk_val_prob = sk_model.predict_proba(X_val_s)[:, 1]

print("Validation log loss:", log_loss(y_val, sk_val_prob))
print("Validation AUC:", roc_auc_score(y_val, sk_val_prob))

# Turn the data into PyTorch tensors (PyTorch's own kind of number table)
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32)

# Model 2: manual PyTorch loop
# Start with all weights and the bias at zero
w = torch.zeros(X_train_t.shape[1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)

learning_rate = 0.5
epochs = 5000

for epoch in range(epochs):
    # 1. Predict: weighted sum, then sigmoid to get probabilities
    z = X_train_t @ w + b
    p = torch.sigmoid(z)

    # 2. Measure: log loss
    loss = -(y_train_t * torch.log(p) + (1 - y_train_t) * torch.log(1 - p)).mean()

    # 3. Find the slope: PyTorch computes the gradient of the loss for w and b
    loss.backward()

    # 4. Step: move each weight a little downhill, then reset the gradients
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if epoch % 1000 == 0:
        print("epoch", epoch, "loss", loss.item())

# Predicted probabilities on the validation group
with torch.no_grad():
    manual_val_prob = torch.sigmoid(X_val_t @ w + b).numpy()

print("Manual validation log loss:", log_loss(y_val, manual_val_prob))

# Model 3: standard PyTorch workflow (nn.Module + torch.optim)
torch.manual_seed(42)


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        # One linear layer: holds the weights and the bias
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        # Returns the weighted sum (before the sigmoid)
        return self.linear(x).squeeze(1)


nn_model = LogisticRegressionModel(X_train_t.shape[1])
loss_fn = torch.nn.BCEWithLogitsLoss()  # sigmoid + log loss in one
optimizer = torch.optim.SGD(nn_model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    optimizer.zero_grad()            # reset the gradients
    z = nn_model(X_train_t)          # 1. predict
    loss = loss_fn(z, y_train_t)     # 2. measure
    loss.backward()                  # 3. find the slope
    optimizer.step()                 # 4. step downhill

# Predicted probabilities on the validation group
with torch.no_grad():
    nn_val_prob = torch.sigmoid(nn_model(X_val_t)).numpy()

# Business costs (assumptions stated in the report)
PRICE = 50          # price of the product
DISCOUNT = 0.20     # discount shown to visitors predicted not to buy
CONVERSION = 0.10   # share of leaving visitors that the discount would convince
COST_MISSED_SALE = PRICE * CONVERSION      # $5: predicted "will buy", but the visitor left
COST_WASTED_DISCOUNT = PRICE * DISCOUNT    # $10: predicted "won't buy", but would have bought anyway


def total_cost(y_true, prob, threshold):
    y_true = np.asarray(y_true)
    pred_buy = prob >= threshold                              # True = predicted to buy, no discount
    missed_sales = np.sum(pred_buy & (y_true == 0))           # predicted buy, did not buy
    wasted_discounts = np.sum(~pred_buy & (y_true == 1))      # predicted no buy, did buy
    return missed_sales * COST_MISSED_SALE + wasted_discounts * COST_WASTED_DISCOUNT


# Naive baselines: no model at all
print("Cost, discount for everyone:", total_cost(y_val, np.zeros(len(y_val)), 0.5))
print("Cost, discount for no one:", total_cost(y_val, np.ones(len(y_val)), 0.5))

# Try every threshold from 0.01 to 0.99 on the validation group and keep the cheapest
thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
costs = [total_cost(y_val, sk_val_prob, t) for t in thresholds]
best_threshold = thresholds[int(np.argmin(costs))]
print("Best threshold:", best_threshold, "with cost:", min(costs))

# Final evaluation on the test group, using the threshold chosen on validation
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

sk_test_prob = sk_model.predict_proba(X_test_s)[:, 1]
with torch.no_grad():
    manual_test_prob = torch.sigmoid(X_test_t @ w + b).numpy()
    nn_test_prob = torch.sigmoid(nn_model(X_test_t)).numpy()

# Naive baseline: every visitor gets the average purchase rate from training.
# That is below the threshold, so it means "discount for everyone".
baseline_test_prob = np.full(len(y_test), y_train.mean())

results = []
for name, prob in [
    ("Naive baseline", baseline_test_prob),
    ("scikit-learn", sk_test_prob),
    ("Manual PyTorch loop", manual_test_prob),
    ("nn.Module + torch.optim", nn_test_prob),
]:
    results.append({
        "Model": name,
        "Log loss": round(log_loss(y_test, prob), 4),
        "AUC": round(roc_auc_score(y_test, prob), 4),
        "Total cost ($)": total_cost(y_test, prob, best_threshold),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save everything the dashboard needs, so it never has to retrain
out = Path("artifacts")
out.mkdir(exist_ok=True)

# The three trained models and the scaler
joblib.dump(sk_model, out / "sk_model.joblib")
joblib.dump(scaler, out / "scaler.joblib")
torch.save({"w": w.detach(), "b": b.detach()}, out / "manual_model.pt")
torch.save(nn_model.state_dict(), out / "nn_model.pt")

# The prepared test group and the results table
np.save(out / "X_test.npy", X_test_s)
np.save(out / "y_test.npy", y_test.to_numpy())
results_df.to_csv(out / "results.csv", index=False)

# The threshold and the cost assumptions
settings = {
    "best_threshold": float(best_threshold),
    "price": PRICE,
    "discount": DISCOUNT,
    "conversion": CONVERSION,
    "feature_names": X.columns.tolist(),
    "train_purchase_rate": float(y_train.mean()),
}
with open(out / "settings.json", "w") as f:
    json.dump(settings, f, indent=2)

print("Saved artifacts to", out)