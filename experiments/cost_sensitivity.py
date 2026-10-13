"""
How sensitive is the decision to the assumed cost ratio (common.py: COST_FN : COST_FP)?

Run (after `uv run python main.py train`):
    uv run python experiments/cost_sensitivity.py

For each ratio it picks the threshold that minimises cost on the VALIDATION
split with the saved scikit-learn model, and compares against the cheapest
constant policy (approve everyone or reject everyone). Test data is not used.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

import common as cm
from features import TARGET

data = pd.read_csv(cm.ARTIFACTS / "dataset_engineered.csv.gz")
val = data[data["split"] == "val"]
pre = joblib.load(cm.ARTIFACTS / "preprocessor.joblib")
model = joblib.load(cm.ARTIFACTS / "sklearn_logreg.joblib")

y = val[TARGET].to_numpy().astype(int)
p = model.predict_proba(pre.transform(val.drop(columns=[TARGET, "split"])))[:, 1]
base_rate = y.mean()

print(f"default rate on validation: {base_rate:.3f}")
print(f"{'FN:FP':>6}{'break-even':>12}{'best thr':>10}{'model cost':>12}{'approve-all':>13}"
      f"{'reject-all':>12}{'cheapest const.':>17}{'saving':>9}")
for ratio in [1, 2, 3, 5, 10]:
    thr = cm.best_threshold(y, p, cost_fn=ratio, cost_fp=1.0)
    model_cost = cm.total_cost(y, (p >= thr).astype(int), ratio, 1.0) / len(y)
    approve_all, reject_all = base_rate * ratio, 1 - base_rate
    cheapest = "approve-all" if approve_all <= reject_all else "reject-all"
    best_const = min(approve_all, reject_all)
    print(f"{ratio:>4}:1{1 / (1 + ratio):>12.3f}{thr:>10.2f}{model_cost:>12.4f}{approve_all:>13.4f}"
          f"{reject_all:>12.4f}{cheapest:>17}{(1 - model_cost / best_const) * 100:>8.1f}%")
