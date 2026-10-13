import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load saved test artifacts
test_artifacts = joblib.load("models/test_artifacts.pkl")
y_test = test_artifacts['y_test']

probs_dict = {
    "Scikit-Learn": test_artifacts['sklearn_probs'],
    "Manual PyTorch": test_artifacts['manual_probs'],
    "Standard PyTorch": test_artifacts['std_probs']
}

threshold = 0.5
baseline_preds = np.zeros_like(y_test)
results = []

# Naive Baseline metrics
results.append({
    "Model": "Naive Baseline",
    "Accuracy": accuracy_score(y_test, baseline_preds),
    "Precision": precision_score(y_test, baseline_preds, zero_division=0),
    "Recall": recall_score(y_test, baseline_preds, zero_division=0),
    "F1-Score": f1_score(y_test, baseline_preds, zero_division=0)
})

# Model metrics
for name, probs in probs_dict.items():
    preds = (probs >= threshold).astype(int)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1-Score": f1_score(y_test, preds, zero_division=0)
    })

# Print markdown table
df_results = pd.DataFrame(results)
print("\n--- COPY THE TABLE BELOW INTO YOUR REPORT.MD ---\n")
print(df_results.to_markdown(index=False))
print("\n------------------------------------------------\n")