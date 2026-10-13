import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure output directory exists for saved models
os.makedirs("models", exist_ok=True)

# Set seed for reproducibility across all three methods
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. LOAD DATASET & PREPROCESSING
# ==========================================
df = pd.read_csv("data/telco_churn.csv")

# Clean 'TotalCharges' (convert empty strings/spaces to median)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Drop non-predictive columns
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# Simple Feature Engineering
df['ChargesPerMonthTenure'] = df['TotalCharges'] / (df['tenure'] + 1)

# Encode Target (Yes/No -> 1/0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode Categorical Features
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=['Churn'])
y = df['Churn']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (fit strictly on training data to prevent leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert arrays to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)


# ==========================================
# 2. NAIVE BASELINE
# ==========================================
# Always predicts the most common class (0: No Churn)
majority_class = y_train.mode()[0]
y_pred_naive = np.full_like(y_test, fill_value=majority_class)
naive_acc = accuracy_score(y_test, y_pred_naive)


# ==========================================
# 3. METHOD 1: SCIKIT-LEARN
# ==========================================
sklearn_model = LogisticRegression(random_state=42)
sklearn_model.fit(X_train_scaled, y_train)
sklearn_preds = sklearn_model.predict(X_test_scaled)
sklearn_probs = sklearn_model.predict_proba(X_test_scaled)[:, 1]


# ==========================================
# 4. METHOD 2: MANUAL PYTORCH LOOP (Session 5)
# ==========================================
n_samples, n_features = X_train_tensor.shape
# Weights and bias initialized manually with autograd
w_manual = torch.randn(n_features, 1, requires_grad=True)
b_manual = torch.zeros(1, requires_grad=True)

learning_rate = 0.1
epochs = 500

for epoch in range(epochs):
    # Forward pass: Sigmoid(X * w + b)
    linear_out = torch.matmul(X_train_tensor, w_manual) + b_manual
    predictions = torch.sigmoid(linear_out)
    
    # Binary Cross-Entropy Loss manually
    loss = -torch.mean(y_train_tensor * torch.log(predictions + 1e-8) + 
                       (1 - y_train_tensor) * torch.log(1 - predictions + 1e-8))
    
    # Backward pass
    loss.backward()
    
    # Manual Gradient Step Update
    with torch.no_grad():
        w_manual -= learning_rate * w_manual.grad
        b_manual -= learning_rate * b_manual.grad
        w_manual.grad.zero_()
        b_manual.grad.zero_()

# Evaluation for Manual PyTorch
with torch.no_grad():
    manual_probs = torch.sigmoid(torch.matmul(X_test_tensor, w_manual) + b_manual).numpy().flatten()
    manual_preds = (manual_probs >= 0.5).astype(int)


# ==========================================
# 5. METHOD 3: STANDARD PYTORCH MODULE
# ==========================================
class LogisticRegressionModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

std_model = LogisticRegressionModule(n_features)
criterion = nn.BCELoss()
optimizer = optim.SGD(std_model.parameters(), lr=0.1)

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = std_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Evaluation for Standard PyTorch
std_model.eval()
with torch.no_grad():
    std_probs = std_model(X_test_tensor).numpy().flatten()
    std_preds = (std_probs >= 0.5).astype(int)


# ==========================================
# 6. EVALUATION & SAVE ARTIFACTS
# ==========================================
def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }

print("\n--- MODEL COMPARISON ---")
print(f"Naive Baseline Accuracy: {naive_acc:.4f}")
print("Scikit-Learn:", get_metrics(y_test, sklearn_preds))
print("Manual PyTorch:", get_metrics(y_test, manual_preds))
print("Standard PyTorch:", get_metrics(y_test, std_preds))

# Save models & artifacts for app.py
joblib.dump(sklearn_model, "models/sklearn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
torch.save({'w': w_manual, 'b': b_manual}, "models/manual_pytorch.pt")
torch.save(std_model.state_dict(), "models/std_pytorch.pt")

# Save test set and predictions for Gradio visualization
test_artifacts = {
    'X_test': X_test,
    'y_test': y_test,
    'sklearn_probs': sklearn_probs,
    'manual_probs': manual_probs,
    'std_probs': std_probs
}
joblib.dump(test_artifacts, "models/test_artifacts.pkl")

print("\nAll models and artifacts saved successfully in 'models/' folder!")