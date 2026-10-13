"""
Assignment 1: F1 Pit Stop Prediction

Business framing: Predict whether a car will pit on the next lap.
F1 strategy engineers use this to anticipate rival undercuts and decide
whether to bring their own car in to defend track position.

Target: PitNextLap (binary)
Model: logistic regression.
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)
from pathlib import Path

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# 1. LOAD
print("Loading data...")
df = pd.read_csv(ROOT / "train (2).csv")
print(f"  {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Target: {df['PitNextLap'].value_counts().to_dict()}  (positive rate: {df['PitNextLap'].mean():.3%})")


# 2. FEATURE ENGINEERING
# Softer compounds degrade faster so we assign a score.
COMPOUND_SOFTNESS = {'WET': 0, 'INTERMEDIATE': 1, 'HARD': 2, 'MEDIUM': 3, 'SOFT': 4}
#expected stint length, the number of laps, by compound.
COMPOUND_LIFE = {'WET': 20, 'INTERMEDIATE': 25, 'HARD': 45, 'MEDIUM': 35, 'SOFT': 25}

FEATURE_NAMES = [
    'TyreLife', 'tyre_life_sq', 'tyre_wear_ratio',
    'compound_code', 'compound_x_age',
    'is_soft', 'is_medium', 'is_hard',
    'LapTime_Delta', 'Cumulative_Degradation',
    'RaceProgress', 'race_remaining',
    'Stint', 'LapNumber', 'Position', 'Position_Change',
    'PitStop',
]


def make_features(df: pd.DataFrame) -> np.ndarray:
    d = df.copy()
    d['compound_code']   = d['Compound'].map(COMPOUND_SOFTNESS).fillna(2).astype('float32')
    d['life_expected']   = d['Compound'].map(COMPOUND_LIFE).fillna(35).astype('float32')
    d['tyre_wear_ratio'] = (d['TyreLife'] / d['life_expected']).astype('float32')
    d['tyre_life_sq']    = (d['TyreLife'] ** 2).astype('float32')
    d['compound_x_age']  = (d['compound_code'] * d['TyreLife']).astype('float32')
    d['race_remaining']  = (1.0 - d['RaceProgress']).astype('float32')
    d['is_soft']         = (d['Compound'] == 'SOFT').astype('float32')
    d['is_medium']       = (d['Compound'] == 'MEDIUM').astype('float32')
    d['is_hard']         = (d['Compound'] == 'HARD').astype('float32')
    return d[FEATURE_NAMES].fillna(0).values.astype('float32')


X = make_features(df)
y = df['PitNextLap'].values.astype('float32')


# 3. RACE-BASED TRAIN / VAL / TEST SPLIT
# Laps from the same race must stay together
df['_key'] = df['Race'] + '|' + df['Year'].astype(str)
race_keys = df['_key'].unique().tolist()

rng = np.random.default_rng(SEED)
rng.shuffle(race_keys)

n = len(race_keys)
n_tr = int(0.70 * n)
n_va = int(0.15 * n)

train_set = set(race_keys[:n_tr])
val_set   = set(race_keys[n_tr:n_tr + n_va])
test_set  = set(race_keys[n_tr + n_va:])

mask_tr = df['_key'].isin(train_set).values
mask_va = df['_key'].isin(val_set).values
mask_te = df['_key'].isin(test_set).values

X_tr, y_tr = X[mask_tr], y[mask_tr]
X_va, y_va = X[mask_va], y[mask_va]
X_te, y_te = X[mask_te], y[mask_te]

print(f"\nSplit: train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te):,}")
print(f"Positive rate — train: {y_tr.mean():.3%}  val: {y_va.mean():.3%}  test: {y_te.mean():.3%}")


# 4. SCALE
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr).astype('float32')
X_va_s = scaler.transform(X_va).astype('float32')
X_te_s = scaler.transform(X_te).astype('float32')

# Class weight for imbalance (negative / positive ratio)
pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
print(f"Class imbalance pos_weight: {pos_weight:.2f}")

N_FEAT = X_tr_s.shape[1]


# 5. SKLEARN LOGISTIC REGRESSION 
print("\n[1/3] Training sklearn LogisticRegression...")
sk_model = LogisticRegression(
    C=0.5,
    class_weight='balanced',
    max_iter=500,
    solver='lbfgs',
    random_state=SEED,
)
sk_model.fit(X_tr_s, y_tr)
sk_proba_te = sk_model.predict_proba(X_te_s)[:, 1]
sk_proba_va = sk_model.predict_proba(X_va_s)[:, 1]
print("  done")


# 6. MANUAL PYTORCH LOOP
# Implements logistic regression from scratch, raw tensors, manual autograd, manual parameter update — no nn.Module, no optimizer object.
print("\n[2/3] Training manual PyTorch logistic regression...")

X_tr_t = torch.from_numpy(X_tr_s)
y_tr_t = torch.from_numpy(y_tr).unsqueeze(1)
X_te_t = torch.from_numpy(X_te_s)

w_man = torch.zeros(N_FEAT, 1, requires_grad=True)
b_man = torch.zeros(1, requires_grad=True)

LR_MAN     = 0.05
EPOCHS_MAN = 40
BS_MAN     = 4096
pw_t       = torch.tensor(pos_weight)
history_man = []

for epoch in range(EPOCHS_MAN):
    perm = torch.randperm(X_tr_t.shape[0])
    epoch_loss = 0.0
    n_batches  = 0
    for start in range(0, X_tr_t.shape[0], BS_MAN):
        idx = perm[start: start + BS_MAN]
        Xb  = X_tr_t[idx]
        yb  = y_tr_t[idx]

        logits = Xb @ w_man + b_man
        probs  = torch.sigmoid(logits)
        # Weighted binary cross-entropy (manual formula)
        loss = -(pw_t * yb * torch.log(probs + 1e-8)
                 + (1 - yb) * torch.log(1 - probs + 1e-8)).mean()

        loss.backward()

        with torch.no_grad():
            w_man -= LR_MAN * w_man.grad
            b_man -= LR_MAN * b_man.grad
        w_man.grad.zero_()
        b_man.grad.zero_()

        epoch_loss += loss.item()
        n_batches  += 1

    avg_loss = epoch_loss / n_batches
    history_man.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1:>3}/{EPOCHS_MAN}  loss={avg_loss:.4f}")

with torch.no_grad():
    man_proba_te = torch.sigmoid(X_te_t @ w_man + b_man).squeeze().numpy()


# 7. STANDARD PYTORCH (nn.Module + optim)
print("\n[3/3] Training standard PyTorch (nn.Module + Adam)...")


class LogReg(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.fc = nn.Linear(n_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # logits; sigmoid applied at inference / inside loss


nn_model  = LogReg(N_FEAT)
optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

dataset = TensorDataset(X_tr_t, y_tr_t)
loader  = DataLoader(dataset, batch_size=4096, shuffle=True)

EPOCHS_NN = 40
history_nn = []
for epoch in range(EPOCHS_NN):
    nn_model.train()
    epoch_loss = 0.0
    n_batches  = 0
    for Xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(nn_model(Xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches  += 1

    avg_loss = epoch_loss / n_batches
    history_nn.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1:>3}/{EPOCHS_NN}  loss={avg_loss:.4f}")

nn_model.eval()
with torch.no_grad():
    nn_proba_te = torch.sigmoid(nn_model(X_te_t)).squeeze().numpy()


# 8. EVALUATE
def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'auc':       roc_auc_score(y_true, y_proba),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'conf_mat':  confusion_matrix(y_true, y_pred),
    }

# Naive baseline: always predict majority class which here is no pit stop
naive_proba = np.full(len(y_te), y_tr.mean(), dtype='float32')

results = {
    'sklearn': compute_metrics(y_te, sk_proba_te),
    'manual':  compute_metrics(y_te, man_proba_te),
    'nn':      compute_metrics(y_te, nn_proba_te),
    'naive':   compute_metrics(y_te, naive_proba),
}

print("\n── Results on held-out test races ──────────────────────────────────")
print(f"{'Method':<22} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Prec':>7} {'Rec':>7}")
print("─" * 57)
for name, m in results.items():
    label = {'sklearn': 'scikit-learn', 'manual': 'Manual PyTorch',
             'nn': 'PyTorch nn.Module', 'naive': 'Naive baseline'}[name]
    print(f"{label:<22} {m['accuracy']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f} "
          f"{m['precision']:>7.4f} {m['recall']:>7.4f}")


# 9. SAVE ARTIFACTS
print("\nSaving artifacts...")

with open(MODELS_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(MODELS_DIR / "sklearn_logreg.pkl", "wb") as f:
    pickle.dump(sk_model, f)

torch.save({'w': w_man.detach(), 'b': b_man.detach()},
           MODELS_DIR / "manual_weights.pt")

torch.save(nn_model.state_dict(), MODELS_DIR / "nn_model.pt")

# Draw a stratified sample for feature-distribution plots in the dashboard
sample_idx = np.random.choice(len(y_tr), min(15_000, len(y_tr)), replace=False)
train_sample = pd.DataFrame(X_tr_s[sample_idx], columns=FEATURE_NAMES)
train_sample['target'] = y_tr[sample_idx]

dashboard_data = {
    'feature_names':   FEATURE_NAMES,
    'n_features':      N_FEAT,
    'X_test':          X_te_s,
    'y_test':          y_te,
    'proba_sklearn':   sk_proba_te,
    'proba_manual':    man_proba_te,
    'proba_nn':        nn_proba_te,
    'results':         results,
    'history_manual':  history_man,
    'history_nn':      history_nn,
    'pos_weight':      pos_weight,
    'sk_coeffs':       sk_model.coef_[0].tolist(),
    'train_sample':    train_sample,
}

with open(MODELS_DIR / "dashboard_data.pkl", "wb") as f:
    pickle.dump(dashboard_data, f)

print("All artifacts saved to models/")
print("\nTraining complete.")
