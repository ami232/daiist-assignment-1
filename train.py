from __future__ import annotations

import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

MATCHES_PATH = DATA_DIR / "dota2_500k_matches_ids.csv"
HERO_MAP_PATH = DATA_DIR / "dota_hero_id_map.csv"
ATTR_PATH = DATA_DIR / "complete_opendota_attributes.csv"

MODEL_ROWS = 20_000
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
RANDOM_SEED = 42
MANUAL_LEARNING_RATE = 0.10
STANDARD_LEARNING_RATE = 0.05
EPOCHS = 300

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def parse_team(value: object) -> list[int]:
    """Convert OpenDota's comma-separated team field into five hero IDs."""
    try:
        return [int(x) for x in str(value).split(",") if str(x).strip()]
    except (TypeError, ValueError):
        return []


def load_metadata() -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
    hero_map = pd.read_csv(HERO_MAP_PATH)
    attrs = pd.read_csv(ATTR_PATH)

    hero_meta: dict[int, dict[str, object]] = {}
    for row in attrs.itertuples(index=False):
        roles = ast.literal_eval(row.roles_list) if isinstance(row.roles_list, str) else []
        hero_meta[int(row.hero_id)] = {
            "name": str(row.name),
            "primary_attr": str(row.primary_attr),
            "roles": list(roles),
        }
    return hero_map, hero_meta


def load_recent_valid_matches(valid_hero_ids: set[int]) -> pd.DataFrame:
    usecols = ["match_id", "radiant_win", "radiant_team", "dire_team"]
    df = pd.read_csv(MATCHES_PATH, usecols=usecols)

    # Higher match IDs are newer, so start with the most recent records.
    df = df.sort_values("match_id", ascending=False).reset_index(drop=True)
    radiant_parsed = df["radiant_team"].map(parse_team)
    dire_parsed = df["dire_team"].map(parse_team)

    valid_mask = []
    for radiant, dire in zip(radiant_parsed, dire_parsed):
        all_ids = radiant + dire
        valid = (
            len(radiant) == 5
            and len(dire) == 5
            and len(set(all_ids)) == 10
            and all(hero_id in valid_hero_ids for hero_id in all_ids)
        )
        valid_mask.append(valid)

    valid_mask = np.asarray(valid_mask, dtype=bool)
    df = df.loc[valid_mask].copy()
    df["radiant_ids"] = radiant_parsed.loc[valid_mask].values
    df["dire_ids"] = dire_parsed.loc[valid_mask].values
    df = df.head(MODEL_ROWS).copy()

    if len(df) < MODEL_ROWS:
        raise RuntimeError(
            f"Only {len(df)} valid matches found. Expected at least {MODEL_ROWS}."
        )

    # Oldest -> newest so train/validation/test mimic predicting future matches.
    return df.sort_values("match_id", ascending=True).reset_index(drop=True)


def build_feature_names(
    hero_ids: list[int], roles: list[str], attributes: list[str]
) -> list[str]:
    names = [f"hero_{hero_id}_side" for hero_id in hero_ids]
    names += [f"role_diff_{role.lower()}" for role in roles]
    names += [f"attr_diff_{attribute}" for attribute in attributes]
    return names


def featurize_match(
    radiant_ids: list[int],
    dire_ids: list[int],
    hero_ids: list[int],
    hero_index: dict[int, int],
    hero_meta: dict[int, dict[str, object]],
    roles: list[str],
    attributes: list[str],
) -> np.ndarray:
    """Create hero-presence and team-composition features for one draft."""
    vector = np.zeros(len(hero_ids) + len(roles) + len(attributes), dtype=np.float32)

    # Signed hero indicator: +1 Radiant, -1 Dire, 0 not picked.
    for hero_id in radiant_ids:
        vector[hero_index[hero_id]] = 1.0
    for hero_id in dire_ids:
        vector[hero_index[hero_id]] = -1.0

    offset = len(hero_ids)
    for i, role in enumerate(roles):
        radiant_count = sum(role in hero_meta[h]["roles"] for h in radiant_ids)
        dire_count = sum(role in hero_meta[h]["roles"] for h in dire_ids)
        vector[offset + i] = float(radiant_count - dire_count)

    offset += len(roles)
    for i, attribute in enumerate(attributes):
        radiant_count = sum(
            hero_meta[h]["primary_attr"] == attribute for h in radiant_ids
        )
        dire_count = sum(hero_meta[h]["primary_attr"] == attribute for h in dire_ids)
        vector[offset + i] = float(radiant_count - dire_count)

    return vector


def build_matrix(
    df: pd.DataFrame,
    hero_ids: list[int],
    hero_meta: dict[int, dict[str, object]],
    roles: list[str],
    attributes: list[str],
) -> np.ndarray:
    hero_index = {hero_id: i for i, hero_id in enumerate(hero_ids)}
    return np.vstack(
        [
            featurize_match(
                radiant,
                dire,
                hero_ids,
                hero_index,
                hero_meta,
                roles,
                attributes,
            )
            for radiant, dire in zip(df["radiant_ids"], df["dire_ids"])
        ]
    ).astype(np.float32)


def time_split(df: pd.DataFrame, X: np.ndarray, y: np.ndarray):
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    return {
        "train": (df.iloc[:train_end].copy(), X[:train_end], y[:train_end]),
        "val": (df.iloc[train_end:val_end].copy(), X[train_end:val_end], y[train_end:val_end]),
        "test": (df.iloc[val_end:].copy(), X[val_end:], y[val_end:]),
    }


def sigmoid_np(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def evaluate(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def best_validation_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, int]:
    """Minimize total wrong draft calls (FP + FN) using validation data only."""
    best_threshold = 0.5
    best_cost = 10**9
    for threshold in np.linspace(0.10, 0.90, 161):
        predictions = probabilities >= threshold
        false_positive = int(((predictions == 1) & (y_true == 0)).sum())
        false_negative = int(((predictions == 0) & (y_true == 1)).sum())
        cost = false_positive + false_negative
        if cost < best_cost:
            best_threshold = float(threshold)
            best_cost = cost
    return best_threshold, best_cost


def train_manual_pytorch(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, float]:
    """Session 5 manual approach: raw tensors, sigmoid/BCE, autograd, manual updates."""
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    weights = torch.zeros(X_tensor.shape[1], dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    def forward(X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(X @ weights + bias)

    def bce_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
        per_sample = -(
            y_true * torch.log(y_pred)
            + (1 - y_true) * torch.log(1 - y_pred)
        )
        return per_sample.mean()

    for _ in range(EPOCHS):
        y_pred = forward(X_tensor)
        loss = bce_loss(y_pred, y_tensor)

        loss.backward()

        with torch.no_grad():
            weights -= MANUAL_LEARNING_RATE * weights.grad
            bias -= MANUAL_LEARNING_RATE * bias.grad

        weights.grad.zero_()
        bias.grad.zero_()

    return weights.detach().numpy(), float(bias.detach().item())


def train_standard_pytorch(X_train: np.ndarray, y_train: np.ndarray) -> torch.nn.Module:
    """Session 5 appendix approach: nn.Sequential + Adam + BCELoss."""
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(X_tensor.shape[1], 1),
        torch.nn.Sigmoid(),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=STANDARD_LEARNING_RATE,
    )
    loss_fn = torch.nn.BCELoss()

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        y_pred = model(X_tensor).squeeze(-1)
        loss = loss_fn(y_pred, y_tensor)
        loss.backward()
        optimizer.step()

    return model


def main() -> None:
    hero_map, hero_meta = load_metadata()
    hero_ids = sorted(hero_meta)
    roles = sorted({role for meta in hero_meta.values() for role in meta["roles"]})
    attributes = sorted({str(meta["primary_attr"]) for meta in hero_meta.values()})

    matches = load_recent_valid_matches(set(hero_ids))
    X_raw = build_matrix(matches, hero_ids, hero_meta, roles, attributes)
    y = matches["radiant_win"].astype(np.float32).to_numpy()
    feature_names = build_feature_names(hero_ids, roles, attributes)

    splits = time_split(matches, X_raw, y)
    train_df, X_train_raw, y_train = splits["train"]
    val_df, X_val_raw, y_val = splits["val"]
    test_df, X_test_raw, y_test = splits["test"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    # Naive baseline = always use the training-set Radiant win probability.
    baseline_probability = float(y_train.mean())
    baseline_test_prob = np.full(len(y_test), baseline_probability, dtype=np.float32)

    # Same unregularized logistic regression objective for all three implementations.
    sklearn_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    sklearn_model.fit(X_train, y_train.astype(int))
    sklearn_val_prob = sklearn_model.predict_proba(X_val)[:, 1]
    sklearn_test_prob = sklearn_model.predict_proba(X_test)[:, 1]

    manual_weights, manual_bias = train_manual_pytorch(X_train, y_train)
    manual_test_prob = sigmoid_np(X_test @ manual_weights + manual_bias)

    standard_model = train_standard_pytorch(X_train, y_train)
    standard_model.eval()
    with torch.no_grad():
        standard_test_prob = (
            standard_model(torch.tensor(X_test, dtype=torch.float32))
            .squeeze(-1)
            .numpy()
        )

    threshold, validation_cost = best_validation_threshold(
        y_val.astype(int), sklearn_val_prob
    )

    model_probabilities = {
        "Naive baseline": baseline_test_prob,
        "scikit-learn": sklearn_test_prob,
        "Manual PyTorch": manual_test_prob,
        "PyTorch nn.Module": standard_test_prob,
    }

    metrics = []
    for model_name, probabilities in model_probabilities.items():
        metrics.append({"model": model_name, **evaluate(y_test.astype(int), probabilities)})
    metrics_df = pd.DataFrame(metrics)

    print("\nTest-set comparison at threshold 0.50")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        f"\nValidation threshold minimizing wrong draft calls: {threshold:.3f} "
        f"(validation cost={validation_cost})"
    )

    id_to_name = dict(
        zip(hero_map["id"].astype(int), hero_map["localized_name"].astype(str))
    )
    hero_names = {hero_id: id_to_name.get(hero_id, hero_meta[hero_id]["name"]) for hero_id in hero_ids}

    # Save all objects needed by app.py. app.py never trains anything.
    joblib.dump(sklearn_model, ARTIFACT_DIR / "sklearn_model.joblib")
    joblib.dump(
        {
            "scaler": scaler,
            "hero_ids": hero_ids,
            "hero_names": hero_names,
            "hero_meta": hero_meta,
            "roles": roles,
            "attributes": attributes,
            "feature_names": feature_names,
        },
        ARTIFACT_DIR / "preprocessor.joblib",
    )
    torch.save(
        {"weights": torch.tensor(manual_weights, dtype=torch.float32), "bias": manual_bias},
        ARTIFACT_DIR / "manual_torch.pt",
    )
    torch.save(standard_model.state_dict(), ARTIFACT_DIR / "standard_torch.pt")

    predictions = pd.DataFrame(
        {
            "match_id": test_df["match_id"].to_numpy(),
            "actual": y_test.astype(int),
            "baseline_prob": baseline_test_prob,
            "sklearn_prob": sklearn_test_prob,
            "manual_torch_prob": manual_test_prob,
            "standard_torch_prob": standard_test_prob,
        }
    )
    predictions.to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False)
    metrics_df.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)

    target_distribution = (
        matches["radiant_win"]
        .value_counts()
        .sort_index()
        .rename_axis("radiant_win")
        .reset_index(name="count")
    )
    target_distribution.to_csv(ARTIFACT_DIR / "target_distribution.csv", index=False)

    pick_counts = {hero_id: 0 for hero_id in hero_ids}
    for radiant, dire in zip(matches["radiant_ids"], matches["dire_ids"]):
        for hero_id in radiant + dire:
            pick_counts[hero_id] += 1

    hero_pick_counts = pd.DataFrame(
        {
            "hero_id": hero_ids,
            "hero": [hero_names[hero_id] for hero_id in hero_ids],
            "pick_count": [pick_counts[hero_id] for hero_id in hero_ids],
        }
    ).sort_values("pick_count", ascending=False)
    hero_pick_counts.to_csv(ARTIFACT_DIR / "hero_pick_counts.csv", index=False)

    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "sklearn_coefficient": sklearn_model.coef_[0],
        }
    ).sort_values("sklearn_coefficient", ascending=False)
    coefficients.to_csv(ARTIFACT_DIR / "feature_coefficients.csv", index=False)

    metadata = {
        "model_rows": int(len(matches)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "n_features": int(X_train.shape[1]),
        "baseline_train_radiant_win_rate": baseline_probability,
        "validation_selected_threshold": threshold,
        "validation_cost_at_selected_threshold": int(validation_cost),
        "business_cost_definition": "1 cost unit per wrong draft call: FP + FN",
        "oldest_match_id": int(matches["match_id"].min()),
        "newest_match_id": int(matches["match_id"].max()),
        "manual_learning_rate": MANUAL_LEARNING_RATE,
        "standard_learning_rate": STANDARD_LEARNING_RATE,
        "epochs": EPOCHS,
    }
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"\nSaved training artifacts to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
