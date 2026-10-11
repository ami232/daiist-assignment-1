"""Winsorize and standardize, with every statistic learned on the training block only."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLIP_QUANTILES


class Preprocessor:
    """Clip each feature to its training 1st/99th percentiles, then standardize with training mean and std.

    Why clip: the test years include COVID, where some features sit far outside anything seen
    in training. A linear model extrapolates those values straight into extreme logits.
    Why standardize: gradient descent in PyTorch converges badly when features live on very
    different scales, and the L2 penalty treats every coefficient the same, which is only
    fair if the features share a scale. scikit-learn gets the same inputs so all three
    implementations solve exactly the same problem.
    """

    def __init__(self, features: list[str]):
        self.features = list(features)
        self.lower = self.upper = self.mean = self.std = None

    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        x = train[self.features]
        self.lower = x.quantile(CLIP_QUANTILES[0])
        self.upper = x.quantile(CLIP_QUANTILES[1])
        clipped = x.clip(self.lower, self.upper, axis=1)
        self.mean = clipped.mean()
        self.std = clipped.std(ddof=0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features].clip(self.lower, self.upper, axis=1)
        return ((x - self.mean) / self.std).to_numpy(dtype=np.float32)

    def save(self, path: Path) -> None:
        state = {"features": self.features}
        for name in ("lower", "upper", "mean", "std"):
            state[name] = {k: float(v) for k, v in getattr(self, name).items()}
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        state = json.loads(Path(path).read_text())
        p = cls(state["features"])
        for name in ("lower", "upper", "mean", "std"):
            setattr(p, name, pd.Series(state[name])[p.features])
        return p
