| Model | ROC-AUC | PR-AUC | Log loss | Threshold | Precision | Recall | Cost / 100 KOIs |
|---|---|---|---|---|---|---|---|
| Naive baseline (train prior) | 0.500 | 0.362 | 0.654 | 0.00 | 0.362 | 1.000 | 63.8 |
| scikit-learn LogisticRegression | 0.984 | 0.969 | 0.148 | 0.24 | 0.873 | 0.971 | 8.3 |
| PyTorch manual loop | 0.984 | 0.969 | 0.148 | 0.32 | 0.896 | 0.968 | 7.5 |
| PyTorch nn.Module + optimizer | 0.984 | 0.969 | 0.151 | 0.25 | 0.875 | 0.971 | 8.2 |
