| Model | Threshold | ROC AUC | PR AUC | Log loss | Recall | Precision | Cost per client |
|---|---|---|---|---|---|---|---|
| scikit-learn | 0.26 | 0.7659 | 0.5354 | 0.4387 | 0.552 | 0.508 | 0.4157 |
| manual PyTorch | 0.26 | 0.7659 | 0.5354 | 0.4387 | 0.551 | 0.508 | 0.4160 |
| nn.Module + optim | 0.27 | 0.7660 | 0.5360 | 0.4384 | 0.531 | 0.519 | 0.4198 |
| naive baseline (approve everyone) | - | 0.5000 | 0.2212 | 0.5284 | - | - | 0.6635 |
