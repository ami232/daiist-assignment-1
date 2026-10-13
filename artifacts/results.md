| Model | ROC-AUC | Avg precision | Log-loss | Threshold | Calls | Precision | Recall | Profit (EUR) |
|---|---|---|---|---|---|---|---|---|
| scikit-learn | 0.5885 | 0.4528 | 0.7785 | 0.15 | 3429 | 0.447 | 0.645 | 88,350 |
| Manual PyTorch | 0.5863 | 0.4517 | 0.7905 | 0.13 | 3743 | 0.443 | 0.698 | 95,210 |
| Standard PyTorch | 0.5863 | 0.4517 | 0.7905 | 0.13 | 3743 | 0.443 | 0.698 | 95,210 |
| Baseline: call everyone | 0.5000 | 0.3845 | 1.1457 | 0.00 | 6179 | 0.385 | 1.000 | 128,290 |
| Baseline: call nobody | 0.5000 | 0.3845 | 1.1457 | 1.01 | 0 | 0.000 | 0.000 | 0 |
