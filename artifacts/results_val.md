| model | label | mae | rmse | r2 | rmsle | bias | mae_peak_hours |
|---|---|---|---|---|---|---|---|
| baseline_mean | baseline_mean | 231.127 | 306.996 | -0.765 | 1.483 | -202.144 | 509.508 |
| baseline_hour_profile | baseline_hour_profile | 145.389 | 192.344 | 0.307 | 0.717 | -143.229 | 310.715 |
| baseline_last_week_level | baseline_last_week_level | 90.018 | 130.729 | 0.680 | 0.569 | -25.005 | 172.883 |
| sklearn | scikit-learn Ridge | 45.981 | 74.066 | 0.897 | 0.279 | -1.353 | 83.383 |
| torch_manual | PyTorch manual loop | 45.981 | 74.067 | 0.897 | 0.279 | -1.352 | 83.383 |
| torch_standard | PyTorch nn.Module + optim | 45.981 | 74.107 | 0.897 | 0.279 | -0.956 | 83.275 |
