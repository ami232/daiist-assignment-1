| model | label | mae | rmse | r2 | rmsle | bias | mae_peak_hours |
|---|---|---|---|---|---|---|---|
| baseline_mean | baseline_mean | 171.394 | 241.501 | -0.435 | 1.440 | -132.987 | 486.900 |
| baseline_hour_profile | baseline_hour_profile | 99.766 | 143.610 | 0.492 | 0.657 | -73.800 | 266.608 |
| baseline_last_week_level | baseline_last_week_level | 83.132 | 123.637 | 0.624 | 0.695 | -11.177 | 182.564 |
| sklearn | scikit-learn Ridge | 44.600 | 72.966 | 0.869 | 0.429 | -6.573 | 82.109 |
| torch_manual | PyTorch manual loop | 44.600 | 72.966 | 0.869 | 0.429 | -6.572 | 82.108 |
| torch_standard | PyTorch nn.Module + optim | 44.536 | 72.872 | 0.869 | 0.429 | -6.528 | 81.946 |
