| feature_set | target | n_columns | mae | rmse | r2 | rmsle | bias | mae_peak_hours | n_peak_hours |
|---|---|---|---|---|---|---|---|---|---|
| A_raw_numeric | cnt (raw) | 10 | 157.334 | 195.710 | 0.283 | 1.277 | 6.270 | 255.749 | 616 |
| A_raw_numeric | log1p(cnt) | 10 | 158.954 | 223.200 | 0.067 | 0.984 | -5.760 | 257.271 | 616 |
| B_onehot_calendar | log1p(cnt) | 48 | 98.432 | 139.081 | 0.638 | 0.573 | 13.523 | 153.610 | 616 |
| C_hour_x_workingday | log1p(cnt) | 72 | 65.052 | 108.161 | 0.781 | 0.321 | 34.325 | 114.032 | 616 |
| D_weather_terms | log1p(cnt) | 76 | 50.403 | 82.030 | 0.874 | 0.293 | 15.087 | 88.705 | 616 |
| E_plus_lag_level | log1p(cnt) | 77 | 45.961 | 74.034 | 0.897 | 0.279 | -1.467 | 83.379 | 616 |
| E_alt_cyclical_hour | log1p(cnt) | 36 | 91.783 | 133.575 | 0.666 | 0.563 | -33.347 | 188.272 | 616 |
| E_alt_plus_trend | log1p(cnt) | 78 | 46.465 | 74.638 | 0.896 | 0.280 | -2.174 | 84.436 | 616 |
