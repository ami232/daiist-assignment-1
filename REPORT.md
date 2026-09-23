# Assignment 1 Report

- **Name**: Jimena Navarro
- **Student ID**: jnavarro.ieu2021
- **Email**: jnavarro.ieu2021@student.ie.edu
- **Group**: 

## Dataset

I used the hourly Capital Bikeshare file from the UCI Bike Sharing dataset (Fanaee-T and Gama, 2013). One row is one hour in Washington, D.C., from 1 January 2011 through 31 December 2012. The raw file has 17,379 hours and 17 columns. The target is `cnt`, the number of rentals in that hour.

I picked it because the decision is hourly, the series has a shape that a straight hour number cannot represent, and it is small enough to retrain from scratch (about 1 MB). The daily file is in `data/` and I did not model it: a daily total is the wrong unit for moving bikes before a commute.

Two columns are not usable as inputs. `casual + registered` equals `cnt` on every row. A Ridge fit on those two columns alone has test MAE of about 0.01 bikes. That is leakage, so both columns are dropped. `atemp` correlates 0.988 with `temp`, so I kept temperature in °C (`temp * 41`) and dropped the feeling temperature.

## Business / real-life framing

The user of the model is a rebalancing desk. One hour ahead, they decide how many bikes should be available system-wide. Under-staging means missed trips. Over-staging means a van moved bikes that sit idle.

That framing fixed four pipeline choices.

**Target.** The desk moves bikes, not user types, so the target is total rentals. I tried training on `log1p(cnt)` because the raw counts are skewed (median 142, max 977). On the validation quarter that fit had MAE 186 bikes, against 50 for the raw count, and a negative R². The hour-to-hour link is closer to "add some bikes" than "multiply by a factor", and `expm1` turned a few bad log predictions into huge counts. The shipped target is the raw hourly count.

**Split.** The decision is always about a future hour, so the split is chronological:

| Split | Hours | Window |
|---|---:|---|
| Train | 12,591 | 8 Jan 2011 – 30 Jun 2012 |
| Validation | 2,208 | 1 Jul 2012 – 30 Sep 2012 |
| Test | 2,098 | 1 Oct 2012 – 31 Dec 2012 |

Validation is only for the feature blocks, the Ridge `alpha`, and the learning rate. The test quarter is reported once.

A random 80/20 split of the same rows, same features, same Ridge, scores test MAE 38.4. The chronological test MAE is 48.6. In the random holdout, 79% of test hours have their previous hour inside the training set, so `lag_1` is almost a neighbor of a training target. That number is not the error the desk would have faced in October 2012.

**What is known at decision time.** Last hour's count, yesterday, and last week are already observed, so those lags are allowed. This hour's weather in the file is the weather that actually occurred. I treat it as a stand-in for a forecast the desk would already have. If the forecast is wrong, the weather coefficients will not deliver what the test score suggests.

**Decision rule.** The regression predicts a mean. The desk should not stage the mean. I priced a missed rental at $5 and an unnecessary bike at $1. Those are planning assumptions, not numbers estimated from the fit. With that ratio the newsvendor quantile is 5/6, about 83%. On the test residuals that quantile is a buffer of about +45 bikes. The Gradio slider is there so that buffer can be moved and the quarter's cost moves with it. At +0 the test-quarter cost is $284,245. At +45 it is $228,203.

## Data preparation & feature engineering

The two-year span contains 165 missing hours. Lags are shifts on a complete hourly clock, not shifts of the CSV row. Otherwise a three-hour gap would look like one hour. Hours with no observed `cnt` are dropped after the shifts. The first week, and any hour whose lag hour is itself missing, cannot form `lag_168`. That removes 482 rows. Modeling starts at 8 January 2011. 16,897 hours remain.

Humidity is 0% for 22 hours, all on 10 March 2011, which is inside training. That is not a real humidity reading. I set those cells to missing and filled them with the training-period median, 62%. Wind speed is 0 on 12.5% of hours. Some of those are probably calm and some are a filled-in missing value. I did not replace them. I added `wind_is_zero` so the spike at exactly 0 can get its own coefficient (about −7 bikes).

`weathersit` 4, heavy storm, occurs 3 times. I folded those hours into precipitation. Three rows cannot support their own coefficient.

UCI's season code calls January–March "spring". I did not use `season`. Month is encoded as sin and cos of the month index, which does not depend on that label.

The validation MAE below is Ridge with `alpha = 1`, scored in bikes. Each row adds a block.

| Feature block | Validation MAE |
|---|---:|
| Raw columns (hour, month, weekday, season, weather code as numbers, plus year, holiday, working day, temp, humidity, wind) | 158.2 |
| Cyclical hour, month, and weekday, plus weather flags and a day-count trend | 140.2 |
| Plus weekday rush flags, a weekend midday flag, and working-day × hour sin/cos | 104.1 |
| Plus temp × hour, humidity × temp, precip × rush, and the wind-zero flag | 96.6 |
| Lags only: previous hour, yesterday, last week | 57.8 |
| Interactions plus those lags | 51.8 |
| Above, plus change in temp, humidity, and precip versus the same hour last week (shipped) | 50.2 |
| Training-mean baseline on the same validation hours | 196.6 |
| Same hour last week, as a forecast by itself | 56.0 |

What this table is saying:

- A plain hour number is the wrong shape. Weekday demand spikes near 8:00 and 17:00. Weekend demand spikes in the middle of the day. Sin/cos of hour is one smooth wave, so it only gets part of the way (140 vs 158). The rush and leisure flags are what cut validation MAE from 140 to 104. The dashboard's hour profile is that picture.
- Weather and calendar still matter after lags are in. Lags alone are 57.8. Adding the commute and weather features gets to 51.8.
- Change versus last week is the last block I kept. A degree warmer than the same hour last week is worth about +2 bikes in the fitted coefficient, and rain relative to last week is about −28 bikes. Validation MAE moves from 51.8 to 50.2. Small, and it is a real mechanism: last week's count is a bad copy when the weather changed.
- A trailing 24-hour mean, shifted so it does not include the current hour, does nothing once `lag_1` and `lag_24` are present (validation MAE 50.16 versus 50.16 on the same rows). I left it out.
- `days_since_start` has a tiny coefficient, about 0.02 bikes per day. That is expected. Last week's count already carries the 2011-to-2012 growth, so the trend column has little left to explain. Mean demand still rose from about 144 rentals an hour in 2011 to about 235 in 2012. The lag is how the model sees that level.

The shipped coefficients that match the story, in bikes per unit, from the scikit-learn fit (the two PyTorch fits match these to many decimals):

| Feature | Bikes per unit |
|---|---:|
| Precipitation during a rush hour | −64 |
| Weekday evening rush (16:00–19:00) | +60 |
| Weekday morning rush (7:00–9:00) | +53 |
| Weekend or holiday, 10:00–16:00 | +45 |
| Precipitation compared with last week | −28 |
| Same hour last week (`lag_168`) | +0.41 per bike |
| Previous hour (`lag_1`) | +0.31 per bike |
| Same hour yesterday (`lag_24`) | +0.17 per bike |
| Holiday | −8 |

The three lag weights add to about 0.89. The model is mostly a blend of recent history, with the rush and weather columns correcting it. The holiday weight is the failure I care about, discussed below. Humidity × temperature has a coefficient near zero after the other columns are in. The interaction block as a whole earned its keep (96.6 down from 104 before lags, and 51.8 versus 57.8 after lags). I did not delete that one dead column after looking at the fit.

Inputs are standardized with a `StandardScaler` fit on the training hours only. The target is also standardized for training, with the training mean and standard deviation, and predictions are mapped back to bike counts. Negative counts are clipped to 0 after that. On the test quarter, 22 of 2,098 hours were negative before the clip, mostly quiet nights. All three methods are clipped the same way.

## Modeling: three implementations, one model

The model is linear regression. Counts are continuous, and the business quantity is "how many bikes", not a yes/no.

Regularization is Ridge with `alpha = 1`. On validation, alpha 0.1, 1, and 10 all give MAE 50.17–50.19. Alpha 100 is 50.4. Alpha 1000 is 53.0, which is where the penalty starts to flatten the rush-hour corrections. I kept alpha 1 so collinear calendar columns are damped and the fit is still the accurate one.

The two PyTorch fits minimize mean squared error plus `(l2 / 2) * ||w||^2`, with the bias unpenalized. scikit-learn Ridge minimizes the sum of squares plus `alpha * ||w||^2`, also without penalizing the intercept. Those are the same problem when `l2 = 2 * alpha / n_train`. I did not use `weight_decay` on the optimizer, because that would shrink the bias too.

Both PyTorch versions start at zero weights, use full-batch SGD, learning rate 0.1, and 3,000 epochs, in float64. The manual loop is the Session 5 pattern: `y_hat = X @ w + b`, `loss.backward()`, then `w -= lr * w.grad` inside `torch.no_grad()`. The standard version is `nn.Linear` plus `torch.optim.SGD`, with the same penalty written on `linear.weight`. They are the same update, so they should match. They do.

Learning rate 0.05 also reaches the Ridge weights (largest gap about 8e-5 after 3,000 epochs). Learning rate 0.2 diverges: the loss explodes. Lags and the calendar columns are correlated, so some directions of the loss are steep and 0.2 steps over them. I kept 0.1, which lands on the closed form.

Test quarter, October–December 2012, predictions in bikes:

| Method | MAE | RMSE | R² |
|---|---:|---:|---:|
| scikit-learn Ridge | 48.59 | 71.71 | 0.874 |
| Manual PyTorch loop | 48.59 | 71.71 | 0.874 |
| `nn.Module` + SGD | 48.59 | 71.71 | 0.874 |
| Baseline: predict the training mean | 157.12 | 208.12 | −0.060 |
| Baseline: same hour last week | 69.56 | 114.85 | 0.677 |

Largest absolute weight gap between Ridge and the manual loop is 3.1e-7. Between the manual loop and `nn.Module` it is about 3e-17. The largest test-set prediction gap between Ridge and the manual loop is 0.00004 bikes. The MAE values above match to the printed precision. Agreement here is not a coincidence I am papering over: the loss, the penalty, the initialization, and the learning rate were set so that gradient descent solves the same problem Ridge solves in closed form.

Beating the training mean is easy and not the bar. Same-hour-last-week is the bar an operator would actually use. On the summer validation quarter that baseline already has MAE 56.0, and the model only improves it to 50.2, because one July week looks like the next. On the autumn/winter test quarter, last week is a worse copy (MAE 69.6) and the weather and calendar corrections matter more (MAE 48.6).

## Limitations & next steps

The holiday flag subtracts about 8 bikes. That is far too small for the days the test error is actually about. On Thanksgiving, 22 November 2012, the day total was 2,425 rentals and the forecast said 4,461. On Christmas the totals were 1,012 versus 2,851. The flag was estimated from a mixture of ordinary holidays in the training window, and a single binary cannot tell Christmas from a minor weekday off. A next step is separate indicators for the handful of shutdown-style holidays, or a "days until Christmas / Thanksgiving" feature that is known in advance.

15 October 2012 has hourly MAE about 90, but the day totals almost match (5,875 actual versus 5,837 predicted). The hours are wrong in a way that cancels across the day. For staging, the hourly error is the one that counts. A model with a sharper peak, or separate morning and evening residuals, would be the next experiment. It would no longer be one linear regression.

Hurricane Sandy is mostly absent rather than mis-predicted. 29 October 2012 has one recorded hour in this file, and 30 October has none. A linear model fit on ordinary weather will not represent a shutdown that was never in the training target.

Clipping negative predictions at 0 is a patch. 22 test hours needed it. The log target was the principled version of "counts cannot be negative", and it failed for the reason in the framing section. A Poisson or Tweedie model would respect the count and the variance that grows with the mean. That is a different model from the one this assignment asks for.

The prediction is system-wide. Rebalancing is station-level. A good citywide total can still leave the wrong dock empty. Station data is a different dataset.

Realized weather stands in for a forecast. The test score assumes the desk already knows this hour's temperature and precipitation. A lagged weather forecast, or weather from the previous hour only, is the honest production feature, and the MAE would rise.

The 2011–2012 growth is carried by `lag_168`. That works inside this file. It does not tell you what happens if the system stops expanding. I would not trust the day-count trend on its own past December 2012.

## Generative AI use disclosure

I used Cursor's coding agent (Grok) to implement `train.py` and `app.py`, to run the training script, and to format this report from the numbers that script printed. The dataset, the one-hour-ahead rebalancing decision, the chronological split, which feature blocks to keep, and the reading of the holiday failures are the decisions this report is accountable for. The tables match `artifacts/metrics.json` from `uv run python main.py train`.
