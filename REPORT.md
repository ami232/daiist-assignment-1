# Assignment 1 Report

Draft for review. The business idea and modelling preferences below came from my responses. The results interpretation and limitations are marked as AI-assisted draft text and need my review and revision before submission under the course AI policy.

- **Name**: Javier Corredera
- **Student ID**: 19654
- **Email**: jcorredera.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

Source: [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset), Fanaee-T (2013), https://doi.org/10.24432/C5W894. Licensed CC BY 4.0 according to UCI.

The committed `data/hour.csv` contains 17,379 rows and 17 columns, covering 2011-01-01 to 2012-12-31. Each row records one observed hour of system-wide Capital Bikeshare rentals in Washington DC. The target `cnt` counts rentals, including casual and registered users. It is not a count of bikes available in the fleet.

I chose bike rentals and demand because I am interested in operating a small rental location in a city center, where rent can be expensive. Predicting demand could help keep less inventory on site while planning when additional bikes may be needed.

## Business / real-life framing

My hypothetical scenario is a bike-rental shop in the center of a large city, such as New York. I want to help the operator plan bike availability. Space in city centers can be limited because rent is expensive. Estimating busy periods would help an operator decide when more bikes may be needed, including replenishment during peak season.

My chosen priority is to give unusually large prediction errors more weight. The comparison will therefore use RMSE as its main metric.

This dataset records Washington DC system-wide rentals. The shop scenario is illustrative: the data cannot validate demand for one shop, transfer performance to New York, or determine the exact number of bikes a shop needs. The model predicts hourly rental counts as a demand indicator.

I chose calendar-only inputs so future predictions do not require knowing the actual future weather. Temperature and all other weather measurements are excluded. I chose to train on earlier dates and evaluate on later dates, because the planning scenario concerns future demand. Training uses dates before July 2012, validation uses July–September 2012, and the final test uses October–December 2012. All hours on the same day stay together.

The feature experiment tests how hourly demand changes by season. The model stays linear in its coefficients: season–hour combinations are indicator columns. RMSE is the main comparison metric because I selected large misses as my priority; MAE and R² are also reported.

## Data preparation & feature engineering

The original CSV is committed so training needs no download. The audit found no missing cells or duplicate hours. There are 165 unrecorded hours inside the date range; they remain absent rather than being filled with assumed zero rentals. Rows are sorted by timestamp before splitting.

| Split | Dates | Recorded hours |
|---|---|---:|
| Train | 2011-01-01 to 2012-06-30 | 13,003 |
| Validation | 2012-07-01 to 2012-09-30 | 2,208 |
| Test | 2012-10-01 to 2012-12-31 | 2,168 |

The target is `cnt`. `casual` and `registered` are excluded from the inputs because their sum equals the target. The identifier and all weather columns are also excluded. Inputs are year, holiday, working day, weekday, season, and hour. All are calendar information.

Categorical variables become indicator columns (one-hot encoding). The feature experiment compares separate hour and season indicators (35 total columns) with season–hour combination indicators (104 total columns). The combination gives each season its own hourly pattern. It replaces the separate hour/season columns rather than adding a redundant copy of them.

The category ranges are fixed from the data dictionary. A StandardScaler is fitted on training rows only and then applied unchanged to validation and test. The target is standardized with its training mean and standard deviation for all three implementations, then predictions are converted back to rental counts. The app loads this saved preprocessing and uses the same feature function.

| Feature representation | Validation RMSE | Validation MAE |
|---|---:|---:|
| Separate hour and season | 150.070 | 115.294 |
| Season × hour | 143.458 | 107.527 |

**AI-assisted draft interpretation, for student review:** I would keep the season–hour feature for this experiment because its validation RMSE is lower, from about 150 to 143. This indicates that treating each season as having its own hourly pattern helped on the validation period. One comparison is not proof that it will improve every future period.

## Modeling: three implementations, one model

All three versions are linear regression with the same L2 regularization (Ridge), feature matrix, dates, and target scaling. This is one weighted sum plus an intercept, including in the PyTorch version; there are no hidden layers.

The common training objective is:

`mean((prediction - standardized_target)^2) + lambda * sum(weights^2)`

The intercept is not penalized. The fixed regularization strength is `lambda = 0.01`. scikit-learn minimizes a summed rather than averaged squared error, so its equivalent `alpha` is `n_train × lambda = 130.03`. Both PyTorch loops use a learning rate of 0.05, full training batches, and zero initial weights and bias. They stop when the largest absolute gradient is below `1e-8`, with a maximum of 20,000 iterations; this run stopped at iteration 5,479. Float64 is used on CPU for the numerical comparison. No hyperparameter search was performed.

The scikit-learn version uses Ridge's solver. The manual version uses raw tensors, `backward()`, a parameter update inside `torch.no_grad()`, and clears accumulated gradients. The standard version uses `nn.Module`, `nn.Linear`, `optimizer.zero_grad()`, `backward()`, and `optimizer.step()` with SGD. Both optimize the same expression.

The naive baseline always predicts the training-period mean (about 167.71 rentals/hour). It does not learn seasonal or hourly patterns. All metrics below use the same 2,168 test hours and unrounded predictions.

| Method | Test RMSE | Test MAE | Test R² | Negative predictions |
|---|---:|---:|---:|---:|
| scikit-learn | 134.928 | 103.362 | 0.5520 | 0 |
| Manual PyTorch | 134.928 | 103.362 | 0.5520 | 0 |
| Standard PyTorch | 134.928 | 103.362 | 0.5520 | 0 |
| Training-mean baseline | 208.056 | 156.565 | -0.0652 | 0 |

The maximum absolute prediction difference between scikit-learn and either PyTorch implementation is 0.000291 rental/hour. Between the two PyTorch implementations it is about `1.14e-13`. Their normalized training objectives agree to better than `1e-10`. The small difference from the solver is due to the gradient loops stopping at a finite tolerance.

**AI-assisted draft interpretation, for student review:** The model improves on predicting the same average every hour: test RMSE falls from about 208 to 135, approximately a 35% reduction. MAE is about 103 rentals/hour, so the errors are still substantial. RMSE gives large misses extra weight; it is not a claim that every prediction is within 135 rentals. The test R² is about 0.55, compared with -0.065 for the training-mean baseline. This provides a useful illustration of demand forecasting, but does not validate inventory decisions for a small shop.

The methods agree because their feature data and mathematical objective match. Their training workflows differ, rather than the type of model. The dashboard comparison is built from the saved trained models; no training occurs at startup or when a control changes.

## Limitations & next steps

**AI-assisted draft text, for student review:**

- The data describes Washington DC system-wide rentals in 2011–2012. It has neither individual shop demand nor current New York activity. Before using it for the shop scenario, collect recent shop-level bookings, rentals, stock availability, return times, and lost demand when no bike is available, then validate predictions on later shop dates.
- Rental counts are not the same as fleet size. A bike can be rented more than once, and the data does not say how many bikes were simultaneously unavailable. Inventory planning needs rental duration, returns, capacity, and operating costs in addition to demand forecasts.
- Calendar-only inputs cannot respond to unexpected weather, disruptions, or special events. This was an intentional input choice to avoid using unavailable future observations. A later extension could use weather forecasts available at the actual prediction time and compare that change on validation data.
- A separate hourly curve for each season still averages over differing day types. For example, the dashboard's 2012-10-01 18:00 scenario predicts about 418 rentals when 613 were recorded. An additional hour-by-working-day experiment could be evaluated on validation data before changing the final model.
- The final test covers one three-month period. Performance across later years or other cities is unknown. More historical years and repeated chronological evaluations would be needed to assess stability. Missing recorded hours also limit what daily totals mean; the dashboard sums only observed hours.
- The regression gives point estimates rather than an inventory guarantee. A real deployment would need an uncertainty estimate and a decision rule tied to the cost of insufficient versus excess stock.

