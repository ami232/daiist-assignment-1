# Assignment 1 Report

- **Name**: Tomas Lock
- **Student ID**: 18613
- **Email**: tlock.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

The Seoul Bike Sharing Demand dataset, from the UCI Machine Learning Repository
(https://doi.org/10.24432/C5F62R). Each row is one hour of Seoul's public bike-sharing
system, from 1 December 2017 to 30 November 2018: 8,760 rows and 14 columns. The target
is the number of bikes rented in that hour. The other columns are the date and hour,
weather (temperature, humidity, wind speed, visibility, dew point, solar radiation,
rainfall, snowfall), the season, whether it was a public holiday, and whether the
system was operating. After cleaning (see below), 8,465 rows remain.

I picked it because I'm getting into cycling, so I have a feel for why and when people
ride, and that helps with designing features. It also has a time dimension, which makes
the split a real decision rather than a default.

## Business / real-life framing

The model supports Seoul's bike-sharing operator in planning the next day. Each day, the
operations team uses a forecast of tomorrow's hourly rentals to decide how many bikes
(and free docks) need to be in service across the city at each hour, and when to schedule
staff to redistribute bikes.

- **Target:** the total number of bikes rented in each hour of the next day, citywide.
  The data is citywide, not per station, so the model supports city-level planning
  (fleet size, staffing), not decisions about individual stations.
- **When:** one day ahead. In practice, the weather inputs would come from a weather
  forecast. This dataset contains the actual weather, so I assume the forecast is
  accurate, which makes my results somewhat optimistic (see Limitations).
- **Only open hours:** days when the system is closed for maintenance are known in
  advance, so the forecast for them is simply 0 and they are removed from the model's data.
- **Cost of errors:** predicting too few bikes is worse than predicting too many. If
  customers find no bike, the company loses that ride and may lose the customer to
  another type of transport for good. Too many bikes only wastes staff time and leaves
  bikes idle.
- **When errors matter most:** at rush hours, because that's when most customers (and
  potential customers) are affected. A miss at 6pm costs far more than a miss at 4am.

### How this framing drives the pipeline
- **Split:** chronological, by whole days, because the model is used to predict days it
  has never seen, after the days it learned from. A random split would let the model
  learn from days after the ones it is tested on, which is impossible in real use.
  Train: Dec 2017–Aug 2018 (6,528 hours). Validation: Sep 2018 (624 hours, used for every
  decision). Test: Oct–Nov 2018 (1,313 hours, used once, at the end).
- **Consequence for features:** with this split, the training data contains no autumn,
  so the model can't learn anything about the "Autumn" category of the `seasons` column.
  I drop `seasons` and rely on the weather columns, which describe what each season is
  actually like.
- **Features:** only information available one day ahead (calendar, holidays, weather
  forecast). Nothing that would only be known on the day itself.
- **Metric:** RMSE in bikes, because it penalises large errors more than small ones, and
  the largest errors happen at rush hours, when they matter most. I also report RMSE on
  weekday rush hours only (hours starting 7, 8, 9 and 17, 18, 19), to measure performance
  where it matters most for the business. MAE and R² are reported as supporting numbers.
- **Safety buffer:** linear regression treats over- and under-predicting the same, but
  for this business under-predicting is worse. The dashboard has a buffer slider (plan
  for forecast + X%) that shows the trade-off between hours with too few bikes and bikes
  left idle. The model is unchanged; the buffer is a planning rule on top of it.

## Data preparation & feature engineering

### Cleaning
- **Removed the 295 non-functioning hours.** All of them have exactly 0 rentals: the
  system was closed, so 0 doesn't measure demand. There are 13 such days (all full days
  except 7 hours on 6 October), and 10 of them fall between mid-September and November,
  so keeping them would also have distorted the validation and test sets.
- **Dropped dew point.** Temperature and humidity explain 99% of it (R² = 0.990), so it
  adds almost no information and makes coefficients unstable.
- **Dropped seasons.** The training period contains no autumn (see the split above).
- No missing values in the dataset.

### Features
- **Hour × day-type interaction (one-hot).** Working days have two sharp commuting peaks
  (8am and 6pm); weekends have no morning peak and one smooth afternoon peak, plus more
  late-night riding. So the effect of the hour depends on the type of day. I made one
  yes/no column for every (working day, hour) and (day off, hour) pair. Working-day hour 0
  is left out as the reference, because all 48 columns always add up to 1 and would
  duplicate the intercept, leaving 47 columns. I chose one-hot over hand-made hour groups
  because groups would mean choosing boundaries myself, which is subjective and could
  differ between weekdays and weekends.
- **Public holidays count as days off,** like weekends: no commuting.
- **Weather:** temperature, humidity, wind speed, visibility, solar radiation, rainfall,
  snowfall.
- **Scaling:** all features standardised with a scaler fitted on the training period
  only, so nothing from validation or test leaks in. It also makes gradient descent
  converge much faster in the PyTorch versions.

### Round 2: fixing what the error diagnostics showed
After the first version, I plotted the errors against temperature, rain, and the hourly
pattern by month. The biggest errors (above 30°C, rainy hours) happen in months the model
trained on, so the problem was the model's shape (straight lines), not unseen data:

1. **temperature²:** the model kept adding bikes as it got hotter, but people stop riding
   when it's too hot. A squared term lets the effect rise and then fall.
2. **is_raining (yes/no):** most rainy hours have light rain, so a straight line in mm
   barely lowered the prediction. But any rain puts cyclists off.
3. **Log target:** the model added the same fixed rush-hour effect in January and June,
   but real peaks are small in winter and big in summer. With a log target, effects become
   multiplicative, so peaks scale with conditions. Predictions are converted back to bikes
   before evaluation.

| Version (validation set) | RMSE | Rush-hour RMSE |
|---|---|---|
| Round 1: basic features | 414.7 | 705.1 |
| + temperature² + is_raining | 397.8 | 661.3 |
| + log target (final) | 358.1 | 461.1 |

The log target lost when I first tried it, before round 2 (validation RMSE 442.9 vs
414.6). It only helped once the temperature and rain problems were fixed.

## Modeling: three implementations, one model

**Model:** linear regression, because the target (rentals per hour) is a continuous
count. The same model is trained three ways, on the same split and the same features,
minimising the same loss (mean squared error on log rentals):

- **scikit-learn:** `LinearRegression`, which solves for the best coefficients exactly.
- **Manual PyTorch:** raw tensors `w` and `b` starting at zero, autograd for the
  gradients (`loss.backward()`), and a hand-written update step inside
  `torch.no_grad()`, resetting the gradients with `.grad.zero_()` after each step.
- **Standard PyTorch:** `nn.Linear`, `nn.MSELoss` and `torch.optim.SGD`, also starting at
  zero. `optimizer.step()` and `optimizer.zero_grad()` do what I did by hand above. I used
  SGD rather than Adam so both PyTorch versions use exactly the same update rule, and only
  the API differs.

Both PyTorch versions use full-batch gradient descent, learning rate 0.1, 1,000 epochs.
No regularisation: I tested Ridge with alpha from 0.1 to 100 on the round-1 features and
validation RMSE barely changed (414.6–418.6), as expected with 6,528 rows and only
54 features at that point (56 in the final version).

### Results (in bikes)

| Validation (Sep) | RMSE | Rush-hour RMSE | MAE | R² |
|---|---|---|---|---|
| scikit-learn | 358.1 | 461.1 | 260.8 | 0.735 |
| PyTorch manual | 358.1 | 461.1 | 260.8 | 0.735 |
| PyTorch standard | 358.1 | 461.1 | 260.8 | 0.735 |
| Naive baseline (train mean) | 807.0 | 1394.6 | 631.5 | −0.344 |

| Test (Oct–Nov) | RMSE | Rush-hour RMSE | MAE | R² |
|---|---|---|---|---|
| scikit-learn | 408.3 | 640.1 | 324.9 | 0.471 |
| PyTorch manual | 408.3 | 640.1 | 324.9 | 0.471 |
| PyTorch standard | 408.3 | 640.1 | 324.9 | 0.471 |
| Naive baseline (train mean) | 589.1 | 954.2 | 458.0 | −0.102 |

### Do the three agree?
Yes. Their test predictions differ by at most about 0.23 bikes. The small difference
exists because PyTorch uses float32 (less precise numbers) and stops after 1,000 gradient
steps, while scikit-learn solves the problem exactly in one go. The manual and standard
PyTorch versions are practically identical, because they do the same calculation.

### Validation vs test
On test, the final model is worse than on validation, and also worse than the version
without the log target (test RMSE 332.7, rush-hour 527.4). Both versions under-predict in
autumn: on dry weekdays at 10–18°C, October averaged about 1,100 rentals per hour, against
about 700–1,000 in spring at the same temperatures. This could be the system growing
during 2018 or autumn riding habits; one year of data can't separate the two. The log
version under-predicts more, because converting back from logs tends to land on the low
side. I kept the log target because the decision was made on validation. Switching
because of the test result would mean choosing with the test set, and the test score
would no longer be an honest estimate.

## Limitations & next steps

- **The model systematically predicts too few bikes.** Without a buffer, it under-predicts
  73.6% of validation hours and 92.5% of test hours; on test, a buffer of about +60% is
  needed to get shortages below half of the hours. Next steps: with more years of data,
  add a feature for long-term growth and train on the most recent months; correct the
  low bias from converting back from logs. Until then, the operator should use the
  safety buffer.
- **Validation is only one month.** September alone couldn't reveal the autumn problem.
  With more data, I would validate on several months in turn instead of one.
- **Actual weather instead of forecasts.** In real use, the inputs would be next-day
  weather forecasts, which are less accurate, so real errors would be larger. With
  historical forecast data, I would train and evaluate on forecasts instead.
- **Citywide only.** The model can't say which stations need bikes, so it can't directly
  plan redistribution routes. That would need station-level data.
- **Only one year.** Seasonal patterns appear once, so the model can't learn whether they
  repeat or how they change from year to year.

## Generative AI use disclosure

I used Claude (Anthropic) throughout the assignment, and Cursor as my code editor:

- **Setup:** step-by-step help with Git, GitHub, uv and Cursor.
- **Code:** Claude wrote the code for `train.py`, `app.py` and the exploration plots,
  implementing the decisions described in this report.
- **Decisions:** Claude explained the options and trade-offs at each step, pointed out
  issues (for example, that dew point is calculated from temperature and humidity, and
  that the training period contains no autumn), and recommended some choices, such as the
  split periods and the metrics. I evaluated these and decided what to use. The
  observations from the plots, the business framing, and the decisions on non-functioning
  hours, dew point, the hour × day-type interaction, holidays, and the round-2 features
  are mine.
- **Report and notes:** Claude helped with wording and formatting, based on my notes and
  results.
