# Assignment 1 Report

- **Name**: Sebastián Delgado-Aparicio
- **Student ID**: 20013
- **Email**: sdelgadoapar.ieu2022@student.ie.edu
- **Group**: BBADBA 5B 

## Dataset

**UCI Bike Sharing Dataset, hourly file** (`data/hour.csv`), from Capital
Bikeshare in Washington D.C. (Fanaee-T & Gama, 2013; UCI ML Repository
dataset 275).

- **One row = one hour** of the bike-share system: how many bikes were
  rented in that hour (`cnt`), plus the calendar (date, hour, month, year,
  holiday, working day) and the weather in that hour (situation category,
  normalised temperature, "feels-like" temperature, humidity, wind speed).
- **17,379 rows × 17 columns**, covering 1 Jan 2011 to 31 Dec 2012. Some
  hours are missing from the file (system outages), so it's not a perfect
  2 × 8,760 grid.
- `cnt` = `casual` + `registered`: the two user types add up exactly to
  the target.

**Why this dataset.** Hourly demand has a lot of structure that a linear
model can't see in the raw columns (two commute peaks on working days, a
single afternoon hump at weekends, a non-linear temperature effect, strong
growth from 2011 to 2012), so feature engineering visibly matters. It's
also a time series, which makes the split decision a real one.

## Business / real-life framing

**Scenario.** Capital Bikeshare's operations team plans the coming week:
how many rebalancing-truck shifts and field staff to schedule each hour, and
where to stage bikes before the peaks. They need a forecast of **total
rentals per hour, up to one week ahead**. Under-forecasting leaves stations
empty at the commute peak (lost rides, unhappy members); over-forecasting
pays for truck and staff hours that aren't needed.

**What this framing decided in the pipeline:**

1. **Target = rentals per hour (`cnt`), modelled as log(1 + cnt).** The
   operator plans in bikes, so all metrics are reported in bikes/hour after
   converting predictions back with expm1. The model is fitted on the log
   scale because demand effects are multiplicative (rain cuts demand by a
   *percentage*, not by a fixed number of bikes) and the raw count is
   heavily right-skewed.
2. **Only information available a week in advance may be a feature.**
   - `casual` and `registered` are dropped: they sum to the target, so
     using them would be leakage. They're also only known after the hour
     happens.
   - "Same hour last week" (168 hours earlier) is allowed, because when you
     forecast up to 7 days ahead that value is always already observed.
     "Same hour yesterday" would not be safe for days 2-7 of the week, so
     it isn't used.
   - Weather: the dataset has *observed* weather. The operator would use a
     *forecast* of it instead. I use observed weather as a stand-in for a
     good weather forecast, which makes my errors slightly optimistic (see
     Limitations).
3. **Time-based split, not random.** A random split would put, for example,
   Wednesday 18:00 in the test set and Wednesday 17:00 and 19:00 in
   training. The model would effectively interpolate between known
   neighbours, and it would see 2012's higher demand level while
   "forecasting" 2012. Both make test scores look better than real
   forecasting would be. So the data is split chronologically:

   | Split | Period | Rows |
   |---|---|---|
   | Train | 8 Jan 2011 – 30 Jun 2012 | 12,738 |
   | Validation | 1 Jul – 30 Sep 2012 | 2,208 |
   | Test | 1 Oct – 31 Dec 2012 | 2,129 |

   Validation is used to choose the regularisation strength, the learning
   rate and the features. The final models are then **refit on train +
   validation**, because with a fast-growing system the most recent months
   are the most informative, and evaluated **once** on the test quarter.
4. **Metric = MAE in bikes/hour.** Staffing and truck plans scale roughly
   linearly with how many bikes you're off by, so the average absolute miss
   is the number the operator cares about and is easy to communicate ("on
   average we're off by ~46 bikes in an hour"). RMSE is reported alongside
   it because it punishes big misses at peak hours more, which is when
   stock-outs happen. R² is reported for context only.
5. **Baselines an operator would actually use without a model**: (a) "same
   hour last week", and (b) the historical average for that hour on that
   type of day (working vs. non-working).

## Data preparation & feature engineering

All feature code is in `features.py`, which both `train.py` and `app.py`
import, so the app builds inputs exactly as training did.

**Data-quality and leakage decisions**

- Dropped `casual`, `registered` (leakage) and `instant` (a row id).
- Dropped `atemp` ("feels-like" temperature). It's about 0.99 correlated
  with `temp`, so the two would split one effect between two unstable
  coefficients.
- Dropped `season`, because it's a coarser version of `mnth`. `weekday` is
  also dropped, since the day-type split below captures the difference that
  matters (working vs. not) and the rest is noise for this model.
- Weather category 4 (heavy rain/snow) appears only 3 times in 17k hours.
  That's too few to estimate its own coefficient, so it's merged into
  category 3 ("bad weather").
- The lag feature is looked up **by timestamp**, not by shifting 168 rows.
  The file has gaps, so a row shift would point at the wrong hour after
  every outage. Rows with no value a week earlier (the first week of 2011
  plus the week after each gap) are dropped: 304 rows, under 2% of the
  data. That's more honest than imputing them.

**Engineered features (67 in total), and why each one exists**

| Feature | Why |
|---|---|
| Hour of day × day type: 23 indicators for working days + 23 for non-working days, plus a `workingday` shift | Working days have two commute peaks (08:00 and 17-18:00). Weekends and holidays have one broad afternoon hump. A single "hour" number can't express either shape, and even shared hour dummies force both day types onto the same curve. |
| Month indicators (11) | Seasonality isn't a straight line in month number: December isn't "more" than January. |
| `year_2012` | The system grew roughly 65% from 2011 to 2012. |
| Weather indicators (mist, bad) | Weather categories aren't evenly spaced numbers. |
| `temp`, `temp_sq` | Demand rises steeply from cold to mild and then flattens (and dips slightly) in the hottest hours. The squared term lets a linear model bend. |
| `humidity`, `windspeed` | Used as is. |
| `log_cnt_lag_week` | Same hour last week, on the log scale like the target. It carries the recent level of the system (growth, events) that the calendar can't. |
| `holiday` | Public holidays behave like weekends, with some differences of their own. |

All features are **standardised** using means and standard deviations
computed on the fitting data only. This is required for gradient descent:
with features on very different scales, one learning rate is too big for
some weights and too small for others. It also makes coefficients
comparable in size.

**Feature ablation: does each step earn its place?** Each step is added in
turn, fitted on train, scored on validation, using plain least squares:

| Step | # features | Val MAE | Val RMSE | Val R² |
|---|---|---|---|---|
| 1. Raw columns as numbers, raw target | 9 | 157.8 | 195.9 | 0.28 |
| 2. + hour / month / weather as categories | 42 | 111.0 | 145.2 | 0.61 |
| 3. + separate hour profile per day type | 65 | 86.6 | 112.7 | 0.76 |
| 4. + log(1+cnt) target | 65 | 63.5 | 103.5 | 0.80 |
| 5. + temperature squared | 66 | 49.4 | 78.8 | 0.88 |
| 6. + same hour last week (final) | 67 | 44.9 | 72.3 | 0.90 |

**Draft Analysis:**
> Adding the features one at a time showed me that most of the improvement came
from feature engineering, not from the model itself. With the raw columns the
error was 158 bikes per hour, and with all my features it went down to 45, a
reduction of more than 70% using exactly the same linear regression.

The biggest jump came from treating the hour as a category and giving working
days and weekends their own hour profile. Before that, the model could not
represent the two commute peaks at 8:00 and 17:00-18:00, because a single
"hour" number can only make demand go up or down in a straight line.

Using log(1 + rentals) as the target also helped a lot (from 87 to 64), because
things like rain or cold reduce demand by a percentage rather than by a fixed
number of bikes, and on the log scale those effects become additive, which is
what a linear model can handle.

What surprised me most was temperature squared, which lowered the error by
another 14 bikes per hour. Demand grows quickly from cold to mild weather and
then levels off, and the squared term lets the model follow that curve. The
lag feature (same hour last week) only improved the error by about 4.5,
probably because the year feature already captures most of the growth of the
system between 2011 and 2012.

## Modeling: three implementations, one model

**Model: linear regression (with an optional ridge penalty) on log(1 +
cnt).** The target is a continuous count, so this is regression, not
classification. All three implementations minimise the same objective on
the same standardised features and the same split:

    mean((y − Xw − b)²) + (α / n) · ||w||²

scikit-learn's `Ridge` minimises `sum(...) + α||w||²`. Dividing by *n*
doesn't move the minimum, so the PyTorch versions use the mean form with
penalty α/n. The intercept *b* is not penalised in any of them.

1. **scikit-learn** `Ridge`: closed-form solution.
2. **Manual PyTorch loop**: raw tensors `w` and `b` with
   `requires_grad=True`. Each step computes the loss, calls `.backward()`,
   then updates `w -= lr * w.grad` inside `torch.no_grad()` and zeroes the
   gradients.
3. **Standard PyTorch**: an `nn.Module` wrapping `nn.Linear`, with
   `nn.MSELoss` plus the same explicit L2 term, and `torch.optim.SGD`. I
   deliberately did *not* use SGD's `weight_decay`: it would also shrink
   the bias and uses a different scaling, so the objective would no longer
   match scikit-learn's.

Both PyTorch versions start from all-zero weights and use **full-batch**
gradient descent (all ~15k rows every step, 3,000 steps). Mini-batches
would add noise to every step, and the two versions would no longer be
comparable step for step. The dataset is small enough that full-batch is
fast (a few seconds).

**Tuning (on validation only)**

- Ridge α ∈ {0, 1, 10, 100, 1000}: validation MAE was 44.94, 44.95, 45.07,
  46.54 and 51.50. **α = 0 was best**, so the final model is effectively
  ordinary least squares.
- Learning rate ∈ {0.01, 0.03, 0.1, 0.3, 1.0}: 0.01 and 0.03 hadn't fully
  converged in 3,000 steps (validation MAE 48.2 and 45.5). 0.1 and 0.3
  reached the same optimum (44.94). **1.0 diverged** to NaN, because the
  step overshoots the minimum and grows every iteration. **lr = 0.3** was
  chosen.

**Results on the test quarter (Oct–Dec 2012), bikes/hour**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline: same hour last week | 70.60 | 117.71 | 0.660 |
| Baseline: hour × day-type average | 79.91 | 119.62 | 0.649 |
| scikit-learn Ridge | **45.83** | **73.02** | **0.869** |
| PyTorch manual loop | 45.83 | 73.02 | 0.869 |
| PyTorch nn.Module + optim | 45.83 | 73.02 | 0.869 |

**Do the three agree?** Yes, to numerical precision:

- The largest coefficient difference between any two implementations is
  2 × 10⁻⁶.
- The largest prediction difference on the test set is 0.001 bikes/hour.

The remaining differences come from PyTorch working in 32-bit floats while
scikit-learn uses 64-bit, plus gradient descent stopping a hair short of
the exact optimum. The two PyTorch versions follow the same path (same
start, same full-batch gradients, same update rule), as the overlapping
loss curves in the dashboard show.

> **Draft analysis:**
> My model reduced the error of the best simple forecast ("same hour last week")
from 70.6 to 45.8 bikes per hour, about 35% better.

The three implementations ended up with practically identical results: the
coefficients differ by less than 0.000002 and the predictions by less than
0.001 bikes. This is what I expected, because linear regression with squared
error has only one minimum. scikit-learn calculates it directly, while the two
PyTorch versions walk towards it step by step with gradient descent, so as long
as they converge they have to arrive at the same place. The tiny differences
that remain come from PyTorch using 32-bit numbers instead of 64-bit, and from
gradient descent stopping very slightly before the exact minimum.

To get there, the learning rate mattered. With 0.01 the model had not finished
learning after 3,000 steps, while with 1.0 the steps were too big and the loss
exploded to infinity. I chose 0.3, which converged quickly and safely.

Regularisation did not help: the best alpha was 0, so the final model is plain
linear regression. With around 15,000 rows and only 67 features the model has
plenty of data and is not overfitting. Its remaining errors come from
situations it has no information about, like holidays or storms, and shrinking
the coefficients only made the predictions worse.

## Limitations & next steps

- **Holidays and unusual events cause the biggest errors.** The worst days in
  the test period were Black Friday (23 Nov), when the model predicted too many
  rentals because the dataset marks it as a normal working day, Hurricane Sandy
  (30 Oct), when the system was closed, Thanksgiving and Christmas. To fix this
  I would add a feature that flags special days such as the day after
  Thanksgiving and the Christmas week.

- **One abnormal week can mislead the next one.** On 6 November, exactly one
  week after Hurricane Sandy, the model predicted about 91 bikes per hour too
  few, because its "same hour last week" value came from the storm day. Using
  the median of the last 2-4 weeks instead of only last week would make this
  feature less sensitive to one bad week.

- **I used the real weather instead of a weather forecast.** In practice the
  operator only knows the forecast when planning the week, so the real error
  would be somewhat higher than mine, especially 5-7 days ahead. With more data
  I would train the model on archived weather forecasts.

- **Peak hours are the hardest to predict.** The error is about 98 bikes per
  hour at 8:00 and 17:00-18:00, double the overall average of 46, and these are
  exactly the hours when empty stations matter most. I could give peak hours
  more weight during training or build a separate model for them.

- **Predictions are slightly too low on average.** Because I convert the
  predictions back from the log scale, the model predicts around 5 bikes per
  hour less than the real average (215 vs. 220). A standard correction, called
  smearing, could remove this bias.

- **Limited data.** The dataset only covers two years and the total of the
  whole system. The year feature cannot predict growth in 2013, so in practice
  the model would need to be retrained regularly, and an operator would really
  need forecasts per station, which this dataset does not include.

## Generative AI use disclosure

I used Claude (Anthropic) through Claude Code as a coding assistant & teacher. I chose
the dataset and the business framing by myself, but Claude helped write
the code for the pipeline (`features.py`, `train.py`) and the Gradio
dashboard (`app.py`). I then went through every part of the code with it to understand,
step by step, so that I can explain each decision, and I rewrote the
analysis and conclusions in my own words. 
