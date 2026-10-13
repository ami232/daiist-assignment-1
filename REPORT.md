# Assignment 1 Report

- **Name**: João Paulo Prado
- **Student ID**: 19398
- **Email**: jtobar.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

- **Source:** Seoul Bike Sharing Demand, Kaggle
  (https://www.kaggle.com/datasets/kashyapankush/seoul-bike-sharing-data),
  committed as `data/SeoulBikeData.csv`.
- **One row** = one hour of public bike rentals in Seoul, with the weather
  and calendar conditions for that hour.
- **Size:** 8,760 rows × 14 columns, covering 1 Dec 2017 – 30 Nov 2018
  (24 rows per day). No missing values, no duplicate rows.
- **Target:** `Rented Bike Count` (hourly number of bikes rented).
- **Why this dataset:** it captures hourly bike-share demand together with the
  conditions (time, weather, season, holidays) that plausibly drive it, which
  is the same problem I want to address for BiciMAD.


## Business / real-life framing

- **Scenario:** the Madrid city government uses this model to better predict
  hourly demand for BiciMAD, its government-funded public bike rental scheme.
  As a frequent BiciMAD user, I have noticed that at certain times of day
  there are not enough bikes available across the city; better demand
  predictions would let the city plan bike availability.
- **Target definition:** the hourly rented bike count, predicted from the
  available variables (time, temperature, rain, etc.). Because the target is
  a continuous count, the model is a **linear regression**.
- **Split strategy:** a **random** split rather than a time-based split.
  With only one year of data, a time-based split would leave some seasons
  only in training (e.g. winter) and others only in test (e.g. summer), so
  the model would be evaluated on conditions it never saw. A random split
  lets both sets cover the full range of every variable.
  - The random split is done by **whole days**, not individual hours: every
    hour of a given day goes to the same set. Neighbouring hours on the same
    day are very similar, so splitting hour-by-hour would let the model see
    8:00 and 10:00 in training and be tested on 9:00, which makes test
    scores look better than they would be for predicting a genuinely new
    day. Splitting by day keeps all seasons in every set while testing on
    complete days the model has never seen.
  - Days are split into train / validation / test. The validation set is
    used only for the small amount of tuning allowed (regularisation
    strength, learning rate); the test set is used once for the final
    comparison.
- **Evaluation metric:** **MAE (mean absolute error)** in bikes per hour —
  the average number of bikes the prediction is off by, which is directly
  interpretable for the city when planning bike availability.

## Data preparation & feature engineering

- **Non-functioning days removed:** all rows with `Functioning Day = No`
  (295 rows) are dropped. On those days bikes were not available to rent,
  so they contain no information about demand. We then drop the column entirely since only the `Functioning Day = Yes` entries remain.
- **Hour of day:** I expect fewer rentals late at night and early in the
  morning, and more during the morning and afternoon, so the hour is a key
  feature.
  - A raw `Hour` column (0–23) would only let linear regression fit one
    slope ("demand rises/falls steadily through the day"), which cannot
    represent this pattern.
  - Hour is therefore encoded **cyclically with sin/cos**:
    `hour_sin = sin(2π·hour/24)`, `hour_cos = cos(2π·hour/24)`. This maps
    23:00 next to 00:00 on a circle, preserving the fact that the end of one
    day flows into the start of the next — information that one-hot
    encoding would lose, since it treats every hour as an unrelated
    category.
  - **Two sin/cos pairs.** One pair can only draw a single smooth wave per
    day (one peak, one low point). The EDA shows weekdays have two peaks —
    8:00 (people going to work) and 18:00 (people leaving work) — so a second
    pair at twice the frequency, `sin(4π·hour/24)` and `cos(4π·hour/24)`, is
    added to allow two peaks per day.
- **Weekday vs. weekend:** a `weekend` flag (Saturday/Sunday = 1) derived from
  `Date`, since weekends could have more bike rentals and the daily pattern
  differ from working days (commuting).
- **Weekend × hour interactions.** In the EDA, weekends follow a smoother
  daily curve without the two commuting peaks seen on weekdays. On their own, the hour terms and the weekend flag can only give weekends a different *level*, not a
  different *shape* through the day. The two ideas do different jobs (two
  pairs = the curve can have two peaks; interactions = weekdays and weekends
  can have different curves), so both are used: each of the four hour terms
  is multiplied by the weekend flag (`weekend_x_hour_sin1`, ...).
- **Weather:** I expect rainfall, snowfall and lower temperatures to reduce
  rentals.
- **Holidays:** I expect holidays to increase rentals. We use a binary flag
  to record this.
- **Weather variables kept: temperature, solar radiation, rainfall,
  snowfall. Dropped: humidity, wind speed, dew point, visibility.** Decided
  after a correlation heatmap in the EDA (`eda.ipynb`, also
  shown in the dashboard).
- **Seasons:** seasons affect demand and are categorical, so they are
  one-hot encoded.
  - One season (Autumn) is left out as the reference category: with an
    intercept in the model, all four dummy columns would always sum to 1 and
    be perfectly collinear with it.
- **Scaling:** all 17 features are standardised (mean 0, std 1) with
  statistics computed on the **training** days only, so no information from
  validation/test leaks in. Standardising also lets gradient descent in the
  two PyTorch versions use one learning rate for every feature, and makes
  the coefficients comparable in size.
- **Final feature set (17):** `hour_sin1, hour_cos1, hour_sin2, hour_cos2,
  weekend, weekend_x_hour_sin1, weekend_x_hour_cos1, weekend_x_hour_sin2,
  weekend_x_hour_cos2, holiday, temp, solar, rain, snow, season_spring,
  season_summer, season_winter`.
- **Code organisation:** the cleaning and feature code lives in
  `features.py` and is imported by `eda.ipynb`, `train.ipynb` and `app.ipynb`, so
  the dashboard's what-if predictions are built exactly the way the models
  were trained.
- **Target transformation: the model predicts `log(1 + count)`**, and
  predictions are converted back to bikes with `exp(prediction) − 1` before
  computing MAE, so all reported errors are in bikes per hour. Reasons:
  1. *No impossible predictions.* A linear model on the raw count can
     predict negative bike rentals; after the back-transform, predictions
     can never go below 0.
  2. *Effects act as percentages, not fixed amounts.* On the raw scale, a
     linear model says e.g. "rain removes 300 bikes" at every hour — at
     4:00, when ~150 bikes are rented, that is impossible. On the log scale,
     each feature multiplies demand by a factor ("rain cuts demand by X%"),
     which scales sensibly between quiet and busy hours.
  3. *Skew.* The raw count is right-skewed (skewness 1.14 after cleaning:
     most hours are low, a few are very high); on the log scale it is
     -0.77. A few very busy hours therefore pull the fit less.

## Modeling: three implementations, one model

- **Model:** linear regression, because the target (hourly bike count) is
  continuous.
- **Split:** 247 train days (5,921 hours), 52 validation days (1,248 hours),
  54 test days (1,296 hours). All four seasons appear in every set.
- **Tuning (validation MAE, bikes per hour):**
  - Ridge α: 0.01 → 232.76, 0.1 → 232.76, 1 → 232.76, 10 → 232.82,
    100 → 233.52, 1000 → 246.83. Chosen: α = 0.01.
  - Learning rate (both PyTorch versions, 3,000 full-batch epochs): 0.01,
    0.03, 0.1 and 0.3 all → 232.76. Chosen: 0.3.
- **How the three are made comparable:** all three minimise the same
  objective — squared error on `log(1 + count)` plus an L2 (Ridge) penalty
  of the same strength on the weights, not the intercept. scikit-learn's
  `Ridge` solves it exactly. The two PyTorch versions use full-batch
  gradient descent from zero initial weights, with the penalty rescaled to
  `α/n` because they average the squared error instead of summing it.
  - Manual loop: raw tensors `w`, `b` with `requires_grad=True`,
    `loss.backward()`, then `w -= lr * w.grad` inside `torch.no_grad()`,
    and the gradients are zeroed by hand.
  - Standard: an `nn.Module` wrapping `nn.Linear(17, 1)`, `nn.MSELoss`, and
    `torch.optim.SGD`. The L2 term is added to the loss by hand rather than
    through SGD's `weight_decay`, because `weight_decay` would also penalise
    the bias and uses a different scale.
- **Baselines:** (1) predict the training median count for every hour;
  (2) predict the training median for each hour-of-day × weekend
  combination, which is a lookup table with no weather information.

**Test set (54 unseen days), errors in bikes per hour:**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline: overall median | 505.1 | 652.1 | −0.033 |
| Baseline: hour × weekend median | 423.0 | 544.1 | 0.281 |
| scikit-learn (Ridge) | 241.7 | 367.6 | 0.672 |
| Manual PyTorch loop | 241.7 | 367.6 | 0.672 |
| Standard PyTorch | 241.7 | 367.6 | 0.672 |

**Agreement:** largest coefficient difference from scikit-learn is below
0.000001 for both PyTorch versions. Largest prediction difference on the test
set is 0.0008 bikes (manual) and 0.0027 bikes (standard).

Yes, the three models agree. This happens because they minimise the same
objective: even though they use different methods, they all converge to the
same values. The baseline serves as a comparison: it does not use any
features, while the models include all of the weather and context
features.

## Limitations & next steps

- RMSE (368) is much larger than MAE (242), so a minority of hours carry
  large errors; these are mostly during peak hours. One way to fix this
  would be to add a more flexible hour encoding.
- The error histogram has a longer tail on the negative side, meaning big
  under-predictions. One way to solve this would be to penalise
  under-predictions more.
- This is exactly the problem identified in the business framing, showing
  that there is clearly room for improvement in BiciMAD's bike planning.

## Generative AI use disclosure

- Claude Code (Anthropic, Claude Opus 5.5) was used to write all of the
  code: data loading and cleaning (`features.py`), the EDA notebook, the
  training pipeline with the three model implementations, and the Gradio
  dashboard. It was also used to format this report.
- The business framing, the choice of dataset, target and metric, the
  decision to drop non-functioning days, the cyclical hour encoding, the
  choice of weather features (from the EDA heatmap), and the season and
  holiday encodings are my own decisions.
- Some design options were proposed by the assistant, and I reviewed and
  adopted them: splitting randomly by whole days rather than by hours, a
  second sin/cos pair and the weekend × hour interactions, the
  `log(1 + count)` target, standardisation, and the second baseline.
  The rationale for each is written above, and I can explain each one.
- The interpretation of the results and the limitations are my own.
