# Assignment 1 Report

- **Name**: Raad Abu Dail
- **Student ID**: 18940
- **Email**: Rabdulhameed.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

The California Housing Prices data set is based on 1990 U.S. Census data. 
Each row is a census block group (district), around 600 to 3,000 individuals 
in size, not an individual home. . One row represents a census block group
(district), which is roughly 600-3,000 people, not an individual house. This
matters because it rules out using the model for single-home appraisal and shapes 
the business framing below toward a district/region level decision instead.

- **20,640 rows, 10 columns** originally: `longitude`, `latitude`,
  `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`,
  `households`, `median_income`, `median_house_value` (target), and
  `ocean_proximity` (categorical).
- **19,675 rows** after removing price-capped districts.

This data set was selected because it is easy to understand, does not have any 
individual information of people (it uses census information on districts), 
and even though it is one of the classic data sets, it has issues in terms
of quality of data (there are missing values, censored target variable, and 
a highly skewed category).

## Business / real-life framing

**Scenario**: A real estate investment group needs a quick approximation of 
the average house price of a California district, given demographic and 
geographic variables, in order to evaluate acquisition prospects and assess
local mortgage pricing without having to wait for an actual market evaluation.
The prediction unit is a district, which is accurate to the level of the 
data collected.

This framing drove three pipeline decisions:

1. **Target definition.** Approximately 4.8 percent of the districts have a 
   `median_house_value` value of $500,001, which is due to a reporting limit set 
   in the original census data gathering process. Therefore, this is not an 
   actual value but rather a cap that should be avoided when training the model.
2. **Split strategy.** This is a single snapshot in time (1990 census) with
   no time axis, which means that the time split will not divide past from 
   future. So, a random split makes more sense than a time split. However, 
   I used stratified split based on `ocean_proximity` feature because the least
   frequent class might disappear randomly.
3. **Decision metric.** An investment-screening audience thinks in dollars,
   not R². I report MAE in dollars as the headline metric throughout as it 
   is what actually informs decision making in such scenario.

## Data preparation & feature engineering

- Removed price-capped rows: 965 out of 20,640 rows (4.7%) had 
median_house_value equal to $500,001 and were removed before any splitting.
- Missing data: total_bedrooms had 207 missing values. Filled with the median 
value, calculated only on the training split (not on validation/test splits or 
the entire dataset, to avoid information leakage.)
- Split: 64% training / 16% validation / 20% testing, stratified by 
ocean_proximity. I confirmed verification worked correctly: the rare ISLAND 
category (only 5 rows overall) was properly distributed 3/1/1 among 
train/validation/test, not 5/0/0 which could happen due to sampling.
- Categorical encoding: ocean_proximity was one-hot encoded, since its 5 
categories lack natural ordering, with the encoder trained only on the training
data.
- Feature engineering:
  - rooms_per_household, bedrooms_per_room, population_per_household: district 
  totals provide little insight without considering the household size (5,000 
  rooms can be an indicator of a crowded area or a sparsely-populated one).
  - dist_to_nearest_city_km: haversine distance from each district to the 
  nearest of 5 main California population centers. Proximity to hubs is known 
  to affect house prices, but is not accounted for by latitude/longitude data.
- Skew correction: log1p transformation to total_rooms, total_bedrooms, 
population, households and median_income, due to strong right skew, that would
otherwise enable just a few districts to dominate a linear fit.
- Scaling: All numerical features standardized with the scaler trained only on
the training data. Standardization is required for convergence of the two 
gradient descent methods within a reasonable number of iterations and is 
performed identically in case of scikit-learn method, ensuring that all 
three methods learn from the same feature matrix.
- Target: intentionally not transformed to logarithm to keep the dashboard 
predictions in dollars without a need to perform back-transformation afterwards.
- Target scaling for the two PyTorch methods only: targets in raw dollars 
(around $200k) create too large gradients for plain gradient descent. As a 
result, target was standardized only in these two cases, with predictions 
inverse-transformed to dollars before metric computation. The solver in 
scikit-learn's method is not gradient descent-based and works well with the 
raw dollar scale, ulitmately learning on the untransformed target.

## Modeling

Linear regression (target is a continuous dollar value), with L2
regularization, implemented three ways on the identical preprocessed
train/val/test split:

1. **scikit-learn** `Ridge`, regularization strength (`alpha`) tuned via a 
7-value grid search on validation MAE → best `alpha = 0.01`.
2. **Manual PyTorch loop**: raw tensors, `w`/`b` as leaf tensors with 
`requires_grad=True`, loss computed by hand (MSE + an explicit L2 penalty term),
gradients from `.backward()`, parameters updated manually inside 
`torch.no_grad()`, gradients zeroed by hand each step (no `nn.Module`, no 
`torch.optim`). Learning rate and regularization strength jointly grid-searched
(12 combinations, short runs) on validation best MAE `lr = 0.1`, `lam = 0.0`.
3. **Standard PyTorch**: a single `nn.Linear` layer trained with 
`torch.optim.SGD`. Reuses the manual loop's `(lr, lam)` directly rather than
re-searching, because both implementations compute the identical update 
mathematically. One thing I had to handle explicitly: PyTorch's `weight_decay`
adds `weight_decay * param` to the gradient, which corresponds to a loss penalty
of `(weight_decay/2) * sum(param**2)`. To match the manual loop's 
`lam * sum(w**2)` term exactly, I set `weight_decay = 2 * lam`, and used 
separate parameter groups so the bias term isn't penalized.

All three were tuned on validation MAE and reported on a held-out test set,
against a naive baseline:

| Method | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| Baseline (predict mean) | $78,782 | $98,109 | -0.000 |
| scikit-learn Ridge | $44,828 | $60,227 | 0.623 |
| Manual PyTorch loop | $44,828 | $60,228 | 0.623 |
| Standard PyTorch (`nn.Module` + `optim`) | $44,828 | $60,228 | 0.623 |

**Do the three agree?** Yes, all are consistent with MAE up to a few cents in 
both validation and test cases. It was expected and intended since the three 
tests correspond to the same model fit through three different methods, 
not three separate models. Therefore, almost identical results make up the 
proper result and at the same time validate the manual approach since if there
had been an error in the implementation, it could not have produced the same 
high degree of consistency with the nn.Module/optim approach. Each of the three
approaches independently determined a low level of regularization (alpha=0.01, 
lam=0.0). With 12,592 training samples and 17 predictors, there is no chance of
overfitting the data with such a simple linear model.

The model cuts MAE by roughly 43% versus the naive baseline ($78,782  to 
$44,828), and R² of 0.623 means it explains about 62% of the variance in
district home values from these features alone.

## Limitations & next steps

- Scope is district-level rather than home-level. This model produces an 
estimate of the typical value of a district, not the value of any particular 
property. It can neither serve for individual loan underwriting nor individual
home appraising, and is only meant to screen neighborhood. Fixing this issue 
would require parcel-level data, which are not available in this data set.
- Censored target variable. The exclusion of the 965 price-capped districts 
from the training results in missing training data above around $500k, hence 
the danger of underestimating the actual price in real high-valued districts 
due to lack of forecast foundation. In future work, we can use an uncensored 
and newer housing dataset to build the entire price range distribution model.
- Category 'ISLAND' is memorized rather than learned. Given the number of 
five rows total, the model is unable to learn any kind of generalized pattern 
about island districts, so, predictions on any such district must be considered
unreliable. To go forward, we need more data for the islands.
- The target variable is kept in raw dollars and not log-transformed for better
interpretability in the dashboard. This leads to heteroscedastic residual error,
In the future, it makes sense to build a log-target variant of this model and 
compare percentage error across price ranges.
- Unexplained variance. R²=0.623 means that about 38% of variance is 
unexplained, which might be explained by factors missing from this dataset.

## AI use disclosure

Claude was used for help with the data set choice, the business framing, and 
for code error debugging.
