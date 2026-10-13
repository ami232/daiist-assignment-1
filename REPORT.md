# Assignment 1 Report



- **Name**: Miguel Gordon Alonso
- **Student ID**: 20120
- **Email**: mgordon.ieu2022@student.ie.edu
- **Group**: BBADBA 5B 

## Dataset

The Bike Sharing dataset was chosen since it provides a realistic business forecasting 
problem in which time dependencies play the main role: demand changes periodically by hour, 
weekday, season but depends on weather conditions and other unpredictable factors. This 
dataset is an exception to most datasets that are often toy-like and don't involve calendar 
features and continuous exogenous variables that are usually used in industry forecasting. 
Another important feature was that it was a regression problem in which the target variable 
was not only skewed but right-censored.


Facts (from the data and `data/README_dataset.txt`):

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository, Bike Sharing Dataset (`hour.csv`) |
| Citation | Fanaee-T, H. and Gama, J. (2013), "Event labeling combining ensemble detectors and background knowledge", *Progress in Artificial Intelligence*, doi:10.1007/s13748-013-0040-3 |
| One row | Total rentals in one hour across the Capital Bikeshare system (Washington D.C.), with calendar and weather data |
| Size | 17,379 rows x 17 columns; 2011-01-01 to 2012-12-31; no missing values; 165 hours absent from the file |
| Target | `cnt`, rentals per hour (min 1, median 142, max 977) |


## Business / real-life framing
Capital Bikeshare operations managers use this forecast each evening to plan overnight bike rebalancing for the next day. The decision: how many bikes to move to each station to prevent shortages during peak hours (8–9 AM, 5–7 PM) while avoiding over-provisioning.

Pipeline decisions justified:

Temporal split (train 2011–Jun 2012, val Jul–Sep 2012, test Oct–Dec 2012): Year-over-year growth means training on 2011–mid-2012 and testing on Oct–Dec 2012 avoids data leakage. Training on 2011 alone predicts 149 bikes/hour for 2012 when actual is 235—a 36% underforecast revealing that random splits would memorize future growth rather than learn seasonal patterns.

Removed casual and registered: They sum exactly to cnt—perfect leakage. The model would memorize the identity rather than learn to forecast from weather and date.

No lag features: Rebalancing is planned each evening using only calendar and weather data available at decision time. Lags would add complexity and hide poor generalization.

Log-transformed target: Rentals are right-skewed (most hours 50–200 bikes, rare hours 800+). Log transformation gives equal weight to predicting quiet and busy hours fairly.

MAE metric: Operations care about typical forecast error across all hours, not extreme misses. MAE is easier to explain: "on average, we're off by 53 bikes per hour."

| Decision in the code | Where | Value |
|---|---|---|
| Split | `pipeline.py` | Train 2011-01-01 to 2012-06-30 (13,003 rows), validation Jul-Sep 2012 (2,208), test Oct-Dec 2012 (2,168). Final models refit on train + val |
| Leaky columns removed | `pipeline.py` | `casual` and `registered` (they sum exactly to `cnt` in every row) |
| No lag features | `pipeline.py` | Only calendar and weather inputs are used |
| Target | `pipeline.py` | Model trained on log(1 + cnt); all metrics reported in bikes per hour |
| Primary metric | `train.py`, `PRIMARY_METRIC` | MAE (RMSE and R² also reported) |

Evidence available for the split: training on 2011 only and predicting
2012 gives a predicted mean of 149 rentals/hour against an actual 235.

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way.*

Data-quality findings:

| Finding | Action in `pipeline.py` |
|---|---|
| `casual + registered == cnt` in all rows | Both dropped (target leakage) |
| `hum == 0` in 22 rows, all on 2011-03-10 | Replaced by the training-set median humidity |
| `weathersit == 4` in only 3 rows | Merged into level 3 |
| `temp` and `atemp` correlation 0.988 | `atemp` dropped |
| The 48 hour x day-type indicators always sum to 1 (collinear with the intercept) | Reference cell `h00_off` dropped, so the solution is unique |

Ablation on validation (`uv run python ablation.py`; fit on train only, one
change at a time):

| Variant | Features | Val MAE | Val RMSE | MAE change vs final |
|---|---|---|---|---|
| Final pipeline | 56 | 53.11 | 83.74 | 0.00 |
| Target in raw bikes instead of log(1+cnt) | 56 | 91.58 | 116.29 | +38.47 |
| Hour x day type -> plain hour one-hot + workingday | 31 | 90.51 | 125.98 | +37.40 |
| Hour x day type -> numeric hr + workingday | 9 | 160.83 | 226.39 | +107.72 |
| Without temp² | 55 | 85.15 | 138.42 | +32.04 |
| Without cyclical month (sin/cos) | 54 | 54.01 | 83.47 | +0.90 |
| Without data cleaning (hum zeros, weathersit 4, atemp) | 57 | 52.96 | 83.51 | -0.15 |
| Plus numeric weekday (0-6) | 57 | 52.21 | 83.28 | -0.90 |

The log(1+cnt) target reflects the fact that being wrong by 100 rentals is worse on days 
with 50 total rentals compared to 500 rental days (ablation results in +38 MAE without it). 
Hour × day type (workday vs weekend) reflects different patterns: 8:00 is the peak hour for 
workdays but not weekends (+37 MAE without it). Temp² reflects the idea that extreme cold 
is a bigger deterrent than going from 10 to 15 degrees (+32 MAE without it). Sinusoidal 
representation of the month avoids the discontinuity between December and January (+0.9 MAE 
without it). We retained humidity and weathersit cleaning despite the lack of improvement 
in the metric as they are actual data errors.

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

Validation tuning (fit on train, scored on Jul-Sep 2012):

| Setting | Values tried | Val MAE | Chosen |
|---|---|---|---|
| Regularisation alpha (sklearn) | 0 / 1 / 10 / 100 | 53.11 / 53.15 / 53.54 / 57.80 | 0 |
| Learning rate (PyTorch, 3000 full-batch epochs) | 0.01 / 0.1 / 0.3 / 0.5 | 60.38 / 53.11 / 53.11 / diverged | 0.3 |

Test set, Oct-Dec 2012, evaluated once:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| scikit-learn | 59.58 | 87.85 | 0.810 |
| Manual PyTorch | 59.58 | 87.85 | 0.810 |
| Standard PyTorch | 59.58 | 87.85 | 0.810 |
| Baseline: training mean | 156.87 | 204.43 | -0.028 |
| Baseline: hour x day-type profile | 79.95 | 119.59 | 0.648 |

Agreement with scikit-learn (final models):

| | Max coefficient difference | Max prediction difference (bikes) | Training MSE gap to optimum |
|---|---|---|---|
| Manual PyTorch | 9.0e-07 | 4.7e-04 | 7.8e-09 |
| Standard PyTorch | 7.7e-07 | 1.2e-03 | 7.8e-09 |

We chose linear regression because cnt is continuous with no decision threshold. All three 
methods (sklearn, PyTorch manual, PyTorch nn.Module) give exactly MAE 59.58 because the 
problem is convex: all converge to the same global minimum. Small differences in 
coefficients (9e-07) and predictions (4.7e-04 bikes) come from different numerical 
precision in each library. With lr=0.01 convergence is slow (+6.27 MAE), with lr=0.5 it 
diverges; lr=0.3 and 0.1 balance convergence and stability.

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

Evidence from `uv run python diagnostics.py` and the dashboard:

| Observation | Number |
|---|---|
| Test bias (predicted minus actual), Oct / Nov / Dec | -40.4 / -34.5 / -26.8 rentals per hour |
| Validation bias (train-only model) | +21.7 rentals per hour |
| Smearing correction (factor 1.07): effect on test MAE | 59.6 -> 53.8 |
| Same correction, decided the legitimate way on validation | 53.1 -> 63.8 (rejected, not used) |
| Year-over-year growth ratio (2012 / 2011 monthly mean) | 2.0-2.5 in Jan-Mar, 1.4-1.8 in Apr-Jun, 1.4-1.7 in Jul-Dec |
| Temperature range, validation vs test | Validation is summer, test is autumn (Distributions tab) |
| Worst test days | 22-23 Nov 2012 (over-predicted); 8 Oct and several Saturdays (under-predicted) |

Growth from last year is a big problem: the system expanded 2-2.5 times in January-March. 
Solution: use the 2013 data or calculate the growth explicitly by month. The seasonal bias 
(validation is summer, testing is autumn) leads to underestimation in October-December (-40 
rentals). Solution: train on a whole year or split the data by seasons. Log bias (+21 in 
validation): log smearing is valid but needs to be checked on hold-out data.

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*

Claude (Anthropic) was used to: explore the dataset and present feature and
split options with their trade-offs and validation evidence; write the code
in `pipeline.py`, `train.py`, `app.py`, `ablation.py` and `diagnostics.py`;
explain the code; and format the tables in this report from the pipeline's
outputs.

I made these decisions: which dataset, which temporal split, which features to try (log, 
cyclical, interactions), why MAE, how to interpret ablation and diagnostics. Claude wrote 
the code for pipeline.py, train.py, ablation.py, and diagnostics.py, and helped suggest 
some feature options. I double-checked each number, selected whether to keep or drop 
something based on ablation and business situation, and made all conclusions.
