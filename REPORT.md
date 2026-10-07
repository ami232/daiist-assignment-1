# Assignment 1 Report

- **Name**: Antonio Meneses Escalante
- **Student ID**: 19015
- **Email**: ameneses.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

The dataset utilized on this assignment is the "Walmart Dataset" from kaggle. This one is derived from Walmart's 2014 recruiting competition. Each row represents one store's sales in one week and the date is the last day of that week. This dataset contains 6,435 rows × 8 columns, which is 45 stores × 143 weeks, with dates ranging from 5 Feb 2010 to 26 Oct 2012. There are no missing values. 

The reason why this dataset was chosen is primarily because it ignites a clear business analysis, in this case what variables are moving along with the sales of each store. Moreover, analyses like this one are often repeatly done in the workplace, making this a highly transferable assignment. 

## Business / real-life framing

This model might be used primarily by directors, individuals that need to oversee sales performance such as regional managers or operations directors. This one would be able to answer questions such as "what is driving sales up or down" and "what should each store be selling".

First, since decisions such as these ones require quantities and not yes/no answers, a regression model was chosen. Following, it was also decided that the target should be in log form, causing the coefficients to become % effects that apply proportionally to every store. Stores range from $0.26M to $2.1M a week, so a fixed dollar effect would be wrong for most of them.

Considering that past sales would swamp the drivers being explained, these were not included. Additionally, a chronological split was used since a random split would leak neighbouring weeks into training. Finally, a 52-week test set was utilized in order to include a holiday season and a 13-week validation set was used only for tunning.

The main metric is WMAE: the average absolute error in dollars, with holiday weeks counting five times as much as a normal week; MAE, MAPE and R² are reported alongside.

Holiday weeks count five times as much because they are the most important weeks of the year for the stores, with sales far above a normal week (about +40% at Thanksgiving), so a forecast that is too low leads to understocking and lost sales. 

## Data preparation & feature engineering

After initial data exploration, several critical aspects were uncovered. First, it was found that Store alone accounts for 91.7% of the variance. Second, the holiday_flag was missaligned, showing the flagged christmas week as the week after christmas, failing to show the real peak of store sales. Finally, exploration showed that labor day and the Super Bowl had almost no lift or effect.

The best option was to create a separate flag for each holiday (Super Bowl, Labor Day, Thanksgiving, post-Christmas) plus a new pre-Christmas flag for the real peak (the week dated 18–24 December), since with the original holiday flag the model would get one weight for all holidays. Additionally, Store is a label, not a quantity: fed in as a number, a linear model would assume sales rise or fall steadily from store 1 to store 45. Therefore its one-hot encoded, giving each store its own baseline level.

CPI was found to vary almost entirely between stores, with 99.9% of its variation due to stores, something already captured by the store-level columns. Within each store, the remaining variation is simply a slow, steady upward trend over time. Furthermore, its coefficient is unstable, changing from 0.18 under one regularization strength to 0.28 under another. A coefficient that shifts this much cannot reliably be interpreted as the “effect of inflation.” Removing it barely changed test performance.

Replacing CPI with a time trend made test WMAE worse (≈$93K vs. $86K), and stronger regularization hurt validation WMAE (≈$70K at alpha 0.1 vs. $97K at alpha 10), because the penalty shrinks the real differences between stores.

## Modeling: three implementations, one model

I used a ridge-regularized linear regression on log sales, because its coefficients read directly as % effects. All three implementations minimise the same ridge objective (average squared error plus a penalty on large weights) on the same data; because PyTorch averages the errors while scikit-learn sums them, the PyTorch penalty uses alpha/n.

| Implementation | How it trains |
|---|---|
| scikit-learn | `Ridge` solves for the weights directly. |
| Manual PyTorch | Autograd computes the gradients; weights are updated by hand. |
| Standard PyTorch | `nn.Module` + `nn.MSELoss` + `optim.SGD` do the same steps. |

Tuned on validation: alpha = 0.1 and learning rate 0.5 (1.0 diverged), both PyTorch versions run 20,000 full-batch steps. Final models were refitted on train + validation and tested once; the baseline is each store's average weekly sales.

Results (test year, Nov 2011 – Oct 2012):

| Model                    | WMAE ($) | MAE ($) | MAPE (%) | R²     |
|--------------------------|---------:|--------:|---------:|-------:|
| Baseline (store average) | 149,737  | 102,549 | 9.37     | 0.8991 |
| scikit-learn             | 85,788   | 75,545  | 7.87     | 0.9645 |
| Manual PyTorch           | 85,756   | 75,524  | 7.87     | 0.9645 |
| Standard PyTorch         | 85,757   | 75,524  | 7.87     | 0.9645 |

The coefficients agree within 0.0014 and the test predictions within 0.06%, because all three minimise the same convex objective (a single lowest point) on the same data. The tiny gaps come from PyTorch's 32-bit arithmetic (vs. 64-bit in scikit-learn) and from gradient descent stopping after 20,000 steps, just short of the minimum that scikit-learn computes directly.

Coefficients as % effects on weekly sales (scikit-learn; the other two are effectively identical):

| Driver                      | Effect (%)      | Relative to / per       |
|-----------------------------|----------------:|-------------------------|
| Temperature                 | +1.93           | +1 std (18.9 °F)        |
| Fuel_Price                  | −0.71           | +1 std ($0.45/gal)      |
| Unemployment                | −3.81           | +1 std (1.9 pts)        |
| Super Bowl / Labor Day week | +3.03 / +5.56   | normal week, same month |
| Thanksgiving week           | +39.93          | normal November week    |
| Pre- / post-Christmas week  | +40.64 / −24.41 | normal December week    |

The gain over the baseline is much larger in WMAE (43%) than in MAE (26%), so most of it comes from holiday weeks, which a store average cannot anticipate. Overall, the model explains about 65% of the variance that store averages miss. For the executive, the holiday calendar and each store's own sales level dominate, while temperature, fuel prices and unemployment move sales by only a few percentage points. Temperature correlates negatively with sales within a store (−0.10), but has a positive coefficient (+1.9%), because the coldest weeks are the holiday months. Once month and holidays are accounted for, a warmer-than-usual week is linked to slightly higher sales.



## Limitations & next steps

Some of the limitations identified were:

1. The validation quarter contains only Labor Day: The regularization strength and learning rate were chosen on a validation set whose only holiday is Labor Day, so they are optimised for ordinary weeks rather than the holiday peaks that matter most to the business. The proper fix, rolling time-series cross-validation with holiday seasons inside the validation windows, would require more years of data: this dataset has only one holiday season before the test year, and it is needed for training. 
2. Training contains only one holiday season: Every holiday effect was learned from the 2010 season alone, so anything unusual about that year is built into the coefficients. I would estimate the holiday effects separately on the 2010 and 2011 seasons and compare them to check they're stable.
3. The model assumes the same % holiday lift for every store: The model applies one percentage lift per holiday to every store, although large and small stores may react differently. I would first check the model's holiday-week errors store by store. If some stores are consistently over- or under-predicted, I would let the holiday effect vary by store type or size, both available in the original competition's stores.csv

## Generative AI use disclosure

Code: Claude wrote the code with my assistance (features.py, train.py, models.py, app.py and the exploration notebook cells) and debugged it. 

Decisions: I chose the dataset and defined the business user and questions. For the other design decisions (separate holiday flags, log target, chronological split, metric, baseline), I used AI to list the trade-offs and afterwards I made the final choice. Dropping CPI, adding month features and using a separate validation set were proposed by Claude, and I adopted them after reviewing the evidence.

Report: I wrote the first draft. Claude then revised it for completeness and reading comprehension, mainly the descriptive parts (dataset, pipeline, methods, settings and results tables). The reasoning behind the metric, the CPI explanation, the agreement explanation, the interpretation of results and the limitations are written by me.
