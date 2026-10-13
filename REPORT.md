# Assignment 1 Report

- **Name**: Niccolò Pragliola
- **Student ID**: 19873
- **Email**: npragliola.ieu2022@student.ie.edu
- **Group**: This is an individual assignment

## Dataset

**Online Retail II** — UCI Machine Learning Repository (id 502, CC BY 4.0). All invoice
lines of a UK-based online giftware retailer, many of whose customers are wholesalers.

| | |
|---|---|
| Period | 2009-12-01 → 2011-12-09 |
| Raw invoice lines | 1,067,371 (8 columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country) |
| Lines with a Customer ID (committed file) | 824,364 (22.8% of lines have no customer) |
| Customers | 5,942 in 41 countries (91% UK) |
| Committed file | `data/online_retail_ii.csv.gz`, 11.5 MB, built by `prepare_data.py` |

The raw data is one row per invoice line; the modelling table is one row per
**customer × quarterly cutoff** (see below), 17,703 rows over 5,212 distinct customers.

I chose Online Retail II because its dated, customer-linked transactions make it possible to study repeat purchasing over time and build features from real buying behavior. The data has no target column, so I define lapse from each customer's transaction history: a customer lapses when they place no order in the 90 days after a cutoff. This turns the task into predicting a future outcome from prior activity rather than looking up an existing label. Lines without a Customer ID cannot be connected to a customer's earlier purchases or later outcome, so they cannot contribute valid customer-level features or labels.

## Business / real-life framing

A retailer's marketing team makes the decision before each quarter: which currently active customers should receive a 10% win-back voucher? Lapse means no purchase in the next 90 days, matching the quarterly decision cycle and giving every snapshot a fixed outcome window. The model scores customers who ordered at least once in the preceding 180 days, so it uses recent behavior to identify people who are still engaged enough for a win-back offer to be actionable. Customers without an order in that lookback are outside this campaign's target population.

**Voucher economics used (assumptions):**

| Parameter | Value |
|---|---|
| Discount on the next order | 10% |
| Gross margin | 30% |
| Save rate (share of would-be lapsers a voucher brings back) | 15% |
| Quarters a won-back customer keeps ordering | 4 |
| Mailing cost per voucher | £1 |
| Break-even lapse probability `d / (s·(g·L − d) + d)` | **0.377** |

A true positive is a voucher sent to someone who would otherwise lapse: with the assumed 15% save rate, the expected contribution is 15% × (four quarters of gross margin − the 10% discount on the first quarter's order) × usual quarterly spend, minus the £1 mailing cost. The discount sits inside the 15% because only customers who actually come back use the voucher. A false positive is sent to someone who would order anyway; the voucher gives away 10% of their usual quarterly spend, plus the mailing cost, without changing the purchase decision. A false negative receives no voucher despite lapsing, so the campaign earns or loses £0 relative to sending nothing, but misses a possible win-back; a true negative also receives nothing and has £0 incremental campaign profit. Four quarters approximates a year of continued orders after winning a customer back; with one quarter, the sensitivity table shows that the validation-tuned policy sends no vouchers and earns £0, while the current 0.38 threshold in the dashboard produces a loss. I trust the 15% save-rate assumption least because no randomized campaign result measures how many lapsers a voucher actually changes; retention for four quarters is also uncertain.

**Split — stacked quarterly snapshots, by time:**

| Split | Cutoff | Customers | Lapse rate |
|---|---|---|---|
| train | 2010-06-01 | 2,652 | 0.500 |
| train | 2010-09-01 | 2,822 | 0.380 |
| train | 2010-12-01 | 3,465 | 0.620 |
| train | 2011-03-01 | 3,338 | 0.576 |
| validation | 2011-06-01 | 2,658 | 0.500 |
| test | 2011-09-01 | 2,768 | 0.393 |

A random row split could put later snapshots in training and earlier snapshots in validation or test, letting the model learn from future periods; the same customer also appears at multiple quarterly cutoffs. The first cutoff, 2010-06-01, is the first one with the full 180 days of history after the data begins on 2009-12-01. The last, 2011-09-01, is the final cutoff whose following 90-day outcome is observed before the data ends on 2011-12-09. Using several earlier training quarters provides more examples across different seasonal conditions; the lapse rates range from 0.38 to 0.62, showing that the base rate changes substantially over the year. I choose the voucher threshold by validation campaign profit because this decision is about expected pounds earned, while accuracy and F1 treat errors in ways that do not reflect the different financial consequences of false positives and false negatives.

## Data preparation & feature engineering

**Cleaning (in `train.ipynb`):**

| Step | Lines |
|---|---|
| Lines with a customer | 824,364 |
| After dropping exact duplicate lines | 797,885 |
| After keeping 5-digit product codes only (drops POST, M, ADJUST, BANK CHARGES, …) | 794,223 |
| After dropping zero-price lines | 794,163 |

Cancellations (invoices starting with "C") are not counted as orders but are used for the
return-rate feature.

**Features** (all computed from the 180 days before each cutoff, standardised with a scaler
fit on the training snapshots only):

| Group | Features |
|---|---|
| RFM | `log_recency`, `log_orders`, `log_revenue` |
| Order behaviour | `log_avg_order`, `log_lines_per_order`, `log_products`, `days_since_first`, `one_time`, `log_overdue`, `recent_share` |
| Returns & country | `return_rate`, `uk` |
| Season of the cutoff | `season_spring`, `season_autumn`, `season_winter` (reference: summer) |
| Christmas | `xmas_share`, `xmas_x_autumn` |

**Ablation on validation** (groups added one at a time, kept if validation log loss improves):

| Group added | # features | Val log loss | Val AUC | Kept |
|---|---|---|---|---|
| RFM | 3 | 0.5906 | 0.7484 | ✓ |
| + order behaviour | 10 | 0.5882 | 0.7497 | ✓ |
| + returns & country | 12 | 0.5879 | 0.7499 | ✓ |
| + season | 15 | 0.5821 | 0.7520 | ✓ |
| + christmas | 17 | 0.5820 | 0.7526 | ✓ |

Money and count variables are strongly right-skewed, with a small number of wholesalers far above the median, so logarithms keep those extremes from dominating the model. `log_overdue` compares a customer's time since last order with their own average gap between orders, distinguishing someone who is late for their usual rhythm from someone whose last order was simply a while ago; one-time buyers receive zero because their personal rhythm cannot be estimated. `days_since_first` uses only the 180-day lookback because the source begins in December 2009 and does not reveal customers' complete histories before then. Christmas-heavy spending is uncommon and the dashboard distributions overlap substantially, so it is only a modest signal; the autumn interaction lets that relationship differ in the pre-Christmas quarter. Seasonality helped most, improving validation log loss from 0.5879 to 0.5821; the Christmas group helped least, from 0.5821 to 0.5820, but was retained because it still improved validation log loss and slightly improved AUC. I kept the features within the historical lookback and did not use future transactions as predictors, which would leak the outcome into the model.

## Modeling: three implementations, one model

**Model:** L2-regularised logistic regression (binary target).

| | scikit-learn | PyTorch manual loop | PyTorch `nn.Module` + `optim.SGD` |
|---|---|---|---|
| Objective | `C·Σ logloss + ½‖w‖²` | `mean logloss + (λ/2)‖w‖²` | same, via `weight_decay = λ` on weights only |
| Regularisation | C = 1.0 (tuned on validation) | λ = 1/(C·n) = 8.145e-5 | λ = 8.145e-5 |
| Optimiser | lbfgs, tol 1e-10 | full-batch gradient descent, manual update | full-batch SGD |
| Learning rate | — | 2.0 (tuned) | 2.0 (tuned) |
| Init / epochs | — | zeros / 36,760 | random / 33,819 |
| Gradient norm at solution | 3.31e-8 | 1.00e-8 | 1.00e-8 |
| Training objective | 0.5738258063 | 0.5738258063 | 0.5738258063 |

C search: validation log loss is flat between C = 0.1 (0.58222) and C = 100 (0.58205);
best at C = 1 (0.58198). Learning-rate search (2,000-epoch budget): 0.05 → 2.0 all head to
the same objective (faster for larger rates); 4.0 diverges.

**Agreement:**

| Pair | Max abs weight difference | Max abs test probability difference |
|---|---|---|
| sklearn vs manual | 7.39e-5 | 2.78e-6 |
| sklearn vs module | 7.38e-5 | 2.78e-6 |
| manual vs module | 5.19e-9 | 1.98e-10 |

**Test results** (threshold 0.38, chosen on validation for each model):

| Model | AUC | Log loss | Brier | Vouchers sent | Share of them that lapsed | Profit vs no vouchers |
|---|---|---|---|---|---|---|
| scikit-learn | 0.7371 | 0.5777 | 0.1997 | 1,497 | 0.551 | **£9,772** |
| PyTorch manual | 0.7371 | 0.5777 | 0.1997 | 1,497 | 0.551 | **£9,772** |
| PyTorch nn.Module | 0.7371 | 0.5777 | 0.1997 | 1,497 | 0.551 | **£9,772** |
| Baseline: train lapse rate for everyone | 0.5 | 0.7061 | 0.2565 | — | — | — |
| Baseline: no vouchers | — | — | — | 0 | — | £0 |
| Baseline: voucher to everyone | — | — | — | 2,768 | 0.393 | −£95,687 |
| Baseline: recency rule (no order for ≥ 40 days, tuned on validation) | — | — | — | 1,653 | 0.480 | £691 |

- Model minus recency rule, bootstrap over test customers (2,000 resamples): **£9,122**, 95% CI [£699, £15,445].
- Calibration on test: mean predicted lapse 0.381 vs observed 0.393, ECE 0.023.
- Validation-optimal threshold 0.38 vs break-even formula 0.377.

Logistic regression is a suitable choice because lapse is binary, the model returns probabilities that can be compared with an economic threshold, and its coefficients are interpretable. The three implementations should reach the same solution because they use the same training data, scaled features, and L2-regularized logistic-loss objective. Scikit-learn minimizes `C × sum(log loss) + ½‖w‖²`; dividing by `C·n` gives the PyTorch objective with `λ = 1/(C·n)`. The intercept is excluded from weight decay because scikit-learn does not penalize it, so penalizing the bias would change the model being fit. They agree closely: test probabilities differ by at most 2.78e-6, and the two PyTorch fits' weights differ by only 5.19e-9; scikit-learn's weights differ by about 7.4e-5 because its solver stops at its own convergence tolerance. The learning rate controls how quickly optimization reaches the minimum, not which minimum is preferred, so I selected it by training objective under an equal epoch budget; 4.0 diverged. Validation loss is nearly flat from C = 0.1 to 100, suggesting performance is not very sensitive to regularization across that range, with C = 1 giving the lowest value. The validation profit-optimal threshold of 0.38 is close to the economic break-even probability of 0.377; that agreement is consistent with useful calibration, and the held-out check is also reasonably close (mean predicted 0.381 versus observed 0.393, ECE 0.023), though it does not prove perfect calibration. The estimated £9,772 versus £691 recency-rule advantage is meaningful on this test snapshot: the bootstrap 95% interval for the difference is £699 to £15,445, entirely above zero, but it covers only one quarter and depends on the assumed voucher economics.

**Sensitivity of the decision to the economics** (threshold and recency rule re-tuned on
validation each time; same trained model):

| Quarters retained | Save rate | Break-even | Val threshold | Test profit, model | Test profit, recency rule |
|---|---|---|---|---|---|
| 1 | 0.05 | 0.909 | 0.90 | £0 | £0 |
| 1 | 0.15 | 0.769 | 0.90 | £0 | £0 |
| 1 | 0.25 | 0.667 | 0.90 | £0 | −£925 |
| 2 | 0.05 | 0.800 | 0.90 | £0 | £0 |
| 2 | 0.15 | 0.571 | 0.52 | −£505 | −£62 |
| 2 | 0.25 | 0.444 | 0.48 | £2,817 | £2,460 |
| 4 | 0.05 | 0.645 | 0.64 | −£40 | −£1,080 |
| **4** | **0.15** | **0.377** | **0.38** | **£9,772** | **£691** |
| 4 | 0.25 | 0.267 | 0.215 | £29,557 | £20,768 |
| 6 | 0.05 | 0.541 | 0.52 | £38 | −£1,038 |
| 6 | 0.15 | 0.282 | 0.215 | £25,088 | £16,584 |
| 6 | 0.25 | 0.190 | 0.215 | £63,072 | £51,319 |

The table shows that voucher profitability depends strongly on both the save rate and how long a won-back customer continues ordering. With only one retained quarter, even the most favorable tested save rate leads the validation-tuned policy to send no vouchers, so the programme is not worth running under those assumptions. At four quarters and a 15% save rate, targeted vouchers outperform the recency rule; higher save rates or longer retention raise the potential return, while lower values can erase it or make it negative. The business should run the programme only when measured incremental retention and margin are expected to cover the discount and mailing costs; a randomized pilot should establish those economics before broad rollout.

## Limitations & next steps

The voucher save rate and post-win-back retention are assumptions rather than measured effects. I would run a randomized voucher holdout among eligible customers and track incremental orders, margin, and repeat purchasing over the following year to estimate both. The model predicts lapse risk, not whether a voucher changes a customer's behavior, so it may spend on people who would return anyway; a follow-up experiment could support an uplift model that estimates each customer's treatment effect. Customers recur across the four training snapshots, so the training rows are not fully independent: the effective sample size is smaller than 12,277, which makes the coefficients and the validation-based choices less certain than they look (the test bootstrap is not affected, since the test set is a single snapshot with each customer once). I would add a customer-grouped evaluation, such as grouped cross-validation over the training snapshots, as a robustness check. Finally, the test set contains only one autumn quarter, so results may not generalize to other seasons; I would reserve additional later quarters for testing and report performance separately by season.

## Generative AI use disclosure

Claude helped write the pipeline and dashboard code while the framing, decisions and analysis are all personal.