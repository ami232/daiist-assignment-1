# Assignment 1 Report

- **Name**: Eduardo Alfaro
- **Student ID**: 19911
- **Email**: ealfaro.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

I used the **Bank Marketing** dataset from the UCI Machine Learning
Repository (Moro, Cortez & Rita, 2014; CC BY 4.0), specifically
`bank-additional-full.csv`, which is saved in `data/`.

- **One row** is one client that a Portuguese bank called during its phone
  campaigns to sell a term deposit.
- **Size**: 41,188 rows, 20 input columns and the target `y`.
- **Target**: `y` says whether the client subscribed to the deposit. Only
  **11.3%** did, so the classes are imbalanced.
- **Columns**: the client's profile (age, job, marital status, education,
  default, housing loan, personal loan), details of the current campaign
  (channel, month, weekday, number of calls), results of previous campaigns
  (`pdays`, `previous`, `poutcome`) and five macroeconomic indicators
  (Euribor, employment, prices, consumer confidence).
- **Data quality**: missing values appear as `"unknown"` in six columns,
  `pdays = 999` means "never contacted before", and there are 12 duplicate
  rows.

I picked it because it raises real decisions. The rows are ordered by date
(May 2008 to November 2010), there is a leakage trap (`duration`), and the
two types of error have very different costs.

## Business / real-life framing

**The decision.** The bank's sales team has to decide **which clients to
call next**. The call center can't call everyone. I assumed it can call
**25% of the clients in each campaign** (each month). The model gives every
client a probability of subscribing, and the team calls the top 25% of the
list.

**Target.** `y = 1` if the client subscribed and `0` if not. Because the
target is binary, I use **logistic regression**.

**What is known before the call.** The model can only use information that
exists before the call is made. That rules out **`duration`** (call length).
It is only known after the call ends, and on its own it gives an AUC of 0.82,
so including it would produce a model that looks great but can't be used, so
I excluded it. The variables I expected to matter most were **job** and the
**loan/credit information**. Job does matter (students 31% and retired 25%
subscribe, blue-collar only 7%), and so does default (no = 12.9%, unknown =
5.2%). The housing and personal loan flags barely change anything (10.8% to
11.6%).

**Temporal split.** The model is meant to choose whom to call in *future*
campaigns, so I train on the past and test on later periods. A random split
would let 2010 information leak into training. The data also changes a lot
over time:

| Period | Calls per month | Subscription rate | Euribor |
|---|---|---|---|
| 2008 | thousands | 3% to 6% | ~4.9% |
| 2009 | hundreds to a few thousand | 9% to 57% | 1.6% to 0.7% |
| 2010 | ~130 to 310 | 45% to 58% | ~0.7% to 1.0% |

**Cost of errors.** I estimated the two errors like this:

- **False positive (calling someone who doesn't subscribe): about 5 EUR.**
  An agent costs about 15 EUR an hour, a contact takes about 8 minutes
  (dialing, the ~4.3-minute call and wrap-up), which comes to about 2 EUR per
  contact. Each client gets about 2.6 contacts per campaign, so roughly 5 EUR.
- **False negative (not calling someone who would have subscribed): about
  100 EUR.** I assumed an average deposit of 5,000 EUR kept for a year, with a
  2% margin for the bank.

So missing a subscriber costs about **20 times more** than a wasted call.
These are my own assumptions, not data from the bank, so the dashboard lets
you change them.

**Metric.** If cost were the only thing that mattered, the best threshold
would be 5 / (5 + 100) ≈ 0.05, which means calling almost everyone. The call
center can't do that, so capacity is the real limit. What matters is that the
calls we *can* make reach people who will subscribe. My main metric is
therefore **precision at 25% capacity per month**: in each month, call the
top 25% of clients and measure how many of them subscribe. I also report:

- **lift** (precision divided by the positive rate), because the positive
  rate is very different in each period;
- **recall**;
- **business cost** (FP × 5 + FN × 100).

Accuracy is useless here, because predicting "no" for everyone already gets
about 89%.

## Data preparation & feature engineering

Everything is in `train.ipynb`, in this order:

1. **Rebuilt the date.** There is no year column, but the rows are in date
   order, so every time the month goes backwards a new year starts. This
   gives each row a year and month, which I need for the split.
2. **Dropped `duration`** (leakage, see above).
3. **Removed the 12 duplicate rows.** One mistake I found: the first time, I
   looked for duplicates *after* dropping `duration`, and it removed 1,784
   rows. Without the call length, many different calls look identical, so I
   now look for duplicates on the original data.
4. **Kept `"unknown"` as its own category.** Not knowing something is
   informative (default = unknown subscribes 5.2% vs 12.9% for "no"), and
   real clients with unknown values will keep coming.
5. **Dropped `pdays`.** 999 means "never contacted", not 999 days, and a
   linear model would read it as a very large number. I first replaced it
   with a yes/no flag, but the flag turned out to be almost a copy of
   `poutcome` (all "success" clients were flagged). `poutcome` already says
   never contacted / failure / success, so I kept only that.
6. **Temporal split:**

   | Set | Period | Rows | Subscription rate |
   |---|---|---|---|
   | Train | May 2008 to May 2009 | 36,214 (88%) | 6.7% |
   | Validation | Jun 2009 to Dec 2009 | 2,904 (7%) | 39.1% |
   | Test | 2010 | 2,058 (5%) | 52.1% |

   I chose these cuts so that validation and test are both in the "new"
   period (low Euribor, high conversion). That way, decisions made on
   validation transfer better to test.

7. **Feature engineering** (all decisions based on training data only):
   - **`month`: dropped.** The month effect is really a "which year"
     effect: May converts 3% in 2008, 9% in 2009 and 58% in 2010. Also,
     September never appears in train.
   - **`age`: age bands** (≤25, 26-30, 31-40, 41-50, 51-60, >60). The
     relation is U-shaped (10.7% for ≤25, 5.4% for 41-50, 36.2% for >60), and
     a single linear weight can't capture a U.
   - **`campaign`: `log(1 + campaign)`.** It has a long tail (median 2,
     max 56). The log means each extra call matters less, and a few extreme
     clients can't dominate the weight.
   - **Macro indicators: kept only `euribor3m`.** Euribor, employment
     variation and number of employees are 0.96 to 0.98 correlated, which
     makes the weights unstable. One indicator is enough to describe the
     economic situation.
   - **`housing`/`loan` unknown: one flag `loans_unknown`.** The same 990
     clients are "unknown" in both, so the two columns were identical.
8. **Encoding and scaling.** One-hot encoding for categoricals, with the
   first category as the reference. Then **every column is standardised,
   including the one-hot columns**. This turned out to be key for the
   comparison. The encoder and scaler are fitted on train only.

In total the model uses **41 features**.

## Modeling: three implementations, one model

**Model:** logistic regression, because the target is binary

All three versions minimise exactly the same function: the mean log loss
plus an L2 penalty λ/2·‖w‖², with λ = 1/(C·n). The intercept is not
penalised. That is the same thing scikit-learn does with its `C` parameter.

- **scikit-learn**: `LogisticRegression(C=1)`.
- **Manual PyTorch**: tensors with `requires_grad`, the loss written by
  hand, `backward()`, and the update `w = w - lr * grad`.
- **Standard PyTorch**: `nn.Linear` + `BCEWithLogitsLoss` + `torch.optim.SGD`,
  with the L2 penalty as `weight_decay` on the weights only.

**Hyperparameters (chosen on validation):**

- **C = 1.** Validation precision barely moved for any C ≥ 0.001 (53.0% to
  53.7%), which is within noise (±1.8 points), so I kept the default. Very
  strong regularisation (C = 0.0001) made it worse (50.3%).
- **Learning rate = 0.5.** With 0.01 or 0.1, gradient descent hadn't
  converged after 2,000 epochs. From 0.5 up, the results matched
  scikit-learn.
- **10,000 epochs**, full batch.

**Results:**

| Split | Model | Precision@25% | Lift | Recall@25% | Business cost | AUC |
|---|---|---|---|---|---|---|
| Val | Baseline (random 25%) | 36.9% | 0.94 | 23.6% | 88,990 EUR | 0.50 |
| Val | scikit-learn | 53.2% | 1.36 | 34.0% | 76,600 EUR | 0.603 |
| Val | Manual PyTorch | 53.2% | 1.36 | 34.0% | 76,600 EUR | 0.603 |
| Val | nn.Module + optim | 53.2% | 1.36 | 34.0% | 76,600 EUR | 0.603 |
| Test | Baseline (random 25%) | 50.9% | 0.98 | 24.4% | 82,365 EUR | 0.50 |
| Test | scikit-learn | 75.9% | 1.46 | 36.4% | 68,820 EUR | 0.698 |
| Test | Manual PyTorch | 75.9% | 1.46 | 36.4% | 68,820 EUR | 0.698 |
| Test | nn.Module + optim | 75.9% | 1.46 | 36.4% | 68,820 EUR | 0.698 |

On the test set, 76 out of every 100 calls end in a subscription, compared
with 51 when calling at random. That is about 13,500 EUR less in business
cost. Test looks better than validation partly because more people subscribed
in 2010, which is why I also compare lift (1.36 vs 1.46).

**Do the three agree? Yes, but not at first.** In the final version, the
probabilities differ by less than 0.001, the weights by less than 0.005, and
the three models pick exactly the same clients to call. Getting there took
three fixes:

1. **One-hot columns were not scaled.** Rare categories (e.g. age > 60, 163
   clients) are columns that are almost always 0, so their gradient is tiny
   and gradient descent moves very slowly in that direction. scikit-learn
   uses a smarter optimiser (L-BFGS) that doesn't have this problem. Before
   scaling, the probabilities differed by up to **0.54**. After scaling, the
   difference was almost zero.
2. **Duplicate columns** (`housing_unknown` = `loan_unknown`, and the
   `pdays` flag ≈ `poutcome`). They split one effect between two weights.
   Removing them made the weights easier to interpret.
3. **Not enough epochs.** `previous` and `poutcome_nonexistent` are −0.975
   correlated, which leaves an almost flat direction in the loss. With 3,000
   epochs there was still a 0.018 difference; with 10,000 it was below 0.001.

The lesson: the three methods solve the same problem. When they disagree,
the cause is the optimiser not reaching the minimum yet, not the model.

**Largest weights:** `euribor3m` (−0.37, lower Euribor means more
subscriptions), `contact = telephone` (−0.19 vs mobile), `poutcome =
success` (+0.16), university degree (+0.12), `log_campaign` (−0.10, more
calls means less likely) and `default = unknown` (−0.10). Euribor is the
biggest weight, but it doesn't change the ranking *within* a month, because
everyone called in the same month has the same Euribor.

## Limitations & next steps

- **The world changes between periods.** The model learned with 6.7%
  positives and is tested on 52%. In the dashboard's predicted-vs-actual plot
  it clearly underestimates probabilities on test (the average predicted
  probability is 28%, while the real rate is 52%). Ranking per month is not affected, but
  a fixed threshold is. The threshold that calls 25% in validation (0.32)
  calls **37%** of clients in test, more than the capacity. *Next step:*
  retrain regularly on the most recent months, or recalibrate the
  probabilities on recent data.
- **The model is modest** (AUC 0.60 on val, 0.70 on test). Without
  `duration`, a linear model with these columns can't separate clients very
  well. *Next step:* try interactions (e.g. age × job) or tree-based models,
  and use more client data (balance, products held).
- **Small and noisy evaluation sets.** Validation has 2,904 rows and test
  2,058, and some months have fewer than 200 clients. Differences of 1 to 2
  points are noise. *Next step:* evaluate on several consecutive time
  windows (rolling evaluation) instead of a single split.
- **`campaign` has a bit of leakage.** It counts the calls of the current
  campaign *including the last one*, and clients who say "no" keep getting
  called. *Next step:* use only the number of calls made *before* each
  contact, which needs the call-by-call log.
- **Who was called is not random.** In 2010 the bank made far fewer calls
  and probably chose its clients better, so the data reflects the bank's
  past decisions. *Next step:* keep a small random group of calls in each
  campaign to measure the real effect.
- **The costs are assumptions.** 5 EUR and 100 EUR are my estimates.
  *Next step:* get real figures from the bank. Meanwhile, the dashboard lets
  you change them.
- **Euribor in test is below anything seen in training** (0.63 to 1.05 vs
  1.24 to 5.05), so the model is extrapolating.

## Generative AI use disclosure

I used Claude (Anthropic) as an assistant throughout the project:

- It suggested candidate datasets and I chose Bank Marketing.
- At each step it ran the exploratory analysis, explained the options and
  their trade-offs, and **I made the decisions**: the business framing,
  excluding `duration`, the temporal split and its cut points, how to handle
  `unknown`/`pdays`/duplicates, the feature engineering choices, the 25%
  per-month capacity, C and the learning rate.
- It proposed the cost estimates (5 EUR / 100 EUR) based on assumptions I
  reviewed.
- It wrote the code (`train.ipynb`, `app.py`) and found and investigated the
  disagreement between the three implementations with me.
- I wrote an initial shorter and simpler report,then he drafted this report from my decisions and the results. I reviewed and
  edited it.

## Generative AI use

Per the syllabus AI Policy: disclosed AI use is fine and must be stated in
REPORT.md's disclosure section. Within that policy, here's how it applies
to this assignment specifically:

- **Fine to use AI for**: boilerplate and common operations, like loading
  a dataset, saving a trained model, and especially plotting and
  presenting results in the Gradio dashboard.
- **Use your own judgement for**: the decisions that are the point of this
  assignment, like which features to design, how to frame the business
  problem, and the implications of your design choices. AI can write the
  code for a decision, but the decision itself has to be yours.
- **REPORT.md**: the ideas and findings in it must be your own. AI may
  help with formatting, not with generating the analysis or conclusions.

None of this changes what's expected of you: you have to be able to
explain every decision in your submission as if you made it yourself,
because you did. Using a tool to help write it doesn't transfer the
understanding requirement to the tool.
