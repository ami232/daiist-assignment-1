# Assignment 1 Report

- **Name**: Nicolas Aguado
- **Student ID**: 20446
- **Email**: naguado.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

**Source.** Lending Club loans (a US peer-to-peer lender), in the version published
on Zenodo (record 11295916): 1.35 million loans from 2007–2018 with 15 columns. This
version only keeps information known when someone applies for a loan, plus a
`Default` column (1 = the loan defaulted).

**One row = one loan.** Columns include annual income, debt-to-income ratio (DTI),
loan amount, FICO score, employment length, loan purpose, home ownership, US state,
zip code, a title and description written by the borrower, and the issue date.

**What is in the repo.** `prepare_data.py` (run once by hand) downloads the full file,
keeps loans issued 2012–2015 and takes a random sample of 20,000 loans, saved as
`data/loans_sample.csv` (2.6 MB). About 18% of them defaulted.

**Why 2012–2015.** Loans from 2018 show a much lower default rate (16% vs 23% in
2016–2017). They are not safer: many had not had time to default yet when the data
was collected. Before 2012 there are very few loans.

**Why this dataset.** Lending is a real decision where the two kinds of mistake cost
different amounts, the data has a time order that matters, and the raw columns need
real work before they are usable.

## Business / real-life framing

**The decision.** A lender gets a loan application and must decide whether to fund
it. The model estimates the probability that the loan defaults; if it is above a
threshold, the lender rejects the application.

**What each mistake costs** (my assumptions, set in `pipeline.py`):
- Approving a loan that defaults: lose about **60% of the amount lent**.
- Rejecting a loan that would have been repaid: miss about **15% of the amount lent**
  in interest.

**How this shapes the pipeline:**

1. **Only application-time information.** The model is used when someone applies, so
   it can only use what is known at that moment. I also dropped the description
   column, because it tells you *when* the loan was issued, not how risky it is (see
   below).
2. **Split by time, not at random.** In real life the model is trained on past loans
   and used on new ones. So I train on loans from **2012–2014** (10,643 loans, 17.6%
   default) and test on **2015** (9,357 loans, 20.1% default). A random split would
   mix 2015 loans into training and hide that 2015 was a riskier year. The test shows
   this matters: the model expected 18.1% of 2015 loans to default, but 20.1% did.
3. **The threshold comes from costs, not accuracy.** Accuracy is useless here:
   approving everyone is already 80% accurate. Since a default costs 4 times more
   than a wrong rejection (60% vs 15%), it is worth rejecting a loan once its default
   probability is above about **20%**. Testing every threshold on the last 6 months
   of training data, the cheapest one was **0.17**, close to that 20%.

## Data preparation & feature engineering

All transformations live in `pipeline.py`, so training and the app apply exactly the
same steps.

**Features:**

| Feature | What I did | Why |
|---|---|---|
| FICO score | used as is | Strongest single signal: default rate falls from 24% for the lowest scores to 11% for the highest. |
| DTI | capped at 60 | Higher debt, more risk (14% → 27% default). The cap removes one extreme value. |
| Income | log | Very uneven (median $64k, max $9.5M). The log stops a few huge incomes from dominating. |
| Loan amount | log | Same reason. |
| Loan-to-income | amount ÷ income | How big the loan is compared with what the borrower earns (12% → 27% default). |
| Employment length | text → number of years (0–10) | Models need numbers, not text. |
| Employment not informed | 1 if missing, else 0 | Not reporting it is a warning sign: 23% default vs about 19% for the rest. |
| Purpose, home ownership, state | one column per category | Categories with fewer than 50 loans are grouped together so the model does not learn from tiny groups. |

**Preprocessing learned from training data only.** Missing values are filled with
the training median and numbers are rescaled to a common scale (mean 0, std 1). These
are calculated on the training loans only and then applied unchanged to the test
loans, so no information from 2015 leaks into training. The common scale also lets
the PyTorch versions train with a single learning rate.

**Columns I dropped:**
- `experience_c`: the same value for every loan, so it tells us nothing.
- `zip_code`: over 800 different values, too many to learn from. Location can also
  act as a proxy for protected characteristics, which is a problem in lending.
- `title`: mostly repeats the loan purpose.
- `desc`: written for 61% of 2012 loans but 0% of 2015 loans (Lending Club stopped
  asking for it). A "has description" feature would really mean "issued early",
  not "low risk". I tested it and it does not help.

**What worked and what didn't (2015 test set):**

| Features used | ROC AUC | Log loss |
|---|---|---|
| Raw numbers only (FICO, DTI, income, amount) | 0.648 | 0.4836 |
| Engineered numeric features | 0.652 | 0.4791 |
| **Full model** | **0.666** | **0.4750** |
| Full model without loan-to-income | 0.667 | 0.4749 |
| Full model without state | 0.665 | 0.4745 |

- The engineered features help, but the gains are modest.
- **Loan-to-income turned out to be redundant.** On its own it is the strongest
  signal, but the model already has log income and log amount, which together carry
  the same information. A simpler model would drop it.
- **State adds nothing.** Removing it even slightly improves the log loss; its
  coefficients are probably noise from small groups.

## Modeling: three implementations, one model

**Model.** Logistic regression, because the target is yes/no (default or not) and it
gives a probability, which the cost-based threshold needs.

All three versions minimize the same thing: the log loss plus a small penalty on
large coefficients (L2 regularization). scikit-learn sets the penalty strength with
a parameter `C` instead of λ; I converted between them so all three use the same
penalty.

- **scikit-learn:** `LogisticRegression`.
- **Manual PyTorch:** the weights as raw tensors, the loss written by hand, autograd
  for the gradients, and the update step written by hand.
- **Standard PyTorch:** an `nn.Module` with one linear layer, a built-in loss and the
  `torch.optim.SGD` optimizer.

**Tuning (kept to a minimum).** I tried four penalty strengths on the last 6 months
of training data and picked λ = 0.001. The results were almost the same for all of
them, so regularization barely matters here (10,000 loans for 63 coefficients). The
learning rate for both PyTorch versions was fixed at 0.5.

**Results on the 2015 test set** (threshold 0.17):

| Model | ROC AUC | Log loss | Cost (USD) | Loans rejected |
|---|---|---|---|---|
| Baseline: approve everyone | 0.500 | 0.5034 | 17,301,165 | 0% |
| scikit-learn | 0.6656 | 0.4749 | 12,527,550 | 48.1% |
| PyTorch, manual loop | 0.6656 | 0.4749 | 12,527,550 | 48.1% |
| PyTorch, nn.Module + optim | 0.6656 | 0.4749 | 12,527,550 | 48.1% |

The model lowers the cost of the 2015 loans by **$4.8M (−28%)** compared with
approving everyone.

**Do the three agree? Yes, but not at first.** In the final version their
predictions match to the sixth decimal place, and the two PyTorch versions are
identical, because they run the same algorithm; the only difference is who applies
the update.

With 3,000 training steps, PyTorch did **not** fully match scikit-learn: predictions
differed by up to 0.007 and the intercept was clearly different. The reason: the
category columns of each variable always add up to 1, just like the intercept. So
the model can shift weight between the intercept and those columns without changing
its predictions much. Gradient descent moves very slowly in that direction, while
scikit-learn's solver is smarter and gets there quickly. With 30,000 steps the gap
disappeared. The loss curves are shown in the dashboard.

## Limitations & next steps

1. **Modest accuracy (AUC 0.67).** This version of the data has few variables.
   *Next:* use the full Lending Club file and add credit-history variables that are
   also known at application time (credit card usage, recent credit checks, past
   late payments).
2. **2015 was riskier than the model expected** (18% predicted vs 20% actual).
   *Next:* retrain regularly on the most recent loans.
3. **The costs are my assumptions, and the chosen threshold rejects 48% of
   applicants**, which no real lender would accept. On the test set, a threshold of
   0.20 costs almost the same ($12.47M) and rejects only 36%. *Next:* estimate the
   real loss and interest figures from the full data, and add a minimum approval
   rate.
4. **We only see loans that were approved.** People Lending Club rejected are not in
   the data, so the model never sees the riskiest applicants.
5. **Missing loan term and an external `Default` label.** 3-year and 5-year loans
   behave differently, and I could not check exactly how `Default` was defined.
   *Next:* rebuild the target from the raw data and add the term.
6. **Fairness.** Location can stand in for protected characteristics, and state adds
   nothing, so the next version should drop it.

## Generative AI use disclosure

I used Claude (Anthropic) to help set up the environment, write the code
(`prepare_data.py`, `pipeline.py`, `train.py`, `app.py`), explore the data, propose
the default modeling choices, and draft this report. I reviewed every decision and
result and can explain each of them.
