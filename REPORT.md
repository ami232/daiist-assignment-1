# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Pablo Torres
- **Student ID**: 19910
- **Email**: ptorres.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A | BBADBA 5B]

## Dataset

*What is it, where did you get it, what does one row represent, how many
rows/columns, and why did you pick it.*

**HMEQ (Home Equity)**, from Kaggle (kaggle.com/datasets/ajay1735/hmeq-data),
stored in `data/hmeq.csv`: 5,960 rows, 13 columns, under 1 MB. 

**One row = one applicant for a home-equity loan** (a loan guaranteed by the
applicant's house). The target `BAD` is 1 if the applicant defaulted or became
seriously delinquent, and 0 if they repaid. 19.9% of rows are `BAD = 1`.

| Column | Meaning |
|---|---|
| BAD | Target: 1 = defaulted, 0 = repaid |
| LOAN | Loan amount requested |
| MORTDUE | Amount still owed on the current mortgage |
| VALUE | Current value of the property |
| REASON | Purpose: debt consolidation or home improvement |
| JOB | Job category |
| YOJ | Years at current job |
| DEROG | Number of major derogatory credit reports |
| DELINQ | Number of delinquent credit lines |
| CLAGE | Age of the oldest credit line (months) |
| NINQ | Number of recent credit inquiries |
| CLNO | Number of credit lines |
| DEBTINC | Debt-to-income ratio |

**Why I chose it:**
- **I am interested in default risk in private banking.** Deciding who will
  and will not repay a loan is a core banking problem. HMEQ is retail
  home-equity lending rather than private banking, but it is a clean, public
  dataset to learn the same ideas on.
- **The two kinds of mistake cost very different amounts.** Lending to
  someone who does not repay loses a lot of money, while turning away a good
  client only loses some profit. This makes the choice of decision threshold
  meaningful.
- **The data is messy** (gaps, very uneven amounts, text columns), so
  preparing it well really matters.
- **It is small and easy to explain.** It fits in the repo, trains in
  seconds, and every column has a clear business meaning.
- **A simple model fits the problem.** The Kaggle description says a bank must
  be able to explain why it rejects an applicant, and logistic regression is
  easy to explain.

## Business / real-life framing

*The hypothetical scenario this model serves, and what that scenario implies
for how you built the pipeline — target definition, whether a time-based
split was necessary and why (or why not), which metric should drive the
decision threshold and why.*

A bank must decide whether to approve or reject each home-equity loan application, and the model gives the probability that the applicant will default. The two mistakes cost very different amounts (my own assumptions): 
- approving someone who defaults loses about 10,000 EUR, 
- rejecting someone who would have repaid loses about 1,500 EUR.
 So the goal is to lose as little money as possible, not to be right as often as possible, and I judge models by average cost per applicant (accuracy would mislead, since approving everyone is already ~80% accurate). The target `BAD` was already a clean yes/no label, and because only about 1 in 5 applicants defaults, I kept that proportion in every split. The data has no dates, so I used a random 60/20/20 split instead of a time-based one, which means I cannot show how the model would behave on future applicants. The decision threshold is not the usual 0.5: in theory it should be 1,500 / (1,500 + 10,000) = 0.13, and in practice I pick it on the validation set by choosing the cheapest one, using the test set only once at the end.

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

I split the data first and learned everything below from the train set only, then applied it unchanged to validation and test. The biggest data-quality issue is missing values: on train, 750 applicants have no debt-to-income ratio (`DEBTINC`) and 62.9% of them defaulted versus 8.5% of the rest, so filling the gap alone would hide a strong clue. I therefore added "was missing" flags for `DEBTINC` and `VALUE` (the latter is only ~65 rows, so weaker evidence) plus a count of missing values per applicant, and filled the remaining gaps with the median, which is not thrown off by extreme values. Loans of 10,000 or less default far more (30.7% vs 17.2%), and a straight-line model cannot capture that jump, so I added a "small loan" flag (the 10,000 cut-off is a round number chosen by looking at train). I also added two ratios common in banking, loan-to-property-value and mortgage-to-property-value; on train the first showed no clear pattern, so I treat it as a hypothesis, not a proven gain. The very skewed amounts (`LOAN`, `MORTDUE`, `VALUE`, `CLAGE`) were log-transformed, the 16 extreme `DEBTINC` values above 60 were capped at 60, numbers were put on the same scale (needed for gradient descent and so the regularisation treats all features equally), and `REASON` and `JOB` were turned into 0/1 columns with a separate "Missing" category, ending with 26 columns.

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

I used **logistic regression**, because the target is yes/no and the bank must be able to explain why it rejects someone. I built the same model three ways: scikit-learn, a PyTorch loop written by hand (compute the prediction, measure the error, let autograd find the gradients, then update the weights myself), and the standard PyTorch way (`nn.Module` with an optimiser). To make them truly comparable, all three use the same train/validation/test split, the same loss and the same regularisation strength, chosen from {0.01, 0.1, 1, 10} on validation log loss. I did not tune the learning rates for score (1.0 for the manual loop, 0.05 with Adam); I only checked that training had converged by comparing against scikit-learn.

| Model | AUC | Log loss | Threshold | Cost per applicant (EUR) |
|---|---|---|---|---|
| scikit-learn | 0.8829 | 0.3075 | 0.15 | 551 |
| Manual PyTorch | 0.8829 | 0.3075 | 0.15 | 551 |
| nn.Module + Adam | 0.8829 | 0.3075 | 0.15 | 551 |
| Baseline: approve everyone | 0.50 | 0.4999 | - | 1,997 |
| Baseline: reject everyone | 0.50 | 0.4999 | - | 1,201 |

All numbers are on the test set; the threshold was chosen on validation (0.15, close to the theoretical 0.13). The model loses 54% less money than the better naive rule (rejecting everyone) and 72% less than approving everyone.

**Do the three agree? Yes.** Their predicted probabilities differ by at most 0.0005, which is too small to change a single decision, so AUC, log loss and cost are identical. This is expected: it is one problem with a single best answer, and the three methods are three different ways of walking to it, stopping at very slightly different points. It also depends on training long enough: with scikit-learn's default stopping setting, its probabilities differed from the PyTorch ones by up to , so I tightened it .

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Limitations & next steps

- **Small validation and test sets.** Each has only about 238 defaulters, so the chosen threshold (0.15, versus 0.13 in theory) and the cost in euros are rough estimates, not exact figures. With more data I would choose the threshold with cross-validation and report a margin of error.
- **The costs are my assumptions** (10,000 and 1,500 EUR). The best threshold moves with them, so with real figures from a bank I would recompute it.
- **Missing data may be a quirk of how the data was collected.** A missing debt-to-income ratio predicts default here, but it probably reflects the bank's process (for example, unverified applications), not something about the borrower. If that process changed, the model would get worse. I would ask the data owner why it is missing.
- **No dates, so no test on the future.** A random split cannot show whether the model keeps working on newer applicants. With dated data I would train on older loans and test on newer ones.
- **Some feature choices are not proven.** The 10,000 cut-off for "small loan" was chosen by looking at train, and the loan-to-value ratio showed no clear pattern there. 
- **A straight-line model has a ceiling.** Published results on this dataset report that tree-based models (random forest, gradient boosting) reach a clearly higher AUC (about 0.92 to 0.98) than logistic regression (about 0.79), although the splits and preparation differ, so those numbers are not directly comparable to my 0.883. The assignment requires logistic regression, so I did not try them, but they are the obvious next step.

## AI use disclosure

I used an AI assistant (Claude) while working on this assignment.

**What the AI did**
- Explained the steps of the assignment and helped me plan the workflow.
- Suggested candidate datasets acording to the requirements and my interests.
- Assisted to draft code for the train/validation/test split, the feature engineering and preprocessing, the three model implementations, the threshold and cost analysis, and the Gradio dashboard.
- Searched the web for published results on this dataset, used only as context in the Limitations section.
