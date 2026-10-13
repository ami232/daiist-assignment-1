# Assignment 1 Report

- **Name**: Santiago Llorca
- **Student ID**: 19952 
- **Email**: sllorca.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

The data comes from the public Lending Club loan dataset on Kaggle
(`accepted_2007_to_2018Q4`, accepted loans only). Each row is one consumer
loan, with the borrower's application-time information and the final outcome
of the loan.

From the full file (~2M loans) I kept only 36-month loans issued between
January 2012 and December 2014. A 36-month loan issued by the end of 2014 had
reached its final outcome by the time the data was closed in 2018, so the
target is reliable. I then drew a random sample of 18,000 loans (seed 42) so
the file is small enough to commit and retrain from scratch (2.9 MB). The
result has 18,000 rows and 26 columns, with a default rate of 13.4%.

I picked it because it has a natural business decision (approve or reject),
a target I have to define myself, a clear time dimension, and a mix of
numeric and categorical variables where feature engineering matters.

## Business / real-life framing

**Scenario.** A consumer lender must decide, for each new 36-month loan
application, whether to approve or reject it. The model estimates the
probability that the loan will default, and the loan is approved if that
probability is below a threshold.

**Target definition.** `default = 1` if the loan ended as "Charged Off" and
`0` if "Fully Paid". Loans in other statuses (e.g. Current, Late) are dropped
because their outcome is not final.

**Only application-time information.** The model must only use what the lender
knows at the moment of the decision. Post-origination columns (payments
received, recoveries, last payment date) are excluded because they leak the
outcome. I also exclude `int_rate`, `grade` and `sub_grade`: these are
produced by Lending Club's own scoring model, so using them would mean
inheriting its answer instead of building an independent assessment of the
applicant.

**Asymmetric costs, which set the threshold.**
- Approving a loan that defaults (false negative) costs roughly
  LGD × loan amount, where LGD is the share of the principal not recovered.
- Rejecting a loan that would have been repaid (false positive) costs the net
  interest margin I would have earned.

Assumptions (not taken from the data, to be varied in the dashboard):
LGD = 65% and net margin = 10% of the loan amount. A loan is worth
approving when (1 − p) × margin > p × LGD, i.e. when p < margin /
(margin + LGD) ≈ 13%. So the threshold is chosen by **minimizing expected
cost on the validation set**, not fixed at 0.5. Because the default rate is
13.4%, accuracy is misleading (a model that approves everyone scores 86.6%),
so I also report ROC-AUC and PR-AUC, which do not depend on the threshold.

**Time-based split.** A random split would train on loans issued after some of
the loans I evaluate on, leaking the future. The split is by issue date:
train 2012–2013, validation January–June 2014 (to choose regularization,
learning rate and threshold), test July–December 2014 (used once at the end).

**Baselines.** (1) Approve every loan. (2) A single-variable rule on FICO
score.

## Data preparation & feature engineering

**Split and leakage rules.** Loans are split by issue date (train: 8,549
rows; validation: 4,176; test: 5,275). Every quantity learned from data
(medians for imputation, 1st/99th percentile clip bounds, the list of frequent
categories, mean and standard deviation for scaling) is fitted on the training
set only and reused unchanged on validation and test. The default rate drifts
upward over time (12.8% train, 13.4% validation, 14.3% test), which a random
split would have hidden.

**Columns excluded.**
- `term`: constant (all loans are 36 months).
- `issue_d`: used only for the split.
- `int_rate`, `grade`, `sub_grade`: outputs of Lending Club's own scoring
  model (see framing).
- `installment`: with a fixed term it is a monotonic function of the interest
  rate relative to the loan amount, so it would leak `int_rate` through the
  back door. Loan burden is captured by `loan_to_income` instead.
- `addr_state`: 50 levels on ~8.5k training rows leave many states with very
  few loans, so I left it out by design (not tested empirically).

**Features built (27 columns after encoding).**

| Feature | Construction | Rationale |
|---|---|---|
| `fico_mid` | mean of `fico_range_low/high` | Most direct solvency signal; the two columns are near-duplicates |
| `loan_to_income` | `loan_amnt / annual_inc` | Size of the request relative to the borrower's income |
| `log_annual_inc`, `log_loan_amnt` | natural log | Both are right-skewed; linear models handle logs better |
| `credit_history_years` | years between `earliest_cr_line` and `issue_d` | Length of credit history |
| `emp_length_num`, `emp_length_missing` | "< 1 year" to 0, "10+ years" to 10; missing imputed with the train median plus an indicator | Missing employment length is informative: 20.1% default vs 13.0% for the rest (949 loans) |
| `has_delinq_history`, `mths_since_delinq` | indicator of a recorded delinquency; months imputed with the train median | 52.7% of values are missing; I treat a missing value as "no recorded delinquency". The data only weakly supports this (12.9% default vs 13.9%) |
| `log_revol_bal`, `revol_util`, `dti` | log1p of the balance; others used directly | Pressure on revolving credit and debt burden |
| `delinq_2yrs`, `inq_last_6mths`, `pub_rec_any` | counts clipped at the 99th percentile; `pub_rec` as a binary flag | Long-tailed counts |
| `home_ownership`, `verification_status`, `purpose` | one-hot; categories below 1% of the training set grouped as "rare"; most frequent level is the reference | Few useful levels, avoids unstable dummy columns |

**Other data-quality decisions.** Winsorizing at the 1st/99th percentile
(`loan_to_income`, `credit_history_years`, `revol_util`, `dti`, `delinq_2yrs`,
`inq_last_6mths`, `log_revol_bal`) limits the influence of extreme values.
All features are standardized, which matters for the PyTorch models: with
unscaled inputs, gradient descent would converge at very different speeds per
feature.

**Caveats found in the results.**
- `loan_to_income`, `log_annual_inc` and `log_loan_amnt` are linearly related
  (the log of the ratio is the difference of the other two), so individual
  coefficients are not interpretable as separate effects. L2 regularization
  keeps predictions stable despite this.
- `has_delinq_history` gets a negative coefficient even though raw default
  rates are slightly higher for borrowers with a delinquency record. The
  coefficient is conditional on `fico_mid`, which already captures that
  information.
- - `verification_status=rare` has no visible bar in the dashboard's coefficient
  plot: no verification level falls below the 1% threshold (the smallest has
  5,418 of 18,000 loans), so that column is all zeros and its coefficient is
  zero. It is harmless but redundant.



## Modeling: three implementations, one model

**Model.** L2-regularized logistic regression, because the target is binary
(default / no default). The three implementations use the same split, the same
preprocessed matrices and the same regularization:
1. **scikit-learn** `LogisticRegression` (L-BFGS, `tol=1e-8`).
2. **Manual PyTorch loop**: raw tensors, `binary_cross_entropy_with_logits`
   plus an explicit L2 term, `loss.backward()`, a manual update under
   `torch.no_grad()` and manual gradient zeroing; full-batch, 3,000 iterations.
3. **Standard PyTorch**: `nn.Linear` + `BCEWithLogitsLoss` + `torch.optim.SGD`
   with `weight_decay`; same iterations.

**Making the regularization equivalent.** scikit-learn minimizes
C x sum(log-loss) + 0.5 x ||w||^2. Dividing by C x n gives mean log-loss +
(1 / (2 C n)) ||w||^2, so both PyTorch versions use lambda = 1 / (C x n), with
n = 8,549. The bias is not penalized in any of the three.

**Hyperparameters (only these were tuned, on validation).** C was chosen from
{1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 0.1, 1} by validation log-loss: **C = 0.003**
(0.3824), an interior optimum. My first grid stopped at 1e-3, which was the
edge of the grid, so I widened it. The learning rate was chosen from
{0.1, 0.5, 1.0} for both PyTorch models; all three give the same validation
log-loss (0.3824), so the choice is a tie broken in favor of 0.1. The problem
is convex and regularized, so the learning rate only changes the speed of
convergence, not the solution.

**Results (test set, July-December 2014; threshold chosen on validation).**

| Model | Threshold | Val AUC | Test AUC | Test PR-AUC | Test cost ($) | Saving vs approve-all |
|---|---|---|---|---|---|---|
| Baseline: approve all | none | 0.500 | 0.500 | 0.143 | 6,299,865 | 0.0% |
| Baseline: FICO rule | reject if FICO <= 684 | 0.578 | 0.586 | 0.175 | 5,168,810 | 18.0% |
| scikit-learn | 0.134 | 0.625 | 0.638 | 0.221 | 4,801,419 | 23.8% |
| PyTorch manual loop | 0.134 | 0.625 | 0.638 | 0.221 | 4,801,419 | 23.8% |
| PyTorch nn.Module | 0.134 | 0.625 | 0.638 | 0.221 | 4,801,419 | 23.8% |

**Do the three agree?** Yes. The largest absolute difference between any
learned weight and the scikit-learn weight is about 1e-5, and the bias gap is
below 1e-5. With scikit-learn's default tolerance (`tol=1e-4`) the gap was
1.6e-4 in the weights and 7e-5 in the bias, and test cost was 4,800,579 for
scikit-learn vs 4,801,419 for both PyTorch models. The gap was the same for the
two PyTorch versions, so I suspected the solver stopped early. I tested it by
tightening `tol` to 1e-8 and 1e-12: the coefficients stop changing (checked on
the first three coefficients and the intercept), and after rerunning training
with `tol=1e-8` the gaps dropped to ~1e-5 and the cost became identical across
the three models. The regularized problem has a single optimum, so all three
converge to it regardless of initialization (zeros for the manual loop,
random for `nn.Linear` with a fixed seed).

**Reading the numbers.** The model beats both baselines on all metrics
(23.8% cost saving vs 18.0% for the FICO rule). An AUC of 0.638 is modest, as
expected after excluding `int_rate`, `grade` and `sub_grade`. In the
dashboard, rejecting everyone (5.74M) costs less than approving everyone
(6.30M) on the test set, because the test default rate (14.3%) is above the
13.3% break-even.

## Limitations & next steps


- **Temporal drift.** The default rate rises across the splits (12.8%, 13.4%,
  14.3%). In the dashboard's decile plot the observed default rate is above
  the predicted one in the upper deciles of the test set, i.e. the model
  slightly underestimates risk. The threshold chosen on validation (0.134) is
  higher than the best one in hindsight on test (0.117). Next step: refit on a
  rolling window, recalibrate the scores, and monitor the default rate against
  the break-even.
- **Cost assumptions.** LGD = 65% and margin = 10% are assumptions, not
  estimates. Next step: estimate recovery and realized interest from the
  payment columns of the raw data (`recoveries`, `total_pymnt`), which are
  valid for measuring outcomes but not for use as features, and use
  loan-specific rather than flat margins.
- **Selection bias.** The data only contains loans that Lending Club approved
  (FICO scores start around 660), so the model never saw the rejected
  applicants and may behave differently on them. Next step: reject inference
  with the rejected-loans file, or a small randomized-approval experiment.
- **Modest discrimination.** A linear model with excluded lender-score
  variables reaches AUC 0.638. Next step: interactions or a tree-based model
  as a benchmark, though that would go beyond this assignment's scope.
- **Small samples and no uncertainty estimates.** Train has 8.5k loans,
  validation 4.2k (used to choose C and the threshold) and test 5.3k; I report
  no confidence intervals. Next step: bootstrap intervals on the cost and AUC
  and use more of the original ~2M loans.
- **Residual leakage not measured.** I excluded `installment` and
  `int_rate`, but did not run an experiment adding `installment` back to
  measure how much it would inflate the AUC, nor an ablation by feature
  group, nor a test of `addr_state`. These are the first checks I would add.
- **Interpretability.** Because of the collinearity noted above, individual
  coefficients should not be read as isolated effects.

## Generative AI use disclosure

I used Claude (Anthropic) as a study and coding assistant throughout this
assignment:
- **Dataset and business case.** Claude proposed candidate datasets and design
  options; I chose the Lending Club loan-default problem and the option of
  excluding the lender's own score (`int_rate`, `grade`, `sub_grade`).
- **Code.** Claude wrote the code for the data preparation script, the feature
  engineering module, the evaluation functions, the three training
  implementations and the Gradio dashboard. I ran all of it myself, checked the
  outputs against each other (for example the three-way comparison and the
  tolerance experiment) and reran the full pipeline from scratch.
- **Git/GitHub workflow.** Claude guided me through cloning my fork,
  branching, committing, pushing and opening the pull request.
- **Report.** Claude drafted the report text from the results of my runs; I
  reviewed it against my outputs.

All final decisions and the content of this report are my responsibility, and
I can explain each step of the pipeline.
 