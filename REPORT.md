# Assignment 1 Report

- **Name**: Juan Pablo Miro-Quesada
- **Student ID**: 19858
- **Email**: jmiroquesada.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A]

## Dataset

*I selected the Telco Customer for four reasons. First, it supports a realistic business decision: which customers a retention team should contact with an offer. The two kinds of error have very different costs, since losing a customer is far more expensive than giving an unnecessary discount, so the decision threshold has to be set by cost rather than left at the default 50%. Second, the raw data needs genuine cleaning and feature work, including TotalCharges stored as text with blank values, many categorical columns, a relationship between tenure and churn that is not linear, and add-on services that can be combined into new features. Third, at about 7,000 rows and under 1 MB, it is small enough to commit to the repo and retrain quickly. Fourth, its binary target suits logistic regression, which makes it a clean case for comparing scikit-learn, a manual PyTorch loop, and the standard nn.Module workflow.*

## Business / real-life framing

*This model supports a monthly retention campaign at a telecom company. Each month, the retention team scores every current customer and decides who receives a proactive offer, such as a discount or plan upgrade. The team cannot contact everyone, so the model’s job is to identify the customers most likely to leave. The two kinds of error have different costs. Missing a customer who leaves costs their future revenue, which I assume is about $400 (roughly $65 per month for six months). Offering a discount to someone who would have stayed costs about $50. These figures are assumptions, and a miss is eight times more expensive than a wasted offer. For that reason the decision threshold is chosen by minimizing expected cost on a validation set, not by accuracy or a default 50% cutoff, and under these numbers the cost-optimal threshold is roughly 11%. The target is the existing Churn column (Yes/No), which needed no redefinition. The dataset is a single snapshot with no dates, so a time-based split is not possible. I used a random split stratified on Churn into train, validation, and test sets, which keeps the roughly 26% churn rate in each part. The cost of this choice is that the results cannot show how the model would perform on future customers.*

## Data preparation & feature engineering

**Target:** the existing `Churn` column, no redefinition needed.
**Split:** 60/20/20, stratified on `Churn` (4,225 / 1,409 / 1,409 rows, 26.5% churn in each
  part). The data has no dates, so a time-based split was not possible. A random split cannot
  show how the model would do on future customers.
**Threshold:** chosen per model on the validation set by minimizing 400 x FN + 50 x FP over a
  grid from 0.01 to 0.99. The test set was used only for the final table. The theoretical
  threshold is 50 / 450 = 0.11, and the validation-chosen thresholds were 0.13 to 0.14 across
  runs (0.14 in the final run).*

Data quality:

`customerID` was dropped (identifier, no signal).
`TotalCharges` is stored as text. The blank rows were all customers with `tenure == 0`
  (verified with an `assert`, not assumed), so they were set to 0.

Tenure experiment (validation set, same logistic regression):

| Tenure form | Val AUC | Val log-loss |
|---|---|---|
| raw | 0.8583 | 0.4025 |
| log1p | 0.8632 | 0.3961 |
| 5 bins | 0.8596 | 0.4012 |

`log1p(tenure)` was kept. Churn by tenure bucket was 52.9% / 35.9% / 28.7% / 20.4% / 9.5%.

Other choices, tested against rules set before the run (simplifying variants adopted if
log-loss was no more than 0.001 worse; adding variants adopted only if at least 0.002 better):

| Variant | Change in log-loss | Decision |
|---|---|---|
| Electronic-check flag instead of 4 payment categories | +0.0001 | adopted |
| Merge "No internet/phone service" into "No" | +0.0001 | adopted |
| Contract as 0/1/2 instead of one-hot | +0.0001 | adopted |
| Add `TotalCharges` | 0.0000 | rejected |
| Add count of add-on services | 0.0000 | rejected |

Supporting numbers: churn by contract 42.7% / 11.3% / 2.8%; electronic check 45.3% versus
15-19% for other payment methods; `TotalCharges` correlated 0.83 with tenure and 0.65 with
`MonthlyCharges`.

Final inputs: 5 numeric columns (`tenure_log`, `MonthlyCharges`, `SeniorCitizen`, `is_echeck`,
`contract_ord`), standardized, plus one-hot columns with one reference category dropped, 19
columns in total. The scaler and encoder were fitted on the training split only.

## Modeling: three implementations, one model

All three are plain logistic regression, with no class weights, on the same split.

Test-set results (thresholds chosen on validation):

| Model | AUC | Log-loss | Cost | Per customer |
|---|---|---|---|---|
| scikit-learn | 0.8403 | 0.4206 | $35,900 | $25.48 |
| manual PyTorch | 0.8403 | 0.4206 | $35,900 | $25.48 |
| nn.Module + optim | 0.8403 | 0.4206 | $35,900 | $25.48 |
| naive (contact nobody) | 0.5 | 0.5787 | $149,600 | $106.17 |

Contacting everyone would cost $51,750. Against these, the model cuts cost by about 76%
(versus nobody) and about 31% (versus everyone).

Do the three agree? Not at first:

| Run | Settings | Max weight gap vs sklearn | Max probability gap | PyTorch gradient norm |
|---|---|---|---|---|
| 1 | C=10, 20 columns, 20k epochs | 1.21 | 3.3e-2 | 1.1e-4 (from `diagnose.py`) |
| 2 | `drop="first"`, C=30, 100k epochs | 0.33 | 1.1e-2 | 2.7e-5 |
| 3 | C=0.3, 300k epochs | 1.6e-6 | 2.6e-7 | 3.3e-16 |

The manual and `nn.Module` versions matched each other to about 1e-15 in every run.

What I found: with run 1's settings, the manual loop's objective approached sklearn's from above
as epochs rose (0.40722145 at 20k, 0.40713489 at 50k, 0.40710125 at 400k; sklearn 0.40710120).
Run 1's design matrix had a smallest eigenvalue of 1.6e-17, because the three `InternetService`
columns sum to 1 and duplicate the intercept. After the fixes it was 2.4e-4, the Hessian
condition number was 1.3e3, and the predicted epochs for a million-fold error reduction was
about 52,000. The remaining 1.6e-6 gap is sklearn's own stopping tolerance (its gradient norm is
6e-8). The PyTorch L2 strength is lam = 1/(C x n) = 7.89e-4, and the intercept is not
regularized in any of the three. **[WRITE: why does a near-singular problem slow gradient
descent, and why was C=30 worse than C=0.3?]**

Process notes: the rule for choosing C (smallest C within 0.001 of the best validation
log-loss) was introduced after seeing run 2, not before. The weight-agreement criterion was
loosened from 1e-3 to 1e-2 before run 3. One diagnostic run used stale code and was discarded.


## Limitations & next steps

Known limitations:

- No time dimension, so the random split cannot show performance on future customers.
- The $400 and $50 costs are assumptions, and the cost ignores what a successful offer earns
  or costs.
- Test AUC (0.840) is below validation AUC (0.863), with about 370 churners per split.
- The cost-optimal threshold is flat near its minimum (cost moved by $350 between runs).
- It is a sample dataset, a linear model, and a single split.

## Generative AI use disclosure

*Claude was used to better understand the assignment, help with the actual code, and better understand the output. I interacted with it to be more understanding of the outputs and see if I missed anything. It also helped me write more clear. "*
