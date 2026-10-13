# Assignment 1 Report

- **Name**: Javier Comin
- **Student ID**: 20106
- **Email**: jcomin.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

UCI "Default of Credit Card Clients" (dataset id 350): clients of a Taiwanese bank with six months of history (April to September 2005) and whether they defaulted in October 2005. One row is one client. There are 30,000 rows and 25 columns (an ID, 23 inputs and the target), and the default rate is 22.1%. I converted the original `.xls` to CSV (2.9 MB, `data/credit_default.csv`). I chose it because the target is binary with a natural business cost, the six-month payment history leaves room for feature engineering, and it retrains in about 40 seconds.

## Business / real-life framing

**Scenario.** Each month a card issuer flags high-risk clients (for example, cutting the limit or sending them to early collections) using only their last six months of repayment behaviour. The target is "defaults next month"; the dataset already defines it, and all inputs precede it.

**Costs.** A missed default (false negative) costs 3 and wrongly flagging a good client (false positive) costs 1. The ratio is an assumption. At 3:1 the break-even threshold is 0.25, just above the 22.1% base rate, so "approve everyone" (0.664 per client) is the cheapest constant policy. That is the lender's status quo, a meaningful baseline. At 5:1 the cheapest constant policy would be "reject everyone", which no lender does. The model's saving depends strongly on the ratio: on validation it is 38.7% at 3:1, 28.1% at 5:1 and 7.5% at 10:1.

**Split.** Random, stratified on the target, 60/20/20 (18,000 / 6,000 / 6,000). A time-based split was not possible: the data is one snapshot with no dates, and IDs are unique, so no client appears in two splits.

**Threshold.** It minimises business cost, chosen on validation and evaluated once on test. Accuracy would mislead: approving everyone already gets 77.9%.

## Data preparation & feature engineering

**Excluded:** ID, SEX and MARRIAGE (protected characteristics or close proxies; using sex in credit decisions is restricted in the EU). Putting them back raised validation AUC by only 0.0013 and 0.0008. I kept AGE and EDUCATION.

**Data quality.** EDUCATION codes 0, 5 and 6 (345 clients) are undocumented tiny groups, so I merged them into 4 ("others"). Negative BILL_AMT (1,930 rows) are credit balances after overpaying, so I left them as they are. Preprocessing is fitted on the training split only, and one category is dropped per one-hot variable, otherwise the dummies are collinear with the intercept.

**Features** (tested on validation only):
- `max_delay`: worst delay in the six months.
- `n_months_delayed`: months late (default rate 11.7% with 0, 70.3% with 6).
- `severe_delay_last`: last month 2+ months late (69.6% default rate against 17.0%).
- `log_limit`, and `log1p` of the six PAY_AMT columns, both heavy-tailed.

**Impact.** Validation AUC went from about 0.724 to 0.772, almost all from the delay features (removing them gives 0.742).

**What did not work.** Credit utilisation and payment-ratio features changed AUC by less than 0.001, six more candidates by at most +0.0005, and a log of BILL_AMT by +0.001. The standard error of AUC here is about 0.01, so these are noise and I dropped them (`experiments/feature_ablation.py`).

## Modeling: three implementations, one model

I used **logistic regression**: the target is binary and a cost-based threshold needs probabilities. All three minimise the same objective, mean log loss + (lambda / 2) * ||w||^2 with lambda = 1 / (C * n) and an unpenalised intercept.
- **scikit-learn:** `LogisticRegression`, C = 1 (the smallest C within 1e-4 of the best validation log loss).
- **Manual loop:** raw tensors, hand-written loss, autograd and a manual update; full-batch gradient descent, 5,000 steps, lr = 1.0.
- **`nn.Module` + `optim`:** `nn.Linear`, `BCEWithLogitsLoss`, Adam, batches of 512, 60 epochs, lr = 0.01, `weight_decay` on the weights only.

Test results (6,000 clients, threshold chosen on validation):

| Model | Threshold | ROC AUC | PR AUC | Log loss | Recall | Precision | Cost per client |
|---|---|---|---|---|---|---|---|
| scikit-learn | 0.26 | 0.7659 | 0.5354 | 0.4387 | 0.552 | 0.508 | 0.4157 |
| manual PyTorch | 0.26 | 0.7659 | 0.5354 | 0.4387 | 0.551 | 0.508 | 0.4160 |
| nn.Module + optim | 0.27 | 0.7660 | 0.5360 | 0.4384 | 0.531 | 0.519 | 0.4198 |
| naive baseline (approve everyone) | - | 0.5000 | 0.2212 | 0.5284 | - | - | 0.6635 |

**Do they agree?** Yes. The objective gaps against scikit-learn are -4e-7 (manual) and +1.2e-4 (nn), the largest coefficient differences are 0.007 and 0.055, and AUC differs by 0.0001. The nn's higher cost (0.4198 against 0.4157, about 25 cost units) I read as noise: it uses a different threshold (0.27 against 0.26).

At first they did not agree: coefficients differed by up to 0.28 (manual) and 2.4 (nn). Four causes:
1. One-hot dummies collinear with the intercept, fixed by dropping one category.
2. EDUCATION = 0 had 11 training clients and no defaults, so with weak regularisation its coefficient drifts towards minus infinity. Merging rare categories fixed it.
3. Minibatch noise in the nn's Adam: full-batch training cut the mean probability difference from 0.0077 to 0.0004.
4. Extreme PAY_AMT values: `log1p` cut the largest probability gap from 0.15 to 0.04.

## Limitations & next steps

- **One snapshot, no time dimension.** I cannot test drift, and a 22% default rate is probably not representative today. I would use data from several periods and a time-based split.
- **Flat, assumed cost ratio.** I would use real margin and loss given default, scaled by each client's exposure.
- **One test split, no uncertainty.** I would add bootstrap confidence intervals for AUC and cost.
- **Only a linear model and hand-made features.** I would benchmark against gradient boosting to see what the linear form misses.
- **Fairness only partly handled.** AGE and EDUCATION can still act as proxies; I would audit flag and error rates by group.

## Generative AI use disclosure
I used AI to help explain the project and track errors. I also used it to fix some mistakes when I didn't know what was going wrong, and to check that the entire project was complete before submitting it.
