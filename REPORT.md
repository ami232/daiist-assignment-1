# Assignment 1 Report



- **Name**: Yago Moreno
- **Student ID**: 19134
- **Email**: ymoreno.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

**Source.** UCI Machine Learning Repository, "Bank Marketing" (#222), Moro, Cortez & Rita (2014). Phone campaigns of a Portuguese bank selling term deposits, May 2008 to Nov 2010. 41,188 rows, 20 inputs plus the target `y` (subscribed yes/no, 11.3% yes). I removed 12 exact duplicates, leaving 41,176 rows. Each row is one client at their last contact. Rows are in date order but there is no year column.

**Why this dataset.** I chose it for four main reasons. First, the target is a real decision with money attached: whether a phone call is worth making, with a clear cost per call and value per sale. Second, the rows follow the calendar, so how to split the data is a genuine modelling decision rather than a default, and a naive split fails visibly. Third, the data contains realistic traps (a column known only after the call, a code that looks like a number, many "unknown" values), so feature engineering has real content. Fourth, it is a binary target with an imbalanced class, which makes a logistic regression comparison across three implementations meaningful. The 41k rows are above the brief's rough guide of about 20k, but the brief calls that guide "not a strict cutoff", the file is only 5.8 MB, and the full training run takes about 2.5 minutes.

**What I noticed in the data.**

- `duration` (call length) is only known after the call ends, so it can't be used to decide whether to call.
- `pdays = 999` is a code for "never contacted" (96% of rows), not a real number of days.
- `default = "yes"` appears only 3 times, but "unknown" appears for 21% of clients.
- `housing` and `loan` are "unknown" on the same 990 rows.
- The five macro-economic columns have the same value for every client in a given month, and three of them (`euribor3m`, `emp.var.rate`, `nr.employed`) are 0.91-0.97 correlated.
- The campaign changed a lot over time:

| Period | Calls | Subscription rate |
|---|---|---|
| May-Nov 2008 | ~27,700 | 3-6% |
| Mar-May 2009 | ~8,500 | 9-45% |
| Jun 2009-Nov 2010 | ~4,500 | 35-57% |

Call volume dropped about 10x while the success rate rose from about 5% to about 50%. This is consistent with a shift to targeted calling, but the data doesn't say so directly; it is my interpretation of the pattern.

## Business / real-life framing

**Decision.** Before each call, a call centre decides whether this client is worth phoning. The model gives a probability of subscription for each client.

**Economics (assumed, editable in the dashboard).** €5 per call, €50 per subscription. A call pays off on average if P(subscribe) > 5/50 = 10%.

**How the framing changed my pipeline:**

| Framing point | Decision it drove |
|---|---|
| The model must work before dialling | `duration` is excluded. The current call is counted in `campaign`, so I use `campaign - 1` (prior contacts) |
| The model is trained on past campaigns and used on the next one | Chronological split, not a random one |
| Accuracy means little with 11% positives, and what matters is money | The threshold is chosen by expected profit on validation |
| The call centre has limited capacity | Results are also reported when only the top 10/25/50% of clients can be called |

## Data preparation & feature engineering

### Split

My first attempt was a 70/15/15 split by row order. Test ROC-AUC was 0.479, worse than random, and the mean prediction was 13.6% against an actual rate of 38.4%.

**Why it failed.** The training rows were almost all from the 2008 mass-calling period, when only about 5% of clients subscribed. The macro columns take only about 26 distinct values (one set per month), so the model could use them to work out which month a row came from. In the test period (May 2009 - Nov 2010) every test row had a `euribor3m` value below anything seen in training, and about 45% of rows were also out of range for `cons.conf.idx` and about 41% for `cons.price.idx`, so the model extrapolated, and some relationships even reversed: low consumer confidence went with "yes" in training and with "no" in test. The result was a mean prediction of 13.6% when the real rate was 38.4%, and a ranking worse than random. The calendar split fixes most of this because training includes part of the new period: on the test set, `euribor3m` is never out of range (0%), though `cons.price.idx` (28%) and `cons.conf.idx` (11%) still partly are.

I switched to a calendar-based split:

| Split | Period | Rows | Subscription rate |
|---|---|---|---|
| Train | May 2008 - Dec 2009 | 39,118 | 9.1% |
| Validation | Mar - Jun 2010 | 879 | 54.2% |
| Test | Jul - Nov 2010 | 1,179 | 50.6% |

There are no rows for Jan-Feb 2010. Training now includes part of the new period, so the model has seen macro conditions similar to those in 2010. On the test rows, 0% of `euribor3m` values are outside the training range, compared with 100% in the first attempt. `cons.price.idx` (28%) and `cons.conf.idx` (11%) are still partly out of range, so the calendar split removes the `euribor3m` extrapolation entirely but not all extrapolation. I chose this split after seeing the first one fail, so the test set is not completely untouched. As a diagnostic, a random split gives ROC-AUC 0.788, which is optimistic because 2010 rows end up in training and the model effectively sees the future. The year is reconstructed from the month order (a decrease such as Dec to Mar marks a new year) and is used only for splitting, never as a feature.

### Features

All transforms in `engineer()` are row by row and learn nothing from the data. Anything learned (means, standard deviations, category lists) is fit on train only.

| Feature | Built from | Reason |
|---|---|---|
| `age_band` (<25, 25-29, 30-39, 40-49, 50-59, 60+) | age | The effect is U-shaped; one coefficient per band can capture that, a straight line cannot |
| `job`, `marital`, `day_of_week` | one-hot | Categorical |
| `education` | "illiterate" merged into "basic.4y" | Only 18 clients |
| `default_unknown` | `default == "unknown"` | "yes" has only 3 cases, so not disclosing carries the signal |
| `housing_yes`, `loan_yes`, `loan_info_unknown` | housing, loan | Flags; one "unknown" flag because both are unknown on the same rows |
| `cellular` | contact | Mobile vs landline |
| `log_prior_contacts` | `log1p(campaign - 1)` | Known before the call; log because of the long tail (max 56) |
| `recent_contact` | `pdays != 999` | 999 is a code; as a number it would be the largest value and dominate |
| `days_since_contact_capped`, `previous_capped` | pdays (cap 30, 0 if never), previous (cap 3) | Recency for contacted clients; stops rare high counts dominating |
| `poutcome` | one-hot | Outcome of the previous campaign |
| `euribor3m`, `cons.conf.idx`, `cons.price.idx` | as-is | Macro context, known before dialling |

**Left out on purpose:**

- `duration`: leakage.
- `month`: here it works as a stand-in for the time period, not as a seasonal effect (for example, "October" in training is 67 rows from 2008). Adding it hurt in the ablation.
- `emp.var.rate`, `nr.employed`: highly correlated with euribor, and they hurt in the ablation.
- A `contacted_before` flag: identical to `poutcome != "nonexistent"`.

### Ablation (validation set, scikit-learn, C = 1)

| Variant | Columns | ROC-AUC | PR-AUC | Log loss |
|---|---|---|---|---|
| Raw columns, no engineering | 52 | 0.686 | 0.710 | 0.637 |
| **Final set** | 43 | 0.706 | 0.725 | 0.629 |
| minus client | 18 | 0.712 | 0.737 | 0.625 |
| minus credit | 39 | 0.706 | 0.724 | 0.629 |
| minus contact | 37 | 0.685 | 0.710 | 0.704 |
| minus history | 38 | 0.586 | 0.585 | 0.684 |
| minus macro | 40 | 0.681 | 0.713 | 0.853 |
| plus month | 52 | 0.634 | 0.670 | 0.727 |
| plus collinear macro | 45 | 0.698 | 0.718 | 0.754 |

History matters most, and contact and macro clearly help. Month and the collinear macro columns hurt. The credit group makes no difference. Removing the client group scores slightly better (+0.006 ROC-AUC), but this is within the noise (about ±0.015 on a set this small), so I kept it.

### Preprocessing

Categoricals are one-hot encoded, then every column (dummies included) is standardised using training statistics. I standardised the dummies because rare ones, such as `poutcome_success` (~2% of rows), made the problem badly conditioned, and gradient descent converged very slowly. After standardising, the manual loop matches scikit-learn. The cost is that coefficients now mean "change in log-odds per 1 standard deviation".


## Modeling: three implementations, one model

I use logistic regression because the target is binary (subscribed or not) and I need a probability for each client to apply the cost threshold; L2 regularisation keeps the coefficients stable given the correlated features. It is trained three ways on the same 39,118 x 43 training matrix: scikit-learn, a manual PyTorch loop (full-batch gradient descent, hand-written loss and update), and PyTorch `nn.Module` + `optim`. The PyTorch versions add an L2 penalty written so that all three minimise the same objective as scikit-learn's `C`.

**Tuning.** C = 1, chosen from {0.001, 0.01, 0.1, 1, 10} by validation log loss, taking the strongest regularisation within 0.001 of the best (C = 1 and C = 10 are indistinguishable on 879 rows). Learning rate: 1.0 for the manual loop (2,000 epochs), 0.01 for the `nn.Module` version (Adam with a cosine schedule, 60 epochs). The schedule is needed because mini-batch noise otherwise leaves the weights jittering around the optimum; in earlier development runs, probabilities without it were noticeably further from scikit-learn's.

### Three-method comparison (test set: Jul-Nov 2010, 1,179 calls, 50.6% subscribed)

| Model | ROC-AUC | PR-AUC | Log loss |
|---|---|---|---|
| scikit-learn | 0.7445 | 0.7409 | 0.5997 |
| PyTorch, manual loop | 0.7448 | 0.7406 | 0.5995 |
| PyTorch, `nn.Module` + Adam | 0.7446 | 0.7405 | 0.5997 |
| Naive: constant 9.1% | 0.5000 | 0.5064 | 1.2600 |

The naive baseline's log loss is poor because 9.1% is far from the test rate of 50.6%.

### Agreement with scikit-learn

| Model | Max abs. coef. diff | Intercept diff | Max prob. diff | Mean prob. diff |
|---|---|---|---|---|
| Manual loop | 0.0068 | 0.0006 | 0.0102 | 0.0020 |
| `nn.Module` + Adam | 0.0066 | 0.0008 | 0.0098 | 0.0022 |

The three models agree closely because the objective is convex and they use the same data, scaling and penalty. The remaining small gap is probably due to scikit-learn's L-BFGS stopping at its default tolerance (1e-4); during development a tighter tolerance closed it.

### Business results (test set, €5 per call, €50 per subscription)

| Capacity | scikit-learn model | Precision | Random calls |
|---|---|---|---|
| 10% (117 calls) | €4,265 | 82.9% | €2,377 |
| 25% (294 calls) | €9,930 | 77.6% | €5,974 |
| 50% (589 calls) | €17,255 | 68.6% | €11,967 |
| 100% (everyone) | €23,955 | 50.6% | €23,955 |

The profit-maximising threshold on validation is 0.01, i.e. call everyone, because 54% of validation clients subscribe and almost every call pays off. With unlimited capacity the model adds nothing. Its value is in ranking when capacity is limited: at 25% capacity it earns 66% more than random calling.

Coefficient note: the largest coefficient is `euribor3m` (-0.89), but I read it mostly as a stand-in for the 2009-10 targeted-calling period, not as a causal effect. Correlated features share credit (for example `poutcome_success` and `recent_contact`), so coefficients should not be read causally.

### Dashboard

The Gradio app loads the saved models and never retrains. It has tabs for the model comparison (including predicted vs actual), feature and target distributions, and a threshold and capacity slider with confusion matrix and business cost.


## Limitations & next steps

- **Regime change and selection bias.** The bank's calling policy changed over time, and another change would break the model again. *Fix:* retrain on a rolling window and track calibration (predicted vs actual rate) every month.
- **Macro features act partly as a time proxy.** They help only while new values stay within the training range, and this is already partly violated in the final test set (28% of `cons.price.idx` and 11% of `cons.conf.idx` values are out of range). *Fix:* keep a version without macro features as a sanity check, and check that new macro values fall within the training range before trusting the model.
- **Small validation and test sets** (about ±0.015 AUC). *Fix:* walk-forward evaluation over several consecutive test periods, reporting the spread of results instead of one number.
- **The split was chosen after a failed first attempt,** so the test set is not completely untouched. *Fix:* in a real project, fix the split rule from the deployment setting before looking at any results, and keep one final period unseen until the end.
- **Costs are assumed** (€5 / €50). *Fix:* ask the bank for its real cost per call and value per subscription, then rerun the threshold and capacity analysis (the dashboard already lets these be edited).
- **Final models are fit on train only.** *Fix:* once the settings are chosen, refit on train + validation so the model includes the most recent months.
- **Linear model:** interactions are included only if engineered by hand. *Fix:* try one or two interactions, for example `cellular` x `recent_contact` or `age_band` x `job`, and keep them only if validation improves.

**Which matters most.** In my view, regime change. The model's good test result depends on the bank's calling policy in mid-to-late 2010 resembling the end of the training period, and a policy change would break it without warning. That is why I would put the rolling retraining and monthly calibration check first if the bank deployed it.

## Generative AI use disclosure

Used Claude for discussion on which methodology to follow. Also, helped me compare different datasets, as well as solve any coding problem I encountered, helping me see the error and developing a functional code.
