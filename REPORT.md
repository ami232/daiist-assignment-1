# Assignment 1 Report

- **Name**: Gonzalo Angulo
- **Student ID**: 18620
- **Email**: gangulo.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

**UCI Online Shoppers Purchasing Intention** (UCI ML Repository, dataset 468), committed at
`data/online_shoppers_intention.csv`. 12,330 rows × 18 columns; one row is one browsing
session on an e-commerce site over one year. The target is `Revenue` (did the session end in a
purchase?), positive in 15.5% of sessions.

Columns: page counts and time spent on three page types (administrative, informational,
product), Google Analytics metrics (`BounceRates`, `ExitRates`, `PageValues`), `SpecialDay`
(closeness to a holiday), `Month`, `Weekend`, `VisitorType`, and four ID codes
(`OperatingSystems`, `Browser`, `Region`, `TrafficType`).

I picked it because it supports a concrete business decision, the class imbalance makes the
decision threshold matter, and the raw columns (heavily skewed counts and durations, ID codes
stored as integers, a possibly leaky metric) mean feature engineering actually changes results.

## Business / real-life framing

**Decision supported:** the shop shows a "10% off / free shipping" pop-up to sessions the model
flags as likely buyers, to stop them abandoning before checkout.

**Cost of errors (assumptions):** average order €80 at ~25% margin ≈ €20 margin per order.
- **False negative** (likely buyer gets no pop-up): assuming half of them abandon anyway,
  expected lost margin ≈ **€10**.
- **False positive** (pop-up shown to a non-buyer): coupon leakage and sharing, brand
  dilution, training customers to wait for discounts ≈ **€2**.

A missed buyer is 5× worse than a wasted pop-up, so the theoretical break-even threshold is
2 / (2 + 10) ≈ 0.17, not 0.5. The model is judged on **total € cost on the test set**, not
accuracy. The threshold is tuned on the validation set and only then applied to the test set.

**How the framing drove the pipeline:**
- **Time-based split.** In production the model always scores *future* sessions, so a random
  split would let it learn from December and be tested on February. I split by month:
  train Feb–Sep (6,968 sessions, 11.7% buyers), validation Oct–Nov (3,531, 24.8%, used only
  for threshold tuning), test Dec (1,706, 12.7%). There is no year or day column, so this is the
  finest ordering available.
- **`Month` dropped as a feature.** After a month split, the test month never appears in
  training, so a one-hot `Month` would carry no information at test time. `SpecialDay` is kept
  as the seasonality signal.
- **`PageValues` handled as possible leakage.** Google Analytics computes it from revenue of
  pages visited, aggregated over sessions that include converting ones, so it may partly
  encode the outcome. The purchase rate is 3.9% when it is 0 and 56% when it is > 0. Rather than
  guess, I trained two feature sets: **full** (with it) and **deployable** (without it), and
  report both.

## Data preparation & feature engineering

**Data quality**
- No missing values.
- **125 exact duplicate rows dropped.** On inspection they are all single-page, zero-duration
  sessions. These are most likely distinct visitors who bounced instantly, not logging
  errors. I dropped them because they are indistinguishable and add no information. At 1% of
  rows the effect is negligible.

**Engineered features** (all defined in one `engineer()` function so they are identical for
every split):
| Feature | Why |
|---|---|
| `total_pages`, `total_duration` | Overall engagement in the session |
| `product_time_share` = product time / total time | Is the visitor shopping or just browsing info pages? |
| `sec_per_product_page` | Depth of attention per product viewed |
| `single_page` flag | Instant bounces are a distinct behaviour (spike at BounceRate = 0.2) |
| `is_new_visitor` | New visitors buy *more* often (24.9% vs 13.9%), worth an explicit flag |
| `has_pagevalue` (full set only) | The jump from 0 to > 0 matters more than the size |
| `log1p` of all counts and durations | Long right tails (durations reach 60,000 s); log makes them usable by a linear model |

**Preprocessing** (`ColumnTransformer`, **fitted on train only**):
- `StandardScaler` on numeric features, so all coefficients are on one scale and gradient
  descent behaves.
- `OneHotEncoder` on `OperatingSystems`, `Browser`, `Region`, `TrafficType`. These are ID
  codes, not quantities (Region 9 is not "more" than Region 1). Codes with < 50 training rows
  are grouped as "infrequent", and unseen codes in later months fall into that group.
- Result: 52 features (full) / 50 features (deployable).

## Modeling: three implementations, one model

**Model:** logistic regression, because the target is binary and the decision needs a
probability to threshold. Same L2 penalty in all three: sklearn's default C = 1.0, reproduced in
PyTorch as `0.5 · (1 / (C·n)) · ||w||²` on the weights only (not the bias), which is the same
objective.

1. **scikit-learn** `LogisticRegression(C=1.0)` (L-BFGS, solved to convergence).
2. **Manual PyTorch:** raw tensors `w`, `b`, hand-written BCE + L2, `loss.backward()`, manual
   update `w -= lr * w.grad`, full-batch gradient descent.
3. **Standard PyTorch:** `nn.Module` with one `nn.Linear`, `BCEWithLogitsLoss`, `optim.SGD`
   with `weight_decay` on the weight only.

Both PyTorch versions start from zeros, lr = 1.0, 10,000 full-batch epochs.

**Results on the December test set** (threshold tuned on Oct–Nov validation):

| Features | Method | Threshold | AUC | Log loss | Business cost |
|---|---|---|---|---|---|
| — | Naive baseline (train purchase rate) | 0.01 | 0.500 | 0.380 | €2,980 |
| full | scikit-learn | 0.05 | 0.900 | 0.223 | €762 |
| full | manual PyTorch | 0.05 | 0.900 | 0.223 | €764 |
| full | standard PyTorch | 0.05 | 0.900 | 0.223 | €764 |
| deployable | scikit-learn | 0.10 | 0.749 | 0.337 | €1,916 |
| deployable | manual PyTorch | 0.10 | 0.748 | 0.337 | €1,920 |
| deployable | standard PyTorch | 0.10 | 0.748 | 0.337 | €1,920 |

Reference policies on the test set: show the pop-up to nobody = €2,160, to everyone = €2,980.

**Do the three agree?** Yes, after investigation.
- **Manual and standard PyTorch match exactly** (coefficient difference 0.0000). They compute
  the same gradient with the same initialisation; `nn.Linear` + `optim.SGD` is just a packaged
  version of the manual loop.
- **sklearn vs PyTorch:** max probability difference 0.032 (full) / 0.013 (deployable). Test
  AUC and cost are practically identical.
- **What went wrong first:** with lr = 0.5 and 3,000 epochs, the gap was up to 0.15 in
  probability. The largest coefficient disagreement was on `log_total_pages`, which is nearly
  collinear with `log_ProductRelated` (most pages viewed are product pages). Collinearity
  makes the loss surface a long flat valley, so gradient descent moves very slowly along it.
  L-BFGS uses curvature information and reaches the optimum directly. Raising lr to 1.0 and
  epochs to 10,000 closed the gap. The remaining coefficient differences (≤ 0.22) sit in those
  same collinear directions, where different weight combinations give nearly the same
  predictions.

**Key findings**
- **PageValues carries most of the signal:** AUC 0.90 with it vs 0.75 without.
- **The deployable model is still useful:** €1,916 vs €2,160 for the best naive policy, a
  saving of about 11%. At threshold 0.10 it shows the pop-up to 59% of sessions and catches 88%
  of buyers.
- **Seasonality affects threshold tuning.** The naive baseline, tuned on high-season Oct–Nov
  (25% buyers), chose "show everyone", but in December "show nobody" was cheaper. The models'
  tuned thresholds (0.05–0.10) sit below the theoretical 0.17 for the same reason: trained on
  low-season months, they under-predict in high season, and the validation-tuned threshold
  compensates.

## Limitations & next steps

1. **PageValues may leak the outcome.** The 0.90 vs 0.75 gap is the cost of uncertainty. *Next:*
   recompute page values only from sessions *before* each session's date, so the feature is
   legitimately available at decision time.
2. **Whole-session features, mid-session decision.** Counts and durations are session totals,
   but a pop-up fires while the session is in progress, so results are optimistic. *Next:*
   collect event-level clickstream data and build features from the first N pages only.
3. **Propensity, not uplift.** The label says who bought, not who bought *because of* a
   pop-up. Some flagged buyers would have bought anyway, so the discount is partly wasted.
   *Next:* an A/B test with a randomised pop-up to train an uplift model.
4. **Seasonal shift.** Training months are low season; calibration drifts in Nov–Dec (see the
   predicted-vs-actual plot in the dashboard). *Next:* multiple years of data to train on past
   holiday seasons, or periodic re-calibration of the intercept.
5. **One year, Jan and Apr missing**, and the cost numbers are assumptions. *Next:* replace
   them with the shop's real margin and coupon-redemption data. The dashboard slider already
   shows how the optimal threshold moves.

## Generative AI use disclosure

I used Claude (Anthropic) throughout this assignment: environment setup and Git commands,
suggesting the dataset, exploratory-analysis code, the training notebook code (feature
engineering, the three implementations, evaluation), diagnosing the convergence gap between
sklearn and PyTorch, the Gradio dashboard (`app.py`), and drafting this report from my choices.
I chose the time-based split, testing PageValues both ways, dropping duplicates, and the
discount pop-up framing; the cost assumptions, train/validation/test cut and engineered features
were proposed by Claude and reviewed by me. I rewrote this report in my own words and can
explain every decision in it.
