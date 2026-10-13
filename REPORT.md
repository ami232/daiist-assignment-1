# Assignment 1 Report

- **Name**: Felipe Gómez
- **Student ID**: 19369
- **Email**: fgomez.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

**IBM Telco Customer Churn** (`data/telco_churn.csv`, downloaded from IBM's
public repository `IBM/telco-customer-churn-on-icp4d`). One row is one
residential customer of a fictional telecom operator at a single point in
time: 7,043 rows × 21 columns (customer ID, demographics, the services they
subscribe to, contract and billing details, tenure in months, monthly and
total charges, and whether they left in the last month: `Churn`). The churn
rate is 26.5%.

It is a good fit for this assignment because the binary target maps directly
onto a real decision (who should receive a retention offer), the costs of the
two types of errors are clearly different, and the raw columns need real
work before a *linear* model can use them: categorical service columns with
redundant levels, a tenure effect that is strongly non-linear, a charges
column with blank values, and interactions between product and contract.

## Business / real-life framing

**Decision supported:** each month the retention team decides which
customers receive a retention offer (e.g. a discount or a free upgrade). The
model estimates each customer's probability of churning and the team contacts
everyone above a threshold.

**Assumed costs** (in `features.py`, editable in the dashboard):

- Every offer sent costs **€100** (whether or not the customer would actually
  have left: true positives and false positives both cost €100).
- Every churner we fail to contact (false negative) costs **€500** of lost
  future margin.
- Customers correctly left alone (true negatives) cost nothing.

So `cost = 100 × (TP + FP) + 500 × FN`.

**What this framing implies for the pipeline:**

1. **Target:** `Churn == "Yes"` → 1. It is already defined in the data; no
   construction needed. The target is a "left in the last month" flag, so the
   features describe the customer at that moment.
2. **Split strategy:** the dataset is a *single snapshot* with no date or
   signup-cohort column. No future rows exist to leak into training, so a
   time-based split is impossible and also unnecessary. I used a **stratified
   random 60 / 20 / 20 split** (train / validation / test, seed 42). Stratifying
   keeps the 26.5% churn rate identical in every split, which matters because
   the decision threshold is chosen on validation and must transfer to test.
   The split assignment is saved (`artifacts/split.csv`) so the dashboard
   evaluates on exactly the same test customers.
3. **Metric that sets the threshold:** not accuracy and not 0.5. Missing a
   churner is 5× more expensive than a wasted offer, so the threshold is
   chosen by **minimising the business cost on the validation set**. In theory
   the break-even probability is offer / churn cost = 100 / 500 = 0.20. The
   validation-optimal threshold came out at **0.17**, close to that. Model
   selection (C, learning rates) uses validation **log loss**, because the
   threshold decision needs well-calibrated probabilities and not only a good
   ranking.
4. **Baseline:** the naive model predicts the training churn rate for
   everybody, so it can only choose between "contact everyone" (€140,900 on
   test) and "contact no one" (€187,000). I also added a common-sense rule:
   "contact every month-to-month customer" (€103,600).

## Data preparation & feature engineering

All transformations live in `features.py` and are shared by `train.py` and
`app.py`, so the dashboard applies exactly the same transformation.

**Data-quality decisions**

- `TotalCharges` is stored as text, and 11 customers have a blank value. All
  11 have `tenure = 0`: they signed up but have not been billed yet. The
  correct value is therefore **0**, not a mean or median imputation.
- `"No internet service"` (in the 6 add-on columns) and `"No phone service"`
  (in `MultipleLines`) were collapsed to `"No"`. That information is already in
  `InternetService` / `PhoneService`. Keeping it would create perfectly
  collinear dummy columns.
- `customerID` is dropped (an identifier). `gender` is **deliberately
  excluded**: it has no predictive signal in this data (churn rate is ~26% for
  both genders), and a retention policy should not target offers on it.

**Engineered features and why**

| Feature | Rationale |
|---|---|
| `log_tenure`, `tenure_band` (0–6, 7–12, 13–24, 25–48, 49–72 m) | Churn falls from 53% in the first 6 months to 9.5% after 4 years, steeply at first and then flat. A linear model can't follow that curve on raw tenure. The log and the bands let it do so. |
| `is_new_customer` (tenure ≤ 1) | 61% of first-month customers churn, against 23% of everyone else. This is the single largest positive coefficient after fibre. |
| `log_total_charges` | TotalCharges is heavily right-skewed. |
| `price_change` = MonthlyCharges − TotalCharges / tenure | Intended to capture recent price increases, a classic churn trigger. **Didn't work**, see below. |
| `n_addons` (count of the 6 add-ons), `n_protection` (OnlineSecurity + TechSupport) | Engagement / lock-in. Protection add-ons have a strong negative effect (−0.40 per add-on), separate from the overall count. |
| `single_service` | Phone-only or internet-only, with no add-ons. |
| `fiber_month_to_month` | Interaction: fibre + month-to-month customers churn at 55%, against 14% for everyone else. A linear model cannot create this interaction itself. |
| `paperless`, `SeniorCitizen`, `has_partner`, `has_dependents` | Binary 0/1 indicators. |
| `Contract`, `InternetService`, `PaymentMethod`, `MultipleLines`, `tenure_band` | One-hot encoded, with the first level dropped to avoid perfect collinearity with the intercept. |

**Preprocessing:** a `ColumnTransformer` standardises the 7 numeric
features, passes the binary ones through, and one-hot encodes the
categoricals. It is **fit on the training split only** and saved to
`artifacts/preprocessor.pkl`. The result is 26 model inputs.

**What didn't work / what I changed**

- **`auto_pay` flag, removed.** The first version had a flag for "automatic
  payment method". It is *exactly* `1 − ElectronicCheck − MailedCheck`, i.e. a
  linear combination of the PaymentMethod dummies and the intercept. The
  optimum was then not unique, and the three implementations returned
  visibly different coefficients for the payment features. For example,
  `ElectronicCheck` was +0.25 in scikit-learn and −0.11 in the manual loop,
  even though the predictions were nearly the same. Removing it made the
  scikit-learn and manual-PyTorch coefficients agree to within 0.08.
- **`price_change` has no linear signal** (coefficient ≈ −0.02). Its relation
  to churn is U-shaped (by quintile: 31% / 16% / 37% / 17% / 32%). In this
  dataset, TotalCharges looks like a noisy tenure × MonthlyCharges rather than
  a true billing history, so the "recent price change" interpretation does not
  hold. I kept it to document the attempt; it is harmless.
- **Regularisation barely matters.** On validation log loss, C was tuned over
  {0.01 … 100}. The loss improves monotonically and flattens after C ≈ 3
  (0.3984 at C = 3, 0.3983 at C = 100), and C = 100 was selected. With 4,225
  training rows and 26 features the model is far from overfitting, so the
  penalty is effectively negligible.

## Modeling: three implementations, one model

**Model:** L2-regularised **logistic regression**, because the target is
binary and the business decision needs a probability to compare against a
cost-based threshold.

All three implementations minimise the **same objective** on the same
standardised training matrix:

`mean BCE(w, b) + λ/2 · ‖w‖²,  λ = 1 / (C · n_train)`

This is exactly scikit-learn's objective (`½‖w‖² + C · Σ BCE`) divided by
`C · n_train`. The intercept is not penalised in any of the three.

1. **scikit-learn:** `LogisticRegression(C=100, lbfgs, tol=1e-8)`.
2. **Manual PyTorch:** raw tensors `w`, `b` with `requires_grad`, an
   explicitly written BCE loss plus the L2 term, `loss.backward()`, then a
   manual `w -= lr * w.grad` update. Full batch, 3,000 epochs. The learning
   rate was tuned on validation over {0.05, 0.2, 0.5, 1.0} → **1.0**.
3. **Standard PyTorch:** an `nn.Module` with one `nn.Linear`,
   `BCEWithLogitsLoss`, mini-batches of 256 from a `DataLoader`, `Adam` with
   a cosine learning-rate schedule, and the same explicit L2 term on the
   weights. I deliberately did *not* use `weight_decay`, because it would also
   decay the bias and, with Adam, is not equivalent to L2. 200 epochs. The
   learning rate was tuned over {0.001, 0.003, 0.01} → **0.01**.

**Results on the held-out test set** (1,409 customers; threshold chosen on
validation by minimum business cost):

| Model | ROC AUC | PR AUC | Log loss | Threshold | Precision | Recall | Offers | Cost (€) | €/customer |
|---|---|---|---|---|---|---|---|---|---|
| scikit-learn | 0.8347 | 0.6463 | 0.4267 | 0.17 | 0.452 | 0.872 | 721 | 96,100 | 68.20 |
| Manual PyTorch loop | 0.8348 | 0.6464 | 0.4267 | 0.17 | 0.453 | 0.872 | 720 | 96,000 | 68.13 |
| nn.Module + torch.optim | 0.8342 | 0.6456 | 0.4268 | 0.17 | 0.455 | 0.874 | 718 | 95,300 | 67.64 |
| Naive baseline (contact everyone) | 0.5 | 0.265 | 0.5787 | – | 0.265 | 1.000 | 1,409 | 140,900 | 100.00 |
| Naive baseline (contact no one) | 0.5 | 0.265 | 0.5787 | – | – | 0.000 | 0 | 187,000 | 132.72 |
| Rule: contact all month-to-month | 0.71 | 0.393 | – | – | 0.412 | 0.866 | 786 | 103,600 | 73.53 |

All three models cut the retention budget's cost by **~32% (≈ €45k per 1,409
customers)** against the best naive policy, and by ~7% against the
month-to-month rule of thumb.

**Do the three agree?** On predictions, yes. On some individual
coefficients, not fully. The details:

- **scikit-learn vs manual loop:** essentially identical. The train
  objective is 0.408154 vs 0.408159, coefficient correlation 0.999, max
  coefficient difference 0.08, max per-customer probability difference 0.03.
  The remaining gap is because 3,000 full-batch gradient-descent steps have
  not *completely* converged along the poorly conditioned directions. L-BFGS,
  as a second-order-ish method, gets there.
- **scikit-learn vs nn.Module/Adam:** the metrics agree to the third decimal
  and the mean per-customer probability difference is 0.007 (max 0.07). But
  the train objective is slightly higher (0.40851), coefficient correlation is
  0.946, and some coefficients differ a lot. Examples: `MonthlyCharges` is
  −0.32 vs +0.10, `InternetService_Fiber optic` 1.02 vs 0.50, and the tenure
  bands are about −0.54 vs −0.84.
- **Why:** these features come in strongly correlated groups.
  `MonthlyCharges` is almost determined by the services the customer has
  (fibre, multiple lines, number of add-ons; correlation with `n_addons` is
  0.72). `tenure`, `log_tenure`, `log_total_charges` and the tenure bands all
  describe the same thing. Along those directions the loss is almost flat, so
  very different weight combinations give the same predictions. With
  near-zero regularisation (C = 100) nothing pins the solution down. Mini-batch
  Adam with a decaying learning rate stops at a nearby point on that flat
  valley, while L-BFGS goes to the exact minimum. This is a property of the
  feature set (collinearity), not a bug. It also means **individual
  coefficients on those correlated features should not be interpreted
  causally**. The stable conclusions shared by all three are: two-year
  contracts (−1.6), no internet (−1.2), protection add-ons (−0.4 each) and
  dependents reduce churn; new customers (+0.7), fibre, electronic check
  (+0.39) and paperless billing (+0.42) increase it.

## Limitations & next steps

- **No time dimension.** A real deployment would score customers *this
  month* to predict churn *next month*. Without snapshot dates I cannot run a
  time-based (out-of-time) validation. With monthly snapshots, I would train on
  months ≤ t and test on month t + 1.
- **Costs are assumptions.** €100 / €500 are not real numbers, and the model
  also assumes every contacted churner is saved. With data on the offer's
  actual acceptance rate *p*, the cost becomes `100·(TP+FP) + 500·(FN + (1−p)·TP)`,
  which moves the threshold upwards. The dashboard lets you change the two
  costs to see how sensitive the policy is.
- **Collinear feature groups** make individual coefficients unstable across
  optimisers (see above). To interpret coefficients I would keep a single
  representation of tenure (bands *or* log), drop `MonthlyCharges` in favour
  of the individual services, and use more regularisation.
- **Linear decision boundary.** The only interaction is the hand-built
  fibre × month-to-month one. Others (e.g. senior × electronic check, tenure ×
  contract) would need either more hand-crafted terms or a tree-based model.
  I would compare against gradient boosting to measure how much the linear
  restriction costs.
- **Dataset realism.** It is IBM's sample dataset, partly synthetic (see the
  `price_change` finding), so the effect sizes should not be taken as real
  telecom behaviour.

## Generative AI use disclosure

This submission was produced with **Claude Code (Anthropic)**. The AI chose
the dataset and the business framing, designed and implemented the feature
engineering, wrote `features.py`, `train.py` and `app.py`, ran the
experiments, and drafted this report. I reviewed the work and am responsible
for being able to explain every decision in it.
