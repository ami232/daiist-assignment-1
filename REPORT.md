# Assignment 1 Report

- **Name**: Angel Benito Irache
- **Student ID**: 20131
- **Email**: abenito.ieu2022@student.ie.edu
- **Group**: BBADBA 5A


## Dataset

**Hotel Booking Demand** (Antonio, Almeida & Nunes, 2019, *Data in Brief*),
taken from the TidyTuesday mirror (2020-02-11). One row is one booking at
either a City Hotel (Lisbon) or a Resort Hotel (Algarve), with arrival dates
from 1 July 2015 to 31 August 2017. The full file has 119,390 rows and 32
columns; 37.0% of bookings were canceled.

**Sampling.** The brief asks for about 20k rows. I took a uniform random sample
of 20,000 rows (`random_state=42`, script `data/make_sample.py`) and sorted it
by arrival date. Unlike taking the most recent 20k rows (only ~5 months), this
keeps the whole 26-month window, so seasonality and year-on-year changes
are still present and a temporal split is still possible. The sample matches
the full data: 37.7% canceled (vs 37.0%), 66% City / 34% Resort (same).
Only the sample (`data/hotels_sample.csv`, 2.9 MB) is committed.

**Exploration findings that shaped the pipeline.**
- `reservation_status` matches the target 1:1 (Canceled / No-Show vs Check-Out).
- Nulls carry meaning: `company` is 94% null (= not a corporate booking),
  `agent` 14% null (= booked directly, 25% cancel vs 39%).
- 0.6% of the sample (124 rows) has 0 nights or 0 guests (not real stays); ADR has values
  <= 0 (complimentary stays) and one value of 5,400.
- "Non Refund" deposits are canceled 99.4% of the time - the opposite of what
  the name suggests, and a known oddity of this dataset.
- 27% of rows are exact duplicates. There is no booking ID, so I kept them as
  separate bookings (most likely group bookings).
- Cancellation rate moves between 31% and 43% by quarter, so time matters.

**Why this dataset.** The target is clear, the business decision is real, and
the data has the problems that make feature engineering matter: skewed
numerics, high-cardinality categoricals, meaningful nulls and several columns
that leak the future.

## Business / real-life framing

A hotel wants to estimate, **at the moment a booking is made**, how likely it
is to be canceled, to decide (a) whether to accept an extra booking for those
dates (overbooking) and (b) whether to ask for a non-refundable deposit.

**Target**: `is_canceled` (1 = canceled, including no-shows).

**Prediction moment -> allowed columns.** Only information the hotel has when
it accepts the booking. The dataset is a snapshot taken at the end of each
booking and does not record when fields were filled in, so "at booking" is the
only moment I can defend. Excluded for that reason:
`reservation_status`, `reservation_status_date` (the outcome itself),
`assigned_room_type` (assigned at check-in: "room changed" bookings cancel
5% vs 42%), `booking_changes`, `days_in_waiting_list`,
`total_of_special_requests`, `required_car_parking_spaces` (all accumulate
after the booking; a guest who cancels early never adds them). `adr` is kept:
the price is agreed at booking and is present for canceled bookings too.

**Split: temporal, by arrival date, 70/15/15.** A random split would train on
2017 bookings and test on 2015 ones, i.e. use the future. I split by *arrival*
date, not booking date, because the dataset was cut by arrival date: bookings
made in 2017 only appear if they arrived before September 2017, so a test set
of late booking dates would contain only short lead times (23% cancel rate for
2017Q3 bookings). Arrival-date splitting mimics "train on stays that have
finished, predict upcoming arrivals".

| Split | Rows | Arrivals | Cancel rate | Non-refund share |
|---|---|---|---|---|
| Train | 13,910 | 2015-07-01 -> 2017-02-10 | 37.0% | 13.9% |
| Validation | 2,976 | 2017-02-11 -> 2017-05-22 | 40.1% | 13.2% |
| Test | 2,990 | 2017-05-23 -> 2017-08-31 | 39.8% | 7.4% |

Validation is used for every choice (features, C, the empirical threshold
shown for comparison); test is used for the final evaluation, with one
disclosed exception: the decision to adopt the theoretical threshold (see
Limitations, point 2). Every month in val/test also occurs in train.

**Metric and threshold: business cost.** Both errors are priced per booking,
in revenue terms (ADR x nights, ADR clipped at the train 99th percentile):
- *False negative* (missed cancellation): empty room, cost = 1 x revenue.
- *False positive* (flagged, overbooked, guest arrives): relocation +
  compensation + reputation, cost = 2 x revenue.

The multipliers are assumptions and are editable in the dashboard. Because
both costs scale with the same revenue, the expected-cost rule "flag if
p x FN > (1 - p) x FP" gives the same threshold for every booking:
**p > FP / (FP + FN) = 2/3**. Accuracy is not used to decide anything (it
weights both errors equally). I also report log-loss (probability quality,
which the threshold rule depends on), ROC-AUC, PR-AUC, precision and recall.

## Data preparation & feature engineering

**Cleaning.** Dropped 124 rows with 0 nights or 0 guests (not real stays; they
would also make the weekend ratio and ADR per person undefined). `children`
nulls -> 0.

**Everything learned from data is learned on train only**: the top-10
countries and agents, the ADR cap (99th percentile = 227.09), the scaler means
and standard deviations, the one-hot categories.

| Feature | Why |
|---|---|
| log(1 + lead time) | Strongly right-skewed (median 69, max 737 days); longer lead time = more time for plans to change |
| log(1 + nights), weekend-night ratio, arrival weekday | Length and type of trip (leisure vs business) |
| total guests, has kids/babies | Families plan more and cancel less |
| Month as sin/cos | Seasonality, December next to January |
| log previous cancellations, log previous non-canceled bookings, ever canceled, repeated guest | Customer history; counts are mostly 0 so log + flag |
| hotel, deposit type, market segment, distribution channel, customer type, meal, reserved room (one-hot) | Booking terms known at booking time |
| country: top 10 + Other + Unknown | 177 countries, 139 with < 100 rows; Portugal (domestic guests) gets its own level |
| agent: top 10 + other + none; has company | Nulls mean "direct" / "not corporate" |
| ADR clipped, ADR per person, ADR = 0 flag | Price level; complimentary stays behave differently |
| log lead time x City, log lead time x Non Refund | Lets the lead-time effect differ by hotel and deposit type, which a linear model cannot learn alone |

Numeric columns are standardised and categoricals one-hot encoded with the
first level dropped. Final design: **70 columns**.

**Choices made on validation** (scikit-learn, C = 1, validation log-loss; one
axis at a time):

| Choice | Options (val log-loss) | Kept |
|---|---|---|
| Lead time | log 0.4192 / buckets 0.4233 / both 0.4204 | log |
| Month | sin/cos 0.4192 / one-hot 0.4196 / one-hot + week 0.4192 | sin/cos (tied at 4 decimals; ties go to the first-listed, simpler option: 11 fewer columns) |
| Interactions | with 0.4192 / without 0.4195 | with |
| Meal + room type | with 0.4192 / without 0.4193 | with |

The differences are in the third or fourth decimal. How a feature is
represented mattered much less than which information is allowed in (see the
ablation below). Lead-time buckets were the clearest loser: they throw away
the ordering inside each bucket.

**What didn't work: duplicate columns.** My first feature set also had
`is_portugal` and `has_agent`. They duplicated information already in the
`country_grp_PRT` and `agent_grp_none` one-hots (`has_agent` is just the
complement of the latter), so X'X was singular: its smallest eigenvalue
(of X'X/n, intercept column included) was ~1e-16, i.e. zero. scikit-learn's
L-BFGS solver gave an answer anyway, but the two gradient-descent versions
crawled along those directions and had not converged after thousands of
epochs (max probability gap to sklearn 0.06). I found the problem from the
eigenvectors, and removing the two columns fixed it.

**Leakage ablation.** Same model and C, adding the excluded post-booking
variables back (test set):

| Variant | Test ROC-AUC | Test PR-AUC | Test cost (EUR, t = 2/3) |
|---|---|---|---|
| Booking-time features (final) | 0.8133 | 0.7466 | 620,705 |
| + special requests | 0.8446 | 0.7921 | 555,637 |
| + parking | 0.8372 | 0.7704 | 592,592 |
| + booking changes | 0.8267 | 0.7664 | 604,360 |
| + room changed | 0.8223 | 0.7544 | 622,877 |
| + waiting list | 0.8132 | 0.7467 | 620,825 |
| + all post-booking | 0.8727 | 0.8244 | 484,970 |

The leaky columns would add up to 6 AUC points and cut cost by 22%. That gain
is information the hotel does not have when the booking arrives, so the
offline score would overstate what the model can do in production. They stay
out. Waiting list adds nothing, so excluding it costs nothing either.

## Modeling: three implementations, one model

**Model: logistic regression**, because the target is binary, the threshold
rule needs probabilities, and the loss is convex, so all three
implementations should reach the same solution.

**Same objective in all three**: mean binary cross-entropy + (lambda/2)||w||^2,
intercept not penalised, with lambda = 1/(C n_train). This is scikit-learn's
objective divided by C n, so the same C means the same model.
- **scikit-learn**: `LogisticRegression(C=1.0)`, L-BFGS, `tol=1e-10`.
- **Manual PyTorch**: raw float64 tensors, hand-written BCE with `logsigmoid`
  plus the L2 term, `loss.backward()`, update `w -= lr * grad` with no
  optimizer, full batch, starts at zero.
- **Standard PyTorch**: `nn.Linear` module, `BCEWithLogitsLoss`, `optim.SGD`
  full batch with `weight_decay = lambda` on the weights only (it adds
  lambda w to the gradient = the L2 term), default random start.

**Tuning.** C from {0.001, ..., 10} on validation log-loss -> **C = 1**
(0.3 to 3 are within 0.0002). Learning rate from {0.1, 0.3, 1, 3, 5}, 10,000
epochs -> **lr = 3** for both (lr = 5 oscillates and ends far from the
optimum). I first picked the learning rate by
validation log-loss, which chose lr = 1. That run had not converged (max
probability gap to sklearn 0.023): stopping short of the optimum works like
early stopping, so a validation metric rewards it. The learning rate is an
optimisation setting and C already handles generalisation, so I switched to
picking the learning rate that reaches the lowest training objective.

**Results on the test set** (models at threshold 2/3; baselines as described):

| Model | Log-loss | ROC-AUC | PR-AUC | Precision | Recall | FP | FN | Business cost (EUR) |
|---|---|---|---|---|---|---|---|---|
| Baseline: accept all (= majority class) | 0.6738 | 0.500 | 0.398 | - | 0.000 | 0 | 1,190 | 678,828 |
| Baseline: rule "Non Refund or previous cancellation" | - | - | - | 0.900 | 0.188 | 25 | 966 | 615,508 |
| scikit-learn | 0.5055 | 0.8133 | 0.7466 | 0.862 | 0.246 | 47 | 897 | 620,705 |
| Manual PyTorch | 0.5054 | 0.8133 | 0.7466 | 0.862 | 0.246 | 47 | 897 | 620,705 |
| nn.Module + optim | 0.5054 | 0.8133 | 0.7466 | 0.862 | 0.246 | 47 | 897 | 620,705 |

For reference, the same models at a 0.5 threshold cost 658,240 EUR on test,
and at the validation-optimal threshold (0.51) 651,879 EUR. The validation
log-loss of all three was 0.4192.

**Do the three agree? Yes, to the precision the optimisers reach.**

| Pair | Max abs. prob. difference | Mean abs. prob. difference | Max abs. coefficient difference | Same decision, thresholds 0.3-0.7 |
|---|---|---|---|---|
| sklearn vs manual PyTorch | 3.8e-3 | 5.9e-5 | 0.028 | >= 99.9% (100% at 2/3) |
| sklearn vs standard PyTorch | 4.6e-3 | 6.5e-5 | 0.028 | >= 99.9% (100% at 2/3) |
| manual vs standard PyTorch | 7.3e-4 | 6.3e-6 | 0.004 | 100% |

Decision agreement was checked at 0.3, 0.4, 0.5, 0.6, 2/3 and 0.7, because the
dashboard slider lets the user move the threshold. The worst case is 0.6,
where 3 of the 2,990 test bookings sit between the sklearn and PyTorch
probabilities. The training objective is 0.3704873 (sklearn), 0.3704875
(manual) and 0.3704876 (standard), and all test metrics agree to the fourth
decimal.

**The remaining gap is convergence, not a different model.** The same
PyTorch runs, measured against scikit-learn after 1k, 3k and 10k epochs:

| Epochs | Max abs. coefficient gap (manual / standard) | Max abs. prob. gap | Objective gap |
|---|---|---|---|
| 1,000 | 0.82 / 0.87 | 0.12 | 4.6e-4 |
| 3,000 | 0.40 / 0.42 | 0.028 | 4.8e-5 |
| 10,000 | 0.028 / 0.028 | 0.004 | 2.4e-7 |

Every gap shrinks steadily with training, which is what you expect if all
three are heading for the same optimum. Why it is slow:
- The gradient-descent error along each direction shrinks by a factor
  (1 - lr x h) per epoch, where h is an eigenvalue of the objective's Hessian
  (computed at the sklearn optimum, intercept included). For logistic
  regression this Hessian is X'WX/n plus lambda on the weight coefficients
  (not on the intercept), with W = p(1 - p) per booking.
- The largest eigenvalue is 0.52. Near the optimum, gradient descent is
  stable only for lr < 2/0.52 = 3.9. That is why lr = 3 works and lr = 5
  oscillates.
- The smallest eigenvalue is 8.4e-5, almost exactly lambda (7.2e-5): along
  that direction the data adds almost no curvature, only the L2 penalty does.
  After 10,000 epochs the error along it is still (1 - 3 x 8.4e-5)^10000 ≈ 8%
  of where it started.
- That direction is the one-hot column `distribution_channel_Undefined`, which
  has **one** row in train, so the data barely constrains its coefficient.
  (In the plain X'X/n its eigenvalue is 7.1e-5, essentially
  1/n = 1/13,910 = 7.2e-5. Lambda is also 7.2e-5 only because C = 1 makes
  lambda = 1/n.)
- The largest coefficient gaps sit on that column, the intercept and the
  market-segment dummies that trade off against it. They affect very few
  bookings, which is why the mean probability gap is about 6e-5.

L-BFGS uses curvature information, so it is not slowed down by this. Manual
and standard PyTorch run the same algorithm and differ only in their starting
point (zeros vs PyTorch's random initialisation), so they are about 5x to
10x closer to each other than to sklearn.

**How I got there (what didn't work first):**
1. With duplicate columns (see above), the PyTorch models were 0.06 away in
   probability after 3,000 epochs.
2. After removing them, a learning rate chosen by validation loss (lr = 1)
   left a 0.023 gap. Choosing it by training objective (lr = 3) closed it.

## Limitations & next steps

1. **The later periods differ from training: partly season, partly a real
   year-on-year change.** Comparing summer 2017 (test) with *all* of train
   mixes the two, because train's only full summer is 2016. So each
   evaluation window is compared with the same calendar dates one year
   earlier (those rows are in train):

   | Window | Cancel rate | Resort cancel | Cancel rate excl. Non Refund | Mean lead time | Non Refund share |
   |---|---|---|---|---|---|
   | 2016-02-11 -> 05-22 (val dates, 2016) | 35.1% | 25.9% | 25.1% | 78 | 13.8% |
   | 2017-02-11 -> 05-22 (validation) | 40.1% | 27.6% | 31.0% | 92 | 13.2% |
   | 2016-05-23 -> 08-31 (test dates, 2016) | 38.2% | 32.8% | 30.1% | 125 | 11.6% |
   | 2017-05-23 -> 08-31 (test) | 39.8% | 36.9% | 35.0% | 141 | 7.4% |
   | All of train | 37.0% | 26.4% | 26.9% | 100 | 13.9% |

   - Most of the apparent "Resort 26% -> 37%" and "lead time 100 -> 141" is
     seasonality. Summer 2016 already had 32.8% and 125 days.
   - What is a real change: among refundable bookings, cancellations rose by
     5-6 points year on year in both windows (25.1% -> 31.0%, 30.1% -> 35.0%),
     and the Non Refund share fell from 11.6% to 7.4% in summer.
   - The model misses exactly that rise. Excluding Non Refund bookings it
     predicts 26.5% vs 31.0% actual on validation, and 30.1% vs 35.0% on test.

   **The ROC-AUC drop is partly composition.** Non Refund bookings are the
   easy ones (99-100% canceled), and their share halves in test.
   - All bookings: ROC-AUC 0.877 (val) -> 0.813 (test), a 0.064 drop.
   - Excluding Non Refund: 0.817 -> 0.771, a 0.046 drop.
   - So the Non Refund share explains about a quarter of the drop. The rest
     is not explained by it: it could be a harder season, a different year or
     other shifts in the mix (see the next paragraph).

   With these data I cannot separate "summer is harder to predict" from "2017
   behaves differently", because the only earlier summer is inside the
   training set. (I first read the small gain from retraining on train +
   validation as evidence of drift rather than lack of data. That inference
   was wrong: neither set contains summer 2017, so it cannot distinguish the
   two, and I dropped it.)

   *Next*: a rolling-origin evaluation over several years, so each season is
   evaluated out of sample at least once. A cheap fix to test is re-estimating
   the intercept on the most recent months: validation already showed the
   same 4-5 point under-prediction before test began.
2. **The validation-optimal threshold did not transfer.** The cost-minimising
   threshold on validation was 0.51. On test it costs 651,879 EUR vs 620,705
   EUR at 2/3. In an earlier run (before the duplicate-column fix) the
   validation threshold was 0.48 and cost more than accepting every booking. I
   adopted the theoretical 2/3 as the official threshold. **Disclosure**: I made
   that choice after seeing the test result, so the test set is no longer fully
   untouched for this one decision. It is defensible because 2/3 comes from
   the cost assumptions and not from any data, but a clean check would need a
   new, later period. *Next*: a rolling-origin evaluation (several consecutive
   validation windows) to see how stable the threshold is.
3. **On cost, the model and a one-line rule are indistinguishable.** On test,
   "flag if Non Refund or previously canceled" costs 615,508 EUR and the model
   620,705 EUR: a 5,196 EUR gap over 2,990 bookings. To check whether that gap
   is real, I ran a block bootstrap over the 15 arrival weeks of the test set
   (2,000 resamples). Weeks are resampled instead of rows because bookings in
   the same week share season and events.

   | Difference (EUR) | Observed | 95% interval | Share of resamples < 0 |
   |---|---|---|---|
   | model - rule | +5,196 | [-20,958, +30,601] | 34% |
   | model - accept all | -58,123 | [-106,172, -9,154] | 99% |
   | rule - accept all | -63,319 | [-106,785, -24,165] | 100% |

   Both clearly beat accepting every booking. Between the model and the rule
   the interval easily includes zero, so the honest statement is "no
   detectable difference", not "the model is worse". With only 15 blocks the
   intervals are approximate, so I rely on their sign pattern and not on the
   exact endpoints.

   Why the model does not win: with FP twice as expensive as FN, only
   bookings with p > 2/3 are flagged (the model flags 340 test bookings, the
   rule 249). The two sets overlap heavily: 222 bookings are flagged by both.
   The 118 flagged only by the model cancel 60% of the time, just below the
   2/3 needed for a flag to pay off at a 2:1 cost ratio, so on test they cost
   slightly more than they save. The 27 flagged only by the rule cancel 7% of
   the time. The model's advantage is ranking (ROC-AUC 0.81; the rule has only
   two levels) and the probabilities themselves. *Next*: use the probabilities
   where a yes/no rule cannot help, for example expected cancellations per
   arrival date (sum of p) to size overbooking.
4. **"Non Refund" looks like a data artefact; the model does not need the
   column itself.** 99-100% of these bookings are marked canceled, which is implausible
   for real non-refundable bookings. It is probably how some group or agency
   blocks were recorded. I measured the dependence directly by refitting
   without `deposit_type` and its interaction (same C) :

   | Variant | Val ROC-AUC | Test ROC-AUC | Test ROC-AUC excl. Non Refund | Test log-loss | Test cost (EUR) |
   |---|---|---|---|---|---|
   | Final (with deposit_type) | 0.877 | 0.813 | 0.771 | 0.506 | 620,705 |
   | Without deposit_type | 0.869 | 0.809 | 0.769 | 0.517 | 626,120 |

   Removing the column costs 0.004 AUC and 0.9% more cost, but test log-loss
   rises from 0.506 to 0.517, far more than the differences between feature
   representations on validation (at most 0.004), so the probabilities do
   use it. Other booking-time features (agent, country, market segment,
   customer type, lead time) probably reconstruct much of it, which I have
   not tested. This shows the model does not need `deposit_type` itself. It
   does not show that the artefact has no influence, because correlated
   features can carry it. What does show signal beyond the artefact is the
   "excl. Non Refund" column: on refundable bookings alone the model still
   ranks at ROC-AUC 0.77. *Next*: ask the hotel how these bookings were
   recorded.
5. **Costs are assumptions.** FP = 2 x revenue is a guess. Revenue uses ADR
   clipped at the train 99th percentile (227.09 EUR/night) rather than the
   raw ADR. The full dataset contains a rate of 5,400 EUR/night, about 50
   times the average ADR (~100 EUR). A single such error would weigh as much
   as dozens of ordinary bookings and could flip a model comparison alone.
   Clipping makes the cost measure robust to such outliers, at the
   price of under-stating the few genuinely expensive bookings. The threshold
   is not affected: both costs scale with the same revenue, so 2/3 holds
   whatever the clip. *Next*: get actual relocation costs. The dashboard
   already lets the user change both multipliers and shows the new optimal
   threshold.
6. **Linear model.** Interactions had to be added by hand (two of them). A
   gradient-boosted tree model would capture more, at the cost of
   interpretability and of this assignment's three-way comparison.

## Generative AI use disclosure

I used Claude (Anthropic) in two ways.

**Claude Code, inside the repository.** It wrote the code
(`data/make_sample.py`, `pipeline.py`, `train.py`, `app.py`), downloaded and
explored the dataset, ran and debugged the pipeline (including finding the
duplicate-column problem) and produced a first draft of this report.

**Claude chat (claude.ai).** I used it to compare candidate datasets and
business problems, to review my feature list, and the finished code. In those conversations the assistant suggested options and analyses that I then adopted: predicting at
booking time, excluding the post-booking columns and running the leakage ablation, the comparison with the same dates of 2016, the ROC-AUC excluding Non Refund bookings, the block bootstrap, the refit without `deposit_type`, and the agreement and convergence checks.

I chose the dataset, the business framing, the prediction moment, the split, the cost multipliers and the threshold from the options discussed, and I reviewed all the code and text. The figures come from running the code (the exploration and `main.py train`), not from the assistant's estimates.
