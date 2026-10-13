# Assignment 1 Report

- **Name**: Isabella Lentini
- **Student ID**: 20211
- **Email**: ilentini.ieu2022@student.ie.edu
- **Group**: BBADBA 5B 

## Dataset

I used the "Hotel Reservations" dataset from Kaggle
(`ahsan81/hotel-reservations-classification-dataset`). A copy is committed in
`data/hotel_reservations.csv` (3.1 MB), so the pipeline runs offline and
`scripts/fetch_data.py` shows where it came from.

Each row is one hotel booking. There are 36,275 rows and 19 columns: the
number of adults and children, weekend and week nights, meal plan, parking,
room type, lead time (days between booking and arrival), arrival year, month
and day, market segment, whether the guest has stayed before, their previous
cancellations and stays, the average price per room, the number of special
requests, and the booking status. Arrivals go from July 2017 to December 2018,
and 32.8% of all bookings were canceled.

I kept all 36,275 rows even though the brief suggests about 20k, because the
file is small and the extra rows give me a full year for training plus two
separate later periods for validation and testing.

I picked this dataset because it supports a real decision that hotels make
every day, and because the raw columns are not ready to use: the date is
split into three columns and has an invalid day in it, lead time is very
skewed, some rooms cost nothing, some bookings have zero nights, and the
cancellation rate changes a lot over time. So the feature engineering and
the split actually matter.

## Business / real-life framing

The model is for a hotel revenue manager who has to decide how many extra
rooms to sell (overbooking). If the manager knows which bookings are likely
to cancel, they can sell those rooms again instead of leaving them empty. The
model flags each booking as "will cancel" or "will not cancel".

**Target.** `y = 1` when `booking_status` is "Canceled", and 0 otherwise.

**Costs of mistakes.** Both mistakes are measured in euros, using the value
of the stay (average price per room times the number of nights):

| Mistake | What happens | Cost |
|---|---|---|
| Missed cancellation (false negative) | The guest cancels, the room was not resold and stays empty | 1 × stay value |
| False alarm (false positive) | The room is resold but the guest shows up, so someone has to be relocated to another hotel and compensated | 1.5 × stay value |

A false alarm costs more because the hotel loses the money and also has to
pay for the relocation and deal with an unhappy guest. The 1.5 is my own
assumption, not a real figure from a hotel.

This framing drove three concrete decisions in the pipeline:

1. **Time-based split.** In real use the model is trained on past bookings and
   used on future ones, so I split by arrival date instead of randomly:

   | Split | Arrivals | Rows | Cancel rate |
   |---|---|---|---|
   | Train | Jul 2017 to Jun 2018 | 20,127 | 26.6% |
   | Validation | Jul to Sep 2018 | 8,318 | 44.8% |
   | Test | Oct to Dec 2018 | 7,830 | 35.8% |

   A random split would let the model learn from bookings that come after the
   ones it is tested on, and it would hide something important: the
   cancellation rate moves a lot over time (14.8% of 2017 arrivals canceled
   against 36.7% in 2018). With a time split the test score shows how the
   model would really do on a new period. Validation is used to choose the
   regularization, the learning rate and the threshold, and the test set is
   used once at the end.

2. **The threshold is chosen by euro cost, not accuracy.** Flagging a booking
   is worth it when the expected cost of not flagging is bigger than the
   expected cost of flagging:

   p × 1.0 × value > (1 − p) × 1.5 × value, which gives p > 1.5 / 2.5 = 0.6

   So with perfect probabilities the threshold should be 0.6. In practice I
   tried every threshold from 0.01 to 0.99 on the validation set and kept the
   one with the lowest total cost, which was 0.54 (more on why it is lower in
   Limitations). Accuracy treats both mistakes as equal and ignores that a
   long expensive stay matters more than a cheap one night stay, so it is the
   wrong measure for this decision.

3. **Model quality measures and class imbalance.** To compare models without
   a threshold I use ROC AUC (how well risky bookings are ranked above safe
   ones) and log loss (how good the probabilities are, which matters because
   the cost rule needs real probabilities). I did not use `class_weight`,
   because changing the weights would distort the probabilities, and the
   cost based threshold already deals with the imbalance.

## Data preparation & feature engineering

All cleaning and features live in `pipeline.py`, so the app prepares a new
booking in exactly the same way as training does.

### Data quality decisions

| Problem | What I did | Impact |
|---|---|---|
| 37 bookings dated 29 Feb 2018, a date that does not exist | Moved them to 1 Mar 2018 and added an `invalid_date` flag | Rows are kept and get a valid date for the split. The flag lets the model treat them differently if they behave differently (its coefficient ended up small, −0.12) |
| 78 bookings with zero nights | Kept them and added a `zero_nights` flag | They could be day use or data errors. Dropping them would lose real bookings, and the flag keeps them from confusing the nights features. Their stay value is 0, so they never add to the business cost |
| 545 bookings with a price of 0 (354 Complementary, 191 Online) | Kept them and added an `is_free` flag | Free rooms behave differently (the flag lowers the odds of canceling to about half). Like zero nights, they cost nothing when wrong |
| A few bookings with 9 or 10 children | Capped children at 3 | The next highest value is 3, so these are almost certainly typing errors that would pull the scaled feature around |

There were no missing values.

### Engineered features

| Feature | Built from | Why |
|---|---|---|
| `log_lead_time` | `lead_time` | Lead time is very skewed (most bookings are made a few weeks ahead, some more than a year). log(1 + x) shrinks the long tail and handles 0 |
| `log_total_nights` | weekend + week nights | Same reason, the length of stay is skewed |
| `weekend_share` | weekend nights / total nights | Separates leisure weekend trips from business week trips, without repeating the total nights information (0 for zero-night stays) |
| `arrival_weekday` | the arrival date | Weekday patterns (Sunday and Friday arrivals cancel more than Monday ones) |
| `has_prev_cancellation`, `has_prev_stay` | previous cancellations and previous stays | Almost everyone has 0, so the counts are mostly empty with a few large values. A yes/no flag is more stable |
| `invalid_date`, `zero_nights`, `is_free` | see the table above | Data quality flags |

The raw date columns, the raw lead time and nights, the previous counts and
`Booking_ID` are not given to the model, since they are replaced by the
features above (or, for the ID, carry no information).

**What did not work: month features.** I first added the arrival month and
booking month as sine and cosine, so that December sits next to January. They
made the model worse: validation ROC AUC went from 0.828 without them down to
0.808 with them, and log loss went from 0.512 up to 0.581. The reason is that
I only have one full year of data, so the model cannot tell a seasonal pattern
apart from a change between years. It learned from 2017 that October to
December bookings rarely cancel (about 10%), but in 2018 the same months had
36%. So I removed them. To be honest about this check: while looking into it I
also looked at the test set, which means the test set was not completely
untouched when I made this decision.

### Preprocessing (fitted on the training set only)

- **Numeric features** are standardized (mean 0, standard deviation 1). This
  puts all features on the same scale, which gradient descent in PyTorch needs
  to converge, and it makes the coefficients comparable.
- **Categorical features** (meal plan, room type, market segment, weekday) are
  one-hot encoded. Categories with fewer than 50 training rows are merged into
  one "infrequent" group (this caught Meal Plan 3 and Room_Type 3), because a
  coefficient learned from a handful of rows is mostly noise. Categories never
  seen in training also go into that group. One column per feature is dropped
  so the one-hot columns do not duplicate the intercept, and the dropped one is
  the most common category in training (Meal Plan 1, Room_Type 1, Online,
  Monday). That way every coefficient reads as "compared with a typical
  booking".
- **Flags** (0/1) pass through unchanged.

The scaler and encoder are fitted on the training data and only applied to
validation and test, so no information from later periods leaks into
training. The final model sees 33 features.

### What drives cancellations

Since the features are standardized, the coefficients can be compared. The
biggest ones (with the effect on the odds of canceling):

- **Lower risk:** requesting a parking space (odds × 0.19), Offline market
  segment compared with Online (× 0.20), each extra standard deviation of
  special requests (× 0.30), Corporate segment (× 0.44), repeated guests
  (× 0.49).
- **Higher risk:** lead time (one standard deviation of log lead time
  multiplies the odds by about 3.1), room price (× 1.7), Aviation segment
  (× 2.0).

This makes sense: guests who ask for parking or special requests have
concrete plans, while bookings made far ahead online are easy to cancel for
free.

## Modeling: three implementations, one model

The target is binary and the decision rule needs probabilities, so I used
**logistic regression** with an L2 penalty. I trained it three ways on the
same split and the same 33 features:

1. **scikit-learn** `LogisticRegression`.
2. **Manual PyTorch**: weight and bias tensors, the loss written out by hand,
   `loss.backward()` for the gradients, and a hand-written gradient descent
   step.
3. **Standard PyTorch**: an `nn.Module` with one `nn.Linear` layer,
   `nn.BCEWithLogitsLoss`, and `torch.optim.SGD`.

### Making them solve exactly the same problem

For the comparison to mean anything, all three have to minimise the same
function, so I matched them on purpose:

- **Same penalty.** scikit-learn minimises C × (total log loss) + ½ × (sum of
  squared weights). Dividing by C × n gives average log loss + (λ / 2) × (sum
  of squared weights) with λ = 1 / (C × n) = 1 / (0.1 × 20,127) ≈ 0.0005. The
  manual version adds this penalty to the loss. The standard version uses the
  optimizer's `weight_decay = λ`, which adds λ × w to each gradient, the same
  thing.
- **Bias not penalised**, like scikit-learn. In the standard version this
  needed two parameter groups, with weight decay only on the weights.
- **Same starting point and data:** weights start at zero, every step uses
  all training rows (full batch, no `DataLoader`, so no randomness), and all
  tensors are 64-bit.
- **scikit-learn solved to the exact minimum:** I used the `newton-cholesky`
  solver with a very strict tolerance (1e-10). The default stopping rule
  stops a little early, which would show up as a small difference that is
  not a real disagreement.

### Tuning (on validation only)

Following the brief I only tuned the regularization strength and the learning
rate.

| C | Validation log loss |
|---|---|
| 0.01 | 0.5131 |
| **0.1** | **0.5121** |
| 1 | 0.5161 |
| 10 | 0.5174 |
| 100 | 0.5177 |

C = 0.1 was best, and the two PyTorch versions use the λ that matches it.

For the learning rate I tried 0.1, 1 and 4, with 5,000 steps each. Gradient
descent on this loss is only stable below 2 / L, where L is the steepest
curvature of the loss. For this data that limit is about 4.7, so 4 is the
largest safe value. With 4 the training objective reached 0.429961, while 0.1
was still at 0.430657 after 5,000 steps (not fully converged). Both PyTorch
versions chose 4.

### Results on the test set (Oct to Dec 2018)

Each model uses the threshold it chose on validation. The naive baseline gives
every booking the training cancellation rate (0.266) as its probability.

| Model | Threshold | ROC AUC | Log loss | Accuracy | Precision | Recall | Cost (€) | Saved vs baseline (€) |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0.27 | 0.500 | 0.673 | 0.642 | 0.000 | 0.000 | 971,694 | 0 |
| scikit-learn | 0.54 | 0.856 | 0.460 | 0.775 | 0.802 | 0.494 | 693,294 | 278,401 |
| PyTorch (manual) | 0.54 | 0.856 | 0.460 | 0.775 | 0.802 | 0.494 | 693,294 | 278,401 |
| PyTorch (standard) | 0.54 | 0.856 | 0.460 | 0.775 | 0.802 | 0.494 | 693,294 | 278,401 |

The model ranks bookings well (AUC 0.856 against 0.5) and cuts the cost of
mistakes by about 29% (€278k over three months) compared with doing nothing.
When it flags a booking it is right 80% of the time, but it only catches
about half of the cancellations. That is the cost rule at work: false alarms
are expensive, so it only flags bookings it is fairly sure about.

On validation the results were similar: AUC 0.828, and the cost at the chosen
threshold was €884,954 against €1,556,789 for the baseline. Using 0.54
instead of 0.5 saved about €13k there, and using the theoretical 0.6 would
have cost about €69k more than 0.54.

### Do the three agree?

Yes. Compared with scikit-learn, the largest difference in any weight is
0.000005, the bias difference is below 0.000001, and the largest difference
in any predicted probability is 0.000001. The two PyTorch versions match each
other to every printed digit. All metrics are identical to three decimals,
and the predictions at the threshold are identical, so the cost is the same
to the cent. The only visible difference is in the seventh decimal of ROC AUC,
which comes from those tiny probability differences changing the order of a
few bookings with almost equal scores.

This is expected, because the problem has a single best solution (the loss is
convex and the penalty makes the minimum unique), and all three methods were
set up to find it. The methods only differ in how they get there: Newton's
method in scikit-learn takes about 10 big steps using the curvature, while
gradient descent in PyTorch takes 5,000 small steps using only the slope.
The tiny gap that is left is what remains of gradient descent not being fully
at the exact minimum, plus rounding.

## Dashboard

`uv run python main.py app` opens a Gradio dashboard that only loads the
saved files in `artifacts/` (the fitted preprocessor, the three models,
`predictions.csv` and `results.json`). Nothing is trained or refitted at
startup. It has five tabs:

- **Model comparison:** the test results table and a predicted vs actual
  (calibration) plot for the three models on validation or test.
- **Threshold and cost:** a threshold slider with the confusion matrix, the
  business cost, the cost curve and a histogram of probabilities, for any
  model and period.
- **Data:** distributions of the features against the target, and the
  cancellation rate over time.
- **What drives cancellations:** the model coefficients.
- **Predict one booking:** enter a booking and get the probability from all
  three models, using the same cleaning and features as training.

## Limitations & next steps

1. **Only one year of data, so season and year are mixed up.** This is why
   the month features failed, and the model has no way to know about
   seasonality. With two or three more years I would add the month features
   back and check them on a full year of validation.

2. **The cancellation rate drifts, so the probabilities run low.** The model
   learned from a period where 26.6% canceled, but validation had 44.8% and
   test 35.8%. The calibration plot shows the model underestimates
   cancellations, which is why the best threshold on validation (0.54) is
   below the theoretical 0.6. Recall also dropped from 0.63 on validation to
   0.49 on test. I chose to keep the models trained on the training period
   only instead of retraining on train and validation together, so that the
   saved models are exactly the ones I evaluated. The downside is that the
   model does not know about the most recent months. In real use I would
   retrain regularly on a rolling window of recent bookings, and recalibrate
   the probabilities on the latest period before choosing the threshold.

3. **The cost model is simplified.** The 1.5 relocation factor is my
   assumption. The model also treats every flagged booking as one overbooked
   room, but in reality a hotel only overbooks when it is close to full, and
   the real cost depends on how many guests show up on the same night. With
   real relocation costs and daily occupancy data I would turn this into a
   per night decision (how many rooms to overbook given the predicted
   cancellations for that night).

4. **The test set was looked at once before the end.** This happened during
   the month features check. The decision would have been the same based on
   validation alone, but the test score is slightly less clean than it should
   be. Next time I would only open the test set once, at the very end.

5. **The baseline's "best" threshold of 0.27 means nothing.** The baseline
   gives everyone 0.266, so any threshold above that flags nobody and any
   threshold below flags everybody. Flagging nobody was cheaper, so 0.27 is
   the same as "never overbook".

6. **A linear model cannot catch interactions.** For example, a long lead
   time probably matters more for Online bookings than for Corporate ones. I
   would test a few interaction terms chosen from business sense (lead time ×
   segment, price × room type) and keep only the ones that improve
   validation.

7. **Limited information about the dataset.** The Kaggle page does not say
   which hotel this is or how the data was collected, so I could not check
   things like the zero-night and free bookings with the source.

## Generative AI use disclosure
Claude (Claude Code) wrote the code after I approved a plan at each stage.
You made the framing, cost, cleaning, split and feature decisions.
