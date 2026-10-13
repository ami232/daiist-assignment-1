# Assignment 1 Report

- **Name**: Hugo Marsiglia
- **Student ID**: 19610
- **Email**: hmarsiglia.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

**Source.** City of Chicago Data Portal, "Food Inspections" (dataset `4ijn-s7e5`):
https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5.
Downloaded on 13 October 2026 through the portal's public API with
`data/download_data.py`. The result is committed as
`data/chicago_restaurant_inspections.csv` (9.9 MB) and training reads only
that file.

**One row** is one visit by a city health inspector to one restaurant.

**Size.** 85,889 inspections of 16,447 licensed restaurants, from 1 July 2019
to 30 June 2026, in 11 columns:

| Column | Meaning |
|---|---|
| `inspection_id` | Unique id of the visit |
| `license` | The restaurant's business licence number; links a restaurant's visits together |
| `dba_name` | Trading name |
| `risk` | City risk class: Risk 1 (High), Risk 2 (Medium), Risk 3 (Low) |
| `zip` | ZIP code |
| `inspection_date` | Date of the visit |
| `inspection_type` | Canvass (routine), Complaint, License, a re-inspection of one of those, and a few rarer types |
| `results` | Pass, Pass w/ Conditions, Fail, or a visit with no outcome (Out of Business, No Entry, Not Ready, Business Not Located) |
| `latitude`, `longitude` | Location |
| `violation_codes` | Code numbers of the violations cited, for example `3\|47\|55` |

The only change made at download time is to the last column: the portal's
free-text `violations` field (inspector comments, about 1 KB per row) is
reduced to the list of violation code numbers. Everything else happens in
the pipeline.

**Why the file is larger than the ~20k-row guide.** The model predicts
22,205 of these rows: the routine inspections with an outcome from July 2022
onwards. The other rows are the same restaurants' earlier inspections, and
they are what the features are built from. The whole pipeline retrains from
scratch in about 30 seconds.

**Why I picked it.** I like this dataset because it connects to a real
decision: which restaurant should be inspected first. A restaurant's past
results can be used to predict a future outcome, which makes sense to me and
makes it easy to explain why the features are useful.

## Business / real-life framing

The user is a city health department that schedules routine inspections. It
would use the model to decide which restaurants to visit next: the riskier
ones move up the routine inspection queue.

How that framing shaped the pipeline:

| Question the framing raises | What the pipeline does |
|---|---|
| Which inspections does the model predict? | Routine ("Canvass") inspections only: 22,205 rows. Complaint, licence and re-inspection visits are never targets, but they count as history. |
| What is the target? | `Fail` = 1. `Pass` and `Pass w/ Conditions` = 0. The 3,779 routine visits with no outcome (3,625 Out of Business, 149 No Entry, 3 Not Ready, 2 Business Not Located) are excluded. |
| How is the data split? | By date. Train: July 2022 to June 2024 (10,360 inspections, 27.3% fail). Validation: July 2024 to June 2025 (6,509, 22.3%). Test: July 2025 to June 2026 (5,336, 22.8%). Each period is a whole number of years, so every season is represented equally. |
| Which metric sets the threshold? | The total cost of mistakes: $450 for a missed fail, $150 for a wasted priority visit (3 to 1). The cheapest threshold on the validation year is 0.26; the theoretical value for a perfectly calibrated model is 150 / (150 + 450) = 0.25. |

**Why Fail only.** I wanted to predict an actual inspection failure, so I
kept the target focused on that outcome. Counting Pass w/ Conditions as well
would answer a broader question: whether a restaurant gets either outcome. A
conditional pass can still involve problems, but it is different from a
recorded failure. A Fail triggers a re-inspection, whereas the issues behind
a conditional pass are corrected during the visit. Predicting Fail therefore
focuses the priority queue on restaurants likely to require further
enforcement and inspection resources.

**Why 3 to 1.** The $150 and $450 figures are assumptions. Missing a failing
restaurant can delay the discovery of its problems, so I give that mistake a
higher cost than an unnecessary priority visit. I chose 3 to 1 as a moderate
starting assumption: it gives missed failures substantially more weight
without assuming the stronger penalty of 5 to 1. I would test both ratios to
see how much the inspection priorities change.

**Why a split by date.** The model will predict future inspections using
information that is already available, and splitting by date recreates that
situation: it learns from earlier inspections and is tested on later ones.
With a random split it could train on newer inspections while being tested
on older ones, which gives a less realistic picture of future performance.

## Data preparation & feature engineering

**The rule behind every feature.** A feature may only use information dated
strictly before the inspection it describes. Constants that are learned
from data (the base fail rate, medians for imputation, chain sizes, the
scaler's means and standard deviations) are learned on the training period
only.

**Data-quality decisions**

| Issue | Decision | Size |
|---|---|---|
| No usable licence number | Dropped: the visit cannot be linked to any restaurant's history | 94 of 85,889 rows |
| Routine visit with no outcome | Not a target (see above) | 3,779 rows |
| Same restaurant, two routine rows on one day | Keep one | 10 rows |
| July to September 2026 looked incomplete when downloaded (about half the usual monthly volume) | Data stops at 30 June 2026 | n/a |
| Restaurant has no inspection with an outcome in the previous 3 years (new, or not inspected for a long time) | Counts are 0; the smoothed fail rate falls back to the training base rate (27.3%); days since last inspection, average violations and last violations take the training medians (369 days, 3.8, 3); the three violation flags take 0; `no_history` = 1 | 4.2% of training rows, 2.3% of test rows |
| History gets longer as the calendar advances | Every history feature looks back a fixed 3 years, so a 2023 inspection and a 2026 inspection are described with the same amount of history. The file starts in July 2019, exactly 3 years before the first predicted inspection. | n/a |

**The 16 features.** All are standardised with training means and standard
deviations before modelling. The coefficient is from the scikit-learn model
and is per standard deviation of the feature.

| Group | Feature | How it is built | Coefficient |
|---|---|---|---|
| Static profile | `risk_high` | 1 if the city risk class is Risk 1 (High) | +0.052 |
| Static profile | `log_chain_size` | Log of the number of licences sharing the restaurant's normalised name, counted on inspections before the validation year | -0.075 |
| Static profile | `month_sin`, `month_cos` | Month of the year as a point on a circle | -0.064, -0.087 |
| Area track record | `zip_fail_rate_12m` | Fail rate of routine inspections in the same ZIP over the previous 365 days, shrunk toward the city rate with 20 pseudo-inspections | +0.264 |
| Own record | `fail_rate_3y` | (fails + 2 x base rate) / (inspections + 2) over the previous 3 years | +0.186 |
| Own record | `log_n_prior` | Log of 1 + inspections with an outcome in the previous 3 years | -0.142 |
| Own record | `no_history` | 1 if there is no inspection with an outcome in the previous 3 years | +0.047 |
| Own record | `prev_canvass_fail` | 1 if the most recent routine inspection failed | -0.047 |
| Own record | `log_days_since_last` | Log of 1 + days since the most recent inspection | +0.092 |
| Own record | `log_complaints` | Log of 1 + complaint-driven inspections in the previous 3 years | +0.092 |
| Violation history | `mean_viol_prior` | Average number of violations cited per inspection, previous 3 years | +0.182 |
| Violation history | `prev_n_viol` | Violations cited at the most recent inspection | +0.131 |
| Violation history | `pest_any` | 1 if violation 38 (insects, rodents, animals) was cited in the previous 3 years | +0.077 |
| Violation history | `temp_any` | 1 if a temperature-control violation (codes 18 to 24, 33) was cited | +0.068 |
| Violation history | `mgmt_any` | 1 if a person-in-charge, certificate or staff-knowledge violation (codes 1, 2, 3, 5) was cited | +0.092 |

**Why the features are built this way**

- **Logs of counts and elapsed days.** These variables are skewed, and their
  effects may have diminishing returns. Logs reduce the influence of
  unusually large values; using `log(1 + x)` also accommodates zeros.
- **Smoothed fail rates.** A restaurant or ZIP with very few inspections
  could otherwise receive an extreme rate from one or two outcomes.
  Smoothing pulls uncertain estimates toward a broader average while
  allowing substantial history to dominate.
- **Month on a circle.** Sine and cosine represent the annual cycle so that
  December and January are close together, allowing the model to learn a
  smooth seasonal pattern.
- **Standardisation.** Putting features on comparable scales improves
  gradient-descent behaviour and prevents the L2 penalty from treating
  features differently merely because their measurement units differ. The
  scaling parameters are learned from training data only.

**What each group adds** (scikit-learn, validation year)

| Groups in the model | Features | ROC AUC | Log-loss |
|---|---|---|---|
| Static profile | 4 | 0.539 | 0.5350 |
| + Area track record | 5 | 0.646 | 0.5100 |
| + Own record | 11 | 0.671 | 0.5016 |
| + Violation history | 16 | 0.702 | 0.4880 |

| Group removed from the full model | ROC AUC lost |
|---|---|
| Area track record | 0.031 |
| Violation history | 0.031 |
| Own record | 0.013 |
| Static profile | 0.000 |

**What did not work**

- The static profile (risk class, chain size, season) adds nothing once the
  history features are in: validation AUC is 0.702 with or without it.
- `prev_canvass_fail` ends up with a small negative coefficient (-0.047)
  once the restaurant's fail rate is in the model.
- Tried during exploration and dropped before the final pipeline: a flag for
  "the last inspection of any type failed" (almost never true, because a
  fail is followed by a re-inspection that passes); a hand-washing violation
  flag (coefficient near zero); the citywide fail rate over the previous 90
  days (no change) or 365 days (AUC about 0.002 higher but worse log-loss);
  one-hot ZIP codes (replaced by the rolling ZIP fail rate). These checks
  are not part of `train.py`.

**Why the ZIP's recent fail rate predicts well.** In my opinion it captures
changing local conditions, for example the type of restaurants being
inspected and inspection practices. Because it uses recent outcomes, it can
respond when an area's pattern changes. The correlation of -0.07 between ZIP
fail rates in the training years and in the test year (see Limitations)
suggests that the same ZIPs do not always have the highest fail rates. I
therefore interpret the feature as a changing local signal rather than as
evidence that certain areas are permanently risky.

**The static features.** Risk class, chain size and season may carry useful
information that the restaurant's history already captures. The lack of any
AUC improvement means they added little to this particular model; it does
not prove they never matter. I kept them to show their contribution
alongside the history features, but the ablation gives no predictive
justification for retaining them. For deployment I would prefer the simpler
model without them, provided its later-period performance remains
comparable.

**The negative weight on a failed last inspection.** It may reflect
restaurants correcting their problems after failing. It could also result
from overlapping history features. I would not interpret it as proof that
failing an inspection lowers a restaurant's risk.

## Modeling: three implementations, one model

**Model.** Logistic regression, because the target is binary. All three
implementations minimise the same objective on the same standardised
training matrix:

    mean log-loss + (l2 / 2) x sum of squared weights     (bias not penalised)

scikit-learn's `C` and the PyTorch `l2` are tied by `l2 = 1 / (C x n)`.
`C` = 0.1 was chosen on validation log-loss, which gives `l2` = 0.000965
with n = 10,360.

| Implementation | How it is optimised |
|---|---|
| scikit-learn | `LogisticRegression`, L-BFGS solver, run to convergence |
| PyTorch manual loop | The Session 5 pattern (section 9.2): raw weight and bias tensors, a `forward` function ending in a sigmoid, a hand-written `bce_loss`, `loss.backward()`, then `w -= lr * w.grad` inside `torch.no_grad()` and `.grad.zero_()`. Full batch, lr 1.0, 500 epochs. The only addition is the L2 term in the loss. |
| PyTorch `nn.Module` + `torch.optim` | The Session 5 appendix pattern (A.2): `nn.Linear` followed by a sigmoid, `nn.BCELoss`, and `optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()`. `torch.optim.SGD` with `weight_decay` on the weights only supplies the L2 term. Two additions: `DataLoader` mini-batches of 256 instead of the full batch, and the learning rate (1.0) multiplied by 0.9 after each of the 60 epochs. |

**Results on the test year** (July 2025 to June 2026, 5,336 inspections, same
split for every row, threshold 0.26 chosen on validation)

| Model | ROC AUC | PR AUC | Log-loss | Brier | Fails found in riskiest 20% | Prioritised | Precision | Recall | Cost per 1,000 inspections |
|---|---|---|---|---|---|---|---|---|---|
| Naive baseline: training fail rate for everyone | 0.5000 | 0.2281 | 0.5422 | 0.1781 | 20.0% | 0.0% | n/a | 0.0% | $102,633 |
| Rule of thumb: rank by own past fail rate | 0.5833 | 0.2855 | 0.5404 | 0.1765 | 27.1% | 30.5% | 30.0% | 40.1% | $93,553 |
| scikit-learn | 0.6488 | 0.3419 | 0.5129 | 0.1677 | 32.5% | 30.7% | 33.9% | 45.6% | $86,301 |
| PyTorch manual loop | 0.6488 | 0.3419 | 0.5129 | 0.1677 | 32.5% | 30.7% | 33.9% | 45.6% | $86,301 |
| PyTorch `nn.Module` + `torch.optim` | 0.6488 | 0.3420 | 0.5129 | 0.1677 | 32.5% | 30.7% | 33.9% | 45.5% | $86,329 |

Each baseline uses its own cheapest threshold from the validation year. For
the naive baseline that threshold means "prioritise no one"; "prioritise
everyone" would cost $115,789 per 1,000.

On the validation year the three models score ROC AUC 0.7021, 0.7021 and
0.7022, log-loss 0.4881, 0.4881 and 0.4880.

**Do the three agree?**

| Compared with scikit-learn | Largest gap in any coefficient | Largest gap in any test-year probability |
|---|---|---|
| PyTorch manual loop | 7.4e-08 | 1.1e-07 |
| PyTorch `nn.Module` + `torch.optim` | 1.6e-03 | 1.8e-03 |

The `nn.Module` version makes a different decision for 5 of the 5,336
test-year restaurants: it prioritises 1 that scikit-learn does not and
leaves out 4 that scikit-learn prioritises. That is the $28 difference in
cost per 1,000.

The gap of the `nn.Module` version was investigated with two extra runs. Its
one algorithmic difference from the manual loop is mini-batching, so one run
removes the learning-rate decay and the other removes mini-batching
altogether:

| `nn.Module` run | Gradient steps | Largest coefficient gap to scikit-learn |
|---|---|---|
| Mini-batches of 256, constant learning rate, 60 epochs | 2,460 | 8.8e-02 |
| Mini-batches of 256, learning rate x 0.9 per epoch, 60 epochs (the saved model) | 2,460 | 1.6e-03 |
| Full batch, constant learning rate, 500 epochs (the manual loop's recipe) | 500 | 7.3e-08 |

The third run changes two settings at once, batch size and number of epochs,
so it shows that the manual loop's recipe reproduces scikit-learn inside
`nn.Module`; it does not isolate the effect of batch size alone.

**Why they agree, and why the third is slightly off.** All three
implementations fit the same mathematical model; only the training differs.
The manual loop uses the whole training set for every update. The mini-batch
version uses smaller groups of rows, so each update sees fewer of them. With
a constant learning rate the weights keep moving around the best solution;
reducing the learning rate helps them settle, which explains the much
smaller gap.

**Tuning.** Only two things were tuned.

| `C` | Validation log-loss | Validation ROC AUC |
|---|---|---|
| 0.001 | 0.4942 | 0.7003 |
| 0.01 | 0.4887 | 0.7020 |
| 0.1 (chosen) | 0.4881 | 0.7021 |
| 1 | 0.4880 | 0.7021 |
| 10 | 0.4880 | 0.7021 |

Values that tie at three decimals are treated as equal and the strongest
regularisation among them is kept.

| Learning rate | Manual loop: largest coefficient gap to scikit-learn | `nn.Module`: largest coefficient gap to scikit-learn |
|---|---|---|
| 0.01 | 3.9e-01 | 4.4e-01 |
| 0.1 | 1.6e-02 | 1.2e-02 |
| 1.0 (chosen for both) | 7.4e-08 | 1.6e-03 |
| 5.0 | 5.3e-01 (does not converge) | 1.8e-03 |

The learning rate was judged on the training objective, not on validation.

## Limitations & next steps

Evidence the pipeline produces on how far the result can be trusted:

- ROC AUC is 0.702 on the validation year and 0.649 on the test year.
- Quarter by quarter it ranges from 0.62 to 0.73, including inside the
  training period: 0.656, 0.680, 0.623, 0.657, 0.683, 0.659, 0.648, 0.688
  (training), 0.699, 0.729, 0.721, 0.677 (validation), 0.622, 0.665, 0.625,
  0.682 (test).
- Refitting the same model on the 24 months before each test quarter does
  not help: AUC 0.619, 0.664, 0.621, 0.681, against 0.622, 0.665, 0.625,
  0.682 for the frozen model.
- Which ZIP codes fail most is not stable. Across the 33 ZIPs with at least
  60 routine inspections in every period, the correlation of ZIP fail rates
  is +0.41 between the training years and the validation year, +0.16 between
  validation and test, and -0.07 between training and test.

**Which number I would quote.** The test AUC of 0.65, because it measures
performance on the latest held-out year. The higher validation AUC of 0.70
appears consistent with a more favourable period: performance swings across
quarters even within training, and refitting on fresher data does not
improve the test result. This suggests temporal variability rather than
simply outdated training data, although it does not establish the exact
cause.

**What the health department should expect.** Useful but imperfect
prioritisation. On the test year the highest-ranked 20% of inspections
contained about 32.5% of the failures, better than the 20% a random
selection would find. However, the model misses many failures and its
performance changes over time, so it should support scheduling alongside
routine coverage.

**The four limitations I consider most important**

1. **Only inspected restaurants have outcomes.** The results may not
   represent every restaurant. I would evaluate the model prospectively and
   include a random audit sample.
2. **Inspector identity is missing.** Differences between inspectors may
   influence the outcome, which can bias the results. I would obtain that
   information and check how much outcomes differ between inspectors.
3. **The costs are assumptions.** A different cost ratio changes which
   restaurants are prioritised. I would obtain the real figures from the
   department.
4. **Chain matching is approximate.** Similar names may incorrectly group
   unrelated restaurants. I would check the matches against ownership
   records, or remove the feature if it adds no benefit.

## Generative AI use disclosure

I used Claude Code (Anthropic) throughout this assignment.

- **Code.** All the Python in this repository (`data/download_data.py`,
  `features.py`, `models.py`, `train.py`, `charts.py`, `app.py`) was written
  by the AI.
- **Exploration.** The AI ran the exploratory analysis and proposed
  candidate datasets, target definitions, cost ratios and feature groups,
  each with measured evidence and a recommendation.
- **Decisions.** I chose the dataset, the target definition (Fail only), the
  3 to 1 cost ratio and the feature groups from those options.
- **Report.** The factual parts of this report (the dataset description, the
  tables, the lists of what the pipeline does) were drafted by the AI from
  the pipeline's output. For the analysis and conclusions, the AI asked me
  questions with the relevant numbers; I answered them in my own words and
  the AI edited my answers for clarity.
