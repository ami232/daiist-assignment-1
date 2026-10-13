# Assignment 1 Report

- **Name**: Sergio García
- **Student ID**: 19912
- **Email**: sgarcia.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

**Bank Marketing (with social/economic context)**, UCI Machine Learning Repository
(Moro, Rita & Cortez, 2014; CC BY 4.0), file `bank-additional-full.csv`, committed in `data/`.

- One row = one phone call to a client during a Portuguese bank's term-deposit campaigns.
- 41,188 rows, 20 inputs + target `y` (did the client subscribe?). The overall subscription rate is about 11%.
- Rows are **ordered by date, May 2008 to November 2010**. That matters for the split (see below).
- This is above the ~20k-row guide. I kept the full file because the temporal split needs the
  whole date range. Without it the test set would not cover a later period.

Why this dataset: it has messy, realistic features (a sentinel value, a leaking column, a
U-shaped age effect, collinear macro indicators) and it supports a clear cost-based decision.

## Business / real-life framing

**Decision supported:** before each outbound call, the campaign manager decides **whether this
client is worth calling**. Each call costs money (agent time). Each subscription brings margin.

Assumptions (hypothetical, editable in the dashboard): a call costs **EUR 10** and a subscription
is worth **EUR 80** net. Calling is worth it when `p × 80 > 10`, i.e. `p > 0.125`.

How this framing drives the pipeline:

1. **Target definition / leakage.** `duration` (call length) is only known *after* the call.
   If duration = 0 then the answer is always "no". It would make the model look great and be useless for a
   *before-the-call* decision, so it is dropped. `campaign` (calls so far in this campaign) *is*
   known before the next call, so it stays, as `campaign − 1`.
2. **Time-based split.** The bank trains on past campaigns and scores future ones. A random split
   would put 2010 calls into training and leak future economic conditions. I use the oldest 70%
   for training, the next 15% for validation and the newest 15% for test, in row (= date) order.


   | Split | Rows | Period | Subscription rate |
   |---|---|---|---|
   | Train | 28,831 | 2008-05 → 2009-04 | 5.6% |
   | Validation | 6,178 | 2009-04 → 2009-05 | 10.7% |
   | Test | 6,179 | 2009-05 → 2010-11 | 38.5% |

   The base rate changes sharply over time: there were far fewer calls after 2009, and a much larger
   share of them succeeded. A random split would have hidden this completely.
3. **Metric and threshold.** Accuracy is misleading: "call nobody" is ~89% accurate. The business
   ranks clients and calls the top ones, so I tune on **average precision** (precision-recall
   area). I set the **decision threshold to maximise profit on validation**, not to 0.5. Test is
   used once, for the final numbers.

## Data preparation & feature engineering

All row-wise transformations are in `pipeline.engineer()`. Scaling and one-hot encoding
(`ColumnTransformer`) are **fit on the training split only**.

| Feature | What I did | Why |
|---|---|---|
| `duration` | dropped | Post-call information → target leakage (above). |
| `pdays` | `never_contacted` flag (pdays = 999) + `days_since_prev` (0 if never) | 999 is a sentinel. As a raw number a linear model reads it as "contacted very long ago" and mixes two meanings into one slope. |
| `campaign` | `log1p(campaign − 1)` | Calls *before* this one (known at decision time). Long tail up to 56 with diminishing effect → log. |
| `previous` | `log1p(previous)` | Same long-tail reasoning. |
| `age` | age (scaled) + `age_lt25`, `age_ge60` flags | Subscription is U-shaped in age (students, retirees). One linear term can't bend. |
| `default` | `default_not_no` flag | Only 3 rows are "yes". "unknown" vs "no" is the useful signal. |
| job, marital, education, housing, loan, poutcome, contact, day_of_week | one-hot, `handle_unknown="ignore"` | Nominal categories. "unknown" is kept as its own level because missingness itself is informative. |
| macro (euribor3m, nr_employed, emp_var_rate, cons_price_idx, cons_conf_idx) | standardised, kept as a group | Economic context at call time. Highly collinear, which L2 regularisation handles. |
| `month` | one-hot, tested as a group | Captures campaign timing/seasonality, but it is entangled with the year and the economy, so it might not generalise forward in time. Whether to keep it was left to the validation ablation. |

**Feature-set ablation (on validation, scikit-learn only):** I compared `client+history+contact`,
`+macro`, `+macro+month` × C ∈ {0.01, 0.1, 1, 10}, scored by validation average precision
(full grid in `artifacts/tuning.csv`, also shown in the dashboard).
The winner was **`+macro+month` with C = 0.1**. Validation is only one month after training, so the
economic context and month effects still carry over there. On the test period (2009-05 → 2010-11)
they carry over much less (see Limitations).

## Modeling: three implementations, one model

**Model:** L2-regularised **logistic regression** (binary target). All three versions minimise
the *same* objective:

`mean log-loss + ½·λ·‖w‖²`, with `λ = 1 / (C · n_train)`, intercept not penalised.

That is scikit-learn's objective divided by `C·n`. Without matching λ, the three versions would
optimise different problems and could not be expected to agree.

- **scikit-learn:** `LogisticRegression(C, lbfgs)`.
- **Manual PyTorch:** raw `w`, `b` tensors, a hand-written numerically stable log-loss, `loss.backward()`,
  and a manual `w -= lr·grad` step under `torch.no_grad()`. Full batch, zero init.
- **Standard PyTorch:** `nn.Linear` in an `nn.Module`, `BCEWithLogitsLoss`, `torch.optim.SGD`, with
  `weight_decay = λ` on the weights only (bias in its own param group with no decay).
  Full batch, zero init.
- Learning rate chosen from {0.1, 0.5, 1.0} by the **lowest final training objective** (it is an
  optimisation setting, not a generalisation one). Chosen values are in `artifacts/meta.json`
  (`lr_manual`, `lr_standard`). Epochs: 4000.

**Test results** (threshold picked on validation, test used once):

| Model | ROC-AUC | Avg precision | Log-loss | Threshold | Calls | Precision | Recall | Profit (EUR) |
|---|---|---|---|---|---|---|---|---|
| scikit-learn | 0.5885 | 0.4528 | 0.7785 | 0.15 | 3,429 | 0.447 | 0.645 | 88,350 |
| Manual PyTorch | 0.5863 | 0.4517 | 0.7905 | 0.13 | 3,743 | 0.443 | 0.698 | 95,210 |
| Standard PyTorch | 0.5863 | 0.4517 | 0.7905 | 0.13 | 3,743 | 0.443 | 0.698 | 95,210 |
| Baseline: call everyone | 0.5 | 0.3845 | 1.1457 | — | 6,179 | 0.385 | 1.000 | **128,290** |
| Baseline: call nobody | 0.5 | 0.3845 | 1.1457 | — | 0 | — | 0 | 0 |

**Do they agree?**

- **Manual vs standard PyTorch: identical** to 4 decimals. Same objective, same full-batch gradient
  descent, same zero start. The only difference is who writes the update step (me vs `optim.SGD`).
- **PyTorch vs scikit-learn: very close in ranking, not identical.** ROC-AUC 0.5863 vs 0.5885,
  mean absolute difference in test probability 0.011 (max 0.135), but the intercept differs by
  about 1.47. scikit-learn's L-BFGS reaches the optimum. Plain gradient descent with a fixed
  learning rate had not fully converged after 4000 epochs. The slow direction is the
  **intercept vs. the one-hot columns**: every category has its own dummy, so adding a constant to all
  dummies of one feature and subtracting it from the intercept gives the same predictions. Only the
  small L2 penalty pins down the solution, so the loss surface is almost flat along that direction.
  Collinear macro features (euribor3m, nr_employed, emp_var_rate) add a second flat direction.
- **Why profit differs:** the thresholds chosen on validation came out at 0.15 vs 0.13 (small
  probability differences shift the profit-optimal cut-off by one grid step). Profit is very
  sensitive to the threshold in this period, so the gap in profit is bigger than the gap in AUC.
- *Fix I would try:* drop one level per categorical (`drop="first"`) to remove the redundant
  direction, and/or train for more epochs. Both should close the intercept gap.

**Against the baseline:** all three models beat the naive baseline on ranking (AUC ≈ 0.59 vs 0.5,
AP ≈ 0.45 vs 0.385), but **"call everyone" earns more** (EUR 128k vs EUR 88–95k). The reason is the
business maths, not the code. In the test period 38.5% of clients subscribe, far above the 12.5%
break-even. Almost everyone is worth calling, so any threshold that skips clients loses money.
The threshold was tuned on validation, where the rate was 10.7% and being selective paid off. That
choice does not transfer to a period where the base rate has tripled.

## Limitations & next steps

- **Concept drift is the main problem.** The subscription rate goes 5.6% → 10.7% → 38.5%.
  The calibration plot shows the models under-predicting badly in test: clients scored around
  0.05–0.10 actually subscribe at around 28%. The curve is also fairly flat, which matches the weak
  test AUC (≈0.59). *Next step:* retrain on a rolling window. Re-estimate the intercept (base
  rate) and the profit threshold on the most recent month before each campaign, instead of
  fixing them once.
- **Macro features extrapolate.** Interest rates (euribor3m) in the test period sit at or below the
  low end of what training saw. A linear model extrapolates linearly, which may be wrong.
  *Next step:* bucket the macro indicators, or model the economy separately from client propensity.
- **Business numbers are assumed.** EUR 10 / EUR 80 are hypotheses. The optimal threshold moves with them
  (try it in the dashboard). *Next step:* get real call-centre cost and deposit margin.
- **One row is a call, not a client.** The same client can appear several times. *Next step:* group by client
  (no ID is available here) to avoid overlap between splits.
- **Linear model.** Interactions (e.g. age × job) are not captured. *Next step:* add a few
  hand-picked interactions before moving to a non-linear model.

## Generative AI use disclosure

I used Claude (Anthropic) to generate the pipeline code (`pipeline.py`, `train.py`, `app.py`) and
to draft this report, including the analysis. I ran the code myself, all results come from my run,
and I reviewed every decision so I can explain it.
