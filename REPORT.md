# Assignment 1 Report

- **Name**: Tomás Roschge
- **Student ID**: 19045
- **Email**: troschge.ieu2022@student.ie.edu
- **Group**: BBADBA 5A 


## Dataset

**Predict Students' Dropout and Academic Success**, UCI Machine Learning Repository (dataset #697,
Realinho et al., 2022), committed as `data/students.csv`. One row = one undergraduate student of a
Portuguese higher-education institution: 4,424 rows × 37 columns (demographics, admission data,
socio-economic background, 1st- and 2nd-semester curricular units, three macro-economic indicators)
and a 3-class `Target`: Graduate 2,209 / Dropout 1,421 / Enrolled 794. No missing values, no duplicates.

Data finding: the three macro columns (unemployment, inflation, GDP growth) take only **10 distinct
combinations**, and they match Portugal's yearly figures for 2008–2017 (e.g. unemployment 16.2 % in 2013,
GDP −4.06 % in 2012). They therefore identify the **enrollment cohort**, which `pipeline.py` reconstructs
as a `cohort` column. Students per cohort range from 362 to 571.

I picked this dataset because, besides the fact that an analysis with Claude Code determined that it was fit to comply with the objectives of this assignment, the topic really interested me. I believe this kind of analysis are the ones that capture deep patterns that otherwise are hard to find. With a problem as serious as studens dropping out of school, I found this could be very interesting to analyze and, who knows, maybe provide a potential tool for solution.

## Business / real-life framing

Decision supported: **at the end of each student's 1st semester, the university decides who is offered a
mentoring programme** to prevent dropout. Costs (assumed): a missed dropout (FN) loses ≈ **€3,000** of
remaining tuition; mentoring a student who would have graduated anyway (FP) costs ≈ **€300**.

How this framing drives the pipeline:

| Framing implication | Pipeline decision |
|---|---|
| The decision is made at the end of semester 1 | All 2nd-semester columns are dropped (they don't exist yet at decision time, so using them would be leakage) |
| The model will be applied to *future* cohorts | Split by cohort, not randomly: train 2008–2013 (2,311 rows), validation 2014–2015 (681), test 2016–2017 (638) |
| "Enrolled" = still studying at the end of the normal course duration, final outcome unknown | Excluded; target = 1 if Dropout, 0 if Graduate |
| Errors have asymmetric costs (10:1) | Threshold chosen by minimising €-cost on the validation cohorts, then applied unchanged to test; C chosen by validation log loss |

**The Scenario:** The university has a limited budget for mentorship programmes. At the end of the 1st semester, the model assigns each student a dropout probability, so the university can offer the mentorships to those who are most at risk of dropping out.The result is a list of priorities on who to call.


**Why at the end of the first semester?:** The moment the enrollment is made, we only have demographic data stated on the application form, which is not useful to predict behavior (or would be poor indicators and turn out to be a non-representative analysis). On the other hand, at the end of the second semester, a lot of students would have already abandoned and it might be somewhat late to intervene. This is why we choose to do it with data at the end of the first semester; there is representative behavioral data and still time to act. The evidence here lies on the features itself; the 2 with the most weight on the entire model are from the first semester (`s1_failed_units` and `s1_approval_rate`)

**Why take out second semester and tuition/debtor (leakage):** The model can only use what actually exists at the moment of decision. therefore, data from the 2nd semester cannot be used, as the test would look better than how it actually will perform.
Additionally, we do not know when tuition/debtor were registered, and 94% of the ones who are not up-to-date are dropouts. The inference was made that these are most likely a consequence of the dropout, not a cause or previous signal. 
AUC dropped from 0.909 to 0.895. Better to have a worst metric that is actually realistic.

**Why a cohort-based split instead of a random one?:** The issue with random split is that it mixes same-year students in train and test. The model would already know the context from those years and metrics would turn out really optimistic. 
Additionally, the dropout rate between cohort changes between train (42%), validation (33%) and test (34%).Temporal split measures if the model endures that change, while the random split hides it. 

**Why are "enrolled" excluded?:**These are the students who, at the end of the natural period for the bachelor, were still enrolled, so we do not know if they ended up graduating or dropping out. 
If we chose to identify them as no-droppers, we would be teaching the model that the student advances slowly but doesn't drop out, and that might not be accurate. Excluding them leaves a clear target. 

**Why is the threshold chosen by cost and not by accuracy?:** Errors do not cost the same; a non-detected dropout costs us 3.000€, whilst an unnecessary mentorship costs us 300€.
Moreover, accuracy treats them equally: with a threshold of 0.5, the model would miss many dropouts to save up some mentorships
With a ratio of 10:1 in cost, we get 300 / (300 + 3,000) ≈ 0.09. Validation was 0.11 for sklearn. 


## Data preparation & feature engineering

All features are built in `pipeline.engineer_features`. The preprocessor is fitted on the train cohorts only
(`pipeline.build_preprocessor`).

- **1st-semester performance:** `s1_approval_rate` = approved / enrolled units, `s1_failed_units`,
  `s1_evals_per_unit`, `s1_no_evaluations`, `s1_units_without_eval`, `s1_has_credits` (credited units = transfer students).
- **Grade = 0 is not a grade:** in the raw data a 1st-semester grade of 0 means "nothing approved". `s1_no_grade`
  flags it, the grade itself is set to missing and imputed with the train median. Without this, a linear model
  would treat 0/20 as a real, extreme grade.
- **Financial:** `scholarship` only (tuition and debtor flags excluded, see *Dropped*).
- **Background:** `log_age_at_enrollment` (age is right-skewed: median 20, max 70), `mature_student` (≥ 23),
  `application_order`, `parents_higher_ed` (0/1/2, from 29 qualification codes mapped to higher-ed or not),
  `non_portuguese` (97.5 % are Portuguese, so 21 nationality codes collapse to one flag), `not_single`, `evening`,
  `male`, `displaced`, `special_needs`, plus admission and previous-qualification grades.
- **High-cardinality codes** (`course` 17 levels, `application_mode` 18, `prev_qualification` 17): one-hot,
  with levels that have < 30 train rows grouped into "infrequent" (also catches unseen levels at test time).
- **Dropped:** all 2nd-semester columns (leakage), the macro columns (cohort identifiers with only 10 values:
  they would learn cohort effects that can't extrapolate to new cohorts), parents' occupations (32 noisy codes each),
  `International` (duplicate of nationality), and **`Tuition fees up to date` + `Debtor`**: the data dictionary
  doesn't say when they were recorded, and 457 of the 486 students not up to date (94 %) are dropouts, so the flag
  most likely records the dropout (a student who left stops paying) instead of preceding it. With both features the
  test ROC AUC was 0.909 (tuition was the largest coefficient, +1.57); without them it is 0.895.
- **Scaling:** numeric features are standardised (needed for gradient descent to converge at one learning rate,
  and makes L2 penalise all features on the same scale). Binary flags are left as 0/1.

Final design matrix: 54 columns.

### What didn't work and why each choice matters.

**1. Tuition/debtor:** Tuition was the strongest coefficient out of the whole model, but 94% of those that were not up-to-date were dropouts, so it was possible leakage. By taking them out, AUC decreased from 0.909 to 0.895. A coefficient that dominates that much can be an important alarm. 

**2. Macro columns:** The three macro columns take only 10 distinct combinations, which match Portugal’s yearly figures for 2008–2017, so they identify the enrollment year. Instead of using them as features, they were useful to perform the cohort split. With them as features, the model would learn yearly patterns that cannot be applied to other years. 

**3. Grade = 0:** Means the student did not pass any subject, not a 0/20. It was marked flagged, set as missing and and completed with the median. If not, the model would take it as a real, extremely low grade. 

**4. Honesty disclaimer:** Besides from tuition, I did not test each feature by its own. The next step would be to take out a group of features at a time and test the AUC in validation.



## Modeling: three implementations, one model

Model: L2-regularised **logistic regression** (binary target). All three versions minimise the same objective
on the same standardised train matrix:

`mean BCE + λ/2 · ||w||²`, with `λ = 1 / (C · n_train)`, bias not penalised.

This matches sklearn's `0.5·||w||² + C·Σ logloss` divided by `C·n`.

- **scikit-learn:** `LogisticRegression(C=0.03)`, lbfgs. C picked from {0.01 … 10} by validation log loss
  (0.01 → 0.362, 0.03 → **0.353**, 0.1 → 0.353, 10 → 0.372).
- **Manual PyTorch:** raw tensors `w`, `b`, explicit sigmoid + BCE + L2, `loss.backward()`, and a manual
  `w -= lr·w.grad` step inside `torch.no_grad()`. Full batch, 4,000 epochs.
- **Standard PyTorch:** `nn.Module` with one `nn.Linear`, `BCEWithLogitsLoss`, and `torch.optim.SGD` with
  `weight_decay=λ` on the weights only. `weight_decay` adds `λ·w` to the gradient, which is the gradient of `λ/2·||w||²`.
- **Learning rate** (both torch versions): picked from {0.05, 0.1, 0.5, 1.0} by the final training objective,
  since it is an optimisation setting, not a model setting → 1.0 (all four reach 0.33885 within 1e-5 after
  4,000 epochs).

### Results on the test cohorts (2016–2017, n = 638, dropout rate 34.3 %)

Threshold chosen on the validation cohorts.

| Model | ROC AUC | PR AUC | Log loss | Threshold | Recall | Precision | Flagged | Cost (€) | €/student |
|---|---|---|---|---|---|---|---|---|---|
| Naive baseline (train rate 0.423) | 0.500 | 0.343 | 0.656 | 0.01 | 1.000 | 0.343 | 100 % | 125,700 | 197 |
| scikit-learn | 0.895 | 0.879 | 0.346 | 0.11 | 0.904 | 0.538 | 57.7 % | 114,000 | 179 |
| PyTorch manual loop | 0.895 | 0.879 | 0.346 | 0.13 | 0.890 | 0.580 | 52.7 % | 114,300 | 179 |
| PyTorch nn.Module + optim | 0.895 | 0.879 | 0.346 | 0.13 | 0.890 | 0.580 | 52.7 % | 114,300 | 179 |
| Policy: mentor nobody | – | – | – | – | 0.000 | – | 0 % | 657,000 | 1,030 |
| Policy: mentor everybody | – | – | – | – | 1.000 | 0.343 | 100 % | 125,700 | 197 |

Agreement with sklearn on the test set:
- **Manual loop and nn.Module:** max |Δ probability| = 0.0009 and max |Δ coefficient| = 0.0018 vs. sklearn. The two
  torch versions differ from each other by at most 1.2e-7 in probability (float32 precision).
- **Intercepts:** sklearn −0.185 vs. torch −0.188.
- **Same decisions as sklearn:** 95.0 % of test students when each model uses its own validation-chosen threshold
  (0.11 vs 0.13). **100 %** when they use the same threshold (checked at 0.11 and 0.13).
- **Why the thresholds differ:** on the validation cohorts the cost curve is jagged. At 0.11, sklearn costs €120,900 and
  torch €123,900. The difference is one FN: exactly one validation student (an actual dropout) gets 0.11013 from
  sklearn and 0.1098 from torch, so they land on opposite sides of 0.11. At 0.13 the costs are €122,100 and €121,800.
  That flips the argmin. Logged for all three models in `artifacts/metrics.json`
  (`agreement.val_cost_by_threshold_near_optimum`) and shown in the dashboard's agreement panel.

Largest coefficients (log-odds, same sign and size in all three):
- **Increase dropout risk:** `s1_failed_units` +1.06, course 9853 +0.57, `mature_student` +0.30, `log_age_at_enrollment` +0.24.
- **Decrease dropout risk:** `s1_approval_rate` −0.90, `scholarship` −0.50, course 9500 −0.38, `s1_grade` −0.36.

### Do the three agree? 

**1. They match:** Probabilities differ by at most 0.0009, coefficients by 0.0018, and the AUC is identical (0.895). The three minimize the loss function (BCE + L2 with lambda = 1/(C·n), without penalizing the bias) over the same data. The problem is convex, so it was a unique optimum.

**2. 95% vs 100%:** It is not that the models differ, it is the threshold choice. With the same threshold they make the same decision for 100% of students. Each chose it on validation, and there a single student (who actually dropped out) has 0.11013 in sklearn and 0.1098 in torch. That unique FN of 3000€ changes the optimal threshold from 0.11 to 0.13. This shows how fragile the threshold is in order to choose it with a tiny validation

**3. Intercept (-0.185 vs -0.188):** There is margin because the one-hot columns of each category add up to 1, same as the bias column. If I increase the intercept and decrease all the coefficients of one categorical variable by the same amount, predictions don't change, only the L2 penalization does (a little). 
So, the loss function is almost flat in that direction. Both get to the same loss, but stop at different points; sklearn by its solver's tolerance and torch because of the float32 precision. 

**4. Business:** The model costs 114,000€, and mentoring all of the students costs 125,700€. It saves 11,700€ and mentors 58% of students instead of 100%. The model only wins by a small margin because, with a cost of 10:1, not detecting a dropout is so expensive that "mentoring everyone" is a strong baseline. The value of the model lies in NOT MENTORING the 42% with less risk.

- In case the mentorship would be more expensive or if the slots are limited, the model's advantage would grow exponentially. It can be tested by switching the FP value in the dashboard

## Limitations & next steps

**1. Unstable threshold:** Validation has 681 students and one single student moved the threshold. The solution would be to validate with different cohorts on sliding windows (train until year N and validate on N+1, repeating) and average the threshold.

**2. Predicting risk ≠ predicting the effect of mentoring:** The model predicts who drops out, not to whom the mentorship will change the result. The solution would be to implement a random pilot (some students get the mentorship and others don't) to train an uplift model.

**3. Simplified cost:** It ignores that mentoring a TP costs 300€, assumes that the mentorship saves 100% of those who get it, and the costs are estimated. The solution would be to use the programme's real success rate metric of the program and perform a sensitivity analysis (the dashboard already allows or changes in the costs).

**4. Sensitive features:** Gender, nationality, and age could make the model target certain groups more. The solution would be to audit the recall and the rate of targeted students by group, and measure how much it gets lost by taking them out.

## Generative AI use disclosure

Claude Code (Anthropic, Claude Opus) was used to:
- propose and analyse the suitability candidate datasets and cost figures; I chose the dataset, framing and costs;
- write the whole code in `pipeline.py`, `train.py` and `app.py`;
- write the factual/descriptive parts of this report (dataset facts, pipeline description, results tables).
- build the analytical dashboard
- Help with gramar corrections
- Help with organising interpretations of results and understanding them in simple terms in order to write the justifications, framings, validations, results, and limitaitons

The framing justification, interpretation of results and limitations are my own.
