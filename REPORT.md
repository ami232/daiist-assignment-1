# ASSIGNMENT 1 REPORT

* Name: Paul
* Student ID: 19816
* Email: pcoyral.ieu2022@student.ie.edu
* Group: PPLEDBA

## Dataset

UCI Wine Quality (Cortez et al., 2009): Portuguese vinho verde, red and white files merged with an `is_red` flag.

- **One row** is one wine batch.
- **Size:** 6,497 rows; 11 lab measurements, `is_red`, and `quality` (expert median score, 0–10).
- **Why I chose it:** cheap objective lab data paired with an expensive subjective expert label is a natural screening problem.

## Business / real-life framing

A winery uses routine lab results to decide which batches to send to its expensive tasting panel as premium candidates.

- **Target:** premium = quality ≥ 7 (19% of batches).
- **Split:** 60/20/20 stratified. No time-based split, because there is no date column and each batch is scored independently.
- **Threshold:** set by business cost, not accuracy. A false positive wastes €50 of panel time; a false negative loses €200 of margin. Flag when p > 50/(50+200) = **0.20**, which is also the threshold that minimised cost on validation.

## Data preparation & feature engineering

- **Duplicates:** I removed 1,177 exact duplicates **before** splitting (5,320 rows left), so the test score isn't inflated.
- **Features:** clipping extreme values, log transforms for skewed columns, colour × feature interactions (red and white behave differently), splines for "sweet-spot" features (SO₂, volatile acidity, density), then standardisation. This gives 33 features.
- **Ablation (5-fold CV):** CV AUC rose from 0.837 (raw) to 0.846 (final).
  - I dropped chemistry ratios: they added nothing and made the problem much harder for gradient descent (condition number ~1,000× higher).

## Modeling: three implementations, one model

I used **logistic regression**, because the target is binary and the cost-based threshold needs probabilities. All three versions minimise the same log loss + L2 objective (C = 1).

Results on the test set (1,064 batches, threshold 0.20):

| Model | ROC AUC | Precision | Recall | Test cost |
|---|---|---|---|---|
| scikit-learn | 0.8105 | 0.401 | 0.743 | **€21,600** |
| Manual PyTorch loop | 0.8105 | 0.401 | 0.743 | **€21,600** |
| nn.Module + optim | 0.8105 | 0.401 | 0.743 | **€21,600** |
| Naive baseline (never flag) | 0.5 | — | 0 | €40,400 |

**The three agree.** The largest difference between their predicted probabilities is about 1e-5, and they make zero different decisions. The objective is convex, so gradient descent reaches the scikit-learn solution, but it needs about 20,000 full-batch steps. The model saves **€18,800 (−47%)** compared with the baseline.

## Limitations & next steps

- **Costs are assumed.** I'd use the winery's real panel cost and margin, then recompute the threshold.
- **Precision is only 40%.** Lab chemistry overlaps between premium and ordinary wines. I'd add grape variety or vintage as features.
- **No producer or date data.** I'd re-validate on new vintages or other wineries.
- **Single split.** I'd add bootstrap confidence intervals to show how stable the saving is.

## Generative AI use disclosure

I used AI to:
- help write and debug the code and dashboard;
- explain the methods;
- format this report.

I reran the pipeline and checked every number myself.
