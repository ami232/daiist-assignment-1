# Assignment 1 Report

- **Name**: Bernarda Andrade
- **Student ID**: 19119
- **Email**: bandrade.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

I used the *Speed Dating & Attraction Outcomes* dataset by sergionefedov on
Kaggle, a synthetic dataset modelled on the Columbia speed dating
experiment by Fisman & Iyengar (2002–2004). Because it is synthetic, I understand that it does not
describe real people, so any patterns found can help show whether the methods used can predict an outcome. Each row
is a four minute date between a man and a woman at a speed dating event, with
both people's ages, whether they share a race or field, a shared interests
score, the ratings each gave the other (attractiveness, sincerity,
intelligence, fun, ambition), their stated preferences and self ratings, how
often each goes out, and each person's yes/no decision. The file has 14,504
rows and 30 columns covering 120 events and 2,548 participants (each attends
one event), with no missing values. The target, `match`, is 1 when both people
said yes. I chose this dataset because I find
it interesting to analyse human attraction, especially to find out whether it
is possible to predict outcomes like a mutual match.

## Business / real-life framing

This model helps a speed dating organiser. During each event every participant writes a scorecard that rates each partner. Many participants, however, only send their yes/no answers days later, or they never send them. The morning after the event, the organiser uses the model to decide which pairs are likely to match mutually. The organiser then sends a "you two should talk" nudge to those pairs while interest is still fresh. Because the model makes predictions after the dates, the model can use both profile information (age, field of study, race, stated preferences, self ratings, how often each person goes out) and the scorecard ratings each person gave their partner. The question the model tests is whether a mutual match can be predicted from how the date went, not only from who the people are. I expected a real but moderate signal.

**Target.** The target is `match`, which is set to 1 when both participants said yes. I dropped `male_decision` and `female_decision` because `match` is computed directly from those two decisions, so keeping them would leak the answer into the features.

**Cost of errors.** In this case a false positive is a notification sent to two people who would not have matched. This could be annoying and may lead users to ignore future suggestions. A false negative is a real match that never reconnects, which is exactly the outcome the organiser's service tries to create so in this case it would be lossing opportunity to satisfy its customers. Therefore I assume a missed match costs three times as much as a wasted nudge, so business cost equals FP × 1 + FN × 3. I chose the decision threshold that mnimizes this cost on the validation set instead of using the default 0.5.

**Metric for comparing models.** To compare models independently of the threshold, I used ROC‑AUC and PR‑AUC. PR‑AUC is especially suitable because matches are relatively rare (19.6% of dates). Accuracy would be misleading because a model that always predicts "no match" already scores about 80%.

**Split.** I did not use a time based split. The data has no dates and `event_id` is only an identifier, so there is no "future" that could leak into training. Instead I split by `event_id`, keeping whole events together in train, validation or test. Each participant has 6–15 dates (11 on average) within a single event. A random split would place the same person in both train and test. The model could then memorise individuals instead of learning general patterns, which would inflate the test score. That is why I split by `event_id` to keep it together.

## Data preparation & feature engineering

**Data quality and leakage.** The file has no missing values, so nothing was
imputed. I
verified that `match` equals `male_decision` AND `female_decision` on every
row, then dropped both decision columns, since keeping them would have caused model memorization. That left 24 candidate features. The ID columns are kept
aside to build the split and are never model inputs. Every choice below was
made with 5 groups cross validated  by `event_id` on the 84 training
events (validation and test were not used), and the number to beat is the
naive baseline (CV PR-AUC 0.198,(no better than random). All features are
standardised with a scaler fitted on the training split only.

**Decision 1: "Both interested" features (dropped).** A match needs both
people to say yes, so I added `min` and product features of the two people's
ratings, plus `overall_by_male` and `overall_by_female` (the average of the
five scorecard ratings each person gave the other). I expected them to help
because logistic regression combines its inputs by adding them. Grouped CV
showed no gain (0.397 vs. 0.397 for the base set), so I dropped them. The
reason is that logistic regression doesn't add up probabilities. A match happens when he says yes and she says yes, so
P(match) ≈ P(he says yes) × P(she says yes), and multiplying probabilities is
the same as adding their logarithms. So it had no sense to keep them, that is why I made the decision to drop this feature."

**Decision 2: Rater calibration (dropped).** Some people rate everyone a 9, so
I compared each rating with the rater's own average, such as "she rated him 2
points above her usual." This is allowed in my scenario because all of a
person's scorecards exist the morning after, and the averages use ratings
only, never decisions. I expected calibration to help, but grouped CV showed
it slightly lowered PR-AUC (0.394 vs. 0.397), so I removed it. One possible
reason is that with only 6 15 dates per person, the personal average  was just noise.

**Decision 3: Preference weighted ratings (kept).** I multiplied how much a
person says they value attractiveness and intelligence by how attractive
and intelligent they rated their partner. Stated priorities often predict
real choices poorly, so I treated this as an experiment and kept it only if it
improved CV PR-AUC. It improved it from 0.397 to 0.403, so I kept it. This
was the only engineered group that helped.

**Decision 4: `goes_out` encoding (ordinal).** The categories have a natural
order (rarely = 0 to very often = 3), so one ordered column keeps that
information with fewer features than one hot encoding. CV showed no clear gain
from one hot (0.396 vs. 0.397), so I kept the ordinal version.

**Decision 5: Profile features (kept).** Age, race, field and preferences are
almost uncorrelated with matches: on their own they reach only 0.237 CV PR-AUC
(raw ratings alone: 0.394). But removing them from the final set lowers CV
PR-AUC from 0.403 to 0.394, so they matter in combination. The model uses them
as context for interpreting the ratings: stated preferences get negative
coefficients alongside the positive preference weighted features, so the same
rating counts differently depending on how much the person says they care
about that attribute.

Each engineered group was compared with the same base set and kept only if it raised mean CV PR-AUC. The final model uses 28
features: the base set plus the four preference weighted features.

| Variant (grouped 5-fold CV, training events) | Features | CV ROC-AUC | CV PR-AUC (± std) |
|---|---|---|---|
| Naive baseline | 0 | 0.500 | 0.198 ± 0.006 |
| Profile features only | 13 | 0.570 | 0.237 ± 0.017 |
| Raw ratings only | 11 | 0.716 | 0.394 ± 0.008 |
| Base: profile + raw ratings | 24 | 0.717 | 0.397 ± 0.009 |
| Base + both-interested (dropped) | 34 | 0.718 | 0.397 ± 0.008 |
| Base + rater calibration (dropped) | 37 | 0.715 | 0.394 ± 0.009 |
| **Base + preference-weighted (final)** | **28** | **0.722** | **0.403 ± 0.007** |
| Base, `goes_out` one-hot instead of ordinal (rejected) | 30 | 0.717 | 0.396 ± 0.009 |
| Final set without profile features | 15 | 0.716 | 0.394 ± 0.008 |

## Modeling: three implementations, one model

**Why logistic regression.** The target is binary, and I need a probability
for each pair, not just a class, because the organiser's decision depends on a
cost based threshold. Linear regression can output values below 0 or above 1
and is not fitted to a yes/no outcome, so it cannot give a trustworthy
probability. Logistic regression also has interpretable coefficients, which
helps me say which parts of a date relate to a match.

**Implementations and how I made them comparable.** I fitted the same model
three ways: scikit-learn, a manual PyTorch loop, and the standard
`torch.nn.Module` / `torch.optim` workflow. To compare them fairly, I used the
same 28 features (standardised with training-set statistics) and the same L2
penalty. scikit-learn's regularisation strength C = 0.1 was chosen by grouped
CV; the equivalent PyTorch penalty is λ = 1/(C·n) = 0.00094, which makes the
PyTorch loss equal to scikit-learn's objective divided by C·n, so both have the
same optimum. None of them penalises the intercept.
The PyTorch versions use full batch gradient descent with learning rate 0.5
for 3,000 epochs. I tried 0.05, 0.1, 0.5 and 1.0 and chose 0.5 because it
reached the lowest training loss (1.0 was equally good); the loss stopped
improving after about 1,100–1,300 epochs, so 3,000 leaves a safe margin. For
each model I picked the threshold that minimises FP×1 + FN×3 on the
validation set and then evaluated once on the test set.

| Implementation | Test ROC-AUC | Test PR-AUC | Threshold | Test cost (FP×1 + FN×3) |
|---|---|---|---|---|
| scikit-learn | 0.696 | 0.403 | 0.29 | 841 |
| Manual PyTorch loop | 0.696 | 0.403 | 0.29 | 841 |
| `nn.Module` / `optim` | 0.696 | 0.403 | 0.22 | 898 |
| Naive baseline | 0.500 | 0.191 | n/a (nudge no one) | 1,065 |

**Do the three agree?** Yes, the models agree; only the chosen threshold
differs. Their predictions agree: the maximum absolute
difference in predicted probability is 0.0005, and ROC-AUC and PR-AUC are the
same to 3 decimals. The final training losses match to 7 decimals (0.4435409
for all three), and the coefficients differ by at most 0.001. The best model
reaches a test PR-AUC of 0.403 against a baseline of 0.191, roughly double,
which fits my expectation of a real but moderate signal. It lowers the test
cost from 1,065 to 841 (−21%).

**What the coefficients say.** Because all features are standardised, the
coefficients can be compared directly. The strongest is `shared_interests`
(+0.53): feeling that you have interests in common relates to a match more than
any single rating. Next come the preference weighted attractiveness ratings
(+0.45 when the female is the rater, +0.33 when the male is): being found
attractive by someone who says attractiveness matters to them. Stated
preferences on their own are negative (−0.34 female, −0.23 male), which is the
"context" effect from Decision 5: for the same weighted score, a person who
claims to care a lot about an attribute is harder to impress. Fun and
sincerity ratings add smaller positive effects (about +0.19 each). These are
associations in synthetic data, not causes.

**Coefficient differences in an earlier version.** Before I dropped the "both
interested" features, a few coefficients differed between scikit-learn and the
two PyTorch versions (e.g. `prod_overall` −0.42 vs. −0.31, `overall_by_male`
0.21 vs. 0.09), even though their predictions and training objective were
almost identical. The cause was perfect multicollinearity: `overall_by_male`
and `overall_by_female` are exactly the average of five raw ratings that were
also features, so many coefficient combinations gave the same predictions.
Gradient descent approaches the regularised optimum very slowly along this
flat direction, while scikit-learn's solver reaches it. Dropping that group
(for the reasons in Decision 1) removed the problem, which is why the final
coefficients agree.

**Why `nn.Module` / `optim` has a different threshold.** Although its
probabilities match the other two almost exactly, its cost minimising
validation threshold came out at 0.22 instead of 0.29. The validation cost
curve is very flat between those values: at 0.22 it is 891 for this model
(894 for scikit-learn), and at 0.29 it is 893 for both, so a 2 point
difference decides which threshold "wins." On the test set this changes the
cost from 841 to 898. The difference comes from threshold selection on a
small validation set (18 events), not from the model.

## Limitations & next steps

**Synthetic data.** The patterns I found describe the generator, not real
daters, so the results show whether the method can find a signal, not how
attraction works. For example, preference weighted ratings helped here, but
they may not in real data. With more time I would run the same pipeline on
real event data, such as the original Columbia speed-dating data, and compare
the size of the signal.

**Small validation set.** The validation set has only 18 events, so the chosen
threshold could change with a different split. This already happened between
implementations: `nn.Module` / `optim` picked 0.22 instead of 0.29 because the
validation cost curve is almost flat, which raised its test cost from 841 to
898 even though its predictions are the same. I would use repeated
event-grouped CV over all 102 training and validation events, choose the
threshold on pooled out-of-fold predictions, and report the spread across
folds.

**Feature decisions rest on small differences.** The fold-to-fold standard
deviation of CV PR-AUC is about 0.007–0.009, while most feature-group effects
were smaller than that (both-interested −0.0005, calibration −0.003, one-hot
−0.001). Even the preference-weighted gain (+0.006) is within about one
standard deviation. I would repeat the grouped CV with several random event
shuffles and compare feature sets fold by fold, keeping a group only if it
wins consistently.

**Modest PR-AUC.** A test PR-AUC of 0.403 (ROC-AUC 0.70) means the model is
far from perfect: at the chosen threshold it nudges 408 pairs, of which only
158 (39%) actually match, and it misses 197 of the 355 real matches. Logistic
regression can only use the interactions I build by hand, so I would try
gradient boosting, which finds interactions itself, and compare it on the same
grouped CV.

**Assumed cost ratio.** The 3× ratio is my assumption, and the best threshold
moves with it. The dashboard already lets the user change both costs and
watch the cost curve and threshold respond; as a next step I would report the
cost minimising threshold for 2×, 3× and 5× to show how sensitive the decision
is, and estimate the true ratio from the organiser's data on how nudges affect
retention.

**Hypothetical scenario.** The idea that decisions arrive late is an
assumption I made, not something in the data. In practice, scorecards may also
be missing for some participants. I would test how performance changes when
some scorecard ratings are missing, and compare against a profile only model
that could run before the date (CV PR-AUC 0.237, barely above the 0.198
baseline).

## Generative AI use disclosure

I used Claude (Anthropic) throughout this assignment. It helped me choose the dataset, wrote the pipeline and dashboard code, investigated why coefficients differ between implementations, and corrected and formatted wording in this report. It also suggested initial reasoning for the scenario and for my feature decisions, which I reviewed and edited. The framing, the 3:1 cost ratio, and the keep and drop decisions on features are mine, and I based them on the grouped CV results. I also re ran the notebook, confirmed no person appears in two splits, compared the results with the naive baseline. Lastly I also wrote the analysis in the report. 
