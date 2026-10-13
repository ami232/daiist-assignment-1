# Assignment 1 Report

- **Name**: Ana Maria Cea Marquez
- **Student ID**: 19756
- **Email**: acea.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

The data comes from the **snldb** project (github.com/hhllcks/snldb), which scraped
snlarchives.net. It covers every season of *Saturday Night Live* from season 1
(1975) to season 46 (April 2021). It is spread over several relational tables;
I used three of them:

- `casts.csv`: who was in the cast each season, whether they were a featured
  player, whether they anchored Weekend Update, how many episodes they did and
  whether they joined or left mid-season.
- `appearances.csv`: about 55,000 rows, one per appearance of a person in a
  segment of an episode.
- `titles.csv` and `episodes.csv`: what type each segment was (sketch, cold
  open, Weekend Update…) and where each episode falls in its season.

**One row of my modelling table is one cast member in one season** (e.g.
"Kate McKinnon, season 40"). After building the target and dropping the rows
explained below, the table has **577 rows**, 14 engineered features and a
binary target. All CSVs together are about 4 MB, so they are committed to the
repo and the pipeline rebuilds everything from scratch.

I picked it because I'm a big SNL fan and have followed the show for years, so
I already had a feel for how the cast changes from season to season and which
kinds of cast members tend to stick around. It also turned out to be a good
dataset for this assignment: the modelling table does not exist and has to be
built by joining and aggregating several tables, which is where the feature
engineering becomes non-trivial.

## Business / real-life framing

**Target:** `renewed = 1` if the person is in the cast again **the following
season**, 0 otherwise.

**Who uses it:** NBC's casting department. Every summer they have to plan
auditions and scouting for the next season before the producer announces
who is coming back, and many departures are not even the producer's decision
(people leave for film or other shows). A model that flags, at the end of each
season, which cast members are likely *not* to return lets them start looking
for replacements early. The model only ranks people by risk; the renewal
decision itself stays with the producer.

How this framing drove the pipeline:

- **Target definition.**
  - Season 46 is dropped: there is no season 47 in the data, so its label is
    unknown.
  - Someone who leaves and comes back years later counts as *not renewed*,
    because the question is about next season's cast.
  - The 17 people who had already left **during** the season (`last_epid`
    filled in) are dropped. At the end-of-season decision point they are
    already gone, so there is nothing to predict, and keeping them would
    teach the model something trivially obvious.
- **Time-based split, not random.** The model is used to predict next season
  from past seasons, so it must be evaluated the same way: train = seasons
  1–30, validation = 31–37, test = 38–45. This matters here because the
  renewal rate drifts a lot: 74% in train, 86% in validation and 87% in test.
  A random split would mix 2015 seasons into training to predict 1990 and hide
  that drift.
- **Metric for the threshold: business cost, not accuracy.** "Everyone is
  renewed" already gets 86.6% accuracy on test while catching zero
  departures, so accuracy is useless here. For the casting department the two
  errors are not equal:
  - **Missed departure** (model says "renewed", person leaves): a hole in the
    cast with no replacement lined up. **Cost 3.**
  - **False alarm** (model says "leaves", person stays): some wasted scouting
    time. **Cost 1.**

  The 3:1 ratio is an assumption, not a measured number. The threshold is
  chosen to minimise this cost (see below), and the dashboard lets you change
  both costs and watch the confusion matrix and total cost move.

## Data preparation & feature engineering

Every feature only uses information available **at the end of the season the
row describes**, so nothing leaks from the future across the time split. The
`StandardScaler` is fitted on training rows only.

| Group | Feature | What it captures |
|---|---|---|
| role | `featured` | Featured player rather than main cast |
| role | `update_anchor` | Anchors Weekend Update, a very visible role |
| role | `joined_mid_season` | Joined after the premiere (a trial / late addition) |
| tenure | `prior_seasons` | Seasons already completed (counted from earlier rows only) |
| tenure | `stuck_featured` | Still featured after 2+ seasons, i.e. never promoted |
| screen time | `apps_relative` | Appearances per episode **divided by that season's cast average** |
| screen time | `apps_relative_change` | Change in `apps_relative` vs. their own previous season |
| screen time | `cold_open_per_episode` | Cold-open appearances per episode |
| screen time | `update_per_episode` | Weekend Update appearances per episode |
| screen time | `second_half_share` | Share of their appearances in the second half of the season (< 0.5 = being used less as the season went on) |
| characters | `log_characters` | log(1 + distinct recurring characters played) |
| characters | `log_impressions` | log(1 + distinct impressions) |
| context | `cast_size` | Size of the cast that season (more competition) |
| context | `recent_renewal_rate` | Average renewal rate of the previous 5 seasons |

Main decisions:

- **Normalising by season.** The show changed a lot in 45 years (cast size,
  segments per episode), so 10 appearances per episode in 1980 is not the same
  as in 2015. `apps_relative` compares each person with their colleagues that
  season instead of using raw counts.
- **log1p on characters and impressions.** Most people have 0–2, a few have
  10+. The log stops those few from dominating the coefficient.
- **`recent_renewal_rate` and leakage.** Renewals for season *s−1* are known
  by the end of season *s* (they either came back in *s* or not), so a rolling
  mean over the *previous* seasons is legitimate. It lets the model follow the
  drifting base rate. Season 1 has no past and gets 0.5.
- **What I left out on purpose.** `actors.csv` has gender, but I did not use
  it: a model that informs casting decisions should not be using gender.
- **Data quality.** Some seasons have extreme renewal rates. After seasons 5
  and 10 nobody was renewed, because the whole cast was replaced. I kept them
  because they are real events, and `recent_renewal_rate` and `cast_size`
  give the model some context for them.

**Ablation** (validation, sklearn, dropping one group at a time):

| Model | Val AUC | Val log loss |
|---|---|---|
| all features | 0.761 | 0.360 |
| without role | 0.739 | 0.371 |
| without tenure | 0.747 | 0.362 |
| without screen time | **0.682** | **0.405** |
| without characters | 0.753 | 0.375 |
| without context | 0.744 | 0.378 |

Screen time is by far the most useful group. Tenure adds the least.

## Modeling: three implementations, one model

**Model:** logistic regression, because the target is binary. All three
implementations minimise the same objective:

> mean log loss + ‖w‖² / (2·C·n), with the bias not penalised.

- **scikit-learn:** `LogisticRegression(C)`. sklearn minimises
  ½‖w‖² + C·Σ log loss. Dividing by C·n gives the objective above with the
  same optimum, which is why the PyTorch versions use a penalty of 1/(2·C·n).
- **Manual PyTorch:** raw `w` and `b` tensors with `requires_grad`, sigmoid
  and BCE written by hand, `loss.backward()`, then `w -= lr * w.grad` inside
  `torch.no_grad()` and gradients zeroed every step.
- **Standard PyTorch:** `nn.Module` with one `nn.Linear`,
  `BCEWithLogitsLoss` and `torch.optim.SGD`. I did not use `weight_decay`,
  because it would also shrink the bias, unlike sklearn.

**Process (in this order, so test is touched only once):**

1. **C** chosen on validation log loss among {0.01, 0.1, 1, 10}: C = 10
   (weak regularisation; 14 features on ~360 rows does not overfit much).
2. **Learning rate** among {0.05, 0.5}: both reach the same training loss
   (0.4748), as expected for a convex problem; 0.5 kept.
3. **Threshold** chosen with **walk-forward validation**: for each season *s*
   from 21 to 37, fit on all seasons before *s* and predict *s*. Pooling these
   gives 222 rows and 29 departures (AUC 0.687), instead of only 13 departures
   in validation, so the threshold is much less noisy. Chosen threshold:
   P(renewed) < **0.66** → flag as leaving.
4. **Final models refit on train + validation (seasons 1–37)** with C, lr and
   threshold fixed, since those are the most recent seasons before test.
5. **Test (seasons 38–45, 127 rows, 17 departures)** evaluated once.

**Results on test:**

| Model | Threshold | Accuracy | Recall (departures) | Precision (departures) | Cost | ROC AUC |
|---|---|---|---|---|---|---|
| Baseline (always "renewed") | – | 0.866 | 0.00 | – | 51 | 0.500 |
| scikit-learn | 0.5 | 0.803 | 0.00 | 0.00 | 59 | 0.623 |
| manual PyTorch | 0.5 | 0.803 | 0.00 | 0.00 | 59 | 0.623 |
| nn.Module PyTorch | 0.5 | 0.803 | 0.00 | 0.00 | 59 | 0.623 |
| scikit-learn | 0.66 | 0.677 | 0.65 | 0.24 | **53** | 0.623 |
| manual PyTorch | 0.66 | 0.669 | 0.65 | 0.23 | 54 | 0.623 |
| nn.Module PyTorch | 0.66 | 0.669 | 0.65 | 0.23 | 54 | 0.623 |

**Do the three agree?** Yes. The coefficients match to about the third
decimal, and the largest difference in predicted probability on test is
1.6·10⁻⁴. The small gap comes from (a) PyTorch using float32 vs. sklearn's
float64 and (b) gradient descent stopping after 5,000 epochs without hitting
the exact optimum, while sklearn uses a second-order solver (lbfgs). That
tiny gap is enough to flip one person who sits right on the threshold, which
is why the cost is 53 for sklearn and 54 for both PyTorch versions.

**Reading the results honestly:**

- At the default 0.5 threshold the model is useless: it flags 8 people, all
  of whom stayed, and misses all 17 departures. Almost every predicted
  probability is above 0.5 because most people are renewed, which is why the
  threshold has to come from the costs.
- With the cost-based threshold it catches 11 of 17 departures (65%), but only
  about 1 in 4 people it flags actually leaves. At a 3:1 cost ratio the cost
  (53) is still slightly **worse than the baseline (51)**. This is the base-rate
  problem: with only 13% departures in test, false alarms outnumber hits.
- The model does rank people better than chance (test AUC 0.62), but AUC drops
  from 0.76 on validation to 0.62 on test. My explanation is that recent
  seasons renew almost everyone, so the few departures left are more likely
  voluntary, and screen-time data cannot predict that.
- **What I tried.** A first version (4 basic features, threshold picked on the
  13 validation departures, trained on seasons 1–30 only) had a test cost of
  65. Adding the engineered features, the walk-forward threshold, the
  train+val refit and `second_half_share` brought it to 53. That is a clear
  improvement, but it does not beat the baseline, and I stopped there rather
  than keep tuning against the test set.

## Limitations & next steps

- **Firings vs. voluntary exits are the same 0.** The data does not say why
  someone left. With more time I would hand-label departures from press
  coverage (e.g. Wikipedia cast pages) and either model only non-voluntary
  exits or add a separate class.
- **Very few positives.** 17 departures in test means one or two people change
  the cost noticeably. A rerun of the whole process with different cost ratios
  shows this: the model ties the baseline at 2:1 (34 vs 34), loses at 3:1
  (53 vs 51) and 4:1 (86 vs 68), and only wins at 6:1 (94 vs 102). Rather than
  trust one test number, I would report a rolling evaluation over all recent
  seasons.
- **The data stops in 2021**, so the most recent seasons are missing. Re-running
  the snldb scraper would add 4–5 seasons of test data.
- **Coefficients are not causal.** Several features are correlated (e.g.
  `apps_relative` vs. `update_per_episode` and `log_characters`), so some signs
  look counter-intuitive (`apps_relative` is slightly negative). I only read
  them as "what the model uses", not as reasons people get renewed.
- **Missing drivers.** Popularity outside the show, film offers or contract
  length almost certainly matter and are not in the data.
- **Practical use.** As an automatic yes/no it does not pay at 3:1. I would use
  it as a ranking: the dashboard's "Human review band" sends the uncertain
  middle to a person, and with cuts at 0.6 / 0.85 only 2 of the 17 departures
  slip through as "auto-renewed", at the price of reviewing 75 of 127 people
  by hand.

## Generative AI use disclosure

I used Claude (Anthropic) throughout this assignment and helped me with data loading, feature computation, the three model
implementations, evaluation and the Gradio dashboard and helped me 
download the dataset. 
