# Assignment 1 Report

- **Name**: Eloy
- **Student ID**: 19367
- **Email**: esanchez.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

I used the TMDB Box Office Prediction dataset from Kaggle, only `train.csv`
(because the test file has no revenue). One row is one movie: 3,000 rows and
23 columns (budget, genres, cast, date, runtime, collection, revenue...), with
release years from 1921 to 2017. I wanted to work with movies, and I picked
this one because it has far fewer than 20k rows and everything is in a single
file.

## Business / real-life framing

The model helps a cinema decide which movies to give more screens to before
release. Since the decision is made before the movie comes out, I only use
information from the dataset that is known before the movie is released in
cinemas. I split the data by date (70/15/15): train up to April 2010,
validation up to September 2013 and test from September 2013 to July 2017,
because a cinema always predicts new movies.

I treat it as classification because the cinema's decision is yes or no, not
an exact figure. Success = revenue above the median of the training set (about
$17.3M), which gives 50% hits in train and 49% in test. At first I thought of
"failure = does not recover its budget", with a success meaning that, besides
recovering the budget, it makes a significant extra profit, but 26.6% of the
train movies have no budget. Limitation: being above the median does not mean
the studio made money, but the model is for cinemas, and a cinema cares about
whether the movie fills screens.

For a cinema, missing a hit is worse than giving screens to a flop, because it
loses market. I made an estimate with the help of AI and assumed that missing
a hit costs 3 and giving screens to a flop costs 1 (an assumption, not data).
The threshold (0.30) is the one that minimizes that cost on validation. I use
cost and not accuracy because the two errors do not cost the same.

## Data preparation & feature engineering

I ended up with 36 variables: `budget_desconocido`, `log_budget`, `runtime`,
`es_saga`, 12 for the month (one-hot) and 20 for the genre (one-hot).

- **Budget.** 26.6% of the train movies have a budget of 0, and only 23% of
  those are hits, versus 60% of the rest, so a missing budget already says
  something. That is why I added an "unknown" column. I replaced the zeros with
  the median of the real budgets so that the model does not read 0 as "a free
  movie", and I used the logarithm because the budget is very skewed (mean
  $20M, maximum $300M) and a few blockbusters would dominate the model. The
  cost is that the real budget of those movies is lost and the median is only
  a guess.
- **Month as one-hot.** The month number is not a real scale (December is not
  "more" than January) and the pattern is not linear: June, July and December
  have around 62-65% hits, and January and September around 32-35%. One-hot
  gives each month its own weight.
- **`popularity` excluded.** It is measured after the release, so it mixes the
  hype before release with the result, and the cinema would not have it when
  it decides. However, there are always big productions and movies with more
  hype than others, and they tend to be better indicators of success, so I use
  the franchise effect (Spiderman, Avengers) as `es_saga`. This only captures
  movies that belong to a saga, not all hype (a highly anticipated original
  movie does not count), and `log_budget` also partly captures big
  productions.
- **Cast and crew excluded.** They are long texts with many names. I left them
  out to keep it simple, but they could help (see limitations).
- **Two-digit years in dates.** Pandas reads `65` as 2065. This was affecting
  146 movies, all from before 1969. I subtracted 100 years when the year came
  out after 2019. Without that, those old movies would have been sorted as if
  they were from the future and would have ended up in the test set, which
  should only contain the most recent ones.
- **`runtime`.** 2 movies had no value and 12 had 0 (impossible). I treated
  them as missing data and used the train median. Median and not mean because
  a few very long movies (up to 248 min) would pull the mean up, and not 0
  because that would say the movie has no duration.
- **Everything computed from the data comes from train only** (medians, genre
  list, scaler and the success median). Validation and test stand for the
  "future", and if I used them there would be information leakage and the test
  results would look better than they really are.

## Modeling: three implementations, one model

I use logistic regression because the decision is yes or no and it gives a
probability, which lets me move the threshold according to the cost of each
error. A linear regression would predict an exact revenue, which is very
skewed and not what the cinema needs, and then I would still have to turn it
into a decision. The three versions use the same standardized variables and
the same L2 regularization (C=1). I did not tune `C`: I kept scikit-learn's
default and used the same strength in the two PyTorch versions so that they
can be compared. The `lr` (0.5) and the epochs (5000) were chosen only so that
the manual loop reaches the same solution as scikit-learn, not to improve the
results.

Results on the test set at threshold 0.30:

| Model | Accuracy | AUC | Missed hits | Screens to flops | Cost |
|---|---|---|---|---|---|
| scikit-learn | 0.784 | 0.89 | 24 | 73 | 145 |
| PyTorch manual loop | 0.784 | 0.89 | 24 | 73 | 145 |
| PyTorch nn.Module + optim | 0.784 | 0.89 | 24 | 73 | 145 |
| Baseline: always "hit" | 0.491 | 0.50 | 0 | 229 | 229 |
| Baseline: always "flop" | 0.509 | 0.50 | 221 | 0 | 663 |

With the default threshold (0.5) the cost is 193, so the cost-based threshold
saves 48 points on movies it never saw.

The three agree because they are the same model: they minimize the same thing
(log loss plus an L2 penalty of the same strength, which in PyTorch is the
`l2` term in the manual loop and `weight_decay = 1/(C*n)` in the optimizer),
and the problem has a single best solution. The small differences (up to about
0.002 in probability) come from how each one gets there: scikit-learn uses a
different optimizer (L-BFGS) and the PyTorch ones use gradient descent, which
ends up very close to the optimum but not exactly on it. When I tried
`lr=0.1` and 2000 epochs in the manual loop, the gap with scikit-learn grew
from about 0.002 to about 0.029, because it had not converged. That was enough
to move the best threshold from 0.30 to 0.31 and the test cost from 145 to
146. So the three versions only agree if the optimization runs long enough.

## Limitations & next steps

- **Success is not profit.** Being above the median does not mean the movie
  made money. With data on tickets sold and screen costs from the cinema
  itself, I would define success as profit per screen.
- **Missing budgets.** 26.6% of train (30% of test) has no budget and I used
  the median. I would look for the real ones from another source (TMDB API or
  IMDb) and see how much the results change if I remove those movies.
- **The 3:1 ratio is invented.** The threshold depends on it and I have no data
  to support it. I would ask a cinema for its real margin per screen. The
  dashboard slider shows how the cost changes with the threshold, but the
  ratio is fixed at 3:1.
- **A single split and a single test period (2013-2017).** The industry changes
  (streaming, prices) and I did not adjust money for inflation. I would use
  several time windows and adjust for inflation.
- **Variables I left out.** Cast and crew (for example the director's track
  record or the lead actors) probably add information. I would build them using
  only movies released before each release, so as not to leak the future.
- **The model is linear.** It does not capture combinations such as a high
  budget within a saga. I would try interactions or a tree-based model and
  compare the cost.
- **Hyperparameters not tuned.** I used C=1 by default. With more time I would
  try several values (0.01, 0.1, 1 and 10) and choose the one with the lowest
  cost on validation, never on test.

## Generative AI use disclosure

I used Claude throughout the project. It helped me understand the problem statement and set up the project, find a suitable dataset (I chose the movie dataset), and conceptualize the problem. I proposed the cinema scenario—based on the idea that missing out on a hit is worse than screening a flop—and suggested excluding popularity; Claude proposed defining "success" using the training set median and a 3:1 ratio, a suggestion I discussed and accepted. Claude helped write most of the code for `train.py` and `app.py` (which I then executed and verified) and structured the report using a template; I filled in the content—drafting the descriptions, decisions, and work based on my choices—while Claude reviewed and edited the text to ensure a well-structured, coherent final report.
