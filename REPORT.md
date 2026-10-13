# Assignment 1 Report

- **Name**: Bruno Sanchis Dorado
- **Student ID**: 20262
- **Email**: bsanchis.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

FIFA 20 player data, originally scraped from sofifa.com and published on
Kaggle ("FIFA 20 complete player dataset"). I downloaded `players_20.csv`
from a public GitHub copy
(`github.com/apoorva-21/fifa-analysis`, file `data/players_20.csv`).

- One row is one football player in the game FIFA 20.
- The original file has 18,278 rows and 104 columns (8.9 MB).
- The file in this repo, `data/fifa20_players.csv`, keeps the same 18,278
  rows but only the 18 columns I looked at (1.5 MB): name, age, height,
  weight, overall, potential, market value, positions, preferred foot,
  international reputation, weak foot, skill moves, and the six main skills
  (pace, shooting, passing, dribbling, defending, physic).

I picked it because the columns are easy to understand without a data
dictionary, and because the obvious way to model it does not work well: a
straight line on the raw value fails, so the features need some thought.

## Business / real-life framing

**Scenario.** A club with a small budget wants to find bargains. The model
says how much a player with a given profile (age, potential, skills,
position) is usually worth. If the real price of a player is clearly lower
than that, the scouting team should take a look at him. The model produces
a shortlist; it does not decide who to buy.

What this means for the pipeline:

- **Target.** Market value in euros (`value_eur`), so this is a regression
  and I use linear regression. I predict the log of the value. Values go
  from €10K to €105.5M with a median of €775K, and for a scout being 20%
  wrong matters about the same on a cheap player as on an expensive one.
  The log turns "percent wrong" into something a line can fit.
- **Columns I refused to use.** If the model is supposed to tell me what a
  player *should* cost, it cannot be fed other prices.
  - `release_clause_eur` is calculated from the value, so it leaks the answer.
  - `wage_eur` is negotiated together with the value; it is another price.
  - `overall` is the game's own summary rating and the game sets the value
    almost directly from it. Adding it drops the validation RMSE from 0.32
    to 0.18, but then the model only copies the game's formula and the
    features I built stop mattering. I left it out on purpose.
- **Split.** Random 70 / 15 / 15 (11,222 / 2,405 / 2,405 players), fixed
  seed. A time-based split was not needed: the data is one single picture
  of FIFA 20, there are no dates, so nothing from the future can leak into
  training. Each player appears only once.
- **Metric.** There is no decision threshold because this is not a
  classification. The metric that matters for the business is the typical
  percentage error in euros (median absolute % error), because the bargain
  finder ranks players by "how far below the model's value is the real
  price". If the model is usually 20% off, a 20% discount means nothing.
  I also report RMSE and R² on the log scale (what the models actually
  minimise) and the mean absolute error in euros.

## Data preparation & feature engineering

**Data-quality decisions**

- **Goalkeepers removed (2,036 rows).** Their pace, shooting, passing,
  dribbling, defending and physic are empty; they have their own
  goalkeeper columns instead. Filling the gaps with averages would invent
  numbers that mean nothing for a goalkeeper. The model is for outfield
  players only.
- **Players with value 0 removed (210 outfield rows).** They are free agents
  with no club, so there is no price to learn from, and log(0) does not
  exist.
- This leaves 16,032 players with no missing values.

**Features** (17 in total). Twelve are used as they come: age, potential,
height, skill moves, weak foot, international reputation and the six
skills. Five are engineered. I checked each one on the validation set
(lower RMSE is better; basic features only = 0.5025):

| Added to the basic features | Validation RMSE (log) | Verdict |
|---|---|---|
| nothing | 0.5025 | starting point |
| squared age term | 0.3307 | by far the most useful |
| `best_skill` (highest of the six skills) | 0.4909 | small help |
| position flags (midfielder / attacker) | 0.4986 | very small help |
| `is_left_footed` | 0.5024 | no real effect |
| all five | 0.3234 | final model |

- **Squared age term.** Median value rises until about 26–27 and then
  falls. A line can only go up or down, so with age alone the model cannot
  describe this. Adding a squared term lets it draw a curve. This single
  feature cut the error by a third.
- **`best_skill`.** A striker with bad defending is not a worse striker.
  The six skills added together punish specialists, so I added the best of
  the six as "how good is he at his main job".
- **Position flags.** Defender / midfielder / attacker from the first
  position listed. Defender is the reference group, so two 0/1 columns are
  enough; a third would repeat information the model already has.
- **`is_left_footed`.** I expected left-footed players to be worth more
  because they are rarer. The data does not show it. I kept it because it
  does no harm, but it did not work.

**Scaling.** All features are standardised with a `StandardScaler` fitted
on the training set only, so nothing about validation or test players is
used. Gradient descent needs this because the features have very different
sizes.

## Modeling: three implementations, one model

Linear regression with an L2 penalty, because the target is a continuous
number. All three versions minimise the same thing: mean squared error on
log value plus `alpha × sum of squared weights` (divided by the number of
rows in PyTorch, to match how scikit-learn's `Ridge` counts it).

- **scikit-learn**: `Ridge(alpha)`, solved exactly.
- **Manual PyTorch**: `w` and `b` tensors, `loss.backward()`, update by hand
  inside `torch.no_grad()`, then `.grad.zero_()`. Learning rate 0.05, 5,000 epochs.
- **Standard PyTorch**: `nn.Linear`, `nn.MSELoss`, `torch.optim.Adam`.
  Learning rate 0.02, 3,000 epochs.

Results on the same test set (2,405 players):

| Model | RMSE (log) | R² (log) | Mean error (€) | Typical error (%) |
|---|---|---|---|---|
| Naive baseline (always the training mean) | 1.3521 | 0.000 | 2,277,463 | 78.1 |
| scikit-learn | 0.3209 | 0.9437 | 756,708 | 20.4 |
| PyTorch manual loop | 0.3209 | 0.9437 | 756,708 | 20.4 |
| PyTorch nn.Module + optim | 0.3209 | 0.9437 | 756,678 | 20.4 |

**Do the three agree?** Yes, now. The largest weight difference against
scikit-learn is 0.000003 for the manual loop and 0.00006 for Adam.

**They did not agree at first, and this is why.** My first squared term
was plain `age²`. scikit-learn and Adam gave RMSE 0.3206, but the manual
loop gave 0.3292 and its weights for age and age² were 3.75 and −3.36
instead of 4.63 and −4.19. The loss was still going down after 5,000
epochs: it had not finished. The reason is that `age` and `age²` are
almost the same column (correlation 0.995). When two features move
together, plain gradient descent moves very slowly along the direction
that separates them. scikit-learn solves the equations directly and Adam
adapts its step size, so they were not affected.

Instead of just adding more epochs, I changed the feature to
`(age − 25)²`, where 25 is the rounded average age. It describes exactly
the same curve, but its correlation with age is only 0.33. With that
change the manual loop converges in under 1,000 epochs and matches
scikit-learn.

Adam also needed one adjustment: with learning rate 0.05 it stayed about
0.001 away from scikit-learn's weights because it keeps jumping around the
minimum. With 0.02 the gap is 0.00006.

**Regularization.** I tried alpha = 0, 1, 10, 100, 1000 on validation.
100 was the best (0.32318) but 0 gives 0.32337, a difference in the
fourth decimal. With 11,222 rows and 17 features the model is not
overfitting, so regularization barely matters here. I report that rather
than pretend the tuning helped.

## Limitations & next steps

- **The value comes from a video game, not from the transfer market.** EA
  sets it with a formula. The model learns that formula, so a "bargain" is
  a player the formula prices low compared with similar players, not
  necessarily a cheap player in real life. Next step: use real transfer
  fees (e.g. Transfermarkt) as the target.
- **A big discount can simply be a model error.** The typical error is
  20%, so discounts under about 20–25% are noise. The model also overvalues
  players over 33 by about 14% (median), because a squared curve does not
  fall as fast as real values do at the end of a career, and undervalues
  players between €2M and €10M by about 23%. The top of the bargain list is
  full of 36-year-olds for that reason. Next step: add age bands (0/1
  columns) instead of one curve, and check the errors by age again.
- **`potential` is EA's opinion**, and the game computes it knowing the
  player's overall rating, so some of the information I removed with
  `overall` comes back through it. A real scout would have to estimate
  potential himself.
- **No goalkeepers.** They would need their own model with the
  goalkeeper skill columns.
- **One season only.** I cannot check whether a "bargain" really went up
  in value the following year. With the FIFA 21 file I could test exactly
  that, and that test would need a time-based split.
- **No wages or contract length**, which is what often makes a real
  signing cheap or expensive.

## Generative AI use disclosure

I used Claude Code (Anthropic) heavily in this assignment:

- It wrote the code in `train.py`, `app.py` and `features.py`, in the style
  of the Session 5 PyTorch notebook.
- It suggested candidate datasets and options for each design decision. I
  chose among them: the FIFA dataset, the "find bargains" framing, leaving
  out `overall`, wages and the release clause, and excluding goalkeepers.
- It found and fixed the disagreement between the manual loop and
  scikit-learn (the `age²` problem described above).
- It wrote the first full draft of this report from those decisions and
  from the real numbers produced by `train.py`. I reviewed it.
