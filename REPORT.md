# Assignment 1 Report

- **Name**: Samer
- **Student ID**: 19583
- **Email**: sroz.ieu2022@student.ie.edu
- **Group**:

## Dataset

I used NBA games from 2004 to 2022 from Kaggle, [nathanlauga/nba-games](https://www.kaggle.com/datasets/nathanlauga/nba-games), using `games.csv` and `teams.csv`. I committed the seasons 2011-12 to 2022-23 in `data/`, which is 15,371 rows and 2.3 MB.

Each row is one game. It has the date, the season, the two team IDs, the final points for each team and some box score stats like shooting percentages, assists and rebounds.

After keeping only regular season games that have a score, there are 13,445 games. I picked this dataset because I follow basketball and because the raw table has no useful pre game features at all. Every feature has to be built from each team's history, so feature engineering is most of the work.

## Business / real-life framing

A sportsbook has to post an opening point spread for every game before betting opens. The spread is the book's guess of the home team's final margin. If the line is far off, sharp bettors take the other side and the book loses money. So the model predicts the home team's margin, home points minus away points, using only what is known before tip off.

This framing drove the pipeline in four ways.

1. **Target.** I predict the margin in points, not who wins. A book needs to know by how much, not just the winner, so this is a regression problem and I use linear regression.
2. **No leakage from the game itself.** The box score columns are only known after the game ends. They would make the model look great and be useless before tip off, so I never use them. Every feature is built with `shift(1)` inside each team and season, so a game only sees the games before it.
3. **Time based split.** A book sets lines for a season that has not happened yet, using the seasons before it. The league also changes over time, for example the home edge moves from season to season. A random split would mix games from every season into both training and test, which hides that drift and gives a test score that is too optimistic. I train on 2012-13 to 2017-18, use 2018-19 to pick the regularisation strength and learning rates, and test once on 2021-22. The final models are refit on 2012-13 to 2018-19.
4. **Metric.** I use MAE in points, because every point of error is a point a bettor can use against the book. The models are still fitted with squared error since that is what linear regression minimises, and squared error punishes big misses more, which a book also cares about. I also report the share of games within 3 points and how often the sign of the prediction picks the winner. In the dashboard a slider shows how many games the line misses by more than N points.

I left out 2019-20 and 2020-21. COVID stopped the season, the playoffs were played in a bubble with no real home team, and most of 2020-21 was played in empty arenas. Home court advantage dropped from about 2.7 points to 0.9 points in 2020-21, which shows up in the Home court by season tab of the dashboard. Training on those seasons would teach the model a home edge that does not exist with fans in the building. 2022-23 is only half a season in the data, so I did not use it either.

## Data preparation & feature engineering

**Cleaning.** Game IDs starting with 2 are regular season games. I dropped preseason games because teams play their bench, playoff and play in games because rotations change and the same teams keep playing each other, 29 duplicate game IDs, and games with no score.

**Team game log.** I turned each game into two rows, one for each team, with the margin from that team's side. All features are computed per team and season on this log, using only earlier games.

These are the final features. All of them are home minus away, or a flag.

| Feature | What it is | Why |
|---|---|---|
| `rating_diff` | Season to date average margin, shrunk toward a prior | Main measure of team strength |
| `form_diff` | Average margin over the last 10 games | Picks up injuries, trades and hot streaks that the season average is slow to show |
| `rest_diff` | Days of rest, capped at 4 | Tired teams play worse |
| `home_b2b`, `away_b2b` | Second night of a back to back | Strongest fatigue signal |
| `altitude` | Home team is Denver or Utah | Both arenas are far above sea level |

**Shrinking the team rating.** After one game a team's average margin is just that one game, which is mostly noise. So the rating is a weighted average of the team's results so far and a prior, `(n * avg + 10 * prior) / (n + 10)`. The prior is last season's average margin pulled halfway to 0, because rosters change over the summer. After 10 games the team's own results count as much as the prior. Teams with no history start at 0. I set K at 10 and the prior weight at 0.5 before looking at results, then checked a few other values in the table below.

**Feature check on the validation season.** This uses Ridge and MAE in points, so lower is better. The validation season only has 1,230 games, so small differences can be luck. Each row shows the difference from the final set and its standard error, computed game by game. I treat anything smaller than about two standard errors as noise.

| Features | Val MAE | vs final set | Std error |
|---|---|---|---|
| Constant, home court only | 11.365 | +1.191 | 0.148 |
| Rating, not shrunk | 10.436 | +0.262 | 0.057 |
| Rating, shrunk, K 10, prior 0.5 | 10.252 | +0.077 | 0.028 |
| Shrink check, K 5 | 10.277 | +0.102 | 0.029 |
| Shrink check, K 20 | 10.241 | +0.067 | 0.034 |
| Shrink check, prior weight 0 | 10.322 | +0.147 | 0.037 |
| Shrink check, prior weight 1 | 10.240 | +0.065 | 0.036 |
| + form | 10.235 | +0.061 | 0.024 |
| + rest and back to backs | 10.187 | +0.013 | 0.011 |
| + altitude, final set | 10.175 | 0 | 0 |
| Final set + away road trip | 10.178 | +0.003 | 0.003 |
| Final set + team home edge | 10.181 | +0.007 | 0.008 |

- Shrinking clearly helps. It is 0.18 points better than the raw season average, which is more than 3 standard errors.
- Using last season as the prior helps too. It beats shrinking toward 0 by about 0.07 points, around 2 standard errors.
- The exact K and prior weight do not matter much. K 5, 10 and 20 and prior weights 0.5 and 1 are all within about one standard error of each other, so I kept the values I set at the start and did not tune them.
- Form and rest are real gains. Form adds 0.017 and rest with back to backs adds 0.048.
- Altitude is borderline. It adds 0.012, which is about one standard error.

**What did not work.**

- **Away road trip.** This counted how many away games in a row the visitor had played, to capture travel fatigue. It changed nothing, it was 0.003 worse and its coefficient was about 0, so I dropped it. My guess is that a long road trip only hurts through tired legs on back to backs, which the back to back flag already covers, but I did not test that.
- **Team specific home edge.** This was a team's home margin minus its overall margin. It came out slightly worse, well inside the noise. A team only plays about 41 home games a season, so that number is too noisy to help.
- **Altitude is not a clean altitude effect.** It is just a flag for two teams, so it also picks up anything else about playing in Denver or Utah, like their crowds. With only two teams I cannot separate those.

**Scaling.** Features are standardised with `StandardScaler`, fitted on the training data only. This matters for the two PyTorch versions, because gradient descent is slow when one feature is in the tens and another is 0 or 1.

## Modeling: three implementations, one model

**Model.** I use linear regression with an L2 penalty, which is Ridge, because the target is continuous. All three versions minimise the same thing. sklearn Ridge minimises `sum((y - Xw - b)^2) + alpha * sum(w^2)`. Dividing by n gives `mean((y - Xw - b)^2) + (alpha / n) * sum(w^2)`, which is the loss I wrote in both PyTorch versions. I wrote the penalty by hand instead of using `weight_decay` so the bias is not penalised, the same as in Ridge.

- **scikit-learn.** `Ridge(alpha)`, solved exactly.
- **Manual PyTorch.** Raw tensors `w` and `b` with `requires_grad`, then the loss, `backward()`, `w -= lr * w.grad` inside `torch.no_grad()`, and zeroing the gradients after each step. Full batch, 1,000 epochs.
- **Standard PyTorch.** `nn.Linear`, `nn.MSELoss` plus the penalty, and `torch.optim.Adam`. Full batch, 1,000 epochs.

**Tuning.** I picked alpha on 2018-19. Values from 0.1 to 1,000 gave almost the same validation MAE, between 10.174 and 10.196, and only 10,000 was clearly worse at 10.476. With 7,379 training games and 6 features there is very little to overfit, so regularisation barely matters here. I used 100.

The learning rate's job is to reach the minimum of the training loss, so I picked it by the lowest final training loss and not by validation MAE. sklearn's exact minimum is 149.0555. In the manual loop, lr 0.1 and 0.5 reach it and lr 0.01 stops at 149.0557. With Adam, lr 0.1 reaches it, 0.01 stops at 149.40 and 0.001 stops at 166.08. Both versions use 0.1.

**Results on the test season.** This is 2021-22, 1,230 games.

| Model | MAE | RMSE | R2 | Within 3 pts | Picks winner |
|---|---|---|---|---|---|
| Naive baseline, home team by 2.7 | 12.212 | 15.337 | -0.004 | 12.1% | 54.4% |
| scikit-learn | 11.245 | 14.256 | 0.132 | 17.3% | 64.6% |
| Manual PyTorch | 11.245 | 14.256 | 0.132 | 17.3% | 64.6% |
| Standard PyTorch | 11.245 | 14.256 | 0.132 | 17.3% | 64.6% |

All three beat the baseline by about 1 point of MAE and pick the winner 65% of the time, against 54% for always picking the home team.

**Agreement.** The three versions agree. All coefficients match to 3 decimals, and the biggest gap between any two models on any test game is 0.000 points.

One thing came up during tuning. With lr 0.01 the manual loop gets the training loss almost right, 149.0557 against 149.0555, but the rating weight is visibly off, 4.833 against 4.854. That is because `rating_diff` and `form_diff` have a correlation of 0.78. Lots of mixes of the two give almost the same loss, so the loss surface is very flat in that direction and a small learning rate barely moves along it in 1,000 steps. The loss looks finished while the weights are not. A learning rate of 0.1 gets all the way there.

**Coefficients in real units.** These are from the sklearn model.

- Every point of rating difference is worth 0.83 points of margin.
- An away team on a back to back is worth 1.3 points to the home team, and a home team on a back to back costs it 1.2.
- A home game in Denver or Utah is worth 1.8 extra points on top of the normal home edge.
- The base home edge is about 2.4 points.

## Limitations & next steps

- **The test season was harder than the past.** Margins in 2021-22 are more spread out, with a std of 15.3 against 13.2 to 14.4 in earlier seasons, and more blowouts. That is why test MAE is 11.2 while validation was 10.2. Weighting recent seasons more would help the model keep up with changes in the league.
- **Home court shrank after COVID.** The model learned its home edge from 2012 to 2018, but 2021-22 only had 1.7 points. On average it favours the home team by 0.9 points too much. A fix is to estimate the home edge from only the last season or two instead of all the training years.
- **No player information.** The biggest moves in a line come from who is playing, like stars resting, injuries and trades. This data has nothing on injuries or lineups. The players table, `games_details.csv`, could be used to build a minutes missing feature.
- **No comparison with real lines.** Without historical betting odds I cannot say how this compares to what books actually post. Adding closing lines would show whether the model ever finds games where the market is wrong.
- **Linear only.** Effects like rest are probably not linear, since 3 days of rest is not three times as good as 1 day. Binning rest into categories is a simple next step that still keeps the model linear.

## Generative AI use disclosure

I used Claude, made by Anthropic, during this assignment. It suggested some dataset ideas and I chose the NBA point spread one. It wrote most of the code for loading the data, the training loops, saving the models and the Gradio dashboard. It also drafted this report from the results, and I went through and edited it. I checked every decision in the pipeline and every number in this report and can explain each one.
