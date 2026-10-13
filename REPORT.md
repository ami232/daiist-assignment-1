# Assignment 1 Report

- **Name**: Moritz Hanbuch
- **Student ID**: 19140
- **Email**: mgobbels.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

I used a dataset containing football match results and betting odds from football-data.co.uk, which compiles public results and odds (not official league data). The data is split across 32 CSV files covering 4 leagues (Premier League, La Liga, Bundesliga, Serie A) across 8 seasons (2018/19 to 2025/26). Each row represents one match. In total there are 11,568 rows and about 5 MB of data, which is under the guideline of about 20k rows or 20 MB.

Each raw CSV file contained between 61 and 132 columns, but only 33 were common across all 32 files, and my pipeline uses 15 of those: date, *teams*, *full-time goals* and result, *shots*, *shots on target*, *yellow and red cards*, and Bet365 home odds. The columns in italics exist for both the home and away team (I included a full list of the columns at the bottom of this section).

All the files were latin-1 encoded, and only 3 of the Bet365 home odds were missing. I excluded these from the profit calculation, but since they are in the training seasons, they never affect the reported results. I also ran a validation check confirming that every team played the right number of fixtures, ensuring complete seasons and checking for possible name inconsistencies. But this found no problems. 

Overall, I chose this dataset because it fitted the requirements well, and because I had not yet worked with a dataset containing betting odds and was curious whether I could find something valuable. We normally use basic datasets (movies, etc.), but this one seemed different and also matched my interest in football and sports. It also allowed a lot of creativity for feature engineering, since match stats are only available during and after the match, not before when the bettor would place the bet. Hence I had to build pre-match features (explained in section 3).

List of all columns: 
Date, HomeTeam, AwayTeam, FTHG, FTAG (full-time goals for home and away), FTR (full-time result), HS, AS (shots for home and away), HST, AST (shots on target for home and away), HY, AY (yellow cards for home and away), HR, AR (red cards for home and away), and B365H (Bet365 odds for a home win)

## Business / real-life framing

I chose the following scenario: a bettor stakes a flat one unit (for simplicity) on the home team whenever the model’s probability of a home win clears a certain threshold. If he wins, he is paid the odds (Bet365 odds column) minus 1, and if he loses it costs him 1.

For this I defined my target variable as home win (FTR == “H”), a binary variable saying whether the home team won or not. On the other side, draws or away wins both count as 0 since the bettor loses his stake in both cases. 

In this case a time-based split was necessary, because the bettor always predicts the future. If I used a random split, future match results and performances would inform earlier matches and therefore leak data. In the end I decided to use the following splits: warm-up 2018/19 (to handle cold start of features), train 2019/20 to 2023/24, validation 2024/25, and test 2025/26, touched once.

The metric that actually drives the threshold is validation profit with a minimum of 50 bets (to avoid picking and relying on a few lucky bets), and not accuracy, because the bettor cares about units won. In general, a win at short odds pays very little, which means accuracy and profit can disagree. In the end one shared threshold is picked from the validation set using the scikit-learn model and then applied to all three models on the test set.

Another condition I decided on is that the model never sees the odds, meaning they are only used to compute the profit. Although this is a trade-off, I decided on it because it frames the scenario better: a bettor only makes real money when the model is right and the bookmaker’s price is wrong. Without odds as an input, the model can find the likely winner (my goal) and does not focus on finding mispriced odds. This also made me focus more on the feature engineering of finding the winner, which is one of the largest parts of the assignment. 

Finally, I decided to use two different baselines. One serves as an accuracy bar and predicts the majority class (“not a home win”, from the train base rate). The second serves as the profit bar and bets on every home team. Therefore, we have two different baselines that measure two distinct things.


## Data preparation & feature engineering

Note: I never chose features or constants by looking at the data, only by reasoning. I used descriptive checks on the data as sanity checks, but never as the selection method.

*Feature Engineering:*

I engineered a total of 26 features, all computed only from earlier matches (earlier dates). Examples:
- Form: points per game over the last 10 matches, to measure the team’s current momentum, plus venue-specific points per game over the last 5 (home matches for the home team, away matches for the away team). I chose the window sizes (10, 5 and 38) purely based on reasoning. For example 10, because 10 matches are around a quarter of a season, which seemed to be a good estimate of how often a team’s performance might change and should be reassessed. And the 5-match venue window keeps the venue-specific form recent.
- Attack and defence: goals scored and conceded per game over the last 10 games.
- Shot quality: shots on target for minus against, last 10 games.
- Shot volume: all shots for minus against, last 10 games. I kept both on purpose, since shots on target say more about the quality of the chances and total shots about how many a team creates. L2 regularization in the model handles the correlation between them.
- Long-run strength: goal difference per game over the last 38 matches (around one season) to take longer time periods into account too.
- Discipline: cards per game over 10 matches (yellow = 1, red = 2).
- Fatigue: days since the last match (capped at 14). However, this has the limitation that the dataset only has domestic matches, so fatigue is understated for teams playing in the Champions League or Europa League. I still decided to keep it, because the variable should still capture short rest and fixture congestion in domestic play, and the limitation is noted in Limitations.
- League one-hot: home advantage differs across leagues, so I added a one-hot encoding of the league to let the model account for it.
- Elo rating: I built an Elo rating for each team, with one rating pool per league. I used Elo to summarise a team’s whole history and the strength of its opponents in one number which the rolling averages don’t capture. K = 20 (how much one result moves a rating), home advantage = 65 (rating points added to the home team). I also added an automatic pull-back at the end of each season (adjust for summer team changes), where one third of each rating is moved toward the league mean. Again, I used all of these constants by reasoning, not tuning.
- Cold-start flags: one flag each for the home and away team, set to 1 when any of that team’s features fell back to a default because of too little history.

*Data Quality Decisions:* 

Cold start problem: 
Promoted teams have no earlier data, or at least I did not have that data. To handle this, I implemented a default from an expanding percentile of earlier matches in the same league, using only matches before that date. I used different quartiles for different features: bottom quartile for all the “higher is better” features (form, attack, shot quality, shot volume, long-run strength), top quartile for goals conceded, and median for cards and rest. I also implemented that a new team’s Elo starts at the bottom quartile of last season’s ratings. I used this instead of the league average, since the average would not capture the performance as accurately (promoted teams are usually weaker). Once a team has a minimum number of earlier matches (5 for the 10-match features, 10 for the 38-match feature, 3 for venue form), I start using the team’s moving average again. This affected 323 out of 7,230 training rows (4.5%), and 40 rows each in validation and test. The only limitation is that relegated teams that return carry stale form and Elo, not a true blank slate.

COVID seasons: 
Initially I wondered whether I needed to adjust for COVID seasons. I thought that with fewer or no fans, home advantage would be smaller and team performance might be different. However, in a quick look at the training data I only found a slight dip (home-win rate of 0.405 in 2020/21 vs 0.434 to 0.465 in the other training seasons), which was not different from normal season-to-season variation, so I kept all seasons and did not flag them.

Fouls and cards: 
I suspected that fouls and cards depend heavily on the referee, and they might not even vary much across teams. However, red cards directly impact matches and are therefore important to keep. At the same time, I decided to drop fouls, because they add no mechanism that cards don’t already carry. As a last check, in a quick look at the training data I compared the difference in cards between top and bottom teams (sanity check only - I did not use it to select the feature).

Leakage test: 
I implemented a leakage test that, for sampled matches in every split, deletes all later rows, recomputes the features, and checks that they are identical. I also added a check on one test match: removing the home team’s last earlier match must change its features, which shows the test can detect a change, so a passing test means something. The test recomputed 30 sampled matches (10 per split) and all of them matched, so for those matches no feature used information from the future.

Scaler: 
I fitted the StandardScaler on the training data only, because the gradient-based models need features on similar scales, and fitting on train only stops validation and test information from leaking into the mean and standard deviation.


## Modeling: three implementations, one model

I used logistic regression with L2 regularization, because the target is binary. I trained the model the three ways the task asked for: scikit-learn, a manual PyTorch loop (my own gradient update, no torch.optim), and nn.Linear with torch.optim.SGD. I set the PyTorch penalty so that all three solve the same problem. I only tuned C (the regularization strength) and the learning rate (the gradient-descent step size), using validation log loss (chosen C = 0.01). C was tuned with scikit-learn and reused for both PyTorch versions. The learning rates chosen were 1.0 (manual loop) and 0.3 (nn version). The nn version uses SGD with momentum 0.9, while the manual loop uses plain gradient descent. This is the one real difference between the versions.

Results on the test season 2025/26, same split for all rows

| Test 2025/26 | Bets | Profit | ROI | Acc @ 0.5 | Log loss |
|---|---:|---:|---:|---:|---:|
| scikit-learn | 63 | +2.09 | +3.3% | 0.6521 | 0.6127 |
| Manual PyTorch | 63 | +2.09 | +3.3% | 0.6521 | 0.6127 |
| nn.Module + optim | 63 | +2.09 | +3.3% | 0.6521 | 0.6127 |
| Majority class | 0 | 0.00 | n/a | 0.5643 | 0.6849 |
| Bet every home team | 1,446 | -129.62 | -9.0% | 0.4357 | n/a |

And as we can see, all three implementations agree. The largest probability difference on test is about 1.6e-7, and none of them makes a different bet. This is also what I expected, because logistic regression with L2 has one best solution, so any correct implementation ends up there. A disagreement would have meant a bug or a run that had not converged. This actually happened in an early version. My selection rule picked a run that had not finished converging, which created a small gap to scikit-learn. I fixed it so that only runs that fully converged are considered. In the end, all learning rates that converged gave the same validation loss, so the learning rate only changes how fast the model gets there. (Manual lr = 0.01 hit the 30,000 iteration cap so I excluded it as not converged.)

Overall, the model has real predictive signal and it makes fewer mistakes than just guessing “not a home win” every time. Its log loss is lower (0.6127 vs 0.6849) and it is right more often at the 0.5 cut-off (65% vs 56%). This means the features tell the model something real about who wins.
However, in my business scenario predicting well is not the same as making money. At the 0.770 threshold the model placed 63 bets and made +2.09 units. That is a small sample, and a few lucky or unlucky results can shift the total by several units. The 95% bootstrap interval for the profit (re-drawing the test matches many times) is -4.87 to +8.55, which includes zero, so I cannot rule out that my +2.09 is just luck and as a result I would not say the model has a real betting advantage. 

But it also does clearly beat the strategy of betting on every home team, which lost 129.62 units. This is the case because home teams only win about 43% of the time and the bookmaker’s margin eats the rest, while my model avoids most of those bad bets by only betting when it is confident (clears the threshold).


## Dashboard

I added this section to describe the dashboard, since it is a required part of the assignment.

The Gradio dashboard has four tabs: headline results, model comparison, threshold explorer and distributions. It loads only the saved artifacts and never retrains. At startup it also re-runs the saved models and cross-checks their predictions against the saved ones, and fails if they differ.


## Limitations & next steps

- Only 6 of 99 thresholds I could choose from were profitable on validation, and 0.770 was the best of them (the thresholds right next to it earned +1.97 at 0.765 and +1.54 at 0.775). So my threshold is not very stable, and I would not trust the +2.09 on test too much. As a next step I would test several thresholds across more seasons.
- The odds are never an input, so my model finds likely winners, not mispriced odds (this was also my design decision). Overall the model won 54 of 63 bets, but the average odds were only 1.21, so each win paid little. As a next step I would only bet when my model’s probability is clearly higher than the one implied by the odds (1 divided by the odds). But this would also change my whole framing, so it is a trade-off at the same time.
- I only had one test season, which included just 63 bets and an interval that includes zero. I would like to try this with more seasons of data and testing it on several seasons in a row, or also add more leagues.
- My final model never learned from the 2024/25 season, because I only used it to pick C and the threshold. Next, I would probably try to refit it on train plus validation once C and the threshold are fixed.
- Right now my model only backs the home team, so it cannot tell the bettor when to back a draw or an away win. As a next step I would build a complete model for all three outcomes, also using the draw and away odds.
- As mentioned in the feature engineering section the data has only domestic matches, so fatigue is understated for teams playing in Europe. So if I actually had European fixture data, I could count the matches in the last 14 days instead of the days since the last match.
- My cold start solution was good for now but it has the problem I already mentioned - a relegated team that returns carries old form and Elo, not fresh information about the team as it is now. As a next step I would give less weight to the old history of teams that dropped out of the league and came back
- Home advantage differs by league, but the league one-hot only shifts the baseline. So the next step could be one model per league.
- I set the window sizes and Elo constants by reasoning, so the 10-match window could be too short or too long. As a next step I would run hyperparameter tuning on them (e.g. windows of 8, 10 and 15) on validation.


## Generative AI use disclosure

- Coding: I used Claude Code for all the code, as the task allows - loading the data, the feature engineering, the training pipeline (train.py), saving the models, and the Gradio dashboard (app.py), including the plotting and the explanatory captions, which I reviewed again afterwards. The code implements decisions I made.
- Discussing features and design: I used Claude to talk through feature ideas and trade-offs, but every decision came from me, for example the bettor scenario, one shared threshold, the league one-hot, Elo, the cold-start defaults, and not choosing any feature or constant by looking at the data.
- Understanding and review: I used it to explain concepts to me and to check my code against the task description.
- The report: I used it to help me structure my thoughts, suggest improvements, and fix grammar and wording in the end. I reviewed everything again and only kept points I understand and can explain.
