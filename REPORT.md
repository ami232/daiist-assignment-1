# Assignment 1 Report

- **Name**: Alejandra Gómez Vallejo
- **Student ID**: 19766
- **Email**: agomez.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset
*What is it, where did you get it, what does one row represent, how many
rows/columns, and why did you pick it.*

For this first assignment I used matches data from the Spanish First and Second Divisions (LaLiga and LaLiga Hypermotion), downloaded from Football-Data.co.uk (https://football-data.co.uk/spainm.php). This British website publishes historical football results and statistics, and it is built mainly for betting. Each season and division comes as a separate CSV file. I downloaded every season from 2014-15 to 2025-26, which gave 24 files. As explained later, I dropped 2025-26 in the end, because the Pinnacle odds stop partway through that season.

Each row is one football match. The raw files have the expected number of matches in every season: 380 in the First Division (20 teams, 38 matchdays) and 462 in the Second Division (22 teams, 42 matchdays). That makes 10,104 matches in total. The columns are not the same across files. Each file has between 52 and 131 columns, and 182 different columns appear across all 24 files. In general, they include:

- Match information: division, date, kick-off time, home and away team.
- Results: full-time and half-time goals and result (home win / draw / away win).
- Match statistics: shots, shots on target, fouls, corners, yellow and red cards, for each team.
- Betting odds: 1X2 odds, over/under 2.5 goals and Asian handicap odds from several bookmakers (Bet365, Pinnacle, William Hill, etc.), plus market averages and maximums, both pre-match and at closing.

Final size. After cleaning, which is described in the next section, the dataset has 7,855 matches and 29 columns. It covers the First Division from 2014-15 to 2024-25 and the Second Division from 2017-18 to 2024-25. For odds, I kept only Pinnacle, the one bookmaker available in every season for pre-closing and closing of both divisions.

I chose this topic and datasets because I'm doing my thesis on football and, although my thesis is about player signings rather than betting, it also starts in the 2014-15 season and I considered this business case as a good way to get familiar with football data and the Spanish leagues. 


## Business / real-life framing

*The hypothetical scenario this model serves, and what that scenario implies
for how you built the pipeline — target definition, whether a time-based
split was necessary and why (or why not), which metric should drive the
decision threshold and why.*

**Scenario** 
I am a bettor who, before each match in LaLiga and LaLiga
Hypermotion, decides whether to stake 1 € on the home team at Pinnacle's pre-match odds or to stay out. The model does not need to predict the results: instead, it estimates the home-win probability and compare it
with the probability implied by the bookmaker's odds. I bet only when the model gives the home team a better chance than the market does.

**Target** 
The target is binary: `home_win = 1` and `0` for a
draw or an away win. This matches the decision, because a home bet loses in both of the other cases. The classes are reasonably balanced (45.1% home wins), so no resampling was needed. The cost is that the strategy can only back home teams. Since the bet is done before kick-off, every feature must be known by then. Match statistics only enter as averages over each team's previous 5 matches, and Elo is stored before it is updated with the result. Odds are deliberately not used as features: they are the benchmark to beat.

**Time-based split** 
A time-based split was necessary because a bettor can only learn from matches already played, and a random split would let 2024 matches predict 2023 ones (same for the Elo):
- Train: from 2014-15 to 2022-23
- Validation: 2023-24
- Test: 2024-25

The dates do not overlap and the home-win rate is stable across the split. The scaler is fitted on train only, and all tuning is done on validation. The test season is used once, at the end.

**Metrics** 
I use two metrics for two different decisions:
- Log loss to compare the models: What matters is that the probabilities are accurate, because they are then compared with the market's. Accuracy is not useful here because always predicting "no home win" already gets 54.6% right.
- ROI on validation to choose the threshold: I bet on the home team when my probability beats the market's by at least t: `p_model − p_Pinnacle ≥ t`. I chose t = 0.05, because it's the first value from which the ROI stops being clearly negative.


## Data preparation & feature engineering
*What you engineered and why, and any data-quality decisions you made along the way — e.g. "segment X had defective data, so I excluded it and used a population-average default for scope Y at inference time; the impact of that choice is Z."*

**Data preparation**: 
I started ensuring that all files had the right number of rows (380 for First Division; 462 for the Second). Then i looked at the shapes of each. As said before, the columns were not the same across the different files. Since I wanted to have an homogeneous dataset, I looked at the number of seasons I have for that column per division, and the percentage of observations I had in those seasons where I do have that column. I went to the data dictionary (https://football-data.co.uk/notes.txt) and start discarding columns:
- First I quit the Asian handicap betting ods because it was a different kind of bet to the 1x2 we are used to in Spain.
- Then I quit the columns in which there is no information for more than 1/2 of the seasons (1-5).
- I looked at the data available for the different bethouses, and Pinacle was the only one with data for all of the seasons, so I decided to just focus on this one and discard the res (same with the market ones because they are an average of different houses the data is not available for all seasons)
- I renamed the columns using the dictionary mentioned before, to be more clear on what I was working on.
- I quit the ods of over 2.5 goals because I am not gonna use market data to train my model (I would just copy it), it's not my target, and I don't need it to compute my benefit.
- I transformed the Division column from SP1/SP2 to First and Second. I looked at the format and transform the Match Date from object to datetime.
- I quit time of the match because I missed 5 seasons, web scrapping was going to consume me too much time for not a big benefit, I didn't wanna quit such amount of information of my dataset, and it was too much information to inpute with whatever strategy. I also did a chi-square test that gave me p = 0.468, so it ensured I could quit it.

I moved to the NAs and see number per season and division. I had different findings:
- In the Second Division (2018-19 season) I had 21 matches with lot of NAs. After inspection and research I found that all of them were for Reus, and Reus that season was sent out on January 18th for not paying their players. The times coinceeded and I drop these rows.
- Half of the Pinacle quotes were not available from half of the 2025/26 season. That was going to be my test set. Without these data, I can't compute the benefit. Therefore, I decided to quit this season, and use the previous one for test.
- Then I had don't have data for Second Division in the first 3 seasons, so I dropped them. Therefore, this Division data starts from 2017-18. Implications: teams that went from second to first division between 2015 and 2017 start without data from the second division, in the 2017-18 most second division teams start without previous data (but same happens for the first division in the 2014-15 season).
- The NAs regarding market data were not important since they are not in the test set, and in the validation season (2023-24) there are just 2. For training seasons, I am not using that data.

I ensured I had no duplicates, that the teams available made sense, and that the result was aligned with the goals.

I defined the target: predicting if the local team is going to win, see the statistic for the data available (45.1% local victories) and create my target column with 0 and 1. The percentage was similar in both divisions and seasons.

**Feature Engineering**:
I had 2 types of features:
- Some known before th match
- Some known just after/during the match

So the most important thing is that for a match at X day, I can just train with data available before that day. I built 3 types of features:
- How well is the team playing: average of the previous 5 matches (goals, shots, etc). Before taking the rolling mean, the values are shifted by one match (shift(1)) within each team, so a match never uses its own statistics, just earlier matches. To deal with the cold start problem, I discarded the lines that had no 5 previous matches for both teams available - these values were computed per team level and then transformed to match level. The number of matches I lose in validation it's not a significant. I also create the differences columns.
- How well a team is: I did an Elo using the same logic as chess rankings: all teams start with 1350 in Second Division and 1500 in First. Before each match, Elo computes the expected resut based on the differences between each team. After the match, each team wins or losses points based on how surprising the result has been. This summaries in one column, the whole history (avaliable data) of the teams (so winning Barça gives more points that winning Espanyol). This has k= 20 (how much puntuation is moved after each match - standard quantity in football), home advantage of 60 extra points, and 1/3 of regress (At the start of each season, each team moves one-third of the way toward its division's average. Rosters change in the summer. It also determines promotions: a team that is promoted moves closer to the First Division average.). Important: to deal with leakage, I first save the Elo and then update the result, Therefore, the feature of each match just reflects previous matches. If we see the final ranking, it makes sense (Barça, Real Madrid and Atlético de Madrid are first, second, and third, respectively).
- Circumstances (resting dates): limitation: I just have data for LaLiga and there are teams playing European competitions.

In the case of columns based on the 5 previous matches, each feature exists for the home team, the away team and as a difference (home − away). The model only uses the differences, because using all three would be perfectly collinear (diff = home − away).

I group features by families (strength/Elo; form/goals; context/resting dates; volume/shots-corners; discipline/fouls) to see if quiting x group makes the model worst. Before this, I do the split:
- Train: from 2014-15 season to 2022-23
- Validation: 2023-2024
- Test: 2024-2025

The split is not random because I cannot learn from the future to predict the past. 

I scale the features (compute with training data and then apply also in validation and test). The reason of the scaling is easy: everything has to be in the same unit and for instance Elo and yellow cards are in very different units and the L2 penalty would punish coefficients differently just because of their scale, and gradient descent in PyTorch converges badly when features have very different ranges.


## Modeling: three implementations, one model
*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

I started defining 2 baselines (what I need to surpass):
- Baseline 1= % of local victories in train (always the same)
- Baseline 2: market probability (Pinnaple)

I used these 4 metrics:
- Log loss and brier: measure how well probabilities are (not just if you were right about the result or not)
- AUC: measure capability of ordering the matches
- Accuracy: % of correct answers with threshold 0.5.

**Logistic Regression with sckit learn**:
I use a logistic regression because my outcome is a binary variable (1 being locals winning). I tried C = 0.001, 0.01, 0.1, 1 and 10. Validation log loss was the same (0.6430) for C between 0.1 and 10, and only got worse with C = 0.001. Since they tied, I kept the most regularized one, C = 0.1. I train the model, looked at the coefficients and see that:
- diff_elo is the most important one: it makes sense because Elo already takes into account past results weighted by the opponent's strength
- I do ablation to see the value loss by quitting the different group features described in the previous section, and decide to remove context and discipline because they did not bring enough value. 

| Validation log loss         | Value  | Change   |
|-----------------------------|-------:|---------:|
| All features                | 0.6432 |          |
| Elo only                    | 0.6464 | +0.0032  |
| Without strength (Elo)      | 0.6604 | +0.0172  |
| Without form                | 0.6448 | +0.0016  |
| Without volume              | 0.6448 | +0.0016  |
| Without discipline          | 0.6435 | +0.0003  |
| Without context (rest days) | 0.6427 | −0.0005  |

I choose C again but eliminating the features we mentioned (I stay with the same), do the scaling, and perform the logistic regression. The trained model is saved to artifacts/, and the Gradio app loads it from there without retraining.

**Manual Pytorch**:
I replicated the teacher's notebook adding an L2 penalization to make sure the 3 methods optimice the same function than sklearn, and therefore, have similar coefficients. Based on the graphic, I choose the learning rate to be 0.5. The trained model was sent to Gradio.

**Standard Pytorch**:
L2 is only applied to the weights, not to the bias. The learning rate chosen is 0.01 (is different than the one of manual because Adam automatically adjusts the step size for each parameter, so it usually uses smaller lr values). The trained model was sent to Gradio.

**Business rule**:
I bet to the local team if p_model − p_Pinnacle ≥ t. I choose t to be 0.05. With this, the results are the following:

Results on the test set (2024-25)
| Model              | Log loss | Brier  | AUC   | Accuracy |
|--------------------|---------:|-------:|------:|---------:|
| Constant           | 0.6891   | 0.2480 | –     | 0.545    |
| Pinnacle (market)  | 0.6275   | 0.2190 | 0.691 | 0.637    |
| sklearn            | 0.6428   | 0.2256 | 0.668 | 0.614    |
| PyTorch manual     | 0.6428   | 0.2256 | 0.668 | 0.614    |
| PyTorch standard   | 0.6428   | 0.2256 | 0.668 | 0.614    |



| Model             | Bets | Hit rate | Profit     | ROI    |
|-------------------|-----:|---------:|-----------:|-------:|
| sklearn           | 208  | 0.385    | −34.92 €   | −0.168 |
| PyTorch manual    | 208  | 0.385    | −34.92 €   | −0.168 |
| PyTorch standard  | 208  | 0.385    | −34.92 €   | −0.168 |

The three models agree: we have the same coefficients to 4 decimal places, identical validation and test metrics,
and exactly the same betting decisions. This was not the case at first: with its default tolerance (tol = 1e-4), sklearn differed from both PyTorch versions by up to 0.001 per coefficient, while the two PyTorch versions already matched each other. Refitting sklearn with tol = 1e-10 closed the gap to 4 decimals, so the difference came from sklearn’s stopping criterion, not from the model. 

Think that the models were comparable because:
- All three models use the same 10 final features,
the same `StandardScaler` (fitted on train only) and the same train/validation
split. 
- I also made sure they minimise the same objective. 
- tol = 1e-4

Results make sense because i achieved the expected behavior:
- I surpass the baseline 1 
- I did not achieve to surpass the market

## Dashboard
uv run python main.py app launches a Gradio app that only loads the saved scaler and models from artifacts/; nothing is retrained. It has four tabs:

- Model comparison: metrics for the three models and both baselines, a predicted-vs-actual (calibration) plot, and a plot showing that the three models give the same probability for every match.
- Distributions: the distribution of each feature by outcome, and the home-win rate by season and division.
- Betting threshold: a slider for t that updates the confusion matrix (bet / no bet vs. home win / no home win), the profit in €, the ROI, and the cumulative profit through the season.
- Match explorer: for a single test match, the three models’ probability next to Pinnacle’s.

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

**Limitations**:
- The models don't surpass the market. Moreover, in matches where it detects an advantage, it is overly optimistic. With t = 0.05, during validation, it predicted an average of 52% home wins, Pinnacle predicted 43%, and the actual result was 45.5%. In testing, it lost 16.8%. The underlying reason is that it uses only publicly available statistics from previous matches, whereas the market takes into account lineups, injuries, rotations, and the money wagered. Therefore, I should add information available before the match that today I don't have such as line-ups, injuries, etc.
- All decisions depend on a single validation season (832 matches and about 200 bets), so the standard error of the ROI is around ±7.5 points. This is why the ROI went from +0.2% on validation to −16.8% on test. To deal with this I could use walk-forward validation (train up to season N, evaluate on N+1, and repeat over several seasons).
- The model can only bet on the home team. I could train a second binary model ("away win"), or fit a multinomial regression (I did not do it because of the assignment instructions).
- Second Division teams start 2017-18 with no history, so teams promoted in those years reach the First Division with no past record, and 228 matches were excluded. I could have used those seasons for Elo only. 
- Odds are missing for half of 2025-26, so the most recent
season could not be used as the test set. I could have used the market average or Bet365 odds for that season, after checking that they give similar results to Pinnacle.
- Elo was not tuned and it is the most important feature: K, the home advantage and the mean-reversion factor are standard values that I did not test. Therefore, the next step would be training these parameters.
- Rest days only count only league matches. In ablation we saw that this feature because it does not see cup or European games. Next step would be adding fixtures from those competitions.
- The market probability is computed in a simplified way: Removing the margin proportionally ignores the favourite–longshot bias. Next step would be compare against Shin's method.
- The betting strategy is basic: It always stakes 1 € at the pre-closing odds. Next step: measure closing line value (if the odds decrease after the bet).


## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*

I originally did my work in Google Colab, and I used Claude to reorganize that code into this repository in the
format required (train.ipynb and app.ipynb running through main.py) and made it do the dashboard. I used it for most of the code: sometimes I told it what I wanted to do and it wrote the code, and sometimes it proposed the next step. I ran every step myself, checked the outputs, and asked it to explain the code line by line when I did not understand it.

I also used it to explain concepts I needed for my decisions (closing vs pre-closing odds, Elo, regularization across the three methods, ablation). For the main decisions it often laid out options with a recommendation; some decisions I proposed myself (e.g. dropping columns present in fewer than half of the seasons, keeping a single bookmaker, dropping 2025-26 and Segunda 2014-17), and for the others I chose after checking the evidence in my own data.

Finally, I used it to help me translate my notes from my mostly Spanish Colab notebook into English for this report.
