# Assignment 1 Report

- **Name**: Jaime Fernandez
- **Student ID**: 18632
- **Email**: jfernandezr.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset
For this assignment, I used NFL game results from the `nflverse/nfldata` project (`games.csv`,
https://github.com/nflverse/nfldata). I saved a frozen copy of the dataset (1999–2025, only games already played) to make the pipeline reproducible.

Each row represents one NFL game. The original file had 7,276 games and 39 columns with information such as teams, week, rest days, roof/surface, weather, quarterbacks, betting lines, and final scores.  
I kept only regular‑season games and removed ties, ending up with 6,952 games.

(The frozen copy I created contains 7,276 games. The original nflverse file includes more rows, including future scheduled games.)

I chose this dataset because the outcome is clearly binary (home win or not), the data has a natural time order, and most of the useful signal (recent form, strength, rest) must be engineered from past games.

## Business / real-life framing

**Question:** Will the home team win this regular‑season game?

**Decision:** A coach uses the prediction to decide whether to rest starters.  
- If the model is confident the home team will win, starters rest.  
- Otherwise, starters play.

There are two types of errors with different costs:

| Error | What happens | Cost |
|------|--------------|------|
| **False positive** | Model says “win”, starters rest, team loses | **3** |
| **False negative** | Model says “no win”, starters play, team would have won anyway | **1** |

From these costs, the optimal decision threshold is **0.75**, not 0.5.

This comes from comparing expected costs: saying “win” costs (1 − p)·3 and saying “no win” costs p·1. “Win” is better when (1 − p)·3 < p, which simplifies to p > 3/(3 + 1) = 0.75.

### Pipeline implications

- **Target:** `home_win = 1` if the home team scores more points.  
- **Time‑based split:**  
  - Train: 1999–2021 (5,868 games)  
  - Validation: 2022 (269 games)  
  - Test: 2023–2025 (815 games)  
- **Leakage removed:** final scores, results, totals, overtime, and all betting lines.  
- **Baseline:** always predict “home win” → test accuracy **54.2%**.  
- **Main metric:** total business cost at threshold 0.75.


## Data preparation & feature engineering

Key rule: **a game’s features can only use information from earlier games**.

Since the dataset doesn’t include team strength, I built it using point margin (points scored minus points allowed) from previous games.

Final features:

| Feature | What it measures | Why |
|---------|------------------|-----|
| **pd5_diff** | Home minus away average margin over last 5 games | Recent form |
| **pd16_diff** | Same over last 16 games | More stable strength |
| **rest_diff** | Home rest days minus away rest days | More rest → fresher |
| **div_game** | 1 if both teams are in the same division | Rivalry games often closer |

Correlations with winning (train):  
- pd5_diff: **0.28**  
- pd16_diff: **0.30**  
- rest_diff: **0.02**  
- div_game: **−0.03**

To prevent leakage, each team’s past margins are computed using a shift(1), which removes the current game from the rolling window. The script check_no_future_leakage also scrambles all scores from the first test day onward and verifies that features for games on that day do not change, while features for later games do.

**Data-quality decisions**

- First games of each team have no past margin - filled with 0.  
- Weather and surface had many missing values and didn’t improve validation performance → removed.  
- Features were standardized using a scaler fitted only on the training set.

Temperature and wind have many missing values, mostly because roofed stadiums do not record them. Surface does not have many missing values, but it has nine inconsistent spellings (e.g., “grass” vs “grass ”)

I tested other windows and an Elo‑style rating. Results were similar, but the 5+16 combination was the simplest and performed best on the validation season.

## Modeling: three implementations, one model

I used **logistic regression**, since the target is binary and the output is a probability that can be compared to the 0.75 threshold.

I implemented the same model in three ways:

1. **scikit‑learn LogisticRegression**  
2. **Manual PyTorch loop**  
3. **PyTorch nn.Module + optim.SGD**

All three used:
- The same objective (binary cross‑entropy + L2 penalty)  
- The same optimizer (full‑batch gradient descent, 3,000 epochs)  

Only the two PyTorch implementations use full‑batch gradient descent with 3,000 epochs and zero initialization. scikit‑learn uses the lbfgs solver, which is a quasi‑Newton method, not gradient descent, but it optimizes the same objective on the same standardized inputs.

**Hyperparameters (the only two I tuned, on the 2022 validation season).**
- Regularization: **C = 1.0**  
- Learning rate: **0.1**

The validation loss was essentially flat for all C ≥ 1 and for all tested learning rates (0.1, 0.5, 1, 2). Because tuning made almost no difference, I chose the most cautious values: C = 1.0 and learning rate = 0.1.

**Results on the test set (2023-2025, 815 games):**

| | Accuracy @0.5 | Log loss | AUC | Cost @0.75 | False pos. | False neg. | Games rested |
|---|---|---|---|---|---|---|---|
| scikit-learn | 0.6319 | 0.6386 | 0.6896 | 418 | 19 | 361 | 100 |
| Manual PyTorch | 0.6319 | 0.6386 | 0.6896 | 418 | 19 | 361 | 100 |
| nn.Module + optim | 0.6319 | 0.6386 | 0.6896 | 418 | 19 | 361 | 100 |
| Baseline: always predict home win | 0.5423 | 0.6905 (train win rate) | - | 1,119 (always rest) | - | - | 815 |
| Baseline: never rest starters | - | - | - | 442 | 0 | 442 | 0 |

All three implementations agree completely. They learn the same coefficients to four or five decimals, and the largest difference in predicted probability between any two methods on the test set is about 1.5×10⁻⁶. Logistic regression with an L2 penalty is a convex problem, so there is only one optimum. By matching the objective using the conversion λ = 1/(C·n), using the same standardized inputs, and starting the PyTorch models from the same point, all three converge to the same solution.

## Interpretation
- The model clearly beats the baseline: **63.2% vs 54.2% accuracy**.  
- Coefficients match football intuition:  
  - pd16_diff (long‑term strength) is most important.  
  - pd5_diff also matters.  
  - rest_diff and div_game contribute very little.  
- Business impact is **small but positive**:  
  - Model cost: **418**  
  - “Never rest” cost: **442**  
  → about **5% improvement**.

## Limitations & next steps

1. **Small and not fully reliable business gain.**  
   Bootstrap interval: −9 to +54. More seasons needed.

2. **Costs (3 and 1) are invented.**  
   Real thresholds depend on actual team‑provided costs.

3. **Model overestimates home advantage recently.**  
   Could recalibrate the intercept using recent seasons.

4. **Features ignore opponent strength and quarterback changes.**  
   Next step: opponent‑adjusted margins or Elo, and QB change indicators.

5. **Only one validation season.**  
   Better: rolling‑origin cross‑validation across multiple seasons.

6. **Linear model cannot capture interactions.**  
   Could test interactions or tree‑based models.

7. **Simplified decision scenario.**  
   Real rest decisions depend on standings, injuries, and are player‑specific.

8. **Test set was not perfectly untouched.**  
   I printed some test results during exploration, though final choices used validation.


## Generative AI use disclosure
I used Claude (Anthropic) as an assistant only for allowed purposes in this assignment: to compare candidate datasets and help me decide on the NFL games data, to draft boilerplate code in train.py and app.py, to assist with GitHub and Visual Studio steps, and to suggest wording and formatting improvements for this report. All substantive decisions have been made on my own. I ran all code locally in Visual Studio Code and verified the outputs myself.
