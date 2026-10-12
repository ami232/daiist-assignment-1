# Assignment 1 Report

- **Name**: Leopoldo Sirolli
- **Student ID**: 19170
- **Email**: lsirolli.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A]

## Dataset

I got the data from tennis-data.co.uk, and collected the excel files of all games in the ATP tour from years 2018 to 2026.
Each line represents one game, with tournament, location, date, series, court and so on, then the results of each set of the game and the betting quotes for both players from bet365 and Pinnacle.

Raw data was composed of 19,685 rows and 43 columns
After cleaning it was composed of 19,017 rows and 50 rows after A/B reshape and dropping of uncompleted matches.

I chose this dataset because I enjoy watching tennis and although never bet, I wanted to see how effectively I could not only predict a game result, but theoretically make money by betting on multiple games.

## Business / real-life framing

What I envision for this project is a user that aims to maximise porfit from a number of individaul bets on tennis ATP matches. I don't expect to beat everyday bookmakers on every match but rather find a few matches where the book makers price could be off and bet on those.

The time based split was necessary because I used 2018 dataset to warm up my model and the new features such as Elo I constructed, then I used 2019-2023 data to train, 2024 to valdiate and 2025 to test as the 2026 data was incomplete as the season is still under way.

The threshold will be expected value being --> model probability x odds - 1 and the decision metric will be ROI as accuracy will ignore price.


## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way, e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*


Non completed matches were removed due to walkovers, reitrements, awarded games and so on (668) and mising or invalid odds (93) for a total of 761 removals, approx. 3.9% of the raw data. 
The first 668 were removed from the start as they served no purpose to the investigation nor results, the other 93 were removed after the feature loop as they helped updating players info.

Missing rankings were filled with 2000, essentially unranked players in the ATP.
2026 season excluded fue to incomplete season and 2018 used only for warm up as mentioned previously.

Used only Bet365 quotes because only boom,aker present in all files for consistency, dropped all other boomakers. And I chose not to use any results  from the Wsets or Lsets as this would be target leakage as this is information created during the game and the model would predict a result straight off the score.

Engineered feauters:

- elo_diff 
    - Overall Elo rating (like chess), start at 1500 and K=32
    - K is how much a players Elo moves after each game --> new Elo = Elo + K * (actual result -          expeted result), actual result being 1 is won
    - correlation with y = +0.342
    
- elo_surf_diff
    - Elo adjusted for surface
    - correlation with y = +0.318

- form_dff
    - win rate over last 10 matches (0.5 if no history)
    - correlation with y = +0.225
    
- surf_wr_diff
    - carreer win rate on said surface smoothed (wins+1)/(played+2)
    - correlation with y = +0.255
    
- fatigue_diff
    - matches played in last 14 days
    - correlation with y = +0.108

- exp_diff
    - log(1+matches played in dataset)
    - correlation with y = +0.211
    
- log_rank_diff
    - log(A rank) minus log(B rank)
    - correlation with y = -0.341


Fatigue_diff I figured to be positive as the more recent matches I saw as more likely to win  since tennis is played in tournaments and one loss means direct elimination, therefore two people that meet each other in a tournament played the same numebr of games in that tournament, but if one got eliminated earlier in the previous tournament he played less matches and could be and could be a valuable indication to our esitmate

Elo_diff and log_rank_diff are obivously very similar inversely, a high elo points to a low (better) ranking


StandardScaler fitted on train (2019-2023) only, applied to validation and test. Raw scales before scaling: elo_diff std 159, form_diff std 0.24.


## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

| Model (test season 2025) | Accuracy | Log loss |
|---|---|---|
| Naive: higher-ranked player wins | 0.6478 | n/a |
| Bookmaker (Bet365, margin removed) | 0.6757 | 0.5929 |
| Logistic regression: scikit-learn | 0.6534 | 0.6176 |
| Logistic regression: manual PyTorch | 0.6534 | 0.6176 |
| Logistic regression: nn.Module + torch.optim | 0.6534 | 0.6176 |

| Betting (threshold 0, chosen on validation) | Bets | ROI | Win rate | Avg odds | Back every favourite |
|---|---|---|---|---|---|
| Validation 2024 | 2,015 | -14.4%


They agree, absolutely, on the test made on the 2025 season of the ATP we had identical accuracy (0.6534) and log loss (0.6176) for all three. After all they are a weighted sum of the same 7 scaled features through sigmoid trained (once again) on the same rows.
Furthermore the obective was the same, sklearn's C was translated into the PyTorch pensalty (lambda = 1/(2*c*n)). Without this they would be optimizing slightly different thigngs.

The logistic regression loss has a single minimum meaning its convex and any method that gets close enough returns essentiually the same answer.

Conclusion: differences are too small to change any prediction or decision meaningfully, they can all be treated equally in the end.