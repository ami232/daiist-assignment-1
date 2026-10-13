# Assignment 1 Report

- **Name**: Matias Viguera Barrientos
- **Student ID**: 18542
- **Email**: mviguera.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A]

## Dataset

This data was found in Open Powerlifting (openpowerlifting.org), which is a public data archive with results of competitions from powerlifting. The dataset until this day (October 2026) counts with 4 million results from federations around the world, around 827MB, given my current location, I've decided to work with the Spanish federation (AEP: Asociacion Española de Powerlifting), and chose the most common competitions in this area, squat, bench press and deadlift. For the extraction we have prepare_data.py, which gave us data/aep_raw_sbd.csv, dataset found in the repository.

Now lets go to the details and what information specifically it has this data set. One row represents one lifter in one competition, and it includes sex, age, bodyweight, weight class, the nine attempts (negative value means failed attempt), best lift per discipline, total, the place, scoring points, the meet's name and date. There are 14,102 rows and 30 columns that covers 5,899 lifters and competitions go from February 2015 to June 2026. If there is no total, that means the lifter was disqualified (usually they wouldn't be able to lift in the three attempts).

For the final purpose of this model, I've decided to live the lifter names as anonymous IDs. This was done because the results need to only be connected with who did it, and that other results were connected to that same lifter, adding this way privacy. 

Why this dataset. Originally I was doing another dataset that was related to hypertrophy in the gym and how it could predict the muscle growth and lifting more, getting to lift more, show more results. However, the dataset chosen was really messy, missing a lot of values and was in general not able to be used with the purpose I wanted. Thankfully, this dataset was found and adapted the model to this one. Following the gym perspective and with powerlifting that is something I am very interested. I do go a lot the gym, and used to be really interested in powerlifting competitions. Regarding why the dataset, from a more construction perspective, this one was perfect, its real, openly licensed data, small enough to commit and retrain in minutes and with good feature engineering from each lifter's competition history.

## Business / real-life framing

The main user of this model, should be a coach/self-coached lifter that wants to plan the competitive season. When using this app they are trying to answer the total that the lifter might achieve in the next competition. Supporting specific decision-making situations like "Is the lifter ready for a national championship?" or "Is competitive season a real objective, or should it focus in regional first". The target is the next competition's total in kilograms, a continuous value, so this is a regression problem and here we find a linear regression.



The prediction should be done some weeks before the competition, so for te information that is accessible is just the one available the weeks before the competition, includes previous competitions + what is know for registration (sex, age, etc.). For the values extracted during the competition day, they will be excluded, this include the nine attempts, best lifts, total, place and points scores. I also decided to not include the bodyweight (the actual one), because it is measured during the competition day, and replaced it by the weight class. There is a little issue when doing this, accuracy might look compromised, however, it means the model will only use data that it counts with.



In order for the model to work, there must be a previous total, so anyone that is doing it for the first time, won't be able to work with this model. Training was done until December 2024, validation was on January to June 2025, and test was July 2025 to June 2026. 



We have the main metrics now, Mean Absolute Error (MAE) in kilograms, this is used for coaches to understand it directly ("on average, off by X kg"). R2 (R-squared) is being used, but can be a bit misleading, totals range from 110 to 950kn, so even the naive baseline can score R2 = 0.956 while being off by 22.3kg on average. 



For this model, overpredicting is the biggest mistake that can happen, a lifter who just came from a long place, paid the entry fees and was not able to reach the weight that was told, can lose even more motivation, affecting their confidence in the app and themselves. I have established a safety margin in the dashboard, this lowers every prediction by a fixed number of kgs. My personal recommendation is 10kg, this reduces to 31% the overprediction and large overshoots go down to 5.6%, the average error rises only to 17.9kg (from 16.2kg). Of course a number such 20 kg can be even safer, however 20 is already a crazy number and is an escalation quite big compared to 10 kg.



## Data preparation & feature engineering

There are a couple of rows I decided to remove:
Doping Disqualifications (DD) and no-shows (NS). No value and the results were void, we are not using them in the lifter's history. 
One exact duplicate.
12 lifter IDs of two different people who were merged under the same ID. 

Leaving this with 14,004 rows from 5,887 lifters. 

Disqualified is the 3% of rows, without counting total, they are not targets and are skipped in the computations, so when we refer to previous total, it means the valid total. However, we are still using previous number of disqualifications as features.

Features used are per lifter, from earlier competitions only, for this we used pandas groupby and shift(1).
Anchor: prev_total (last valid total) and gap_to_best (best previous total minus last total). 
Progression: trend (change between the last two totals) and has_trend (0 when only the previous meet exists).
Experience: n_prev_valid, years_competing and n_prev_dq.
Timing: days_since_last (more time equals more training, however there is still more uncertainty).
Lifter: Age and male. The target distribution it shows to be model given that women's total differs heavily from men's.
Registration Information: class_limit and is_plus_class (weight class), class_vs_prev_bw (class limit minus the bodyweight at the previous meet, going up is more weight, therefore more strength) and is_national (national championship).
prev_misses: if they have failed attempts in previous meets. Many misses shows a total understated of the lifter's real ability. 

I decided to use gap_to_best because the best total usually showed to be pretty near the prev_total, so it could show a near duplication, giving us a gradient descent converge very slowly. 

Some ideas I tested, were junior lifter flag (under 24), it didn't improve validation error because Age was already getting it.

| Features | Validation MAE (kg) |
|---|---|
| Naive baseline: same as last meet | 29.9 |
| Previous total only (learned) | 23.2 |
| + history shape | 22.5 |
| + experience & timing | 19.6 |
| + lifter (age, sex) | 18.7 |
| + registration (weight class, national) | 17.6 |
| + previous missed attempts | 17.5 |

Leakage controls works because we are just working with info from previous meet, and making it strictly earlier than what we want to predict. 

Finally for the data, I had four missing ages which were filled with training median, all features are standardize. Scaling statistics and medians are all in the training set only. 

## Modeling: three implementations, one model

The model's objective it to predict next competition's total in kg, which is continuous, so we are doing a linear regression. The three implementations use the same 15 standardised features, the same time-based split and the same objective (MSE with no regularization). The last point matters, scikit-learn's Linear Regression is unregularized, so PyTorch versions must be the same. 

* scikit-learn: LinearRegression, which solves least squares directly (no iterations).
* Manual PyTorch loop: raw tensors w and b initialized at zero, MSE loss, gradients from loss.backward(), and a manual update w -= lr * w.grad inside torch.no_grad(). Full-batch gradient descent, learning rate 0.1, 3,000 epochs.
* Standard PyTorch: a torch.nn.Module wrapping nn.Linear, nn.MSELoss and torch.optim.SGD with the same learning rate and epochs (default random initialization).

Since the features are standarized, the learning rate of 0.1 is stable. Training loss falls from 276,000 to 527 withing the first 500 epochs. 

| Model | MAE (kg) | R² | Mean error (kg) | Overpredicted (%) |
|---|---|---|---|---|
| Naive baseline (same as last meet) | 22.3 | 0.956 | −15.5 | 22.5 |
| scikit-learn | 16.2 | 0.977 | 1.5 | 52.8 |
| PyTorch manual loop | 16.2 | 0.977 | 1.5 | 52.8 |
| PyTorch nn.Module | 16.2 | 0.977 | 1.5 | 52.8 |

The model is reducing the average error by 27%, which compared to the baseline, which is extremely biased (-15.5kg) because it ignores lifters usually improve between competition, it is a really good improvement. 

Errors are lower on the test period than validation, in both the baseline and the model. Suggesting the periods July 2025-June 2026 were easier to predict, not that the model is better. 

So, can we say that the three agree? Well, yes, they produce identical metrics and coefficients identical to two decimals. Agreeing wasn't automatic, I did investigate the representation of the best previous total affects it: 

| Configuration | Agreement between the three models | Why |
|---|---|---|
| A. `prev_total` + `best_prev_total` + `gap_to_best` | Predictions identical; coefficients almost identical (63.90 / 63.50 / −1.30; nn.Module 63.89 / 63.51) | `gap_to_best` = best − prev exactly, so infinitely many coefficient combinations give the same predictions; the individual coefficients can't be interpreted |
| B. `prev_total` + `best_prev_total` | Disagree: `prev_total` coefficient 90.77 (scikit-learn) vs 74.53 (PyTorch), `best_prev_total` 36.87 vs 53.21; predictions up to 10.97 kg apart, although MAE is identical (17.5 kg) | The two features are almost identical, so the loss surface is nearly flat in that direction: after 3,000 epochs the training loss is still creeping down (528.0 → 527.5) towards the 527.0 that configuration C reaches by epoch 500 |
| C. `prev_total` + `gap_to_best` (final) | Exact: max difference 0.0 kg | Same information in two weakly correlated features, so gradient descent converges to the exact solution |

The coefficient say that the previous total dominates (127kg), next is time since last competition (+8.1), sex (+7.7), moving up weight class (+7.6) and age (-5.9). Previous attempts add 1.9 which is consistent with the bounce back idea.

## Limitations & next steps
1. Merged identities may remain. The source identifies lifters by name, and my rule only catches merged identities that produce impossible jumps (a drop over 25%, or two totals on the same day). Two people with the same name and similar strength would go undetected and add noise to their histories. Next step: add a consistency check on age. A real person’s age must increase in line with the competition dates, so an ID whose recorded age jumps backwards or forwards inconsistently is probably two people.
2. One federation, one country. The model was trained and tested only on Spanish raw lifters, so it may not generalize to other federations, where competition calendars and levels differ. Next step: run the same pipeline on another IPF federation from OpenPowerlifting (e.g. Italy’s), first evaluating the Spanish model as is, then retraining.
3. 46% of the lifters in data have competed only one, showing there is a big amount of people that just have one competition. This falls in the first-time competitors, making it more difficult to calculate their prediction. Next Step: separate cold start model using only sex, age and weight class. Making it less accurate but could still be useful.A straight-line model for a curved reality. 
4. Improvement isn’t linear: beginners improve fast and then plateau, and strength eventually declines with age. A linear model can only add a fixed amount per unit of experience or age. Next step: stay with linear regression but add transformed features, such as experience buckets (first year, 1–3 years, 3+), age squared, or the logarithm of days since last competition.

5. The safety margin is the same for everyone. Uncertainty clearly differs between lifters (e.g. after a long break, or for beginners), yet the dashboard subtracts the same 10 kg from every prediction. Next step: model the size of the error as a function of features like `days_since_last` and `n_prev_valid`, and give each lifter a margin proportional to their own uncertainty.
## Generative AI use disclosure

I used Claude (Anthropic) throughout this assignment:
Dataset selection: After finding out that the original dataset was going not give me what I was looking for, I asked claude for other ideas in topics I liked, Claude researched and audited candidate datasets (bank marketing, UFC, League of Legends, chess, OpenPowerlifting) for size, leakage and data quality. I chose OpenPowerlifting and the Spanish federation subset.
Code: Claude wrote prepare_data.py and adapted the code in train.ipynb  from the previous dataset to the new one, moreover, it refined it to make it give what we were trying to get, also it did the whole code for app.ipynb 
Decisions: Claude explained the options and trade-offs for the business framing, scope, split, metric and data-quality rules. I made the final choices, including the season-planner product, the 10 kg safety margin and adding the missed-attempts feature.
Report: Claude drafted and idea of what we had, then I refined it and add what I thought was missing along with my own ideas. The limitations part was completely done by claude, since I was not reaching good next steps for the limitations found.
