# Assignment 1 Report

- **Name**: Roberto Falcato
- **Student ID**: 20219
- **Email**: rfalcato.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

The data is from the Formula 1 World Championship data set based on Ergast (Kaggle), a relational set of 14 tables. I used seven: results (base table), races, drivers, constructors, circuits, qualifying, status. Each row represents a driver in a single Grand Prix. After limiting to seasons 2014-2024 (hybrid-engine era), the modelling table has 4,626 rows and 13 model features. To avoid leakage and respecting the repository size limit, we excluded post-race tables (lap_times, pit_stops) and post-race standings . The dataset is chosen for two reasons; the target must be built from multiple tables and the time structure calls for careful split and feature design.

## Business / real-life framing

FantasyGP is a fantasy F1 platform, which will suggest drivers who are most likely to score points (top-10 finish) in the next race. The model runs after qualifying and before the race, so only the starting grid, the qualifying results and previous races can be used. This leads to three pipeline decisions:

- scores_points = (points > 0) * 1. This is equivalent to a top ten finish (zero mismatches) for 2014-2024, so the target is clear.
- Split: chronological, since a random split would allow predictions of past races to be influenced by future races. Train 2014–2019 (2,487 rows, 48.7% positive), validation 2020–2021 (780 rows, 50.0%), test 2022–2024 (1,359 rows, 50.0%).
- Threshold metric: expected misclassification cost. A false positive (recommending a non-scorer) has a cost of 1, a false negative (missing a scorer) costs 2, because the value of the platform is in finding undervalued drivers. This cost is minimised on the validation set to choose the threshold (which is not fixed at 0.5). With calibrated probabilities, the theoretical cut-off is 1/(1+2) ≈ 0.33.

## Data preparation & feature engineering

I combined the seven F1 tables into one table with one row per driver per race (2014 to 2024, 4,626 rows). The model makes predictions after qualifying and before the race so I only used what is known by then: starting grid, qualifying results, how often a driver and team finished in the top 10 in their last 5 races, how often they did not finish in the last 10, and age and experience of the driver. Recent form only uses earlier races so a race doesn’t see its own result. The data is split by time (train 2014-2019, validation 2020-2021, test 2022-2024).

Two data-quality decisions:
- 79 rows (1.7%) start from the pit lane, which the data records as grid position 0 and would look like pole position. I changed it to last place. Without this, the model score (AUC, where 0.5 is guessing and 1 is perfect) drops from 0.8686 to 0.8657.
- 199 rows (4.3%) have missing values, mostly rookies with no history. I filled them with the typical value instead of deleting them. The impact is tiny (filling with zeros gives 0.8681).

## Modeling: three implementations, one model

I used logistic regression, because the question is a binary and I need a probability, so I can choose the threshold using the cost of each type of mistake. I trained the same model three ways on the same split: scikit-learn, a PyTorch loop written by hand, and the standard PyTorch tools (`nn.Module` + `optim`). Results:

| Model | AUC | Accuracy | Cost per driver-race |
|---|---|---|---|
| Naive baseline (starts in top 10) | 0.8318 | 0.7712 | 0.3444 |
| scikit-learn | 0.8686 | 0.7962 | 0.2884 |
| PyTorch manual loop | 0.8686 | 0.7962 | 0.2870 |
| PyTorch `nn.Module` + optim | 0.8686 | 0.7962 | 0.2870 |

All three beat the baseline by the same amount, and their probabilities differ by maximum 0.0015. The tiny gap with scikit-learn exists because it keeps training until it fully converges, while my PyTorch versions stop after a fixed number of steps. The small cost difference (0.2884 vs 0.2870) comes from a single driver that falls on the other side of the threshold.

## Limitations & next steps

- **The basic grid already provides a good starting point.** In fact, it achieves an AUC of 0.8318 on its own. When we add our model to the mix, the score increases to 0.8686, but the difference isn't huge. It seems that most of the useful information is concentrated at the starting point of the grid. So, what's next? We should try adding more data that the grid doesn't currently include, like the team's recent performance compared to others in similar races. This could help us see if the score can be improved even further.
- **One thing didn't work out.** When the grid is compared to a teammate, it gets a positive score (+0.175), but if you're starting behind your teammate, it should actually lower your chances. And it turns out, the model works a bit better without it (0.8693). It's likely that this feature is similar to the grid and qualifying position, but I haven't checked that yet. * Next step:* see how closely it's related to them and get rid of it if necessary.
- **Let's try a different approach.** We started by testing the threshold using just two seasons -2020 and 2021, which gave us 780 rows of data. But the threshold we got, 0.28, isn't the best one when we look at the test set, where 0.34 works better. So, what's next? We should try choosing the threshold using more seasons. For example, we could test each season against all the previous ones, one by one.
- **My testing was limited to just one period and a single type of model.** I based my judgments on the years2022-2024, but I should note that any differences below 0.001 AUC aren't really reliable. A simple straight-line model can't handle complex situations either, like when a strong team is at a circuit where it's tough to overtake. * What's next:* I plan to repeat the evaluation, looking at each season separately, and also try using a tree-based model to see if it can find more insights.

## Generative AI use disclosure

I used Claude throughout this assignment. What it did:

- **Dataset and framing:** Suggested dataset options and helped me decide which of the 14 Formula 1 tables to use. I chose Formula 1 dataset from its options.
- **Code:** Assisted during the development of `train.ipynb` and where it really helped as I had not coded an app before on my own was suring the development of `app.ipynb`. However it did not fully developed the code completely without my inputs and my understanding.
- **Text:** Helped me brainstorm and organize my report. Not as in writing the report for me but it helped me understand what each section was demanding, and how to frame it based on my project
- **Explanations:** Explained parts of the setup code I did not undertand, the results shown in the app, and helped me with the setup overall.
