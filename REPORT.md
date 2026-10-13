# Assignment 1 Report

- **Name**: Lucas Agustin Vucinovich
- **Student ID**: 18140
- **Email**: lvucinovich.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

The data I used is the French Bakery Daily Sales dataset, which contains the real sales records of one bakery in France between 2 January 2021 and 30 September 2022. It covers 148 different products, from the classic baguette and croissant to regional specialities like Kouign Amann and Far Breton. The dataset was originally published on Kaggle by Matthieu Gimbert, but I used a cleaned daily version from the GitHub repository tlukanie/FrenchBakerySales_TimeSeriesForecasting, specifically the file data/daily_sales_french_bakery.csv.

The file is around 2 MB and has 57,046 rows and 4 columns, where each row is the sales of one product on one day. The column unique_id is the product name (for example TRADITIONAL BAGUETTE or CROISSANT), ds is the date of the sale, y is how much that product sold that day in euros (so it is revenue, not units), and unit_price is the price of one unit in euros.

Since I wanted to predict the bakery’s total revenue, I added up all the products for each day. This gave me a table with one row per day, 637 days in total, where the target is the total daily revenue in EUR. The bakery was open on 600 of those days, making about 931 EUR on average and 3,184 EUR on its best day. On the remaining 37 days there were no sales at all, which means the shop was closed.

I chose this dataset for a few reasons. First, it is real business data and not synthetic, so the patterns and the problems in it, like closures, holidays and summer peaks, are real too. Second, it has a clear and continuous target, daily revenue, which is a good fit for linear regression. It also connects to a concrete business decision, since forecasting revenue a week ahead can help the bakery plan staff shifts. The feature engineering is not trivial either: revenue depends on the day of the week, the season, holidays and recent sales, so I had to build all of these features myself from the date, the sales history and external French holiday calendars. Because it is a time series, I also had to split the data by time instead of randomly, otherwise information from the future would leak into the training set. Finally, the dataset is small enough to keep in the repo and retrain the model from scratch in a few seconds.

## Business / real-life framing

The model is meant for the owner of the bakery, who plans the staff shifts one week in advance because employees need to know their schedule with that much notice. Understaffing means queues and lost sales, and overstaffing means paying wages for nothing, so the model forecasts each day’s total revenue seven days ahead to help decide how many people to schedule.

This shaped the pipeline. The target is total daily revenue in euros, because staffing depends on the overall volume of the business, not on individual products. I removed the 37 closed days from training, since a zero there means the shop didn’t open, not that demand was low, and the owner already knows those days in advance, so their forecast is simply zero. I also used the model to estimate how much revenue was lost on those closed days: about 15,000 EUR over the 23 closed days that have enough sales history. The seven-day horizon also ruled out some features: I couldn’t use yesterday’s sales, so I used sales from the same weekday last week (lag7) and the average of the same weekday over the last four open weeks (roll4). I dropped weather for the same reason, since it isn’t known a week ahead, and it also made validation worse.

A time-based split was necessary, because a random split would let the model see the future during training. I trained on January 2021 to March 2022, validated on April and May 2022 to choose features and alpha, and kept June to September 2022 as a test set, which I only used to report results, never to make a decision.

Since this is regression, the equivalent of a threshold is the safety margin added on top of the forecast. Falling short costs more than overshooting (lost sales against a few extra hours of wages), and the model underpredicts on 59% of the days in the test set, so a margin makes sense. I use MAE in euros as the main metric because it is easy to explain to the owner, plus an asymmetric cost metric. The dataset has no cost information, so I made up two example costs to be able to build the margin: 0.6 per euro short and 0.3 per euro over. They are not measured values; I only chose them so that a euro of shortfall costs twice as much as a euro of surplus, because I believe falling short is worse for the bakery. An extra employee is not fully wasted, since they can restock, clean or prepare the next day's production, while a lost customer cannot be recovered. Because the margin is a decision, I choose it on the validation set, which gives +6%. On the test set, that margin lowers the cost from 11,791 EUR to 10,150 EUR, about 14% less than planning with no margin. Since the costs are invented, the exact result matters less than the method, which is why the dashboard lets the user change them. With the bakery's real product margins and wage costs, the same method would give the real margin.

## Data preparation & feature engineering

I added up the 148 products for each day and filled in the missing calendar dates, which gave 637 days. The 37 closed days (zero sales) were removed from training and treated as missing when computing past sales, because counting them as 0 € would artificially lower lag7 and roll4 for the next week. I also dropped the first open day of each weekday (7 days), because they have no earlier same weekday to compute lag7 from, leaving 593 days: 413 for training, 60 for validation and 120 for test. The StandardScaler is fitted only on training data, so nothing from the future leaks in.

Since the model is linear, every pattern has to be given to it as a feature. Day of the week and month are one-hot encoded, because weekends sell much more and summer tourism creates a strong peak. lag7 (sales on the same weekday last week) and roll4 (average of the last four open days with that weekday) capture the recent level of demand. Public and school holidays come from official French sources and are stored in data/. I used Zone B school holidays because I inferred from the regional products that the bakery is in Guérande, though the dataset doesn’t say so. Finally, weekend_summer marks weekends in July and August, since a Sunday in August sells more than the two effects added together, and a linear model can’t learn that interaction on its own.

I chose the features by adding them in groups and checking the MAE:

| Features (cumulative) | Validation MAE (EUR) | Test MAE (EUR) |
|---|---|---|
| Calendar | 254 | 326 |
| + lag7 | 245 | 262 |
| + roll4 | 243 | 255 |
| + holidays | 195 | 202 |
| + weekend_summer | 195 | 198 |
| + weather *(dropped)* | 201 | 191 |

Holidays gave the biggest improvement. I dropped weather because it worsened validation (195 to 201) and isn’t known a week ahead. It did improve test slightly (198 to 191), but I decided based on validation, which is the right approach, and using a weather forecast is left as a next step.

Two last points: y is revenue in euros, not units, and 14 of the 37 closed days are the Wednesdays from January to April 2021, when the bakery closed every Wednesday. Since there was no earlier open Wednesday to compute lag7 from, the lost revenue from closures is estimated only for the other 23 days.

## Modeling: three implementations, one model

I used linear regression because the target, daily revenue in euros, is a continuous number and not a category. More specifically I used Ridge, which is linear regression with an L2 penalty. Several of my features are correlated (lag7 and roll4 measure almost the same thing, and there are many one-hot columns), and the penalty keeps the coefficients from becoming unstable. I chose alpha = 30 on validation from 0, 1, 10, 30, 100, 300 and 1000. Since alpha = 0 is plain linear regression, that option was tested too and performed worse.

To make sure the three implementations are really the same model, they all use the same 22 features, the same scaler fitted on training data and the same split. In PyTorch I wrote the loss to match sklearn’s exactly, MSE + (alpha / n) × ‖w‖², without penalising the bias, as sklearn does. I also standardised the target in PyTorch so the gradients are stable, and converted the predictions back to euros afterwards. The manual version uses plain tensors, loss.backward() and updates the weights by hand, while the second one uses nn.Linear with torch.optim.SGD. Both train for 3,000 epochs with a learning rate of 0.03.

Choosing the learning rate taught me something. At first I picked it by validation MAE, and 0.003 won, but with that value the model hadn’t finished converging and the PyTorch coefficients were about 3 € away from sklearn’s. The learning rate isn’t a parameter of the model but of the method used to find it, so I switched to choosing it by the lowest final training loss. An unconverged model acts like hidden regularisation, and once I fixed this the three versions matched.

Results on the test set (June to September 2022, 120 days):

| Model | MAE (EUR/day) | RMSE (EUR) | R² | Days under-forecast |
|---|---|---|---|---|
| Baseline: train mean | 515.17 | 695.56 | −0.41 | 63% |
| Baseline: same weekday last week | 282.06 | 405.30 | 0.52 | 48% |
| scikit-learn Ridge | 198.04 | 275.06 | 0.779 | 59% |
| PyTorch manual loop | 198.04 | 275.06 | 0.779 | 59% |
| PyTorch nn.Module + optim | 198.04 | 275.06 | 0.779 | 59% |

The three implementations agree: the largest difference in predictions is 0.002 € and in coefficients less than 0.001. They aren’t exactly identical because sklearn solves the problem directly while PyTorch approaches the solution step by step with gradient descent, so after 3,000 epochs it is extremely close to the exact solution but not exactly on it, and the small rounding errors of thousands of float32 updates add up differently.

The model beats the strong baseline (same weekday last week) by about 84 € per day, roughly 30% less error, and beats the mean baseline by far. The mean baseline even has a negative R², because the test period is summer while the mean comes from a training period that mixes all seasons. The model underpredicts on 59% of days, more than the strong baseline, which supports the safety margin from the business section.

Looking at the coefficients (in euros per standard deviation, since the features are standardised), the largest are school holidays (+123), Sunday (+94), lag7 (+86), July (+80), August (+72), roll4 (+69) and public holidays (+61). This matches the story of the data: tourism, summer and weekends drive the bakery’s sales.

## Limitations & next steps

The biggest limitation is the validation set. It covers only 60 days of spring, while the test set is summer, so it doesn’t represent the period the model is evaluated on. This showed up more than once: in an earlier version, validation picked alpha = 300, which made test results worse, and the best safety margin was +6% on validation but +14% on test. To fix this I would use rolling-origin cross-validation, with several successive time cuts, so the hyperparameters are chosen across different seasons and not from a single spring.

Related to this, the model has seen very little history. The training data runs from January 2021 to March 2022, so the only summer it learned from is 2021, and it is then tested on summer 2022. More years of data would help it learn the high season properly.

The model also underpredicts systematically in the high season. Over the whole test set it falls short on 59% of days, and in July and August it underpredicts by about 122 € per day on average, while in June it is roughly on target (it actually overpredicts by about 25 €). It is additive, so it adds up fixed effects in euros, but in the high season the effects seem to multiply, which is why I had to add weekend_summer by hand. A concrete fix would be to predict the logarithm of revenue, so that effects become percentages (for example “+20% in August”) instead of fixed amounts, while the model stays linear.

Weather was only tested with actual observed weather, which isn’t available seven days ahead, so I dropped it even though it improved test (198 to 191). A fair test would use historical forecasts, meaning what the forecast said a week before each day. Open-Meteo keeps past forecasts made several days ahead (its Previous Runs API), which could be used for this.

Finally, the model predicts revenue, but the owner really needs to know how many people to schedule and at what times, and the data is daily with no hourly detail. The next step would be a conversion rule built from past shift records (for example one employee per X € of sales) and hourly data to separate mornings from afternoons. On top of that, the location and the costs (0.6 and 0.3) are my own assumptions and should be confirmed with the owner, using real product margins and the real hourly cost of an employee.

## Generative AI use disclosure

I used AI in a few specific parts of the project. At the start, I used it to check whether the dataset I found was suitable for the assignment. While coding, it helped me fix bugs and clean up some of the more complex parts of the code. I also used it to search for external datasets that could improve my predictions, such as the holiday and weather data, to build the dashboard, and to proofread and correct my report.
All the decisions, the analysis and the answers in this assignement are my own, and most of the code was written by me by hand.