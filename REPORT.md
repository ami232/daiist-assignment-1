# Assignment 1 Report

- **Name**: Andrea Saxod
- **Student ID**: 18996
- **Email**: asaxod.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset
The dataset is based on daily OHLCV (Open, High, Low, Close, Volume) data for the S&P 500. The data was retrieved from Yahoo Finance using the yfinance library and covers January 2014 to December 2024 (about 11 years). In terms of raw trading days, we have around 2767, of which 2516 were usable after feature engineering because some features, like 52 Week High, require 252 days of past data before they can be calculated. The raw columns in this dataset are Date, Open, High, Low, Close, and Volume; each row represents one trading day for the S&P 500 index. I chose this dataset because it's challenging to build useful features from raw price data. I started with only 5 numbers per day, and I had to turn them into something the model could learn from. Also, the data split had to be done carefully, as we cannot mix past and future data. 

## Business / real-life framing
Regarding this scenario, we are looking at a trading desk in an investment firm that runs a simple strategy. Every evening, the model observes today's market data and predicts whether the S&P 500 will close higher tomorrow. If our model predicts "Up," the desk buys overnight; if it predicts "Down," the desk does nothing. 

This business framing is what guided the next 3 important decisions of the pipeline:

1) The target was defined as binary: 1 if tomorrow's close is higher than today's close, or 0 if it is lower. This was built by shifting the Close column forward by one day for comparison. However, the very last row of the dataset has no tomorrow, so it gets dropped;

2) To split the data, a time-based split was used. This was the most critical decision in the timeline. If I had randomly shuffled the data, a data point from 2023 could have ended up in the training set while a point from 2020 ended up in the test set. The model would then have learned from the future to predict the past, which is look-ahead bias. In that case, we would probably get very good test results, but the model would be useless in real-world scenarios where we never have tomorrow's data. I split the data into 70% training (2015 to 2021), 15% validation (2022 to mid-2023), and 15% testing (mid-2023 to 2024);

3) Finally, the way in which I think of errors is important, as not all mistakes are equal. In this case, a false positive meant the desk bought the index, and the market dropped, a real financial loss. A false negative meant the desk stayed out and the market went up; a missed opportunity but with no loss. Because real losses hurt more than a missed gain, I set the cost of false positives at $ 10,000 and false negatives at $5,000. These are purely illustrative numbers that reflect a real trading concept. A conservative desk should raise the decision threshold above 0.5 so the model signals buy only when it is very confident. 

## Data preparation & feature engineering
I started with 5 raw columns, and I built 25 features across 8 categories:
- Returns (4 features): I calculated the percentage change in the closing price over 1, 5, 10, and 21 trading days. These capture whether the market has been in an uptrend or downtrend, and how fast. For example, a market in a 21-day rise is very different to one that has recently had a big drop;
- Moving average ratios (6 features): I computed the simple moving averages over 5, 10, 21, and 50 days, then expressed the current price as a ratio to each of them. I also computed two crossover ratios: the 5-day average divided by the 21-day average, and the 10-day average divided by the 50-day average. I chose ratios because in 2014 the S&P 500 was around 1,800, while in 2024 it was at 6,000, meaning a completely different scale; ratios let us compare performance over time;
- Volatility (5 features): I measured how daily returns have been jumping around the last 5 and 21 days, alongside their ratios. I also computed the intraday range, which measures the difference between the daily high and low price relative to the closing price. Markets move in periods; a volatile day is usually followed by more volatile days;
- RSI (1 feature): This is the relative strength index over 14 days. It measures whether the price moves have been mostly up or down. It ranges from 0 to 100; above 70 means the market has been rising sharply recently and may be due for a pullback. Below 30 means the opposite. This is a mean reversion signal;
- MACD (3 features): This is the difference between the 12-day and 26-day exponential moving average, plus a signal line (9-day average of MACD) and their difference. It is one of the most common momentum indicators, as when the MACD crosses above the signal line, it is usually considered a buy signal;
- Volume (2 features): I computed today's volume relative to the 21-day average, alongside the one-day change in volume. When markets move on high volume, the change is considered more impactful than a low-volume move;
- Relative position (2 features): This is where I measured the current price relative to the 52-week high and 52-week low. A market near its all-time high behaves differently than a market that has dropped meaningfully;
- Day of week (1 feature): Monday through Friday is encoded as 0 to 4. There is a known pattern called the Monday effect, where Mondays tend to show slightly weaker returns;
- Bollinger Band position (1 feature): This measures where the price sits within its recent range. A value near +1 means the price is at the top of its range; near -1, it's at the bottom;
- Cleaning decisions: rows where any feature was missing were dropped, not filled in with hypothetical values. Dividing by zero (which can happen when volume is zero on a half day) was handled by returning 0 instead of infinity. All features have been scaled using StandardScaler, which subtracts the mean and divides by the standard deviation. I fit the scaler only on the training data; otherwise, if I fit it on the test data, I would include the mean and standard deviation of the training data, which is a form of data leakage.  

## Modeling: three implementations, one model
The model is logistic regression, which is a binary classifier. It takes all 25 features, multiplies each by a weight, adds them with a bias term, passes the result through a sigmoid function to get a probability between 0 and 1, and then compares that probability to a threshold to decide "up" or "down".

I trained this model using three different methods on the same data split:
1) Scikit-learn: I used LogisticRegression with L-BFGS (an optimization method that uses second-order math to jump straight to the best weights) and C = 1.0. C controls L2 regularization, which penalizes large weights; it stops the model from relying too heavily on a single feature and helps prevent overfitting;
2) Manual PyTorch: I created a weight matrix and bias as raw tensors, made a forward pass, then called the backward function that triggered autograd. This calculates how much the weights should change. The weights would then be updated by hand using the following function: W -= lr * W.grad. I then used mini-batch stochastic gradient descent, training on small chunks of 256 samples at a time, with a learning rate of 0.01 for 200 epochs;
3) Standard PyTorch: This is where I defined a model class with one linear layer (which handled the matrix and bias internally). For the loss, I used binary cross-entropy that takes raw outputs before sigmoid. It is more numerically stable because it avoids taking the log of very small numbers. Regarding the weights, I used the Adam optimizer, an advanced version of stochastic gradient descent that automatically adjusts the learning rate for each weight based on its gradient history; it converges faster. I started with a learning rate of 0.01, a batch size of 256, and 200 epochs.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Naive baseline (always Up) | 0.5661 | 0.5661 | 1.0000 | 0.7230 | 0.5000 |
| Scikit-learn LR | 0.4921 | 0.5407 | 0.6822 | 0.6033 | 0.4640 |
| Manual PyTorch LR | 0.5053 | 0.5477 | 0.7243 | 0.6237 | 0.4827 |
| Standard PyTorch LR | 0.4894 | 0.5443 | 0.6028 | 0.5721 | 0.4726 |

Did my three models agree:
Yes, they agreed more or less. Regarding accuracy, they landed around 49-51 percent, and 46-48 percent for the AUC. The small differences came from three things:
1) They used different optimizers: L-BFGS finds the exact best solution, while SGD takes many small noisy steps, and Adam takes the smart adaptive steps;
2) Scikit-learn applies L2 regularization by default, while PyTorch models do not.
3) The mini-batch randomness in the PyTorch models means they take slightly different paths to the solution;
However, as logistic regression has a convex loss function, all three models do end up nearly in the same place.

Why did all models perform worse than the naive baseline: 
The naive baseline only predicted up; it got 56.6% because the market happened to go up on 56.6% of test days. The models try to be selective and predict down on some days, but they get it wrong often, which hurts their overall accuracy. This is expected because the Efficient Market Hypothesis states that public information is already reflected in stock prices. Technical indicators like RSI and moving averages are built from public price data, so that information is already priced in. If a simple logistic regression could predict the S&P 500, every trader would use it, and the edge would instantly disappear. The fact that the model cannot be a coin flip suggests the pipeline is working and markets are efficient to some extent. 

## Limitations & next steps
- The model is too simple. Logistic regression assumes a linear relationship between features and the probability of going up. Financial markets are non-linear and change with behavior depending on regimes. A tool such as gradient-boosted trees could capture these non-linear patterns, but it would still struggle using daily index returns;
- The market regime changed from the training to the test: the training data covers a long bull market and the COVID crash, while the test covers the post-rate-hike recovery. Features that were useful in one regime may not work in the next. A better approach is to retrain the model on a rolling window, for example, refitting the model every month on the most recent 2 years of data;
- The target is extremely noisy: predicting whether the S&P500 goes up or down tomorrow is one of the hardest financial problems. A less noisy target might be the direction over the next 5 days or whether the next day move is larger than 0.5%;
- No transaction costs: the business cost in the dashboard assumes a fixed dollar amount per wrong trade, but a real trading strategy would need to account for bid-ask spreads, slippage, and position sizing. What matters in practice is the Sharpe ratio, not just raw accuracy;
- Missing data sources: I only used price and volume data, which is the most basic and efficiently priced information. Real desks use additional data like bond yields, VIX, sector rotation patterns, macroeconomic releases, and sentiment data from news and social media. These alternative data sources may carry signals that are not yet fully priced in.
  
## Generative AI use disclosure
I use AI for boilerplate code such as loading data, saving models to disk, and building the dashboard. All decisions, such as which dataset to use, how to frame the business model, feature engineering, and so on, are my own. 
