# Assignment 1 Report
**Name:** Antonio Cabral de las Casas  
**Student ID:** 19360
**Email:** acabral.ieu2022@student.ie.edu 
**Group:** BBADBA A

## Dataset
I chose an historical dataset from SPY which is an ETF that tracks the S&P500 because I am very passionate about the stock market.

The dataset comes from thsi repo in github: "https://github.com/OStochastic/Daily-SPY-data-from-2000-2025", it contains the daily opening, closing, highs and lows prices and trading volume. One row = one trading day.

After cleaning and preparing the data I had 6433 observations.

## Business / Real-life Framing
My goal was knowing if I could predict if SPY would go up or down during the next trading day, which could help an everyday investor decide to buy SPY at the next market open and sell it at the close.

The target is 1 if the next day closing price is higher than its opening price and 0 otherwise.

I divided the data chronologically by: 70% training, 15% validation and 15% testing to avoid using future information.

I also gave false positives a cost of 3 and false negatives a cost of 1, mainly because I wanted to penalize buying before a negative day much more than missing a positive one. I selected the decision threshold using validation data.

## Data Preparation & Feature Engineering
First I cleaned the dataset, sorted the dates and removed missing or invalid values.

Then I created 12 features including past returns, volatility, moving averages, price range, trading volume and day-of-week indicators.

I chose these indicators because they can help identify market trends and changes in the market.

I also used StandardScaler fitted only on the training data to put the features on a similar scale and avoid data leakage.

## Modeling: Three Implementations, One Model
I used logistic regression because the target has two possible outcomes: UP or DOWN.

I trained the same model using scikit-learn, manual PyTorch and standard PyTorch, with L2 regularization.

This were the results (I asked chatgpt to do this table):
| Model | Accuracy | ROC AUC | Cost / 100 |
|---|---:|---:|---:|
| Scikit-learn | 46.22% | 0.5062 | 53.78 |
| Manual PyTorch | 46.22% | 0.5064 | 53.78 |
| Standard PyTorch | 46.22% | 0.5064 | 53.78 |
| Majority baseline | 53.89% | 0.5000 | 138.34 |

All models gave very similar results, there are only very small differences that are probably caused by how each method optimizes the weights.

However and not at all surprising (this were very simple models and institutions work with much better data...) the ROC AUC was close to 0.5 so the models were not good at predicting the direction at all.

The selected threshold was 0.61 which reduced our hypothetical cost but always predicting DOWN would give almost the same cost (53.89). 

## Limitations & next steps
The main limitation is that historical prices alone were not enough to predict SPY movements accurately. The model also uses hypothetical business costs, not real trading profits or losses.

If I continued the project, I would add indicators like the VIX and interest rates, test different market periods, include transaction costs and maybe add an LLM that could read the news in seconds and add them to the thesis of the day. Eve with all that, I doubt that I could get some kind of edge, and if I did, I dont think it would take long for it to disappear. The sad truth about being a retail investor is that you fight against institutions which are "gods" in the sector, that's why I always recommend to always go for the long term and never try to outsmart "the casino".

Overall, I learned that correctly training a model does not mean it will make good predictions.

## Generative AI Use Disclosure
ChatGPT helped me write and understand the Python code and interpret the results. I ran the models, wrote the report and tested the dashboard myself.