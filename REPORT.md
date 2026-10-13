# Assignment 1 Report

- **Name**: Maria Fernandez Gomez-Monche
- **Student ID**: 19784
- **Email**: mfernandezg.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

I used a Kaggle dataset called Vehicle Sales Data (`car_prices.csv`): 558837 used car sales at USA whole sale auctions between December 2014 and July 2015.
One row = one car sold at auction (year, make, model, trim, body, transmission, state, condition score from 1-49, odometer(miles), color, seller, VIN, mmr, sellingprice, sale date). (mmr` is a third-party estimate of the car's wholesale market value)

The full dataset is not used. I cleaned it by dropping rows with missing mmr, sellingprice, date and wrong prices (mmr < 1000 or sellingprice < 500), then samples 9000 sales from December-March, 2500 from may and 2,500 from june. 

I picked this dataset because a car's price depends on many  things such as the year its from (age), miles, condition, seller and geography. Also there are raw columns that are messy (inconsistent), there are missing transmissions and  conditions so feature engineering is important.

## Business / real-life framing

We are a dealership that buys used cars at auctions including lease returns. To decide how high to bid on each car so we use the dataset to predict how far the final auction price is from its reference market value (mmr (third-party estimate)).

- **Target**: sellingprice / mmr - 1 (example: +0.05 = the car closed 5% above mmr). It is relative because our bidding rule is also relative to mmr so cheap and expensive cars can be compared. 

- **What the model does and does not know**: it does not know what a car is worth to us or what profit we will make: the dataset has no costs and no resale prices. It only anticipates how high other buyers bids usually go. 

- **Bidding rule**: max bid = mmr x (1 + prediction - X), never above 1.2 x mmr. The 20% cap is a business assumption, not estimated from data. X is a safety cushion subtracted to protect us from overestimating. It is fixed from the model validation errors, never by eye and never looking at the test set: X = the 75% quantile of (prediction - actual) in May, so that after subtracting it the model still overestimates in about 25% of validation cars. The 75% is a business choice (how much overestimation we tolerate).

- **Which error hurts most: overestimating.** If the model predicts a higher close than the real one, we bid too high which means that we pay above the value. This means that money leaves our cash and stops us from buying other cars that are worth it. Losing an auction by bidding low is an opportunity cost. The dashboard therefore shows both: how often and by how much the bid lands above the final price and how many cars are lost.

- **Temporal split**: train December to May, validate in May, test in June (April has very few sales, so it is left out). 
When we use the model we only know the past, so a later test simulates it. A random split would inflate the result because sales from the same lot, or even the same car would fall in train and test. I did not measure the size of that inflation here. In the temporal sample only 1 VIN is shared between each pair of splits. The risk of this split is that we train on winter months and test on summer (which is also a limitation).

- **Metrics**: RMSE is the main metric because it punishes large errors, which are the very wrong bids. It is complemented with the mse prediction - actual (if it is positive the model overestimates). The signed error does not replace RMSE: it can be near zero if the model fails equally upwards and downwards. 


## Data preparation & feature engineering

Row features are in `common.engineer()`. Everything that learns statistics (median imputation, scaling, which categories are frequent enough to get a column...) is in a `ColumnTransformer` fitted on the training split, then applied to validation/test.

- age (sale year − model year) and age_sq: the price falls with age, but not in a straight line.
- log_odometer and odometer_per_year: mileage is very skewed, so I use its log. 80k miles on a 3-year-old car is not the same as on a 10-year-old. Odometer is capped at 300k.
- condition, condition_sq, condition_x_age and condition_missing: condition is the strongest signal. The square and the interaction with age let it matter differently on older cars. The missing flag keeps “unknown condition” as information, and the missing values are filled with the median.
- seller_is_lease: 1 if the seller’s name contains “lease”. It is a proxy for lease returns, which the scenario highlights.
- body_group: the body column has inconsistent labels (“Sedan”/”sedan”, “SuperCrew”, “G Sedan”...) so I map them by keyword to 10 groups.
- make, state, color, seller and transmission: one-hot encoded. Categories with fewer than 100 training rows are pooled as “infrequent”.

To decide which features to keep, I compared versions of the model on the validation set in May. The number is the RMSE, which is the models typical error in percentage points: lower is better. 

## Modeling: three implementations, one model

**Model**: 

- scikit-learn: I used Ridge, which finds the best weights directly with a formula (no step-by-step training). It has one setting, alpha, that controls how much the weights are kept small. I tried values from 0.1 to 3000 on the validation set (May) and alpha = 100 worked best.
- Manual PyTorch: I wrote the training loop by hand. The model starts with all weights at zero. At each step it makes predictions for all the cars, measures the error, asks PyTorch (loss.backward()) which direction improves it, and nudges the weights a small step in that direction (w -= lr * w.grad). It repeats this 4,000 times. The error it minimises is the average squared error plus a small penalty on large weights, which is the same ridge idea.
- Standard PyTorch: the same model built the usual way, with nn.Linear, the nn.MSELoss error and the torch.optim.SGD optimizer doing the steps for me. The ridge penalty comes from the optimizer’s weight_decay setting (set to 2 * alpha / n so it matches scikit-learn’s version) and is not applied to the bias. It also starts from zero and runs 4,000 steps.

All three solve the same problem, so they use the same alpha and should end up with the same model.

Test set (June, 2500 cars), in percentage points of mmr:
Results on the test set (June 2500 cars). Numbers are in percentage points (pp) of mmr. For each model I show three measures: the typical error (RMSE), the average bias (predicted − actual, positive means it predicts too high) and the average absolute error (MAE). Lower RMSE and MAE are better.

Trusting mmr gives a typical error of about 19 points. My three models all get about 18, so they are a bit better. On average trusting mmr predicts about 1 point too high and my models about half a point too low.

The three models give exactly the same results and all of them beat the baseline. On the validation set (May) the same thing happens: the baseline’s typical error is 19.5 and all three models get 18.4.

Do the three agree? Yes. Once the learning rate is chosen well, the three give practically the same predictions: the biggest difference between them are so tiny that they are just computer rounding. This makes sense because the problem has a single best answer and all three reach it. They would only differ if the PyTorch versions trained for too few steps, used too small a learning rate (this happened with 0.01), or used a different penalty than scikit-learn (that is why weight_decay is set to 2 * alpha / n).

The bidding cushion X and the result. I set X  using only the validation errors (May): it is the value for which about 25% of May bids still ended above the final price. Then I applied it once to the June test set, which the model had never seen.

Trusting mmr needs a cushion of about 8.7 points: about 20% of bids ended above the final price, by around $1,100 on average, and we lost about 80% of the cars (bid too low). With my linear models the cushion is about 7.5 points: about 20% of bids ended above the final price, by around $1,000 on average, and we also lost about 80% of the cars.

The baseline needs a bigger cushion (8.7 vs 7.5) for the same protection, because it is wrong by more. On May, 25% of bids end above the final price by design. The 20% cap never mattered in June.

**Limitations & next steps**
Trained in winter, tested in summer. The model learns from December to March and is tested in June, and cars may not sell the same way in summer. May and June also behave a bit differently, so the cushion that worked in May was slightly too cautious in June. With more months of data, I would test on several periods.
The gain over trusting mmr is small. The model is better than just trusting mmr, but not by much. Most of how high other buyers bid looks like random noise that the car’s characteristics cannot explain. Next step: try a more flexible model, such as gradient boosting.

A few extreme cars have a big effect. Some cars sell for much more or much less than mmr, and they pull the model’s error up. A method that is less affected by extreme values could help.
A bigger cushion protects us but loses cars. The more cautious we are, the fewer overpayments, but the more cars we lose. How cautious to be is a business decision, and the data cannot tell us which is best because it has no profit or resale prices. Adding real resale prices would allow that.

Lease returns are only an approximation. I mark a car as a lease return when the seller’s name contains “lease”. It did not improve the results. A real lease flag in the data would be better.

I used a sample, not all the data. I used a small sample (14,000 sales) of the more than half a million available, so rare makes and states are grouped together.

The condition score seems to mix two scales. Some cars have a score from 1 to 5 and the rest from 11 to 49. The 1-5 scores seem to be a separate scale (5 looks like an excellent car), but the model reads them as very low scores on the 1-49 scale. This is my own guess, not documented in the dataset. Next step: multiply the scores of 5 or less by 10 and retrain.


## Generative AI use disclosure

I used Claude (Anthropic) as a coding and writing assistant. I wrote the business framing (the bidding rule, the target definition, the temporal split and the choice of metrics) and Claude helped turn it into the project. Claude helped me wirte the data sampling script, the feature engineering and code (`common.py`, `train.py`) and the Gradio dashboard (`app.py`) as well explained concepts to me (for example the cushion X and the `condition` score) and checked the data when I asked questions.

I ran the code on my own computer, checked that the results matched the report, and decided to leave the `condition` variable as it is and list it as a limitation.