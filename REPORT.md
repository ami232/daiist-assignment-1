# Assignment 1 Report

Ignacio Leon Fierro Botin-Sanz De Sautuola
ifierro.ieu2022@student.ie.edu
20206 
[BBADBA A]



## Dataset

I used the UCI Bike Sharing Dataset (hourly version, "data/hour.csv"). It comes from the Capital Bikeshare system in Washington D.C. and covers 2011 and 2012. Each row is one hour and it contais the date, hour, whether it was a working day, the weather (temperature/humidity/wind/weather situation) and the number of bikes rented in that hour ("cnt"). It has 17,379 rows and 17columns. There are no missing values but 165 hours (rows) are missing from the file.

I chose it because the target is easy to understand (bikes rented per hour) but the relationship with the raw columns is not a straight line. I noticed that on working days demand has two peaks (morning and evening rush hour) and on weekends it has one peak around midday so a linear model only works if you build the right features.



## Business / real-life framing

**Scenario** Every evening the operations team of the bike-share company has to plan the next day to know how many bikes the trucks need to move between stations and how many staff to have there ready to respond to the demand. For that they need a forecast of rentals for each hour of tomorrow which is based on the calendar and the weather forecast.

This affects the pipeline in four ways:

- **Target** Total rentals per hour ("cnt"). I dropped "casual" and "registered" because they add up to "cnt", so they would be giving the model the answer, the two kinds of customers are simply people who paid for a single ride and people with a membership. The model predicts "log(1 + cnt)" instead of "cnt" because rentals are very skewed (median 142/max 977), the log also makes effects proportional, for example rain removes a percentage of demand not a fixed number of bikes, and the prediction can never be negative. All errors are converted back to bikes as we dony want a log, we apply the inverse to get bikes once again and all the errors in the report are all measured in bikes.

- **Split** I split by time istead of having it be random. The first 70% of days are for training (1 Jan 2011-25 May 2012, 12,139 rows) and then next 15% for validation (26 May-12 Sep 2012, 2,640 rows) and the last 15% for test (13 Sep-31 Dec 2012, 2,600 rows). A random split would mix hours from the same day in train and test and the real planner never has that information as they only know the past so it wouldnt make sense for it to be done at random.

- **Features** The only things that they know the evening before are the calendar and the weather forecast. They unfortunately dont have the number of rentals in the previous hour as they are planning for the full next day (although this would have been a good feature).

- **Metric** This is a regression so there is no decision threshold and I use MAE in bikes per hour (how many bikes I am off on average) and next to it the bias which is the average of prediction - actual because for operations predicting too low (empty docks/lost rides) is way worse than predicting too high. The regularisation strength was chosen using validation MAE although the differences were small as I will explain in more detail in the modelling section when I talk about the tuning.



## Data preparation & feature engineering

I added features step by step and fitted each version with scikit-learn (alpha = 1) to see what each step changed. MAE and bias are in bikes per hour:

| Feature set                       | Features | Val MAE | Val bias | Test MAE | Test bias |
--------------------------------------------------------------------------------------------
| 1. raw columns (hour as a number) | 7        | 148.2   | -76.4    | 156.1    | -132.1    |
| 2. + one-hot hour                 | 29       | 103.2   | -64.6    | 121.8    | -110.9    |
| 3. + hour x working-day           | 52       | 73.8    | -50.8    | 109.6    | -102.5    |
| 4. + temperature curve (temp²)    | 53       | 89.5    | -85.6    | 99.5     | -91.7     |
| 5. + growth trend (final)         | 54       | 60.1    | 35.9     | 49.3     | 13.7      |

- Using the hour as a number means demand can only go up or down in a straight line during the day which is wrong. One-hot hours fix the shape but working days with peaks at 8h and 17-18h and weekends with one peak around 13h have different shapes so I gave every (hour/ day type) combination its own dummy. One of the 48 possible cells whcih in this case is "00h on a non-working day" is left out as the reference.

- Demand goes up with temperature but flattens and drops in the hottest hours and the squared term lets the model capture that. I removed "atemp" which is essentially a "feels-like temperature" because it is 0.99 correlated with "temp" so there is no point in using it.

- I used One-hot to show how rain is much worse than mist for example. I also used "clear" as the reference as it is the normal condition. Category 4 (heavy storm) only has 3 rows in two years so I merged it into category 3 as it is a category so rare that it is hard to learn from it so there is no point keeping it separate unless it occurs more, which it doesnt.

- The Growth thrend is essentailly a clock, it is the Years since 1 Jan 2011. The system grew a lot as average hourly rentals were 144 in 2011 and 235 in 2012. With the time split that I chose the model trains mostly on 2011 and is tested on late 2012 so it needs a way to understand and learn the growth.

- Continuous columns are standardised with the mean and standard deviation of the training set only. This matters for gradient descent; scikit-learn would give the same answer without it.


Things that did not work as expected:

- Without the growth trend every version predicts too low (test bias around -100 bikes/hour) and step 3 is barely better than the simple baseline of "average for that hour and day type" (109.6 vs 116.4 MAE). The trend was the biggest single improvement.

- "temp²" actually just made validation worse (73.8 to 89.5) but it did make the test better (109.6 to 99.5). The validation period is the summer of 2012 which was the hottest period with the highest demand. I kept the feature because the drop in demand at very high temperatures is visible in the data, but the evidence is mixed.

- 22 rows (10 March 2011) have humidity 0, which looks like a sensor error but I left them in because they are only 0.2% of the training rows so its not a lot of impact.



## Modeling: three implementations, one model

The model is linear regression with an L2 (ridge) penalty because the target is continuous. All three versions minimise the same losswhich is mean squared eeror on "log(1 + cnt)" plus "alpha / n * sum(w²)" (the bias is not penalised). This is the same objective as scikit-learn's Ridge divided by n.

- scikit-learn: "Ridge(alpha)" which solves the problem exactly.

- Manual PyTorch: weight and bias tensors with "requires_grad=True", "loss.backward()" to get the gradients and then "w -= lr * w.grad" inside "torch.no_grad()" and zeroing the gradients by hand. Full batch, 3,000 epochs.

- Standard PyTorch: "nn.Linear" + "torch.optim.SGD" with the same loss and the same zero initialisation also full batch and 3,000 epochs.

Tuning: alpha in {0.1, 1, 10} by validation MAE (60.2 / 60.1 / 59.2, so 10 is the chosen one although the differences are very tiny) and learning rate in {0.01, 0.1, 0.5} by final training loss (0.3289 / 0.2065 / 0.2065 so I chose 0.5).

| Model                                      | Val MAE | Val RMSE | Val bias | Test MAE | Test RMSE | Test bias |
-----------------------------------------------------------------------------------------------------------------
| Baseline: training average                 | 226.88  | 299.83   | -199.73  | 187.15   | 263.38    | -153.17   |
| Baseline: average for that hour & day type | 146.08  | 191.14   | -143.83  | 116.41   | 166.79    | -97.10    |
| scikit-learn                               | 59.22   | 92.02    | 33.39    | 48.85    | 77.35     | 6.08      |
| PyTorch manual loop                        | 59.22   | 92.02    | 33.39    | 48.85    | 77.35     | 6.09      |
| PyTorch nn.Module + optim                  | 59.22   | 92.02    | 33.39    | 48.85    | 77.35     | 6.09      |

The models agree as the biggest difference in any coefficient between the PyTorch versions and scikit-learn is 0.0018 and the biggest difference in any test prediction is 0.38 bikes which is really little. The two PyTorch versions give the same result because they start from the same point and take the same steps. The tiny gap with scikit-learn is because gradient descent stops after 3,000 steps while scikit-learn solves exactly. This depends on the learning rate as with lr = 0.01 the loop is still far from the minimum after 3,000 epochs (loss 0.329 vs 0.206) so it would not have matched scikit-learn.

Compared with the best baseline (116.4 MAE) the model brings the test error down to 48.9 bikes per hour.



## Limitations & next steps

- The trend is a straight line as it already over-predicts on validation (+33) and would keep growing forever so a better option would be to retrain regularly on recent data or to use last week's demand at the same hour as a feature (because it is known a day ahead).

- The model is trained on observed weather but the planner only has a forecast so the real error would be larger. I would need historical forecasts to measure this.

- The test months (Sep to Dec) only appear once in training, in 2011 when the system was smaller, so season and growth are hard to separate. More years of data would help.

- In the weekly view of the dashboard the model misses the highest rush-hour and weekend peaks which are the hours that matter most for rebalancing.

- After choosing alpha I could have retrained on train + validation to use more recent data.

- Rebalancing happens per station, and this dataset has no station information.



## Generative AI use disclosure

I used Claude (Anthropic). It proposed the dataset and the options for the target and features; I chose the hour x working-day and temperature-curve features, and it added the growth-trend feature after the ablation showed systematic under-forecasting. It helped me with the code in "features.py", "train.py" and "app.py".