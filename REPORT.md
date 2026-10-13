# Assignment 1 Report

- **Name**: Armand Hubler
- **Student ID**: 19095
- **Email**: ahubler.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

This project does not rely on any superficial dataset available on Kaggle. Instead, I believe the main challenge in data science lies in correctly understanding your business problem and your data to create value and deliver either useful information or justify decisions.
It all started from a previous hackathon in which I competed, where my team was tasked to predict hourly energy production and map plant codes to anonymised bids.
For this project, I wanted to think of another player in the market, the battery. In Spain, over half of the production comes from renewable plants that are non-dispatchable.
The data in this project was scraped from 3 sources:

- OMIE: Iberian market operator. It provides the day-ahead prices for Spain and Portugal and the aggregated bid curves of the joint Iberian day-ahead market (MIBEL).
- ESIOS: the platform of REE, the Spanish grid operator. This provides us with demand forecasts, wind and solar forecasts and imbalance prices.
- ENTSO-E: European transparency platform containing the energy mix in Spain and the transfer capacity from France to Spain.

Due to the high data volume and the limited requirements, the data was first fetched before getting aggregated, normalised and re-encoded to fit in less than 20 MB, a relatively small dataset size to train a neural network.
The final shape looked as follows (January 2019 to September 2026):

- OMIE: 67,847 rows × 70 columns for the bid curves, one row per hour (cumulative supply and demand MW at fixed price levels), plus the Spanish and Portuguese prices.
- ESIOS: 5 series (demand, wind and solar forecasts, up and down imbalance prices), one value per hour.
- ENTSO-E: 5 generation series (gas, coal, run-of-river hydro, reservoir hydro, pumped storage) and the France to Spain transfer capacity, one value per 15 minutes or per hour.
  The OMIE prices, ESIOS and ENTSO-E series are stored together in long format: 1,601,382 rows × 3 columns (datetime, series, value), one row per value of a series at a given time. For training, everything is aggregated into 2,090 rows (one row = one day) and 43 features.

This data was chosen due to its relevancy with the chosen topic as well as its high volume, granularity and variety, making it ideal for feature engineering.

## Business / real-life framing

The business case applies to the Spanish energy market, which consists of energy producers (wind farms in Castilla y León, solar farms in Andalusia, or even nuclear, hydro and gas plants), retailers and trading companies (Iberdrola, Endesa, Moeve) that buy and sell on behalf of clients, big industrial consumers that bid directly on the market, and newer players such as battery operators.
Each economic agent takes rational decisions to defend its own interests. Renewable energy producers are forced to sell their production, sometimes at negative prices due to low demand and high competition, which causes large price differences and encourages the arrival of batteries.
According to [RenewableDB](https://renewabledb.com/battery-spain), batteries usually operate in 4-hour cycles. I model one battery of 1 MW, which can therefore store 4 MWh.
Each day, it can charge for 4 hours and discharge at a higher price for 4 hours later.
There is however a 15% energy loss in the round trip, according to [NREL](https://atb-archive.nrel.gov/electricity/2019/index.html?t=st).
I therefore modelled the target variable as follows:
<br> `0.85 × mean(4 highest) − mean(4 lowest) > €20/MWh`, where the 4 highest hours come after the 4 lowest, and €20 is my estimate of the battery wear caused by one cycle. I asked an LLM to look for the operating cost of a battery. $7–17/MWh wear cost was mentioned in [this paper](https://arxiv.org/pdf/1904.07771) (Xu et al.), which I rounded up to be more conservative.<br>

The data is time-dependent, which requires careful attention to the split. When tuning the model, it is important to take this into account and use appropriate tools like TimeSeriesSplit.
Ignoring it would cause a look-ahead bias and add future information that is not known at inference time.
Regarding the decision threshold, it should be driven by the profit generated. The data is heavily imbalanced, so it would inflate regular metrics like accuracy.
Opportunity cost arises when the model makes a false negative (type II error), because we miss a profitable day. Although recall would be a good solution, I opted for log-loss, which rewards good probabilities rather than just correct labels.
The threshold itself should be set on profit, but this only works if the probabilities are calibrated (a 70% call should be profitable about 70% of the time), which can be checked with the expected calibration error (ECE) before choosing it.

## Data preparation & feature engineering

The first step was to only use information known at decision time.
The auction for day D closes at 12:00 on D-1, so each series was shifted accordingly:

- 0 days for forecasts,
- 1 day for prices and bid curves
- 2 days for actual production.
  Since October 2025, prices are published every 15 minutes, so everything was averaged to hourly.
  Constant series such as offshore wind (which does not exist in Spain but are available in the dataset) duplicates and series with very little signal were dropped.
  The bid curves which is very large (about 1.5 billion rows) were reduced to the MW offered at fixed price levels.
  Training starts in 2021, as the market before had very different spreads.
  Moreover, extreme values like the 2022 gas crisis were winsorized (consist of clipping to a certain percentile), missing values filled with the median, and features standardised.
  It is important to note that they have all been fitted on train and infered on validation/test set to avoid any leakages.
  The 43 features are daily averages:
- calendar (weekday, month, holidays)
- lagged prices and forecasts
- z-scores against recent history (prices are not stationary so comparing each value to its own recent mean and spread makes it comparable across years),
- fundamentals like residual load and renewable share
- bid-curve features like the price implied by yesterday's curve. We estimate how much power will be needed today and read its price on yesterday’s supply curve. For example, if 25 GW cleared yesterday and today’s forecast has 1 GW more demand but 3 GW more wind and solar, about 23 GW is needed. If yesterday’s sellers offered 23 GW at €40/MWh or less, the implied price is €40.

## Modeling: three implementations, one model

I used logistic regression, as the target is binary (cycle or not) and it gives a probability that can be compared to a threshold.
It is also simple and easy to interpret, which fits a first model. I also added a naive model whose prediction depends on yesterday's spread level.
In total, my train file has 3 models (scikit-learn, a manual PyTorch loop and a standard `nn.Linear` with the Adam optimizer).
Each one was slightly tuned with Optuna on the same time-based CV, and tested on the same 418 days (August 2025 to September 2026).

| Model            | Threshold | Accuracy | ROC AUC | Log-loss | ECE (raw → calibrated) | Profit (€/MW) |
| ---------------- | --------- | -------- | ------- | -------- | ---------------------- | ------------- |
| scikit-learn     | 0.50      | 0.95     | 0.91    | 0.15     | 0.025 → 0.040          | 103,742       |
| Manual PyTorch   | 0.55      | 0.95     | 0.90    | 0.17     | 0.026 → 0.048          | 103,742       |
| Standard PyTorch | 0.55      | 0.95     | 0.91    | 0.16     | 0.069 → 0.045          | 103,742       |
| Always cycle     | –         | 0.95     | 0.50    | 0.36     | 0.053                  | 103,742       |
| Naive            | –         | 0.93     | 0.64    | 0.50     | 0.072                  | 102,785       |

We can observe that the metrics are very similar, so the three models agree, with all of them outperforming the naive baseline.
I experimented with the manual model's loss function, and it ends up first on profit and accuracy, although only by a small margin.
Instead of treating all errors the same way, each day is weighted by how far its spread is from the €20 cost. For example, a day with a €100 spread gets a weight of 80, while a day with a €22 spread gets a weight of 2: getting the first day wrong costs 40 times more, which matches the money really lost.
The manual loop updates the weights with plain gradient descent while the standard version uses the Adam optimizer, so even on the same data they do not necessarily converge to exactly the same weights.
The regularisation in the standard PyTorch model did not seem to help (its log-loss is worse, 0.18).

## Limitations & next steps

- The Spanish market changed between my train/validation and test sets: 73% of days were profitable before August 2025, against 95% after. This led me to review my target to better take operational costs into account.
- The gap between the naive PnL and the best model is only €960/MW over the test period, so the real challenge is not the decision itself but the timing (which hours to charge and discharge), which could be a natural next step.
- I tuned the threshold and calibrated the probabilities on the validation folds (244 unprofitable days out of 1,390), leaving the test set untouched. The raw probabilities were underconfident (about 70% predicted against 82% observed, ECE 0.11–0.16), so I first applied Platt scaling and then picked the most profitable threshold, which came back to about 0.5.
- Calibration also did not carry over well. It improved the standard PyTorch ECE but worsened the other two, because it learned a correction for a period with 82% profitable days and the test period has 95%. Since calibration changes neither the ranking nor the test profit, and worsens log-loss for two of the three models, I did not carry it into the dashboard.
- This pipeline relies on three assumptions around the target (85% efficiency, €20 wear cost and a 4-hour battery), and it could be worth running a Monte Carlo simulation on these parameters.
- Finally, battery operators can also be paid for helping the grid stay balanced (balancing services), which would increase the PnL.

## Generative AI use disclosure

I used Claude Code with a custom harness built with Oh-My-Pi. It helped me write the scrapers, data loaders, plotting functions, the Gradio dashboard and the repo structure, following the constraints I set. I also used it to challenge my target definition, find sources on battery costs.
