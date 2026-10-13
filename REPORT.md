# Assignment 1 Report


- **Name**: Marcela Mendes Gimenes Funabashi
- **Student ID**: 20251
- **Email**: mmendesgimen.ieu2022@student.ie.edu 
- **Group**: BBADBA 5A
## Dataset

The dataset used in this project comes from the Brazilian Central Bank and was accessed through the python-bcb interface, which queries public datasets provided by the BCB API. Because the underlying database is extremely robust—containing a lot of financial time series, including indicators from other Mercosur countries—I initially fetched only the macroeconomic variables relevant to domestic inflation dynamics: the official IPCA inflation rate, the IPCA-15 preview, the SELIC target interest rate, the USD/BRL exchange rate, the ICBr commodity price index, and the IBC-Br economic activity index. Each row in the dataset represents a month of aggregated macroeconomic activity and financial market prices, spanning from January 1999 through October 2026. In its preprocessed form, the dataset contains 334 rows (months) and the 6 features previously stated (columns). I chose this dataset primarily because it matched the use case I wanted to investigate, but it also satisfies the file-size guideline (ensuring that the pipeline can retrain quickly from scratch), and it would allow me to explore different areas of feature engineering, especially those that pertain to the time series.
## Business / real-life framing


This model serves an asset management firm seeking an early-warning system to protect nominal fixed-income portfolios against inflation-driven bond devaluations. The target variable is defined as a binary indicator showing whether the official IPCA inflation rate in month (t+1) exceeds 0.50%, which is an economic threshold representing an annualized inflation rate above 6.17% that breaches the BCB's current upper tolerance ceiling and rapidly decreases real portfolio yields. When the model signals high risk, the fund manager automatically executes derivative hedging positions (such as buying IPCA futures) to shield portfolio capital. On the other hand, when low risk is predicted, the portfolio remains unhedged to avoid unnecessary transaction costs.

Therefore, this context requires a chronological, time-based split–training on 2003-2017, validating on 2018-2020, and testing on 2021 onwards–rather than a random split. This is because a random split would introduce severe temporal data leakage, since information from the future months would contaminate past training features. A strict time-based split also allows us to stress-test the model on the post-2021 global inflation shock and evaluate whether it could successfully extrapolate and safeguard capital in unforeseen circumstances.

Because the financial loss of a missed inflation spike (false negative that leaves millions in bond holdings exposed to market drawdowns) exceeds the minor transaction fee of a false alarm (false positive where hedging was purchased unnecessarily), the pipeline is driven primarily by Recall, rather than simple accuracy. The asymmetric business risk directly dictates how the decision threshold is calibrated. Instead of defaulting to a 0.5 classification cutoff, the decision threshold is derived from minimizing a total business cost metric, ensuring that the model is sensitive enough to catch incoming spikes. In other words, because the primary business goal is minimizing missed inflation spikes (False Negatives), Recall is reported as the main operational diagnostic. However, rather than tuning the decision threshold on validation data to maximize Recall directly, the threshold was set structurally at $\tau = 0.25$ using expected cost minimization.

## Data preparation & feature engineering


The first thing I checked was the time scope of each of the regressors, along with whether there is any missing data. I found that metrics were collected at different initial dates, such as ipca15 in May 2000 while IBC-Br  in January 2003. Some metrics are also updated more rapidly than others, such as SELIC, which already has data available for October 2026. Since IPCA is our target variable, I removed all rows after the last valid IPCA date. Within the main timeframe, there was no missing data, which makes sense since these are crucial government statistics. When time-plotting the variables and color-coding them by train, validation, and test splits, I saw that the validation set covers the lead-up to COVID-19 and its hardest year (2020), while the test set covers the post-COVID recovery years. This could be a limitation for how well the models handle totally unexpected events, but it lets us test how resilient the models are under extreme economic stress. Looking at the plots, I also noticed sharp spikes that line up with real events, like a big drop in economic activity (IBC-Br) in 2020 caused by COVID. Lastly, looking at the raw numbers showed that exchange rates (usdbrl), commodity prices (icbr), and economic activity (ibcbr) are levels rather than rates. This is a problem because levels mostly track time passing rather than economic changes, so they had to be converted into percentage rates.

Looking at the target variable, there is no major class imbalance overall, but the base rate (the chance of target = 1) changes across different time periods. For example, it was low (0.250) in the 2018–2020 validation period while the economy slowed down during COVID, but it rose to 0.471 in the test set due to post-COVID supply issues and global commodity price jumps. For a simple naive baseline, I calculated persistence, which checks how often the next month is high based on whether the last published month was high. A logistic regression model is only useful if it beats this simple guess. Plotting the target by month showed clear seasonality, meaning early months in the year have higher rates of high inflation because of factors like school tuition resets, minimum wage increases, summer holidays, and Carnaval.

I designed all features using data strictly from the training period so there would be no data leakage. To decide which features to keep and change, I picked one metric for each main economic idea driving inflation. Features that started as raw levels—like exchange rates, commodity prices, and economic activity—were turned into percentage rates, and every feature only uses data that was already published by the decision date. I added 11 month dummy variables (12 - 1) to capture seasonality without causing perfect multicollinearity. In total, this gave 17 features. Rolling the data to match dates created NaN values at the very start of the sample, so I removed those incomplete rows. Checking the correlation between features and the target showed that the highest correlation was around 0.4, which is fairly low. There is also little overlap between features. Finally, because the numeric features are on totally different scales, standardization is necessary so that gradient descent in the two PyTorch models can converge properly at the learning rate, and the scaler was fitted strictly on the training data.
## Modeling: three implementations, one model


a. Model Selection & Preprocessing

To predict whether next month's inflation will exceed a growth of 0.50%, binary logistic regression was chosen across all 3 implementations. This is because the business context involves a binary decision (hedge vs not hedge), rather than trying to estimate a specific measure. Numeric features were standardized using parameters fitted strictly on training data to prevent future data leakage, while the 11 binary monthly dummy variables were left unscaled so their regression coefficients directly measure the change in log-odds of. a spike for that month vs January.

b. Decision Rule derived from Business Costs

The classification decision threshold was fixed based on the underlying economics of portfolio risk management. A missed inflation spike (False Negative) leaves millions in unhedged bond holdings exposed to severe market drawdowns, whereas a false alarm (False Positive) incurs only a minor hedging transaction fee. Assuming a missed spike costs 3x more than a false alarm ($\text{COST}_{FN} = 3 \cdot \text{COST}_{FP}$), hedging is triggered whenever the predicted probability  $p$ makes not hedging more expensive in expectation:

$$p \cdot \text{COST}_{FN} > (1 - p) \cdot \text{COST}_{FP} \quad \Longleftrightarrow \quad p > \frac{\text{COST}_{FP}}{\text{COST}_{FP} + \text{COST}_{FN}} = \frac{1}{1 + 3} = 0.25$$

This 0.25 threshold explicitly prioritizes recall over precision. Fixing the threshold from business costs avoids tuning it on validation data, where only 9 high-inflation months occurred and threshold tuning would fit random noise. Because this decision relies on well-calibrated probabilities, test-set log loss is reported alongside business cost.

c. Naive Baseline

The baseline benchmark I chose was the persistence rule: predict high inflation in month t+1 if the last published month (t-1) was above 0.50%. Because monthly inflation exhibits momentum, repeating the most recent observation is the simplest rule of thumb a manager could follow. A logistic regression model is only valuable if it beats this simple persistence rule.

d. Implementations

All 3 models minimized the exact same L2-regularized mean log loss objective, leaving the bias parameter unpenalized.

* Scikit-learn (LogisticRegression): minimizes log loss scaled by inverse regularization strength $C$. The parameter C was selected using validation log  loss rather than validation business cost because log loss rewards calibrated probabilities across every single validation month rather than ranking models on only 9 positive events.

* Manual PyTorch Loop: built using raw PyTorch tensors, explicit matrix multiplications, autograd gradient tracking, and a hand-written gradient descent update step. The loss function was scaled by $1 / (C \cdot n)$ to match scikit-learn's exact loss surface. The learning rate was tuned to find the fastest convergence rate without diverging.
* Standard torch.nn.Module + torch.optim: built using nn.Linear(input_dim, 1) and nn.BCEWithLogitsLoss(), updated via torch.optim.SGD. The L2 penalty was added explicitly to the total loss term rather than using SGD's weight decay parameter, ensuring the bias parameter remained unpenalized to match scikit-learn.

e. Model Results & Agreement

The table below evaluates all 3 implementations against the naive persistence baseline on out-of-sample test set (2021 onwards):

| model | log_loss | roc_auc | recall | precision | missed_spikes | false_alarms | business_cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| persistence (baseline) | NaN | NaN | 0.688 | 0.688 | 10.0 | 10.0 | 40.0 |
| scikit-learn | 0.651 | 0.745 | 1.000 | 0.471 | 0.0 | 36.0 | 36.0 |
| manual PyTorch | 0.651 | 0.745 | 1.000 | 0.471 | 0.0 | 36.0 | 36.0 |
| nn.Module + optim | 0.651 | 0.745 | 1.000 | 0.471 | 0.0 | 36.0 | 36.0 |

All 3 implementations reach the same mathematical solution. Because binary logistic regression optimizes a convex loss surface, the gradient descent and second-order optimization algorithms converge to the same global minimum.  Furthermore, they have the same objective, the bias was unpenalised in all 3, and training run to convergence.

On the test set, all three models yield exact identical results across every metric: a log loss of 0.651, an ROC-AUC of 0.745, a recall of 1.000, 0 missed spikes, and 36 false alarms, lowering total business cost from 40.0 down to 36.0. This complete numerical agreement confirms that scikit-learn, manual PyTorch, and torch.nn.Module all solved the exact same regularized optimization problem to full convergence.

The models achieve a perfect recall of 1.000 because they decide to hedge in every single month of the test set. This occurs due to three factors working together:

* Under a 3:1 cost ratio (where a missed spike costs 3× a false alarm), hedging is worth it whenever the predicted risk of an inflation spike exceeds 1/4 = 25%. The model only decides "don't hedge" if it is more than 75% sure that next month will be calm.
* About 41% of training months and 47% of test months were above 0.50% inflation. Before looking at any feature, a typical month carries a risk of around 40%. To skip a hedge, the model's features must push a month's risk from 40% down below 25%, which would require strong evidence of a safe month.
* Validation (2018–2020) showed that confident models with larger C values performed worse when the economic regime changed. Strong regularization shrinks all coefficients toward zero, keeping predicted probabilities close to the baseline average. On the test set, predicted probabilities only range from 0.32 to 0.59, so no month falls below 0.25.

In a way, this behaves like insurance. If a damaging event happens almost half of the time, insurance is cheap compared to the potential loss, and you cannot reliably predict the safe months. In cases like this, staying continuously insured is the most rational decision. The models would only become selective if inflation spikes became rarer, missed spikes were cheaper relative to false alarms, or the features provided enough signal to confidently call specific months safe.

## Limitations & next steps


The primary limitation of this framework is that its optimal policy collapses into a static "always hedge" rule, making the machine learning model operationally redundant despite outperforming the baseline in total business cost. Because a high baseline spike frequency (~41–47%) and an aggressive 3:1 cost structure establish a low 25% decision threshold, the model’s heavy regularization ($C = 0.003$) keeps predicted probabilities between 0.32 and 0.59—never low enough to confidently declare a month safe despite achieving a 0.745 ROC-AUC. To turn this into a truly selective system, future work should maybe incorporate other indicators, such as forward-looking market indicators, switch from binary classification to continuous IPCA forecasting to calculate exact dollar-denominated losses, and explore non-linear architectures (like XGBoost or Hidden Markov Models) that can capture subtle calm windows without over-smoothing probabilities during economic shifts.

In terms of pre-training, the process of selecting the features can be a significant limitation that impacts the overall performance of the models, and is something that should be taken into account for better pipelines (e.g. keeping more regressors from the start). Other limitations are that the validation period is a different period with only 9 spikes, the 0.50% threshold stay fixed while the inflation target changes over time, and that the results are very dependent on the cost ratio. These are significant considerations that would make for a more robust model training and evaluation.

## Generative AI use disclosure

Claude Code was used to assist with code writing and debugging and proposing design options. I chose the framing, features, costs and baseline, questioned and analyzed the results, and directed the changes.
