# Assignment 1 Report

- **Name**: Jean Carlos
- **Student ID**: 19832
- **Email**: jcarlos.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

The IBM Telco Customer Churn sample from Kaggle has 7043 rows (1 per customer) and 21 columns: demographics, account details, 9 services and the target Churn (left within the last month; 26.5% did). I chose it because retention targeting is a real business decision and its raw columns need feature engineering (a curved tenure effect, ordered contracts, duplicate columns).

## Business / real-life framing

Each month, a retention team offers 1 free month to customers whose predicted churn probability exceeds a threshold. A free month only costs REVENUE if the customer would otherwise have paid: it is wasted on loyal customers but free for churners who leave anyway. I assume a persuaded churner stays 12 more months (plausible, as churners had already stayed 18 months on average) and that the offer persuades 30% of the churners it reaches.

Contacting a churner therefore gains 0.30 × (12 − 1) = 3.3 months of their bill on average, while contacting a loyal customer wastes one. Contacting pays when p × 3.3 > (1 − p), so above p = 0.233; as both sides scale with the bill, this applies to every customer. The business number is the campaign's expected net value versus no campaign ($0). This drives 4 decisions:

1. Target: The 11 customers with 0 months of tenure cannot have churned yet, so I removed them.
2. Split: 1 dateless snapshot with 1 row per customer makes a time-based split unnecessary; a 60/20/20 split stratified on Churn keeps the churn rate equal across sets.
3. Regularisation is chosen by validation log-loss, because the 0.233 rule needs accurate probabilities.
4. Threshold is chosen on validation by maximising EXPECTED NET VALUE, as that is what the team gains or loses. It is 0.24, close to the theoretical 0.233, confirming accurate probabilities. The test set only reports results.

## Data preparation & feature engineering

TotalCharges is blank only for the removed tenure-0 customers, so no imputation was needed. Preprocessing is fitted on training data only, and every column is standardised so the L2 penalty treats them equally. Each feature decision was tested with 5-fold cross-validation on training and validation data. The final 17 columns reach ROC-AUC 0.8489 and log-loss 0.4120, versus 0.8427 and 0.4201 for the 30 raw one-hot columns, winning all 5 folds.

1. log_tenure: churn falls steeply then flattens (62% after one month, 10% after 4 years), and the logarithm makes this near-linear. It is the largest gain (raw tenure: 0.8418).

2. contract_months (1, 12 or 24): each extra month cuts the odds of churning by a similar 12–15%, so 1 column matches two one-hot columns.

3. electronic_check: electronic-check customers churn at 45%, the other three at 15–19%, so 1 flag replaces all 4 and scores slightly better (0.8489 vs 0.8487).

4. One 0/1 column per service: plain one-hot encoding gives each internet add-on a "No internet service" column identical to InternetService_No, and MultipleLines a "No phone service" column duplicating PhoneService. Duplicates let many sets of weights fit equally well, so coding these answers as 0 removes all 7 without losing information.

I dropped MonthlyCharges (R² 0.9988 from the services) and TotalCharges (correlation 0.9996 with tenure × bill), as they repeat existing information, and gender, which has no predictive value (26.2% vs 26.9% churn). An add-on count, a tenure ≤ 3 months flag and a fibre × month-to-month flag added nothing.

## Modeling: three implementations, one model

Logistic regression suits a binary target and gives the probability the decision rule needs. All 3 implementations minimise average log-loss plus an L2 penalty on the weights, but not the bias, which only sets the overall churn level. C = 0.1 was chosen by validation log-loss and converted for PyTorch as λ = 1 / (C × n training customers). The manual version follows Session 5: raw tensors starting at 0, hand-written sigmoid and log-loss, autograd and a hand-written update. The standard version uses nn.Module with Adam from random weights, adding the penalty by hand because weight_decay would also penalise the bias. As the penalty leaves a single best solution, reaching it from different starts and optimisers is a real test of agreement. Learning rates (1.0 for gradient descent, 0.01 for Adam) were chosen by lowest final training loss.

Test-set results (1,407 customers, threshold 0.24):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Log-loss | Expected net value |
|---|---|---|---|---|---|---|---|
| scikit-learn | 0.7292 | 0.4947 | 0.8690 | 0.6305 | 0.8545 | 0.4069 | $55,268 |
| Manual PyTorch | 0.7292 | 0.4947 | 0.8690 | 0.6305 | 0.8545 | 0.4069 | $55,268 |
| Standard PyTorch | 0.7292 | 0.4947 | 0.8690 | 0.6305 | 0.8545 | 0.4069 | $55,268 |
| Naive baseline | 0.2658 | 0.2658 | 1.0000 | 0.4200 | 0.5000 | 0.5791 | $25,846 |

The model contacts 87% of churners, and 49% of those contacted churn; the low threshold is deliberate, as a missed churner costs 3.3 months of revenue and a wasted offer one. The baseline gives everyone 26.6%, above the threshold, so it contacts everyone and ranks no better than chance. The model also beats contacting all month-to-month customers ($51,032).

The 3 implementations give the same weights, probabilities and training loss (0.406050965), apart from rounding-level differences below 0.0000001. An earlier version with duplicate columns and no PyTorch penalty did not: MonthlyCharges weighed +0.47 in PyTorch but −0.21 in scikit-learn.

## Limitations & next steps

The data is 1 snapshot, so I cannot test the model on future customers; monthly data would allow training on earlier months and testing on the next. Features are recorded with the label, so customers already leaving (for example, cancelling add-ons) may leak signal; I would use only earlier data. The 30% save rate is an assumption, but the model beats both simple policies for any rate from 10% to 50%, which also covers persuaded customers staying 5 to 19 months; a randomised trial would measure the real rate. Contact and serving costs are ignored, and capacity is assumed unlimited (47% are contacted); with fixed capacity I would contact only the riskiest. The modest 8% gain over the month-to-month rule makes behavioural data (usage, complaints) the best next step. The synthetic IBM data may also not reflect a real company; I would validate the model on real customer data.

## Generative AI use disclosure

I used Claude entirely for the plots, the Gradio dashboard, and app.py, but when it came to the train.py code, I explored the data myself (churn rate by category and correlations between columns), used Session 5 for inspiration for Manual PyTorch, came up with the feature ideas (log_tenure, contract_months, YES/NO service columns), and then asked Claude whether I had missed anything. The feature set, business framing, split strategy, and choice of threshold metric are my decisions. Finally, for the report, I wrote the analysis and conclusions myself and used Claude to improve readability, help shorten some sections, and create the Markdown table.