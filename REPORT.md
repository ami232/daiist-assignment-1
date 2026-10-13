# Assignment 1 Report

- **Name**: Juan Felipe Ortiz Lejarde
- **Student ID**: 20417
- **Email**: jortiz.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

I picked the IBM Telco Customer Churn dataset from
[IBM's GitHub repository](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
saved in `data/telco_churn.csv`. I chose this dataset because it has less than 10k rows which is within the
expected value and it has valuable information that could build a good predictive model. Here is an overview
of the features:

7,043 rows x 21 columns. One row = one customer of a fictional US telecom provider at a single snapshot in
time; there is no date column. Example: the first customer in the file.

| Group | Columns | Kind of data | Example (customer `7590-VHVEG`) |
|---|---|---|---|
| Identifier | `customerID` | Unique ID | 7590-VHVEG |
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Categorical (Yes/No, Female/Male, 0/1) | Female, 0, Yes, No |
| Account | `tenure` | Numeric (months as a customer, 0-72) | 1 |
| | `Contract`, `PaperlessBilling`, `PaymentMethod` | Categorical | Month-to-month, Yes, Electronic check |
| Charges | `MonthlyCharges`, `TotalCharges` | Numeric ($) | 29.85, 29.85 |
| Phone services | `PhoneService`, `MultipleLines` | Categorical | No, No phone service |
| Internet services | `InternetService` | Categorical (DSL / Fiber optic / No) | DSL |
| | `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Categorical (Yes / No / No internet service) | No, Yes, No, No, No, No |
| **Target** | `Churn` | Categorical (Yes/No) | No |

The goal is to predict whether a user will churn. This is a very interesting task because of the inherent
class imbalance (1,869 churners, 26.5%, vs 5,174 non-churners, 73.5%) and its potential to gain extra
revenue that otherwise would be lost if the model performs well.

## Business / real-life framing

Losing a customer is very costly for any company, and companies have been trying to solve this problem for
a long time. The first step to avoid the lost revenue is to understand why people leave, and then act before
they do so the company prevents as much of it as possible.

A target of 1 means the customer left (`Churn = Yes`). Each of those customers is something the company can learn
from, so that it can stop others from going down the same path. Once the model is trained, the company has valuable
information it can use to avoid churn. A company that does nothing will inevitably lose customers. A company
that calls everyone wastes money on people who weren't even planning on leaving, which lowers what the
campaign earns. It could even push some of them to start thinking about churning.

These are the assumptions I made. The save rate of 0.30 comes from Verbraken, Verbeke & Baesens (2013),
"A Novel Profit Maximizing Metric for Measuring Classification Performance of Customer Churn Prediction
Models", *IEEE Transactions on Knowledge and Data Engineering*, where it is the default offer-acceptance rate.

- Call cost: $50 per customer called.
- Horizon: a customer who leaves takes 12 months of their monthly bill with them (MonthlyCharges x 12).
- Save rate: the offer keeps 30% of the called customers who would otherwise leave.

After studying the data, I set a gain-based threshold instead of 0.5. Compared with doing nothing, calling a
customer who would leave gains on average 0.30 x MonthlyCharges x 12 - $50 (about +$220 for a $75/month
customer), while calling a customer who would stay loses $50. The threshold that gives the highest net gain on
the validation set is 0.28 for all three models. On the test set, this gives a net gain of $53,450, compared
with $39,189 at a 0.5 threshold and $30,941 when calling everyone.

I used a stratified split to account for the class imbalance: it keeps the same churn rate in each split
(26.5% in train, validation and test). I split the dataset into three parts, train (60%, 4,225 customers),
validation (20%, 1,409) and test (20%, 1,409), so that each one has a separate job. The models learn their
weights on train, and every choice (the regularisation strength C, the features I kept and the threshold) is
made on validation. The test set stays hidden until the very end and is used only once, for the final numbers.
If I made choices by looking at the test set, its results would be too optimistic, so keeping it untouched is
what tells me whether the model really generalises to customers it has never seen.

I didn't use a time-based split because the dataset is a single snapshot in time and it doesn't have dates,
so there is no way to know which customers came earlier or later.

## Data preparation & feature engineering

When I first looked at the data, I found 11 missing values in `TotalCharges`. They belong to customers
with a tenure of 0, who had never been billed. I fixed this by setting their total to their `MonthlyCharges`,
assuming it was their first month.

Then I cleaned the variables to reduce redundancy. For example, the six internet add-on columns (such as
`OnlineSecurity`) had a third value, "No internet service", which only repeats `InternetService = No`. I
turned it into a plain "No", and did the same with "No phone service" in `MultipleLines`.

Looking more closely, I found a very important insight in the relationship between tenure and churn: newer
customers are much more likely to churn than older ones. Churn is 62.0% in the first month, 35.9% at 7-12
months and only 6.6% after five years, so it drops fast at the start and then flattens. A linear relationship
would not capture all of the information in this variable, so I took the log of tenure, log(1 + tenure). On
the validation set, this improved the log loss from 0.4018 to 0.3959 and the PR-AUC from 0.677 to 0.690, a
small but consistent gain.

I then used ablation to check whether I had included variables that weren't really necessary, and found
that two of my engineered features, `charge_diff` (the change in a customer's price over time) and
`fiber_x_monthly` (fiber optic and a month-to-month contract together), had a negligible effect on the final
prediction: removing either one changed the validation metrics only in the fourth decimal (see the table
below). I decided to drop them, prioritising model explainability over a tiny difference in the metrics.

| Variable tested | Decision | Reason |
|---|---|---|
| Raw `tenure` | Replaced by `log_tenure` | Churn falls fast in the first months and then flattens; the log fits that shape better (validation log loss 0.4018 vs 0.3959) |
| `charge_diff` | Dropped | Weight ~0 (-0.015); its U-shaped pattern is explained by tenure, price and contract, which are already in the model |
| `fiber_x_monthly` | Dropped | Weight ~0 (+0.043); on the log-odds scale the two risks do not add up to more than their sum, so the interaction adds nothing |
| `OnlineBackup`, `DeviceProtection`, `StreamingTV`, `StreamingMovies` as separate columns | Only counted through `n_addons` | Giving each its own weight did not improve validation, and together with `n_addons` they would be exact duplicates |

I used a standard scaler for the numerical features, so that they are all on the same scale: the
regularisation then treats every feature equally, and gradient descent converges faster. For the categorical
features I used a one-hot encoder, which creates n - 1 columns for a feature with n levels, since having all
columns equal to 0 already means the base level. Both were fitted on the training set only and then applied
to validation and test.

Final model: 4 numeric features (`SeniorCitizen`, `log_tenure`, `MonthlyCharges`, `n_addons`) and 11 categorical
ones (`gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`,
`TechSupport`, `Contract`, `PaperlessBilling`, `PaymentMethod`), giving 19 columns after encoding. I kept both
`gender` and `SeniorCitizen` for consistency, since both are sensitive attributes and both have near-zero
weights (+0.041 and +0.059).

## Modeling: three implementations, one model

I used a logistic regression because this is a classification problem: the target is binary (the customer
churns or not), and the model outputs a probability of churning that the threshold turns into a decision.

I also used L2 (ridge) regularisation, which keeps the weights from growing too large and helps with
multicollinearity: `MonthlyCharges` is almost a combination of the service columns, and the penalty keeps
those weights stable. I chose the regularisation strength C by testing several values ({0.01, 0.1, 1, 10})
and keeping the one with the lowest log loss on the validation set (C = 10).

These are the three implementations I used, all trained on the same split and minimising the same objective
(mean log loss + the same L2 penalty, bias not penalised, float64):

1. **scikit-learn:** `LogisticRegression(C=10, solver="lbfgs")`.
2. **Manual PyTorch:** raw tensors, a hand-written log loss, `loss.backward()` for the gradients and a
   hand-written gradient descent step (learning rate 2.0, 30,000 epochs).
3. **Standard PyTorch:** an `nn.Module` with one `nn.Linear` layer, `BCEWithLogitsLoss` and the
   `torch.optim.LBFGS` optimiser.

Results on the test set (1,409 customers, threshold chosen on validation; net gain is compared with doing
nothing):

| Model | ROC-AUC | PR-AUC | Threshold | Precision | Recall | Called | Net gain | Net gain at 0.5 |
|---|---|---|---|---|---|---|---|---|
| scikit-learn | 0.840 | 0.655 | 0.28 | 52.0% | 78.6% | 565 | $53,450 | $39,189 |
| Manual PyTorch | 0.840 | 0.655 | 0.28 | 52.0% | 78.6% | 565 | $53,450 | $39,189 |
| Standard PyTorch | 0.840 | 0.655 | 0.28 | 52.0% | 78.6% | 565 | $53,450 | $39,189 |
| Baseline: call nobody | 0.5 | 0.265 | - | - | 0% | 0 | $0 | - |
| Baseline: call everyone | 0.5 | 0.265 | - | 26.5% | 100% | 1,409 | $30,941 | - |

Compared with the best naive baseline (calling everyone), my model makes about $22,500 more ($53,450 vs
$30,941) while making 60% fewer calls (565 instead of 1,409).

## Limitations & next steps

1. **Size of the dataset.** With 7,043 customers the dataset is very small, and the three-way split reduces
   it further: the test set has only 1,409 customers (374 churners), so a few customers can move the net gain
   by thousands of dollars. More data, or cross-validation instead of a single validation set, would make the
   results more reliable.
2. **The costs are assumed.** The call cost, the 12-month horizon and the 30% save rate are not measured on
   this company. I would need real data from a retention campaign (calling a random group and comparing its
   churn with a similar group that was not called) to estimate them.
3. **One threshold for everyone.** The value of saving a customer depends on their bill, so I could set
   different thresholds for different customers, for example calling when p x 0.3 x 12 x MonthlyCharges is
   above $50. Clustering customers into segments and setting a threshold per segment would also help.
4. **Only a linear model.** I would like to test more flexible models such as Random Forest or XGBoost, to see
   if they can capture patterns that logistic regression misses and improve the performance.

## Generative AI use disclosure

AI was used to generate parts of the code, pipeline and the gradio dashboard. All design choices were made by me. AI was also ran to run multiple tests to make ensure the robustness of the pipeline. 