# Assignment 1 Report

- **Name**: Tessa Correig Martra
- **Student ID**: 19614
- **Email**: tcorreig.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

I'm using the IBM Telco Customer Churn dataset with 7,043 rows and 21 columns originally. Each row represents one customer of a telecom company in California, with demographic info (gender, senior citizen status, partner, dependents...), account details (tenure, contract type, billing method...), subscribed services (phone, internet, security, streaming...), billing amounts and a binary target: whether the customer churned (left the company) or not. The `customerID` column was excluded from the model's 20 training features since it has no predictive value, but was kept alongside the pipeline as reference metadata and then split in parallel with the train/test data so predictions can be traced back to a specific customer in the dashboard.

Data set: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

I picked this dataset because it fits the size guideline and has a genuinely imbalanced binary target (with a 26.5% churn rate) that requires handling rather than a 50/50 split and has enough categorical columns that feature engineering meaningfully changes what the model can learn, rather than being a dataset where raw columns already say everything.

## Business framing

**Scenario**: The company is seeing that the churn rate is higher that what they were expecting. At first they thought that the retention team could offer customers that are at risk a discount or different promotions, but at the end they noticed that they couldn't do this for all 7,000 customers they had. This would cost more money than what they would get and also staff time. The model ranks customers by churn risk so the team can specifically target the customers that are more likely to churn rather than selecting the customers that will get the discounting measures, among others, randomly.

This framing drove three concrete pipeline decisions:

1. **Target definition**: `Churn` (Yes/No) was encoded directly to 1/0 since the dataset already defines churn at the customer level.

2. **Split strategy**: I used a random 80/20 split, stratified by `Churn`. The model predicts churn across different customers at a single point in time. Each row is one customer's profile and outcome in this one quarter, with no repeated observations of the same customer across periods. Because of this, there's no future leaking into training risk that a time-based split would normally guard against, since there's no time dimension linking rows together in the first place. The real risk here is class imbalance. With only 26.5% churners, a plain random split could accidentally produce a training set and test set with very different churn rates, making the model's evaluation misleading. That is why I stratified in order to fix this, where both the train (26.53%) and test (26.54%) sets ended up with matching churn rates.

3. **Decision threshold**: I built a hybrid cost model rather than using the default 0.5 value or a single flat cost for both error types:
   - **Cost of a false negative** (missing an actual churner): calculated **per customer**, as 3 months of that customer's own `MonthlyCharges`. This varies by customer, since losing a $100/month customer is a bigger loss than losing a $20/month customer.
   - **Cost of a false positive** (offering a discount to someone who wouldn't have churned): a **flat $50**. In reality, a retention team sets one standard offer, it doesn't scale per customer.

   Because false negatives cost much more than false positives on average in this setup, the threshold decided in order to minimize the cost is not the default 0.5. Rather than estimate this by manually testing a few points, I added a cost vs threshold curve to the dashboard that sweeps every threshold from 0.01 to 0.99 and marks the minimum automatically. This confirms the optimal threshold is **0.19**, with a minimum cost of **$27,594**, well below the $39,993 cost at the default 0.50. The curve is steep and improving from 0 up to about 0.19 where we can see fewer missed churners, which matters more under this cost structure, and then flattens through roughly 0.1–0.35 where later rises steadily as the threshold increases further since more and more actual churners start being missed. 
   
   These three decisions (target definition, stratified split and hybrid cost model) are reflected directly in the Gradio dashboard: the Model Comparison tab shows the three methods agreement, the Distributions tab lets you inspect the engineered features discussed below, and the Threshold & Cost tab lets you move the decision threshold and watch the confusion matrix and business cost update live, including an automatically computed cost-minimizing threshold.

## Data preparation & feature engineering

**Cleaning:**
- Dropped `customerID` as a training feature since it has no predictive value, but I kept a copy of it alongside the pipeline (split together with `X`/`y` using the same random state) so predictions can be traced back to a specific customer for the retention team to act on. Without this, the model's output would be an anonymous list of numbers.
- `TotalCharges` was stored as text with 11 blank entries, all belonging to customers with `tenure = 0` (brand new customers with no charges yet). It was converted to numeric and filled blanks with 0, rather than guessing or dropping those rows.
- Collapsed "No phone service" / "No internet service" into a plain "No", since they're redundant with the phone/internet service flag itself.
- Verified zero remaining missing values and correct dtypes before engineering features.

**Engineered features**, each with a specific hypothesis:
- **`tenure_group`** (binned into 0-12, 13-24, 25-48, 49+ months): churn risk doesn't fall linearly with tenure, it drops sharply after the first year, so binning surfaces that non-linear pattern to a linear model. We can see it in the dashboard's distributions tab where churn is nearly as common as retention in the 0-5 month range, then shrinks to almost nothing by month 60+.
- **`total_services`** (count of subscribed add-ons, 0-7): a proxy for how locked in a customer is, more bundled services means more inconvenience to switch providers. We can also see it in the dashboard, where customers with 0 services skew heavily toward churn, customers with 6-7 skew heavily toward retention.
- **`contract_risk`** (ordinal: month-to-month=0, one-year=1, two-year=2): intentionally ordinal because the order carries real meaning where longer contract terms genuinely correlate with lower risk. This turned out to be the single clearest separator in the whole dataset: month-to-month customers are close to 50/50 churn/retain, while two-year contract customers are overwhelmingly retained.
- **`is_new_customer`** (tenure ≤ 3 months): a simple flag for the highest risk group identified above.
- **`spend_mismatch`** (absolute difference between `MonthlyCharges × tenure` and actual `TotalCharges`): intended to flag plan changes in the middle of the contract period as a signal. I did not separately verify this feature's individual predictive strength beyond its inclusion in the overall model which is a limitation that I noted below.

**Encoding & scaling**: remaining categorical columns were one-hot encoded (`drop_first=True`, avoiding redundant and predictable dummy columns). Numeric columns were standardized using a `StandardScaler` **fit only on the training set, then applied to the test set**. Fitting on the full dataset would leak information about the test set's distribution into training, which of course we don't want.

## Modeling: three implementations, one model

I used **logistic regression**, since the target is binary (churn/no churn).

| Model | Accuracy | ROC AUC |
|---|---|---|
| Naive baseline (always "no churn") | 0.735 | — |
| Scikit-learn | 0.800 | 0.846 |
| Manual PyTorch loop | 0.808 | 0.846 |
| Standard PyTorch (`nn.Module` + `torch.optim`) | 0.806 | 0.846 |

All three real models land within 1 percentage point of accuracy and match to three decimal places on ROC AUC, and all of them beat the naive baseline. **The three methods agree closely.** This was the expected result because all three are solving the same identical mathematical problem, just at three different levels of abstraction where scikit-learn hides the optimization entirely, the manual loop writes gradient descent by hand and the standard PyTorch version uses the framework's `nn.Linear` layer and `SGD` optimizer instead of manual updates.

One small explainable discrepancy worth noting is that the manual PyTorch loop's loss started at exactly `0.6931` (= ln(2), the expected loss when every prediction is 50/50) because I initialized its weights to exact zeros. The standard PyTorch version's `nn.Linear` layer initializes weights to small random values by default, so its starting loss differed run to run until I added the seed, after which results became exactly reproducible across runs. Both converge to nearly the same final loss regardless of this different starting point, which is further evidence they're correctly arriving at the same solution.

I also visually confirmed the three models' predicted probability distributions (not just their summary metrics) are nearly identical in shape since all three show a large spike of confident "no churn" predictions near 0 and a gradual spread toward higher risk, rather than a confident cluster near 1. This reflects the class imbalance, with only ~26.5% churners in training, the model has less data to become highly confident about the churn class specifically, so it's more confident ruling customers out than ruling them in.

## Limitations & next steps

- **Performance ceiling is likely close to what a simple model can achieve on this data.** Published benchmarks on this same dataset, even using more complex methods like random forests and gradient boosting typically land in the same 80-85% accuracy and 0.83-0.85 ROC AUC range. Customer churn has an irreducible random component, such as competitor promotions and personal circumstances, that of course this dataset doesn't capture, so a much higher score would likely indicate overfitting rather than genuine improvement. With more time, I would try engineering interaction features (like contract type × tenure) rather than expecting a fundamentally different score.
- **`spend_mismatch` was not individually validated.** I built this feature on the hypothesis that a gap between expected and actual total charges signals a mid contract plan change, but I never isolated its individual contribution to model performance. With more time, I would have checked its coefficient directly, or dropped it and compared it, to confirm it's pulling its weight rather than just adding noise.
- **Random initialization was only fixed late.** Before adding `torch.manual_seed(14)`, the standard PyTorch model's results shifted slightly between runs (ROC AUC moved from 0.8468 to 0.8458 across two runs with identical code). This is now fixed for reproducibility, but it's worth noting this could have looked like a disagreement between methods if caught at a different point.
- **No temporal or behavioral data.** The dataset is a single quarter snapshot with no usage trends, support ticket history, or complaint records, all of which a real retention team would likely find valuable and which could meaningfully improve the model's precision on the "which customer, right now" question.

## Generative AI use disclosure

I used Claude (Anthropic) throughout this assignment for writing the code in `train.py` and `app.py` (such as data cleaning, feature engineering implementation, all three model training loops and the Gradio dashboard). All the business decisions such as the hybrid cost model's structure and which features to engineer and whywere my own decisions, discussed and refined with AI assistance rather than generated by it. I also used AI to help me write some parts of the report since I didn't know how to explain some concepts in the correct way.
