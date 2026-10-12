# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Ignacio Ascanio Aliño
- **Student ID**: 18719
- **Email**: iascanio.ieu2022@student.ie.edu
- **Group**: [BBADBA 5B]

## Dataset

I chose the Bank Marketing dataset from the UCI Machine Learning Repository. It contains information from direct marketing campaigns conducted by a Portuguese bank, mainly through phone calls, with the goal of getting customers to subscribe to a term deposit.

I am using the bank-additional.csv version, which has 4,119 rows and 21 columns. Each row represents a customer contact during a marketing campaign. There are 20 input features, including customer information (age, job, education), previous marketing interactions, and economic indicators. The last column, y, is the target variable, indicating whether the customer subscribed to a term deposit (yes or no).

I chose this dataset because I am interested in finance and wanted to work on a realistic banking problem. It also contains both numerical and categorical variables, which makes it interesting for feature engineering and preprocessing. Finally, the dataset is small enough to train multiple models efficiently while still being large enough to compare their performance.

## Business / real-life framing

I am working as a data analyst for a bank that wants to make its marketing campaigns more efficient. Instead of calling every potential customer, the bank wants to identify which customers have a higher probability of subscribing to a term deposit. This could help the marketing team prioritize customers, reduce unnecessary calls, and potentially increase the return on its campaigns.

The target variable is binary: 1 if a customer subscribes and 0 if they do not. Therefore, I used logistic regression to predict the probability of subscription.

Since the bank needs to make predictions before contacting customers, I excluded `duration`, which measures how long a call lasted. Including it would create data leakage because that information would not be available when deciding whom to call.

I split the dataset into 70% training, 15% validation, and 15% testing, using stratified sampling to preserve the proportion of subscribers and non-subscribers. Although a chronological split would better represent future campaigns, the smaller dataset does not contain complete dates, making this difficult. This is a limitation of my approach.

For evaluation, I compared accuracy, precision, recall, F1-score, and ROC-AUC against a naive baseline. Since the dataset is imbalanced, I focused on precision and recall rather than accuracy alone.

I also evaluated different decision thresholds using hypothetical assumptions of €5 per call and €50 per successful subscription. I selected a threshold of 0.10 because it maximized the estimated net value on the validation set among the thresholds tested. These financial estimates are illustrative and do not establish the causal impact of making a call.

## Data preparation & feature engineering

During data exploration, I found that the dataset had no standard missing values (`NaN`), but several categorical features contained `"unknown"`. Instead of removing those observations, I kept `"unknown"` as a separate category to avoid losing potentially useful data.

I also discovered that `pdays = 999` indicates no recorded previous campaign contact, rather than 999 actual days. To represent this information more clearly, I created four features:

- **`previously_contacted`:** Indicates whether `pdays` is different from 999.
- **`days_since_previous`:** Preserves actual days since previous contact, replacing 999 with 0.
- **`has_previous_contacts`:** Indicates whether the customer has any recorded previous interactions (`previous > 0`).
- **`has_any_loan`:** Indicates whether the customer has at least one confirmed housing or personal loan.

I removed the original `pdays` column after transforming it and excluded `duration` to prevent data leakage. I also used `StandardScaler` for numerical features and `OneHotEncoder` for categorical features, fitting both transformations only on the training set.

To evaluate the impact of feature engineering, I compared logistic regression using the original features against the engineered features on the same validation set. The engineered model improved accuracy from **89.97% to 90.29%**, recall from **22.06% to 25%**, and F1-score from **0.3261 to 0.3617**.

Although these improvements were relatively small, they suggest that the engineered features helped the model identify more potential subscribers. I therefore kept them in my final model.

## Modeling: three implementations, one model

I used logistic regression because my target variable is binary: whether a customer subscribes to a term deposit or not.

I implemented the same model in three ways: scikit-learn, manual PyTorch using autograd and manual gradient updates, and standard PyTorch using `nn.Module` and `torch.optim.SGD`.

All three models used the same preprocessing and data splits. On the validation set, they achieved similar performance, with 90.29% accuracy. Scikit-learn achieved higher recall and F1-score, while the PyTorch models had higher precision and ROC-AUC.

The two PyTorch implementations produced almost identical results, suggesting that my manual gradient updates were working correctly. The differences from scikit-learn could be explained by different optimization methods, initialization, and convergence.

Accuracy alone was misleading because the naive baseline already achieved approximately 89% accuracy without identifying any subscribers. This is why I also compared precision, recall, F1-score, and ROC-AUC.

I selected **scikit-learn logistic regression** for the final business evaluation because it achieved the highest F1-score (0.3617) and recall (25%) on the validation set at the initial threshold of 0.50. Although the two PyTorch implementations had higher precision (75%) and slightly higher ROC-AUC (around 0.81), their lower recall (17.65%) meant they identified fewer actual subscribers. Since the bank's goal is to identify potential customers to contact, I considered recall particularly important. Scikit-learn also provided a straightforward way to generate subscription probabilities using `predict_proba()`.

However, the default threshold of 0.50 was too conservative for the bank's marketing objective, as it missed 75% of actual subscribers on the validation set. I therefore tested thresholds between 0.10 and 0.90. The threshold of **0.20 maximized F1-score (0.5211)**, balancing precision (50%) and recall (54.41%). However, F1-score does not account for the financial consequences of calling customers.

To make the decision more business-oriented, I introduced hypothetical assumptions of **€5 per call and €50 per successful subscription**. Under these assumptions, the threshold of **0.10 generated the highest estimated net value (€1,595)** on the validation set, compared with €310 from calling everyone and €0 from calling nobody. At this threshold, the model identified 45 of the 68 actual subscribers while recommending calls to only 131 of 618 customers.

Therefore, I selected **scikit-learn with a threshold of 0.10** as my final approach, prioritizing estimated business value over maximizing accuracy or F1-score alone. However, these financial estimates are based on hypothetical assumptions and observed subscriptions, so they do not establish whether the calls actually caused customers to subscribe.

## Limitations & next steps

**1. Class imbalance:** Only 10.95% of customers subscribed, making accuracy misleading. Although adjusting the threshold improved recall, the final model still missed 30 of the 67 actual subscribers in the test set. With more time, I would investigate class weighting and compare precision-recall performance.

**2. Dataset size and time information:** I used the smaller dataset with 4,119 observations. It lacks complete dates, preventing a reliable chronological split. With access to the full campaign data and timestamps, I would evaluate how well the model predicts subscriptions in future campaigns.

**3. Feature engineering:** The engineered features improved F1-score from 0.3261 to 0.3617 on the validation set, but the improvement was relatively small. I tested them together, so I cannot identify which features contributed most. Next, I would test each engineered feature separately.

**4. Business assumptions:** The €5 cost per call and €50 value per subscription are hypothetical. The model also cannot determine whether a customer subscribed because of the call. With real campaign costs and randomized contact data, I would evaluate incremental profitability and test whether targeting customers actually generates additional subscriptions.

## Generative AI use disclosure

I used ChatGPT to help structure the project, understand machine learning concepts, and develop and debug Python code for preprocessing, feature engineering, model training, evaluation, and the Gradio dashboard.
