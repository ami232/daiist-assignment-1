# Assignment 1 Report


- **Name**: Emilia Granja
- **Student ID**:18875
- **Email**: egranja.ieu2022@student.ie.edu
- **Group**: BBADBA 5B 

## Dataset

The dataset I chose is the Telco Customer Churn from an IBM sample dataset of a fictional telecom company. It has 7043 customers and 21 columns, and each row represents one customer and their service, contract, and account information. The target variable is Churn, which shows whether the customer left the company within the last month. It was converted to 1 = Yes and 0 = No. The churn rate is 26.5% (1,869 out of 7,043 customers), so the dataset is imbalanced, with three out of four customers staying. 

The data cleaning was useful to make the dataset consistent for the model. TotalCharges was converted to numeric because it had been stored as text, and the 11 blank values were filled with 0 since those customers were new and had not accumulated charges yet. 

The categories "No internet service" and "No phone service" were simplified to "No" because they repeated information already given by InternetService and PhoneService, so removing them eliminated redundant columns without changing the meaning of the data. After one-hot encoding, this reduced the model inputs from 35 to 28 columns; the ROC-AUC stayed the same (about 0.843), and, contrary to what I expected, it did not reduce the disagreement between the three models.

I chose this dataset because it represents a realistic business problem that fits classification: a telecom company deciding which customers its retention team should contact. The two possible mistakes have very different costs (calling a customer who would have stayed is cheap, while missing a customer who leaves is expensive), which makes the choice of decision threshold a real business question rather than a technical default.

## Business / real-life framing

The situation is that a telecommunications company’s retention team calls customers at risk of churning and offers them something so they stay. The model assigns each customer a probability of churning, and the team calls all those who exceed a set threshold.
There are two possible errors: False positive: calling someone who would have stayed, which has a cost of $20 (agent time + incentive), and a False negative: overlooking someone who later leaves, which costs $500 (lost revenue). The false negative has a bigger impact. 
These are assumptions I made to keep it simple, since the dataset doesn't include what a lifetime value model needs.
The target is churn: customers who left in the last month (1= left, 0=stayed). It was already defined, so I didn't need to build it. TThe split isn't based on time because there is no date column, so customers cannot be sorted chronologically. If the data had dates, a random split could put later customers in training and earlier ones in testing, which would leak the future into the model. Since this dataset is a single snapshot, that risk does not apply here. I chose a random 80/20 split, stratified by churn, where the training set consisted of 5,634 customers and the test set consisted of 1,409. Only about 1 in 4 customers churn, so a purely random split could result in a significantly higher or lower number of churning customers in the test set, which would make the metrics look better or worse simply by chance.
The churn rate remained at 26.5% in both sets. A `random_state=42` was set to ensure consistency and a specific split, so all three models are trained and tested on the same customers, making the comparison fair.

To evaluate the models, accuracy alone would be misleading: the "don't call anyone" option already achieves 73.5% accuracy, but it costs $187,000 because every churner is missed. To compare the three models I used ROC-AUC, since it measures how well they rank customers by churn risk regardless of the threshold. To choose the threshold I used the business cost, calculated as $20 per false positive plus $500 per false negative.

With these costs, the default threshold of 0.5 costs $92,980, which is even worse than simply calling every customer ($20,700). Because losing a customer costs 25 times more than an unnecessary call, it is worth calling even relatively low-risk customers. The cheapest threshold found in the dashboard is around 0.05, with a total cost of about $15,600. At this threshold the team would call about 1,023 of the 1,409 test customers (73%) and catch 369 of the 374 churners.

This means that, with these costs, the model saves money mainly by identifying the customers who are safe not to call, rather than by finding a small high-risk group. If the retention team had limited capacity and could not make that many calls, a higher threshold would be needed, which I discuss in the limitations.



## Data preparation & feature engineering

For cleaning, the dataset was first cleaned to make it consistent and usable for modeling. TotalCharges was converted from text to numeric, and the 11 blank values were filled with 0 because those customers were new and had not accumulated charges yet, so the impact was minimal. The duplicated categories “No internet service” and “No phone service” were replaced with “No” because they repeated information already given by InternetService and PhoneService, which reduced the number of columns from 35 to 28 without changing the AUC. Finally, customerID was dropped because it is only an identifier and has no predictive value.
After cleaning, the categorical variables were one-hot encoded with drop_first=True, and the numerical variables were standardized with StandardScaler so they were on the same scale. Both transformations were based only on the training set, and the test set was then aligned to the training columns, which prevents data leakage and ensures the test evaluation is fair. The final input to the model had 28 columns.
For feature engineering I created several row-level features from each customer’s own data, meaning they were built only from that customer’s information and do not use any information from other customers or from the test set, so there is no leakage. tenure_group was designed to capture the fact that churn is much higher in the first year and then drops sharply rather than in a straight line, which is supported by the churn rates 47.4% / 28.7% / 20.4% / 9.5% across the four groups: 0–12, 13–24, 25–48, and 49–72 months. no_protection identifies customers who have internet but neither online security nor tech support, and these customers churn much more often (49.0% vs. 13.8%), likely because they are more exposed when problems arise. new_month_to_month flags customers with tenure of 12 months or less and a month-to-month contract, which is a high-risk combination because new customers with no long-term commitment have little that keeps them with the company; this group had a churn rate of 51.4% vs. 16.7%. I also tried num_services, but it was dropped because it was just the exact sum of columns the model already had, so it did not add new information. Overall, the engineered features barely changed performance, with ROC-AUC moving from 0.842 to 0.843, because they were built from variables the model already had, such as tenure, Contract, OnlineSecurity, and TechSupport. Still, they were useful because they made the main churn patterns more explicit and easier to explain to the business.

## Modeling: three implementations, one model

I used logistic regression for this model since the output variable is dichotomous (Churn: Yes/No), provides a probability which could be made into an actionable business decision with a cutoff, and is simple and easy to understand. It was done in three different ways using the same split and the same 28 input variables: Scikit-Learn (LogisticRegression(max_iter = 2000), manual implementation of PyTorch where the starting weights and biases were zeros and the sigmoid function and binary cross entropy were coded manually, and PyTorch with torch.nn (nn.Linear, BCEWithLogitsLoss, and torch.optim.SGD). The last two implementations had the following parameters: learning rate = 0.1, number of epochs = 2000, gradient descent using full batch, and no regularization.
All three models beat the naive “never call” baseline accuracy of 0.735, but at the default threshold of 0.5 they are still much more expensive than calling everyone, which shows that threshold selection matters more than accuracy alone.

| Model                   | ROC-AUC | Accuracy @0.5 | Cost @0.5 |
|-------------------------|---------|---------------|-----------|
| scikit-learn            | 0.8434  | 0.8006        | $92,980   |
| PyTorch manual          | 0.8429  | 0.8013        | $92,960   |
| torch.nn                | 0.8430  | 0.8013        | $92,960   |
| Baseline: never call    | –       | 0.7346        | $187,000  |
| Baseline: call everyone | –       | –             | $20,700   |

Agreement between implementations:
The three implementations agreed very well on the overall ranking of customers, since their ROC-AUC values were all around 0.843. The manual PyTorch and torch.nn versions were practically identical, with final losses of 0.4114 and 0.4115, because they perform the same optimization with the same math. Compared with scikit-learn, the individual predicted probabilities differed by up to about 0.09, but the ranking stayed essentially the same.

I tested several possible reasons for this gap. Removing regularization from scikit-learn made the gap larger, so regularization alone was not the explanation. Increasing training from 2,000 to 20,000 epochs reduced the gap and lowered the loss, which suggests the PyTorch versions had not fully converged. Removing `num_services` or the duplicated categories did not reduce the gap, so those were not the cause either. The most likely explanation is that gradient descent converges slowly here because several inputs are strongly correlated, such as `tenure`, `TotalCharges`, `tenure_group`, and `new_month_to_month`. With correlated inputs, many different weight combinations give almost the same loss, so gradient descent moves slowly toward the optimum with small steps, while scikit-learn's solver reaches it more directly.

## Limitations & next steps

**Threshold chosen on the test set.** The optimal threshold around 0.05 and $15,600 was found by looking at the test set itself, so it is optimistic; a better approach would be to use a validation set or cross-validation to choose the threshold and keep the test set only for the final evaluation.


**Flat costs.** The costs were simplified to flat values of $20 and $500; in a real setting, these should be based on actual margin and retention data, possibly with customer-level costs derived from variables like MonthlyCharges.


**Every call assumed to work.** The model assumes that every retention call succeeds, which is unrealistic; this should be replaced with an estimated success rate from an A/B test or historical retention data.


**No call-capacity limit.** At the low threshold, the team would contact about 73% of customers, which may not be feasible; a more practical strategy would be to rank customers and call only the top N the team can handle.


**Single snapshot, no dates, fictional IBM data.** The dataset is a single snapshot with no dates, so a time-based split was not possible; in a real deployment, the model would be used to predict future customers, so it should be tested with a time-based split. Also, because the data is fictional, real-world patterns may be noisier, so the results might not carry over exactly.


**PyTorch models not fully converged.** The PyTorch versions may improve with more epochs, learning-rate tuning, or by adding the same L2 regularization as scikit-learn.


**One single train/test split.** Using only one split means the results may depend on that particular split; cross-validation would be better to check how stable the results are.


**Linear model limitation.** Logistic regression is linear, so it can only capture interactions if they are built by hand, such as new_month_to_month; a useful next step would be to test a non-linear model and see whether it finds additional patterns.

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
Claude was used.
Dataset & framing:
- Suggested candidate datasets (Telco churn, credit default, used cars)
  and checked their size and columns
- Explained classification vs regression options 
- Proposed options for the split (stratified or not) and the cost model

Code:
- Help me write the code in train.ipynb
- Wrote the code for the features after I chose them
- Debugging failing when main.py runs the notebook
- Wrote app.ipynb (the Gradio dashboard)
- Helped with Git/GitHub 

Analysis:
- Suggested the experiments to investigate model and helped interpret the results

REPORT.md:
- Gave an outline and pointed out errors I made 
