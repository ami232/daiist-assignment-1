# Assignment 1 Report

- **Name**: Lucia Bell
- **Student ID**: 19177
- **Email**: lbellsicre.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A]

## Dataset

Name: Telco Customer Churn. 

What it is: dataset from a telecommunications company design to predict customer churn, specially, which customers are likely to cancel their subscription. 

Where did you get it: sourced from IBM's public dataset repository hosted on Github. 

What does one row represent?: each row represents an individual customer with their demographic information, subscribed services, account detailes and their churned actions last month. 

How many rows and columns?: 7,043 rows and 21 columns. 

Why did you pick it: First, it is align with my TFG, as my topic is customer analytics and churn conversion is one of the things I want to work on. Then, it is a binary classification problem that fits the project size constraints while providing numerical and categorical variables that can be used for feature engineering to extract meaningful insights. 

## Business / real-life framing

Hypothetical scenario: A large telecommunications company wants to minimize customer churn by targeting those who are likely to cancel their service in the next month. Therefore, rather than running costly retention campaigns across the whole customer base, the retention team wil be able to engage this users via the model risk scores with personalised discounts, upgrades or loyalty offers. 

Target defition: The target variable is binary representing churning or not. This relates to the business goal of capturing positive churn events before they occur. 

Split strategy (Time-Based vs Random): It is split with 80% train and 20% test as there is no explicit future timestamp specified for each row and the data represents a snapshot of customer accounts at a single moment in time. However, the parameters used for preprocessing (imputation values and scaling) are computed only on the training fold, done to avoid information leakage. 

Decision threshold metric: will be evaluated using a dependent business cost metric in the Gradio dashboard. The business cost of a False Negative (when a churner is not predicted) is high, meaning a loss of Customer Lifetime Value and Monthly Recurring Revenue. Having False Positives (thinking that a customer would turn into a churneer and would lose money by providing a discount) represents a small price to pay. So, the decision threshold can be changed to maximize sensitivity (catching churners) based on the marketing department's retention budget. 

## Data preparation & feature engineering

I did the following actions: 

Feature engineering: I created a variable called "ChargesPerMonthTenure" (total_charges/(tenure+1)) to capture the spending ratio over time. 

Leakage prevention: The StandardScaler was fitted on the training set using .fit_transform(). On the other hand, during testing and dashboard inference, only .transform() to prevent getting future information that could bias the project. 

Inference data alignment: when users input only basic metrics in the Gradio dashboard (tenure, monthly charges and total charges), the system builds a matching dummy Dataframe to ensure the input data structure and number of features match the requierments of the pre-trained models without having a shape mismatch error. 

Missing columns: are automatically imputed with a value of zero and reordered to match X_test.columns exactly (from the original schema).

This strategy ensures structural consistency across all the three models during real-time evaluation, preventing dimensionality errors without requiring live retraining. 


## Modeling: three implementations, one model

A logistic regression model was selected because customer churn prediction is a binary classification task (0 staying and 1 for leaving). Linear regression here would be unsuitable since it predicts continuous values. Logistic egression, applies the sigmoid function to map out values between 0 and 1, wich are probabilities of churn. 

The table is the following: 

| Model            |   Accuracy |   Precision |   Recall |   F1-Score |
|:-----------------|-----------:|------------:|---------:|-----------:|
| Naive Baseline   |   0.734564 |    0        | 0        |   0        |
| Scikit-Learn     |   0.805536 |    0.667785 | 0.532086 |   0.592262 |
| Manual PyTorch   |   0.799148 |    0.64918  | 0.529412 |   0.583211 |
| Standard PyTorch |   0.79489  |    0.640264 | 0.518717 |   0.573117 |

------------------------------------------------

Do the models agree? 

Yes, the three of them achieve very close performance, outperforming the naive baisline. Nevertheless, although they solve the same logistic regression problem, some differences exist due to distinct optimization algorithms like L-BFGS in Scikit-Learn vs gradient descent in PyTorch with specific epchos and learning rates leading to differences in weight initialization or convergence paths too. 


## Limitations & next steps

I found to main limitations: 

Feature Representation Limitation: the pipeline relies on tabular snapshot data without tracking temporal customer behaviour trends over time. With more time, I would incorporate historical time-series logs (like monthly service usage fluctuations, support frequency over the last 6 months) and design moving-windows to capture gradual changes in customer engagement. 

Class imbalance and threshold sensitivity: I saw class imbalance making the default decision threshold suboptimal for catching churners without increasing false positives. So, instead of relying ong a fixed threshold, I would implement cost-sensitive learning during training to penalize false negatives directly, paired with an automated precision-recall curve optimization pipeline related to business costs. 

## Generative AI use disclosure

I used Gemini to help understand the project requirements, clarify technical concepts, and structure the code into a clean, easy to understand implementation. AI was also used as a collaborative guide to draft and refine explanations and metrics tables for the final report.
