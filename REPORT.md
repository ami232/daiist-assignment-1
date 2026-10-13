# Assignment 1 Report

- **Eli Homsi**:
- **18938**:
- **ehomsi.ieu2022@student.ie.edu**:
- **BBADBA 5A**

## Dataset

The Dataset I chosed is "BankChurners", retrieved from Kaggle. It contains 10,127 rows (representing bank customers) and 23 columns, each representing different features; However,is important to note that the 2 last columns were deleted, this as part of the recommendations from the dataset creator.
I picked this dataset because I am really interested in Finance, while also having a strong interest in one company : Nubank. This company is a Colombian fintech that is innovating in the way they provide financial services, democratizing the access to bank accounts. In this sense, I believe that it is interesting to analyzie one of the main conflicts that a bank can have, which is customer churn. For these reason, I found interesting and useful to explore this problemtic through this dataset, especially in an industry and company that im passionate about.


## Business / real-life framing

The scenario this model would serve is a bank's customer retention team. Managers may want to predict which customers are likely to churn, so they can design targeted campaigns and offers to keep those specific customers, such as offering cashbacks, disscounts with affiliated brands, etc.

Moreover, the target column is "Attrition_Flag", which indicates whether a customer is still an existing customer or has attrited. I converted it to 1 (attrited) and 0 (existing), so the model predicts the probability that a customer leaves. The two columns I removed from the dataset were pre-computed predictions of this same target, so keeping them would have let the model copy the answer.

Regarding to the data-split, the dataset is a one-time snapshot with one row per customer and no dates, so there is no time order to protect and a time-based split was not necessary. I used a random split of 70% train, 15% validation, and 15% test, stratified by the target so each part has the same share of churners. Scaling is fitted on the training data only, to avoid leaking information from the test customers. A limitation is that a real bank would train on past customers and predict future ones, which a time-based split would test better, but this dataset has no dates.

Finally, accuracy is misleading here because most customers do not churn, so a model that predicts "nobody leaves" would score high while catching no one. What matters is the cost of each mistake. Missing a customer who leaves loses their future revenue, while a false alarm only wastes a retention offer, so missing a churner is more costly and I favor recall. I set the decision threshold by minimizing a business cost, assuming a missed churner costs the bank 500 and a wasted offer costs 50. I also report precision, recall, and ROC AUC to show how the model behaves at different thresholds.


## Data preparation & feature engineering

I started with an exploratory data analysis. I loaded the dataset, checked if there was any missing values, runned a descriptive analysis of the dataset, checked the data type of every column, and looked at the categories inside each text column.  aAfter having this first look of the data I was dealing with, I then defined the target by converting Attrition_Flag into 1 for customers who left and 0 for those who stayed. Only 1,627 of the 10,127 customers (approximately 16%) churned, so the data is imbalanced. This is why I used a stratified split and did not rely on accuracy alone.

Moreover, I did a correlation analysis on the numeric features to see which had the highest impact on attrition (the target variable). Since correlation does not work on text, for the categorical variables I compared the churn rate of each category against the overall churn rate, to see how far each one was from the average and how relevant it could be for the models. The Kaggle description recommends dropping the last two columns, so I started my list of columns to drop with these two. I also dropped CLIENTNUM, which is only a customer ID and has no relevance for explaining customer behavior. Then, based on each feature's relationship with the target and on business reasoning, I dropped the features that showed little impact on the target variable and/or the ones I believed could have little business relevance in this context, such as gender, dependent count, marital status and education level.

One of the main data-quality issues was the "Unknown" values in Income_Category, the only categorical variable I kept. I kept Unknown as its own income category instead of dropping those customers or guessing their income. Dropping them would have removed 11% of the data, and a bank regularly meets customers who do not disclose their income, so the model needs to be able to score them. Guessing would have meant inventing information. The data supports this choice: the churn rate of the Unknown group is 16.8%, close to the 16.1% average, so it behaves like any other income bracket. To encode this column, I used one-hot encoding, which turns each bracket, Unknown included, into its own 0/1 column. I chose it over numbering the brackets 1, 2, 3 because numbering assumes equal spacing between brackets, and Unknown has no place on that scale. 

For scaling, I used StandardScaler, which rewrites each column to an average of 0 and a spread of 1. The columns are on very different scales; For example, Credit_Limit is in the thousands, while Months_Inactive_12_mon runs from 0 to 6. Without scaling, large numbers would dominate, the manual PyTorch loop could learn very slowly or fail, and the weights could not be compared between features. I fitted the scaler on the training data only and then applied the same averages and spreads to the validation and test sets. If the scaler had also seen the test customers, information about them would leak into training and make the test score too optimistic.

Finally, I checked the impact of reducing the feature selection. In an earlier run with all 32 encoded inputs, the results changed slightly compared with the 16 inputs I kept: the simpler model lost a little recall, and I accepted that in exchange for a smaller feature set that is easier to explain. 


## Modeling: three implementations, one model

For the model selection I chose logistic regression, because the problem is a classification (yes or no) problem: the customer either churns or stays. I trained the same model three ways on the same split and the same scaled data: scikit-learn, a manual PyTorch loop, and the standard torch.nn.Module. All three used the same regularization (C = 0.5, and the equivalent penalty in PyTorch). The two PyTorch models also used the same learning rate (0.5) and the same number of epochs (10,000). After running all the models, the results were the following : 

| Model | Accuracy | Precision | Recall | ROC AUC |
|---|---|---|---|---|
| Baseline | 0.8395 | 0.0000 | 0.0000 | 0.5000 |
| scikit-learn | 0.9000 | 0.7738 | 0.5328 | 0.9093 |
| Manual PyTorch loop | 0.9000 | 0.7738 | 0.5328 | 0.9093 |
| nn.Module| 0.9000 | 0.7738 | 0.5328 | 0.9093 |

As we can see, all three models beat the baseline clearly, as well as the three models agree. Their scores are identical, and the learned weights were almost the same. This was expected, since a regularized logistic regression has one best set of weights, and all three methods use the same data, loss and penalty, so they reach it. The two PyTorch models are the same algorithm written two ways (by hand and with PyTorch's tools), so they match exactly.


## Limitations & next steps

I think the main limitation was related to the amount of churned customers present in the dataset; definitely having so few churn cases in the dataset affects the reability of the models, since they are trained on data that might need more churn cases to learn meaningful relationships between the target and the features.
Moreover, the models could also be improved by having other types of variables, such as cost and revenue per customer, which can help us to make a very interesting and more realistic analysis on the impact of churn customers for the bank. The costs I used in the dashboard are assumptions, not data. I also believe that for the future I should do another type of exploratory analysis to better interpret the impact of the categorical variables on the target, in order to reduce the feature selection to only the most meaningful variables. I made my selection using all the customers (including the test set), mainly based on correlations, churn rates by category and business judgment, without testing what each dropped variable did to the results. Finally, the dataset has no dates, so I used a random split, while a real bank trains on past customers and predicts future ones.

To address these limitations in the future, I would first choose the decision threshold on the validation set using the business cost, and give churned customers more weight during training, which should raise recall without inventing data. I would also gather more real churn cases, or use cross-validation to get more stable estimates. For the creation of new features, I would create revenue and cost per customer columns based on assumptions, or, if possible, gather real data from a bank that has this information, so the cost of a lost customer comes from data and not from my assumption. For the feature selection I would do the exploration and selection on the training set only, so the test set stays untouched. I would compare the validation scores with and without each group of variables, to check that dropping them does not hurt the model. I would also test whether the differences between categories are real or just noise from small groups, for example with a chi-square test, and try grouping categories instead of dropping a whole column. Finally, the models can also be improved with the existance of information about, and if so I would use a time-based split that allow us to predict future churn based on past churned customers.

## Generative AI use disclosure

*I used generative AI tools (Claude and ChatGPT) for support on this assignment. This included explaining the instructions, helping organize the pipeline, writing and debugging code (including parts of the training notebook and the dashboard, which I ran and adapted), explaining concepts, and general guidance on how to approach each step. I also used AI to correct the grammar and improve the coherence of the text in this report. The decisions about the business framing, the feature selection and the conclusions are mine.*
