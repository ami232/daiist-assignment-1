# Assignment 1 Report

- **Name**: Manuela Domínguez
- **Student ID**: 19034
- **Email**: mdominguez.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

The Telco Customer Churn dataset is publicly available on Kaggle (referenced below). This dataset contains a total of 7,043 customer records (each individual customer corresponding to one row) and 21 variables that provide a deep overview of the customer base of the company. Further information provided in these dimensions are such as demographic characteristics (gender, senior citizen status, dependents), accounting information (tenure, contract type, payment method and monthly charges), and the catalog of services the customer is subscribed to (internet, phone, security, or technical support between others).

Moreover, of particular relevance to this project, the dataset also provides the target variable: Churn, which is focused on identifying whether a customer has discontinued their relationship with the company or not (provides a binary outcome where 1 represents customers who have churned and 0 otherwise). With all that said, the remaining variables provide the information that will be employed to explore any potential patterns associated with this customer attrition, and therefore, to estimate this probability of churn. 


Finally, I must state that this subject is of particular interest to me. My choice was driven by a professional interest in this industry, since I am currently working as a consultant on a project for a telecommunications client, where I am involved in budget analysis and I have deepened on some of the key challenges affecting the sector's financial performance. Among these, primarily include customer churn, acquisition of new ones and the return of former others (rejoiners) and its impact on the variations in ARPU (Average Revenue Per User) across customer segments and startegies. This experience has motivated me to use this assigment as an additional way to explore how customer data can help identify the inherent patterns associated with churn. Hence, beyond the preductive exercise itself, I seek to understand how these insights could support more targeted retention strategies, helping, as a consequence, teleco companies to balance the customer retention objectives with their financisl implications. 


## Business / real-life framing

The business scenario considered for this project consists in a telecommunicationes company seeking to reduce customer churn alongside with improving the effiency of its renetion budget. 

In this industry, losing a customer has a bigger impact than reducing the existing customer base, it also represents a loss of recurring revenue. Additionally, not all customers contribute equally to this income, as their ARPU can vary depending on the subscribed services, the contract type or any other possible pricing conditions. 

Hence, the main objective is not to simply identify the quantity of potential churners, but also to help the company to actual prioritize the retention efforts from their strategy to be focused on customers at risk of leaving (aside from considering any additional financial implications of these decisions).

For that reason, in order to address this objective, my focus will be on developing a binary classification model using logistic regression with Churn as the target variable (1: customers who leave, 0: customers who remain). Moreover, it is true that the dataset includes a minimal limitation: it does not include any individual observation dates nor a clear chronological sequence of the customer behavior. For that reason, I will not be employing any time based split, but instead I will emply a straified split, preserving approximately the same proportion of churned and retained customers across the three subsets of train, validation and test split. 

Moreover, throughout the results of the model, I will consider false negatives (from now on FN) as particularly relevant. Failing to identify customers who subsequently leave may result in lost recurring revenue alongside with possible future additional costs on customer acquisition to replace their spot in our customer base. However, it is not the only relevant type I am closy paying attention to, since false positives (from now on FP) will also be carrying a cost. Contacting customers who would have stayed anyway in spite of our efforts, lead to unnecessary spending on retention campaigns that could have otherwise be employed in other customers or other areas. For these two reasons, I will focus on evaluating the model on recall and precision metrics. Nonetheless, since one metric can worsen the other, I will also consider the estimated financial cost of each type of error. In that way, by assigning these illustrative costs to both, FP and FN, I will select the classificiation thresholds that minimizes the total cost on the validation set. As a result, this will end up allowing the model's predictions to not only be based in terms of statistical performance, but also in terms of the potential financial implications for the company. 

## Data preparation & feature engineering

I initually conducted an initial exploratory analysis in order to identify any issues regarding data quality. Throughout this evaluation, I confirmed there were no duplicated observations nor customer IDs, as well as the rest of the variables seem not to have any missing values. 

However, I did identify an inconsistency regarding the TotalCharges variable. This one was initually stored as text (string) in spite of representing a numerical amount. By further investigating the discrepancies within this column, I also realized that there were 11 blank spaces as values, which casually coincided with customers who also had tenure = 0, suggesting my hypothesis that those clients who did not have any charges was because they had recently joined the company, and therefore, had not yet accumulated any spending. Because of this, I decided to fill these blank spaces with 0s rather than inputing with the mean or removing the customers in order to not alter the meaning. In that way, I managed to preserve all the 7.043 observations. 

Furthermore, I also converted our target variable (Churn) into a binary column; assigning the value of 1 to those customers who have churned (left the company) and 0 otherwise to those customers who remained. In that way, the customerID was later additionally excluded from the preductors, since it served ad an identifier, and for our analysis it does not add any meaningful information regarding the customer behaviour. 

Regarding feature engineering beyond the original variables, after analysing basic correlation between our target variable Churn and the rest of the features, I introduced three additional features that aimed to capture any other behavior in the relationship between the customer and the telecommunications provider. 

First, the number of extra services variable was made in order to count the number of additional services subscribed to by each customer. This included online security, online backup, device protection, technical support, streaming TV and streaming movies, In this case, my hypothesis was that those customers who were subscribed to several services, would probably have a stronger relationship with the company, making them therefore less likely to churn. So rather than considering each service independently, this variable gave assumed they had the same weight and provide me as an additional measure of the overall extent of the customer's service adoption. 

On the other hand, I created a binary variable of automatic payment, which distinguished customers using automatic payments such as bank transfers or credit cards from other methods. This variable was because during the exploratory analysis I saw that those customers who payed automatically had a considerably lower churn rate than those paying through electronic checks. This had sense, since, in the end, if you are loyal to a service, the customer tends to prefer an automatic charge in their accounts without having to worry about it. Gence, this behavioral pattern is associated with customer retention, while it also helps me to retain the original payment method information. 

Finally, I introduced another variable; tenure monthly charge interaction, where I planned to show an interaction between customer tenure and monthly charges (calculated by multiplying both variables). This also arised during the exploratory analysis, were I saw that the customers that churned had, on average, shorter tenure and higher monthly charges. Hence, I investigated whether the relationship between monthly charges and churn could depend on how long the customer had been in a relationship with the company. So, rather than letting the logistic regression treat both variables, which had some type of correlation with churn, being treated as independent contributors, I decided to represent their combined effect. However, I later recognized that this feature could be overlaping with the Total Charges column, since accumulated charges are also related to tenure and monthly billing. 

Moreover, when it comes to the data splitting and preprocessing, I divided the dataset into three subsets: i) training (accounting for 60%), ii) validation (20%) and iii) test (20%) using a stratified random split. As mentioned before, this type of approach was selected because the dataset does not contain any individual observation dates that could support a chronological split. Additionally, stratification also helped me preserve the original churn distribution of 26% across all three subsets. The final split consisted of 4,225 training observations, and 1,409 observations for each; test and validation. Later, the same feature engineering rules were applied independently to each subset. 

Since logistic regression requires all inputs to be numerical, categorical variables where transformed to one-hot encoding (since there weren't multiple categories in any variables) transforming categories into binary indicators, while numerical variables were standardized using StandardScaler so that we could compare numerical features on the same scale (which it is particularly relevant for gradient descent optimization).

Nonetheless, to prevent any data leakage, the preprocessing transformations were only made on the training data set and subsequently applied to validation and tests subsets without refitting. So after preprocessing, each observatuon was represented by 48 numerical features. These matrices were lated used across scikit-learn, manual and standard Pytorch, so if I found any differences in perfromance from the models could not be attributed to the different input data. 



## Modeling: three implementations, one model

For this project I did a logistic regression, since our objective relied on predicting a binary outcome; whether would the telecommunication's customer churn or not. Hence, unlike linear regression, the logistic one estimates the probability of belonging to a particular class, so it makes it perfectly suitable for identifying customers at risk of leaving and subsequently applying different classification thresholds depending on our business priorities. 

For that reason, in order to compare the different approaches, I implemented the same logistic regression model in the 3 ways using the same test splits and the same 48 preprocessed features. 

1. First, for scikit-learn I used LogisticRegression. At first I sed its default L2 regularization, however, I subsequently increased C to 1e6 since I wanted to make regularization almost negligible and allow a more comparable optimization objective across all three implementations. 
2. Secondly, in Manual Pytorch I implemented logistic regression through weight and bias tensors, binary cross entropy loss and automatic differentiation through loss.backward(), These parameters were updated then manually by using gradient descent without having to rely on torch.optim. 
3. Finally, in Standard Pytorch y employed torch.nn.Module with a single nn.Linear layer and torch.optim.SGD, while maintaining the same BCE loss and learning rate as in the manual implementation. Additionally, both Pytorch approaches used a 0.05 learning rate, while I also trained them on 3,000 epochs but subsequently tested 10,000 epochs to investigate their convergence. 

In order to evaluate the models, I compared their predictions against a naive model that always predicts the majority class (no churn) because of the dataset class imabalance. Therefore, all resylts below were obtained on the same independent test set of 1,409 customers, using a classification threshold of 0.5. 
* Navive model: accuracy = 73.46%, precision = 0.00%, recall = 0.00%, F1 score = 0.00%, log loss = NA
* Scikit learn: accuracy = 79.84%, precision = 63.80%, recall = 55.61%, F1 score = 59.43%, log loss = 0.42087
* Manual Pytorch: accuracy = 80.34%, precision = 65.30%, recall = 55.35%, F1 score = 59.91%, log loss = 0.41962
* Standard Pytorch: accuracy = 80.34%, precision = 65.30%, recall = 55.35%, F1 score = 59.91%, log loss = 0.41969

The naive dataset achieving an accuracy of 73.46%, which also represents the portion of customers who did not churn, while also achieving zero in the rest of the metrics, show why accuracy is not enough to evaluate churn in an imbalanced dataset. All other logistic regression implementations show improvement upon the baseline, identifying around 55% of customers who churn at the default threshold. Their performance is also similar with small differences, and although the pytorch implementations achieved slightly higher accuracy and F1 scores on the test set, these differences are small so there is no framework superior. 

Additionally, in spite of implementing the same mathematical model, the 3 approaches did not happen to produce the same identical probabilities. Indeed, one important difference was the optimization method, where the scikit-learn uses its own numerical optimizer, whereas both of the pytorch implementations mainly relied on full batch gradient descent with a fixed learning rate. 

During the training, I also observed that increasing the number of pytorch epochs from 3,000 to 10,000 reduced the training loss and bring it closer to the sckit-learn's. But there was a trade.off, since their validation loss also increased slightly. This suggested that further optimization of the training objective did not neccessarily meant improved generalization. So the 2 pytorch implementations produced almost identifcal results, and after changing to 10,000 epochs, the mean absolute difference between their predicted probabilities was near 0.00031. Hence, this supports my conclusion that the manual parameter updates and the standard pytorch optimizer were implementing the same learning procedure, so the small remaining differences were mainly consistent with different parameter initialization and optimizer. 

In conclusion, overall, the comparison of the three implementations produce very similar results when trained on the same data, but still I highlight how the optimization choices and training duration can lead to even the smallest differences in predictions. 

Additionally, I wanted to add more of a business evaluation sense to the project, so I evaluated how different classification thresholds affected the estimated financial cost of retention decisions. Therefore, I introduced an assumption (as illustration) of the costs: 10€ each contacted customer (regardless if the model saw it as a potential churner or not) and 100€ (10 times higher) to each customer who churns without being identified (since it privates the company from future recurring revenue). 

After using the scikit-learn model's validation predictions, I found out that the threshold that minimized this estimated cost function as 0.06 (which was expected since the cost of churning without identifying was much higher). So when it was applied to an independent test set, this threshold achieved a recall of 97.6%, however, that also mean a trade-off where it required contacting 69.3% of the customer base. This is what telecommunications companies have to face between maximizing the churn detection or limiting the commercial expenditure. 

I later also introduced a sensitivity analysis that showed that if we increase the assumed cost per contact from 10€ to 20€ and afterwards 30€, this raised the threshold from 0.05 to 0.22 and 0.33 respectively, resulting in more selective retention strategies but also on lower churn detection. But still, I must declare once again that these financials are just illustrative (not actual savings) since this dataset does not contain any information aboyt campaigns or budgets. 


## Limitations & next steps

Despite all these results, there were some limitations that should be considered. 

In first place, the dataset does not include any dates or timely horizon. This prevented me from implementing a chronological split. Despite solving this limitation with a stratified random split for an initial valuation, it does not fully reflect a real business scenario, since with historical customer data, I could define a specific predicting horizon such as churn within next 3 months, or note some changes in the market such as the entrance of a new competitor. 

Another limitation is the business cost analysis that I implemented. This one relies on hypothetical costs: 10€ for every contacted customer and 100€ for every missed churner. But one of the most important limitations in this case is that it assumes that contacting a correctly identified churner prevents the associated financial loss, which in reality I know it does not happen. So if we had real data, I would incorporate new variables such as retention costs, budget, or customer profitability between others in order to estimate the actual financial value of every intervention. 

Also, although I introduced three engineered features, I did not individually assess their contribution to the preductive performance. As I mentioned before, the variable I created about TenureMonthlyChargeInteraction might possibly overlap with the original feature of TotalCharges. So as a next step, I would conduct another analysis that compares the model performance with and without each feature, and in that way determine whether they genuinely improve or not the predictions. 

Finally, while reducing the classification threshold to the optimal cost one of 0.06 and this increased recall to 97.6% on the test set, it also required contacting a high amount of customers. So such broad campaign might be difficult to implement with a limited budget, so I would therefore try to incorporate some budget constraint to prioritize those customers with a higher risk. 

## Generative AI use disclosure

I used only ChatGPT for the following uses: as a supporting tool through the project to structure the python code, any troubleshoot errors, and developing the Gradio dashboard with a more sophisticated look&feel. I also used it to clarify concepts related to logistic regression, pytorch training, evaluation metrics and the classification and business threshold. 
