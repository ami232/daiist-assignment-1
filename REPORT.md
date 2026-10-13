# Assignment 1 Report



- **Juan Muñoz Quintana**:
- **20007**:
- **jmunozq.ieu2022@student.ie.edu**:
- **A**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Telco Customer Churn

*What is it, where did you get it, what does one row represent, how many
rows/columns, and why did you pick it.*
This dataset is a sample from IBM, its a synthetic dataset describing the customers of a fictional telco company.
Each row of the dataset is a customer and the columns are customer attributes, the target being the customer churn. There are a total of 7,093 rows which are also the number of customers and. 21 columns, 19 attributes an ID and the Churn. I got this dataset from Kaggle and i decided to pick it because it fits the size specifics and the target is binary. 

## Business / real-life framing

A telco decide each month which customers to offer a retention deal to. The model produces a churn probability for each customer and we make the offer when that probability is high enough. Target is Churn = Yes (26.5%). There are no dates in the data so I've used a stratified random split rather than a time based one. The threshold is derived from total business cost rather than from accuracy. We flag the customer when the churn probability exceeds (offer cost / (success rate x customer value)) = 50 / (0.30 x 500) = 0.33.

## Data preparation & feature engineering

I got rid of the customerID column from the database. I noted that 11 clients had equal TotalCharges but none of them had a tenure value greater than zero showing that 11 clients are brand new and haven’t been charged for any service yet. That is the main reason why I replaced the TotalCharges value with 0. The numerical columns were normalized and categorical variables were transformed into one-hot ones, and both transformations were applied only to the training dataset to avoid leakage.

## Modeling: three implementations, one model

Logistic regression, which is used because the target is binary. All three versions-the scikit-learn, a manual PyTorch loop, and nn.Module with optim.SGD-have the same cross-entropy with L2 regularisation on the same split, giving an ROC-AUC of 0.840 and 0.423 in Logloss in the test dataset over the naive baseline of 0.500 and 0.579, respectively, with 171,450 EUR at the 0.33 threshold versus 187,000 EUR for the baseline. They yield the same results because this is a convex problem with a singleton optimum, so any method that converges will achieve the same location. The predicted probabilities for all three differ by less than 3e-6, and no customer is flagged differently.

## Limitations & next steps

Expenditure presumption: presume the cost of a 50 EUR offer, the value for the client of 500 EUR, and the success rate of 30%. The exact threshold would depend on the aforementioned values. In this case, I would obtain the real numbers from the organization or conduct an A/B testing of the offer.

Absence of time factor: the data do not contain any time dimension so that the random partition would not be able to tell whether the model will work for future clients as well. If the data had been dated, I would have split them through the time factor.

Single random split: the results are based on a single test of sample 1409 clients. In this case, I would have used cross-validation to come up with the range. Also, I could use tree-based models or add some interaction features since the linear model cannot take into account interactions, for instance, between contract type and internet service.

Uncalibrated threshold: the 0.33 threshold infers that one uses probabilities that are assumed to be accurate but I have not verified it yet.

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
