# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Shamil Mukhamedov
- **Student ID**: 20352
- **Email**: smukhamedov.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

*What is it, where did you get it, what does one row represent, how many
rows/columns, and why did you pick it.*

I chose a dataset that shows whether or not a person defaulted on their loans given variables such as age, whether or not they employed, amount for loan, interest on loan, the term (in months) during which the loan was supposed to be paid back.

I got this dataset from Kaggle, I downloaded the zip file and ran it locally from my laptop using VS-code (later using GitHub codespace since I have an Intel mac so I wasn't able to properly submit and added my final touches via Github Codespace). 

I picked it because it was a dataset that allowed for binary classification, and because we worked with datasets like these (not this exact dataset, as we never used Kaggle for loan-related datasets) in our previous classes; therefore, using a loan-default-related dataset would allow me to better understand the business implications so I am able to explain them in this report. 

The dataset has 255347 rows and 18. One of them is the target variable, the other is the loanID, so there are 16 variables that affect whether or not a customer may default on a loan.

Git disallowed me from pushing the dataset with over 250 thousand rows, so I subsampled the dataset. As such, the dataset I actually worked with for the feature engineering was 15,000 rows. I applied one-hot encoding so the amount of columns drastically changed after my EDA because the dataset had string values for every variable and that would not be suitable for any of the three AI models that I will have to configure.

## Business / real-life framing

*The hypothetical scenario this model serves, and what that scenario implies
for how you built the pipeline — target definition, whether a time-based
split was necessary and why (or why not), which metric should drive the
decision threshold and why.*

The hypothetical scenario that my model serves is the ability to disapprove or approve loan requests given the probability that a customer will default on that loan given the 16 variables that they are required to disclose at the moment they are requesting the loan. 

When I asked you about the dataset, you asked me to clearly define what a default on a loan is for the context of my dataset: a 0 on the Default column signifies that the laon (specified in the LoanAmount column) was paid back in full + interest (specified in the InterestRate column) within the time frame stated in the LoanTerm column; per contra, a 1 in the Default column states that the loan was not paid back (including added interest) within the specified time frame (LoanTerm column). From my EDA, I saw that the usual loan time frames are between 2-5 years (24 and 60 months since the dataset measures LoanTerm in months). 

The target variable is the Default column, it is a binary column where a 0 means the loan was paid back, a 1 means the customer defaulted on their loan.

Given that there was not time-related column with the exception of the loan duration a time-based split was not necessary, but I (obviously) applied a train/test/validation split as that is what we did in every single AI-related course. Otherwise, the results would overfit and not be robust.

I selected the decision threshold by minimizing the business cost function on the validation set instead of maximizing accuracy. I assignment 10 as the cost for a false negative: in the case of my project, a false neative occurs when a borrower that defaulted was incorrectly classified as unlikely to default. A false positive had a cost of 1: borrower who repayd was incorrectly classified as likely to default. In sum, approving a loan that is defaulted is much more costly than rejecting a borrower who may have potentially repaid. Effectively, the threshold balances two types of errors according to their business impact.

The ROC-AUC and PR-AUC are then used to assess the model's ability to distinguish between defaulting and non-defaulting borrowers; meanwhile, the weighted business costs determine the decision threshold. I based the decision threshold on the validation datasplit in orer to avoid basing it on the finalized results.


## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

First, I removed LoanID as it just identified individual records and did not provide meaningful characteristics for predicting a loan default.

Lucky for me, there were no NAs and missingn values, so I did not have to clean the dataset.

For my EDA, I converted categoerical variables (almost all variables initially) into numerical representations. I then asked Claude if that was the right way to go about things, and it informed me that using one-hot encoding would be better, so that is what I did. I also subsampled my data since that is what you recommended I do when I asked you before one of our sessions. I have discussed the dimensions of the subsampled dataset in the first section of this report: 15,000 rows and 28 columns. Every binary categorical column I converted to 1s and 0s and kept the numerical representation specifically for the Education column as that one; in theory, has order. More precisely, I encoded 0 for high-school, 1 for bachelor's, 2 for master's and 3 for PhD. The subsampled dataset had a default rate of approximately 11.61% as stated in the comments and markdown for my code when I was double-checking everything.

Afterwards, I engineered six additional features to better grasp financial risks associated with each loan:
1) LoanToIncome = LoanAmount/Income in order to measure how large the loan is relative to borrower's income.
2) MonthlyPayment = (Loan Amount * [annual interest rate/12]) / (1 - [1+ {annual interest rate/12}] ^-loan term in months) which shows the monthly payment using the standard fixed-rate loan amortization that we learned in our finance courses during sophomore and third year
3) PaymentToIncome = (MonthlyPayment * 12) / Income in order to estimate annual loan repayments as a proportion of income. Primary purpose of this feature is because loan amounts on their own do not do a good job of depicting affordability; instead, this new feature does. 
4) EmploymentShare = (Months employed / 12) / Age. I did not initially add this feature, but ChatGPT told me it would be a good idea and its rationale was that this features shows employment history as a factor of age, which is reasonable as someone with long-term employment history would be more financially responsible and as such more probable to not default on their loans. 
5) LogIncome = ln(1+Income). Used Numpy for this feature. Income was VERY skewed (as it usually is), while most other features had an even distribution; therefore, the relationship between income and default was more suitable for a linear binary logistic regression model.
6) LogLoanAmount = ln(1 + LoanAmount) similar reasoning as for the variable above. Quite a few of the loans had collosal amounts, which caused this feature's distribution to had a rightward skew. Applying a natural logarithmic function to it fixes this issue.

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

I chose the logistic regression model because my target variable is binary. My business goal was to minimize costs by identifying risky borrowers, given that logistic regression estimates the probability for binary classifications (in this case, default or payback for each loan), it was perfect for my classification problem at hand. 

I trained three models: scikit-learn's LogisticRegression, manual PyTorth gradient-descent loop, and a PyTorch workflow using nn.Linear, BCEWithLogitsLoss since it is the most common loss function for binary classification problems with the Adam optimizer. All three models were compared against the naive baseline, that predicts the training-set default rate for every observation. Lastly, I tuned the regularization strength for scikit-learn and learning rates for two PyTorch implementations using validation data.

I'm not sure how I can format a table into this, but here are the results/metrics for each of the different models: 

1) Naive baseline: 
ROC-AUC = 0.5
PR-AUC = 0.116
Threshold = 0.01
Total business cost = $1,989 (I assumed it is in dollars, but the dataset did not specify the currency for anything - not loan amounts, costs, or loan payback currency)

2) Scikit-learn logistic regression:
ROC-AUC = 0.7562
PR-AUC = 0.317
Threshold = 0.1
Total business cost = $1,386

3) Manual PyTorch Logistic Regression: 
ROC-AUC = 0.7557
PR-AUC = 0.3176
Threshold = 0.1
Total business cost = $1,395

4) Standard PyTorch Logistic Regression:
ROC-AUC = 0.7477
PR-AUC = 0.3089
Threshold = 0.09
Total business cost = $1,382

For the most part, the three agree. With that said, the sci-kit learn has the highest ROC-AUC, with PyTorch as a close second for both the AUC-ROC and PR-AUC. With that said, all three models outperform that naive baseline by a mile. 

The three models have varying metrics because of varying optimization steps: scikit-learn uses its personalized solver, manual PyTorch calculates graduent updates explicitly, while the PyTorch with Adam uses mini-batch traning. With that said, the three models agree on the overall approximate results, but do not produce identical business costs or metrics. 

All three models reduced business costs with the Adam PyTorch model having lowest costs by a slight margin, scikit-learn coming in second and the manual pytorth logistic regression model having highest costs. All three models minimize the baseline costs by more than 25%, which is a very healthy metric for a bank or any financial institution giving out loans.

As we discussed during our last session when going over our class notebook, which model will be chosen depends most on the priorities of a business. Scikit-learn has the best performance based on metrics, but pytorch with adam optimization has the lowest costs. Realistically, a bank would most probably be interested in minimizing costs, and given how close all metrics are going with the PyTorch with the Adam optimzer would be a rational decision.

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

Right off the bat, the main limitation in my project is the fact that I had to subsample the dataset instead of using the entire 250,000+ rows. As you may see in both the code and the markdowns, the average default rate in both the small loan csv and the original data csv were the same. The results may have been more robust and accurate if I were to use the original large dataset, especially given that large datasets with no missing values are very hard to come across. 

On top of that, a good loan dataset should have timestamps because of which I would have been force to use time-splits. I mentioned above that time splits were not necessary because there were no clear timestamps. Given current inflationary statistics, time-stamps are crucial to a model such as this one. I did apply a stratigies random split regardless in order to compare the implementations on a consistent test set. If a dated loan dataset would be available, I would perform the evaluations on subsequent periods in order to test their generalizability overtime. 

The logistic regression model assumes a linear relationship between my features and the default rate (my target variable). I had to add that into my EDA before I could even start building the models as otherwise the results would not make sense and would simply be incorrect given the skew of a lot of the financial features. In the future, I would use more complex models such as gradient-boost trees as they allow for non-linear relationships. 

I do not have actual data on business costs for incorrectly defaulted or incorrectly approved loans. I assumed a 10:1 ratio as that is easier for me to interpret and code, but if I had actual data for specific costs, or even approximate costs, the business implementation would be more reliable than what I currently have.

Lastly, although the dataset had a lot of very useful demographic features present, it missed a few that I believe are required such as country where the bank operates. A dataset with even more columns rather than rows would be optimal, even with my added features and one-hot-encoding columsn present. Overall, the dataset was great, just could be better, not complaining.

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*

I ran my code through AI **after attempding it on my own several times** if I could not get something to work. AI helped me code one of my added features and gave me the *idea* to add features to it. A lot of the AI generation came from AI with the exception of the model that we had to take from the session five notebook as I simply used the session five notebook for that. I wrote the report on my own fully in order to be better-prepared for the in-class report/exam tomorrow. 

I also used AI to help me with the terminal commands that I needed in order to push my commits properly and also because I had to use Github codespace instead of the VS terminal, and my classmates could not help me with that as they do not have intel Macs. I later re-checked the ReadMe of this assignment and found that a lot of that was specified there so AI was redundant for the last part I mentioned, but I did use (try to use it - unsuccessfully) it initially and then just went along the instructions that were present in the ReadMe file of the original Github Repo. All in all, I really enjoyed this assignment, I just hope my report and final grade for this assignment showcase that too.

I do not know how to use Gradio, so I just sent that off to Claude and it seems to me that Claude did everything right will all the necessary fields and graphs present.