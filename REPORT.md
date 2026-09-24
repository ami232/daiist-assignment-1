# Assignment 1 Report

- **Name**: Alejandra Hernandez Espinosa
- **Student ID**: 20122
- **Email**: ahernandeze.ieu2022@student.ie.edu
- **Group**: BBADBA 5A 

## Dataset

data/credit_risk_dataset.csv
From Kaggle
One row is one loan application
32,581 rows and 12 columns
It links to my credit-risk idea, and it has real data problems

## Business / real-life framing

The decision behind is that a bank decides whether to approve or reject each loan application
I excluded "loan_grade" and "loan_int_rate", because the bank sets them after judging the risk, so using them would be leakage. Also AUC would jump from 0.830 to 0.892 if I used them.
The split is random because there's no date column, so a time based split is impossible. I used a stratified 60/20/20 split.
The threshold is set by money, not accuracy mainly because of 3 reasons:
1. Proving a defaulter costs 60% of the loan.
2. Rejecting a good customer costs 10% of the loan in lost profit.
3. Missing a defaulter is about 6× worse, so the threshold is low: 0.16.
60% against 10% means that approving a defaulter costs about 6 times more than rejecting a good customer.

## Data preparation & feature engineering

Cleaning:
- dropped 165 duplicate rows
- dropped ages over 100
- treated impossible employment lengths as missing
- filled missing employment length with the median, and added a "missing" flag (applicants with it missing default 31% vs 21%)

New features: 
log income, log loan amount, loan-to-income ratio (and its square), the renter × loan-to-income interaction, and the "loan above 30% of income" flag

Impact: 
AUC went from 0.810 to 0.830.
There is a 30% flag, meaning that below 0.30, most people repaid, right at 0.30, the defaults jump up suddenly. The jump you can see in the Data tab.

## Modeling: three implementations, one model

As the question is yes or no: will this person default? The simple model to use is logistic regression. It gives each applicant a probability of default between 0 and 1.

1. scikit-learn: Does everything for you in one line. 
2. Manual PyTorch: I start all the weights at zero, measure how wrong the predictions are (the loss), let PyTorch calculate which direction improves each weight, and move the weights a small step in that direction. I repeat this 3,000 times, using all the data at every step.
3. Standard PyTorch: Same idea, but with PyTorch's ready-made tools: nn.Module for the model and torch.optim for the update step. It also learns from small groups of 256 applicants at a time instead of all of them at once.

All three use the same data split, the same features and the same small penalty that stops the weights from growing too large (regularisation).

All three reach about the same quality: an AUC of about 0.83, where 0.5 would be guessing. All three are far better than the naive baseline of approving everyone, which costs about 9.1M against about 3.2M with a model

1. scikit-learn and manual PyTorch agree almost perfectly. They solve exactly the same problem, using all the data at every step, and train until they settle.
2. Standard PyTorch gives almost the same predictions, but some weights look different. Income, loan amount and loan-to-income carry nearly the same information, so the model can split the importance between them in different ways and still predict the same thing. Because standard PyTorch learns from small, noisy groups and stops after 60 rounds, it ended up with a different split. 

The honest conclusion: the predictions are reliable, but the individual weights of overlapping features shouldn't be interpreted one by one.

## Limitations & next steps

1. Random split, not time based = get data with an application date, train on older loans and test on the newest ones
2. The data only includes loans that were approved = collect outcomes for some rejected applicants
3. The 60% and 10% cost figures are assumptions = use the bank's real figures for recovery rates and profit per loan. Meanwhile, the dashboard sliders show how sensitive the result is
4. The currency isn't stated in the data = use a dataset with a known source and currency, ideally from the market where the model would be used
5. Age is used as a feature, which is a fairness and legal concern = etrain without age, check that performance stays the same, and then remove it

## Generative AI use disclosure

I used Claude (AI) to write the code and to help explain concepts, and  the report is written by me.
