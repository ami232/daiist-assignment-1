# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Eli Homsi**:
- **18938**:
- **ehomsi.ieu2022@student.ie.edu**:
- **BBADBA 5A**

## Dataset

*.The Dataset I chosed is "BankChurners", retrieved from Kaggle. It contains 10,127 rows (representing bank customers) and 23 columns, each representing different features; However,is important to note that the 2 last columns were deleted, this as part of the recommendations from the dataset creator. I picked this dataset because I am really interested in Finance, while also having a strong interest in one company : Nubank. This company is a Colombian fintech that is innovating in the way they provide financial services, democratizing the access to bank accounts. In this sense, I believe that it is interesting to analyzie one of the main conflicts that a bank can have, which is customer churn. For these reason, I found interesting and useful to explore this problemtic through this dataset. *

## Business / real-life framing

*The hypothetical scenario this model serves, and what that scenario implies
for how you built the pipeline — target definition, whether a time-based
split was necessary and why (or why not), which metric should drive the
decision threshold and why.

The scenario this model serves is a bank's customer retention team. Managers want to predict which customers are likely to churn, so they can design targeted campaigns and offers to keep those specific customers. The target column is "Attrition_Flag", which indicates whether a customer is still an existing customer or has attrited. I converted it to 1 (attrited) and 0 (existing), so the model predicts the probability that a customer leaves. The two columns I removed from the dataset were pre-computed predictions of this same target, so keeping them would have let the model copy the answer.

**Split.** The dataset is a one-time snapshot with one row per customer and no dates, so there is no time order to protect and a time-based split was not necessary. I used a random split of 70% train, 15% validation, and 15% test, stratified by the target so each part has the same share of churners. Scaling is fitted on the training data only, to avoid leaking information from the test customers. A limitation is that a real bank would train on past customers and predict future ones, which a time-based split would test better, but this dataset has no dates.

**Metric and threshold.** Accuracy is misleading here because most customers do not churn, so a model that predicts "nobody leaves" would score high while catching no one. What matters is the cost of each mistake. Missing a customer who leaves loses their future revenue, while a false alarm only wastes a retention offer, so missing a churner is more costly and I favor recall. I set the decision threshold by minimizing a business cost, assuming a missed churner costs the bank, lets say, 500 and a wasted offer costs 50. I also report precision, recall, and ROC AUC to show how the model behaves at different thresholds.

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Generative AI use disclosure

*I used generative AI tools (Claude and ChatGPT) for support on this assignment. This included explaining the instructions, helping organize the pipeline, correcting and debugging code, explaining concepts, and general guidance on how to approach each step.*
