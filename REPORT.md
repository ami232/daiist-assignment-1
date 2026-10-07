# Assignment 1 Report

- **Name**: Maria del Pilar Guerrero Morales
- **Student ID**: 20373
- **Email**: pguerrero.ieu2022@student.ie.edu
- **Group**: BBADBA 5A 

## Dataset

*What is it, where did you get it, what does one row represent, how many
rows/columns, and why did you pick it.*

I chose the Default of Credit Card Clients dataset from the UCI repository, downloaded through Kaggle. 
The link to the dataset: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

In terms of content, in my dataset each row is one credit card holder at a Taiwanese bank, described by their credit limit, a few demographic fields and six months of history that includes: monthly repayment status, statement amount and amount paid. The file has 30,000 rows and 25 columns. I checked and it has no missing values and no duplicate client IDs. There are 35 rows that are identical apart from their ID, but most of them are accounts with no bills or payments in any month, so I read them as different inactive clients with the same profile and kept them. The target is whether the client will default the next month and it is positive for 22.1% of clients. 

I chose it because during this semester I was working in my internship at Visa and some of the projects we handled explore use cases for AI across commercial banks. While under the new EU AI Act credit authorisations might be a high-risk use case for AI, I was very intrigued by the progress being made with foundation models by players like Nubank and Revolut, which are turning commercial banking into an optimisable, cloud-native and personalised experience. At a smaller scale, I thought it would be an interesting topic to investigate on my own through this assignment.

As a flag, although it looks clean, I did notice it has several undocumented codes for example EDUCATION values 0, 5 and 6, and repayment-status values −2 and 0 as well as 1,930 clients with negative bills, which I handle in the data preparation section below.

## Business / real-life framing

I think the data is inherently suited to a model that supports a monthly risk review at the bank. Hypothetically, every existing cardholder is scored, and clients above a chosen threshold have their credit limit lowered to reduce the bank's exposure before a possible default. This way a bank can manage the expected loss from defaults against the expected interest revenue. Because the features need past data, I'd require at least 6 months of payment history, hence the model only applies to existing customers; new applicants would need a separate solution based on a different set of data (e.g. Experian credit scores). In my analysis I will use the target as given in terms of the expected default for the next month (yes or no).

All clients are observed over the same six months (April to September 2005) and each appears only once, so a random split cannot leak future information into training. I therefore plan to use a stratified 70/15/15 split, which keeps the 22% default rate equal across train, validation and test. This is very important because of the class imbalance between default and non-default. The validation set is used to tune regularisation, the learning rate and the threshold, in other words the hyperparameters. The test set will only be used once, for the final comparison. The cost of this choice is that I cannot test whether the model still works in a later period, such as Taiwan's 2005–06 card-debt crisis that followed, because the economy and the situation would be different: there would be data drift. With the available data it is still worth studying common behaviours and indicators of credit risk.

In terms of the types of errors, the two do not cost the same. Missing a defaulter (FN) means losing much of the outstanding balance, while wrongly lowering a good client's limit (FP) costs some lost revenue and maybe a bit of reputation/ customer satisfaction. For an estimate in terms of cost (to be later used in the model) I set FN = 5 and FP = 1. According to my research there is no single industry standard but the German Credit dataset, also from the UCI has an official cost matrix with these weights. An implication of this is that  accuracy is the wrong metric, since predicting "no default" for everyone already scores about 78%. Instead I will choose the threshold that minimises total cost on the validation set. I also plan on reporting ROC-AUC and PR-AUC as further supporting metrics, which do not depend on a threshold.

Finally, to make this hypothetical scenario more realistic I've decided to exclude the SEX and MARRIAGE variables, because credit regulations commonly prohibit using them in lending decisions, and it will help focus on how past repayment patterns actually predict default.

## Data preparation & feature engineering


Before building anything I removed three columns. `ID` is just a row number, and SEX and MARRIAGE are out for the regulatory reasons I gave in the business framing. I also decided not to feed the twelve raw monthly amounts (`BILL_AMT1–6` and `PAY_AMT1–6`) directly into the model. In the EDA we can see how the tails are extreme (very skewed distributions): bills go up to 1.66M and single payments to 1.68M, while the medians are around 20k and 2k, and the correlation heatmap shows the six bill columns are almost copies of each other. 

The next step was dealing with the undocumented codes I flagged in the Dataset section. I did this based on documentation. The official UCI description only defines EDUCATION 1 to 4 and repayment status −1 and 1 to 9, so the rest comes from the Kaggle page:

- EDUCATION 1 to 4 (graduate school, university, high school, others), with 5 and 6 listed as unknown on Kaggle and 0 not documented anywhere. Due to this I merged 0, 4, 5 and 6 into a single "other" group, which was a limitation.
- repayment status −1 (paid duly) and 1 to 9 (months of delay) from UCI, and from the Kaggle discussion −2 (no consumption) and 0 (use of revolving credit, i.e. paid the minimum but carried a balance).


One thing I could not find an explanation for anywhere is that code 1 (one month late) in the repayment status appears 3,688 times in September but almost never in the other five months, and the two months late is common every month 2600 to 3900. This is inconsistent because months should accumulate. I'm not sure if there were some inconsistencies in collecting the data or maybe they recorded the status differently and changed methodology. Because of this I did not want the model to rely on the exact code in each month, so I used late/not-late flags and counts across the six months and feature engineered variables I discuss further

The remaining data-quality questions were about amounts. 1,930 clients have at least one negative bill, which I read as a credit balance from overpaying, and 3,931 have at least one bill above their limit. I kept both groups but flagged them with the `ANY_NEG_BILL` and `ANY_OVER_LIMIT` flags, and clipped utilisation (bill divided by limit) to the range −1 to 2 so extreme rows do not dominate. This way we kept this information but accounted for the model to understand. 

From there I built 22 features in four groups.

1. Profile group had features like the log of the credit limit, age and the three education dummies. I applied log  because the credit limit is heavily right skewed and logistic regression is linear in its inputs so I wanted to control for this. 

2. The repayment group had feature on how recent and how persistent lateness is: the September delay (capped at 3), late flags for September and August, how many of the six months were late, the worst delay, and how many months the client paid in full or did not use the card. On a note the reason I capped the delay at 3 was how few clients have the higher codes, and how jumpy their default rates are (as discussed inconsistencies in collection)

3. The utilisation group expresses each bill as a share of the client's own limit, with the September value, the six-month average, the change from April to September and the two flags above.

4. The payment group looks at whether the client is actually paying down what they owe. I worked on this by creating a feature with of the share of the previous statement that was repaid (September and the six-month average), how many months had no payment at all, and the log of the average and most recent payment. For the repayment share I paired each month's payment with the previous month's bill, because a payment made in a given month settles the statement from the month before, and I set it to 1 when nothing was owed.

Some way sin which the EDA pointed me towards these choices is that when i plotted the correlation with DEFAULT (target variable) the repayment-status columns are by far the strongest signals and they weaken steadily with age, from 0.32 for September down to 0.19 for April, which is why the recent months get their own features. This makes sense intuitively. In Monthly trajectories defaulters' bills look almost identical to non-defaulters', but when looked at proportion spaid (features I engenieered) they pay roughly half as much each month and their average status worsens every month, which is the argument for ratios instead of raw amounts.

To check this work  (feature engineering) was worth it and not just adding complexity, I trained exactly the same logistic regression, with the same split and tuning, on the raw columns (minus ID, SEX and MARRIAGE). On the test set the raw version reaches a ROC-AUC of 0.713, a PR-AUC of 0.491 and a cost of 0.646 per client, while the engineered version reaches 0.764, 0.523 and 0.574. This proves the raw linear model could not express this ratio / relationship engineered variables or for example categorical relationships like in PAY_1 not being linear steps.

A note on leakage is that every feature is computed only from that client's own row, so building them before the split cannot leak information between clients. The only step that learns anything from the data is the standard scaler, which I fitted on the training set alone and then applied unchanged to validation and test.

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*


Since the target is binary (customer will or will not default) and the decision in my framing compares a probability against a cost-based threshold, the model is a logistic regression and I added L2 regularisation. 

I understood the exercise is that the three versions are the same model, so I made sure they minimise the same thing the average log-loss plus the same regularisation penalty on the weights. What differs is only how each one reaches the minimum as each one uses their solver algorithm. In scikt learn the library does everything with fit, in manual pytorch gradient descent etc and all of the parameters are defined and standard pytorch refers to putorchs ready made building blocks to do it. 

In terms of tuning, I kept it to what the assignment allows and did everything on the validation set. I tuned the regularisation strength once with scikit-learn using a gridsearch to find out what would be the optimal penalty, using validation data, and the best value (C = 0.03) was then reused by both PyTorch models to be consistency. The validation curve is almost flat for anything but very strong regularisation where the model clearly underperformed probably because weights where being so squashed the model couldn't use any of the info.
 
For the two PyTorch models I tuned only the learning rate, again using the  validation data log-loss: the manual loop ended up with 0.5 and the standard workflow (batch learning) with 0.03. The Training tab of the dashboard shows both curves of learning. 

The last thing chosen on validation was the decision threshold. This was done separately per model taking into account as the one that minimises the business cost of 5 per missed defaulter and 1 per wrongly flagged client. Here 1 and 5 where simply used to provide relative terms and dont represent directly a unit/ euro cost etc. Interestingly all three models landed on a common threshold of 0.14–0.15, which is close to the theoretical cut-off of about 0.17 you would expect from that cost ratio if the probabilities are well calibrated. 

The test set was used once, with those thresholds fixed and the results are the following:

| Model | ROC-AUC | PR-AUC | Log-loss | Threshold | Precision | Recall | FP | FN | Total cost | Cost / client |
|---|---|---|---|---|---|---|---|---|---|---|
| Flag nobody | 0.500 | 0.221 | — | — | — | 0.000 | 0 | 996 | 4,980 | 1.107 |
| Naive baseline (train prevalence → flag everyone) | 0.500 | 0.221 | 0.529 | — | 0.221 | 1.000 | 3,504 | 0 | 3,504 | 0.779 |
| scikit-learn, raw features | 0.713 | 0.491 | 0.472 | 0.24 | 0.424 | 0.572 | 775 | 426 | 2,905 | 0.646 |
| **scikit-learn** | 0.764 | 0.523 | 0.444 | 0.14 | 0.353 | 0.760 | 1,387 | 239 | 2,582 | 0.574 |
| **Manual PyTorch** | 0.764 | 0.523 | 0.444 | 0.14 | 0.354 | 0.761 | 1,385 | 238 | 2,575 | 0.572 |
| **Standard PyTorch** | 0.762 | 0.522 | 0.444 | 0.15 | 0.373 | 0.726 | 1,216 | 273 | 2,581 | 0.574 |

The naive baseline predicts the training default rate of 22.1% for everyone, so it cannot rank clients and its best threshold collapses into a single policy for all of them. With my costs, flagging everyone (0.779 per client) is actually cheaper than flagging nobody (1.107). This wa interesting to explore as a baseline to beat. The three logistic regressions bring the cost down to about 0.57 per client, roughly 26–27% below the baseline, while catching about three quarters of the defaulters. However it's also important to acknowledge that recall of about 0.75 with a precision of only about 0.35 means that the model is catching most of real positives but out of all of the positive predictions only being 35% accurate. In this scenario this makes sense because of the cost it's most interesting to catch them incorrectly than to not.

In terms of agreement of the models I would say they agree very closely as they reach a very similar but not perfectly. The manual loop reaches the same solution as scikit-learn. Their final training objective matches to six decimal places, the largest coefficient difference in terms of the final model is onlu 0.016 and the two make the same decision for most/ nearly all of the of test clients. The standard workflow ends slightly higher on the training objective (0.4334 vs 0.4330). Its largest coefficient gaps are on the education dummies (for example `EDU_UNI` 0.18 vs 0.29) and on `PAY_RATIO_MEAN` (−0.17 vs −0.27). So essentially it has reached a different optimum. Still it  agrees with scikit-learn on 95.1% of decisions. Even so in terms of performance its ROC-AUC and total cost are practically identical. 

In the dashboard, the comparison tab shows this in the agreement scatter plots and the grouped coefficient chart, and the loss curves in the *Training* tab show the loss settling. I would say the standard workflow which had mini-batch SGD is the one that has most paks and downs less smooth because with a fixed learning rate it never fully settles because every step's gradient is noisy in comparisson with scikit learn and manual loop which caluclate the slope more precisley by using all of the clients data at evrery step.


## Limitations & next steps

The biggest limitation is the one I already pointed out in the framing. There is no out-of-time validation. All 30,000 clients come from the same six months of 2005, so a random split cannot tell me whether the model would still hold up later, for example during the 2005–06 card-debt crisis of Taiwan which would be the most interesting thing to test. If we had a longer period of data we could do maybe a temporal split and test with out of time validation. 

Moreover another limitation is that the cost matrix is also an assumption. The relative costs are borrowed by German Credit dataset. If working with an actual client bank estimating the actual loss from a default which should also vary depending on the size of the balance outstanding instead ofvbeing a fixed cost should be better. The dashboard lets the costs be changed to see how the optimal threshold moves when different possible assumptions might be taken but it is still static. 

Another limitation is the repayment-status coding which I mentioned looks inconsistent over time, since code 1 exists almost only in September and I could not find any documentation explaining it. My features work around this, but they cannot fix it. 


## Generative AI use disclosure

I worked in Visual Studio Code and had a window with Claude open with access to my project. I walked through it with its assistance. We iterated through key decisions in a conversation e.g. including variables, hyperparameter tuning etc. In terms of code it helped me with the functions and worked following my methodological decisions which we discussed through, and I reviewed it. It was also very useful to produce the tables. The app.py dashboard it did entirely with my instructions of what i wanted to be compared. In terms of the Report I asked it to draw from our conversations to do a write-up of the key decisions we agreed on together and based on that I iterated through it with my own thoughts. I finally asked it to check through the grammar and structure of everything. 