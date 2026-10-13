# Assignment 1 Report

- **Name**: Vittorio Fialdini
- **Student ID**: 18843
- **Email**: vpereirafial.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

The dataset that I picked comes from Kaggle. It is called "Car Insurance Data" (`Car_Insurance_Claim.csv`, uploaded by Sagnik Roy). The file in this repo (`data/Car_Insurance_Claim.csv`) was downloaded from a copy of it on Hugging Face, which needs no account. In this dataset one row is one customer (one person that holds an insurance policy). It describes the person (age group, gender, race, years of experience driving, education, etc.), their car (sedan or not, pre-2015 or not), the client's postal code and their driving record (past infractions). `OUTCOME` is the target, where 1 means that the client made an insurance claim. The dataset has 10,000 rows and 19 columns (1 ID, 17 inputs, 1 target), with an overall claim rate of around 31%. Only two columns have missing values: about 10% of the values are missing in credit score (982 rows) and annual mileage (957 rows). 

The reason for choosing this dataset is that it is one of the only insurance datasets that has gender and race labels for every row, while also fitting the requirements and not having considerable amounts of missing data. Since in this project I am trying to estimate the cost of banning race and gender considerations for insurance companies, this dataset was ideal, as it actually has these two columns, with zero missing values in them.

## Business / real-life framing

With this project I wanted to answer two questions. First, are gender and race (here, race is only recorded as "majority" or "minority") related to how often customers make an insurance claim? Second, if they are, what happens when an insurer is not allowed to use them in its model: how much profit does it lose, and who ends up being declined? The answers could help companies discuss more than whether complying with the current law is profitable. An insurer could go further than the law requires, for example with a stricter model that also ignores postal codes and credit scores, and market itself as the most consumer-friendly option. That model would be slightly less accurate, but it could make up for it through a larger number of customers.

The decision is made by the insurance company each time someone applies: accept them, or reject them. An applicant who is flagged is rejected: we don't sell them a policy, so we neither earn their premium nor pay their claims.

The target is `OUTCOME` (1 = the customer made a claim), the exact event that decides whether accepting someone makes or loses money. The data has no dates, so I couldn't order the applicants in time and used a random split instead. With dates, a time-based split would be better: train on older applicants and test on the most recent ones, which is how the model would really be used. The split is stratified, so each set keeps the same 31.3% claim rate as the full data. Training (60%) fits the fill-in values, scaling and weights; validation (20%) chooses the regularisation strength and checks calibration and the threshold; test (20%) is used once at the end, so the final numbers are not tuned to it.

Declining is worth €0 either way (TP, FP). Accepting a non-claimer earns the premium minus 20.5% expenses, €1,809 (TN); accepting a claimer also pays the €5,795 average claim, a loss of €3,986 (FN). Claim cost, expense ratio and loss ratio come from ANIA's 2025–2026 report. The real Italian average premium (€421.5 for cars, from the IVASS IPER survey, Q2 2026) would make almost every applicant unprofitable, since this data's claim rate is 6.5 times the market's 4.84%, so the premium (€2,275) is set from the training claim rate to give the market's 79.8% loss ratio. Profit sets the threshold, not accuracy, because the two errors cost different amounts. Declining pays when p × €3,986 > (1 − p) × €1,809, so t\* = 1,809 / (1,809 + 3,986) = 0.312. This only holds if the probabilities are calibrated, which I checked on validation (646 expected vs. 627 actual claims).

On a bigger level, the project also helps the company decide which model to use. The results show the insurer would gain very little from ignoring the law, about €31 per applicant (roughly 4% of profit), and that gain would probably be wiped out by the reputational and financial damage, such as fines and lost customers, if it came out that the company was breaking it. Without gender, the model declines 42% of women and 41% of men, compared with 36% and 44% with it, even though women claim less (26% vs. 36%). The question matters because of the Test-Achats ruling (2011), which has banned gender-based insurance pricing in the EU since December 2012.


## Data preparation & feature engineering

The data required significant data preparation and engineering in order to create the models. First, for columns with groups, such as age, experience, education and postal code, I decided to OHE them. Since there are not many groups inside each of these columns, OHE was chosen, as it is simple and cheap to add in this case. Furthermore, simply numbering the groups wouldn't give the same result, as the model would assume that each step changes the risk by the same amount, which it doesn't. Experience, for example, drops the claim rate from 64% to 24%, then 5%, then 2%.

Infraction columns such as tickets, DUIs and accidents stayed as counts, capped at the 99th percentile of the training set, so that a single outlier person with many tickets, for example, would not pull the model's estimate for everyone. Also, experience and ticket count together may confuse the model. Experienced drivers with accumulated tickets (due to many years of driving) claim less, which could make the model think more tickets are good. Because experience is also in the model, the ticket effect is measured between drivers with the same experience, and there more tickets do mean more risk.  

For missing data, the median of a similar group was filled in. Since there were only two columns with missing values, both could be filled using a related column as a proxy. For missing credit scores, the median of all customers at the same income level was used. For missing mileage, the median of all customers with a similar family situation (married or not, children or not) was used.

One true limitation of the dataset was also found here. Postal code 21217 had 72 drivers in the training set (120 in the full data), and all of them had claimed on their insurance. I believe it is not honest to simply drop this, as in a real-world scenario this is data that should be considered. However, this pushes the model to always reject customers from this postal code. 

Every column is rescaled to the same average (0) and spread (1), using the training set. Before that, mileage ran into the thousands, the counts went up to 10, and most other columns were 0 or 1. Without rescaling, training would move some weights much faster than others, and the penalty would treat the columns unfairly.

Note: All of these choices (the fill-in values, the caps, the rescaling) are worked out from the training data only.

## Modeling: three implementations, one model

I used logistic regression, because the target is yes or no (claim or no claim) and the decision needs a real probability between 0 and 1 to compare with t\*. Linear regression predicts a number and can go below 0 or above 1.

All three implementations minimise the same objective: the average log loss plus an L2 penalty on the weights (the bias is not penalised). scikit-learn sets the penalty strength with C, and the PyTorch versions with λ; they are linked by λ = 1 / (C × n), where n is the number of training rows. C was chosen on validation log loss (C = 1, so λ = 1/6,000), because it changes the model itself and the decision needs honest probabilities. The learning rates (1.0 for the manual loop, 0.01 for Adam) were chosen on the lowest training loss, because their only job is to reach the bottom of the loss; choosing them on validation could favour a run that stopped early, which acts as extra regularisation that C already controls.

*Test set (2,000 applicants), Legal version, decision threshold t\* = 0.312.*

| Model | Log loss | Brier | ROC-AUC | Profit per applicant |
|:---|---:|---:|---:|---:|
| Naive baseline | 0.621 | 0.215 | 0.500 | €0 |
| scikit-learn | 0.351 | 0.112 | 0.909 | €833 |
| PyTorch, manual loop | 0.351 | 0.112 | 0.909 | €833 |
| PyTorch, `nn.Module` + `torch.optim` | 0.351 | 0.112 | 0.909 | €833 |

The naive baseline gives every applicant the training claim rate (0.313) as their probability. That is just above t\* (0.312), so it declines everyone and earns €0. Accepting everyone instead would lose €5 per applicant.

The three agree almost exactly. The largest coefficient gap from scikit-learn is 0.0005 (manual) and 0.00001 (Adam), the largest probability gap on any test applicant is 0.00004, and no test applicant gets a different decision. They should agree: the objective is convex, so it has one lowest point, and any method that reaches it finds the same coefficients. The small gaps come from PyTorch using 32-bit numbers and scikit-learn 64-bit, and from the manual loop stopping very slightly short of the exact minimum.

The agreement needed two fixes along the way. First, the manual loop's learning rate was originally chosen on validation log loss. That picked lr = 0.1, a run that had not finished training (training loss 0.33153 vs. 0.33122 for the others), which scored 0.000004 better on validation only by chance. Its coefficients ended up 0.24 away from scikit-learn's. Choosing the learning rate on training loss fixed it. Second, with 32-bit inputs scikit-learn's own fit was not exactly reproducible: two identical fits differed by 0.001, because the age and experience bands overlap and leave some coefficients in a very flat part of the loss. Giving scikit-learn 64-bit inputs fixed it.

The notebook also breaks the agreement on purpose (section 6.6), to show what it depends on. Without the L2 penalty, nothing stops the coefficient of postal code 21217 (where every training applicant claimed), so it keeps growing the longer training runs: 1.47 after 3,000 steps and 1.59 after 9,000, against 1.27 in scikit-learn. With a learning rate that is too small (0.01), training stops far from the minimum, and the coefficients end up as much as 1.29 away from scikit-learn's.


### Legal vs. illegal models

| Version | Features<br>(columns after encoding) | ROC-AUC<br>(ranking of claimers, higher is better) | Profit per applicant<br>(EUR, at t\*) | Share declined |
|:---|---:|---:|---:|---:|
| Full (gender and race) | 26 | 0.919 | €864 | 40.2% |
| Legal (no gender, no race) | 24 | 0.909 | €833 | 41.2% |
| Strict (also no postal code, no credit score) | 20 | 0.884 | €781 | 39.2% |

Three types of models were considered here: Full (using all data, no limitations), Legal (which perfectly follows current legislation, disregarding gender and race) and Strict (which also does not use postal code and credit score). The conclusion from the models is that it is barely profitable to break the law and use race and gender in the analysis. The gain is small: it is roughly 4% of profit, and its 95% interval is wide (€3 to €57 per applicant), but it does not include zero, so the gain is probably real.

| Group | Applicants | Actual claim rate | Declined (Full) | Declined (Legal) | Declined (Strict) |
|:---|---:|---:|---:|---:|---:|
| Women | 984 | 26.0% | 36.0% | 41.9% | 40.1% |
| Men | 1,016 | 36.4% | 44.4% | 40.5% | 38.3% |
| Race: majority | 1,816 | 31.2% | 40.4% | 41.1% | 39.2% |
| Race: minority | 184 | 32.6% | 39.1% | 41.8% | 39.7% |

With gender, the model declines men more than women, which matches their claim rates. Without gender, the gap disappears: women are declined more often (36% → 42%) and men less often (44% → 41%), even though women still claim less. In other words, the ban makes lower-risk women pay for higher-risk men. Race barely changes anything: both groups claim at about the same rate, and their decline rates stay within about 3 points of each other in every version. The minority group is small (184 applicants), so those numbers are less certain.

A model trained to guess gender from the Legal columns reaches a ROC-AUC of 0.73, so gender can be partly recovered, mainly because men in this data have more tickets and accidents. Race can hardly be recovered (0.56, close to the 0.5 of guessing). So the ban on race costs almost nothing, because race tells the model almost nothing about claims. But this does not carry into the decisions: without the gender column, the Legal model declines women and men at about the same rate (42% vs. 41%), so the ban does work on who gets declined. The Full model gains so little because gender adds only a little on top of the other columns (ROC-AUC 0.909 → 0.919).

Dropping postal code and credit score costs €52 per applicant compared with Legal (95% interval €23 to €78), about 6% of profit, so the insurer would need about 7% more customers to make up for it. It removes the rule that declines everyone from postal code 21217, but it does not make gender or race harder to recover (0.71 and 0.56). As a fairness measure it is weak. As a marketing choice it would only pay if being "the fairest insurer" attracts more than 7% extra customers.


## Limitations & next steps

The data in the DS is almost certainly synthetic. The Kaggle page gives no source, and the data has patterns that real records would not. For example, all 3,530 drivers with 0–9 years of experience have exactly zero speeding tickets, DUIs and past accidents, and every mileage value is an exact multiple of 1,000. These all point toward a DS that cannot be extrapolated to a real market context.

Secondly, there are no timestamps or dates in any of the entries. Without dates I had to split at random, so the model is tested on applicants from the same period it was trained on. A real insurer would train on past years and predict next year, and the test would show how much the model degrades when prices, cars and driving habits change. With dated data, I would train on older policies and test on the most recent ones.

The race label is also too general. One yes/no label puts very different groups together, so any differences between them cancel out. This may be why race shows almost no link to claims (31% vs. 33%) and can hardly be guessed from the other columns (ROC-AUC 0.56).

Finally, some results rest on very few people: all 120 applicants from postal code 21217 claimed, so the model declines everyone there, a gain that real data would not give. The race minority has only 184 test applicants, and the €31 gain from gender and race has a wide interval (€3 to €57). With more time, I would group rare postal codes and repeat the comparison over many random splits. A real insurer would also price by risk instead of declining, which would turn the question into how much prices differ by gender or race, the actual change made by the Test-Achats ruling.

## Generative AI use disclosure

I used Claude (Anthropic), through Claude Code in VS Code, as an assistant under my direction. I set the research question and goals, gave the instructions at each stage, and gave the instruction in what steps to follow for framing and feature engineering. Claude acted as an assistant: it wrote and debugged the code that I instructed. It also edited the wording and grammer of the report, following bullet points and interpretations I gave him. 



