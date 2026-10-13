# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**:Marco Celati
- **Student ID**:19219
- **Email**:mcelati.ieu2022@student.ie.edu
- **Group**:BBADBA 5A

## Dataset

This dataset called "Default of credit card clients" is about credit card clients from a bank in Taiwan from April to September 2005 and looks at whether they defaulted on their October 2005 payment. I obtained it from the UC Irvine ML repo online. I also converted it from a .xls to a .csv. There are 30,000 rows and 25 columns, of which 1 is the ID, 23 are features, and 1 is the target. The reason I chose this dataset is because I'm highly interested in financial markets and especially the most recent credit issues with US bonds recently having signficantly increased. Hence, these recent times gave me the specific interest in a dataset focused on credit.

## Business / real-life framing


The hypothetical scenario served is that within a bank, the team who's responsible for retrieving the money from the loans at the end of September 2005 will look at the model's scoring of each client. Flagged clients would get a reminder call before the October due date, and if they say they can't pay, they could be offered a payment plan such that the bank could ideally still make back the money that they were owed, albeit after more time.
The target is whether the client defaulted on their October 2005 payment. Regarding the time-based split, each of our clients had the same result month of October 2005, so since there's only one result month, there's nothing to split. Instead, we used a stratified random split which will have the 22% defaulters in both the train and test.
The metric that should drive my decision threshold is total expected cost, and below are the costs of each outcome:
Client is not flagged, pays:NT$0
Client is flagged, pays:NT$50
Client is flagged, defaults:NT$50+80% of the loss
Client is not flagged, defaults:100% of the loss
Flagging and calling a client costs NT$50 which should represent the time being invested into the call. Whereas when they're flagged and default, we assume that 20% of those people who would've otherwise defaulted are now able to repay the loan following the call which served as a reminder, or due to the payment plan offered to them if they weren't able to pay.
The reason a different metric such as Accuracy shouldn't be used is because accuracy treats every mistake as equally bad, however, in our case a wasted call is nowhere as bad as not flagging a defaulter, one mistake costs us much more than the other.


## Data preparation & feature engineering

I dropped the ID column and renamed some columns to month names. There was also some columns which were hard to interpret like for Education, there was columns 0-6, but only 1-4 were defined as different education levels, and there was nothing for 0, 5, and 6, so I put them into a separate unknown group rather than trying to guess what they could've been. Same for Marriage 0.

In addition, Months late ranged from positive values to -2, which implies some clients paid up to 2 months earlier. I treated 2 months early and 1 month early as simply on time so that the column actually represented months late, and because the results of those who paid on time or those who paid early was very similar and the real risk arose from clients who actually began paying late.

The first feature I created was one for being late in September (late_sep) and one for the number of previous months from April to August in which the client paid late (late_months_apr_aug). I separated them because the September late payment is much more significant towards whether the October payment will be on time, late, or defaulted, compared to late payments of earlier months. I could've made only 1 feature with september going along with the other months, but as it is individually very important for the October outcome, I thought it'd be better to keep it separate.

Another feature I created was credit_usage, which divides the september bill by the client's credit card limit to determine what proportion of their limit they owed in september.

Next, I have payment_ratio, which is the amount they've paid divided by the amount they owed. This allows us to see for instance whether the client aggressively paid back his debt or whether he/she simply paid the minimum amount each month.

The final feature is debt_trend, which is whether the client's debt is growing or shrinking relative to their credit card limit.

Also, something that I had expected was that as long as a client had a clean history in all the months prior to September, I believed that even if they were late in September, they would likely not default in October. However, the results surprised me since even with a good prior history of never being late, if a client was late in September, their default rate was 36% compared to only 11% if they had a good history and were not late in September. So the results reinforced my belief that the september payment would be a key indicator of whether they'd default in October, but it contrasted by thought that as long as they had a good prior history, one late September payment wouldn't be a major difference.

However, a result that followed initial expectations was that out of clients who paid back the smallest share each month had a default rate of 28%, while those who paid the most had a default rate of 17%.

## Modeling: three implementations, one model

I used a logistic regression because the target was whether the client would default or not which was a Yes/No answer. The logistic regression gives us a probability of whether they'll default, and if it's above a certain threshold, we'd call that client.

All 3 implementations used the same 60/20/20 split and had the same 28 columns.


| Model | Threshold | Total cost (NT$) | Clients called | Defaulters caught | Accuracy | AUC |
|---|---|---|---|---|---|---|
| scikit-learn | 0.03 | 26,758,889 | 5,960 | 99.7% | 22.7% | 0.768 |
| PyTorch manual loop | 0.03 | 26,758,889 | 5,960 | 99.7% | 22.7% | 0.768 |
| PyTorch nn.Module + optim | 0.03 | 26,758,889 | 5,960 | 99.7% | 22.7% | 0.768 |
| Naive baseline: call nobody | – | 33,056,492 | 0 | 0.0% | 77.9% | – |
| Naive baseline: call everyone | – | 26,745,194 | 6,000 | 100.0% | 22.1% | – |

All the results are on the same test set of 6,000 clients. The thresholds were chosen on the validation set by minimizing total cost.

All 3 models chose a 3% threshold. The accuracy in the table is misleading from the call nobody baseline because it's only so high because it never makes any of those incorrect phone calls (to people who would have paid anyways). What is more important is the total cost of that baseline is the highest (33 million), which represents all those defaults it did not catch. There's a very small difference in AUC, however, as it is 3s.f., it seems identical within the table above.

## Limitations & next steps

The main limitation I found is I simply assumed that one could do unlimited calls and that they were quite relatively cheap. However, practically I don't think that should actually have been the case because you don't actually have unlimited time to make all those calls, and if you do try to do that, you'd need to hire lots more staff to make all those thousands of calls. The way I think one could address this is adding a constraint of the maximum capacity of calls that the staff can do per month. Moreover, probably the NT$50 cost per call should be increased, otherwise the model would contine simply maximising calls because as of right now, the more calls it makes the lower the total cost, because the amount it saves from catching the defaulters greatly exceeds the amount paid for each of those calls.


## Generative AI use disclosure

I used AI in order to write a part of the code such as the gradio dashboard, but all the decisions, such as the assumptions or the direction I wanted this assignment to take were mine. I did also use it to help brainstorm ideas because I believe it can be a great source of ideas that one might otherwise not have thought of.