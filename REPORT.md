# Assignment 1 Report


- **Name**: Juan Jacobo Rico Fontán
- **Student ID**: 18693
- **Email**: jrico.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

The dataset used contains information about military spending, as well as economic variables such as GDP, inflation, population and trade, for different countries throughout the years. It was downloaded with get_data.py, which created data/raw.csv. The dataset comes from the World Bank API, and it was found with the help of AI after I asked it to find a dataset with these characteristics. I chose this topic because I find it interesting and relevant nowadays. The dataset contains 14,322 rows and 17 columns, with 4,817 rows used for modelling and 147 rows from 2024 without a label. It doesn’t use SIPRI directly because the World Bank publishes the same information while adding economic variables that can be used. Each row represents a country in a specific year. It contains information from 1960 to 2025; however, I decided to model the data from 1990 onwards, as the Cold War could have altered the results of the model.

## Business / real-life framing

I’ve decided to use real and not nominal values because of “fake” increases in demand. If a country increases its spending by the same or a similar percentage as inflation, the spending in terms of units does not change. The idea here is to see if countries are rearming, preparing for something, or currently requiring more military goods due to a war. We used local currency for a similar purpose: we don’t want external factors influencing our findings. If the value of the USD changes while the spending remains the same, the spending would appear to change only because of the dollar’s movement against that currency. Another decision to reduce this noise was raising the bar to a 5% increase in military spending, as we believe this is the minimum value that would be considered a true increase in spending. Because each row’s label comes from the following year (t+1), we split by the year of the features, so every block only uses information that would have been available at that moment.
As mentioned before, another strategic decision was leaving out information before 1990, as the Cold War took place under entirely different circumstances and regimes: the arms race, the space race and the nuclear race increased this spending in unrealistic ways. Also, many former Soviet countries were born after this time frame. The final strategic decision was making a false negative cost more than a false positive, as missing an increase could mean losing that margin, instead of simply running an unnecessary campaign. We haven’t made this error more than twice as costly, as in many cases, even if you realise this increase late, you can still enter the race for the margin, meaning it is not a fully disastrous situation.

The data is split by time: training 1990–2010, validation 2011–2016 and test 2017–2023. A false positive costs €300,000 and a false negative €600,000, which gives a decision threshold of 0.30, chosen on the validation set.


## Data preparation & feature engineering

We had to remove 60 rows due to impossible increases, some infinite and others of more than 150%. These changes come from errors in the series, changes of currency, or false or erroneous values in the dataset.

The four hypotheses are as follows. The first is that programs only last a couple of years, meaning that, except during a longer war, an increase in defense spending only has momentum for a couple of years. The second hypothesis is that it has to be economically viable, meaning that a country will only spend or increase its spending if it has the money to do so. The third is a small defense race: when countries spend less than their neighbours, unless they have a treaty or are part of a specific alliance, they tend to increase their spending to try to match their possible surrounding rivals. And finally, a full defense race, meaning that if your neighbour suddenly increases its spending, you probably must do the same, just in case.
The procedures followed to avoid distorted results were, firstly, to winsorize before standardizing, preventing extreme values from influencing the results. In the same way, to avoid filtering future information into the model and causing bias, we fitted the preprocessing only on the training data. We also used a leave-one-out average for the neighbours, and we did not use the World Bank income group, as it is today’s classification and would give future information to the past. On the other hand, momentum did not help, because budgets usually tend to return to the mean after a peak, and that path towards the mean tends to be noisy. The coefficients show that poorer countries with higher inflation, more oil revenue and higher growth tend to be more likely to increase their spending, whilst countries that already spend a lot compared with their region do not usually increase their spending much.


## Modeling: three implementations, one model

I used logistic regression because the objective has a binary result: does spending rise or not. Logistic regression gives us a probability, which is needed to set the threshold by cost. All three models coincide. Manual PyTorch and nn.Module are identical, while sklearn only varies by 0.01 because gradient descent stops at 3,000 epochs. With more epochs, the difference goes down to 0.0004, meaning it isn’t a bug. If you divide sklearn’s objective by C·n, it becomes the same as the PyTorch objective. The L2 penalty is applied only to the weights, not to the bias, to match scikit-learn. That is why we did not use the optimizer’s weight_decay, which would also penalize the bias. The persistence baseline only outputs probabilities of 0 or 1. When it is wrong with “full certainty”, the log-loss becomes extremely large. The model beats the baselines, but only by a small margin. For the business, the key metric is cost, as the model costs €168.0M compared with €180.9M for the best naive strategy, a saving of 7.1%.

| Model | ROC AUC | PR AUC | Log-loss | Precision | Recall | Cost (€M) |
|---|---|---|---|---|---|---|
| Majority baseline | 0.500 | 0.421 | 0.689 | 0.421 | 1.000 | 180.9 |
| Persistence baseline | 0.542 | 0.444 | 6.105 | 0.473 | 0.443 | 211.2 |
| scikit-learn | 0.615 | 0.507 | 0.676 | 0.466 | 0.842 | 168.0 |
| PyTorch manual loop | 0.614 | 0.506 | 0.675 | 0.466 | 0.842 | 168.0 |
| PyTorch nn.Module + optim | 0.614 | 0.506 | 0.675 | 0.466 | 0.842 | 168.0 |

## Limitations & next steps

The model has three main limitations. First, the signal is weak: military budgets depend on wars, elections and alliances, and none of these are in the data, which is why the test ROC AUC is only 0.61. Adding conflict data from the Uppsala Conflict Data Program, NATO membership and election years as features could improve this. Second, the model’s value depends on the cost ratio. It only saves money when a missed opportunity costs between 1.5 and 2.5 times as much as a wasted campaign, and at a 5:1 ratio, sending the sales team to every country is cheaper. To address this, the real costs should be validated with the company, or the problem could be reframed as selecting the top 20 countries per year if the sales team has limited capacity. Finally, after the invasion of Ukraine in 2022, the model correctly predicted the overall level of increases, thanks to the global growth feature, but not which specific countries would increase their spending. Features measuring geographic proximity to active conflicts could help the model identify these countries.

## Generative AI use disclosure

I used Claude in this assignment. Cloud helped me find the data I asked for, and helped me with the setup of the Visual Studio. It also helped with some code decisions, ans specially corrections. I told him to run several experiments with different settings to find the optimal one to then present, and helped me keep track of the steps taken to then write this report. It finally helped paraphrasing, correcting the grammar and structure of specific parts of my report, and it will help me finalise the steps to hand in the assignment correctly. Finally it also helped me create the table for this report.
