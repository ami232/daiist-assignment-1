# Assignment 1 Report

- **Name**: Clementine Mathieu
- **Student ID**: 19311
- **Email**: cmathieu.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A]

## Dataset

This dataset represents the results from 10 US marathons (New York City, Chicago, Honolulu, Marine Corps, Los Angeles, Philadelphia, Twin Cities, Grandma's, Houston and California International), for the editions from 2000 to 2019.

Source: https://zenodo.org/records/6959864

One row represents one runner's information (sex, age...) and finish time in one edition of one race.

The size of the dataset was initially too large to be able to commit it entirely to GitHub. Thus, my thought process consisted of creating all the features using the entire dataset to keep as much information as possible, before creating a sample containing only Marine Corps runners from 2009 to 2018. As such, the history columns (prev_...) are computed based on all 10 races and all years, meaning that a Marine Corps runner in 2016 carries information from their earlier races.

Raw file: 3,343,216 rows and 15 columns
Committed sample: 17,900 rows and 27 columns (Marine Corps 2009-2018 only)

I picked this dataset because I am currently training for a marathon, have already run one, and am therefore very interested in the topic. Additionally, as a runner currently working in a sports company, I have a good initial intuition regarding which features could impact time predictions and which variables to add. The dataset is also very complete, making it possible to uncover interesting findings. Regarding the project requirements, the raw file contains almost no usable features, meaning that feature engineering is not trivial and had to be done to obtain accurate results. Finally, the project addresses a real-world decision, as race organizers need to plan appropriate cut-off times.


## Business / real-life framing

Scenario:
The model serves the Marine Corps Marathon organiser. A few weeks before race day, the organiser has the list of registered runners (sex, age, and their past results at the 10 races) and needs to plan for the en of the race: how many water stations and medical staff to keep open late, how many sweep vehicles, and whether to offer an early-start wave to the runners most likely to be slow. Roads reopen after a set time, so a runner who is still on the course late is a safety and logistics problem. The model flags, for each registered runner, the probability that they finish after 6 hours.

What it implies for the pipeline:

- Target: Slow = 1 if the finishing time is over 360 minutes, 0 otherwise. The organiser does not need the exact time, they need to know who will still be running late, so a binary target and a logistic regression fit the decision better than predicting minutes. 6 hours is a realistic limit for late-course support, and at Marine Corps it gives 9-17% slow runners per year.
- Only information known before the race: every feature has to be available at registration. This is why Gun Time, Place, Sex Place and Div Place were removed, and why a race only counts as history if it happened strictly before the race being predicted.
- Time-based split (necessary): in real life, the organiser trains on past editions and predicts the next one, so I trained on 2009-2014, validated on 2015 and tested on 2016-2018. A random split would mix years, so the model would learn from 2016-2018 runners and then be tested on 2012 runners, which is impossible in real life and would make the results look better than they are. It would also hide the fact that the slow share goes from 8.8% in train to 15.6% in test, which is exactly what the organiser would face in practice.
- Metric for the threshold: the two errors don't cost the same. Missing a slow runner (false negative) means someone is left on reopened roads without support, while flagging a runner who finishes in time (false positive) only costs an unnecessary early-wave slot or a bit of extra staffing. So accuracy is not the right metric, and neither is the default 0.5 threshold. I set the cost of a FN at 10 and a FP at 1, and chose the threshold that minimises the total cost on the 2015 validation year. In practice this means favouring recall: it is better to plan for a few too many slow runners than too few. For tuning C and the learning rates, I used log loss, because the probabilities themselves are useful to the organiser.



## Data preparation & feature engineering

Data quality decisions:
- Gun Time, Place, Sex Place, Div Place are removed because they leak the finishing time information. In the eda, they all correlate a lot, and thus shouldn't be used in the modelisation.
- NYC 2006 us an exact copy of NYC 2007 (same 38,555 rows and the same winner). I checked online and this corresponded to the 2007 information, so I dropped the 2006 NYC data because otherwise it did as if every 2007 NYC runner had a race history.
- The 'data' and 'Country' variables were dropped because it looked like they were wrongly parsed and sometimes even had the finishing time in them. It would have been hectic to rebuild correctly, and I considered that this could be considered information that we could remove.
- Some runners did not have a name ('UNKOWN RUNNER'), so to avoid merging unrelated people into one single 'runner', I decided to remove them.

Linking runners across races:
The dataset has no runner ID, so I built a way to match them: same name + sex + overlapping birth-year range.
- Name and sex alone is not enough because some names are repeated and have more than 2 years apart, so they're clearly different people (ex: Michael Smith has 258 rows). 
- However, using exact birth year is too strict because the birth year was calculated by doing race year - age which depends on whether the persons birthday falls before race day. I allowed for a 1 year difference because 15% of repeated names were off by one year, which splits one person in 2.
- However, 14% of the ages are missing, which is a problem for linking because no birth year can be worked out. The main cases are:
    Honolulu: 292k rows, 68% of the race.
    Chicago: 128k rows. Whole years have no ages: 2015, 2018, 2019.
    Marine Corps 2019: 18.6k rows, the entire year.
    
    This meant that if I had dropped them, it would have removed everyone's history for those races which was too costly.

- So I added a fallback when age is missing. 99.4% of rows without an age still have an age group which gives a range. I only used 5-year groups because wider groups would merge too many different people.
- Rows with no age and no usable age group can't be linked so gtheir history features stay empty and they are treated as first time runners.
For Marine Corps, that is why 2019 was left out of the sample because even with the fallback, only 18% of its runners could be linked to earlier races, against 54% in 2018.

How I determined what counts as previous information:
The data has years but no race dates, so i started by looking only at the race month. However, some races occured in the same month, so i added the exact dates of each race. A race counts as history only if it happened before the race being predicted. Two races were held on the same day (Chicago and Marine Corps 2000) so they don't count for each other. This is to avoid leaking information from after the race being predicted into its features.

Engineered features:

- Runner history (all 10 races): number of earlier races, races at the same course, median, best, and most recent earlier time, share of earlier races slower than 6 hours, days since the last race, number of different races run
- Course features (earlier editions only): course difficulty = race median time + median across all races that year, averaged over earlier editions. Also course's past slow share.
- Course-adjusted history: earlier times ÷ that race's difficulty, and expected_time = adjusted median x Marine Corps difficulty.
- In train.ipynb: female flag, has_history, log of race counts (they're skewed), expected_margin = expected time − 360 (minutes under or over the cutoff), best_gap = median − best earlier time (how consistent the runner is), and log of days since the last race.

Sampling (sample.ipynb)
Once all the feature engineering was done, I was able to determine the smaller set the modelisation would be taking place on. I filtered for Marine Corps only, from 2009 to 2018.
I chose Marine Corps for it to answer well the decision making requirement, as a single organiser needs to plan late-course support for their own  course. Additionally, at Marine corps, slow finishers are rare but common enough to learn from (9-17%). Lastly, the data for Marine Corps was very complete compared to other races (besides 2019 with no age that was left out).
Lastly, the populations between each year are relatively similar, even though of course, the median time went from 4h37 to about 4h55, the share of women from 40 to 47% and the median age from 37 to 40. These elements will of course have a negative impact on the prediction because the ideal would have been to have perfectly similar sets each year, but, it is normal that more women are now running marathons, and that older people ar now too, and it would be too costly to lose the temporality.

In the end, i chose to select 1700 runners per training year and 2000 per test year, so 17900 rows. The athletes are chosen at random within each year.

Preprocessing:
About 56% of the training rows have no earlier race (first-time marathon runners) meaning that the 'previous' columns are NaN. However, the model would read these as 0 and learn that they finished the previous marathon in 0 minutes. Thus, i decided to input the median time of all the training rows. These values were also inputted in the val and test set. 


## Modeling: three implementations, one model

Which model and why:
I used a logistic regression because the target is binary: Slow = 1 if the runner finishes after 6 hours (360 minutes), 0 otherwise. The organiser does not really need the exact finishing time, they need to know who is likely to still be on the course when the roads reopen. Logistic regression also gives a probability rather than just a yes/no, which means I can move the decision threshold according to the cost of each type of error instead of using 0.5 by default.

Split:
- Train: 2009-2014 (10,200 rows, 8.8% slow)
- Validation: 2015 (1,700 rows, 12.8% slow), used to tune C, the learning rates and the decision threshold
- Test: 2016-2018 (6,000 rows, 15.6% slow), only used once at the end

The slow share almost doubles between train and test, which matches what I saw in the sampling (the median time keeps going up over the years). This makes the test set harder than the training set, but it is also the realistic situation: the organiser trains on past editions and predicts the next ones.

Features used (11): is_female, Age, has_history, log_prev_races, log_prev_same_races, prev_distinct_races, prev_slow_rate, prev_last_time, expected_margin, best_gap, log_days_since_last.
I left out prev_median_time and prev_median_rel_time because they were 0.96-0.998 correlated with expected_margin (they are basically the same past time divided or multiplied by a constant). I also left out course_slow_rate and course_difficulty because they only have one value per year, so they are almost constant in train and give extreme z-scores in test.

Regularisation:
For sklearn, I tuned C on the validation log loss over 0.001, 0.01, 0.1, 1, 10. The validation log loss was lower for C = 10. Strong regularisation is clearly bad, but past C = 1 it makes no difference: with 10,200 rows for 11 features the model barely overfits.
To make all three implementations comparable, I adjusted PyTorch’s regularisation to match sklearn’s (Sklearn sums the loss across all training rows, while PyTorch averages it). Both penalise large weights but leave the bias unpenalised.

Decision threshold:
Missing a slow runner (false negative) costs more than flagging a runner who would have finished in time (false positive): a missed slow runner ends up unsupported on reopened roads, while a false alarm only costs an unnecessary early-wave slot. I set the cost of a FN at 10 and a FP at 1, and picked the threshold that minimises total cost on the validation set, then applied it unchanged to test. This gives a threshold of 0.06 for the three models, which is low but makes sense given the 10:1 cost ratio and the fact that only ~9% of training runners are slow.

Baseline:
Predict the training slow share (0.088) for everyone. With the cost-based threshold this means flagging every runner, so it catches all slow runners but with a lot of false alarms.

Results on the test set (2016-2018, 6,000 runners):

| Model | ROC AUC | Log loss | Threshold | Accuracy | Precision | Recall | Cost |
|---|---|---|---|---|---|---|---|
| Baseline (base rate) | 0.500 | 0.4577 | 0.01 | 0.156 | 0.156 | 1.000 | 5,063 |
| scikit-learn | 0.7758 | 0.3697 | 0.06 | 0.522 | 0.230 | 0.878 | 3,896 |
| Manual PyTorch loop | 0.7757 | 0.3698 | 0.06 | 0.521 | 0.230 | 0.877 | 3,907 |
| nn.Module + optim | 0.7757 | 0.3698 | 0.06 | 0.521 | 0.230 | 0.877 | 3,907 |

All three models clearly beat the baseline: log loss goes from 0.458 to 0.370 and the total cost drops by about 23%. They still catch around 88% of the slow runners while flagging far fewer people. Accuracy looks low (52%), but this is expected and wanted: with a 10:1 cost ratio the model is supposed to over-flag, so accuracy is not the metric that matters here, cost and recall are.

Looking at the coefficients, the strongest predictors are has_history (-0.87, a runner with a race history is less likely to be slow than a first-timer with the same imputed values), prev_last_time (+0.65), Age (+0.39), is_female (+0.32) and expected_margin (+0.29), which matches my intuition as a runner.

Do the three agree?
Yes, almost exactly.
- The manual loop and the nn.Module version are identical (same final training loss 0.25404, same coefficients, same predictions). This is expected because they compute the same loss with the same initialisation, learning rate, number of epochs and precision. The only difference is how the code is written (manual update vs optimizer.step()).
- The difference comes from the optimiser, not the model. sklearn uses lbfgs, which runs until it converges, while the PyTorch versions do 3,000 steps of plain gradient descent. The coefficients that differ the most are has_history, log_prev_races and prev_distinct_races, which are strongly correlated with each other (they all describe whether and how much someone has raced before). The other coefficients (is_female, Age, prev_slow_rate...) differ by less than 0.006.

## Limitations & next steps

- Runner linking is approximate: with no runner ID, two people with the same name and age can be merged, and one person can be split in two (spelling, birth year off by more than 1). Fix: rebuild the Country column as an extra matching field, hand-check a sample of common names, or ideally get a real runner ID.
- Only finishers are in the data: runners cut off or who dropped out are missing, so the real slow share is underestimated and the most extreme cases are never seen. Fix: get the starters list and timing mat splits to add non-finishers as Slow = 1.
- Missing ages: rows that can't be linked are treated as first-timers, and Marine Corps 2019 (the most recent year) had to be left out. Fix: recover ages from the official results pages, or link rare names on name + sex only.
- 56% of runners have no history, so the model mostly relies on age and sex for them. Fix: add other races (half-marathons included) or the estimated finish time runners give at registration.
- No race-day weather, and the course features had to be dropped (one value per year). Fix: add historical weather for each race date.
- The model is linear, but effects like age probably aren't. Fix: add age squared and interactions (has_history × prev_last_time), or compare with gradient boosting.

## Generative AI use disclosure

Used to find the exact dates of the marathons.
Used to make the code cleaner and more efficient.
Used for the Gradio application as I had no prior knowledge of Gradio.