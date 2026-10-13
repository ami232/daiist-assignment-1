# Assignment 1 Report

- **Name**: Emanuel Cohen-Henriquez
- **Student ID**: 18761
- **Email**: ecohenhenriq.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

I used a dataset from TidyTuesday GitHub repository, using data from two hotels in Portugal: a resort in Algarve and a hotel in Lisbon, which use to have 119,390 rows and 32 columns, but I just kept the resort part of the data so I used 40,060 rows and the same amount of columns, but four of them were drop for leakage. I picked it because I went to Algarve a few months ago and really liked it, felt nostalgic and it caught my attention.


## Business / real-life framing

I decided to build a model that predicts the chances of a booking being cancelled by a guest so the hotel can act early. This way, the hotel can call risky guests and ask for a deposit, or try to overbook a little to avoid empty rooms and sustain less costs. The model does this by using a logistic regression, where 1 means the guest cancelled (or never arrived withoud cancelling)and 0 that the user arrived.
As the model needs to be able to predict, it should only contain information they know before it happens, so 4 of the columns were drop because of leakage.

In this case I used a time-based split, consisting on: training Jul 2015–Dec 2016, validation Jan–Apr 2017, test May–Aug 2017. I decided to use this approach as, if I were to use a random one, future dates would end up in the training set and would drive the score up. Because of this, I was able to find that the cancel rate rose from 26.6% to 36.3% in the test months, which a random split would have hidden. A random split would give a fake better score for two reasons (even with the leakage columns hidden), 1) if there are group bookings (different rooms with the same information) and the model picks up some for the training, it will know the outcome for those on the test side. 2) and most importantly, the drift, the world changes after you’ve trained the model, so what it learned from the past no longer matches the present.

As the dataset doesn't contain cost data, I decided that the cost of a missed cancelation is 100 euros per missed cancelation (around the average price at this resort), and for a false alarm is 30 euros (Staff time spent calling, maybe a small discount, or a small risk from overbooking). I used total cost to choose the threshold because a missed cancellation and a false alarm are not equally bad for a hotel, while accuracy, the most common metric, would treat them as equally impactful. To decide the threshold, for each booking, the hotel asks: “is it cheaper to flag it or to leave it?” The model will flag it when the first is bigger: p × 100 > (1 − p) × 30, which means p > 30 ÷ 130 = 0.23. Later I would find out that the 0.23 rule only works if the probabilities are accurate. Because of drift your model’s probabilities are too low, so testing every threshold on the validation data found that 0.13 gives the lowest real cost. On the test months, using the default threshold of 0.5 would have cost the hotel 178,310 euros, versus 84,570 euros with my threshold of 0.13.

## Data preparation & feature engineering

I started with the data cleaning. Some of the changes were: removing bookings with 0 nights (384) and bookings with 0 guests (13); bookings with the country missing were set to "unknown" (464); negative price bookings which had to be a data error (1); and dropping the leakage columns (4), as they contained information only known after arrival.

Next I built 15 number features with the evidence from the data, so they could then be. This were log_lead_time, total_nights, weekend_share, total_guests, has_kids, adr (price per night), price_per_person_night, is_repeated_guest, has_cancelled_before, previous_bookings_not_canceled, booking_changes, log_days_in_waiting_list, total_of_special_requests, has_agent, has_company. For example, the cancel rate rises steadily with lead time (6.5% for bookings made a week or less ahead, 42% for more than 180 days ahead), and since lead time ranges from 0 to 737 days, I used its log so a few extreme values don't dominate. 

I also used  8 category features arrival_month, deposit_type, market_segment, customer_type, meal, distribution_channel, reserved_room_type, and country_group, which turns 125 countries into the 8 most common in training plus “Other”. Portugal cancels about 42% vs about 16% for others. 

Then I had to prepare them for the model, which meant Standardizing the 15 numbers (mean 0, spread 1), so gradient descent works well and the penalty (C) treats them fairly.; using one-hot encoding the 8 categories: one 1/0 column per value; and then everything is fitted on the training data only. That covers the averages for scaling, the category list, and the top-8 countries. It’s applied unchanged to validation and test, so test data never influences the preparation. All of this is fitted on the training data only: the averages for scaling, the category list and the top-8 countries. It is then applied unchanged to validation and test, so the test data never influences the preparation. 

One data-quality issue I noticed is that "Non Refund" deposits cancel 96% of the time, which seems backwards; it is a known oddity of this dataset, so I kept the feature but treat it with caution.



## Modeling: three implementations, one model

As our target is a yes or no (guest will cancel or not) logistic regression would be the best fit, for that we need a probability to compare against the threshold (between 0-1), and linear regression could give us any number, not a probability, which would not fit.

For the three models I used the same set up: 26,552 training bookings, the same 66 inputs, the same penalty (C = 0.01), and the same test set (6,959 bookings, May–Aug 2017). Scikit-learn adds up the loss over all bookings, while PyTorch averages it. So the PyTorch penalty strength is set to 1 ÷ (C × number of training bookings), which makes it exactly the same penalty.

| Model | ROC AUC | Log loss | Accuracy @0.5 | Threshold | Cost on test | Cost per booking |
|---|---|---|---|---|---|---|
| Naive baseline | 0.500 | 0.678 | 63.7% | flag everyone | €132,900 | €19.10 |
| scikit-learn | 0.825 | 0.537 | 71.6% | 0.13 | €84,570 | €12.15 |
| Manual PyTorch | 0.825 | 0.537 | 71.6% | 0.12 | €84,540 | €12.15 |
| Standard PyTorch | 0.825 | 0.537 | 71.6% | 0.12 | €84,540 | €12.15 |

The naive baseline gives every booking the same probability (the training cancel rate, 26.6%), so it cannot tell risky bookings from safe ones (ROC AUC 0.5) and can only flag everyone or no one. All three models clearly beat it and got to the same results, saving the hotel about €48,000 over the four test months. 

However, the three models did not agree at first. After 3,000 training steps, the PyTorch probabilities were up to 0.067 away from the scikit-learn ones. This was not a bug, because the two PyTorch versions matched each other. The reason is that PyTorch moves towards the best weights in small, fixed steps, while scikit-learn's solver takes bigger, smarter jumps, so PyTorch simply needed more steps. When I increased the training to 30,000 steps, the gap dropped to 0.003 and all three gave the same results.

With more time, I would drop one column per category or use a faster optimizer such as Adam to reach the same result in fewer steps.

## Limitations & next steps

As I mentioned before, the dataset didn't contained real costs, so they had to be carefully decided by me. I just tought about how much would be and avg price per night, avg stay, and the cost that the staff reaching out to false alarms would constitute.

Another limitation was that test data is summer data only. As we know hotels are seasonal events, so winter or fall data could be different. One fix could be, once more data exists, test on several periods (e.g. each quarter) instead of one.



## Generative AI use disclosure

I used Claude throughout this assignment. I chose the dataset and the business framing, and picked between options it proposed for the split, the costs and the leakage columns. Claude suggested some of the features, but in the end I choose those that I thought were more valuable. It also wrote the training and dashboard code.
