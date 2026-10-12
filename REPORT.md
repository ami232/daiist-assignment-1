# Assignment 1 Report

- Name: Nicolas Loescher Montal
- Student ID:
- Email: nloescher.ieu2022@student.ie.edu
- Group: PPLEDBA 5A

## Dataset

I used the Netflix Customer Churn and Engagement Analytics dataset from Kaggle
(https://www.kaggle.com/datasets/zeyadmohamed26/netflix-customer-churn-and-engagement-analytics), saved in the repo
as data/netflix_customer_churn.csv. It has 5,000 subscribers and 14 columns and is completely clean, no missing
values or duplicates. The target is churned, whether someone cancelled and about half of them did. The rest of the
columns cover who the person is (age, gender, region), their plan and payment (subscription_type, monthly_fee,
etc.), how much they watch and their habits.

I chose this dataset because a lot of companies run on subscriptions now losing subscribers is a big concern. Predicting who
cancels is a simple yes/no question, and the mix of numbers and categories meant there was actual feature engineering to do.

## Business / real-life framing

Every month Netflix's retention team can send a free month to people who look like they're leaving. Offers cost
money though, so someone has to decide who gets one, and that's what my model does.

To put it in euros I assumed the offer costs one month of fees, a lost subscriber costs six months, and the offer
saves half the churners who get it. With an average fee of EUR 13.68, missing a churner costs EUR 82.10 while a
wasted offer only costs EUR 13.68. Since one mistake is about six times worse than the other, accuracy and F1 are
the wrong things to optimise. Instead I picked the cutoff with the lowest expected cost, using the formula from the
hotel overbooking lab, which gives 0.333. In other words: if someone has more than a one in three chance of leaving,
send the offer. The probabilities turn directly into money, so they need to be honest, that's a big reason I chose
logistic regression.

There is no date column and each row is one subscriber at one moment, so nothing from the future can sneak into
training. A random 60/20/20 split into train, validation and test (same churn rate in each) does the job. Validation
was for tuning, test I only used once.

## Data preparation & feature engineering

Everything here was done on the training data only.

Step one was cutting the dead weight. monthly_fee is just the plan written as a price. Gender, region, device, age
and favorite genre were about as useful as flipping a coin. All of those went. avg_watch_time_per_day is really
watch_hours divided by days since last login, but I kept it since a linear model can't build that ratio by itself.

Step two was looking at how churn behaves, and it doesn't rise smoothly, it jumps. People who haven't logged in for
over a month (one missed billing cycle) suddenly churn much more. Same story for people who watched under 5 hours in
total or less than half an hour a day. On the other side family accounts with 4 or more profiles stay longer, Basic
plan users leave more than Standard or Premium, and people paying with Crypto or Gift Cards leave more too, which
make sense since those don't renew automatically.

Step three was turning those jumps into features, because logistic regression can't see them in the raw numbers.
I made six yes/no flags (active_daily, inactive_30, low_watch, shared_account, basic_plan, prepaid_payment) and kept
two continuous ones: last_login_days for the slow rise in the first month, and log_daily_watch because daily watch
time is very skewed. All 8 are scaled on the training data.

Did it help? Validation log loss went from 0.258 with all the raw columns to 0.116, with less features too. Every
flag mattered, removing any of them made things worse. I also tested log of watch hours as a ninth feature but it
just repeated the others and caused problems in training, so it's out.

## Modeling: three implementations, one model

Logistic regression made sense here: the target is yes/no, I need probabilities I can trust, and the weights are
easy to read. I built it three times with the same loss and the same L2 penalty. In scikit-learn I tuned C on
validation and ended up with C = 100, a weak penalty, which is fine with this many rows and so few features. The
manual PyTorch version is my own weights, loss and update steps with plain gradient descent (learning rate 2.0).
The standard PyTorch one uses nn.Linear, BCEWithLogitsLoss and Adam (learning rate 0.1). Both PyTorch versions ran
for 5,000 epochs and I added the penalty by hand, because Adam's weight decay would also shrink the bias and change
the model.

All three get the same test results, an AUC of 0.995 and 97% accuracy. A model that just guesses the average gets
an AUC of 0.5, so that's a big gap.

### Do they agree?

Now they do, the weights match and the predictions are basically identical. But at first they didn't, and figuring
out why led me to two bugs.

The first was that ninth feature. It overlapped so much with the others that many different sets of weights gave
almost the same loss, so each method settled somewhere else. Removing it fixed it.

The second was harder to spot. The PyTorch intro clamps probabilities so the loss never takes log(0). For about a
fifth of my training subscribers the model was so confident that the clamp kicked in, their gradient became zero,
and my manual model just ignored them without any warning. I rewrote the loss on the logit using softplus, which
avoids log(0) without clamping. BCEWithLogitsLoss already does this in the standard version.

### Calibration and threshold

The probabilities are well calibrated, on validation the model expected almost exactly as many churners as there
really were. So the 0.333 cutoff could be used as it is. Other cutoffs between roughly 0.15 and 0.6 gave nearly the
same cost and 0.50 was technically a bit cheaper, but only because of a handful of people, so I kept 0.333.

What does it mean in money? Per 1,000 subscribers the model saves about EUR 13,300 compared to sending no offers
and about EUR 6,300 compared to sending them to everyone. It catches nearly every churner, and most of the people it
targets really were about to leave. A model that could see the future would only be around 2% cheaper.

### Dashboard

app.ipynb loads the saved models and data from the artifacts folder, nothing gets retrained. You can look at the
test results, compare predictions with what actually happened, see how the three models line up, and check how any
column relates to churn. Moving the cutoff slider or changing the costs updates the confusion matrix, the cost and
the best cutoff right away.

## Limitations & next steps

Honestly the data is probably synthetic. Churn follows exact rules, demographics mean nothing and the scores are
much higher than any real churn model gets. On real data I would expect worse results and I'd have to find the cut
points again.

I also don't know when the data was measured. If last login and watch hours were recorded after people already
decided to leave, the model is partly spotting churn that already happened. Taking features on a fixed date and
counting churn in the following month would fix that, and allow a split by time.

The churn rate is unrealistic too. Here half the people leave, a real service loses a few percent per month. The
model would still catch most churners but there would be way more false alarms, and the probabilities would need
adjusting first.

My cost assumptions are guesses, mainly that the offer works equally on everyone. An A/B test would show how well it
actually works, and an uplift model would show who stays because of it. The dashboard already lets you try other
numbers.

Last thing, the test set is small, so recall could easily be a point or two off, and cross-validation would give a
better range. Also a linear model can't capture combined effects like low watch time and inactivity at the same
time. It wasn't needed here, but on real data I'd try a tree model.

## Generative AI use disclosure

I used ai to help with the Gradio dashboard and to brainstorm the steps of the training process.