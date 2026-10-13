# Assignment 1 Report

- **Name:** Ainhoa Olaechea
- **Student ID:** 18680
- **Email:** aolaechea.ieu2022@student.ie.edu
- **Group:** BBADBA A

## Dataset

I used the Hotel Booking Demand dataset from Kaggle. It comes from a 2019 paper by Antonio,
Almeida and Nunes and has real bookings from two hotels in Portugal: a city hotel in Lisbon
and a resort hotel in the Algarve.

Each row is one room booking. It tells you things the hotel knew when the booking was made,
like how many days in advance it was booked, how many nights, how many guests, the price per
night, the booking channel and the guest's country, and whether the booking was cancelled in
the end. In total there are 119,390 bookings and 32 columns, with arrivals between July 2015
and August 2017. The column I predict is `is_canceled` (1 if cancelled, 0 if the guest came).

I picked this dataset because cancellations are a real problem for hotels and the target is
easy to understand. It also has some traps that made me think carefully: some columns are only
filled in after the booking, the data is ordered in time, and there are many repeated rows from
group bookings.

I only used the City Hotel (79,330 bookings). The main reason is that its demand is much less
seasonal than the resort's. The city hotel's median price only goes from about €80 to €120 a
night during the year, while the resort goes from €48 in January to €188 in August. Because of
that, a cancellation costs the city hotel roughly the same in any month, so I can use one fixed
cost per mistake instead of a different cost for each season.

After cleaning, I took a random sample of 20,000 bookings to stay within the size guideline. I
used a fixed random seed (42) so the same 20,000 bookings are picked every time the code runs,
which makes my results reproducible. The number 42 itself doesn't matter, only that it stays
the same. The sample cancels 41.8% of the time, almost the same as the full city hotel data
(41.7%), so it represents it well.

## Business / real-life framing

The city hotel loses money when guests cancel late, because the room usually stays empty. My
idea is to use the model to decide which guests should pay a deposit:

- **Goal:** lose less money from cancellations and keep the hotel full.
- **Model:** when a booking is made, predict the probability that it will be cancelled.
- **Action:** if the risk is high enough, the guest has to pay a deposit, which they only get
  back if they cancel more than 14 days before arrival.

I chose 14 days by looking at when guests actually cancel. In the city hotel data, 12% of
cancellations happen in the last 7 days before arrival, 19% in the last 14 days and 33% in the
last 30 days. Two weeks gives the hotel time to sell the room again, without being as strict as
a 30-day rule. One downside is that 18.5% of guests book less than 14 days ahead, so for them
the deposit would not be refundable at all.

This framing changed how I built the pipeline:

1. **What I predict.** I kept the normal `is_canceled` column. It's simple, but it isn't a
   perfect match for my policy: a guest who cancels months in advance costs the hotel nothing
   (they get their deposit back), yet the model treats them the same as someone who cancels the
   day before.
2. **Splitting by time.** In real life the hotel learns from past bookings and predicts future
   ones. If I split the data randomly, the model would train on bookings from 2017 and be tested
   on 2015, which is like seeing the future, and the results would look better than they really
   are. So I split by arrival date: the oldest 70% for training, the next 15% for validation and
   the newest 15% for the final test.
3. **Only information known at booking time.** I removed any column the hotel wouldn't know on
   the day of the booking (more on this below).
4. **Removing `deposit_type`.** This column shows whether the hotel already asked the guest for a
   deposit. Since my model is supposed to make that decision, it shouldn't be given the answer.
5. **Choosing the threshold by cost, not by accuracy.** The two possible mistakes don't cost the
   same. A typical booking is worth about €265 (€100 a night × 3 nights):
   - If the model misses a cancellation, the hotel loses money only when the guest cancels in
     the last 14 days, which is about 19% of cancellations. So I estimate this mistake costs
     0.19 × €265 ≈ **€50**.
   - If the model asks a reliable guest for a deposit by mistake, some of them might get annoyed
     and book somewhere else. I assumed 1 in 10 would, so this mistake costs 0.10 × €265 ≈ **€25**.

   These numbers are my estimates, not facts. Since missing a cancellation costs twice as much
   as a false alarm, it makes sense to ask for a deposit a bit more easily than at 50% risk; in
   theory the best cut-off is about 33% (25 / (25 + 50)). I picked the threshold on the
   validation data by finding the one with the lowest total cost, and the dashboard has a slider
   to see how the cost changes.

## Data preparation & feature engineering

### Cleaning

- `children` was missing in 4 rows, so I set them to 0. `country` was missing in 488 rows, so I
  labelled them "Unknown" instead of deleting them.
- An empty `agent` or `company` means the booking was made without one, so I turned these into
  yes/no columns.
- I removed 432 bookings that made no sense: 0 nights, 0 guests, or a price over €1,000 a night
  (one was €5,400). That's only 0.5% of the city hotel data.
- I kept 25,902 rows that are exact copies of other rows. There's no booking ID, so these are
  most likely several identical rooms in one group booking. Each one is a real room that could
  end up empty, and they cancel a lot (61% vs 29% for the rest), so removing them would hide part
  of the problem. Because I split by date and copies have the same arrival date, they always end
  up in the same group, so they don't make the test results look better than they are.

### Removing columns the hotel wouldn't know yet (leakage)

The rule I used: *on the day the guest books, would the hotel already know this?* If not, the
column is "leaking" the answer and has to go.

| Column | Why I removed it |
|---|---|
| `reservation_status` | It says "Canceled" or "Check-Out", which is basically the answer itself. It matched the target perfectly |
| `reservation_status_date` | The date of cancellation or check-out, which happens later |
| `assigned_room_type` | The room the guest actually got at check-in |
| `booking_changes` | Counts changes made after booking, so it's always 0 on the booking day |
| `required_car_parking_spaces` | Not one of the 492 bookings that asked for parking was cancelled. That's too perfect, so it's probably only recorded when the guest arrives |
| `total_of_special_requests` | Guests with no requests cancel 55% of the time vs about 22% with requests, but requests can be added after booking, so I removed it to be safe |

I kept `days_in_waiting_list` because the hotel knows it when the booking is confirmed, which is
exactly when it would decide about the deposit.

### New features

Each new feature only uses information from its own row, so nothing from the test period can
sneak into training.

| Feature | What the data shows | Why I added it |
|---|---|---|
| `total_nights` | 1 night: 30% cancel; 2–3 nights: 47%; 8+ nights: 54% | Weekend + weekday nights in one number |
| `has_agent`, `has_company` | Company bookings cancel 21% vs 43% | The original columns were ID numbers, which are just labels, so a yes/no makes more sense |
| `cancelled_before` | 94% vs 38% | People who cancelled before tend to cancel again |
| `log_lead_time` | Booked 7 days or less ahead: 12%; more than 180 days: 65% | Lead time goes from 0 to over 600 days. Taking the log stops the very big values from dominating |
| `arrival_weekday` | Thursday: 49%; Sunday: 34% | Business and leisure trips behave differently |
| `country` (top 10 + "other") | Portugal: 66% vs 27% for other countries | 133 countries is too many for 20,000 rows. The Portugal column works as a "local guest" flag |
| `revenue_at_risk` (price × nights) | Not strong by itself | Connects the model to how much money is at stake |

I didn't create a separate "is from Portugal" column, because it would be an exact copy of the
Portugal country column, and a linear model gets confused by duplicate columns.

I also removed some date columns. The year, because the test data comes later, so the model would
see a year it never trained on, and the year doesn't help predict future bookings. The week
number, because it repeats the date and the model would treat week 52 as "bigger" than week 1.
The day of the month, because it has no real meaning, while the weekday does. And the month,
because the effect was small (36% to 46%), which fits with the city hotel not being very seasonal.

### Splitting and preparing the data

| Group | Bookings | Arrival dates | Cancelled |
|---|---|---|---|
| Training | 14,000 | 1 Jul 2015 – 18 Feb 2017 | 41.3% |
| Validation | 3,015 | 19 Feb 2017 – 24 May 2017 | 45.1% |
| Test | 2,985 | 25 May 2017 – 31 Aug 2017 | 41.0% |

I split on dates rather than row numbers, so all bookings arriving on the same day stay together.
I used the validation group to choose the settings (C, learning rate and threshold) and only used
the test group once, at the very end.

Before training, I scaled the number columns so they all have a similar range, which matters a
lot for PyTorch, and turned the text columns into 0/1 columns. Categories that appear fewer than
20 times are grouped together. Both steps learn only from the training data and are then applied
to validation and test. In the end the model gets 59 input columns.

## Modeling: three implementations, one model

I used **logistic regression** because my target is yes/no. It gives a probability, not just a
yes or no, so the hotel can decide where to draw the line based on cost. It's also easy to
interpret, because each feature gets a weight.

I trained the same model three ways, on exactly the same data and split:

1. **scikit-learn:** I tried four regularisation strengths (C = 0.01, 0.1, 1 and 10) and kept
   the one with the best validation log loss, C = 1 (0.4745). C = 0.1 and C = 10 were almost the
   same, which suggests the model isn't overfitting much.
2. **Manual PyTorch:** The weights start at 0, the model multiplies
   the features by the weights, turns the result into a probability with the sigmoid, calculates
   the loss, uses `loss.backward()` to get the gradients, and updates the weights. I
   added the same regularisation as sklearn so both solve the same problem. I ran it for 2,000
   rounds and tried three learning rates (0.1, 0.5 and 1.0). 1.0 was best; 0.1 was too slow to
   finish learning in 2,000 rounds.
3. **Standard PyTorch:** the same model built with PyTorch's ready made pieces: `nn.Linear`,
   `BCEWithLogitsLoss` and the `SGD` optimiser, with the same settings. I also started the
   weights at 0 so I could compare it fairly.

Both PyTorch models started with a loss of 0.693, which is what you get by guessing 50% for
everyone, and went down to 0.433, which shows they learned. All the trained models are saved in
the `models/` folder, and the evaluation and dashboard only load them; they never retrain.

### Results on the test set (2,985 bookings, 25 May – 31 Aug 2017)

| Model | Accuracy (cut-off 0.5) | ROC-AUC | Log loss | Business cost (cut-off 0.5) |
|---|---|---|---|---|
| Naive baseline (always predicts 41.3%) | 0.590 | 0.500 | 0.677 | €61,200 |
| scikit-learn | 0.728 | 0.809 | 0.523 | €29,400 |
| Manual PyTorch | 0.727 | 0.809 | 0.523 | €29,525 |
| Standard PyTorch | 0.727 | 0.809 | 0.523 | €29,525 |

The baseline never asks for a deposit, which costs €61,200. Asking everyone costs €44,025. My
model costs about €29,400, roughly half of doing nothing and about €14,600 less than asking
everyone.

### Do the three agree?

**My manual PyTorch and standard PyTorch are basically the same.** The biggest difference between
any of their weights is 0.000005, and they make the same decision for every booking. That shows that the manual loop does what PyTorch's built-in tools do. 

**PyTorch and sklearn make the same predictions but don't have exactly the same weights.** Their
probabilities differ by only 0.0017 on average, and they agree on 99.8% of decisions.

When I looked into it, all the big differences were in categories with very few bookings (between
1 and 35 in the training data). With so few examples, gradient descent learns those weights very
slowly and hadn't finished after 2,000 rounds. sklearn uses a smarter optimiser that gets there
faster. So the difference comes from how the models are trained, not from the model itself, and
it hardly changes the predictions, because almost no bookings fall into those categories.

### Dashboard

Running `uv run python main.py app` opens a Gradio dashboard that only loads the saved models. It
has three tabs: (1) the three models compared, with a chart of predicted vs actual cancellation
rates; (2) charts of the target and of any feature; and (3) a threshold slider that updates the
confusion matrix and the cost in euros.

## Limitations & next steps

| Problem | What I found | What I would do with more time |
|---|---|---|
| **Behaviour changed over time** | In the test period the model expected 46% of bookings to cancel, but only 41% did. The ROC-AUC also dropped from 0.849 to 0.809 | Retrain regularly on the most recent 12 months, and check every month whether predicted and real cancellation rates still match |
| **The best threshold didn't carry over** | The cheapest threshold on validation was 0.29, but on the test set it cost €29,950, slightly more than using 0.5 (€29,400).| Choose the threshold using several time periods instead of just one, so one season doesn't decide it |
| **The test set is only summer** | The results show high season, not the whole year | Test on several periods across different seasons |
| **My target doesn't match my policy exactly** | Early cancellations cost nothing under my policy, but the model counts them like late ones | Create a new target: "cancelled in the last 14 days or didn't show up", and compare the cost |
| **The costs are guesses** | The idea that 1 in 10 guests would leave over a deposit is an assumption | Test it: ask a random group of guests for a deposit and measure how many actually leave |
| **There might be more leakage** | Removing parking and special requests was a judgement call | Ask the hotel when each field is filled in, and only keep the ones known at booking time |
| **Some weights aren't reliable** | Some categories have only 1 to 35 bookings | Group rare categories together more aggressively |
| **Group bookings count once per room** | The 25,902 repeated rows give big groups more weight | With a booking ID, treat each group booking as one decision |



## AI use disclosure

I used Claude through the assignment:
- to explain concepts (leakage, time-based splits, gradient descent, regularisation,
  calibration, business-cost thresholds);
- to write the code for data loading and cleaning, feature engineering, the three training
  implementations, saving and loading models, evaluation, and the Gradio dashboard;

The choices of client hotel, business action, 14-day deadline, features to build and
columns to drop were decided by me.
