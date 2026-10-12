# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Sofia Siso 
- **Student ID**: 19304
- **Email**:ssiso.ieu2022@student.ie.edu
- **Group**: PPLEDBA 5A 

## Dataset
I removed 11 text and link columns from the file to keep the repo small, and the full file is available from Inside Airbnb (had to do this to commit it to the github)

What is it, where did you get it, what does one row represent, how many
rows/columns, and why did you pick it.
The data is the Inside Airbnb "Detailed Listings" file for Madrid
(`listings.csv`), a public collection of Airbnb listings that Inside Airbnb
gathers from the Airbnb website. It is a snapshot from September 2026
(the `last_scraped` dates in the file are around 17-21 September 2026).
I downloaded it from https://insideairbnb.com/get-the-data/
and saved it in the repo as `data/listings.csv`
One row is one Airbnb listing in Madrid
(one flat, room or house advertised on the platform at the time of the scrape).
The raw file has 23,109 rows and 90 columns. Most columns are
either identifiers and links, long free text (descriptions, host bios),
host information, listing characteristics (bedrooms, bathrooms, amenities,
location), booking rules, or review statistics.
After my filtering (see Business framing) I model 3,591 listings:
entire homes with exactly 2 bedrooms, with a listed price of 500 or less.
The nightly price (`price`), stored in the file as text like
"$85.00"
Why I picked it? It is impossible to read the news nowadays without seeing
the words *"fondos buitre"* (vulture funds) or *"crisis inmobiliaria en España"*
(housing crisis in Spain). This project is my way of looking at one very small
aspect of the housing offer in Madrid: Airbnb listings. My model predicts what a normal 2-bedroom flat in Madrid typically charges per night, so a host can see whether their price is far above that. It doesn’t prove or disprove that Airbnb pushes up housing prices. I also chose it because the raw
data is messy enough that feature engineering matters.



## Business / real-life framing

The tool is for people who want to list their flat
on Airbnb in Madrid), and what it tells them is what a normal flat like theirs charges
per night, and a warning if their planned price is far above that. The spanish housing crisis of course is a motivation, because now people are scare dto list their apartments on airbnb, because they wil be acused of driving the priecs up if their partment is listed to expensively. 

**What this implied for the pipeline.**

- **Target definition.** The target is the nightly price. I only model entire homes
  with exactly 2 bedrooms, so the comparison is between similar flats and not
  against villas, single rooms or hotel rooms. This scope fits the business idea; 
  hosts can ask "what do flats like mine charge?"]
- **Price cap.** I dropped listings priced above 500 per night (168 listings),
  because extreme prices are likely errors or luxury listings that
  would distort a linear model, and they dont really fit my idea of a person litsing their airbnb to gain extra money, these luxury flats are, i agree, a problem with the vulture flats so i take them out. The cost is that the tool cannot judge listings above 500.
- **Time-based split: not needed.** The file is a single snapshot from September
  2026, so there is no "future" that a random split could leak into training.
- **Split by host.** I split by host (70/15/15 into train, validation and test),
  so all listings of one host stay on the same side. A new host's flat
  is new to the model, and if one host's similar flats were on both sides the
  model could recognise them and the score would look better than it is.
  I used seed 4, chosen from seeds 0-9 and 42 (11 splits in total) because its
  shares were closest to 70/15/15 (worst gap 0.015). I only looked at split
  sizes, never at model scores.
- **Metric.** The metric is MAE in euros, because it is easy to
  explain to a host, "the model's guess is off by about X euros on average".
  My target is a number (linear regression), so there is no classification
  threshold. The equivalent decision is the "too much" rule below.

The tool warns the host if their planned price is more than X% above the
predicted price. The app has a slider for X, with a default of 50%. I chose this %
because on the validation set it flags about 6.6% of flats,
while 10% would flag 29.4%, mostly because the model is off by about 62 euros
on average. And a slider because the model's error is large, so one fixed number
would look more exact than it is. I chose X using the validation set only, and, 
again chose a high one, also because i didnt want any apartment to be flagged, If i want 
airbnb hosts to actually use this tool its important for me that if they want to gain
extra money they can without it being flagged!



## Data preparation & feature engineering

**Price was text** ("$85.00"), so I removed the "$" and commas and
  converted it to a number.
- **4,324 of 23,109 listings (about 19%) had no price.** I dropped them, because a
  model cannot learn from rows without an answer. I dont think these missing prices were random, but they were irrelevant for my study because they didnt help with the business case. 
- **Six columns were completely empty in this snapshot** (bathrooms, host_since,
  host_response_rate, host_acceptance_rate, instant_bookable,
  host_total_listings_count), so I dropped them.
- **Price in disguise (target leakage).** In the rows I compared,
  `price_quote_price_per_night` was identical to `price`, and the other
  `price_quote_*` columns and `estimated_revenue_l365d` look derived from price.
  I left them out, because the model would read the answer instead of learning it.
- **Bathrooms** were text such as "1.5 baths", so I extracted the number
  ("Half-bath" became 0.5). 74 listings have "0 baths". This probably meant either shared bathroom, error, or simply that what they wanted was to rent was a room, not a whole apartment. 
- **I used `minimum_nights_avg_ntm`** and not `minimum_nights`, because it has no
  blanks (`minimum_nights` had 303).

**Features I engineered, and why.**

- `amenity_count`: the number of amenities listed. I included it because a longer
  list (wifi, kitchen, air conditioning...) suggests a better-equipped flat, which I
  expected to cost more. Listings had between 3 and 90 amenities, with a typical
  value around 30.
- `dist_center_km`: straight-line distance (haversine formula) from the listing's
  coordinates to Puerta del Sol. I included it because I expected flats closer to the
  centre to be more expensive. It was the second strongest link with price in my data
  (correlation -0.28).
- `has_reviews`: 1 if the listing has at least one review. I added it because 420
  listings (about 12%) had no review scores, so I filled those blanks with the training
  median. This flag tells the model which flats are new, so a new listing is not
  mistaken for one with an average rating.
- `bathrooms_num`: described above.
- `district_*`: the district (`neighbourhood_group_cleansed`) as one yes/no column
  per district, with one district dropped to avoid a redundant column.
  Districts are learned from the training set only.

**Feature selection.** I measured the correlation of each numeric candidate with
price on the training set only. Every link was weak (between -0.34 and +0.25).
I kept columns with an absolute correlation of at least 0.10. That gave 12 numeric
columns (accommodates, beds, bathrooms_num, amenity_count, dist_center_km,
calculated_host_listings_count, minimum_nights_avg_ntm, number_of_reviews,
reviews_per_month, review_scores_rating, review_scores_location, has_reviews) plus
20 district columns, 32 in total. The cutoff is arbitrary: `is_superhost` and
`hosts_time_as_host_years` fell just below 0.10 and were dropped by a tiny margin,
along with `is_rental_unit` and `availability_365`. Some kept columns overlap
(accommodates, beds and bathrooms), so the model shares the credit between them.

**Missing values and scaling (training data only).** After the split I filled
blanks in `beds` and the two review scores with the training-set median (3.00,
4.76 and 4.86) and `reviews_per_month` with 0 (a blank means no reviews yet).
I scaled the numeric columns with a standard scaler fitted on the training set.
The medians and the scaler must come from the training set only
(the test set would otherwise influence them), and scaling is needed for the
PyTorch versions because oth PyTorch versions learn by gradient descent: they take 
many small steps, and each step uses one learning rate for every weight. 
That works badly when the columns are on very different scales.



## Modeling: three implementations, one model

**Model.** Linear regression, because my target (nightly price) is a number.
I trained the same model three ways on the same split and the same scaled inputs:
scikit-learn `LinearRegression`, a manual PyTorch loop (raw tensors, autograd,
hand-written update step), and the standard `nn.Module` + `torch.optim` workflow.
Linear regression works here because t uses a clear, simple equation where each variable has a fixed weight or coefficient, making it easy to see which factor drives the output, and uses the same mathematical rule across all data.

**Results (same test set, 498 listings).**

| Method | Train MAE | Val MAE | Test MAE | Test R² |
|---|---|---|---|---|
| Naive baseline (median training price) | 73.78 | 69.89 | 70.88 | -0.02 |
| scikit-learn | 60.88 | 58.67 | 62.09 | 0.29 |
| Manual PyTorch loop | 60.89 | 58.67 | 62.09 | 0.29 |
| `nn.Module` + `torch.optim` | 60.89 | 58.67 | 62.09 | 0.29 |

MAE is in euros per night. All three models beat the baseline by about 9 euros
on the test set.

**Do the three agree?** Yes on predictions. The two PyTorch versions are identical
to within 0.00003 euros, as expected, because they run the same maths with the same
start, loss, learning rate and number of steps. scikit-learn differs from them by at
most 2.39 euros on a single test flat. The fitted weights differ by up to about 5
(at learning rate 0.2; about 19 at 0.05), and the biggest gaps are in the rarest
districts (5 to 12 training flats), whose weights the data barely pins down.
Gradient descent stops a bit short there, but this does not
change the predictions in any meaningful way.

**Tuning**
Tuning was kept light
- Regularization: I tried Ridge with alpha 0.1 to 1000. None beat plain linear
  regression on validation (58.67), so I used no penalty.
- Learning rate (PyTorch): 0.05, 0.1 and 0.2. I chose 0.2 by validation MAE, with
  ties broken by the smaller weight gap to scikit-learn. 

R^2 = 0.29 is week, but it isn’t a sign thats theres anythign wrong. 
The strongest single correlation was only 0.34. A linear model can’t get far from those columns. Also, the baseline was beat on the test set (62 vs 71 EUR). I haven’t compared it to other people’s results, so I won’t claim it’s typical, only that it’s explainable.

## Limitations & next steps

- **Weak model.** Test R² is only 0.29: the model explains less than a third of the
  price differences, and it is off by about 62 euros on average. Photos, view,
  floor and building quality are not in the data. 
- **The "too much" warning must be generous.** Because the typical error is about
  62 euros, a strict rule would only flag the model's own noise. 
- **Capped at 500 euros, 2-bedroom entire homes only.** The tool cannot judge
  luxury listings or other flat sizes. Then i repeat the pipeline for other sizes
  and test a different cap.
- **One snapshot, one seed.** The data is a single scrape (September 2026) and one
  host split, and the test hosts look different from the training hosts (for
  example fewer listings per host). The test score is a noisy estimate. Had to see the
  average over several splits (cross-validation), and add scrapes from other dates.
- **Missing prices (19%).** I dropped them and did not check whether they are
  random. If they are not, the tool only knows listings that show a price.
  *Next:* check the pattern in the missing rows.
- **Borderline features.** The correlation cutoff of 0.10 is arbitrary, and two
  features (superhost, host experience) were dropped by a tiny margin. Some kept
  features overlap (accommodates, beds, bathrooms). Then i compared feature sets on
  validation.
- **The model learns what hosts charge, not what is fair.** If the whole market is
  inflated, so is the model's "typical" price. The tool cannot say whether Airbnb
  raises housing prices. I guess extra data (rents, income by
  district) would help the model. 

## Generative AI use disclosure


I used Claude throughout this assignment.

**Understanding the assignment.** Claude explained the README to me in plain
language: what each file is, what the task requires, how the grading works,
and the difference between forking, cloning and opening a pull request.
It also helped me set up the repo: checking that my local copy points to my
fork, and removing the duplicated nested folder and the extra `train.py` and
`app.py` files that `main.py` does not allow.

**Understanding code and outputs that I did not understand.** Two examples:

1. *The messy `listings.csv`.* When I opened the file in VS Code it looked
   broken, with text spread over many lines. Claude explained that hosts type
   paragraphs with line breaks, that the file keeps that text inside quotation
   marks, and that pandas reads it correctly. It told me to count rows with
   pandas, not by reading the file.
2. *The target-leakage warning, and the `KeyError` and `NameError` errors.*
   Claude pointed out that `price_quote_price_per_night` was identical to
   `price`, so the model would read the answer. It also explained why my
   notebook failed when run through `main.py`: cells depended on code that only
   existed in the notebook's memory (a column I had already deleted, and a
   missing `StandardScaler` import).
   *The uneven split.* My first host-based split gave very different sizes
   (2,399 / 825 / 367 listings, about 67% / 23% / 10%). Claude explained why:
   the split divides hosts, not listings, so a few hosts with many listings
   move a lot of rows at once, and the host counts are the same for every
   seed. It suggested comparing seeds 0 to 9 by split size only, never by
   model score, so that I would not be choosing a split that flatters my
   results. I ran the comparison and picked seed 4, which had the smallest
   gap to 70/15/15 (0.015).
3. Claude helped with with suggesting a list of candidates columns to remove, 
    although i followed my own croteria it was useful to see what the AI thought
4. Claude helped me with calulcating the distance to sol formulas (for my app: 
    the feature distance to center)
5. Claude especially helped me with health checks, for example when creating the app,
   it added a small check after each step so I could see that the step worked before
   moving on: the number of districts and the one district dropped in training
   (21 districts, Arganzuela), a test that the chart functions return charts, and a
   made-up flat in Centro whose three model predictions and warning threshold I could
   verify by hand (224 EUR typical price, so the 50% warning starts at 336 EUR). The same
   happened when saving the models: the check lines showed that models loaded from disk
   give exactly the same predictions as the ones in memory. I then ran the app through
   `main.py app` and confirmed it showed the same results table as training
   (test MAE 62.09).


**Code.** Claude helped me with the last steps especially, the tuning loops,
and the saving of the models, and I ran it and read
its output. 

**Decisions that were mine.** The main coding and intreprettaion. The dataset, the business idea (a pricing tool
for hosts), the scope (2-bedroom entire homes), the price cap of 500, the rule
for choosing the split seed (closest to 70/15/15, never using model scores), the
feature-selection rule (correlation of at least 0.10) and the inclusion of the
district column. 

**REPORT.md.** Claude suggested the structure of each section and filled in the
numbers from my outputs. The explanations and conclusions are my own.
