# Assignment 1 Report

- **Name**: Manuela Carreira
- **Student ID**: 18766
- **Email**: mcarreira.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

The data set comes from a source named insideairbnb.com, it is a data set that has a detailed listing that have all the columns not just the summary version. The dataset has all the listings between June and July 2026, this makes the prices be from this dates. The dataset is inside the repo as data/listings.csv.gz. It is loaded directly by train.py, so every cleaning step can be reproduced from the raw file. 

Each row in the dataset represents an apartment that was published in the platform. And each column is a characteristic of the post. 

Size: 22,708 listings (rows) and 90 columns. After keeping only entire homes there are 15,899 listings, and 13,936 of them have a price.

I chose this dataset because for me now living in Madrid and finding flats or places to stay is a big topic right now even if it is just for vacation and for the summer. I wanted to see how prices are, regulations etc. 

## Business / real-life framing

Like I mentioned before, prices for housing or flats for short term stays in Madrid are a big deal right now, due to the high prices. I made a model that is basically an overpricing detector. It will estimate a listings fair price from what the listing is (all the categories we have). A listing will be flagged (marked as overpriced) if the price goes X% above the "fair" price.

This framing drove three decisions in the pipeline:

- **Target = log(price).** The rule I chose is a price X% above the fair price, so the model predicts the logarithm of the price, which turns its errors into percentage errors.
- **Split by host.** One host has 373 almost identical listings. If they fall into both the training and the test set, the model is "cheating", so all listings of a host go to the same set (train, validation or test).
- **No time-based split.** The data is a single snapshot from June–July 2026, not a history over time, so there is no future to leak into the training data.

## Data preparation & feature engineering

In the dataset the price came written as text and not as a number, so I converted it to a number. Also listings that came without a price were eliminated. Without price it is impossible to classify the listing as fair or overpriced. Extreme prices were kept, the detector searches for these extreme cases. The cost of this decision is that some of them could actually be typos and not real prices. 

There were 12 columns about the host that were completely empty, so I didn't use them. There were other problems, like for example information about bathrooms was missing in about 9% of the listings, but the information was given in another column (bathrooms_text, e.g. "1.5 baths"), so I recovered it from there. When there was a missing value it was filled with the median of the training data. The listings without reviews do NOT have a rating. I filled their rating with the median of the training data before training, and marked it as "not real" with the binary variable has_reviews (1 = it has reviews, 0 = it does not have reviews). 

All of the calculations were made only with the training data. Later these same calculations were applied to the validation and test sets. This is so the test works as data that the model never sees. 

Engineered features: 
29 numeric + 2 categorical features, which become 54 columns after one-hot encoding (each district and property type becomes its own 0/1 column).
The 29 variables are numbers like distance, number of beds, number of bathrooms, number of bedrooms, amenities, rules, host, reviews, demand. And the two categorical ones are the district and the type of property (rental unit, condo, loft, serviced apartment, etc.; rooms were excluded).

Location and size. I expected location and size to be the two main drivers of price. I included both the district and the distance to Puerta del Sol to see how strongly location relates to price, and also how location relates to size, since flats in central areas tend to be smaller. Having both in the same model lets it weigh one against the other: for example, whether a very small flat in Salamanca can cost the same as a much bigger one near Sol.

I decided to include availability and occupancy since both are great indicators of how much demand each listing has during these months. The risk is that a very expensive flat tends
to stay empty, so the model can learn that "empty" goes together with
"expensive" and raise its fair-price estimate for empty flats. That makes it
harder for the detector to flag exactly the overpriced listings it should catch.

I excluded estimated_revenue_l365d, because it is calculated as price × occupancy, so it contains the answer (leakage). Together with occupancy, the model could recover the exact price, and the detector would never flag anything.

## Modeling: three implementations, one model

I used linear regression because I'm predicting a number, the nightly price,
not a yes/no outcome. Specifically I used Ridge, which adds a small penalty to
keep the coefficients from growing too large. All three versions were trained
on the same data and split, and I only tuned the penalty and the learning rate
on the validation set.

| Model | RMSE (log) | R² | MAE (€) | Median % error |
|---|---|---|---|---|
| Naive baseline (€156 for all) | 0.507 | −0.01 | 68.3 | 32.4% |
| scikit-learn Ridge | 0.382 | 0.425 | 51.5 | 22.1% |
| Manual PyTorch loop | 0.382 | 0.424 | 51.6 | 21.9% |
| Standard PyTorch (Adam) | 0.382 | 0.425 | 51.5 | 22.0% |

All three models clearly beat the naive baseline. scikit-learn and the
standard PyTorch version (Adam) ended up practically identical. The manual
loop differed slightly in a few coefficients, mostly in districts with very
few listings. It fell behind because it used the same step size for every
coefficient, so the hardest ones hadn't finished moving by the end of
training. Adam adjusts the step size for each coefficient, so it got much
closer. Training the manual loop for longer would close the gap.

Location matters most, followed by the size of the flat. The further from Sol, the lower the price. Barajas coming out as the most expensive district surprised me because it is not known for an expensive neighborhood.

## Limitations & next steps

**The model only explains part of the price.** Location, size and amenities
explain less than half of why prices differ, because many things that matter
aren't in the data. To improve it, I would add photos of the listing and how
close it is to metro and bus stations or supermarkets.

**Extreme prices include typos.** I kept them on purpose, but some are clearly
input errors, so the detector flags them as overpriced when they're just
mistakes. To fix this, I would correct those typos by estimating what the
listing should really cost based on its other characteristics, like where it
is and how big it is.

**The model can't see events or seasons.** The data is a single snapshot and
the model doesn't know when a flat is rented, so a higher price during a
football final looks like overpricing. To fix this, I would use a calendar of
prices per date to account for events and seasons.

**The manual loop didn't fully converge.** It used the same step size for
every coefficient, so some hadn't finished moving by the end of training.
To fix it, I would train it for more epochs or use a larger learning rate.

## Generative AI use disclosure

I used Claude (Anthropic) throughout this assignment. It helped me set up the
environment (uv, Git, VS Code) and wrote the code for `train.py` and `app.py`:
loading and cleaning the data, the feature-engineering code, the three
training implementations, saving the models and the Gradio dashboard. It also
prepared the factual parts and tables of this report from the pipeline's
results, explained the concepts I didn't understand (like leakage, log
transforms and why gradient descent converges slowly), and turned my notes
into clear English.

The decisions were mine: the dataset, the overpricing-detector framing,
focusing on entire homes, keeping extreme prices, including availability and
occupancy, and keeping monthly rentals with a flag. The reasoning in the
report, such as why the features matter, the limitations and how I would fix
them, is based on my own answers, which Claude helped me phrase.

