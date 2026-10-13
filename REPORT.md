# Assignment 1 Report

- **Name**: Daniel Garcia Laya 
- **Student ID**: 19122
- **Email**: dgarcia.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

I use Inside Airbnb (insideairbnb.com/get-the-data) a project that offers quarterly scrapes of every Airbnb listing in a city licensed under CC BY 4.0. I selected Valencia. Downloaded the detailed listings file for three consecutive quarters. The March snapshot was scraped on 28 March 2026. Contains 4,930 listings. The July snapshot was scraped on 3 July 2026. Contains 7,851 listings. The September snapshot was scraped on 25 September 2026. Contains 7,968 listings. All three snapshots contain the 90 columns.

One row represents one listing as it appeared on the day of the scrape. The row includes the price, availability, reviews, room type, location, amenities and host details. The files are around 10 MB in total so the files fit well in this repository and the entire pipeline can be rebuilt from zero.

The data contains no target variable. I created one: whether a listing remains on the platform in the next snapshot. I chose this dataset because Airbnb is a marketplace, which matches the thesis topic. Keeping hosts is also a business challenge for any platform that relies on supply. The dataset also requires feature engineering because many useful columns appear as text, dates or coordinates.

While exploring the data I found two data problems that changed the pipeline. First the field host_since is empty in all three snapshots and the fields price and estimated_revenue_l365d are empty in the March file. The fields cannot be used as features. I measure host experience with the fields hosts_time_as_host_years and hosts_time_as_host_months instead. Second the March scrape contains listings with a minimum stay of seven nights or less while the July and September scrapes also include about 2,300 listings that require longer stays. That explains why March has fewer listings so the difference is due to the scrape, not growth before summer. To compare like with like I keep only short stay listings (seven nights or less) in both periods: 4,923 listings from March to July and 5,501 listings from July to September. Short tourist stays are also where the platform earns most of its commission. This fits the business case.

## Business / real-life framing

**Scenario.** The host success team at Airbnb in Valencia wants to know, every quarter, which listingsre likely to leave the platform before the next quarter. That way an agent can call those hosts. Help with pricing, availability or reviews. I assume that one call costs €20 in staff time and keeps one in three hosts who would otherwise leave. When a host leaves Airbnb loses its commission on that listings bookings. In July the median listing earned about €8,800 a year one quarter at a 15% commission is about €330 and I use that amount as the value of one lost host. I would rather use each listings revenue but revenue is empty in the March file so using one average value is the honest option. All three numbers are assumptions. They make the decision concrete.

**Target.** Inside Airbnb publishes a scrape of every Valencia listing each quarter. A listing is marked as having left (target = 1) if it appears in one snapshot and is missing from the next snapshot and is marked as having stayed (target = 0) if it appears in both snapshots. I build this target myself so the target has a weakness: a host who hides a listing, for a while also appears as having left. Among short stay listings 6.2% left between March and July and 6.4% left between July and September.

**Split.** I split by time not randomly. The model learns from listings as they appeared in March and whether they were gone by July and the model is tested on listings as they appeared in July and whether they were gone by September. That is how the team would really use the model: trained on the past and applied to a quarter the model has never seen. A random split would mix both quarters so the model would be graded partly on the period it learned from. Would appear better than it really is.

**Threshold.** Accuracy is the wrong measure because so few listings leave. A model that predicts nobody leaves is already ninety‑three point six percent accurate on the test quarter. Helps no one. Instead I choose the decision threshold that gives the lowest total cost on a validation set taken from the training period. Total cost is twenty euros for every host we call plus three hundred thirty euros for every host who leaves without a call plus two thirds of three hundred thirty euros for hosts we call but still lose. A call pays off when the chance of leaving times one third of three hundred thirty euros is than twenty euros, which means a probability, above zero point one eight. I look at the test quarter once at the end to report the final result.

## Data preparation & feature engineering

**Data problems I had to deal with.** The March file only has listings with a stay of 7 nights or less so I removed longer stays from both periods. Otherwise the model would be tested on a kind of listing it never saw. Price and revenue are empty in March so I could not use them as features and for the business cost I used one value per lost host (€330) instead of each listings own revenue. "Host_since" is empty everywhere so I rebuilt host experience from the years and months each host has been on Airbnb.

**Features I built.** Everything comes from the listing at the start of the period. The model only uses what the team would know when making the calls.

- Room type (entire home, private room other). 16% Of private rooms left against 4% of entire homes.

- Bathrooms read from text like "1.5 shared baths" plus a shared bathroom flag.

- Number of amenities counted from a text list.

- Whether the listing has a tourist license. 16% Of unlicensed listings, against 3% of licensed ones.

- Host experience, superhost verified identity and how many listings the host manages.

- Free days in the next 30 and 365 days and the share of the year booked.

- Reviews: total last 12 months last 30 days, days since the last one years since the first one and the rating.

- Minimum stay, distance to the city centre and district.

I used logs for skewed counts like reviews so a few extreme values do not dominate.

**Preprocessing.** 20% of the March to July listings are kept for validation. Missing values are filled with the median numbers are standardised and districts are one hot encoded. All of this is fitted on the training rows so nothing leaks from validation or test. Standardising also helps the PyTorch versions converge.

**Did it help?** With 12 raw columns the validation ROC AUC was 0.786. With my features it was 0.845.

**What did not work.** A flag for rentals was useless since March has none so I dropped it. Several features that worked well in spring were much weaker at the end of summer which I come back, to in the section.

## Modeling: three implementations, one model

The target is yes or no logistic regression is used. Logistic regression gives a probability. That probability must be compared to the cost of a call. The coefficients of regression are easy to explain.

To make all three versions the same model they minimise the thing: log loss plus an L2 penalty. The penalty strength C = 0.03 was chosen on validation data with scikit‑learn. The same penalty is used in both PyTorch versions. The intercept is not penalised, like scikit‑learn.

Both PyTorch versions start from zero. Both use all training rows at every step. Both run 3,000 steps with a learning rate of 1.0, which was the best of the three tried. Class weights are not used, because real probabilities are needed for the cost calculation.

**Results on the test quarter (July to September, 5,501 listings, 353 left):**

| Model | ROC AUC | PR AUC | Threshold | Hosts called | Leavers caught | Total cost |
|---|---|---|---|---|---|---|
| Baseline: call nobody | 0.500 | 0.064 | | 0 | 0 | €116,490 |
| Baseline: call everyone | 0.500 | 0.064 | | 5,501 | 353 | €187,680 |
| scikit-learn | 0.704 | 0.134 | 0.18 | 331 | 50 | €117,610 |
| Manual PyTorch loop | 0.704 | 0.134 | 0.18 | 332 | 50 | €117,630 |
| nn.Module + optim | 0.704 | 0.134 | 0.18 | 332 | 50 | €117,630 |

**Do the three agree?** Yes. The largest coefficient difference between scikit‑learn and the manual loop is 0.008 and no test probability differs by than 0.003. The two PyTorch versions are identical which is same start, same loss, same learning rate, same number of steps. The small gap with scikit‑learn exists because scikit‑learn solves the problem exactly while PyTorch stops after a fixed number of steps. That gap moves one listing across the threshold, which's the €20 difference. All three choose the threshold, 0.18.

**What the model learned.** Hosts with listings, superhosts, experienced hosts, licensed listings and good ratings are less likely to leave. Private rooms, shared bathrooms and an empty calendar next month point towards leaving.

**The honest result.** On validation from the quarter as training the ROC AUC is 0.845. On the quarter the ROC AUC drops to 0.704, which a random split would have hidden. The model ranks listings better than chance (PR AUC 0.134 versus 0.064) but at the chosen threshold the model costs €1,120 more than calling nobody. Under my assumptions the model is not good enough to use mainly because what drives leaving in spring is not the same, as what drives leaving at the end of summer.

## Limitations & next steps

- **Only one training quarter.** The model learned from spring and was tested on summer and the signals changed between the two. With time I would ask Inside Airbnb for older snapshots train on several quarters and retrain every quarter.

- **Noisy target.** Some listings that disappear are only hidden for a while. Of the 307 listings that left between March and July 34 were back in September. With a fourth snapshot I would only count a listing as gone if it stays gone for two quarters in a row.

- **New listings.** 1,000 test listings were new since March and 14% of them left, against 5% of older ones. The model cannot see that a listing is new because there is no snapshot before March. An earlier snapshot would let me add that feature.

- **Guessed costs.** The €20 call, the one in three save rate and the €330 per lost host are assumptions. The save rate matters most and a small trial calling a group of flagged hosts would measure it. Once revenue is available in every period I would value each host by its revenue.

- **Only short stays.** Because of the March scrape the model says nothing, about rentals.

## Generative AI use disclosure

I used Claude (Anthropic) to help me find a complete and suitable dataset by suggesting different options, assist with some parts of the code, and help structure some of the answers and sections of the report.
