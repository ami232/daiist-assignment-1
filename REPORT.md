# Assignment 1 Report

- **Name**: Clara Zsigmond
- **Student ID**:
- **Email**: crubinstein.ieu2022@student.ie.edu
- **Group**: PPLEDBA 5A

## Dataset

- The dataset I chose is the "Brazilian E-Commerce Public Dataset by Olist" on Kaggle (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_orders_dataset.csv). It is a real-world, anonymized commercial dataset containing 100,000 orders placed between 2016 and 2018 across multiple marketplaces in Brazil. 
- Each row in my dataset represents an order. The dataset I chose is actually a combination of 9 datatsets. We have seperately customer data, product data, seller data, etc. I chose the features I thought were relevant, tested, retested, and finally defined and built a final single tabular dataset to use for this project. Each row represneting an order is a conseuqnece of the deicsions I made here, where in the original tables a row had different meanings depending on the table you were looking at. I found that each product individually was not a relevant feature to consider here, which is why I could have a table where eahc row is a order even if an order has multiple products: because the number of products is more relevant than the porducts themselves. This is patr of what made feature engineering and selection important here

- I picked this dataset because I was interested in understanding what features impacted late delivery, which is a big issue for anyone who buys things online! This database seemed robust to me, containing 100k orders from all over Brazil and enough features that a selection process would be interesting and non-trivial. I also thought it would be interesting to work with many tables and kind of play around to understand what felt important vs what actually WAS important, which proved to be very different. I started out just choosing features I thought sounded important, but after analyzing them more deeply I found that while some were, some really weren't. That was the case for month, for example, which i started out thinking was a relevant feature as months with holidays or black friday may see a spike in orders and therefore can suffer to deliver on time, but ultimately it was not a relevant feature.

- My final clean dataset has 20000 rows and is a stratified sample of the original, following the assignment size guidelines to avoid pushing a overly large DS to the repo. After joining the tables to select the features I wanted, I ended up with 5 feature groups and 18 columns, which I will explain in thre third section of this report.

## Business / real-life framing

This dataset could be chosen to serve many different purposes and business problems. In this case, I chose to leverage its extensive data to predict if an order will be late. This is a very relevant metric for companies, and can help them act proactively by alerting customers, changing the estimated delivery date, or changing carriers/delivery method to ensure timely delivery when a order is flagged as at risk for late delivery. This scenario informed many of my decisions.

- My target (order being late or not) was not an existing column in the original dataset, so I built it by using two existing columns: late = delivered > estimated date. 

- Because of my target choice, I also defined that the prediction of if an order would be late would be made at the time of purchase. Looking into the past, when the customer is making a purchase, Olist would be able to immediately know if the estimated delivery date was too ambitious and flag internally for review. This allows the company to warn the customer, push back the estimate, or switch carriers with pleanty of time and can help inform other logistical decisions.

- Decision made at purchase time also means only features *known at purchase time* can be used in training, because they are all that will be available at the time the model actually runs. This was a big part of feature selection.

- This is a model meant to use past data to predict future outcomes, making a time-based split the natural decision here. I chose to train on january 2017 to march 2018, validate from april to may 2018, and test on the final split of june to august 2018. This means everything that happened before helps the model know how to predict the possible next occurrences. If the split had been random using the entire timeframe, we would be using part of the test months as training data, which could result in leakage. If I am testing how the model performs in future data, but I have some of that same data to train the model, that is not realistic of the production version of this model.

- Trying to predict late deliveries means orders that were never delivered are not informative for our goal, so those were dropped.

- Another element that was impacted by the choice of target are the relevant metrics to evaluate at the end. I faced an issue of class imbalance here where realistically only ~8% of all orders were ever late, so accuracy became a very tricky and misleading metric. Just guessing no late orders would give 92% accuracy to the model. More accurate quality judges are log loss, PR AUC and cost. 

- Cost brings me to my last impact of the business framing: the cost of a missed late order is higher than a false alarm. We are talking about late deliveries, and in this context not flagging a late order can cause customer churn, requests for a refund, bad reviews, etc. These are very costly consequences for the business. Flagging a on time order as late also poses costs such as covering expedited delivery or sending an update to the customer, but it is still less costly then losing customers or actively dissuading potential customers from trusting your business. That's why I chose to define a missed late order as having a cost of 5, while a false alarm carries a cost of only 1.

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

#### Data preparation and quality decisions
- 2016 had very little data with only a few orders in a few months, so I removed it from the sample
- I also removed orders that were never delivered, because of my business challenge target being explicitly delivered orders that were late, not that never arrived.
- one hot encoding: i want to understand the effect of state on late deliveries (as in delivery state rj, sp, ce, etc). To do so, I used one hot encoding to creat a matrix where each column is a possible state and each row has exactly one 1 (the rest are 0s) in the state that its being delivered to. 
- To avoid any leakage, I only used features known at purchase time to train the model, this caused me to exclude some features such as approval delay (I was considering creating a a column for how long it took for an order to get approved, but that is only known after purchase)


#### Feature engineering
- I tested 13 feature groups and ultimately kept 5: `promised_days`, `customer_state`, `log_distance_km`, `log_freight` and `n_sellers`. I chose them using **backward elimination**:
    - I started with all 13 groups. Each round, I tried removing each group, refit the model, and dropped the group whose removal gave the best validation log loss. When every remaining removal would worsen the log loss by more than 0.0005, I stopped and kept those groups.
    - I picked a tolerance of 0.0005 because validation only has 156 late orders, so differences smaller than that are mostly noise. When two models are equally good within noise, I prefer the one with fewer features: it's simpler to explain and has less that can break when the data changes.
    - This is where I found that some features I expected to use did not actually contribute to model performance. Month was a strong contender for me for the reasons explained previously (months with events like Black Friday encourage shopping, which can create a shortage of carriers, etc.). But on its own it had a ROC AUC of 0.45, worse than chance! Removing it gave the biggest improvement of the whole process, with validation log loss going from 0.2019 to 0.1959. With only about 15 training months, each month value appears once or twice, so the feature acts more like a timestamp than a seasonal signal: when the model saw a spike in one year it learned it, and when the pattern did not repeat in the validation months the model performed worse.
    - I also identified features I had overlooked. `promised_days` was weak on its own (ROC AUC 0.54), but in combination with the others it was the most costly to remove: dropping it from the final model worsens validation log loss by 0.0137, more than any other group. Once distance is in the model, `promised_days` captures how much "slack" the promise leaves for that specific trip.
    - The final 5 groups score a validation log loss of 0.1962, ROC AUC of 0.756 and PR AUC of 0.163, compared with 0.2186 / 0.5 / 0.055 using no features at all. The feature selection uses exactly the same preprocessing as `train.ipynb`, and the numbers match.

- Other decisions:
    - **Distance:** I measured this as the haversine distance between the seller's and the customer's zip-code centroids, averaged over the order's items (an order can have items shipped from different sellers). Haversine gives the great-circle distance in km, the shortest path over the Earth's surface, using a radius of 6,371 km. Simply subtracting latitude and longitude wouldn't work, because a degree of longitude covers fewer km the further you are from the equator, and Brazil spans from about 5°N to 34°S. I made this choice after some research on how to find distance from ZIP codes in Brazil, and honestly the biggest gain might be interpretability: it tells you "this order travelled about 900 km", which is something we can work with better than other measures that could have a similar impact on the model.
    - **Skewness:** I measured the skewness of my numeric columns and used log1p (log of 1 + x, which also handles zeros) where they had long right tails, so a few very expensive or very long-distance orders don't dominate after scaling. Freight (cost of delivery, or *frete* in Portuguese) skew dropped from 7.94 to 1.21 and price from 7.91 to 0.27. Distance was less skewed (1.64), but I logged it too because the difference between 50 km and 500 km matters more than the difference between 2,000 km and 2,450 km.
    - **Cap:** `n_sellers` is almost always equal to 1, so to avoid a few rare orders with 4 or 5 sellers stretching the scale, I capped it at 3. It is still skewed after the cap (12.13 → 10.26), simply because nearly every order has one seller.
    - These transformations (log and caps) are fixed formulas applied row by row, not learned from the data, so it is safe to apply them before the train/validation/test split.
    - To be honest about their impact: on the same 5 groups, log and caps improved validation log loss only slightly (0.1977 → 0.1962), and ROC and PR AUC barely changed. Most of my gains came from *choosing* the features (dropping month, adding distance), not from shaping them.

- Preprocessing (everything here is fit on the training data only, then applied to validation and test the same way a new order would be processed in production):
    - **Median imputation:** 67 training orders have no distance, because their zip code has no coordinates in the geolocation table. Logistic regression can't handle blanks, so I filled them with the training median, which, unlike the mean, isn't pulled up by a few very long-distance orders.
    - **Standard scaling:** each numeric column is converted to (value − train mean) / train standard deviation, so all features are on the same scale. This matters because gradient descent uses one learning rate for every weight, and the L2 penalty shrinks every weight equally, which is only fair if the features are comparable. It also means each weight can be read as "the effect of one standard deviation more of this feature".
    - **One-hot encoding of `customer_state`:** states have no natural order, so numbering them (SP = 1, RJ = 2…) would invent one. Instead, each state gets its own 0/1 column and its own weight. States with less than 1% of training orders are pooled into one "infrequent" column, so the model doesn't learn unreliable weights from a handful of orders, and a state never seen in training simply gets all zeros instead of causing an error. This gives 13 state columns + 1 "infrequent" column = 14 columns, which with the 4 numeric features makes **18 features** in total (18 weights + 1 bias).

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

1. Logistic Regression:
    - The most important factor dirving my decision here is my target: will this order be late or not. That is a binary target: Yes or No. Linear regression draws a straight line, so it can predict −0.3 or 1.4, which are not probabilities. Here I need probabilities because the deicsion the model drives is to flag if the probability is above a certain threshold.
    - This is also important for the comparison because logistic regression's loss is convex meaning it has a single lowest point. Thiis implies that no matter the method used to sreach for the best weights, there is a single best answer. If all three implementations are correct, they must end up in the same place. If they disagree by a lot, one of them is likely to have some bug.

2. Tuning:
    - I only tuned two hyperparameters as recommended: the regularization strength C and, for the two PyTorch versions, the learning rate. Both were chosen on validation log loss and without looking forward into the test months.
    - **C** controls how strongly the L2 penalty pulls the weights toward zero (smaller C = stronger penalty). I tried 0.001, 0.01, 0.1, 1, 10 and 100 with scikit-learn. A strong penalty clearly hurt (validation log loss 0.2099 at C = 0.001), but from C = 1 upwards the results were flat (0.1962 vs 0.1963). With about 13,000 training orders and only 18 weights there is little room to overfit, so the penalty barely matters. I kept C = 1, which is also the scikit-learn default. 
    - **Learning rate** is the size of each gradient descent step. Because the loss has a single minimum, it doesn't change where the model ends up, only whether it gets there within the 5,000 steps. I tried 0.05, 0.1, 0.5, 1 and 2 with the nn.Module version: the smallest one was still slightly short of the minimum after 5,000 steps (train loss 0.25917), while 0.5 and above all reached about 0.25888. I chose 2.0. A step that large works without overshooting because the features are standardized. scikit-learn has no learning rate because its solver (lbfgs) chooses its own step sizes.

3. Results table:
| Model | Log loss | ROC AUC | PR AUC | Precision | Recall | TP | FP | FN | TN | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| scikit-learn | 0.1828 | 0.636 | 0.066 | 6.2% | 17.9% | 25 | 380 | 115 | 3,349 | 955 |
| Manual PyTorch | 0.1829 | 0.636 | 0.066 | 6.2% | 17.9% | 25 | 381 | 115 | 3,348 | 956 |
| nn.Module + optim | 0.1829 | 0.636 | 0.066 | 6.2% | 17.9% | 25 | 381 | 115 | 3,348 | 956 |
| Baseline: constant (train late rate) | 0.1716 | 0.500 | 0.036 | – | 0% | 0 | 0 | 140 | 3,729 | 700 |
| Baseline: never late | – | 0.500 | 0.036 | – | 0% | 0 | 0 | 140 | 3,729 | 700 |
| Baseline: always late | – | 0.500 | 0.036 | 3.6% | 100% | 140 | 3,729 | 0 | 0 | 3,729 |

**Do they agree:**
| Comparison (validation) | Max difference in probability | Max difference in weights |
|---|---|---|
| sklearn vs manual PyTorch | 0.0077 | 0.029 |
| sklearn vs nn.Module | 0.0077 | 0.029 |
| manual PyTorch vs nn.Module | ~0 (identical) | ~0 (identical) |

- Manual vs nn.Module are identical:
    - Same algorithm (full-batch gradient descent), same start (all zeros), same lr, same 5,000 steps, same penalty, same data. The only difference is clamp + manual log loss vs BCEWithLogitsLoss, which only differ at the level of tiny rounding errors. This proves that optim.SGD with weight_decay is doing exactly what the hand-written update does.

- Both vs sklear are very close but not identical:
    - Max difference in any validation probability: 0.0077. Max difference in any weight: 0.029.
    - This can be explained because they use different search methods and gradient descent hasn’t quite reached the bottom after 5,000 steps. In the output this is clear: loss 0.2593 at epoch 500 and 0.2590 at the end, so we stop while it's still creeping down when we do it manually, and scikit learn goes all the way.

- The most important question is if the difference matters, and the answer is no.
    - All three approaches chose the same threshold (0.20).
    - On test they differ by 1 false alarm (380 vs 381), with the same 25 caught and 115 missed.
    - They all have essentially the exact same cost at the threshold (955 - 956).
    This all shows the difference is minimal


## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

I spent a lot a lot of time overcomplicating things because I thought everything had gone wrong due to a simple reason: on test my model does not beat "doing nothing". On test the model thinks about 10% of orders will be late, but only 3.6% are. It predicts almost 3× too much lateness. So many orders cross the 0.20 threshold, giving 380 false alarms to catch just 25 late orders. At 1 per false alarm and 5 per miss, that costs more than flagging nothing. Simply put, just guessing no orders are late would be less costly than the result of my model. On validation, my model still beats doing nothing: cost 713 vs 780 for “never late.” But on test it loses by 955 to 700 using my 5-1 cost explained above. Its log loss (0.1828) is also worse than just predicting the training late rate for everyone (0.1716). 

Here's the truth: in the time split meant to test the model we trained on earlier months, it seems Olist did much better at delivering on time. They lowered their "promised days", which we believed would indicate more late deliveries, but due to other factors they started delivering more often on time. They may have changed something logistically or factors that were not present in the data or captured by the model. Maybe they increased there number of carriers or were better at delivering multiple orders in the same region. I can't know for sure, but it impacted the models performance greatly because what had previsouly indicated a late order now did not mean the same.

It is still not a useless ranking. We have to remember the metrics that were important. ROC 0.64 means that if you pick one late and one on-time order at random, the model gives the late one a higher probability 64% of the time (chance is 50%), and PR AUC 0.066 vs 0.036 for random is almost double.

One of the biggest features changed meaning, and that’s concept drift: the relationship between a feature and the target changed over time, not just the overall late rate. The model learned a rule from 2017 that stopped being true. This is actually an interesting factor, and it only appeared because of the time based split choice. If I had randomy taken a sample, this might have stayed hidden. The main thing I would do to address this would involve both more time and more data:
    - since a relationship changed, I would need a more recent period of data to capture this changing relationship for future predictions
    - maybe the data lacked the most relevant features like # of carriers, including it or other relevant features wold likely improve the model even to future changes, making it able to capture them
    - ultimately, the biggest possible change would be building a better feature from what we have. Instead of raw promised_days, we could test “promised days minus the typical delivery time for this route” (e.g. seller state → customer state), computed on train. That measures the slack directly, which is what really causes lateness, and it should stay meaningful even if Olist changes its promise policy.
    - if this were real time, we could use a rolling validation window to keep our results accurate
    - the choice to decrease the size of the data was made to keep it reasonable for the project, but it might also cause small changes to be noisy especially given the class imbalance. With more time, I would explore what the results look like using the full data set.


## Generative AI use disclosure

- All of the decisions were my own, but AI (Claude) was very helpful in debugging and writing some of the code. I started most of the training (train.ipynb) and the EDA (in feature_selection.ipynb), brought in the scikit learn linear regression and manual PyTorch as we had done in class, but then had AI help me debug what wasn't working and better organzie and order the code, as well as write parts I was unfamiliar with. I wrote and organized the headings and explanations at first, but had Claude help me review them and polish it within the notebook.
- Everything in the REPORT.md is my own. No AI. (Exception: I didn't know how to properly fomrat a table in .md, so AI helped me format the table in the modeling part)
- the dashboard build code was done with AI.