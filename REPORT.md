# Assignment 1 Report
- **Name**: Sofía Serantes Vázquez
- **Student ID**: 19523
- **Email**: sserantes.ieu2022@student.ie.edu
- **Group**: BBADBA 5A 

## Dataset

The Ames Housing dataset (De Cock,2011), a widely - used dataset of 2,930 individual home sales in Ames, Iowa between 2006 and 2010. 
One row represents one house sale, with 82 columns describing the property's physical characteristics ( size, quality, age, location, amenities) and the sale itself (price and date). I used the full original version pulled from a public GitHub of De Cock's data. 

I picked this dataset because it has a lot of information (around 80 columns) that give room for feature engineering and because the 2006-2010 sale window spans the 2008 housing crash, which forced a real decision about how to split the data rather than a default random split. 

## Business / real-life framing

The model supports a real-estate investor deciding whether to buy a house or not. This decision is based on wether it's priced like a good deal relative to comparable properties, not on its absolute price. This is a binary classification problem rather than a price-prediction one. The question is not "what is this house worth" but "is this house underpriced enough worth looking into further". 

*The target definition:* There is no "good deal" label in the raw data, so I built one. For each house, I computed (price_per_sqft = SalePrice/ Gr Liv Area) and grouped houses into comps by Neighborhood x Overall Qual Tier (low/mid/high) x Yr Sold. Within each comp group, the median of the variable price_per_sqft is that group's "fair value" benchmark. A house is label buy = 1 if its actual price_per_sqft is more than 10% below its comp group's median, else 0. This gave me a 26.8% buy rate ( 785 of 2,930 houses. This data is imabalced but we can work on it)

*Time-based split:* Since comp groups include Yr Sold, a random split could put two houses from the same comp group on opposite sides of the train/test boundary. That means a training house's label would have been partly computed using a held-out test house's price and vice versa. Splitting by year instead closes the leak because no comp group can span both sides of a split that cuts exactly along the same variable (YrSold) the comp groups are defined by. It also makes the model more realistic because it is trained with historical sales(from 2006-2008) and then applied to future listing (2009-2010)

*Decision Threshold:* Because the classes are imbalaced (73.2% is labeled as "no buy"), a naive model that always predicts "no buy" already reaches 74.1% accuracy on the test set with zero recall. It would never flag a single real deal for the investor. 
Accuracy alone is therefore the wrong metric to optimise or report only. This is the reason I also used F1 and ROC AUC as the primary comparison metrics. 

## Data preparation & feature engineering
*Excluded features for the model:*
- SalePrice, price_per_sqft,expected_psf,group_size: These are the columns I used to build the buy/no-buy label. Using them as features would be redundant becuase the model would know if the deal is good or not.
- Yr Sold, Mo Sold: The comp groups were built using Yr Sold, so it's tangled up with the answer similar to the problem with the previous features. Also this variables see disjoint year ranges, so keeping them would let the model key off "which year" rather than real house characteristics. 
- Order,PID: They are row identifiers.
- Garage Yr Blt: A missing value in this row means "no garage" but filling it with 0 would create a misleading "year 0" for a model to learn from. There is a variable "Garage Type" that says "None" when there is no garage so the information is captured cleanly elsewhere. 

*Handling missing values:* There are many Nans in this dataset that mean "this house does not have that feature", not "data is missing". I treated these as an explicit "None" category for categorical columns (Alley, Bsmt Qual, Garage Type, Fireplace Qu, etc.) and 0 for their numeric counterparts (Total Bsmt SF, Bsmt Full Bath, Garage Area, etc.) rather than imputing a mean/mode, which would have invented a fake basement quality for houses that have none. 

*Categorical encoding:* Neighborhood (28 levels) and all other categorical columns were one-hot encoded. I chose one-hot over target encoding specifically to avoid an extra leakage risk and because it is the most transparent choice for a linear model because each coefficient has a direct interpretation. 

Created Features: 
- house_age = Yr Sold − Year Built
- years_since_remodel = Yr Sold − Year Remod/Add
- total_sqft = Total Bsmt SF + Gr Liv Area (combines basement and above-ground living space, which are separate raw columns)
- total_bathrooms = Full Bath + 0.5×Half Bath + Bsmt Full Bath + 0.5×Bsmt Half Bath
- qual_x_cond = Overall Qual × Overall Cond

All numeric features were standardized (zero mean, unit variance) using statistics computed on the training set only, then applied unchanged to the test set.

## Modeling: three implementations, one model
I used logistic regression since my target variable is a binary outcome (buy/not buy), not a continuous number. 

| Method | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Naive baseline (always "no buy") | 74.1% | 0.00 | 0.00 | 0.00 | 0.50 |
| scikit-learn | 81.2% | 0.65 | 0.59 | 0.62 | 0.85 |
| Manual PyTorch loop | 81.3% | 0.67 | 0.55 | 0.60 | 0.85 |
| PyTorch `nn.Module` + `optim` | 80.1% | 0.61 | 0.62 | 0.62 | 0.85 |

At first I did not agree with the results. With C=1.0 (a common default regularization strength), scikit-learn reached 88.8% accuracy on the training set but only 45.5% on the test set, this meant severe overfitting. With 275 features (mostly one-hot columns) and only 1,941 training rows, weak L2 regularization let scikit-learn's solver converge fully to a solution that memorized training-set noise. The two PyTorch versions, run for a fixed number of gradient-descent steps rather than to full convergence, never reached that same overfit optimum, which accidentally acted like early stopping and generalized better, even though it wasn't principled regularization.

I diagnosed this by sweeping scikit-learn's C over {1.0, 0.1, 0.05, 0.01, 0.005} and tracking train vs. test accuracy. Test accuracy peaked around C=0.05 (~81%), matching where the PyTorch versions already were. I then set C=0.05 for scikit-learn and matched it in both PyTorch versions via an equivalent weight_decay = 1/(C × n_train) term, and increased the PyTorch training length so they'd actually converge rather than stop early by accident. After this, all three methods landed within ~1 point of each other on every metric. 

The one hyperparameter I tuned was this regularization strength, following the "tune sparingly" guidance. I did not grid-search the learning rate beyond confirming 2,000 steps at lr=0.1 was enough for both PyTorch versions to converge to the same place as scikit-learn's closed-form-equivalent optimum.

## Limitations & next steps
*Sparse comp groups bias the label toward "no buy."* As described above, 88 of 237 comp groups have fewer than 5 houses. With more time, I'd add a fallback hierarchy (drop Yr Sold, then drop qual_tier, in that order) for any group below a minimum size, so every house is compared against a reasonably-sized reference set instead of sometimes just itself.

*No validation split, hyperparameter tuning touched the test set.* I tuned scikit-learn's C by directly observing test-set accuracy at each value, rather than holding out a separate validation slice (e.g. 2006–2007 train / 2008 validation / 2009–2010 test). For a single, sparingly-tuned hyperparameter this is a minor simplification, but it means my reported test metrics are very slightly optimistic — a stricter setup would tune on a validation split and touch the test set exactly once at the end.

*The $3,000 due-diligence cost in the dashboard is an assumption, not derived data.* The Gradio app's business-cost estimate (profit captured minus diligence cost per flagged listing) uses a fixed, invented per-listing cost. With real data on actual inspection/appraisal costs, or input from an actual investor, this number could be grounded rather than assumed, and the net-value estimate would be more defensible.

*Logistic regression can't capture interactions I didn't engineer by hand.* qual_x_cond was one deliberate interaction term, but a tree-based model would likely find others automatically and probably outperform this approach on raw predictive power. I stayed with logistic regression because the assignment's point was comparing the same model across three implementations, not maximizing accuracy.

## Generative AI use disclosure
I used Claude (Anthropic) throughout this assignment as a collaborative tool, with the division of labor the syllabus AI policy describes: I made the decisions (business framing, target definition and its threshold, split strategy, which features to engineer, how to interpret the three-method comparison discrepancy), and asked Claude to write the corresponding code, explain concepts I didn't understand yet (e.g. why a random split would leak, what a comp group is, what each engineered feature does), and help debug environment/git issues while setting up the project locally. Claude also helped me diagnose the scikit-learn overfitting discrepancy by running a regularization sweep, which I then interpreted and decided how to resolve. The analysis, conclusions, and this report's wording are my own, written after working through each decision in conversation rather than having it handed to me.
