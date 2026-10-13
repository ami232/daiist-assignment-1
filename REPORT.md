# Assignment 1 Report

- **Name**: Alexia Lara Chacon
- **Student ID**: 18964
- **Email**: alara.ieu2022@student.ie.edu
- **Group**: BBADBA 5B 

## Dataset

I chose the Ames Housing dataset from De Cock (2011), ‘Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project’, published in the Journal of Statistics Education 19(3).
- **Source**:  Downloaded from jse.amstat.org via `scripts/download_data.py`, saved as `data/AmesHousing.csv`.
- **Size**: 2930 rows × 82 columns.
- **One row**: one residential property sale in Ames, Iowa, from 2006-2010.  
- **Why I picked it**: I chose it because housing and real estate is a topic I am genuinely interested in and I consider it easy to understand for me when coding a project with it. I can easily correlate or understand what each column means, and I corroborated that it was a proper dataset to use for this project with Claude. 
- **Things I noticed in the raw data**: 
  - Lot Frontage has 490 truly empty cells (17%).
  - SalePrice skew is 1.74 raw and about 0 after taking the log.
  - 245 sales are "Partial" (mostly new builds), with a median price of $250,000 vs $159,000 for Normal sales.
  - In columns like Alley, Bsmt*, Fireplace Qu, Garage*, Pool QC, Fence and Misc Feature, "NA" means the feature is absent, not missing, so I kept it as a category.
 

## Business / real-life framing

**Scenario**: A listing agent is able to use the model to suggest an asking price for a house before it is listed. 
This makes sense for this dataset as the model can help predict an appropriate price based on the sales of other houses. If the listing agent chooses a neighborhood, a living area, a lot area (sq ft), the year it was built, the overall quality from 1-10, basement area, garage area, or the number of bathrooms, the model is able to learn from past sales with these features to then predict a price for a new house from its features. 

| Decision | What I did | Why |
|---|---|---|
| Target | Train on log(SalePrice) | I trained on log(SalePrice) because the price distribution was skewed (1.74 for raw SalePrice and -0.01 after taking the log). Prices have a long tail of expensive houses, and the log made the distribution more symmetric, so a few expensive houses don’t dominate the squared error. Using the log also fits a listing agent’s view that prices behave multiplicatively, as prices tend to rise by percentages. |
| Leakage | Dropped Sale Type, Sale Condition, Mo Sold | The agent prices a house before it is listed, so there is no sale yet. That is why Sale Type and Sale Condition were not used when training the model, as it would be using information that isn’t available at pricing time. These columns carry real price information (Partial sales had a median of $250,000 vs $159,000 for Normal sales), so a model that saw Sale Condition would get an advantage in testing that it couldn’t have in real life. Mo Sold was also removed because an agent cannot know in advance which month the house will sell, although this is a judgment call. I did use Sale Condition once, to remove 3 outlier rows from the training set, before it was dropped as a feature. |
| Yr Sold | Not a feature; used only to split and to compute age | An agent doesn't know the sale year in advance, and 2010 (the test year) is outside the 2006-08 training range, so a linear model would have to extrapolate on it. I only use it to split by year and to compute house age as a stand-in for age at listing. |
| Split | Train 2006-08 (1938 rows), validation 2009 (648), test 2010 (341) | An agent prices future listings from past sales, so a random split would let the model train on sales later than ones it is tested on. I use 2009 to choose alpha so that the 2010 test set stays untouched. |
| Metric | Headline: median absolute % error. Also: share within 10%, RMSE ($) | It reads as "the suggested price is typically off by X%", which fits an agent and matches the log target. The median isn't dominated by a few big misses. RMSE and the within-10% share are for reference. |


## Data preparation & feature engineering

| Decision | What I did | Why |
|---|---|---|
| MS SubClass | Converted to text, one-hot encoded | The values (20, 60, 120...) are dwelling-type codes, not quantities, so a linear model would wrongly treat 60 as three times 20. |
| House Age, Years Since Remodel | Sale year minus build/remodel year, clipped at 0. 1 house had a negative age and 3 a negative remodel gap, all Partial sales. | Age is more directly related to price than a calendar year. Negative ages are meaningless (those houses were sold before they were finished), so I set them to 0. Age uses Yr Sold as a stand-in for age at listing. |
| Total Baths, Total SF, Porch SF | Combined related columns | Related columns describe one thing (bathrooms, living plus basement space, porch space), so a single total is easier for the model to use. I kept the originals, and the Ridge penalty limits the effect of the overlap. |
| Log Lot Area, Log Gr Liv Area | Log of right-skewed sizes | Sizes have a long tail of very large values, and the target is also on a log scale, so logging them makes the relationship closer to proportional. I did not measure the skew or test this against raw sizes. |
| Rating letters (Ex/Gd/TA/Fa/Po/NA) | Mapped to 5/4/3/2/1/0 | The ratings have a natural order that one-hot encoding would throw away. Mapping NA to 0 assumes that having no basement or garage is worse than Poor, which is an assumption. |
| Near-empty columns | Dropped Pool QC, Misc Feature, Alley, Fence | Almost every house is "NA" in them (for example, Pool QC in 2,917 of 2,930 rows), so they add very little information and many near-constant columns. |
| Redundant columns | Dropped 1st Flr SF, 2nd Flr SF, Low Qual Fin SF, Garage Cars, Garage Yr Blt | Gr Liv Area is exactly the sum of the three floor columns, Garage Cars and Garage Area measure nearly the same thing, and Garage Yr Blt overlaps Year Built and is empty for houses with no garage. |
| Lot Frontage | Median imputation plus a "was missing" flag, fit on training data only | 17% of values are empty, so I filled them with the training median and added a 0/1 flag so the model can see which houses were blank. Fitting on training data only avoids leaking test information. |
| Outliers | Removed 3 training rows: houses over 4,000 sq ft that were Partial sales, at about $28-40 per sq ft vs about $107 typical. Kept the 2 large houses that sold for $745k and $755k. | The 3 houses were unfinished at sale and sold far below what their size implies, so they would distort a linear fit. The other 2 follow a plausible price per sq ft, and removing them would leave little data on very large expensive houses. I looked at the data before choosing this rule. |
| Scaling / encoding | StandardScaler and one-hot encoder fit on training rows only | Fitting on training rows only keeps the validation and test sets from influencing the means and categories the model is built on, so the test score isn't overly optimistic. |
| Overall Qual and other ratings | Kept in the model | Overall Qual is the strongest predictor (correlation 0.80). It is an assessor's rating, so I'm assuming an agent can judge quality themselves when walking through the house. I did not test the model without the ratings, so I can't say how much accuracy depends on this assumption. |

I did not test the model without the ratings, or with/without the 3-row removal.

## Modeling: three implementations, one model

- **Model**: linear regression with an L2 penalty (Ridge), trained on log price. The target is a continuous price, so linear regression is the right kind of model (logistic regression is for binary targets). I added the L2 penalty because one-hot encoding leaves about 275 features for 1,938 training rows, and the penalty limits overfitting. 
- **Same objective in all three**: scikit-learn `Ridge`, a manual PyTorch loop (raw tensors, autograd, manual gradient step), and `nn.Module` + `torch.optim.SGD`. Regularization strength alpha chosen on the validation set from {0.1, 1, 10, 100, 1000}: alpha = 1.
- **Naive baseline**: predict the training median price.

Median absolute % error (lower is better):

| Method | Val | Test | Test within 10% | Test RMSE ($) |
|---|---|---|---|---|
| Baseline | 24.55 | 24.79 | 20.8% | 75,315 |
| scikit-learn | 5.45 | 6.51 | 72.1% | 20,265 |
| Manual PyTorch | 5.51 | 6.43 | 70.1% | 19,916 |
| nn.Module | 5.51 | 6.43 | 70.1% | 19,916 |

Convergence check:

| | First run (lr 0.01, 5,000 epochs) | Final run (lr 0.05, 20,000 epochs) |
|---|---|---|
| Manual training MSE (log); sklearn = 0.00754 | 0.0189 | 0.00798 |
| Test houses where PyTorch and sklearn differ by >0.2 (log) | 18 | 0 |
| Median gap (log) | 0.053 | 0.0095 |
| PyTorch test median % error | 7.87 | 6.43 |

- **Do the three agree?** In my first run they did not: PyTorch had 7.87% test error vs 6.51% for scikit-learn, 18 test houses differed by more than 0.2 in log price, and the manual loop's training error was about 2.5 times sklearn's. After training longer with a higher learning rate (20,000 epochs, lr 0.05) the gap mostly closed: 6.43% vs 6.51% test error, 0 houses above 0.2, and a training MSE of 0.00798 vs 0.00754. This is consistent with gradient descent not having fully converged at first. The two PyTorch versions match each other, and PyTorch being slightly better than sklearn on test is within what I'd expect from a 341-house test set, so I don't read anything into it.
- **Tested and ruled out**: the gap was not caused by rare neighborhoods (IDOTRR has 71 training rows, Edwards 129).
- **Not tested**: whether the learning rate or the number of epochs closed the gap, since I changed both at once.
- The two PyTorch versions match each other; the manual loop's printed loss is higher because it includes the L2 penalty and `nn.MSELoss` does not.

## Limitations & next steps

| Limitation | How I'd address it |
|---|---|
| The model tends to under-price the most expensive houses (visible in the predicted-vs-actual plot) | I would check errors by price band, and try adding interaction features or a non-linear model. I have not tested either. |
| New builds (Partial) are about 11% of training rows but about 4% of validation and test | I would report errors separately for Partial and Normal sales, and add a new-build flag or use more recent data. |
| The test set is small (341 houses, January-July 2010 only) | I would use more years of data and evaluate over several time-ordered splits instead of a single test year. |
| The price form fills every feature the agent doesn't enter with a training median or most common value; it assumes no remodel | I would ask for more inputs (kitchen and exterior quality, remodel year) and show a price range instead of a single number. Changing only Overall Qual leaves the other ratings at typical values. |
| PyTorch models are close to but not fully at sklearn's solution (training MSE 0.00798 vs 0.00754) | I would train longer or use a learning-rate schedule and stop when the loss stops changing. Since I changed learning rate and epochs together, I can't say which one mattered. |
| House age uses Yr Sold, which an agent doesn't know in advance | At prediction time I would use the current year as the listing date. In training, the sale year is a close proxy, but I did not measure the difference. |

## Generative AI use disclosure

I used Claude (Anthropic) throughout this assignment. It suggested candidate datasets and business scenarios, proposed options for the split cutoffs, metric, feature engineering, imputation and outlier handling, wrote the code for `train.py`, `app.py` and the exploration and diagnostic scripts, helped with git and Codespaces, and helped draft and word several cells of the tables in this report. I chose among the options (dataset, scenario, columns dropped, engineered features, imputation, outlier rule and split), ran the code, checked the outputs, and reviewed and edited the text.


I used Claude (Anthropic) throughout this assignment. What it did, and what I decided:

**Setup and data**
- Helped me find the dataset source related to housing and real estate as I wanted, save it as a CSV, commit it, and set up my Codespace and git workflow.
- Suggested options for the split cutoffs and the metric. I chose the split and the metric from those options. 

**Leakage and preprocessing**
- I decided which columns an agent would not know at pricing time. Claude helped me check I wasn't missing any leaky columns.
- For Lot Frontage, I first thought of filling with the neighborhood median. Claude suggested median imputation plus a "was missing" flag instead, because it needs no custom code and is fit on training data only, so it can't leak. I chose to use it.

**Feature engineering**
- Claude pointed out that MS SubClass was being treated as a number when it is a category code, and I converted it to text.
- Claude drafted the list of candidate features. I chose which to keep. It also explained the clip on age, and I checked the data: only 1 house had a negative age and 3 a negative remodel gap, so the clip barely affects the model.
- I decided the rule for removing the 3 outlier rows after looking at the data.

**Training**
- Claude helped me write and edit most of the code for the three model implementations. When the manual PyTorch loss differed from the scikit-learn result, it helped me diagnose the gap. I ran the diagnostic script, and training longer (original defaults: `lr=0.01, epochs=5000`; final: `lr=0.05, epochs=20000`) closed most of it. So Claude gave me the bigger numbers to see if that would help. 

**Dashboard**
- Claude wrote most of the code for the Gradio app (`app.py`), and I edited parts of it. I ran it, tested each tab and the price form, and checked that the outputs made sense. This part was the hardest for me to code so Claude helped me more than with the other parts. 

**Report**
- Claude helped me make sure this report made sense after I wrote it, and helped me organize it in tables and in a more readable manner. 


