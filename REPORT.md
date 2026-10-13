# Assignment 1 Report

- **Name**: Julian Consuegra
- **Student ID**: 19376
- **Email**: jconsuegra.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

The dataset was retrieved from the UCI Machine Learning Repository and is named "Online Shoppers Purchasing Intention". 

The rows from this dataset represent one browsing session from a distinct user on an ecommerce site. The dataset takes into account multiple factors of user browsing such as page activity (# of pages viewed), analytics metrics (bounce rate, exit rate), timing (month, weekend, special date) and visitor & technical details (new/returning visitor) which are represented by the columns. 

Size: 12,330 rows (unique users) and 18 columns: 17 inputs plus the target, Revenue.

Target: I will be utilizing the column "Revenue" as a target. The value is "TRUE" if session ended in a purchase and "FALSE" if not. Only about 15% of sessions end up in a purchase.

I picked this  dataset since it brings interesting insights on user behavior and relevant conclusions can be drawn regarding what leads a person to actually purchase products on an e-commerce site. On a personal note, I truly believe that successful strategy campaigns backed by relevant data can turn a business around, and this seemed like an interesting opportunity to pursue this curiosity and measure impact using detailed consumer information. Also, the specificity and amount of factors considered in the dataset seemed valuable since analysis can be performed on a deeper scale producing higher value solutions. 

## Business / real-life framing

This model aims to predict whether a session ends in a purchase through different factors and how this influences their purchasing decision. The business is an online shop/e-commerce and it wants to offer discounts to customers who are likely to leave without buying, in order to incentivize purchase and increase conversion rates. Based on patterns gathered in training data, this model will be able to predict purchasing likelihood for new user sessions in order to apply targeted discounts towards those who are predicted not to buy.

Based on this business scenario, I decided to use "Revenue" as the target since the ultimate goal of the shop is to know if a customer will buy or not. If there are repetitive patterns that affect behavior and lead towards the value under "Revenue" being either "TRUE" or "FALSE", there is a clear strategic opportunity that needs to be addressed. The column/target will be kept and used as is, there is no need to make any modification. 

A time-based split is not possible since the only time variable taken into account is "Month". There is no indication of actual year or day making the true time order unclear. Also, in the dataset every row is a different user so the same user cannot appear in more than one group, which avoids data leakage between training and test. Due to this, it is best to use a random split in which all sessions will be randomly placed in three groups (training 60%, validation 20%, test 20%). With this split, I am able to learn the weights from the training set, choose the threshold on the validation set and finally use the test set to determine final metrics. The split is stratified, so each group keeps the same share of buyers (15.5% of buyers in each group).

With the properties of my dataset, accuracy wouldn't be an insightful metric since approximately 85% of the users do not purchase and a simple model that always predicts that a user won't buy would already give an unrepresentative accuracy of about 85%. 

In this scenario, the threshold will determine who gets a discount and who does not. The model will give an estimate of the probability a customer has of purchasing something in the session. If the probability is less than the threshold it is assumed that there will be no purchase so a discount is triggered and vice-versa. 

However, this strategy can generate two main mistakes: missed sale or wasted discount. Evidently, discounts will not convince every non-buyer to purchase a product which is why to make it more realistic, I am assuming that discounts will convince about 10% of non-buyers to actually purchase at the end of the session. If the model falsely determines that a user will purchase, no discount is shown and the sales revenue from potential converted clients is lost (missed sale). On the other hand, if the model falsely determines that a user will not purchase, the discount will be offered and only the discount amount is lost since the purchase was going to be made either way (wasted discount). For example, assume the shop sells a product at $50, offers a 20% discount to targeted customers and the discount convinces 10% of visitors who were going to leave. A missed sale would cost $5 ($50 x 10%) in lost revenue, while a wasted discount would cost $10 ($50 x 20%). Without this assumption a missed sale would cost the full $50, and showing the discount to every visitor would always be the cheapest option, leaving the model with nothing to decide.

The price, discount size and 10%  are assumed values since the dataset does not contain prices or discount results. The number of each mistake is counted on sessions not used for training, by comparing the model's prediction with the actual Revenue value. A missed sale is a session where the model predicted a purchase but Revenue is FALSE, and a wasted discount is a session where the model predicted no purchase but Revenue is TRUE. For each candidate threshold, the total cost is calculated as:

Total cost = (missed sales × $5) + (wasted discounts × $10). 

With this in mind, the metric that should be used to determine threshold is "Total Cost" as we will choose a threshold that minimizes the total cost of the two mistakes.


## Data preparation & feature engineering

**Data Quality**: The dataset had no missing values requiring no cleaning on this regard. Throughout the whole dataset approximately 1% of rows (125 rows) were exact duplicates which I decided to keep. Almost all of them (94%) are one-page sessions where the visitor left immediately, and none ended in a purchase. Different visitors can easily produce identical sessions like these, so they are most likely genuine and I kept them.   

**Columns Removed**: 
- "PageValues" was dropped because Google Analytics calculates it from completed
  purchases, so it partly contains the outcome the model predicts (data leakage). 
  It was by far the strongest predictor: 56% of sessions with a value above 0 bought,
  against 4% at 0. The model's score will decrease. Dropping this prevents the model from having information that will not be available when making the discount decision.
  
- "Browser", "OperatingSystems" and "Region" were dropped because the purchase rate
  barely changes across their values and there is no business reason for them to
  affect buying. Keeping them would increase noise and model complexity with no benefit at all as a return. 

**Categorical columns.**
- "Month" and "VisitorType" contain text, so they were one-hot encoded: each value
  becomes its own 0/1 column. One value per column is left out ("drop_first"),
  because it can be deduced from the others and would repeat the same information.
  Months were not numbered 1 to 12 because it can't be assumed that there is a positive trend in purchases as the number of the month increases.  

- "TrafficType" was kept because the purchase rate varies from 6% to 28% across traffic
  sources. This column refers to how the visitor arrived at the website (search engine result, ad, newsletter). Sources with fewer than 100 sessions were merged into one "other" group,
  since a handful of sessions gives the model nothing reliable to learn from. I first
  used a cut-off of 200, then lowered it to 100 so that source 20 (198 sessions, 25%
  purchase rate) keeps its own column. After grouping, these were also one-hot encoded (TrafficType_1, TrafficType_2, etc).

**Transformations and new features.**
- The six page-count and time columns were log-transformed ("log1p"). They contain a
  few extreme sessions, up to about 18 hours on product pages, which would otherwise
  pull the weights of a linear model. The log keeps the order of the values but
  shrinks large ones, so the difference between 10 seconds and 2 minutes counts more
  than the difference between 5 and 6 hours. This improved the model slightly.
- A new "NoBounce" flag marks sessions with a bounce rate of exactly 0. They are almost
  half of all sessions and buy more often (18.8% against 12.8%). A linear model cannot
  express a jump at zero on its own, so the flag lets it treat these sessions as a
  group. "BounceRates" was kept as well, since it still separates sessions above zero.
  The gain was small.
- "Weekend" and "Revenue" were converted from True/False to 1/0.

**What did not work.**
- Tested the average time per product page which made business sense but added nothing. After the log transform, the model can already compute it from the two columns it comes from. In general, the transformations added very little improvement and removing "PageValues" limited predictions substantially.

**Scaling**
- All columns were standardized so that all columns are on the same scale, which lets gradient descent train properly and makes the weights comparable (mean 0, spread 1). Values learned from the training set were used only and then applied to validation and test, so no information from those sets leaks into the preparation. 


## Modeling: three implementations, one model

**Model.** Logistic regression, because the target is binary (bought or not).

**Implementations.** All three use the same split and standardized features.
- scikit-learn: "LogisticRegression" with regularization off (`C=np.inf`), so all
  three minimize exactly the same log loss.
- Manual PyTorch: weights start at zero; 5,000 steps of gradient descent (learning
  rate 0.5) using autograd for the gradients. The loss flattened after about 2,000
  steps, so it converged.
- nn.Module + torch.optim: the same loop with "nn.Linear", "BCEWithLogitsLoss" and
  "optim.SGD", same learning rate and steps.

**Threshold.** Every threshold from 0.01 to 0.99 was tried on the validation set, and 0.31 gave the lowest total cost. 

**Test set results** (2,466 sessions):

| Model | Log loss | AUC | Total cost |
|---|---|---|---|
| Naive baseline (discount everyone) | 0.4311 | 0.5000 | $3,820 |
| scikit-learn | 0.3674 | 0.7648 | $3,465 |
| Manual PyTorch | 0.3673 | 0.7648 | $3,465 |
| nn.Module + torch.optim | 0.3673 | 0.7648 | $3,465 |

**Agreement.** The three match: same AUC and cost, log loss within 0.0001, and
predicted probabilities within 0.0043 for any session. The tiny gaps come from
stopping rules: scikit-learn stops when improvement becomes negligible, the PyTorch
versions run a fixed 5,000 steps.

**Against the baseline.** The model saves $355 (9.3%) versus discounting everyone
(7.5% on validation). An AUC of 0.76 is moderate, the price of dropping "PageValues".


## Limitations & next steps
- **Timing of the data.** The features are totals for the whole session, but the
  discount would be shown during the session, before those totals are known. The
  results are therefore somewhat optimistic. *Fix:* rebuild the features from what is
  known at the moment of the decision, for example after the first five pages.
- **No real dates.** Only the month is available, so the split had to be random and
  the model was never tested on later months than it learned from. Buying varies
  strongly by month (2% in February, 25% in November). *Fix:* with timestamps, train
  on earlier months and test on the most recent ones.
- **Assumed costs.** The price ($50), discount (20%) and the share of leavers the
  discount convinces (10%) are assumptions, and the dataset contains no discounts, so
  their real effect cannot be measured. The best threshold and the saving depend on
  them. *Fix:* Run an A/B test showing the discount to a random group of visitors to
  measure its real effect, then update the costs. The dashboard already lets these
  values be changed.
- **`PageValues` dropped.** Removing the leaking column limits the model to an AUC of
  0.76. *Fix:* recompute page values only from purchases made before each session,
  which keeps most of the signal without leaking the outcome.
- **Unnamed traffic sources.** Traffic sources are numeric codes with no published
  meaning, so the model uses them but they cannot guide marketing decisions.
  *Fix:* Obtain the channel names from the shop.
- **One shop, one year.** The data covers a single site and 10 months (no January or
  April), so the results may not transfer to other shops or seasons. *Fix:* Retrain
  on more recent and broader data before using the model in practice.

## Generative AI use disclosure

I used Claude to assist me throughout the assignment and I will specify how in each section:

**Ideation**
Once the assignment was understood, Claude helped me explore possible datasets with their respective business framings that I could work with. Once I chose the option of greater relevance and interest to me, Claude pointed out key considerations that needed to be addressed before starting. For example, pointing out that with my initial assumptions, every customer presented with a discount would purchase so there would be no need in creating a model. Like this, other small details that helped me re frame the problem and polish the solution. 

**Pre-Processing & Engineering Details**
Key concepts of pre processing and engineering were re-clarified by Claude (OHE, leakage, log transformation and stratified split among others) before applying changes. I ran exploratory checks that helped me decide what to modify in the data based on Claude's recommendations and on my own criteria after understanding the bases of this step.

**Coding**
Claude developed the code for "train.py" step by step while I ran and checked each step based on the established business framing. After this, the "app.py" was written by Claude in order to properly develop the Dashboard. I directed its redesign so that it had a professional look and every chart reflected a finding with business relevance.

**Report**
Wording and review of the different sections of the report. Helped with the framing of technical concepts and double checking the overall flow of the report was clear. 
