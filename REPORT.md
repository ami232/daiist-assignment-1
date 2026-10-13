# Assignment 1 Report

- **Name**: Gonzalo Domínguez
- **Student ID**: 18691
- **Email**: gdominguez.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

This is the Online Shoppers Purchasing Intention Dataset (Sakar and Kastro, 2018), available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)). The local file used is `data/shopping.csv`. 

This dataset contains 12,330 rows and 18 columns (17 explanatory variables and 1 binary target, Revenue). 
- Each row is 1 completed browsing session on a website (entered, browsed, left), including information about website activity, characteristics, calendar, and finally whether a purchase took place or not.

The target variable is Revenue, with 1 indicating that the session ended in a purchase and 0 indicating no purchase took place. 15.5% of sessions were purchasing, and 84.5% were non-purchasing, which makes this an imbalanced, binary classification task.

The dataset has no missing values, while 125 rows were found to be duplicates of other rows. I retained them because identical recorded attributes may not mean a session was recorded twice (the dataset has no session identifier to guarantee the distinction).

This dataset is thus a very natural choice — an e-commerce business decision with necessary preprocessing, features with skewed distributions, potential concerns for the later sections of the report, and sets of behavioral and seasonal features. It's also small enough to commit and retrain comfortably. 

It should be noted that each recorded row represents a *completed* session, even though business decisions for this project are made at session entry. 

------

## Business / real-life framing

Take an e-commerce company. 
They have a pretty limited number of customer support agents, so the company wants to identify, at the beginning of a visit, which visitors appear most likely to make a purchase, thus benefitting from immediate assistance.

The predicted target is **not** the additional sales that offering support may lead to, but just the purchase occurrence.

 
- True Positive: a prioritized visitor ended up purchasing. Bears the base support cost.
- False Positive: a prioritized visitor did not end up purchasing. Bears the base support cost.
- False Negative: a non-prioritized visitor ended up purchasing. Bears the missed-opportunity cost.
- True Negative: a non-prioritized visitor did not end up purchasing. Bears no extra cost.

A False Negative thus isn't a lost sale, and a False Positive does not mean the assistance was worthless because of dataset limitations.

----

### **Operating assumptions**
- Assistance budget: the company can assist up to 20% of sessions at most, a figure that is assessed as a monthly proportion during historical validation.
- Business goal to choose the best policy: maximize monthly purchaser recall (while meeting the assistance constraint just outlined)
- **Cost per contact**: 1 (units)
- FN cost: 5 (units)
    - This 1:5 ratio is not proven and remains hypothetical, symbolizing an assumed preference from the department.
- Derived cost formula: 1 * (TP + FP) + 5 * FN

----

### **Temporal evaluation**
I used a monthly chronological split since the intended app wants to predict outcomes for later sessions making use of earlier ones — a random split could mix them from different months and be too optimistic. 

I separated the observations into February-October (Training, 7605), November (Validation, 2998) and December (Testing, 1727). 

The purchase rate changes substantially between these periods, with November doubling that of our Training and Testing partitions. Since the model is trained on an earlier, lower purchase rate population, the model may underestimate probabilities later.

The historical expanding-window folds predict July-November by using prior observations, while "policy replay" evaluates September with July and August, October with July-September, and November with July-October. 

----

## Data preparation & feature engineering

`Revenue` and `Weekend` were converted to integer 0 and 1s. 
Months were mapped to numbers for splitting and calendar transformations.

The final model uses 12 selected input columns
- StandardScaler standardizes the 6 numerical columns
- OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False) transforms the 6 categorical columns into indicators

----

### **Feature availability**
Because of the dataset's row nature (each row measures a completed session) contrasting with this project's ex-ante nature, it had to be clearly studied which features would be available at the beginning of each visit.

| Input group | Fields supplied to the model |
|---|---|
| Originally numeric | `SpecialDay`, `Weekend` |
| Originally categorical | `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType` |
| Calendar engineering (cyclical) | `MonthSin = sin(2π × MonthNum / 12)`, `MonthCos = cos(2π × MonthNum / 12)` |
| Visitor-calendar interaction | `ReturningWeekend = 1[returning] × Weekend`, `ReturningSpecialDay = 1[returning] × SpecialDay` |
| Visitor-source interaction | `TrafficVisitorSegment = TrafficType + "__" + VisitorType`, treated categorically |

The last 3 groups are hypotheses regarding calendar periodicity and relationships between type of visitor, and session context.

Categorical traffic codes are not continuous quantities nor do their numerical distances have any meaning.

This always assumes that technical information (i. e. browser and OS, traffic source info, visitor region...) can be obtained at session entry, based on prior entries. The dataset itself does not specify whether this would be realistic.

I excluded the page count, duration, bounce rate, exit rate, and page value variables because they are recorded based on browsing activity, so it's information unavailable to the model if this is ex-ante.

MonthNum is not a final model input, only being created for splitting and to construct `MonthSin` and `MonthCos`. 

The following information is thus excluded from the model input list: `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`.
- These fields may still appear in retrospective, descriptive dashboard charts, but the predictor does not make use of them.

----

### **Feature ablation output**

*Note: July–November; reported statistics average the five evaluation months; all candidates were compared using a fixed threshold of 0.25*

| Feature specification | Mean monthly recall | Mean monthly contact share | Mean ROC-AUC | Mean log loss | Mean monthly scenario cost/session |
|---|---:|---:|---:|---:|---:|
| A: Original entry-time fields | 0.1167 | 0.0838 | 0.6108 | 0.5036 | 0.9522 |
| B: A + cyclical calendar | 0.1294 | 0.1009 | **0.6133** | 0.5019 | 0.9508 |
| C: B + visitor interactions | 0.1367 | 0.1064 | 0.6118 | 0.5007 | **0.9462** |
| D: C + traffic/visitor segment | **0.1426** | 0.1141 | 0.6119 | **0.5001** | 0.9483 |
| E: D + session activity (ORACLE, not deployable) | **0.7038** | **0.2443** | **0.8436** | **0.4085** | **0.5366** |

D was chosen for the pipeline, despite not being the "winner" for all evaluation quantities. D is prioritised because it achieved the highest average monthly purchaser recall under our threshold — which is perfectly consistent with my main goal (to identify eventual purchasers). It also increases higher average contact share. The improvement over C is in any case not all that substantial and comes at the cost of a slightly higher "cost", which is derived from hypothetical units. 

All four were evaluated on the same expanding window folds, with the same regularization strength and the fixed 0.25 threshold, so the comparison does not show each specification receiving its own optimized policy — more proof to say either D or C would have been fair choices. 

Meanwhile, E added fields only known once a session ended: log-transformed page counts, durations and PageValues; Bounce and Exit rates, total pages viewed... using the same expanding window folds, the advantage seen is tremendous, winning in all departments. This is a key experiment in that it shows the predictive signal of the dataset resides in what visitors do DURING the session — partly because many attributes it uses, such as PageValues, represent behavior close to the purchase itself. This analysis was carried out to justify: 1. why the predictive power struggles to go up under our current ex-ante model, 2. if a mid-session trigger existed, this would have an extremely strong use case, much stronger than right now. 

----

### **Data limitations present during running**
- Validation and Test transformations produce `OneHotEncoder` warnings for categories not found in earlier training folds. 
- The encoded columns include pretty specific traffic-visitor combinations, which may have limited historical basis.

- **Unseen categories:** Some historical validation folds contain OS, browser, and other categories that were not observed in their corresponding training folds — which led to `OneHotEncoder` warnings. The warning is normal, if I've informed myself correctly, because `drop="first"` is also used, so this encoding can be confused with the dropped reference category within that group.
- **Sparse interaction categories:** the combination of `TrafficType` and `VisitorType` creates some very specific categorical mixes/combinations that have little historical support.


## Modeling: three implementations, one model

Since the target is binary, I went for a logistic regression, using L2 regularization, the output of which is a predicted probability for "session purchase". All 3 implementations use the same Feb-Oct rows and 92 column matrix

| Aspect | Scikit-learn | Manual PyTorch | `nn.Module` + `torch.optim` |
|---|---|---|---|
| Model | `LogisticRegression` | Logits `X @ weights + bias` | Single `nn.Linear(n_features, 1)` layer |
| Binary loss | Logistic loss | `binary_cross_entropy_with_logits` | `BCEWithLogitsLoss` |
| Optimization | `lbfgs` | Explicit parameter update after autograd | `torch.optim.SGD` |
| L2 penalty | `C = 0.1` | `sum(weights²)/(2 × n × C)` | Same weight penalty |
| Intercept/bias penalty | Not penalized | Not penalized | Not penalized |
| Learning rate | Solver-managed | 1.0 | 1.0 |
| Iterations / epochs | Converged in 84 iterations in the logged run | 5,000 epochs | 5,000 epochs |
| Floating-point representation | Model input converted to float64 | float64 tensors | float64 model/tensors |

The PyTorch models start with 0 weights and bias, while the manual version computes gradients via loss.backward(), applying parameter updates, then clearing gradients. The standard uses optimizer.zero_grad(), loss.backward() and optimizer.step()

The C candidates were .1 and 1, which were selected using mean historical log-loss. .1 was chosen for the final run (July–August selection scores: 0.457 vs 0.471).

----

### **Model agreement**

| Diagnostic | Scikit-learn | Manual PyTorch | Standard PyTorch |
|---|---:|---:|---:|
| February–October training log loss | 0.344358 | 0.344359 | 0.344359 |
| November validation log loss | 0.582616 | 0.582616 | 0.582616 |

Maximum absolute **December probability** difference: scikit-learn vs manual = **0.00013171**; scikit-learn vs standard PyTorch = **0.00013171**; manual vs standard PyTorch = **0.0**.

The minimum difference is most likely because the full-batch gradient didn't fully converge — it can be seen that, the more epochs, the lower the difference (1000 epochs => 2.32e-02, 20000 epochs => 3.42 e-07)


----

### **December holdout comparison (0.50 classification threshold, "Standard")**

| Model | Accuracy | Precision | Recall | ROC-AUC | Log loss | Scenario cost |
|---|---:|---:|---:|---:|---:|---:|
| Naive constant training rate baseline | 0.8749 | 0 | 0 | 0.5000 | 0.3769 | 1,080 |
| Scikit-learn | 0.8749 | 0 | 0 | 0.6391 | 0.3663 | 1,080 |
| Manual PyTorch | 0.8749 | 0 | 0 | 0.6391 | 0.3663 | 1,080 |
| Standard PyTorch | 0.8749 | 0 | 0 | 0.6391 | 0.3663 | 1,080 |

The learned models have higher ROC-AUC and lower log loss than the constant baselines.

At the default threshold (0.50), all three learned models predict negative class for every December session. Their recall and precision are therefore 0, and their accuracy is identical to the naive's.

The learned models have a ROC-AUC of 0.6391, higher than the constant-probability's 0.5000, while also achieving lower log loss. Thus, the learned probabilities should contain useful information.

----

### **December holdout, frozen 0.25 threshold (Policy A):**

| Model | Contacts | TP | FP | FN | TN | Precision | Recall | Scenario cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scikit-learn | 136 | 31 | 105 | 185 | 1,406 | 0.2279 | 0.1435 | 1,061 |
| Manual PyTorch | 136 | 31 | 105 | 185 | 1,406 | 0.2279 | 0.1435 | 1,061 |
| Standard PyTorch | 136 | 31 | 105 | 185 | 1,406 | 0.2279 | 0.1435 | 1,061 |
| Always contact nobody (reference) | 0 | 0 | 0 | 216 | 1,511 | 0 | 0 | 1,080 |

This policy contacts 7.87% of December visitors. Purchasers represent 22.79% of these contacts, while December's purchasing rate is at 12.51%. 

This means that, under the hypothetical cost scenario outlined earlier, we are saving 19 (units) (1.76%) in comparison to contacting nobody. This is nevertheless an offline evaluation because of a lack of real staffing costs information.

----

### **Historical policy**

I ran 3 fixed-cutoff strategies, replayed on Sep-Nov, using calibration info for each evaluation month.

So for each evaluation month, the threshold was based only with earlier historical folds. The threshold was then applied to the predictions of the evaluation month without using that month's outcomes.

| Historical replay (Sep–Nov) | Mean monthly recall | Mean monthly contact share | Highest monthly contact share | Within 20% in every replay month? |
|---|---:|---:|---:|---|
| A: Conservative historical capacity cutoff | 0.2951 | 0.2288 | 0.3772 | No |
| B: Historical 80th-percentile cutoff | 0.4815 | 0.3839 | 0.5268 | No |
| C: Historical quantile plus cost-based floor | 0.4388 | 0.3316 | 0.3772 | No |

The 3 of them broke the 20% contact-share limit we stated, and so the fallback kept Policy A. This is in no way evidence that A respects capacity.

With no fixed threshold, each Policy re-estimated said cutoff, with Policy A using 0.2 in September which explains divergences.

The replay stats are unweighted monthly averages, explaining the difference in monthly contact share (22.88% v. 16.5% for Policy A)

Policy A's cutoff is 0.25, using historical folds for Jul-Nov.


## Limitations & next steps

- **Temporal change and probability reliability** (average November purchase probability of 16.36% vs. actual positive rate of 25.35%). The difference between the training November and December purchase rates indicates purchasing behavior is likely to change a lot over time. With more historical data, it would be useful to examine probability calibration separately by period, and trying to see if more recent training data improves the reliability of these probabilities.
- **The 20% hypothetical staffing cap is never fully in force**, with all candidates exceeding it at some point. Perhaps an adaptive cutoff of sorts could be implemented — if timestamps existed, these could use actual visitor arrival times and the corresponding staff availability.
- **Very limited temporal information**, since there is nothing deeper than month (week, day, timestamp...). To make matters worse, January and April are absent. Although the cyclical calendar features I incorporated try to represent month to month continuity, it's hard to generalise. The dataset definitely struggles because of that, so further data collection would be very helpful.
- **The cost scenarios are fully hypothetical** and may have led to a biased conclusion. If I could obtain the cost of customer assistance, the specific outcome of a proper assistance, or information on customer-value, that would enhance the choice of economic penalty.
- **No causal effect of assistance**, since the dataset does not record if assistance changed a browser's decision. Perhaps running an experiment that could determine if proactive assistance actually changes purchase likelihood or customer satisfaction would amplify the scope of the project, and ultimately help the business.


## Generative AI use disclosure

AI was used for 5 different purposes:
1. **Code refinement**: more complex and detailed parts of the code were produced by AI, starting off of a baseline created by me.
2. **App development**: the Gradio app is 99% created by AI, as authorised in-class by the instructor — I pulled off a very basic iteration of it by hand but I would have never been able to do it without it.
3. **Completionism**: While evaluating potential improvements, AI came up with a couple good ideas that it helped explain and then implement. This includes the historical policy experiment, and the rolling month validation.
4. **Report formatting cleanup**: for a better understanding of markdown format and also better visibility, AI was used to identify areas where the format's legibility could be improved.
5. **Proper functionality**: identification and understanding of errors in-code and during running, learning how to properly commit, etc.
