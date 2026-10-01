# Assignment 1 Report

- **Name**: Gonzalo Ruiz Feliu
- **Student ID**: 19100
- **Email**: gruiz.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

I used the **Medical Cost Personal Dataset** (`data/insurance.csv`). It comes
from the public GitHub repository of Brett Lantz's book *Machine Learning
with R* (`stedy/Machine-Learning-with-R-datasets`), and the same file is also
on Kaggle. The data is simulated from US demographic statistics, so these
are not real patients.

Each row is one insurance member. There are 1,338 rows and 7 columns: `age`,
`sex`, `bmi`, `children`, `smoker`, `region`, and the target, `charges`, which
is the annual medical cost billed to the insurer in dollars.

I picked it for three reasons. The target is a clear continuous number with
an obvious business use. The dataset is small and clean, so it trains in
seconds and fits easily in the repo. And it has an interesting pattern
hidden in it (smoking and obesity together) that a linear model can only
pick up if you engineer a feature for it.

## Business / real-life framing

**The scenario.** A health insurer wants to estimate how much each member
will cost in medical claims over the next year. When someone enrolls, the
insurer already has the details from their form: age, BMI, number of
children, whether they smoke, and where they live. The model turns that
into an **expected annual cost in dollars**, which the insurer can use for
two things:

1. **Budgeting and reserves.** If you add up the expected cost of every
   member, finance knows roughly how much money to set aside to pay claims.
2. **Prevention programs.** Members with a high expected cost, for example
   smokers with obesity, can be offered a program to quit smoking or lose
   weight. That is usually cheaper than paying for their treatment later.

I deliberately did not frame this as a pricing model. In the US, where the
data comes from, the Affordable Care Act does not allow individual health
premiums to depend on health status such as BMI, so a pricing use case
would not be realistic.

**How this framing shaped the pipeline:**

- **Target.** I used `charges` as it is. It is already exactly what the
  insurer needs to know, so there was no reason to transform it.
- **Split.** There is no date column, and every row is a different person
  seen once, so nothing "from the future" can leak into training. That makes
  a random split valid; a time-based split is not needed. Smokers are only
  about 20% of members but drive most of the cost, so I stratified the split
  by `smoker` to make sure train, validation and test all have the same share
  of smokers.
- **Metric.** My main metric is **MAE in dollars**, because it answers the
  business question directly: on average, how many dollars off is our
  estimate per member? I also report **RMSE**, which punishes big misses
  more. Big misses on very expensive members are what hurt a budget the
  most. In this setting, under-predicting is actually worse than
  over-predicting (the insurer runs short of money), but I kept a standard
  symmetric loss to keep the model simple and list this as a limitation.
- **Excluding `sex`.** Using sex to set insurance premiums or benefits has
  been prohibited in the EU since the *Test-Achats* ruling (CJEU, 2011, in
  force from December 2012), and the ACA does not allow it for pricing
  either. My model is for budgeting and prevention rather than pricing, but
  in practice an estimate like this ends up influencing decisions about
  people, so I dropped `sex` for consistency and fairness. I kept `region`,
  since geographic cost differences are legitimate.

## Data preparation & feature engineering

**Cleaning.** I found one exact duplicate row and removed it. If the same
member is counted twice, that person gets double weight and could even end
up in both train and test. That leaves 1,337 rows. There were no missing
values, so no imputation was needed, and I dropped `sex` for the reasons
above.

**Encoding.** `smoker` became 1/0. `region` was one-hot encoded into three
columns (`region_northwest`, `region_southeast`, `region_southwest`), with
`northeast` as the reference category whose effect goes into the bias.
Keeping all four columns would be redundant because they always add up
to 1.

**The engineered feature: `smoker_obese`.** This is 1 if the member smokes
and has a BMI of 30 or more (the standard obesity threshold), and 0
otherwise. The average charges in the data show why it matters:

|            | BMI < 30 | BMI ≥ 30 |
|------------|---------:|---------:|
| non-smoker |  7,977 $ |  8,856 $ |
| smoker     | 21,363 $ | 41,558 $ |

For non-smokers, obesity barely changes the cost. For smokers, it almost
doubles it. A linear model can only add effects together, so from `smoker`
and `bmi` alone it has no way to learn that the combination is much worse
than the sum of its parts. The new column gives it that information
directly.

**Checking the features.** I compared several feature sets with a
scikit-learn linear regression on the validation set (`train.ipynb`,
section 7.1):

| Feature set | Val MAE | Val R² |
|---|---:|---:|
| A: original columns only | 4,036 $ | 0.736 |
| **B: A + `smoker_obese` (chosen)** | **2,560 $** | **0.841** |
| C: A + `smoker_bmi` (smoker × BMI) | 2,869 $ | 0.837 |
| D: B + `age²` | 2,506 $ | 0.841 |
| E: B + `smoker_bmi` + `age²` | 2,399 $ | 0.850 |

Adding `smoker_obese` on its own cuts the error by 37%, and it works better
than the continuous version, `smoker_bmi`. Adding `age²` and `smoker_bmi` on
top only improves MAE by 50 to 160 $, and that is measured on just 201
validation members, so I went with the simpler set B. The final model uses
8 features: `age`, `bmi`, `children`, `smoker`, `smoker_obese` and the three
region columns.

**Split.** 70 / 15 / 15, which gives 935 / 201 / 201 members. The split is
random and stratified by `smoker` (each set has 20.4 to 20.5% smokers), with
a fixed seed of 42 so it can be reproduced. I only touched the test set
once, for the final results.

**Standardization.** Every feature is rescaled to mean 0 and standard
deviation 1 with a `StandardScaler`. The scaler is fit on the training set
only and then applied to validation, test and any new input in the app, so
nothing from validation or test leaks into training. This matters for
gradient descent: a single learning rate only works well for all the
weights if the features are on a similar scale. All three models use the
same standardized inputs, which also means their weights can be compared
directly. I did not scale the target, so predictions stay in dollars.

## Modeling: three implementations, one model

Since the target is a continuous dollar amount, the model is a **linear
regression** (`prediction = X @ w + b`). All three implementations minimize
the same loss, the **mean squared error (MSE)** on the training set:

1. **scikit-learn** `LinearRegression` solves for the optimal weights
   exactly with linear algebra. There is no learning rate and no epochs.
2. **Manual PyTorch loop.** Raw tensors `w` and `b` start at zero and are
   trained with full-batch gradient descent: forward pass, MSE,
   `loss.backward()`, update inside `torch.no_grad()`, then `.grad.zero_()`.
   1,000 epochs.
3. **Standard PyTorch.** `nn.Linear` + `nn.MSELoss` + `torch.optim.SGD`, with
   the same learning rate and number of epochs. I used plain SGD instead of
   Adam so that both PyTorch versions run exactly the same algorithm. The
   only difference is that `nn.Linear` starts from small random weights
   instead of zeros.

**Learning rate.** This was the only hyperparameter to tune, since plain
linear regression has no regularization. I tried four values with the
manual loop:

| lr | Final train MSE | Val MAE | Val RMSE |
|---|---:|---:|---:|
| 0.001 | 24,541,106 | 2,323 $ | 4,998 $ |
| 0.01 | 20,744,896 | 2,559 $ | 4,541 $ |
| **0.1** | **20,744,886** | 2,560 $ | 4,541 $ |
| 1.0 | diverged (NaN) | – | – |

The learning rate does not change which model is the best one; it only
changes how well gradient descent gets there. So I picked the value with
the lowest training loss, 0.1. With 0.001 the training simply had not
finished after 1,000 epochs, and with 1.0 every step overshot and the loss
blew up. One thing worth pointing out: the unfinished 0.001 model actually
gets a better validation MAE, just by chance (its RMSE is worse). If I had
chosen by validation MAE, I would have picked a model that was not done
training.

**Results on the test set** (201 members, not used for any earlier
decision):

| Model | Test MAE | Test RMSE | Test R² | Val MAE | Val R² |
|---|---:|---:|---:|---:|---:|
| Naive baseline (train mean, 13,298 $) | 9,495 $ | 12,428 $ | 0.000 | 8,557 $ | 0.000 |
| scikit-learn | 2,244 $ | 4,094 $ | 0.891 | 2,560 $ | 0.841 |
| PyTorch manual | 2,244 $ | 4,094 $ | 0.891 | 2,560 $ | 0.841 |
| PyTorch standard | 2,244 $ | 4,094 $ | 0.891 | 2,560 $ | 0.841 |

**Do the three agree? Yes.** Their weights match to two decimals, and the
largest gap between any two test predictions is 0.01 $. That is what you
would expect from the same model, the same loss and the same data.
scikit-learn finds the minimum in one exact step, while the two PyTorch
versions get there step by step and fully converge within 1,000 epochs. The
leftover cent comes from PyTorch using float32 instead of float64.

**What the weights say** (in $ per one standard deviation of each feature):
`smoker_obese` +6,212, `smoker` +5,386, `age` +3,647, `children` +637, `bmi`
only +201, and the regions between −186 and −602 compared with the
northeast. So for a smoker with obesity, both smoking terms add up, while
BMI on its own matters very little. That matches the table in the feature
engineering section.

**What this means for the business.** On average, the estimate is about
2,240 $ off per member (the median error is about 1,480 $), compared with
about 9,500 $ for the baseline. Summed over the whole test set, the
predicted total cost is only 1.2% above the real total, and for budgeting
that total is what matters most.

## Limitations & next steps

1. **Some expensive non-smokers are unexplained.** All 9 test errors above
   10,000 $ are non-smokers who cost much more than predicted (about 6% of
   non-smokers cost over 20,000 $). Nothing in the data explains them; they
   are probably members with an illness or accident that is not recorded.
   *Next step*: add information such as previous-year claims or chronic
   conditions.
2. **The loss treats both kinds of error the same, but under-predicting is
   worse.** The model under-predicts only 19% of members, but those misses
   are much bigger (5,525 $ on average, against 1,479 $ for
   over-predictions). *Next step*: for reserves, add a safety margin based
   on the distribution of validation errors, or train a model that predicts
   a high percentile instead of the average (for example, quantile
   regression).
3. **Small dataset, noisy evaluation.** With 201 members per set, the
   results depend on who ends up in each one: R² is 0.84 on validation but
   0.89 on test. *Next step*: use k-fold cross-validation to report an
   average and a spread instead of a single number, and to check again
   whether `age²` and `smoker_bmi` (sets D and E) are worth adding.
4. **The features are simplified.** `smoker_obese` uses a hard cut at BMI 30,
   so a smoker with BMI 29.9 and one with 30.1 are treated very differently.
   Age also enters linearly, even though cost seems to grow faster at older
   ages. *Next step*: evaluate the alternatives from set E with
   cross-validation.
5. **Simulated US data.** The data is synthetic and the regions are US
   regions, so a European insurer could not use this model as it is. *Next
   step*: retrain on real claims data from the target market.

## Generative AI use disclosure

I used **Claude (Anthropic), through Claude Code**, as an assistant
throughout this assignment.

- **Decisions (mine)**: I chose the by searching on the internet and asking Gemini too,
  , chose the final feature set after comparing the options,
  decided to keep the approach as simpleand neat as possible to better understand
  everything.
- **Code**: Claude helped me with the code, specially in the more technical parts.
  I went through it step by step with Claude's explanations until I understood
  each part.
- **Report**: I drafted a first version of this report, including the
  analysis and the numbers. Claude then helped me paraphrase and improve the grammar.
