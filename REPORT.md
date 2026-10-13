# Assignment 1 Report

- **Name**: Francisco
- **Student ID**: 15701
- **Email**: froca.ieu2021@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

**Ames Housing** (De Cock, 2011, *Journal of Statistics Education* 19(3)),
downloaded from the original source
(`https://jse.amstat.org/v19n3/decock/AmesHousing.txt`) and committed
unchanged as `data/ames.csv`.

- One row = one residential property sale in Ames, Iowa, between January 2006
  and July 2010.
- 2,930 rows × 82 columns: the sale price, sale date and sale conditions,
  plus ~80 descriptors of the property (size, quality ratings, age,
  neighborhood, basement/garage/porch details).
- Why this dataset: it matches my interest in real-estate valuation, the
  feature set is rich and messy enough that feature engineering matters
  (quality ratings stored as text, "NA" meaning "feature absent", many
  partly redundant size columns), and every row has a sale date, which
  lets me test the model the way it would actually be used: on future sales.

## Business / real-life framing

**Scenario.** A real-estate appraiser in Ames wants a quick, explainable
first estimate of a home's market value from its characteristics, to use
as a starting point before the full manual appraisal. The model supports
one decision: *what price should this property be valued at today?*

**How the framing drives the pipeline:**

| Decision | Choice | Why it follows from the framing |
|---|---|---|
| Population | Keep only `Sale Condition = Normal` sales in residential zoning (2,397 of 2,930 sales) | An appraisal estimates the price in an ordinary arm's-length sale. Foreclosures/short sales (`Abnorml`), sales between relatives (`Family`), land adjustments and allocations are priced by different forces, and partial sales of unfinished new homes are priced before the house exists. Training on them would teach the model prices the appraiser should *not* reproduce. |
| Target | `log(SalePrice)` | Appraisal errors matter in relative terms: $20k off is serious on a $100k house and minor on a $500k one. On a log scale, a linear model's errors and coefficients are percentages, and the right-skewed price distribution becomes roughly symmetric. |
| Split | Time-based: train = sales 2006–2008 (1,538), validation = 2009 (558), test = 2010 (301) | The appraiser always values houses using only past sales. A random split would let the model see 2010 prices while "predicting" 2009 ones, i.e. leak the future and overstate accuracy. The period also covers the 2008 financial crisis, so the test shows whether the model holds up after a market shock. |
| Use of validation set | Pick the regularization strength and PyTorch learning rate on 2009 only; touch 2010 once, for the final numbers | Keeps the test year a genuine "unseen future". |
| Metric | MAE in dollars on the test year, after converting predictions back with `exp()` | An appraiser and their client think in dollars: "on average the estimate is off by $X". MAE is also less dominated by a few extreme houses than RMSE. |
| Baseline | Predict the median training-period price for every house | The simplest thing an appraiser could do without a model. Any useful model must clearly beat it. |

Note on the target/metric mismatch: the model is fit by minimizing squared
error on `log(SalePrice)`, but evaluated by MAE in dollars. Training on the
log scale gives every house the same relative weight; reporting in dollars
keeps the result interpretable for the business. `exp()` of a log
prediction estimates the *median* price, not the mean, so a small
downward bias in dollars is expected.

## Data preparation & feature engineering

Code: `features.py` (all transformations), `train.py` (pipeline).

**Data-quality decisions**

- *Two kinds of "null".* In the raw file the text `NA` means "the house does
  not have this" (no alley 2,732, no pool 2,917, no fireplace 1,422, no
  basement 79, no garage ~157), while an empty cell is a genuinely missing
  value. The CSV is read with `keep_default_na=False, na_values=[""]` so the
  two stay apart; `NA` is information, not missing data.
- *Lot Frontage* (490 genuinely missing, 16%): missingness is not random
  (51% missing on cul-de-sacs vs 14% on inside lots). Imputed with the median
  of the house's neighborhood, learned on the training years only, plus a
  `frontage_missing` flag.
- *Garage Yr Blt* dropped: redundant with `Year Built`, and its missing values
  are houses without a garage.
- Remaining gaps (1–23 rows per column) imputed with the training median
  (numeric) or most frequent value (categorical).

**Leakage control.** Only row-wise transformations (each value depends on its
own row) are applied before the split. Everything that learns a statistic —
imputation medians, neighborhood frontage medians, scaler means/stds, the
category list of the one-hot encoder — is fitted on 2006–08 only and then
applied unchanged to 2009 and 2010.

**Engineered features** (67 model inputs after encoding)

| Feature | What | Why |
|---|---|---|
| Quality ratings → 0–5 | `Ex=5 … Po=1, NA=0` for exterior, kitchen, basement, heating, fireplace, garage | Keeps the natural order in one column instead of 5 unordered dummies; "absent" ranks below "poor" |
| Presence flags | `has_garage`, `has_basement`, `has_fireplace`, `has_2nd_floor` | Separates "doesn't have it" from "has a small one" |
| `TotalSF` (log) | above-ground + basement area | Buyers pay for total usable space |
| `BsmtFinType1SF` | finished (type 1) basement area | Added after the feature analysis below: `TotalSF` treats a finished and a bare basement the same |
| `TotalBaths` | full + 0.5 × half, incl. basement | One measure instead of four correlated counts |
| `HouseAge`, `YearsSinceRemod` | measured at the sale year | What affects value is age at the moment of sale |
| `TotalPorchSF` (log1p) | sum of porch/deck areas | Five sparse, skewed columns into one |
| Logs of sizes | `log_TotalSF`, `log_GrLivArea`, `log_LotArea`, `log1p` porch/veneer | With a log target the coefficient is an elasticity; pulls in long tails (Lot Area skew ≈ 11) |
| `Qual_x_logSF` | Overall Qual × log(TotalSF) | The quality premium grows with size; a linear model cannot learn a product unless given it |
| One-hot | Neighborhood, Bldg Type, MS Zoning, Central Air, Foundation | Categories with < 10 training sales merged into "infrequent" (a coefficient from 1–4 houses is noise); unseen categories go there too |

Columns left out on purpose: identifiers, `Sale Condition` (constant after the
filter), `Yr Sold`/`Mo Sold` (a numeric year would make the linear model
extrapolate a trend into an unseen year), near-constant columns, and raw
columns replaced by an engineered version. All numeric inputs are
standardized (needed for gradient descent, and so the L2 penalty treats every
coefficient equally).

**What worked and what didn't — feature analysis** (sklearn Ridge, alpha
re-chosen on validation for each row):

| Feature set | Val MAE 2009 |
|---|---|
| Raw numeric columns only, price in $ (no feature engineering) | $16,821 |
| Same, log target | $13,167 |
| Raw numeric + same one-hot categoricals | $12,445 |
| My first feature set, without the interaction | $12,881 |
| My first feature set | $12,705 |
| Raw numeric + my features + one-hot ("everything") | $11,756 |
| **Final: first set + finished basement area** | **$11,965** |

Basement variants tried on validation: raw finished type-1 area $11,965
(chosen); type-1 + unfinished area $12,059; type-1 + type-2 finished area
$12,208; share of basement finished $12,380; log1p of finished/unfinished
areas $12,478–12,490.

Methodological note (honest disclosure): while running this analysis I also
printed test-year MAEs, so the 2010 test set was seen once before the
basement fix. The fix itself was selected on validation only, but the final
test number should be read as slightly optimistic for that reason.

**Which choice mattered most.** Predicting `log(SalePrice)` instead of the
price in dollars: it lowered the validation MAE from $16,821 to $13,167.
The reason is that the value a feature adds to a house is not a fixed amount
of dollars: it depends on the price level of the house, so a model where the
added value works in percentages is much more accurate. Also, most of the
market is between $100k and $250k, while a few houses above $500k produce the
largest errors in dollars; when those errors are squared in the loss, they
dominate the training. Finally, it fits the business framing: the appraiser
cares about relative error, since $20k off is serious on a $100k house and
irrelevant on a $500k one.

**Why my first feature set lost to the raw columns.** `TotalSF` put finished
and unfinished basement area into the same number, and I dropped the columns
that tell them apart. That made the column less informative, because a
finished basement is worth more than an unfinished one, so the model was
noticeably less precise ($12,705 vs $12,445 for the raw columns). Adding the
finished basement area as a separate feature, while keeping `TotalSF`, fixed
it ($11,965).

**Why I did not use the "everything" set.** It was $209 better on validation
(1.7%), but choosing whatever scores best could mean fitting noise: I would be
adjusting the model to the peculiarities of a single year (2009). Also, I
promised the appraiser an explainable estimate, and with many near-identical
columns the coefficients offset each other, so it is no longer possible to
explain what each characteristic adds to the price.

## Modeling: three implementations, one model

**Model:** ridge regression (linear regression with an L2 penalty) on
`log(SalePrice)`. Linear because the target is continuous and the brief asks
for the same simple model three ways; L2 because many engineered features
are strongly correlated, and the penalty makes the solution unique and stable.

**Making the three implementations solve the same problem**

- sklearn `Ridge` minimizes `Σ(error²) + α‖w‖²`. Dividing by n gives
  `mean(error²) + (α/n)‖w‖²`, which has the same minimum, so both PyTorch
  versions use `λ = α/n` with a mean-squared-error loss.
- The intercept is not penalized in sklearn, so the PyTorch penalty is
  applied to the weights only. That is why `torch.optim.SGD(weight_decay=…)`
  is not used: it would also penalize the bias.
- Manual loop: raw tensors, `loss.backward()`, `w -= lr * w.grad` inside
  `torch.no_grad()`, gradients reset each step. Standard: one `nn.Linear`
  inside an `nn.Module`, `optim.SGD` on the full training set each step.
- Same split, same preprocessed matrices, same starting point (weights 0,
  bias = mean of the training target), 50,000 full-batch epochs.

**Tuning (validation year only):** alpha ∈ {0.1 … 100} with sklearn → **α = 3**
(every alpha from 0.1 to 10 is within $10 of each other: regularization
barely matters with 1,538 rows and 67 inputs). Learning rate ∈ {0.003, 0.01,
0.03, 0.1, 0.3} for each PyTorch version → **lr = 0.1** for both; lr = 0.3
diverges (the stability limit for full-batch gradient descent here is ≈ 0.117,
2 / largest curvature eigenvalue).

**Results on the 2010 test year (301 sales)**

| Model | MAE | RMSE | MAPE | Within ±10% | Bias |
|---|---|---|---|---|---|
| scikit-learn Ridge | **$12,422** | $17,696 | 7.5% | 75.7% | −$1,484 |
| PyTorch manual loop | $12,422 | $17,697 | 7.5% | 75.7% | −$1,484 |
| PyTorch nn.Module + optim | $12,422 | $17,697 | 7.5% | 75.7% | −$1,484 |
| Naive baseline (training median, $157,950) | $47,916 | $68,365 | 28.1% | 22.6% | −$14,945 |

Validation 2009 for reference: MAE $11,965 for all three (baseline $53,831).

**Do the three agree?** Yes, after fixing a convergence problem:

| Comparison (test year) | Largest prediction gap | Largest coefficient gap |
|---|---|---|
| manual vs sklearn | 0.010% | 1.5e-3 |
| nn.Module vs sklearn | 0.010% | 1.5e-3 |
| manual vs nn.Module | 0.0001% | 2.9e-6 |

Training objective reached: 0.0076957 for all three.

**What did not agree at first, and why.** The first version (first feature
set, α = 0.1, bias started at 0, 5,000 epochs) left the PyTorch predictions
up to 4.6% away from sklearn's and individual coefficients up to 6.3 away,
while the two PyTorch versions agreed with each other to 2e-6. Diagnosis:

1. *Dummy-variable trap.* The one-hot columns of each categorical variable
   add up to 1, exactly like the intercept's column. Starting from bias 0,
   gradient descent had to climb to log(price) ≈ 12 and spread that height
   over the intercept and every group of dummies (intercept 5.6 instead of
   12.0; `Central Air_N` and `_Y` both ≈ +2.6). Predictions are identical
   along that direction; only the tiny penalty (λ = 6.5e-5) pulls the
   coefficients back, which takes extremely long.
2. *Near-collinear engineered features* (`log_TotalSF`, `log_GrLivArea`,
   `Qual_x_logSF`, …) make the loss surface a long narrow valley, which slows
   full-batch gradient descent along it. Dropping one dummy per group
   (`drop="first"`) did *not* fix the gap (4.8%) because of these.

Fix: start the bias at the mean of the training target (the best constant
prediction, so no "climbing" is needed) and run 50,000 epochs (≈ 4 s per
run). sklearn is unaffected because it solves the problem in closed form
instead of iterating. The remaining 0.01% gap is float32 rounding.

**In short: why the same formula gave different answers.** They are two
different ways of solving the same formula: sklearn uses linear algebra and
gets the exact minimum in one calculation, while PyTorch uses gradient
descent, which only reaches the same point if it takes enough steps and the
path is not too difficult. What fixed it was starting the bias at the mean of
log(price), so it did not have to climb from 0, together with enough steps
(50,000), because the correlated features make the path long.

**Reading the coefficients.** `Overall Qual` is negative (−0.035), but that
does not mean quality lowers the price: it is just a number that adjusts for
quality being worth less in small houses. It has to be read together with
`Qual_x_logSF` (+0.125); the effect of quality is the sum of both and is
positive (in the dashboard, raising quality from 6 to 8 moves a house from
$191,731 to $216,195).

**Is it good enough?** Yes, as a first estimate before a manual appraisal:
the MAE is $12,422 (7.5%), 75.7% of the 2010 houses are within ±10%, and the
error is 74% lower than the baseline's. I would not trust it on its own,
especially in the neighborhoods and price ranges where it fails most (cheap
houses, Sawyer and NAmes: 15 of 301 houses are off by more than 20%).

## Limitations & next steps

Findings from the pipeline (test year 2010):

- **Cheap houses are over-valued.** By price quartile: Q1 ($55k–129k) +3.1%
  average error, MAPE 9.3%; Q4 ($206k–538k) −2.6%, MAPE 6.1%. 15 of 301
  houses are off by more than 20%; the worst are over-valued houses in
  Sawyer, NAmes, SawyerW and Gilbert (+35% to +52%).
- **No market-trend adjustment.** Sale year is deliberately not a feature, so
  the model assumes 2006–08 price levels; it under-values 2010 sales slightly
  (bias −$1,484).
- **Small, partial test year.** 2010 has only January–July sales (301 houses).
- **Population scope.** Only ordinary sales in residential zoning: the model
  does not cover new construction sold before completion (`Partial`),
  foreclosures, or family sales.
- **Test set seen once before the final feature fix** (see feature analysis).
- **One city, 2006–2010.** Not transferable to other markets or periods
  without retraining.

**Causes and next steps**

- *Over-valued cheap houses:* the dataset lacks information about the
  interior condition of the house. Next step: add condition data (e.g. from
  an inspection).
- *No market trend:* the model has no information about how the market moves
  over time. Next step: add an external house price index as a feature.
- *Small, partial test year:* the test set is small (301 sales) and only
  covers January to July 2010.

## Generative AI use disclosure

I used Claude Code (Anthropic) throughout this assignment.

**My decisions:**

- Dataset (Ames Housing).
- Business framing: an appraiser who needs a first estimate.
- Target `log(SalePrice)`.
- Temporal split 2006–08 / 2009 / 2010.
- Population (Normal sales only).
- MAE as the decision metric.
- Which candidate features to include (from a list Claude proposed, adding the porch feature).
- Ran the experiments and diagnostics: feature analysis, convergence investigation.
- Fixing the basement feature after the feature analysis instead of keeping the first set or using all columns.
- The analysis paragraphs 

**Claude Code:**

- Implemented the code (features.py, train.py, models.py, app.py) under my direction, based on the design decisions listed above. I reviewed, ran, and tested it, and requested changes after the feature analysis and convergence investigation.
- Drafted the Dataset and Framing sections and the factual parts of the feature engineering, modeling and limitations sections.

