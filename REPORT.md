# Assignment 1 Report

- **Name**: Louis de Perier
- **Student ID**: 20159
- **Email**: ldeperier.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

I downloaded the Ames Housing Dataset from the authoritative source at `jse.amstat.org/v19n3/decock/AmesHousing.txt` rather
than from Kaggle, because the Kaggle `house-prices` competition file is a 1,460-row
subset of the same data; the original has **2,930 rows and 82 columns**.

One row is one residential property sale in Ames, Iowa between 2006 and 2010. The
columns describe the physical property (lot size, living area, number of rooms,
basement and garage specifications), its condition and construction quality across
several graded scales, its location by neighbourhood, and the circumstances of the
sale. The target is `SalePrice`, the transaction price in US dollars.

I picked it for three reasons. First, the target is continuous and genuinely
predictable, so linear regression is the honest model choice rather than a forced one.
Second, the feature engineering is non-trivial in several distinct ways at once —
ordered quality grades stored as text, structural versus genuine missingness, and
relationships a linear model cannot express without help. Third, it is small: 960 KB
on disk, which commits to the repository and retrains in under three seconds.

## Business / real-life framing

**The decision.** A mid-size mortgage lender operating in Ames needs to value each
property offered as collateral before approving a loan. A full appraisal by a licensed
appraiser costs roughly $500 and takes five to ten business days. The lender wants an
automated valuation model (AVM) to price the straightforward cases instantly, so human
appraisers are reserved for properties where the model is not confident. The model does
not decide whether to lend, it decides **whether a human needs to look**.

**This makes errors asymmetric, which drives the metric.** Overvaluing means the lender
advances too much money against too little collateral and takes a real loss on default.
Undervaluing means asking for a larger deposit or losing the customer. Errors are also proportional rather than absolute: being
$20,000 wrong on a $100,000 house is a serious failure, while the same $20,000 on a
$700,000 house is within normal market noise. For both reasons I model
**`log(SalePrice)`**, so the model minimises proportional error, and I report **median
absolute percentage error** and **share of properties priced within 10%** rather than
relying on R² alone.

**The threshold is an error tolerance, not a class boundary.** This is a regression
problem, so there is no probability cut-off, but the business still sets a threshold:
the maximum predicted error at which it will skip the appraisal. Tighten it and
valuations are more accurate but fewer cases automate, so appraisal spend stays high.
Loosen it and more cases clear automatically but more bad valuations slip through. The
dashboard exposes this as a slider over tolerance, showing the auto-valued share, the
escalated share, and net cost against the $500-per-appraisal baseline.

**Target definition needs care, because not every sale is a market price.**
`SalePrice` records what changed hands, not what the property was worth. The
`Sale Condition` field marks foreclosures and adjustments (`Abnorml`), transfers between
family members (`Family`), and incomplete new builds sold by the developer (`Partial`),
together about 18% of rows. None are arm's-length transactions, so training on them
teaches a price the lender's use case will never encounter. I therefore define the
target as the sale price of **normal arm's-length sales only**, and state this as an
explicit scope limitation rather than silently dropping rows.

**A random split would leak the future into training, so the split is chronological.**
The data spans 2006–2010, covering the US housing downturn, and the lender's situation
is always the same: a model fitted on past sales prices properties that have not sold
yet. A random split would let 2010 transactions inform predictions about 2007 —
information no deployed model could have — and would report a score the lender cannot
reproduce in production. I split on `Yr Sold`: **2006–2008 train (1,550), 2009
validation (561), 2010 test (301)**. The test score therefore answers the lender's
actual question: how well does this price next year's houses?

## Data preparation & feature engineering

**Scope and row filtering.** Of 2,930 records I kept **2,412**. The 518 dropped are the
517 non-arm's-length sales excluded per the framing above, plus one house over 4,000
square feet that De Cock's own documentation flags as distorting least-squares fits.
`Sale Condition` becomes constant afterwards and is dropped, since a constant column
cannot explain variation. The cost of this choice is a smaller training set and a model
valid only for normal market sales. If the lender ever needs to value a foreclosure,
this model does not apply and must escalate to a human.

**Missing values are two different problems, and conflating them is a modelling error.**
27 columns contain nulls, but most are not missing data at all. `Pool QC` is 99.6% null
because almost no house has a pool; `Alley` (93.2%), `Fence` (80.5%) and `Fireplace Qu`
(48.5%) are the same. These are filled with `"None"` or `0` — the **true value**, not an
imputation. Filling them with a mode or median would tell the model that every house has
an average-quality pool. Only **`Lot Frontage`** (16.7% null) is genuinely unknown, and
it is imputed with the neighbourhood median learned from training rows alone, because
street frontage varies systematically by neighbourhood.

`Garage Yr Blt` needed separate treatment: both fills are wrong. Zero would place the
garage in year zero and destroy any derived age feature; the median would invent a
garage that does not exist. I recorded absence in a `Has Garage` flag and filled the
year with the house's own build year, keeping derived garage age on a sensible scale.

**Ordinal grades stay ordered.** Seventeen columns are ranked scales stored as text
(`Po < Fa < TA < Gd < Ex`). One-hot encoding is the default move and it is wrong here —
it discards the ranking and spends five coefficients where one suffices. I mapped them
to integers with `"None"` at zero, below `"Poor"`, which is the correct position: a house
with no fireplace ranks below a house with a poor one. This assumes equal spacing between
grades, which is certainly false in dollar terms; it is a deliberate simplification.

**Engineered features encode what linear regression cannot compute.** A linear model can
only add weighted columns. It cannot subtract, multiply or wrap its own inputs, so each
of these must be supplied explicitly:

| Feature | Why a linear model needs it |
|---|---|
| `Age`, `Years Since Remodel`, `Garage Age` | Differences from `Yr Sold`; the model cannot compute a difference |
| `Qual x SF` | Interaction: a square foot of excellent construction is worth more than one of poor construction |
| `Total SF`, `Total Baths`, `Total Porch SF` | Consolidate quantities split across 5, 4 and 5 raw columns |
| `Mo sin`, `Mo cos` | Months are cyclical; as integers, December sits 11 units from January instead of adjacent |
| `Has Garage`, `Is Remodeled`, `Is New` | Binary states otherwise buried in sentinel values |

**Leakage control.** All row-wise transforms (structural fills, ordinal maps, arithmetic)
run before the split, because they use no information from other rows. Every statistic
that *does* use other rows — frontage medians, retained category levels, scaling mean and
standard deviation — is fitted on **training rows only** and applied frozen to validation
and test. Categorical levels below 1% frequency in training collapse to `"Other"`, which
also absorbs any level appearing for the first time in 2010. The design matrix is **167
features** after dropping one dummy per category (a full set is collinear with the
intercept and makes the least-squares system singular) and removing zero-variance columns.

**Standardisation is not cosmetic.** scikit-learn solves linear regression in closed form
and is indifferent to feature scale, but both PyTorch versions use gradient descent. With
raw features, `Total SF` spans thousands while `Central Air` is 0 or 1; the loss surface
becomes a long narrow valley and no single learning rate suits both. Standardising is
what makes the three implementations comparable at all.

## Modeling: three implementations, one model

**Model choice.** Linear regression, because the target is continuous. I model
`log(SalePrice)` so the loss penalises proportional rather than absolute error, matching
the business framing. Ridge regularisation at `alpha = 100` was selected on the 2009
validation set.

All three were trained on the same 1,550-row training set with identical features and
identical regularisation, and evaluated on the same untouched 301-row 2010 test set.

| Model | RMSE (log) | R² | MedAPE | Within 10% | RMSE ($) | Fit time |
|---|---|---|---|---|---|---|
| Naive baseline (train mean) | 0.3573 | −0.000 | 23.60% | 22.6% | $67,511 | — |
| scikit-learn `Ridge` | **0.0997** | **0.922** | **6.31%** | **71.4%** | **$17,593** | 0.00s |
| Manual PyTorch loop | **0.0997** | **0.922** | **6.31%** | **71.4%** | **$17,593** | 0.52s |
| Standard `nn.Module` + `optim` | **0.0997** | **0.922** | **6.31%** | **71.4%** | **$17,593** | 1.49s |

Train / validation / test RMSE (log) is 0.0814 / 0.0941 / 0.0997 for all three.

**Do the three agree? Yes — to five decimal places.** Maximum absolute coefficient
difference against scikit-learn is **0.00000** for both PyTorch versions across all 168
parameters, and training MSE is 0.006626 for all three. This is the expected result and
it is worth being precise about why. scikit-learn does not iterate: it solves the ridge
problem in closed form, computing the exact minimum directly. The two PyTorch versions
search for that same minimum by gradient descent. Because the objective is strongly
convex, there is exactly one minimum and no local traps, so given enough epochs at a
stable learning rate they must arrive at the same place. Agreement to five decimals is
evidence that all three are solving the same problem correctly.

**The three differ in cost, not in answer.** The closed-form solve is roughly 150× faster
than the manual loop and 400× faster than the standard workflow. The standard workflow is
the slowest despite doing identical arithmetic, because `nn.Module` and the optimizer add
per-step Python overhead that the raw tensor loop avoids. For a problem of this size the
ergonomic benefit of `torch.optim` is not worth paying for; it would become worth it the
moment the model stopped being a single linear layer.

**What went wrong before it went right (two findings).**

*First*, before regularisation the three did **not** agree exactly. scikit-learn reached
a training loss of 0.006397 against 0.006399 and 0.006408 for the PyTorch versions —
scikit-learn lowest, as it must be, since it computes the exact optimum. Yet both PyTorch
versions scored marginally **better on test** (RMSE 0.1011 and 0.1007 against 0.1015).
With 167 features on 1,550 rows the exact least-squares solution overfits slightly, and
gradient descent halted at 3,000 epochs had not fully arrived there. Early stopping was
acting as weak implicit regularisation. Adding explicit L2 made the problem strongly
convex, which improved conditioning enough that gradient descent now reaches the exact
optimum, and the divergence disappeared.

*Second, and more seriously*, my first regularised run produced an R² of **−3.302** for
the standard PyTorch workflow while the other two were fine. The diagnostic that found it
was the coefficient comparison: correlation with scikit-learn was a perfect 1.000000 but
maximum absolute difference was 0.727. The slopes were right and exactly one parameter
was wrong — the intercept. `torch.optim.SGD` applies `weight_decay` to **every**
parameter including the bias, whereas scikit-learn's `Ridge` never penalises the
intercept and my manual loop penalised only the weights. Log prices centre near 12.0, so
shrinking the bias toward zero destroyed predictions while leaving the slopes untouched.
The fix was parameter groups, setting `weight_decay=0.0` for the bias term. This is worth
recording because the failure is silent in a different configuration: on a target already
centred at zero, the same bug would be nearly invisible.

**Hyperparameters, and the conversion between frameworks.** Only two were tuned, both on
validation with the test set untouched. Learning rate was swept over six values: 0.001
under-converges within 3,000 epochs, 0.05 is the best stable value, and **0.1 and above
diverge to `nan`**, because gradient descent on a quadratic loss converges only while the
step stays below the `2/L` stability bound. Ridge strength was swept over nine values and
`alpha = 100` won, improving validation RMSE from 0.0945 to 0.0941.

Matching regularisation across frameworks required a conversion that is easy to get
wrong. scikit-learn's `Ridge` minimises a **summed** squared error plus `alpha‖w‖²`;
PyTorch's `MSELoss` uses the **mean**, and `weight_decay` corresponds to a penalty of
`(wd/2)‖w‖²`. Equating them gives **`weight_decay = 2·alpha/n`** — with `n = 1550` and
`alpha = 100`, that is 0.129. Passing `alpha` directly into `weight_decay` would have
under-regularised the PyTorch models by a factor of 775 while still producing
plausible-looking output. I verified the conversion empirically rather than trusting the
algebra: coefficients match to 0.0000.

**Business reading of the result.** The model prices **71.4%** of 2010 properties within
10%, against 22.6% for the naive baseline. Typical error is 6.31% of sale price. Under
the framing above, that is the share of valuations that could plausibly skip a $500
appraisal — though see the first limitation below before taking that number at face value.

## Limitations & next steps

**1. Regularisation barely helped, which tells us the model is near its ceiling for this
feature set.** Ridge moved validation RMSE from 0.0945 to 0.0941 — under half a percent.
I expected 167 features on 1,550 rows to overfit badly and for ridge to rescue it, but
the train/validation gap was only 0.0145 to begin with. The feature engineering had
already done the work regularisation would otherwise perform. *Next step:* the remaining
error is more likely structural than variance-driven, so the productive direction is
interaction terms and non-linear basis expansions (splines on `Total SF` and `Age`), not
stronger penalties. A gradient-boosted tree would be the natural benchmark to confirm how
much linear structure is actually leaving on the table.

**2. The test set is 301 rows and only covers January–July 2010.** The data collection
ends mid-year, so the test set contains no autumn or winter sales. Since I engineered
cyclical month features, the test set cannot validate half of their range. *Next step:*
report confidence intervals on the test metrics via bootstrap rather than single point
estimates, and if the lender has access to post-2010 county records, extend the holdout
to a full year before trusting the 71.4% figure operationally.

**3. The time-based split was methodologically correct but empirically did not matter
much.** Median sale price is $157,250 in train and $159,000 in test — Ames barely felt
the national housing crash, so the distribution shift I designed the split to capture was
small. I am keeping the split because it is the right decision under the framing and
because a random split would have produced a number the lender could not reproduce, but I
should not claim it as a source of robustness it did not provide here. *Next step:* test
the same pipeline on a market with a real crash (Las Vegas, Phoenix) to see whether the
approach survives genuine shift.

**4. Modelling in log space introduces back-transform bias.** `exp(mean of log)` is the
geometric mean, which is systematically below the arithmetic mean. Every dollar prediction
in this report is therefore slightly low, and for a lender a systematic downward bias is
the *safer* direction but still a known error. *Next step:* apply a Duan smearing estimator
or a variance correction `exp(σ²/2)` and quantify the shift, rather than leaving it
unmeasured.

**5. Excluding 18% of rows narrows what the model can be used for.** The model is valid
only for normal arm's-length sales, and the lender has no automated way to know a sale is
abnormal *before* it happens — `Sale Condition` is recorded at transaction time, not at
valuation time. *Next step:* train a second classifier predicting `Sale Condition !=
Normal` from pre-transaction features, and use it as a routing gate so that properties
likely to transact abnormally go straight to a human appraiser. This is the honest fix;
without it, the 71.4% figure is optimistic because it is measured on a population the
production system cannot perfectly identify in advance.

**6. The ordinal encoding assumes equal spacing between quality grades.** Mapping
`Po/Fa/TA/Gd/Ex` to `1/2/3/4/5` asserts that the price gap from Fair to Typical equals
the gap from Good to Excellent. This is almost certainly false. *Next step:* fit the
spacing from data by target-encoding each grade on the training set, or test one-hot
encoding for the three or four quality columns with the largest coefficients to see
whether the ordinal constraint is costing accuracy.

## Generative AI use disclosure

I used AI tools such as Claude and ChatGPT to help me write code and enhance my writing skills.
