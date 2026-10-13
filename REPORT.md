# Assignment 1 Report

- **Name**: Javier Arce López
- **Student ID**: 20158
- **Email**: jarcel.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

Ames Housing sales from Dean De Cock (`AmesHousing.csv` in this repo). Each
row is one house sold in Ames, Iowa, with property attributes and the
recorded closing price.

The file is tabular, has a clear price target, and enough features that
preprocessing and engineering actually matter, while staying small enough to
commit and retrain from scratch. That also matches a concrete business use:
checking an asking price against comparable closes.

After dropping three partial-sale outliers (as documented by the dataset creator, De Cock), 2,927 sales remain: 1,938 in
training (2006–2008), 648 in validation (2009), and 341 in test (2010). After preparation, the model matrix has 201 columns.

## Business / real-life framing

A buyer's agent in Ames uses this model before a client writes an offer. The seller has named an asking price. The agent wants an estimate of what that house would close for, based on houses that have already sold, and using only what is known before the deal. That is, the house itself, and the month and year of the search. If the ask sits well above the estimate, the agent will tell the client the ask is aggressive, helping them better estimate the performance of the offer in the market. 

Three partial sales of houses above 4,000 sq ft are removed. Those prices are not comparisons a client could use. The two other houses above that size stay, because their prices match each other.

The split is ordered chronologically. An agent pricing a house in 2010 does not yet have the 2010 sales. The model is fit on 2006–2008, the penalty and the learning rate are chosen on 2009, and 2010 is the test. A random split would train on sales that had not closed yet, resulting in data leackage.

There is no class threshold. A large dollar miss is the mistake that leads to a bad offer, so models are compared by dollar RMSE after the log prediction is converted back to dollars. Dollar MAE is reported next to it so the typical miss is visible too.

Training uses log price because raw prices are strongly right-skewed, and the same dollar miss means something different on a cheap house and an expensive one. The reported errors are still in dollars, which is the unit of the offer.

## Data preparation & feature engineering

**Rows and date fixes.** Three Edwards partial sales above 4,000 sq ft
(orders 1499, 2181, 2182) are removed; their prices are not usable
comparables. Two other large houses stay. Three date typos are corrected on
the full file before the split (garage year 2207 → 2007; remodel before
build → build year; remodel after sale → sale year).

**Missing values.** Blanks that mean “no alley / basement / fireplace /
garage / fence / masonry” become `"None"` (and related areas or baths 0).
Real gaps are filled from 2006–2008 only: lot frontage by neighborhood
median (then overall training median), masonry area by veneer-type median,
basement exposure and finish among houses with a basement, electrical by
training mode, and two incomplete detached garages from other training
detached garages.

**Columns dropped.** Identifiers; `Sale Condition` / `Sale Type` (deal
outcome, unknown beforehand — condition is used only to spot the three
partials); near-constant fields; redundant area/car columns that duplicate
`Gr Liv Area`, basement totals, or `Garage Area`; raw year columns replaced
by ages; `Pool QC` / `Misc Feature` (almost always blank) while keeping
`Pool Area` and `Misc Val`.

`Bldg Type` is also dropped, because `MS SubClass` already encodes it.
Subclass 90 and `Bldg Type = Duplex` were the same column, which left the
matrix one rank short, so the OLS coefficients were not unique.

**Features added.** The model should see how old the house is at the sale,
not the raw calendar years it was built or remodeled. So three ages replace
those year columns:

- `Age` = year sold − year built
- `RemodAge` = year sold − year remodeled
- `GarageAge` = year sold − garage year built, or **0** if the house has no
  garage

Houses without a garage have no garage build year. Filling that blank with
a typical year would invent a construction date the house never had.
`GarageAge = 0` means “no garage age,” which matches treating “no garage”
as a real state rather than a missing number. The raw year columns are
dropped after these ages are built so the same information is not counted
twice.

`Mo Sold` and `Yr Sold` are kept because the agent already knows when the
client is shopping; month and year can move prices, and they are known
before the offer (unlike sale type or sale condition). Encoding differs on
purpose:

- **Year stays a single number.** Training only sees 2006–2008. Validation
  and test are 2009 and 2010. A one-hot year indicator for those later years
  would be all zeros in training, so the model could never learn it. A
  numeric year can still carry a rough time trend into years the model has
  not trained on.
- **Month is categorical (one-hot).** Treating month as 1…12 would tell the
  model that December is twelve times January. Months are labels for season,
  not a linear scale, so each month gets its own coefficient.

**Encoding and scaling.** Every category is one-hot encoded, including the
ordered quality fields, `MS SubClass`, and month. Levels with fewer than 10
training rows are collapsed to `Other`. Levels are sorted A–Z and the first
is dropped, so the intercept stands for it. When a field has no rare level in
training, a value never seen in training gets the most common training level
instead. Otherwise it would silently get the reference level. For expmple, one 2010 house
with a poor kitchen would have been coded as an excellent one.

One-hot `None` levels repeat across fields: a house with no basement is
`None` in all five basement fields, and the same holds for garages. Seven
identical columns are dropped, and the pipeline stops if any others appear.
The 201 columns are standardized with a scaler fit on training only.
Neighborhood medians, the rare-level map, dummy columns, and the scaler never
see validation or test.

**Feature experiments.** Each row changes one choice, or adds one candidate
feature, relative to the final pipeline (sklearn Ridge, α=30, scored on 2009
only; notebook section 15). A difference under about $100 is the size of the
Ridge-versus-OLS gain, which we treat as noise.

| Experiment | Columns | Val RMSE | vs final |
|---|---:|---:|---:|
| Final pipeline | 201 | 18,848 | — |
| Ordered fields as integer codes (Po 1 … Ex 5) | 155 | 19,242 | +394 |
| Keep `Bldg Type`, drop only the Duplex dummy | 204 | 18,806 | −42 |
| Raw years instead of ages | 201 | 18,850 | +2 |
| Month as a number 1–12 | 191 | 18,775 | −73 |
| Restore `Garage Cars` and `Exterior 2nd` | 213 | 18,795 | −53 |
| Restore near-constant fields | 208 | 19,093 | +245 |
| + TotalSF (living area + basement) | 202 | 18,846 | −1 |
| + TotalBaths | 202 | 18,849 | +1 |
| Log of `Lot Area` and `Gr Liv Area` | 201 | 19,135 | +287 |
| + `Overall Qual` × `Gr Liv Area` | 202 | 18,859 | +11 |
| All four candidates together | 204 | 18,877 | +29 |

- **Ordered fields one-hot (adopted).** The pipeline first coded quality
  grades as equally spaced integers. One-hot fits 2009 better by $394, the
  largest effect in the table, so equal steps between grades do not hold. We keep the decision made on validation. 
- **Ages versus raw years: no difference.** With `Yr Sold` in the model,
  `Age = Yr Sold − Year Built` is a linear recombination of columns already
  present, so a linear model can fit the same thing either way. The ages stay
  because they are easier to read, not because they predict better.
- **Sums add nothing.** TotalSF and TotalBaths are sums of columns already in
  the model.
- **Log areas: RMSE +$287, MAE −$207.** They help the typical house and hurt
  expensive ones, which dominate RMSE. All four candidates together show the
  same pattern (RMSE +$29, MAE −$311). Not adopted, because RMSE is the
  selection metric, so we have to be coherent with our previous criteria.
- **Near-constant fields are worse by $245**, which confirms dropping them.
- **`Bldg Type`, month as a number, and `Garage Cars` / `Exterior 2nd` are
  within noise** (−$42 to −$73), so the original choices stay. Month as a
  number doing slightly better with 10 fewer columns suggests seasonality is
  weak in this data.

## Modeling: three implementations, one model

The shared model is linear regression on `log(SalePrice)`, the continuous
target. We used the same features for all three fits. Training minimizes error in log
space, but reported metrics convert predictions back to dollars. The naive baseline predicts 
the training geometric mean (~$167,485) for every house.

**Tuning** A small Ridge α grid picked α=30 due to it having the best
val dollar RMSE among OLS and Ridge. Lasso was checked as a reference though it
was slightly worse than Ridge on val. For
both PyTorch trainers, learning rate was chosen on the same val set with
α fixed at 30; lr=0.05 was best among runs that converged (0.1 diverged). Test
2010 was not used to pick α or lr.

**Results (dollar RMSE / MAE).** Val was used to choose the model. Test provides the reported metrics.

| Model | Train RMSE | Val RMSE | Test RMSE | Test MAE |
|---|---:|---:|---:|---:|
| Naive baseline | 81,753 | 81,985 | 74,697 | 53,098 |
| sklearn Ridge (α=30) | 17,338 | 18,848 | 21,713 | 15,247 |
| manual PyTorch Ridge | 17,338 | 18,848 | 21,713 | 15,247 |
| standard PyTorch Ridge | 17,338 | 18,848 | 21,713 | 15,247 |

The three fits agree, because they are all equivalent implementations.

## Limitations & next steps

**2010 is incomplete.** Sales stop in July, so test error mixes model
quality with a shorter season. We should attempt to find the full data for the year to obtain more accurate and reliable metrics.

**No asking prices in the file.** We only have the close prices, which are very valuable data, but for future we could attempt to collect ask prices, which would help us get a feel of the ask-close spread.

**One-hot quality fields drop the order.** They cost 46 columns, and a
grade that is rare in training has no coefficient of its own: it is grouped
as `Other`, or given the most common grade if it never appears. A middle
ground, such as merging adjacent grades or keeping integer codes for the
fields where equal steps fit, was not tested.

**Non-arms-length sales stay in the target.** Family and abnormal closes
pull the estimate down on purpose for a conservative check. For a
retail-only appraisal, filter to normal sales and refit. We should expect a higher
mean prediction and a different error profile.

**Ridge’s validation gain is small and reversed on test.** α=30 beat OLS by
about $61 RMSE on 2009 and lost about $150 on 2010. We treat regularization as
stabilization, not a proven accuracy lift. Having more years of validation,
 we should re-check whether α=0 is enough.

## Generative AI use disclosure

Cursor was used for implementation and notebook & report formatting.
Dataset choice, business framing, split/metric decisions, feature
engineering choices, and the conclusions in this report are mine.
