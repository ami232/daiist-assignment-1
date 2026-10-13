# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Celia Sanmartín
- **Student ID**: 18798
- **Email**: csanmartin.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Dataset
I used the Ames Housing dataset. 
It contains residential home sales in Ames, Iowa, from 2006 to  2010. I downloaded it from OpenML (dataset id 42165, the training set of Kaggle's "House Prices" competition) using scikit-learn's `fetch_openml` and saved it as `data/ames.csv`.

Each row represents one house sale. The dataset has 1,460 rows and 81 columns: 79 describe the house (size, overall quality, neighbourhood, year built, garage, etc.), plus an `Id` column and the target, `SalePrice`.

I picked it because housing is kind of a trending topic nowadays in Spain mainly and lately i have read a lot about it

## Business / real-life framing

A retail bank uses the model as a second opinion on property valuations when assessing a mortgage application: if the model's estimate and the appraised value differ a lot, the application is flagged for review.

The model predicts `SalePrice` directly, in dollars, rather than its logarithm. I chose this so that the RMSE is expressed in dollars, which is easy for the bank to interpret. The trade-off is that `SalePrice` is right-skewed (skewness of about 1.9), so a few expensive houses can pull the linear model; I discuss this in the limitations.

Because the bank would value houses sold after the model is built, I split the data by sale year instead of randomly: the model is trained on earlier years and tested on the most recent ones. A random split would be misleading because test houses would be sold in the same years as many training houses, so the model would be evaluated on a market it has already seen, while in practice the bank would value houses sold after the model was built. This would make the reported error lower than the real one.

I assume that overvaluing and undervaluing a house cost the bank a similar amount [your reason: for example, losing customers vs. default risk], so I do not need an asymmetric error. I use RMSE as the main metric because it penalises large errors more heavily, and large valuation mistakes are the ones that matter most for this decisionnnnn

## Data preparation & feature engineering
In this dataset many empty cells do not mean missing data but "the house does not have this feature" (pool, alley, fence, fireplace, garage, basement). I filled these 15 columns with an explicit "None" category, so the model can tell a house without a pool from a house with one. The few real gaps were handled as follows: `LotFrontage` (about 18% missing) with the median of the house's neighbourhood, because lot frontage varies a lot between neighbourhoods (neighbourhood medians range from 21 to 91, against an overall median of 69), with the overall median as a fallback; `MasVnrArea` (8 rows) with 0; `Electrical` (1 row) with the most frequent value; and `GarageYrBlt` was dropped because it is undefined for houses without a garage and largely redundant with `YearBuilt` and `GarageType`. All fill values are learned on the training set only and then applied unchanged to validation and test, so no information from later years leaks into training.
`HouseAge` and `RemodAge` (year of sale minus year built / remodelled), `TotalSF` (basement + first + second floor), `TotalBath` (half baths count 0.5) and four yes/no indicators (`HasPool`, `HasGarage`, `HasFireplace`, `HasBasement`). An age is more meaningful for a linear model than a calendar year, because it describes the house at the moment of sale; a total area summarises size in one column instead of three correlated ones. I dropped the original columns that these features summarise, because keeping a total together with its parts adds redundant information, which matters when comparing three implementations of the same linear model. I also dropped `YrSold`: the test year (2010) lies outside the training years, so the model would have to extrapolate.
 Ten quality scales (None < Po < Fa < TA < Gd < Ex) were encoded as 0-5, which assumes each step is worth the same. The other 34 text columns (including `MSSubClass`, a category code stored as a number) were one-hot encoded keeping all categories (dropping the first level of each to avoid redundant columns), which gives 233 model columns for 947 training houses. Numeric columns were standardised, which the gradient-descent models need. The impact of keeping all categories is that some categories in 2009-2010 never appear in the training set (for example some `Exterior1st` values) and are encoded as all zeros, so the model treats them like the reference category. For the bank, valuations of houses with unusual features are therefore less reliable and should go to manual review.


## Modeling: three implementations, one model
I use Ridge regression (linear regression with an L2 penalty). With 233 columns and only 947 training houses, plain least squares starts to overfit: in a preliminary comparison its training RMSE was about $20,900 against about $31,100 on validation. Ridge reduces that gap. The only hyperparameter tuned for the model is the penalty strength `alpha`, chosen on the validation year from [1, 3, 10, 30, 100, 300]; the best value was `alpha = 30` (validation RMSE $28,859). The target is standardised with the training mean and standard deviation, and all three models learn on that scale; predictions are converted back to dollars.

The three implementations.
1. scikit-learn `Ridge(alpha=30)`, which solves the problem directly.
2. A manual PyTorch loop: weights and bias as tensors, a loss of mean squared error plus `(alpha / n) * ||w||^2`, gradients from autograd and a hand-written update `w -= lr * w.grad`.
3. The standard workflow: an `nn.Module` with one `nn.Linear` layer, `nn.MSELoss` and `torch.optim.SGD`, where `weight_decay = 2 * alpha / n` on the weights (not on the bias) gives exactly the same penalty.

Both PyTorch versions use full-batch gradient descent, zero initial weights, `lr = 0.02` and 10,000 epochs. In preliminary tests `lr = 0.01` converged more slowly and `lr >= 0.1` diverged (the loss became `nan`).

**Results** (same test set, 2010 sales; errors in dollars):

| Model | Validation RMSE | Test RMSE | Test MAE |
|---|---|---|---|
| Baseline (always the mean training price) | 80,804 | 80,359 | 58,733 |
| scikit-learn Ridge | 28,859 | 28,647 | 19,075 |
| Manual PyTorch loop | 28,877 | 28,716 | 19,129 |
| nn.Module + torch.optim | 28,877 | 28,716 | 19,129 |

All three models cut the test RMSE by about 64% compared with the naive baseline. The average absolute error is about $19,100, roughly 11% of the average 2010 sale price (about $177,000).

**Do the three agree?** The manual loop and the `nn.Module` are identical (maximum difference between their test predictions: $0), as expected, because they perform the same mathematical steps with different APIs. scikit-learn differs slightly: the test RMSE differs by about 0.2%, the largest difference in one house's predicted price is about $2,667, and the weights differ by at most 0.0124 (in standardised units). The likely cause is that scikit-learn solves the Ridge problem exactly in double precision, while gradient descent approaches that solution step by step in single precision; with 233 correlated columns some directions converge slowly, and the training loss flattens at about 0.1545 after 10,000 epochs, very close to the exact minimum. This is my explanation from the loss curve and the size of the gap; I did not run a separate experiment to prove it.

## Limitations & next steps

1. **Small and noisy test set.** The test year has only 175 sales, and 2010 is incomplete (sales stop in July), so the test error is a noisy estimate of how the model would perform on a full year. With more time I would evaluate with several rolling time windows (train on earlier years, test on the next one) to see how stable the error is.
2. **Skewed target.** `SalePrice` has a skewness of about 1.9 and I predict it in dollars, so expensive houses may be modelled worse than cheap ones. I would compare against a model of `log(SalePrice)` and report percentage error as well as dollar error.
3. **Outliers.** Some houses are extreme (the largest `TotalSF` in the training set is 11,752 square feet, against a 99th percentile of about 4,700) and I did not remove them. I would inspect them and test the effect of excluding or capping them on the error.
4. **Unseen and rare categories.** Categories that do not appear in the training years (for example some `Exterior1st` values in 2009) are encoded as zeros, and many one-hot columns correspond to very few houses. With more data I would group rare categories, or flag houses with unseen categories for manual review.
5. **A linear model with simple encodings.** It cannot capture interactions (for example size in expensive neighbourhoods), and the 0-5 quality codes assume that every step has the same value. I would add a few interaction terms and compare against a tree-based model as a check.
6. **Small hyperparameter search.** `alpha` was chosen from six values using a single validation year, and the PyTorch learning rate and number of epochs were set after a few tests. With more time I would use time-based cross-validation and run more epochs (or an optimiser such as L-BFGS) to close the small gap with scikit-learn.

## Generative AI use disclosure
Helped me with data processing and with  coding doubts.
