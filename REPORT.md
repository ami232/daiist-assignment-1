# Assignment 1 Report

- **Name**: Maria Roman
- **Student ID**: 19991
- **Email**: mroman.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

IBM's public "Telco Customer Churn" dataset (downloaded from the IBM telco-customer-churn-on-icp4d GitHub repository and committed as `data/telco_churn.csv`, about 1 MB). One row is one customer of a telecom operator. It has 7,043 rows and 21 columns: demographics, contracted services, contract and billing information, tenure, monthly and total charges, and the target `Churn` (Yes/No, 26.5% positive). I picked it because it is tabular, small enough to commit, has a clear binary target with a natural business cost, and has many categorical variables where feature engineering is possible.

## Business / real-life framing

Scenario: the retention team of the operator can contact a limited set of customers and offer a discount to keep them. The model scores every customer with a churn probability and the team contacts those above a threshold.

- **Target**: `Churn = 1` if the customer left. The dataset is a single snapshot with no date per row.
- **Split**: because there is no time column, a time-based split is impossible. I used a random split stratified by the target (60/20/20 train/validation/test) so all three sets keep the same 26.5% churn rate. In production a random split could overestimate performance, because the model would be used on future customers; this is listed as a limitation.
- **Leakage control**: the scaler is fitted on the training set only, and hyperparameters and the decision threshold are chosen on the validation set. The test set is used only once, for the final comparison.
- **Metric and threshold**: accuracy is misleading here (always predicting "no churn" already gives 73.5%). The threshold is chosen to minimise business cost, not at 0.5. I assume (hypothetical numbers) a missed churner (false negative) costs 500 EUR of lost customer value and an unnecessary retention offer (false positive) costs 50 EUR. With a 10:1 cost ratio the optimal threshold is low: the cost-minimising threshold on the validation set is about 0.08. The dashboard lets the user change both costs and the threshold.

## Data preparation & feature engineering

- **Data quality**: `TotalCharges` has 11 blank values. All of them are customers with `tenure = 0` (just joined, nothing billed yet), so I set them to 0. `customerID` is dropped because it is an identifier.
- **Encoding and scaling**: categorical variables are one-hot encoded (dropping the first level), and all features are standardised with a scaler fitted on the training set.
- **Engineered features**: `n_services` (number of contracted services), `tenure_group` (0-12, 13-24, 25-48, 49-72 months), `is_month_to_month`, `autopay` (payment by automatic transfer or card) and `price_vs_avg_paid` (monthly charge minus the average amount paid per month so far).
- **What did not work**: I refitted the scikit-learn model removing each feature on the validation set. Validation ROC-AUC stays at about 0.860 and log-loss at about 0.401 in every case (full model 0.8600 / 0.4008; without all five engineered features 0.8602 / 0.4013). So none of them helps. The reason is that a linear model already has the information: `is_month_to_month` duplicates one of the `Contract` dummy columns, `autopay` and `n_services` are linear combinations of existing dummy columns, and a linear model cannot gain from them. `tenure_group` was meant to capture non-linearity but gave no gain either. The honest conclusion is that for a logistic regression on this dataset, feature engineering matters less than expected, and that the ablation is what told me.

## Modeling: three implementations, one model

Model: logistic regression (binary target), because the business needs calibrated probabilities that can be compared with a cost threshold, and a linear model is easy to explain. The same train/validation/test split and the same L2 regularisation strength are used for the three implementations:

1. scikit-learn `LogisticRegression` (C tuned on validation among 0.01, 0.1, 1, 10; best C = 10).
2. Manual PyTorch loop: raw tensors, sigmoid, binary cross-entropy plus L2 penalty, `loss.backward()` and a manual gradient step (full batch, 1,000 epochs).
3. Standard PyTorch: `nn.Module` with one `nn.Linear`, `BCEWithLogitsLoss` and `torch.optim.SGD` (same settings, L2 as `weight_decay` on the weights only).

The learning rate was tuned on validation among 0.05, 0.2, 0.5 (best 0.5 for both). The sklearn C was converted to the PyTorch L2 strength with lambda = 1 / (C * n).

Test-set results (1,409 customers, threshold 0.5 for accuracy and F1):

| Model | Accuracy | F1 | ROC-AUC | Log-loss |
|---|---|---|---|---|
| Naive baseline (always "no churn") | 0.7346 | 0.0000 | 0.5000 | 0.5787 |
| scikit-learn | 0.7864 | 0.5567 | 0.8339 | 0.4290 |
| PyTorch manual loop | 0.7878 | 0.5596 | 0.8328 | 0.4302 |
| PyTorch nn.Module + optim | 0.7878 | 0.5596 | 0.8328 | 0.4302 |

Business cost on the test set with the assumed costs (500 EUR per missed churner, 50 EUR per unnecessary offer): flagging nobody costs 187,000 EUR; the model at threshold 0.5 costs about 97,800 EUR; the model at the cost-minimising threshold of 0.08 costs about 37,000 EUR.

Do the three agree? Yes, closely. The two PyTorch versions give identical results because they implement the same mathematics (same loss, same initialisation at zero, same full-batch gradient descent). scikit-learn differs slightly (ROC-AUC 0.8339 vs 0.8328) because it uses a different optimiser (L-BFGS, run to convergence), while the PyTorch versions use plain gradient descent for a fixed number of steps and may not be fully converged.

## Limitations & next steps

- **Random split on a snapshot**: there is no time information, so the test performance may be optimistic for future customers. With dated data I would train on earlier months and test on later ones.
- **Invented costs**: the 500 EUR and 50 EUR figures are assumptions. With real customer lifetime value and offer cost I would recompute the threshold; the dashboard already allows it.
- **Weak benefit over a simple rule**: flagging every customer costs about 51,750 EUR, so the model at its best threshold (about 37,000 EUR) is better but not dramatically. A real offer also does not retain everyone it reaches; I would add a retention success rate to the cost model.
- **Hyperparameters at the edge of the grid**: the best C (10) and learning rate (0.5) were the largest values I tried, so the real optimum may be higher. I would extend both grids.
- **Feature engineering did not help**: the linear model cannot exploit combinations of existing features. Next step would be interaction terms (e.g. contract type x tenure) or a non-linear model (gradient boosting) to check whether those features help there.
- **Not fully converged PyTorch**: more epochs or a better optimiser (Adam) would reduce the small gap with scikit-learn.

## Generative AI use disclosure

I used ai chatbot to help me with some parts of the assignment. Overall, I ran all the code on my own machine, checked the results and reviewed every section so that I can explain them.
