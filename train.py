#---------------------------------------------------------------------------------------
# 1. DATA LOADING

import pandas as pd
import numpy as np
df = pd.read_csv("data/telco_churn.csv")
print(df.head())

# 1.1. Data exploration
print("\nDataset shape:")
print(df.shape) #7043 rows x 21 columns

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info() # review = totalcharges is a str = should be numerical

#---------------------------------------------------------------------------------------
# 2. DATA QUALITY NUMERICAL CHECK

# 2.1. duplication (of customers and rows)
print("\nDuplicated customer IDs:")
print(df["customerID"].duplicated().sum())

print("\nDuplicated rows:")
print(df.duplicated().sum())

  # we have found out there is no need to erase any duplicates or repeated registers from customer IDs or rows. So effectively, every observation = one unique customer

# 2.2. missing values
print("\nMissing values per column:")
print(df.isna().sum())

print("\nBlank values in TotalCharges:") # total charges aside since it should be numerical, we look for blank spaces, not NAs
print(df["TotalCharges"].str.strip().eq("").sum())

  # initially there seems to not be any missing values from the numerical variables, although we find 11 empty values from TotalCharges

# 2.3. check the target
print("\nChurn distribution:")
print(df["Churn"].value_counts())
  # we aim to check the % of clients that are churning (to check if there is a class imabalance, which is probable there is it)
  # effectively, tehre is a class imbalance. Churn customers = 26.54%. Non-churning customers = 73.46%. Hence, this demonstrates i) that we will have to employ a stratified split and ii) that accuracy as main metrics will not be enough
    # (just a remember) accuracy is not a good metric for imabalanced datasets since the model predicting that no customers churn would achieve a high accuracy while failing to identify a single customer who actually leaves, and in this model we are valuing both

print("\nChurn distribution (%):")
print(df["Churn"].value_counts(normalize=True) * 100)

# 2.4. managing column TotalCharges
# My first hypothesis with this column is that possibly those clients that have missing values in this column also have tenure = 0, meaning they've just joined the company, for that reason we are going to do a check
  # and exactly results shows that we are correct. Because this information also being relevant, we will be assigning them = 0 rather than filling with the mean or erasing these observations

# Identify customers with blank TotalCharges
missing_total = df[df["TotalCharges"].str.strip() == ""]

# Inspect relevant information
print("\nCustomers with missing TotalCharges:")
print(
    missing_total[
        ["customerID", "tenure", "MonthlyCharges",
         "TotalCharges", "Contract", "Churn"]
    ].to_string(index=False)
)

# Check tenure distribution for these customers
print("\nTenure of customers with missing TotalCharges:")
print(missing_total["tenure"].value_counts())


#---------------------------------------------------------------------------------------
# 3. DATA CLEANING

# 3.1. TotalCharges

# Convert TotalCharges from string to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Replace missing TotalCharges with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Verify the conversion
print("\nTotalCharges data type:")
print(df["TotalCharges"].dtype)

print("\nMissing TotalCharges after cleaning:")
print(df["TotalCharges"].isna().sum())

print("\nTotalCharges summary:")
print(df["TotalCharges"].describe())

#3.2. Churn
# now we will have to turn churn into a binary variable (although it is currently "binary" because of Yes and Nos, our model will have to work with numbers)

# Convert Churn into a binary target variable
df["Churn"] = df["Churn"].map({ # to replace the values of a column with the ones we are indicating. 1 = positive, 0 = negative of churn
    "No": 0,
    "Yes": 1
})

# Verify the conversion
print("\nChurn distribution after conversion:")
print(df["Churn"].value_counts())

print("\nChurn data type:")
print(df["Churn"].dtype)

#---------------------------------------------------------------------------------------
# 4. DATA NUMERICAL QUALITY CHECK

# Identify categorical columns
categorical_cols = df.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_cols)

# Display unique values for each categorical variable
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

#---------------------------------------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS

# before creating any new variables, we want to see how churn changes with the different client segments 
  # we use group by in order to group clients according different variables (ex. contract)

# Calculate churn rate by contract type
print("\nChurn rate by contract type (%):")
print(
    df.groupby("Contract")["Churn"].mean() * 100
)

# Calculate churn rate by internet service
print("\nChurn rate by internet service (%):")
print(
    df.groupby("InternetService")["Churn"].mean() * 100
)

# Calculate churn rate by payment method
print("\nChurn rate by payment method (%):")
print(
    df.groupby("PaymentMethod")["Churn"].mean() * 100
)

# Compare average tenure and monthly charges
print("\nAverage tenure and monthly charges by churn:")
print(
    df.groupby("Churn")[["tenure", "MonthlyCharges"]].mean()
)

# 5.1. Analysis interpretation of results
  # contract type = we can see a strong relationship with churn. For instance, customers with monthly contracts show a churn rate of 42.7% in comparison to those of contracts of 2 years which show a 2.8% churn rate. 
  # internet service and payment method = also show huge differneces between categories. For instance, those clients choosing fiber represent 41.9% of churn in comparison to 19.0% of other services such as DSL. 
    # same happens to clients paying through electronic check, which represent almost half of the churn (45.3%) in comparison to others who choose to make automatic payments (15-17% churn)
  # All these patterns could be related to prices, antiguity, or other characteristics, so we should therefore not treat them as causal relationships, these are just some insights that gives us an idea of possible correlations. 
  # tenure and monthy charges = we also see how the clients that churn have (on average) less than half of the tenure than those who don't as well as a monthly fee of +13 monetary units. 
    #This analysis leads us to the hypothesis that the combination of less tenure and high fees lead to more probability of churning. 

#---------------------------------------------------------------------------------------
# 6. TRAIN/TEST/VAL SPLIT

from sklearn.model_selection import train_test_split

# Separate predictors and target
X = df.drop(columns=["Churn", "customerID"]) # X = all variables we will be using to predict. Customer ID is an identificator so we also exclude it
y = df["Churn"] # y = the target variable

# First split: 80% training + validation, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y # since we do not have any chrnonological variable, we will be employing a stratified split in order to try to maintain the same proportion of churn in all of the 3 sets
)

# Second split: 60% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)

print("\nDataset split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nChurn rate by split:")
print("Training:", y_train.mean())
print("Validation:", y_val.mean())
print("Test:", y_test.mean())

#---------------------------------------------------------------------------------------
# 7. FEATURE ENGINEERING

# for the feature engineering, I will create only the essential set of variables that I believe that will both: i) possibly satisfy my hypothesis based on what we have seen in the data exploration and ii) that lets the model truly capture any difficult patterns only using the original ones
def engineer_features(X):
    X = X.copy()

#7.1. Number of extra services: counts the number of additional contracted services. From what we have seen, more services could be related with a higher vinculation to the client 
    extra_services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    X["NumberOfExtraServices"] = (
        X[extra_services] == "Yes"
    ).sum(axis=1)

# 7.2. Automatic payment = indicates if the payment is automatic or not. We have seen that clients with automatic payments tend to demonstrate a lower churn
    X["AutomaticPayment"] = X["PaymentMethod"].isin([
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]).astype(int)

# 7.3. Tenure Monthly Charge Interaction: tenure x monthly fee. Since we have seen that the relationship between price and churn could change depending on the tenure. 
    X["TenureMonthlyChargeInteraction"] = (
        X["tenure"] * X["MonthlyCharges"]
    )

    return X

# 7.4. apply the feature engineering to each dataset
X_train = engineer_features(X_train)
X_val = engineer_features(X_val)
X_test = engineer_features(X_test)

# Verify the new columns
print("\nNew engineered features:")
print(
    X_train[[
        "NumberOfExtraServices",
        "AutomaticPayment",
        "TenureMonthlyChargeInteraction"
    ]].head()
)

print("\nDataset shapes after feature engineering:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


#---------------------------------------------------------------------------------------
# 8. DATA PROCESSING

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

print("\nNumerical columns:")
print(numerical_cols)

print("\nCategorical columns:")
print(categorical_cols)



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Define preprocessing transformations
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), categorical_cols)
    ]
)


# Fit preprocessing ONLY on training data
X_train_processed = preprocessor.fit_transform(X_train) # we use fit instead of preprocessor since fit learns information about the data (mean or std) while transform just applies the rules we have learned without calculating them again

# Apply the same transformations to validation and test
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# Verify transformed data shapes
print("\nProcessed dataset shapes:")
print("Training:", X_train_processed.shape)
print("Validation:", X_val_processed.shape)
print("Test:", X_test_processed.shape)

# Verify that all resulting values are numeric
print("\nProcessed data type:")
print(X_train_processed.dtype)

# Verify there are no missing or infinite values
print("\nAll training values finite:")
print(np.isfinite(X_train_processed).all())


#---------------------------------------------------------------------------------------
# 9. NAIVE BASELINE MODEL

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create a classifier that always predicts the majority class
baseline = DummyClassifier(strategy="most_frequent") 

# Train baseline on training data
baseline.fit(X_train_processed, y_train)

# Make predictions on validation data
y_val_baseline = baseline.predict(X_val_processed)

# Evaluate baseline
print("\nNaive Baseline - Validation Metrics:")
print("Accuracy:", accuracy_score(y_val, y_val_baseline))
print("Precision:", precision_score(y_val, y_val_baseline, zero_division=0))
print("Recall:", recall_score(y_val, y_val_baseline, zero_division=0))
print("F1 Score:", f1_score(y_val, y_val_baseline, zero_division=0))

# since this is a extremely simple model that we will be employing as reference to evaluate whether our logistic regression truly dds any value, since our dataset is 73.46% imabalanced in favor of clients that do not churn, 
  # our baseline will have to always predict that the client does not churn independently of th characteristics, hence, that it is why we were expecting the same value of accuracy as of imbalanced and a recall of 0 for churn. 

#---------------------------------------------------------------------------------------
# 10. LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression

# Initialize logistic regression model
sklearn_model = LogisticRegression(
    C=1e6, # C is the inverse of regulation, so a high C value makes the regulation to be close to 0
    max_iter=5000, # in order to let the optimizator do more iterations to converge
    random_state=42
)

# Train the model
sklearn_model.fit(X_train_processed, y_train)

# Predict probabilities on validation data
y_val_proba_sklearn = sklearn_model.predict_proba(
    X_val_processed
)[:, 1]

# Convert probabilities into binary predictions
y_val_pred_sklearn = (
    y_val_proba_sklearn >= 0.5
).astype(int)

# Evaluate model
print("\nScikit-learn Logistic Regression - Validation Metrics:")
print("Accuracy:", accuracy_score(y_val, y_val_pred_sklearn))
print("Precision:", precision_score(y_val, y_val_pred_sklearn))
print("Recall:", recall_score(y_val, y_val_pred_sklearn))
print("F1 Score:", f1_score(y_val, y_val_pred_sklearn))

# 10.1. First iteration: I will do a second iteration motivated for the fact that recall score is relatively low, and mostly if we are interested in retaining our customers. But for now the threshold is 0.5. 
  # Accuracy: 0.8034066713981547 -> classifies 8out of 10 customers correctly (6.9 porcentual points increase in respect to the baseline)
  # Precision: 0.6600660066006601 -> out of 100 clients that we identify as potential churners, approximately 66 truly churn
  # Recall: 0.5347593582887701 -> out of 100 clients that truly churn, we only detect 53. This is not a good metric because in the end we are leaving without identifying the 46% of customers who churn
  # F1 Score: 0.590841949778434 

# 10.2. Second Iteration: overall we have improved little difference but it has been a change 
  # Accuracy: 0.8048261178140526
  # Precision: 0.6622950819672131
  # Recall: 0.5401069518716578
  # F1 Score: 0.5949926362297496

#---------------------------------------------------------------------------------------
# 11. PYTORCH DATA PREPARATION

import torch

# Convert numpy arrays into PyTorch tensors
X_train_t = torch.tensor(
    X_train_processed, dtype=torch.float32
)

y_train_t = torch.tensor(
    y_train.to_numpy(), dtype=torch.float32
)

X_val_t = torch.tensor(
    X_val_processed, dtype=torch.float32
)

y_val_t = torch.tensor(
    y_val.to_numpy(), dtype=torch.float32
)

print("\nPyTorch tensor shapes:")
print("X_train:", X_train_t.shape)
print("y_train:", y_train_t.shape)
print("X_val:", X_val_t.shape)
print("y_val:", y_val_t.shape)

# with these last lines of code we have maken sure that the tensors have the right dimensions

#---------------------------------------------------------------------------------------
# 12. MANUAL PYTORCH LOGISTIC REGRESSION

# What we are going to do now is to build the logistic regression manually without ysing any nn.Linear nor torch.optim.
# The idea is to understand how the model learns instead of letting a library do all the training and work automatically
# The logistic regression works by multiplying a variable by its weight and summing up the bias. Later applies a sigmoid function that converts this result in a probability (between 0 and 1)
    # Once this is done, it compares the prediction with the correct answer and adjusts the coefficients in order to reduce the error

import torch.nn.functional as F

# Set seed for reproducibility
torch.manual_seed(42)

# Number of input features
n_features = X_train_t.shape[1] # this number is 48

# Initialize weights and bias
w = torch.zeros(n_features, requires_grad=True) # creates an initial coefficient of 0 for each variable
b = torch.zeros(1, requires_grad=True) # indicates pytorch to calculate how the error changes in comparison to these parameters

print("\nNumber of features:", n_features)
print("Initial weights shape:", w.shape)
print("Initial bias:", b.item())


# Training hyperparameters (this values are as at starting point 0.05 and 3000 respectively, but we will try iteratively to see if its enough)
learning_rate = 0.05 # how much we change the coefficients in each update (too high = unstable, too low = takes time to converge)
epochs = 10000 # how many times we will update the parameters using the whole set of training


# Manual training loop
for epoch in range(epochs):

    # 1. Calculate logits
    logits = X_train_t @ w + b # the mathematical operation we have explained before

    # 2. Calculate binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits( # we use cross entropy because we are solving a problem of binary classification so this function icorporates the sigmoid function 
        logits, y_train_t 
    )

    # 3. Compute gradients
    loss.backward()

    # 4. Update parameters manually -> we update the parameters through a gradient descent that it is basically = new weight = past weight - the learning rate in order to try to reduce the loss (we do the same thing with the intercept)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 5. Reset gradients -> in order to prevent the accumulation of iterations
    w.grad.zero_()
    b.grad.zero_()

    # Print training progress
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# output: 
    # Initial bias: 0.0
    # Epoch 0: Loss = 0.6931 -> since all the coefficients start with 0, the model initially asigns a probability of 50% to all customers
    # Epoch 500: Loss = 0.4140
    # Epoch 1000: Loss = 0.4116
    # Epoch 1500: Loss = 0.4105
    # Epoch 2000: Loss = 0.4099
    # Epoch 2500: Loss = 0.4094 -> model is learning so coefficients get the error progressively smaller. the loss starts to stabilize, so the training is progressing towards a minimum but has not converged yet


#---------------------------------------------------------------------------------------
# 13. MANUAL PYTORCH EVALUATION 

# Disable gradient tracking during evaluation
with torch.no_grad(): # we are not going to calculate gradients because we are not going to modify the parameters, we just want to obtain predictions

    # Calculate logits on validation data
    val_logits = X_val_t @ w + b

    # Convert logits into probabilities
    val_probabilities = torch.sigmoid(val_logits) # since we used the binary cross entropy
        # logit = 0 = probability of 0.5; logit = + = probability superior to 0,5; logit = - = probability lower than 0.5

    # Convert probabilities into binary predictions
    val_predictions = (val_probabilities >= 0.5).int()

# Convert tensors to numpy arrays
y_val_pred_manual = val_predictions.numpy()
y_val_proba_manual = val_probabilities.numpy()

# Evaluate the model
print("\nManual PyTorch Logistic Regression - Validation Metrics:")

print("Accuracy:", accuracy_score(y_val, y_val_pred_manual))
print("Precision:", precision_score(y_val, y_val_pred_manual, zero_division=0))
print("Recall:", recall_score(y_val, y_val_pred_manual, zero_division=0))
print("F1 Score:", f1_score(y_val, y_val_pred_manual, zero_division=0))

# we also need to compare the probabilities generated by both models, because even though both models have similar metrics, they could produce different probabilities

# Compare predicted probabilities
probability_difference = np.abs(
    y_val_proba_sklearn - y_val_proba_manual
)

print("\nScikit-learn vs Manual PyTorch:")
print("Mean absolute probability difference:",
      probability_difference.mean())

print("Maximum probability difference:",
      probability_difference.max())

# the idea is that if both methods converge towards an equivalent solution, then their probabilities should be very similar
    # hence, the comparison will enable us to see if we need to adjust the learning rate or train with more epochs

# result: rhe results are pretty similar. from the results we can conclude that accuracy is almost identical, pytorch manual achieves a slightly superior precision, and scikit learn has a better recall and F1 score. 
    # Mean absolute probability difference: 0.014610990879903415 -> this means that on average, the probabilities differ in approximately 1.46 porcentual points
    # Maximum probability difference: 0.08425137574452801 -> this means that for some client, the difference reaches 8.43 porcentual points


#---------------------------------------------------------------------------------------
# 14. STANDARD PYTORCH LR

import torch.nn as nn

# Define logistic regression model
class LogisticRegressionTorch(nn.Module): # in here we are defying our class. And because the class ingerits from nn.Module, pytorch is capable of recognising and managing its parameters

    def __init__(self, input_features):
        super().__init__()

        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)

# mathematically it continues to do x = X·w + b, so we have not introduced any complex neuronal red

# Initialize model
torch.manual_seed(42)

standard_model = LogisticRegressionTorch(
    input_features=X_train_t.shape[1]
)

print("\nStandard PyTorch Model:")
print(standard_model)

# Define loss function
criterion = nn.BCEWithLogitsLoss() # is equivalemmt to the loss function we use in our manual pytorch (combines sigmoid and binary cross entropy)

# Define optimizer
optimizer = torch.optim.SGD( # stochastic gradient descent = how we will use all the training set in each update, we will be doing full batch gradient descent, same as in the manual version
    standard_model.parameters(),
    lr=0.05 # we use same learning rate and epochs in order to facilitate the comparison
)

# Number of epochs
epochs_standard = 10000

#---------------------------------------------------------------------------------------
# 14. STANDARD PYTORCH TRAINING LOOP

for epoch in range(epochs_standard):

    # Reset gradients
    optimizer.zero_grad()

    # Forward pass
    logits = standard_model(X_train_t)

    # Calculate loss
    loss_standard = criterion(logits, y_train_t)

    # Backward pass
    loss_standard.backward()

    # Update parameters
    optimizer.step() # before we had to update each parameter separetely, now we substitute it with this

    # Print progress
    if epoch % 2000 == 0:
        print(
            f"Standard PyTorch - Epoch {epoch}: "
            f"Loss = {loss_standard.item():.4f}"
        )

#---------------------------------------------------------------------------------------
# 15. STANDARD PYTORCH EVALUATION

standard_model.eval() #.eval() indicates pytorch that we are using the model for evaluation not for training. in our model it does not change the calculations since we only have one manual layer, but it is a good practice because layers such as dropout or batchmorm do behave differently during evaluation

with torch.no_grad():

    # Calculate validation logits
    val_logits_standard = standard_model(X_val_t)

    # Convert logits to probabilities
    val_probabilities_standard = torch.sigmoid(
        val_logits_standard
    )

    # Apply classification threshold
    val_predictions_standard = (
        val_probabilities_standard >= 0.5
    ).int()

# Convert tensors to numpy arrays
y_val_proba_standard = val_probabilities_standard.numpy()
y_val_pred_standard = val_predictions_standard.numpy()

# Evaluate model
print("\nStandard PyTorch Logistic Regression - Validation Metrics:")

print(
    "Accuracy:",
    accuracy_score(y_val, y_val_pred_standard)
)

print(
    "Precision:",
    precision_score(y_val, y_val_pred_standard, zero_division=0)
)

print(
    "Recall:",
    recall_score(y_val, y_val_pred_standard, zero_division=0)
)

print(
    "F1 Score:",
    f1_score(y_val, y_val_pred_standard, zero_division=0)
)

# with the results we can conclude the following: 
    # first, that the 3 implementations have very similar results, since in the end we are training the same logistic regression with 3 different processes
    # secondly, that scikit learn presents a slightly higher recall, while the other pytorch models have a slightly higher precision 
    # and finally, that both pytorch models obtain a very similar metrics, but this doesn't mean that their probabilities are identical. Both models can classify directly the same clients with threshold 0.5 but assign them different probabilities
    # and also that even though the lossess continue to decrease, we cannot guarantee that all parameters have fully converged yet

#---------------------------------------------------------------------------------------
# 16. PROBABILITY COMPARISON OF THE 3 MODELS

# Compare Manual PyTorch vs Standard PyTorch
manual_vs_standard = np.abs(
    y_val_proba_manual - y_val_proba_standard
)

# Compare Scikit-learn vs Standard PyTorch
sklearn_vs_standard = np.abs(
    y_val_proba_sklearn - y_val_proba_standard
)

print("\nProbability Comparison:")

print("\nManual vs Standard PyTorch:")
print("Mean absolute difference:", manual_vs_standard.mean())
print("Maximum difference:", manual_vs_standard.max())

print("\nScikit-learn vs Standard PyTorch:")
print("Mean absolute difference:", sklearn_vs_standard.mean())
print("Maximum difference:", sklearn_vs_standard.max())

# by comparing the probabilities, it helps us produce if we are reproducing approximately the same mathematical model 
# we can see that indeed the 3 models do actually produce the same probabilities
# manual and standard pytorch have a small difference, so it means that for a random customer, the probabilities of churn estimated by both implementations differ by roughly 0.061 percentual points on average
    # this has sense because they both use the same logistic regression, BCE, gradient descent, lr = 0.05 and 3,000 epochs, so the small difference might be because of the initialization of both coefficients
# scikit learn vs pytorch as a difference of 1.48 porcentual points, while the maximum reaches 8.49, so it is not a huge difference but sufficiently relevant for us to investigate it

#---------------------------------------------------------------------------------------
# 17. VALIDATION LOSS COMPARISON

from sklearn.metrics import log_loss

# Calculate validation loss for each model
loss_sklearn = log_loss(y_val, y_val_proba_sklearn)
loss_manual = log_loss(y_val, y_val_proba_manual)
loss_standard = log_loss(y_val, y_val_proba_standard) # we use log loss because it evaluates directly the preducted probabilities, penalizing exactly those predictions that are confident but wrong, hence it will help us see if the scikit learn probabilities are better even if their classification metrics are similar

print("\nValidation Log Loss Comparison:")
print("Scikit-learn:", loss_sklearn)
print("Manual PyTorch:", loss_manual)
print("Standard PyTorch:", loss_standard)

# we can see with these results that the difference between the best and worse model is only 0.00036, so this indicates that the 3 of them have a similar performance also when we evaluate directly the probabilities and not only the binary classifications
# but still this doesn't prove that the models have converged yet at exactly the same best. to demonstrate this, we would have to compare the training losses and check if these stabilize when increasing the epochs. 
    # but for this assignment, this is just enough evidence that the 3 implementations behave in a similar way

#---------------------------------------------------------------------------------------
# 18. BUSINESS OPTIMAL THRESHOLD

# As it is written in the report, we must be aware that the error in FP and FN is not the same. For that reason, we are going to state a hypothetical cost. 
    # FP = means when we contact or incentivize a client that it is not going to churn = the hypothetical cost of a FP is 10€ since it could be interpreted as an averge increasing cost of contacting, commercial administration and possible unnecessary incentives
    # FN = means when we do not identify a client that ends up churning = the hypothetical cost will be 200€, this is justified because it represents a simplified approximiation to the economic impact of loosing a client, taking into account the potential future revenue and the lost margin
# but still, because we lack of actual data, this numbers are just merely illustrative. Additionally, we also assume that the TP do not generate any additional costs. Even though this is not completely realistic for a retention campaign, it allows us to start with a simple model

# so our objective will be minimizing the total business cost = (100· FP) + (200 · FN)

from sklearn.metrics import confusion_matrix

# Illustrative business costs in euros
COST_FP = 10
COST_FN = 200

def calculate_business_cost(y_true, y_proba, threshold):

    # Convert probabilities into binary predictions
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    # Calculate total business cost
    total_cost = (COST_FP * fp) + (COST_FN * fn)

    return total_cost, fp, fn, tp, tn


# Test different classification thresholds
thresholds = np.arange(0.05, 0.96, 0.01) # this creates a sequence of thresholds

results = []

for threshold in thresholds:

    total_cost, fp, fn, tp, tn = calculate_business_cost(
        y_val,
        y_val_proba_sklearn,
        threshold
    )

    results.append({
        "Threshold": threshold,
        "TotalCost": total_cost,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "TN": tn
    })

# Convert results into a DataFrame
threshold_results = pd.DataFrame(results)

# Find threshold with lowest estimated cost
best_result = threshold_results.loc[
    threshold_results["TotalCost"].idxmin() # idxmin identifies the index of the threshold that generates the lowest estimated cost. if differentthresholds obtain the similar cost, it will select the first one it finds 
]

print("\nOptimal Business Threshold:")
print(best_result)

print("\nBusiness Cost at Threshold 0.5:")
print(calculate_business_cost(
    y_val, y_val_proba_sklearn, 0.5
)[0])

# what we expect to see is that since we have assign a much more high cost to FN, it is most probable that the optimal threshold is lower than 0.5
# this has a lot of sense because the company would rather identify more clients in risk even if that could imply inneccessary contacting with some customers

# as a result, the optimal threshold that minimizes our cost is 0.06 (6% of churn) and it shows us a 75.6% of decrease in our cost in comparison to a threshold of 0.5
    # this also shows that we detect 362 out of 374 clients that churn, so we just let go 12 clients 
    # we innecessary contact 625 clients
    # and we identify as clients in risk to 987 out of the 1,409 customers, aproximately the 70%
# from the recall perspective, the result is excellent but from the commercial perspective, we are destinating retention actions to a huge part of our business client base

#---------------------------------------------------------------------------------------
# 19. BUSINESS COST SENSITIVITY ANALYSIS

# we afre doing this to compare differemt scenarios of costs to see how the buisness decision changes, so instead of just using a unique function cost, we evaluate 3 scenarios
# for each scenario we: i) try different thresholds, ii) calculate how many FP and FN it generates, iii) multiply each error by its corresponding error and iv) selects the threshold that minimizes the cost inside that scenario. 
# in that way we can study if the decision of contacting many clients depends in our assumptions

# Different hypothetical business scenarios
cost_scenarios = {
    "Retention priority": (10, 200),
    "Balanced scenario": (20, 100),
    "Budget constrained": (30, 75)
}

sensitivity_results = []

for scenario, (cost_fp, cost_fn) in cost_scenarios.items():

    for threshold in thresholds:

        # Generate predictions
        y_pred = (
            y_val_proba_sklearn >= threshold
        ).astype(int)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            y_val, y_pred, labels=[0, 1]
        ).ravel()

        # Calculate scenario-specific cost
        total_cost = cost_fp * fp + cost_fn * fn

        sensitivity_results.append({
            "Scenario": scenario,
            "Threshold": threshold,
            "Cost": total_cost,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "TN": tn
        })

sensitivity_df = pd.DataFrame(sensitivity_results)

# Select lowest-cost threshold per scenario
best_scenarios = sensitivity_df.loc[
    sensitivity_df.groupby("Scenario")["Cost"].idxmin()
]

print("\nBusiness Cost Sensitivity Analysis:")
print(best_scenarios.to_string(index=False))

# the results show us that retention priority (with threshold 0.06) identifies almost all clients that churn but select up to 70% of the sample for our campaigns, which results a very intensive strategy in contacts. 
# the balanced scenario has a threshold of 0.17 = it identifies 88% of the clients that churn and selects approcimately 53% of the sample. reduces contacts but lets more clients escape
# the budget contrained with threshold of 0.33 = selects approximately 35% of the clients and reaches a higher precision, and would be more compatible with a limited capacity campaign

# so for our scenario, I would use the balanced one as illustrative scenario, since it offers a balanced compromise between detecting clients that churn and limiting the unnecessary contacts. 


#---------------------------------------------------------------------------------------
# 20. RETENTION CAMPAIGN

from sklearn.metrics import confusion_matrix

# Hypothetical business assumptions (EUR)
COST_CONTACT = 10
COST_MISSED_CHURN = 100

def calculate_business_cost(y_true, y_proba, threshold):

    # Convert probabilities into binary predictions
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    # Cost of all contacted customers (TP + FP)
    contact_cost = COST_CONTACT * (tp + fp)

    # Cost of missed churners
    missed_churn_cost = COST_MISSED_CHURN * fn

    # Total estimated business cost
    total_cost = contact_cost + missed_churn_cost

    return total_cost, fp, fn, tp, tn


# Evaluate thresholds on validation data
thresholds = np.arange(0.01, 1.00, 0.01)

results = []

for threshold in thresholds:

    total_cost, fp, fn, tp, tn = calculate_business_cost(
        y_val,
        y_val_proba_sklearn,
        threshold
    )

    results.append({
        "Threshold": threshold,
        "TotalCost": total_cost,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "TN": tn,
        "Contacted": tp + fp
    })

threshold_results = pd.DataFrame(results)

# Select the minimum-cost threshold
best_result = threshold_results.loc[
    threshold_results["TotalCost"].idxmin()
]

print("\nNew Optimal Business Threshold:")
print(best_result)

print("\nBusiness Cost at Threshold 0.5:")
print(
    calculate_business_cost(
        y_val, y_val_proba_sklearn, 0.5
    )[0]
)

# before we were calculating (10·FP) + (200·FN) and now we are calculating 10 · (TP + FP) + (100·FN)
# we assume that contacting any cstomer (TP or FP) has a cost while failing to identify a customer who churns (FN) leads to a higher potential loss
# we change the formula so now we also count with the cost of contacting clients that were going to churn (TP)


# however, the results show that the optimal threshold continues to be 0.06 in spite of modifying the formula with a minimum cost of 11.070€ and a cost reduction of 45.3% in comparison to the 0.5 threshold.
# the threshold continues to be that low because the cost of not intervening is still 10 times superior, that it is why the model favours an aggressive retention strategy
# the only problem is that with this threshold we would still directing the campaign to directly 70% of our customer base (like 7 out of 10 customers),w hich can make difficult for us to sustain it if the commercial budget is limited


#---------------------------------------------------------------------------------------
# 21. COST SENSITIVITY ANALYSIS

# now we just want to compare whether the decision changes if our assumption on costs do too. we will maintain the cost of loosing a client and we change the cost of intervention. 
# Compare different hypothetical contact costs while keeping
# the estimated cost of a missed churner constant.

cost_scenarios = {
    "Retention priority": (10, 100),
    "Balanced": (20, 100),
    "Budget conscious": (30, 100)
}

sensitivity_results = []

for scenario, (contact_cost, missed_churn_cost) in cost_scenarios.items():

    for threshold in thresholds:

        y_pred = (y_val_proba_sklearn >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val, y_pred, labels=[0, 1]
        ).ravel()

        total_cost = (
            contact_cost * (tp + fp)
            + missed_churn_cost * fn
        )

        sensitivity_results.append({
            "Scenario": scenario,
            "Threshold": round(threshold, 2),
            "TotalCost": total_cost,
            "Contacted": tp + fp,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "TN": tn
        })

sensitivity_df = pd.DataFrame(sensitivity_results)

best_scenarios = sensitivity_df.loc[
    sensitivity_df.groupby("Scenario")["TotalCost"].idxmin()
]

print("\nUpdated Business Cost Sensitivity Analysis:")
print(best_scenarios.to_string(index=False))

# with this we want to see if the cost of contacting clients makes the optimal threshold to increase or not. in that case, we would recommend the company to do more selective campaigns, sacrifying part of the recall in exchange of reducing the number of contacts
# also it is helpful because these numbers would demonstrate that the retention policy depends as much on the estimated probabilities than on the economic assumptions
    # with our actual function we have --> p(churn) > Cost of contact / cost of missed churn

# indeed, from the results we can see that the retention priority strategy prioritizes detecting almost all the chyrners, but needs to contact 70% of the clients, balanced reduces contacts to 45% but fails to identify 83% of the customers who churn, and finally the budget conscious strategy contacts 34% of the clients with a higher precision but lets churners escape
# with these results I would maintain the retention priroity as principal scenario  


#---------------------------------------------------------------------------------------
# 21. TRAINING VS VALIDATION LOG LOSS

# Scikit-learn training probabilities
train_proba_sklearn = sklearn_model.predict_proba(
    X_train_processed
)[:, 1]

# PyTorch training probabilities
with torch.no_grad():
    train_proba_manual = torch.sigmoid(
        X_train_t @ w + b
    ).numpy()

    train_proba_standard = torch.sigmoid(
        standard_model(X_train_t)
    ).numpy()

# Compare training and validation losses
print("\nTraining vs Validation Log Loss:")

for name, train_proba, val_proba in [
    ("Scikit-learn", train_proba_sklearn, y_val_proba_sklearn),
    ("Manual PyTorch", train_proba_manual, y_val_proba_manual),
    ("Standard PyTorch", train_proba_standard, y_val_proba_standard)
]:
    print(
        f"{name}: "
        f"Train = {log_loss(y_train, train_proba):.6f}, "
        f"Validation = {log_loss(y_val, val_proba):.6f}"
    )

# from the first results we have obtained, we can see that the pytorch models still have margin to improve and reach convergence 
    # scikit learn achieves a lower training loss in comparison to pytorch and suggests that models trained with gradient descent can continue accelerating to the minimum loss function
    # also the losses of the validation are very similar
# hence, I will do one last proof of convergence by changing the number of interactions in pytorch to make the solutions become more near scikit-learn (from 3,000 to 10,000)

# by doing this change, result shows that the models continue learning (the training loss decreases and approximates more to scikit-learn), the generalization capacity has not improve (the validation loss has slightly increase by very little) and both implementations ressemble even more 
# in spite of this, I would leave it in 10,000 epochs since the main objective of the assignment is to show that we are correctly impelementing the same logistic regression 


#---------------------------------------------------------------------------------------
# 22. FINAL TEST EVALUATION

# this code first calculates the probability of churn of all clients using each implementation, then compares the models with the standard threshold of 0.5 which helps us evaluate the predictive performance under the same conditions, and later applies our business threshold of 0.06 in order to calculate the estimated cost and the quantity of customers we will be contacting

# Fixed decision thresholds
DEFAULT_THRESHOLD = 0.5
BUSINESS_THRESHOLD = 0.06

# Scikit-learn test probabilities
y_test_proba_sklearn = sklearn_model.predict_proba(
    X_test_processed
)[:, 1]

# PyTorch test data
X_test_t = torch.tensor(
    X_test_processed, dtype=torch.float32
)

# PyTorch test probabilities
standard_model.eval()

with torch.no_grad():
    y_test_proba_manual = torch.sigmoid(
        X_test_t @ w + b
    ).numpy()

    y_test_proba_standard = torch.sigmoid(
        standard_model(X_test_t)
    ).numpy()

# Store predictions for all three models
test_probabilities = {
    "Scikit-learn": y_test_proba_sklearn,
    "Manual PyTorch": y_test_proba_manual,
    "Standard PyTorch": y_test_proba_standard
}

# Evaluate with the default threshold (0.5)
print("\nFINAL TEST METRICS - THRESHOLD 0.5")

for name, probabilities in test_probabilities.items():

    predictions = (
        probabilities >= DEFAULT_THRESHOLD
    ).astype(int)

    print(f"\n{name}:")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(
        y_test, predictions, zero_division=0
    ))
    print("Recall:", recall_score(
        y_test, predictions, zero_division=0
    ))
    print("F1:", f1_score(
        y_test, predictions, zero_division=0
    ))
    print("Log Loss:", log_loss(y_test, probabilities))

# Evaluate business threshold (fixed using validation)
print("\nFINAL TEST BUSINESS EVALUATION - THRESHOLD 0.06")

for name, probabilities in test_probabilities.items():

    cost, fp, fn, tp, tn = calculate_business_cost(
        y_test, probabilities, BUSINESS_THRESHOLD
    )

    print(f"\n{name}:")
    print("Estimated Cost:", cost)
    print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
    print("Customers Contacted:", tp + fp)

# from the results we can see that the 3 models detect more than 97% of the churners using the business threshold. 
# however, to achieve this number, we would need to contact approximately 70% of the clients (too high), showing that this is the trade off between minimizing FN and controling the business cost


#---------------------------------------------------------------------------------------
# 23. SAVE TRAINED MODELS AND ARTIFACTS

from pathlib import Path
import joblib

# Create artifacts directory if it does not exist -> artifact = a file that contains something generated during training and that we can use afterwards
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# Save scikit-learn logistic regression
joblib.dump(
    sklearn_model,
    ARTIFACT_DIR / "sklearn_model.joblib"
)

# Save fitted preprocessing pipeline
joblib.dump(
    preprocessor,
    ARTIFACT_DIR / "preprocessor.joblib"
)

print("\nScikit-learn model and preprocessor saved.")

# Save manual PyTorch parameters
torch.save(
    {
        "weights": w.detach(),
        "bias": b.detach()
    },
    ARTIFACT_DIR / "manual_pytorch.pt"
)

# Save standard PyTorch model
torch.save(
    standard_model.state_dict(),
    ARTIFACT_DIR / "standard_pytorch.pt"
)

print("PyTorch models saved.")

# Save test data after feature engineering
X_test.to_csv(
    ARTIFACT_DIR / "X_test.csv",
    index=False
)

# Save actual test labels
np.save(
    ARTIFACT_DIR / "y_test.npy",
    y_test.to_numpy()
)

# Save predicted probabilities
np.savez(
    ARTIFACT_DIR / "test_probabilities.npz",
    sklearn=y_test_proba_sklearn,
    manual_pytorch=y_test_proba_manual,
    standard_pytorch=y_test_proba_standard
)

# Save the business threshold
joblib.dump(
    {
        "threshold": BUSINESS_THRESHOLD,
        "contact_cost": COST_CONTACT,
        "missed_churn_cost": COST_MISSED_CHURN
    },
    ARTIFACT_DIR / "business_settings.joblib"
)

print("\nAll dashboard artifacts saved successfully.")


# to compare the naive baseline under the same test set 
y_test_baseline = baseline.predict(X_test_processed)

print("\nFINAL TEST BASELINE:")
print("Accuracy:", accuracy_score(y_test, y_test_baseline))
print("Precision:", precision_score(y_test, y_test_baseline, zero_division=0))
print("Recall:", recall_score(y_test, y_test_baseline, zero_division=0))
print("F1:", f1_score(y_test, y_test_baseline, zero_division=0))
