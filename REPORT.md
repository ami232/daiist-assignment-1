# Assignment 1 Report

- **Name**: Adolfo Gutierrez de Gandarilla Domínguez
- **Student ID**: 19003
- **Email**: fgutierrezde.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

I used the **IBM Telco Customer Churn** dataset. I downloaded it from IBM's
public GitHub ([link](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv))
and saved it in the repo as `data/telco_churn.csv` (948 KB).

- Each row is **one customer** of a phone and internet company.
- It has **7,043 customers and 21 columns**: some personal info (gender,
  senior citizen, partner, dependents), account info (how many months they
  have been a customer, type of contract, payment method, monthly and total
  charges), which services they have (phone, internet, online security,
  streaming…), and the target column `Churn` (Yes/No: did they leave).
- **1,869 customers left (26.5%)** and 5,174 stayed (73.5%).

I chose it because churn is a real problem that companies deal with every
month and the data needs some real cleaning and thinking before it can be used.

## Business / real-life framing

**The situation.** Every month, the company's retention team chooses which
customers to send a **retention offer** to (for example a discount worth
about €150 over a year). My model gives each customer a probability of
leaving, and the team contacts everyone above a certain cut-off (the
threshold). So the question the model answers is: *who should we spend the
retention budget on?*

**What each decision costs (my assumptions):**

| What happens | Cost |
|---|---|
| We don't contact them and they stay | €0 |
| We contact them but they would have stayed anyway (wasted offer) | €150 |
| We don't contact them and they leave (we lose a year of their bill) | 12 × their monthly bill |
| We contact them and they were going to leave (the offer works half the time) | €150 + 50% × 12 × their monthly bill |

I use **each customer's own bill** instead of one fixed amount, because
losing someone who pays €110 a month is much worse than losing someone who
pays €20. At first I used a fixed €900. The average customer who leaves
pays €74.4 a month, and 12 × €74.4 = €893, so €900 was a fair average to choose from, but
using each person's bill is more realistic.

**How this framing changed my pipeline:**

1. **Target:** `Churn = Yes` becomes 1 and `No` becomes 0. This is already
   exactly what I want to predict, so I didn't need to change it.
2. **How I split the data:** the dataset is one snapshot with **no dates**,
   so a random split can't leak "future" information into training. I used
   **60% train, 20% validation, 20% test**. Also I stratified the data per training, validation and test, 
   so that each part keeps the same 26.5% of churners. 
   If I had monthly data I would split by time instead (this is actually a limitation).
3. **How I chose the threshold:** not with accuracy. A model that says
   "nobody leaves" is already 73.5% accurate but saves no money. Instead I
   pick the threshold that gives the **lowest total cost on the validation
   set**, and I only use the test set once at the end.
   Doing the maths: contacting someone is worth it when the money the offer
   is expected to save is bigger than the offer itself:
   `p × 50% × 12 × bill > €150`, which simplifies to **p > 25 / bill**.
   For a typical €75/month customer that means **p > 0.33**, so I expected
   the best threshold to be below the usual 0.5. It also means that
   customers paying less than €25 a month (19.7% of all customers) are
   not worth a €150 offer, no matter how likely they are to leave.
4. **How I compare models:** ROC AUC (how well the model ranks customers
   from risky to safe) and log loss (the thing logistic regression actually
   minimises, so it's the fairest way to check the three versions match).

## Data preparation & feature engineering

All of this is in `train.py`, section 2.

**Cleaning**

- `TotalCharges` was stored as text and **11 rows were empty**. When I
  looked at them, all 11 had `tenure = 0`, meaning they were brand-new
  customers who hadn't been charged yet. So the right value is **0**, not a
  missing value to drop.
- I removed `customerID` because it's just a label and says nothing about
  whether someone leaves.

**Encoding**

- Service columns have three options: Yes / No / "No internet service". The
  third one just repeats what the `InternetService` column already says, so
  I turned each service into a simple **1 (has it) / 0 (doesn't)** column.
- `InternetService`, `Contract` and `PaymentMethod` are one-hot encoded,
  dropping one category each so there are no repeated columns.
- I **scaled** the numbers with `StandardScaler`, fitted **only on the
  training data** so no information from validation/test leaks in.

**New features I created**

| Feature | Why |
|---|---|
| `log_tenure` = log(1 + months as customer) | Churn drops very fast in the first months and then flattens: 53% leave in months 0–6, 36% in months 7–12, only 10% after 4 years. A straight line can't follow that curve, but the log of tenure can. |
| `no_support` | Customer has internet but neither online security nor tech support, so they have nobody to call when something breaks. |
| `bill_increase` = monthly bill − average past bill | Shows whether they pay more now than before. Price increases often make people leave. It also replaces `TotalCharges`, which is roughly tenure × monthly bill and so adds nothing new. |

**Features I removed (I found these problems when comparing the three models, see next section)**

- **`num_services`** (how many extra services the customer has). I added it
  at first, but it is exactly the sum of 7 columns that are already in the
  model, so it adds no new information.
- **`MonthlyCharges`**. When I predicted the monthly bill from the services,
  I got **R² = 0.9988**. The bill is basically a price list: phone ≈ €20,
  fiber ≈ +€25, each streaming service ≈ €10, each extra ≈ €5. Having both
  the bill and the services means the model has two ways of saying the same
  thing. I still use the bill inside `bill_increase` and for the business cost.

**A feature I tried and rejected**

Customers with a month-to-month contract **and** fiber internet leave 55%
of the time (and they are 30% of all customers), so I tried adding a flag
for that group. Validation log loss stayed exactly the same (0.3954). The
model already catches this group by adding up the two separate effects, so
I didn't keep it.

**Did all this help? (validation set, same model)**

| Version | Columns | Val log loss | Val ROC AUC |
|---|---|---|---|
| Naive: one-hot everything, raw numbers | 30 | 0.4013 | 0.860 |
| My features but normal `tenure` instead of log | 23 | 0.4014 | 0.859 |
| My features without `no_support` and `bill_increase` | 21 | 0.3961 | 0.863 |
| My features + month-to-month × fiber (not kept) | 24 | 0.3954 | 0.864 |
| **Final** | **23** | **0.3954** | **0.864** |

To be honest: **`log_tenure` is the change that really matters**. My two new
features only help a little. The biggest benefit of the cleaning isn't the
score, it's having a **smaller and clearer model** where the coefficients
are stable (the biggest one went from 1.58 to 0.78).

## Modeling: three implementations, one model

I used **logistic regression** because the target is Yes/No and the
business needs a **probability** to compare with the threshold.

I trained the same model three ways, on the same split (4,225 train / 1,409
validation / 1,409 test):

1. **scikit-learn**: `LogisticRegression`.
2. **Manual PyTorch**: I create the weights `w` and bias `b` as tensors,
   compute `sigmoid(Xw + b)`, write the loss formula myself, call
   `loss.backward()` to get the gradients, and update the weights by hand
   (`w -= lr * w.grad`), 3,000 times.
3. **Standard PyTorch**: a `nn.Module` with one `nn.Linear` layer,
   `BCEWithLogitsLoss` as the loss and `torch.optim.SGD` to update.

**Making sure they solve the same problem.** scikit-learn uses a
regularisation setting called `C`, while in PyTorch I add the penalty
myself with a value `λ`. They are written differently, so I converted one
into the other: **λ = 1 / (C × number of training rows)**. I don't apply
the penalty to the bias, same as scikit-learn. Without this, the three
versions would be minimising slightly different things.

**Tuning (kept small, using validation log loss):**
- `C`: I tried 0.01, 0.1, 1 and 10. C = 1 and C = 10 both gave 0.3954 (C = 10
  was picked); only 0.01 was clearly worse (0.4014). So regularisation
  barely matters here.
- Learning rate: I tried 0.01, 0.1 and 1. I picked **0.1** for both PyTorch
  versions. With 0.01 the model hadn't finished learning after 3,000 steps
  (0.4000 manual / 0.4005 standard, vs 0.3954 with 0.1).

**Results on the test set** (threshold 0.38, chosen on validation; cost in € for 1,409 customers):

| Model | ROC AUC | Log loss | Accuracy @0.5 | Business cost @0.38 |
|---|---|---|---|---|
| scikit-learn | 0.8402 | 0.4206 | 0.797 | €288,323 |
| Manual PyTorch | 0.8403 | 0.4207 | 0.798 | €288,323 |
| Standard PyTorch | 0.8402 | 0.4206 | 0.797 | €288,323 |
| Baseline (everyone gets 26.5%) | 0.5000 | 0.5787 | 0.735 | €337,971 (contacts nobody) |

**Do the three agree? Yes.** The biggest difference in any coefficient
compared to scikit-learn is **0.004**, and the biggest difference in any
predicted probability is **0.0013**. That tiny gap is because the PyTorch
versions stop after 3,000 steps, while scikit-learn keeps going until it
fully converges.

**At first they did NOT agree.** In my first version all three made the
same predictions (same AUC), but some coefficients were very different:
`MonthlyCharges` was −1.34 in scikit-learn but −0.32 in PyTorch. The reason
was the two repeated features (`num_services` and `MonthlyCharges`). When
two columns say the same thing, there are many different combinations of
weights that give exactly the same predictions. scikit-learn ends up at one
of them, and gradient descent moves towards it very slowly and stops
somewhere else. After removing those two features there was only one best
answer, and all three matched, without losing any accuracy.

**What this means for the business.** With threshold 0.38 the model costs
**€288,323**, compared with **€337,971** if we contact nobody. That's a
saving of **€49,648 (−14.7%)** on 1,409 customers. We contact 444 customers
and catch 248 of the 374 who were going to leave. Using the normal 0.5
threshold would cost €292,421, and contacting everyone would cost €380,336.
The best threshold (0.38) is close to what the maths predicted for a
typical customer (0.33), which suggests the probabilities are reasonable.

## Limitations & next steps

- **The costs are my own guesses** (€150 offer, 12 months of bill lost, offer
  works 50% of the time). The threshold and the € results depend on them.
  *Next step:* run a small test of the offer to get the real cost and
  success rate, put them at the top of `train.py` and run it again. The
  threshold updates automatically.
- **One threshold for everyone.** The cost uses each customer's bill, but
  the decision still uses one threshold for everybody (the dashboard needs
  a slider). Since the break-even is `p > 25 / bill`, each customer should
  really have their own cut-off. On the test set, contacting whoever has
  `p × 6 × bill > €150` would cost **€280,704**, another €7.6k less.
  *Next step:* rank customers by expected saving instead of just probability.
- **No dates.** I can't check if the model still works next month, and a
  random split might make results look better than they really are.
  *Next step:* with monthly data, train on past months and test on the
  following month.
- **Validation vs test gap.** Validation AUC is 0.864 but test AUC is 0.840.
  With only 1,409 customers in each part, some of this is just luck.
  *Next step:* use 5-fold cross-validation to choose C and the threshold.
- **Linear model limits.** It can only learn combinations of features if I
  build them by hand. I tried the most obvious one (month-to-month × fiber)
  and it didn't help. *Next step:* look at the customers the model gets
  most wrong to see what they have in common.
- **`bill_increase` is weak.** Its coefficient is almost 0 (−0.012), and
  removing it only changes validation log loss from 0.3954 to 0.3956. In a
  real version I would drop it to keep the model simpler.

## Generative AI use disclosure

I used **Claude (Anthropic), through Claude Code**, as an assistant during
this assignment. It helped me read and organise the assignment brief,
download the dataset, write the code in `train.py` and `app.py` (loading
the data, the training loops, saving the models and the Plotly/Gradio
dashboard), run and check the results. As well, it helped on the visualisation
of this REPORT.

I went through every decision: the business framing and cost assumptions,
the split and why the three models first disagreed. 
