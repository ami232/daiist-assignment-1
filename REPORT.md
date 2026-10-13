# Diabetes screening in New York: who should get a free test?

| | |
|---|---|
| **Name** | Lucía Muñoz Chicote |
| **Student ID** | 19445 |
| **Email** | lmunoz.ieu2022@student.ie.edu |
| **Group** | BBADBA 5B |
| **Course** | DAIIST, Assignment 1 |

**Short version:** New York can afford to send a free glucose test to only 1 in 10 people. I built a model that ranks people by their risk of diabetes and invites the top 10%. On the test set (3,448 people), it finds **3.4 times more diabetics** than inviting people at random, with the same number of tests. I built the same model three times (scikit-learn, PyTorch by hand, PyTorch the standard way) and all three give the same answer.

---

## 1. Dataset

**What it is.** A big US health phone survey called **BRFSS 2022**, run every year by the CDC. People answer questions about their health, habits and background (age, weight, smoking, sleep, past illnesses...). I used the cleaned version from Kaggle ("Indicators of Heart Disease (2022 UPDATE)", file `heart_2022_with_nans.csv`, 445,132 people from all US states, DC and territories).

**Why I picked it.**

- **It supports a real decision.** Diabetes screening is something health departments actually do, with a limited budget. That gives the model a clear job (who gets a test?) and a clear way to measure success (diabetics found, euros spent).
- **The target is useful and imbalanced.** About 1 in 8 people have diabetes, so plain accuracy is misleading and I have to think about the right metric and threshold, which is the interesting part.
- **It needs real data preparation.** It has missing values, ordered answers ("Poor" to "Excellent"), age in bands and impossible values (people sleeping 24 hours). That gives feature engineering something to do.
- **Some columns are traps.** Several columns are consequences of being diagnosed, not causes. Spotting and removing them is a good test of thinking about the problem, not just the numbers.
- **It's public and large.** Anyone can download it and rerun my work, and it covers the other states and territories, which I can use to test whether the model travels (see limitations).

**What I kept.** Only the people who live in **New York State**, because the story is a New York health department: 17,800 people and 39 columns. The script `data/make_ny_subset.py` shows exactly how this file was made. I did not touch the missing values in that step, so every cleaning decision lives in the main code, where you can see it.

**What I predict (the target).** The question "Has a doctor ever told you that you have diabetes?"

| Answer | What I did | People |
|---|---|---|
| Yes | **1** (has diabetes) | 2,183 |
| No | **0** (no diabetes) | 15,057 |
| "Pre-diabetes" | Removed: it is not diabetes, but it is not a clean "no" either | 402 |
| "Yes, but only during pregnancy" | Removed: that is a different illness that goes away after birth | 110 |
| No answer | Removed | 48 |

That leaves **17,240 people, and 12.7% of them have diabetes.** So roughly 1 in 8.

---

## 2. Business framing

**The story.** The New York State health department wants to find people with diabetes early, before it causes damage. It will mail a free glucose test to some people. It only has money for tests for **10% of the population**.

**The question the model answers.** "Who are the 10% most likely to have diabetes?" The model gives everyone a risk score, I sort the list, and the top 10% get invited.

**The costs I assumed:**

- Sending a test to someone **without** diabetes (a wasted test): **€30**
- **Missing** someone who has diabetes (they get complications later): **€1,000**

**Why I don't use accuracy.** A "model" that says *nobody has diabetes* is already right 87% of the time, and it is useless. What matters is:

- **Precision:** of the people I invite, how many really have diabetes?
- **Recall:** of all the people with diabetes, how many did I find?
- **Cost in euros:** the number the health department actually cares about.

### Who gets invited (the threshold)

- The threshold is set by the **budget**: I invite everyone with a risk score of **0.323 or more**, because that is exactly the top 10%.
- I set it using "out-of-fold" scores: each training person is scored by a model that never saw them. That way the threshold isn't too optimistic, and I never peek at the test set.
- If money didn't matter, the cost-optimal threshold would be much lower (0.041) and I would invite **66%** of people, since missing a case costs 33 times more than a wasted test. The budget is what limits the decision, not the costs.

---

## 3. Data preparation & feature engineering

### Step 1: Remove columns that "cheat"

Some columns predict diabetes only because **the person already knows they have it**. A diagnosed diabetic sees the doctor every year and gets more vaccines and scans. So "had a check-up last year" predicts diabetes, but it is a *result* of the diagnosis, not a *cause*.

I removed: last check-up, chest scan, flu / pneumonia / tetanus vaccines, HIV test, and alcohol. Non-drinkers have twice the diabetes rate (18% vs 9%). A likely explanation is that many people stop drinking after they are diagnosed, but the data can't prove this. I also removed two columns with no sensible link to diabetes (COVID and "high-risk last year").

Keeping them makes the score a bit better (PR-AUC 0.402 vs 0.389), but the model would learn to spot people who are **already diagnosed**, which is the opposite of what screening is for.

I **kept** heart disease and kidney disease, because the official diabetes guidelines (ADA) say these are reasons to get tested.

### Step 2: Split the data, and lock the test set away

- **80% to train, 20% to test.** The test set (3,448 people, 437 with diabetes) is only used **once**, at the very end.
- **Stratified:** both parts keep the same 12.7% of diabetics.
- A random split is fine here because it is one year of survey data: there is no "future" that could leak into the past.
- Instead of a separate validation set, I used **5-fold cross-validation** inside the training data to make every tuning decision.

### Step 3: Create better features

My model is a **logistic regression**. Put simply, it multiplies each column by a weight and adds everything up. That means it can only draw **straight lines**. Every feature I created gives it something it cannot build on its own:

| New feature | Why |
|---|---|
| Age as a number, plus age² | Risk goes up with age but flattens after 80 (23% at 75-79, 18% at 80+). A straight line can't bend; a squared term can. |
| BMI, BMI², and "obese" (BMI ≥ 30) | Risk jumps fast at high weight: 6% at normal weight, 12% overweight, 28% above BMI 40. |
| BMI fixed when missing | If BMI is blank but weight and height are there, I calculate it. Impossible values (below 14, above 70) become missing. |
| Short sleep (< 6 h) and long sleep (> 9 h) | Both too little and too much sleep are linked to risk: a U shape. |
| "Frequent bad days" (14+ of 30) | The CDC's own definition of frequent distress. |
| General health 1-5, teeth lost 0-3 | These answers have an order, so I keep that order as a number. |
| Any heart problem (heart attack OR angina OR stroke) | An "any of these" rule. A model that only adds things can't build it. |

I tried two more features and **removed** them, for different reasons:

- **Number of illnesses:** redundant. It is just the sum of the illness columns, and the model can already add those up by itself.
- **Number of unanswered questions:** it added nothing, because the "was missing" columns (step 4) already carry that information.

### Step 4: Prepare the data (learned from training data only)

- **Missing numbers:** filled with the training median, plus a yes/no column saying "this was missing". Not answering can itself be a signal.
- **Missing categories:** "Missing" becomes its own category.
- **Categories to columns (one-hot):** one category per group is left out as the reference (female, White non-Hispanic, never smoked, never used e-cigarettes). So a positive weight for "daily smoker" means "more risk than someone who never smoked". Without this, the weights are not unique (the "dummy variable trap") and the three models would give different weights.
- **Scaling:** every number is rescaled to mean 0 and spread 1.
- Every number used here (medians, averages, category lists) comes **only from the training data**. Otherwise test information would leak in.

### Did feature engineering help?

Cross-validation on the training data, C = 0.1:

| Features | ROC-AUC | PR-AUC |
|---|---|---|
| Original columns only | 0.8185 | 0.3841 |
| **With my new features (used)** | **0.8193** | **0.3888** |
| Without the non-linear terms (squares, cut-offs) | 0.8161 | 0.3794 |
| With the "cheating" columns (not allowed) | 0.8277 | 0.4022 |

**Honest answer: only a little.** The gain is small and almost inside the noise (± 0.009). The reason: almost all original columns are already categories, and one-hot gives each category its own weight, which already captures a lot of curves. The features that helped most were the age and BMI curves.

---

## 4. One model, three implementations

All three are **the same model** (logistic regression with L2 regularisation) solving **the same equation**. Only the tool is different.

### Settings

- **Regularisation C = 0.1.** I tried 0.001, 0.01, 0.1, 1 and 10 with cross-validation. 0.1, 1 and 10 scored the same, so I chose the simplest model (the strongest regularisation).
- **Learning rate = 2.0 and 3,000 epochs** for PyTorch. I tried 0.01 to 4 and picked the one that got the lowest loss without blowing up. At 4 the loss went up again (the steps were too big and it jumped past the minimum).

### Comparison

| | scikit-learn | PyTorch, manual loop | PyTorch, standard |
|---|---|---|---|
| **How it is built** | `LogisticRegression(C=0.1)` | Plain tensors `w` and `b`. Loss written by hand. `loss.backward()` for the gradients, the update step `w -= lr * w.grad` written by hand | `nn.Module` with one `nn.Linear`, `BCEWithLogitsLoss`, `optim.SGD` |
| **How regularisation is added** | `C = 0.1` | `+ (λ/2)·‖w‖²` added to the loss by hand | `weight_decay = λ` on the weights only |
| **Optimiser** | L-BFGS (smart solver, 64-bit numbers) | Gradient descent, full batch (32-bit) | Gradient descent, full batch (32-bit) |
| **Settings** | C = 0.1 | lr = 2.0, 3,000 epochs | lr = 2.0, 3,000 epochs |
| **Final training loss** | 0.298620 | 0.298620 | 0.298620 |
| **Biggest difference in probability vs sklearn** | (reference) | 0.0017 | 0.0017 |
| **Biggest difference in a weight vs sklearn** | (reference) | 0.0053 | 0.0053 |
| **Same invite / don't invite decision** | (reference) | 3,448 of 3,448 | 3,448 of 3,448 |
| **Diabetics found in the top 10% (test)** | 139 | 139 | 139 |
| **Precision at the budget** | 41.2% | 41.2% | 41.2% |
| **Recall at the budget** | 31.8% | 31.8% | 31.8% |

**How the two settings connect.** scikit-learn and PyTorch write the same penalty in different ways. They match when **λ = 1 / (C × number of training rows)**. I used that formula, so the three are solving exactly the same problem.

**Why they don't match to the last decimal.** scikit-learn uses a different solver and more precise numbers (64-bit vs 32-bit). The differences are tiny and **no decision changes**.

**What went wrong first (and how I fixed it).** At first the three gave different weights. There were three causes: (1) the dummy variable trap, (2) duplicated "was missing" columns, and (3) age and age² being almost the same column (0.99 correlated), which made gradient descent very slow. I fixed all three **in the data** (dropping a reference category, removing the duplicates, centring age² as (age − 50)²), not by forcing the models to agree.

---

## 5. Evaluation & dashboard

### Is the model better than doing nothing smart?

Test set only (3,448 people), inviting 10% of them:

| Who gets invited | Diabetics found | Precision | Total cost (test set) |
|---|---|---|---|
| Random people | 41 | 11.9% | €405,120 |
| The oldest people first (common sense rule) | 56 | 16.2% | €389,670 |
| **My model** | **139** | **41.2%** | **€303,940** |

- **3.4 times** more diabetics found than at random, and **2.5 times** more than the "oldest first" rule.
- **€101,180 saved vs random invitations, on these 3,448 test people.** This is not a saving for the whole state; it shows the size of the gain on a sample.
- ROC-AUC on the test set is about 0.81 (95% range: 0.789 to 0.828).

### Is it fair?

I checked whether the model works equally well for different groups (test set):

- **Women vs men:** ROC-AUC 0.807 vs 0.808. Same quality.
- **Black respondents:** they have more diabetes (19.6%) and get invited more (20.9%), which is what we want. But the model ranks them a bit less well (ROC-AUC 0.774).

I **used** sex and race as inputs on purpose. In medicine there are real biological differences, and diabetes guidelines take ethnic background into account. This is not like hiring, where using sex would be discrimination.

### The dashboard (`app.py`)

- It **never trains anything.** It only loads the saved models from `artifacts/`, rebuilds the same test set from the saved split, and predicts.
- **Results:** test metrics for the three models and the baselines, ROC curves, the training loss of both PyTorch versions, and a check that the three models agree.
- **Prediction vs reality:** a calibration chart (in each risk group, how many really have diabetes) and a dot plot of sklearn vs PyTorch probabilities.
- **Threshold explorer:** a slider for the threshold and boxes for the two costs. It updates the confusion matrix, the cost in euros, a warning when you go over budget, and the comparison with random invitations.
- **Also:** how each feature differs between diabetics and non-diabetics, the weights of the three models, the feature engineering comparison and the fairness audit.

---

## 6. Limitations & next steps

**1. I predict *diagnosed* diabetes, not diabetes.** People who have diabetes but don't know it are labelled "no". Those are exactly the people screening wants to find, so the model learns the profile of people who *were* diagnosed.
**Next step:** use **NHANES**, a CDC survey that also takes a blood test (HbA1c). It shows who has diabetes without knowing it. I would check whether my model ranks those hidden cases high, and if not, retrain using the blood test as the target. Once the programme runs, the test results of invited people become new, correct labels to retrain on.

**2. Everything is self-reported.** People round their weight down, forget, or skip questions.
**Next step:** NHANES also measures weight and height. I would compare self-reported vs measured BMI there, and check how much the model's ranking changes when BMI is shifted by the typical error.

**3. The costs are assumptions.** €30 and €1,000 are reasonable guesses, not real New York figures.
**Next step:** replace them with published figures (cost of an HbA1c test, yearly cost of treating diabetes complications). Note that **who gets invited does not depend on the costs**: the budget sets the threshold. The costs only change the euro numbers, so I would report a range of savings for low and high cost assumptions. The dashboard already lets you change them.

**4. The test set is small.** Only 437 diabetics, so the 41% precision could really be anywhere from about 36% to 47%.
**Next step:** add the New York BRFSS data from other years (2021, 2023) to get more diabetics, and report repeated cross-validation over the whole dataset instead of one 80/20 split.

**5. Most cases are still missed.** With a 10% budget I find about 1 in 3 diabetics (recall 31.8%).
**Next step:** give the health department a **"budget vs diabetics found" curve** (recall at 10%, 15%, 20%... of people invited), computed on the out-of-fold scores, so they can see what extra money buys. Another option is cheaper first steps, like an online risk questionnaire, before sending the full test.

**6. One state, one year.** It may not work as well elsewhere.
**Next step:** The full Kaggle file already includes the other states and territories. I would apply the New York model, **unchanged**, to each one and compare ROC-AUC, precision at 10% and calibration with New York. Where diabetes is more or less common, the ranking may still work while the probabilities are off; then I would only re-adjust the intercept for that state. Repeating it on the 2023 survey would test whether it holds over time.

**7. It's a simple model.** Logistic regression only draws straight lines, plus the curves I gave it.
**Next step:** train a gradient boosting model (for example `HistGradientBoostingClassifier`) on the same split and compare PR-AUC and precision at 10%. If the gain is small, I'd keep logistic regression, because the health department can read its weights.

**8. Weaker for some groups.** The model ranks Black respondents less well (ROC-AUC 0.774 vs about 0.81 overall).
**Next step:** pool other states and years to get more Black respondents, check calibration for each group, and try interactions such as race × BMI. ADA guidelines already use a lower BMI cut-off for Asian Americans, which suggests the BMI effect differs by group.

**9. Removing the "cheating" columns is a judgement call.** I removed them based on reasoning, not proof, and some may also carry real risk information.
**Next step:** test it with NHANES. If a column like "check-up last year" predicts *undiagnosed* diabetes too, it is a real risk signal and could come back. If it only predicts *diagnosed* diabetes, it is a consequence and stays out.

---

## 7. Generative AI use disclosure

I used generative AI (Claude, by Anthropic) as an assistant during this assignment:

- **Code:** to help write and debug the code (`features.py`, `train.py`, `app.py`, `data/make_ny_subset.py`), including finding why the three implementations first gave different weights.
- **Data analysis:** to help explore the dataset, for example diabetes rates by age, BMI, sleep and alcohol, which shaped the target definition, the removed columns and the new features.
- **Design decisions:** Claude proposed recommendations for the main design decisions, such as the test budget (10%), the costs of a wasted test and a missed case, and which features to create or remove.
- **Report:** to help me structure and word this report, and to help me with ideas for the next steps in the limitations section.

I studied each decision so I can explain and defend it. I checked all outputs by running the code myself, and all numbers in this report come from my own run of `main.py train`.

---

## How to run it

```bash
uv run python main.py train   # trains everything and saves to artifacts/
uv run python main.py app     # opens the dashboard (it does not retrain)
```
