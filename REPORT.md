# Assignment 1 Report

- **Name**: Jaime Lopez Baillo
- **Student ID**: 19031
- **Email**: jlopezbaillo.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

---

## Dataset

**What it is:** 
The dataset is a lap by lap telemetry from F1 races between 2022 and 2025. Each row shows the tyre compound fitted, the number of laps the tyres have done (TyreLife), the time lost due to degradation, the race position, and if the car made a pit stop on that lap (PitStop) or will pit on the next lap (PitNextLap).

**Source:** Provided by kaggle as train (2).csv and test (2).csv.

**Size:** 439,140 training rows × 16 columns. 104 unique race-year events showing 20 drivers.

**Why I picked it:** I picked this dataset because I'm a huge fan of formula 1, and nowadays with the new regulations and new tracks added to the calendar, it's very complicated to overtake, so the most interesting moment of the race is the pit stop strategy, these decisions can decide race winners. And it's not only about planning the perfect time to pit but also to know when other cars are going to pit in order to plan undercuts. This dataset has also a clear binary target and good features to work with. 

---

## Business / real-life framing

**Scenario:** A race engineer on an F1 team must decide each lap whether to call their own drivers in for pit stop, either following the initial strategy or to respond to what rival teams are doing. If rivals pit and you don't react in time, they join the track on fresh tyres, set faster laps and undercut you so it cost track positions worth tens of thousands of euros in constructors' points prize money at end of the year.

**The model's role:** Predict whether a rival car will pit on the next lap, one lap in advance, so the strategy desk can prepare a counter pit response.

**Pipeline implications this drives:**

1. **Target definition:** PitNextLap is already binary, 0 = no pit next lap, 1 = will pit next lap. No engineering needed here, but the timing matters because we are predicting one lap ahead which is the window an engineer needs to act.

2. **Split strategy: race-based** If I had split the data randomly, some laps from a race would have end up in the training set and other laps from the same race in the test set. Meaning the model would have already seen that race like its safety cars, pit strategies, etc, which makes the test results look better than they really are. In real use, the model only sees the current lap and has never seen the rest of the race. To avoid this, I split by entire races, every race goes entirely into train and test, and is never divided between them.

3. **Metric:** AUC / ROC summarises performance across all thresholds and is the right metric here because the optimal threshold depends on the cost trade off, missed undercut compared to unnecessary alert. Since the threshold depends on the cost trade off, a missed undercut, false negative, is more expensive than a false alarm, false positive, the threshold should be tuned down from 0.5, and the dashboard allows to do it interactively, so you cana see the total cost depending where you set the threshold. 

---

## Data preparation & feature engineering

**No missing values** the dataset was clean.

**Compound encoding:** The 5 types of compound are wet, intermediate, hard, medium and soft, they are mapped with scores from 0 to 4 based on their degradation rate. I also used one hot encoding for the dry tyres so Soft, Medium and Hard to capture non linear effects. 

**Tyre wear ratio:** I used tyreLife and expected_stint_lenght to normalize the tyre age against the typical life of the tyre so for instance wet= ~20 laps, intermediate= ~25 laps, soft= ~25 laps, medium= ~35 laps, and hard= ~45 laps. For example a Soft tyre with 20 laps has much more degradation than a Hard tyre with 20 laps, and the age alone misses this.

**Quadratic tyre age:** tyre_life_sq: the degradation in F1 is non linear, meaning that the last laps of a set of tyres are way slower than the first ones. So to try to get a better approximation, I squared the TyreLife.

**Interaction:** compound_code × TyreLife: Since soft tyres degrade faster per lap, the slope of time loss compare to age differs by compound. This allows a logistic regression capture this interaction.

**Race remaining:** 1 − RaceProgress: The teams usually pit when there are enough laps left to benefit from fresh tyres and usually extend stints at the end. And this is different from RaceProgress, so both are included.

**PitStop:** current lap: a car that pitted this lap will almost certainly not pit again next lap, so this is a strong negative signal.

**Features excluded:** The features that I excluded are the driver id and race name since they are categorical and including them as dummies would overload the model and also enconding track specific patterns could hur the generalisation across new races. The Laptime is also dropped because we already have data that gives us information about the pace and the raw lap time varies depending on the circuit and the length, so it would add noise. 

**Scaling:** StandardScaler (zero mean, unit variance) fitted on the training races only, applied identically to validation and test.

---

## Modeling: three implementations, one model

**Why logistic regression:** The target is binary, so the logistic regression is the best choice it gives us interpretable coefficients and it's direclty comparable across sklearn and PyTorch.

**Class imbalance handling:** So only about ~20% of the laps are followed by a pit stop, so the data is imbalanced meaning there are many more no pit laps than pit laps. So I gave the laps followed by a pit stop more weight during training. This way the model can't get a good score by always predicting no pit. Scikit learn does this with class_weight='balanced', and PyTorch with pos_weight ≈ 3.87.

### Three-method comparison

| Method | Accuracy | F1 | AUC-ROC | Precision | Recall |
|---|---|---|---|---|---|
| scikit-learn | 0.7607 | 0.5223 | **0.8432** | 0.3899 | 0.7911 |
| Manual PyTorch | 0.7548 | 0.5164 | 0.8364 | 0.3832 | 0.7915 |
| PyTorch nn.Module | 0.7494 | 0.5118 | 0.8331 | 0.3775 | 0.7942 |
| Naive baseline | 0.8346 | 0.0000 | 0.5000 | — | 0.0000 |

All three trained on the same 73 race training set, evaluated on the same 16 race test set. Threshold = 0.5 for accuracy, F1, precision, recall.

**Do the three agree?** Yes, the three methods agree closely: their AUC differs by only 0.010. The small differences come from how well each one finished training, not from a real disagreement. sklearn trains more efficiently, so it ends slightly ahead. All three are much better than the naive baseline on F1 and AUC.

**Why accuracy is misleading:** Accuray is misleading because the naive always predict to not pit baseline scores 83. Because it simply predicts the majority class. This is why F1 and AUC are the real evaluation criteria here. 

**Manual PyTorch implementation notes:** In the manual PyTorch model, the model predicts the probability of a pit stop using sigmoid(X @ w + b). For the loss I did a weighted binary cross entropy. Then pytorch computes the gradients and I updated the weights with w = w - lr * gradient. I didn't use an optimizer and nn.module.

---

## Limitations & next steps

1. **No circuit features:** One limitation is that each track has different pit stop windows. If I include track id and lap count taking into account the length of the circuit it would improve the AUC. So a fixed circuit specific embedding could be added, but it would have a risk of overfitting if it's not regularized.

2. **Race dynamics ignored:** Another limitation that is not taking into account is that safety car periods change the way teams react, meaning that many teams will pit next lap not because of worn tyres but because it becomes an opportunity window. And the model has no data for safety car probability. What can be added would be a variance in lap times, that  could address this issue. 

3. **Single-lap lookahead only:** It is common for F1 teams, to make strategic decisions more than one lap in advance. So adapting the target to pits within the next x laps would make the model more useful for these situations. 

---

## Generative AI use disclosure

I used Claude code as a coding assistant during this assignment. It wrote most of the code, including the training loops, and it was especially helpful for the Gradio dashboard and the Plotly charts, which I had no previous experience with. I also used it sometimes to understand concepts and get ideas about what to analyze in order to improve the results. So I used it to speed up implementation and understanding. 