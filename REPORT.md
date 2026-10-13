# Assignment 1 Report

- **Name**: Yasmin Tandon
- **Student ID**: 20356
- **Email**: ytandon.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset
source: https://www.kaggle.com/datasets/srisyra02/zomato-market-analysis 

Zomato is the largest food delivery/restaurant app in India. Similar to UberEats here in Europe, you can order in or schedule pick ups. And also similar to TheFork, you can rate restaurants, get disco[...]

The dataset is 9551x21 with each row representing one restaurant. I filtered it to only include restaurants located in India and then did some feature engineering. This Brought the effective dataset s[...]

## Business / real-life framing

###Scenario: 
A cloud-kitchen company (delivery only) is choosing which cuisine and price concepts to launch in 2 localities, for example assume Gurgaon and Noida. The model estimates the probability that a concept[...]

###Target: 
The target is binary: success means rating ≥ 3.8. That roughly marks the top fifth of the market. A yes/no target matches the yes/no decision to launch, so I used logistic regression rather than pre[...]

###Training population: The model is trained on all rated Indian restaurants and uses a has_delivery flag - some restaurants were dine in only - here the price sensitivity of customers goes down, they[...]

###Split: There was a risk of leakage between outlets of the same chain: Domino's has 79 rows and Cafe Coffee Day has 83. A random split would test the model on near-copies of restaurants it trained o[...]

## Data preparation & feature engineering

As mentioned above, I did filter the dataset down to only include restaurants located in India

- Price (the hypothesis):
  - log_cost. Log scale, so doubling the price counts the same at any price level.
  - rel_price = price ÷ the average of the training-set median prices of each listed cuisine. This measures how expensive a place is compared with what's normal for its food.
  - rel_price², so the model can draw a hill (a sweet spot).
  - log_cost × delivery.
  - Relative price is clipped to its 1st–99th training percentiles (0.25×–3.71×) before squaring, so a few restaurants charging 10× their cuisine's norm don't dominate.
- Services: has_delivery and has_table_booking. Table booking absorbs the "nice dining room" effect, so that effect isn't wrongly credited to price.
- Brand: is_chain (more than one outlet in India). A new brand gets 0.
- Cuisine:
  - n_cuisines.
  - One flag for each cuisine with at least 50 training restaurants (27 cuisines).
  - cuisine_Other pools the rare cuisines, so the model doesn't learn noisy weights from a handful of examples.
- Location:
  - log(1 + restaurants in locality), as a measure of competition.
  - log(locality median cost), for upmarket versus budget areas.
  - City dummies for Gurgaon, Noida, Faridabad and "Other India", with New Delhi as the reference city.

Every price statistic (per cuisine, per locality, per city) is computed from the training split only. The same values are then applied to validation, test and the app. There are two deliberate excepti[...]


## Modeling: three implementations, one model
Model. I used L2-regularised logistic regression, because the target and the decision are both binary.

One objective for all three. All three implementations minimise the same objective: mean BCE + λ/2·‖w‖², with λ = 1/(C·n) and no penalty on the bias. That is exactly scikit-learn's objective.[...]

Tuning:

- scikit-learn: C was chosen on validation log loss from 0.001–100. C = 0.1 won (λ ≈ 0.00215). Validation log loss barely moved between C = 0.01 and C = 100. With about 4,600 rows and 42 weights[...]
- Manual PyTorch: raw tensors w and b, hand-written sigmoid + BCE + L2, backward(), an update inside no_grad(), then grad.zero_(). It uses full-batch training for 3,000 epochs, starts from zero weight[...]
- Standard PyTorch: an nn.Module with one nn.Linear layer, BCEWithLogitsLoss, Adam with OneCycleLR, full batch, 3,000 epochs and a max_lr search over 0.001–0.1. max_lr = 0.1 was chosen.

Test-set results (validation in brackets):

| Model | Log loss | ROC-AUC | Avg precision | Brier |
|---|---:|---:|---:|---:|
| Naive baseline (p = 0.218) | 0.5253 (0.5253) | 0.500 | 0.219 | 0.1709 |
| scikit-learn | 0.3562 (0.3827) | 0.8693 (0.8399) | 0.6850 | 0.1118 |
| Manual PyTorch | 0.3565 (0.3825) | 0.8694 (0.8399) | 0.6853 | 0.1119 |
| Standard PyTorch | 0.3562 (0.3827) | 0.8693 (0.8399) | 0.6850 | 0.1118 |

All three beat the baseline clearly: log loss drops 32% and ROC-AUC rises from 0.50 to 0.87.

Business decision at t = 0.67 on the test set. The model launches 65 concepts, of which 54 succeed (precision 83%, recall 27%). Total cost is Rs 910 lakh (which would be 91,000,000 rupees or approx.91[...]

- scikit-learn and standard PyTorch agree almost exactly. The largest difference in any test probability is 3×10⁻⁷, and their training objectives are identical (0.379474). Adam with OneCycleLR fu[...]
- Manual PyTorch agrees almost, but not exactly. The largest difference in a test probability is 0.014 and the mean is 0.001. The gap comes from two choices:
  a. lr = 0.1 was picked because it had the lowest validation BCE. After 3,000 epochs it is still slightly short of the optimum (training objective 0.379494 against 0.379474). lr = 1.0 reaches the exa[...]
  b. The remaining difference sits in rel_price (0.37 against 0.43), rel_price² and log_cost. These three features are strongly correlated. That creates a long, flat valley in the loss surface where [...]

What the model says about the hypothesis: The fitted relative-price curve is concave: the coefficient on rel_price² is negative, so the price effect weakens as price rises. But its peak is at 5.0× t[...]



## Limitations & next steps

1. A rating doesnt really indicate profit. A 3.8-rated kitchen can still be losing money. Next step would be to combine the model with order-volume or margin data, or use Votes as a proxy for demand i[...]
2. Only surviving restaurants are listed. Closed restaurants are missing, and 2,139 unrated listings (probably newer ones) were dropped, which is likely to make success look more common than it is. So[...]
3. A  lot of the datadine-in. Only about 36% of restaurants deliver, and only around 120 delivering restaurants cost more than Rs 1,500. So the curve above Rs 1,500 is mostly extrapolation. Next step [...]
4. A single split adds noise. Test scores are higher than validation scores (AUC 0.869 against 0.840), which shows how much results vary between splits of about 900 rows. 
5. The business costs are assumed. Rs 15 lakh and Rs 5 lakh drive the threshold. This doesnt really indicate a real world scenario and are arbitarily chosen numbers.
6. The model only measures association, not cause. Price, cuisine and location are chosen by restaurant owners, and they correlate with things the data doesn't capture, such as food quality. The model[...]

## AI Use
I used AI for dashboard and writing some code (with explicit instructions and my review). It was also used for brainstorming when selecting a dataset.
