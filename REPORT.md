# Assignment 1 Report

- **Name**: Rakan Hourani
- **Student ID**: 19907
- **Email**: rhourani.ieu2022@student.ie.edu
- **Group**: BBADBA 5A 

## Dataset

The dataset is the official census of every commercial place in Madrid, which was published by the Ayuntamiento de Madrid. It’s updated monthly, and records each place’s activity (whether it’s a bar, restaurant, shop…), its current status (whether it’s open or closed) and its location and other features. I decided to keep the food and drink venues. 

I got it from datos.madrid.es from the historical version of the census, and used three snapshots from April 2023, April 2024 and April 2025. One row represents a bar or restaurant that is open in April of a given year. Its features describe it on that date and its target says whether it was still the same business one year later. 

There is 33 622 rows and 17 columns. The row count is above the guide, but it is because both years were needed since the model was trained on 2023 venues and then tested on 2024. The raw files were large so I merged them into one single CSV file (madrid_bars_restaurants.csv) for the data that was needed. 

I picked this dataset because I am interested in the food business industry, and since I live in Madrid I thought it would be interesting to see what features, and patterns lead to a closure of a venue here and how that information can be used to avoid unecessary losses. 


## Business / real-life framing

The model was created for a brewery that lends free equipment (like taps and coolers) to bars and restaurants in exchange for their exclusivity, and incurs losses when a business they have lent to closes within the year. So this model, allows the brewery to classify the venues in madrid and the risk that they bring. This way, they can ask for a deposit from businesses that are more likely to close down within the next year and cut their losses. And if the venue does not close, the deposit was just a precautionary measure (at the risk of losing some customer loyalty).

I defined the target by counting a place that had closed, changed its activity, or had become a new business as ‘did not survive’, because if a place changed its activity or changed its name, the old business still closed. Even if it’s the same owner or place, the old business still failed for a reason. I decided to drop canteens because they are contracted by institutions, while the rest of the venues depend on actual traction and foot traffic. 

I did a time based split because it was important since the brewery would use this model on venues from next year, so the test should copy that by learning from 2023 and checking it on 2024. Having the venues mixed randomly would have allowed data from 2024 into training, where the model would have learned from future values, making the scores look better than it would actually be. 

I think that the metric that should decide the threshold is cost, because based on assumptions, a missed failure costs the brewery €1,500 in lost equipment, whereas a false alarm only costs €150, but with the risk of losing a venue to a rival. Accuracy is not reliable here because it is misleading, by looking great (93.5%) but it would never catch a failure before it happened. It also treats a missed failure and a false alarm as equally bad, when they are not. So the threshold is the one that minimizes total costs on the validation data which is at about 0.08. 

## Data preparation & feature engineering

Data Preparation
I started with the CSV which already had the institutional canteens removed from it (it was about 1 960 venues a year). For all the broken coordinates, venues that had coordinates as 0 or outside of Madrid, were replaced with their neighborhood’s average location. I did this because the ‘competitors within 250m’ feature is calculated from locations. The wrong coordinates would lead to false results and confuse the model. 

I treated certain terrace permit dates like ’31/12/9999’ as unknown because the model would read it as a real date and it looks like its from the future which would lead to a wrong terrace group. 

Opening hours was not used because it was missing for 94% of the venues so it would not be able to teach the model anything reliable. 

I grouped the rare venue types like Internet cafe, or street food because with so little examples, if by chance one of them happened to close the model would then ‘learn’ that Internet cafes are risky and grouping them avoids that overfitting. 

I did one hot encoding for each category to become its own 0/1 column, since the model works with numbers and can’t multiply ‘Retiro’ by a weight. The numeric features are standardized to similar ranges to help train smoothly. The encoding and scaling settings are learned from the training data only, because if they were learned from all the data the information from the test year would leak into training. 

Features 
- District (2.8% Latina vs 14.9% Retiro): the location of the venue drives rent, foot traffic, tourism and local spending
- Venue type: different business models have different risks and successes. The differences between types were small (apart from tabernas which are often long-established, traditional places maybe making them most stable)
- Setting (11.5% shopping centers): tells us where the venue is located in a mall, market, or street and we can see that the ones in shopping centers failed more 
- Not street level: a venue inside a building or on a higher floor, is not as visible to a passer by, so it gets fewer walk ins
- Mixed use: when a bar has other activities than just drinks, it’s less central to the owner. 
- Terrace Permit: having a terrace doesn’t matter but their permit for it does. A recent permit usually means a younger business, which tend to fail early on and an established permit means the venue has already survived several years.
- Competitors within 250m: it was a surprise, since our results showed the opposite of what one would expect. I thought more competition meant more closures but busy areas full of bars are established, hospitality streets with a lot of foot traffic. 


## Modeling: three implementations, one model

I chose to use logistic regression because it gave each venue a probability of not surviving, which is useful for the brewery to be able to compare to the threshold. 

| Model | Validation ROC-AUC | Validation PR-AUC | Test ROC-AUC | Test PR-AUC | Test log loss |
|---|---|---|---|---|---|
| Baseline: base rate | 0.500 | 0.065 | 0.500 | 0.096 | 0.324 |
| Baseline: district average | 0.605 | 0.091 | 0.548 | 0.107 | 0.327 |
| scikit-learn | 0.587 | 0.095 | 0.564 | 0.110 | 0.324 |
| PyTorch manual loop | 0.587 | 0.095 | 0.564 | 0.110 | 0.324 |
| PyTorch nn.Module + optim | 0.587 | 0.095 | 0.564 | 0.110 | 0.324 |

The three models are in agreement, with the same scores and training loss, with weights and predictions minimally differing because each optimiser stops at a slightly different point near the same solution. 

## Limitations & next steps

- A limitation is that the costs are assumptions I have made, I would address this by finding the real figures for the equipment value and lost customers

- 2024 had more failures than 2023 (9.6% vs 6.5%), meaning the threshold chosen on 2023 data was too cautious, and 'deposit from everyone' was slightly cheaper in 2024. The model would need to be retrained every year and the threshold rechecked on recent data

- Modest signal, the model comes out with a score of about 0.56 (ROC-AUC) so it only partly separates failing venues. I would add better data and more details like the opening date, age, or customer reviews of the venues. 


## Generative AI use disclosure

I used AI to help prepare the dataset and to help develop and understand the code for the training notebook, the three models and the Gradio dashboard

I then used it to check and reformat what I wrote for the report and to make sure I responded to everything asked. 