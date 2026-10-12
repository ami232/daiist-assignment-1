# Assignment 1 Report

- **Name**: Sophie Jasmine Leonaard
- **Student ID**: 31860
- **Email**: sjleo.exstudents2026@student.ie.edu
- **Group**: PPLEDBA 5A

## Dataset
This assignment uses the Airline Satisfaction dataset, which was taken from the Marketing Data Analytics course material in the past semester. There are 10,600 rows and 22 columns, where each row corresponds to one observation of a passenger on a particular flight. The data includes many variables such as demographic characteristics (Age, Gender), membership status (Airline Status, Class), behavioral variables (Shopping/Eating Amount, No of Flights), flight-related variables (Flight Date, Delay, Distance, Cancelled), and geographical variables (Origin/Destination State). This dataset was chosen because I wanted to predict a binary categorical outcome using a logistic regression model. In this case, the target variable indicates whether a person is satisfied or not satisfied.

## Business / real-life framing
The main objective is to help the Marketing Division to detect the passengers who may have the tendency of being unsatisfied before they churn. For this purpose, the Not Satisfied class is considered as the positive class (Y=1).
Assuming that False Negative (unsatisfied passenger predicted as satisfied) costs much more than False Positive (satisfied passenger predicted as unsatisfied). Quantifying this statement, the cost of having a False Negative is considered to be equal to the median number of flights per customer (17 flights, from No of Flights column) multiplied by the average price of a ticket (approximately $125) and reduced by 2.5x, because assuming that all the False Negatives lead to customer churn is too pessimistic, resulting in an estimated cost of $850. Meanwhile, the cost of a False Positive is estimated to be about $15 to cover the retention vouchers costs.
Due to the 57:1 cost ratio, the ideal threshold should be very close to 0 since the goal of the model is to try to detect each customer who is not satisfied whatever it takes. However, such a low threshold does not work very well since it will alert every passenger. To balance this, a threshold value of 0.3 is advised for the dashboards. Although the model had been trained using the default value of 0.5, the use of 0.3 as the decision threshold while doing the evaluation helps in capturing more passengers who are at risk but prevents the team from being overwhelmed with false positives. This shows in figures that the cost is greatly reduced using the value 0.3 than 0.5 for the dashboard.
A time split was done using the chronological order of Flight Date and splitting the dataset. The months from January to November early was used for training (80%), while later data used for testing (20%). For this assignment, random split was not chosen because future patterns could affect the model's learning process and lead to data leakage.

## Data preparation & feature engineering
The data cleaning process started by filling the missiong values in 418 rows in various columns: Flight Time, Arrival Delay, and Departure Delay. It was found that 393 of these rows were those where the flight was cancelled, but the other 25 rows had data issues. To solve this problem, a new variable (flight_completed) was created to identify which flights had actually departed.
To manage high-cardinality features in Origin State and Destination State (with 37-40 unique categories each), states were grouped into 4 primary regions (Northeast, Midwest, South, West) according to US Census Bueau classification before implementing one-hot encoding. This step was done to prevent adding too many columns that could cause overfitting while preserving regional behavioral patterns.
Ordinal encoding was used for Airline Status and Class because of the hierarchy of the categories. Meanwhile Gender, Type of Travel, and Airline Name were one-hot encoded with drop_first = True. The numerical variables were standardized using StandardScaler on the training data to prevent data leakage. Lastly, the non-predictive column (CustomerID), original categorical columns that had been replaced, and redundant feature (Month of Flight Date) were dropped.

## Modeling: three implementations, one model
The logistic regression model was used because the target variable is binary. It used three different models, which were scikit-learn, PyTorch training loop, and default PyTorch. These models were assessed using the same test dataset compared to a naive baseline.
The parameter regularization strength for scikit-learn (SKLEARN_C) was examined based on the comparison of the training accuracy (76.98%) and the test accuracy (76.13%). The insignificant difference between them shows that there is no overfitting; hence, SKLEARN_C with C = 1.0 was kept. For PyTorch models on the other hand, the training loss graphs for the two methods (manual looping and nn.Module) had reached stable convergence even before the 500th epoch. Thus, it was clear that the chosen learning rate of 0.1 and 2,000 epochs were sufficient for the task.
The naive baseline had an accuracy of roughly 52.0%, while precision, recall, and F1 Score were all 0.0%. In contrast, the three logistic regression models showed comparable and superior results than the baseline with approximately 76.0% accuracy, 73.0% precision, 60.0% recall at default threshold (0.5), and 66.0% F1 score.
Although all three models were generally aligned, there were still some minor differences, which are considered normal. Scikit-learn uses analytical solver, while PyTorch is based on the gradient descent.

## Limitations & next steps
1. The roughly 60% recall rate means that about 40% of unsatisfied customers will still go undetected. The company could consider adjusting the classification threshold to identify more unsatisfied passengers.
2. The cause behind the 25 rows of data quality issue remains unclear. To perform a better analysis, this needs to be found and corrected by auditing.
3. The dataset consists of only one year worth of data in 2019. The company could add more historical data or start recording relevant data to enable more accurate predictions in the future.

## Generative AI use disclosure
Generative AI tools were used to help dataset selection brainstorming, fixing code bugs, explaining key concepts, building dashboard, and report formatting. However all the essential decisions, such as dataset selection, target class, and splitting methodology, were personally made.