# Assignment 1 Report

- **Name**: Jacobo Galindo
- **Student ID**: 19738
- **Email**: jgalindo.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

 I used the Hotel Booking Demand dataset available on Kaggle. It contains booking-level information for two hotels in Portugal, a City Hotel and a Resort Hotel, including variables related to booking timing, customer characteristics, stay details, distribution channels, previous booking behavior and whether the reservation was ultimately canceled. Each row represents one hotel booking, and the target variable is is_canceled, where 1 indicates that the booking was canceled and 0 that it was not.

 The original dataset contains 119,390 bookings and 32 columns. For this project, I restricted the analysis to Resort Hotel bookings with an arrival year of 2016, resulting in 18,567 observations. I did this for two reasons. First, it kept the modeling sample within a manageable size for the assignment. Second, focusing on one hotel type and one year created a more homogeneous setting, since cancellation behavior can differ across hotel types and across periods.

 I chose this dataset because hotel cancellations represent a realistic business problem with a clear operational decision behind it. It also contains a mix of numerical and categorical variables, which makes it suitable for meaningful preprocessing and feature engineering rather than simply fitting a model to already prepared data.


## Business / real-life framing

The model is designed for a hypothetical hotel setting in which the objective is to estimate the probability that a booking will be cancelled. A decision threshold is then used to classify each reservation as likely to cancel or not. In practice, bookings classified as likely to cancel could be considered for a low-cost preventive action, such as a confirmation reminder.

The target variable is is_canceled, where 1 indicates a cancellation and 0 a completed booking. Since the prediction should rely on information available when the booking is made, I removed variables such as reservation_status and reservation_status_date, which directly reveal the final outcome, as well as other variables that may only be known after the initial reservation.

I used a chronological split instead of a random one because, in a real deployment, the model would be trained on past bookings and applied to future ones. I reconstructed booking_date from the arrival date and lead_time, ordered observations chronologically, and created train, validation and test periods. The validation set was used to select the decision threshold, while the test set was kept for the final evaluation.

The threshold was not chosen based on accuracy alone. A false negative means missing a cancellation that could potentially have been prevented, while, in my simplified business framework, a false positive creates only the cost of an unnecessary intervention. I therefore selected the threshold using an expected incremental profit calculation on the validation set, based on illustrative assumptions of a €5 intervention cost, a 20% probability of preventing a cancellation, and a 70% contribution margin. Under these assumptions, the selected threshold was 0.03, prioritizing high recall while still accounting for the economic cost of false positives.

## Data preparation & feature engineering

I first filtered the original dataset to Resort Hotel bookings from 2016, which resulted in 18,567 observations. Missing values in country were replaced with "Unknown" so that these bookings could still be retained. I removed variables that were either unavailable at booking time, created data leakage, or were not meaningful for this model. In particular, reservation_status and reservation_status_date were excluded because they reveal the final outcome, while company and agent were removed because they are mainly identifier fields and also contained substantial missing data.

I engineered several features to represent booking characteristics more directly. total_nights combines weekday and weekend stays into one measure of the total duration of the booking, while party_size combines adults, children and babies into the total number of guests. I also created binary indicators such as has_children, has_previous_cancellation and has_special_requests. These variables were designed to capture meaningful distinctions without assuming that the effect of the original count variables is perfectly linear. For example, has_children distinguishes bookings with children from those without children, rather than assuming that each additional child changes cancellation risk by the same amount. Similarly, has_previous_cancellation captures whether the customer has any cancellation history, and has_special_requests whether there is any sign of additional engagement with the booking.

Numerical variables were standardized using StandardScaler, which puts them on comparable scales by centering them around their mean and scaling them by their standard deviation. This was useful because variables such as lead_time can take values in the hundreds, while others such as is_repeated_guest are only 0 or 1. Categorical variables were transformed using one-hot encoding with OneHotEncoder(handle_unknown="ignore"), which converts each category into binary indicator columns and prevents errors if a new category appears in validation or test data. Importantly, the preprocessing pipeline was fitted only on the training set and then applied unchanged to the validation and test sets, avoiding information leakage. After preprocessing, the final input contained 139 model features.

## Modeling: three implementations, one model

I used logistic regression because the target, is_canceled, is binary. I implemented the same model in three different ways: scikit-learn, a manual PyTorch training loop with explicit weights and gradient updates, and the standard PyTorch workflow using torch.nn.Module, BCEWithLogitsLoss and torch.optim.SGD. All three models used the same features and the same chronological split. I also compared them against a naive baseline that always predicts the majority class.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Naive baseline | 0.859 | 0.000 | 0.000 | 0.000 |
| Scikit-learn | 0.742 | 0.298 | 0.612 | 0.401 |
| Manual PyTorch | 0.832 | 0.388 | 0.320 | 0.350 |
| Standard PyTorch | 0.834 | 0.387 | 0.305 | 0.341 |

The two PyTorch implementations produced very similar results, which is expected because they implement the same underlying logistic regression model in two different ways. The scikit-learn model behaved differently, achieving substantially higher recall but lower accuracy and precision. Although the naive baseline achieved the highest accuracy, it never identified a cancellation, resulting in zero precision, recall and F1. This shows why accuracy alone is not appropriate for this problem.

The differences between scikit-learn and the PyTorch models are likely due to differences in optimization and convergence. To make the comparison more consistent, I removed the default regularization from the scikit-learn model, since the PyTorch implementations were also unregularized.

## Limitations & next steps

One limitation is that the analysis only uses Resort Hotel bookings from 2016, which makes the sample more consistent but limits how well the results can generalize to other hotels or periods. With more data, I would test the same pipeline across multiple years and hotel types while keeping a chronological evaluation.

A second limitation is that the threshold of 0.03 selected on validation performed much worse on the final test period. It kept recall extremely high but created many false positives, reducing precision and expected incremental profit. With more historical data, I would evaluate the model across several consecutive time periods and monitor changes in cancellation behavior over time.

The business assumptions used for threshold selection are also illustrative rather than based on observed hotel economics. In a real implementation, I would replace them with actual intervention costs, contribution margins and measured campaign effectiveness.

Another limitation is that I used the predicted probabilities to define the decision threshold without explicitly checking whether they were well calibrated. As we did in the last session, a useful next step would be to assess model calibration and, if necessary, apply a calibration method. This would make the predicted cancellation probabilities easier to interpret and could make probability-based decision thresholds more reliable.

## Generative AI use disclosure

I acknowledge the use of ChatGPT (OpenAI) to support coding, debugging, technical explanations, dashboard development, and the structuring and wording of parts of this report.

Examples of prompts used include: “Explain the manual PyTorch training loop step by step,” “Why is StandardScaler used in this model?”, “Why do we fit preprocessing only on the training set?”, “How should I compare the three logistic regression implementations?”, “How can I build the Gradio dashboard without retraining the models?”, and “Help me make this section clearer and more concise.”

I mainly used AI when I was unsure about specific technical concepts or implementation details, particularly in PyTorch and in connecting the trained models to the Gradio app. The outputs helped me understand what the code was doing, identify errors, and improve the structure of the implementation. I reviewed and tested the generated code myself and made the final decisions regarding the business framing, feature engineering, evaluation logic, threshold selection and interpretation of results. AI was therefore used as a support tool for learning, debugging and communication, rather than as a substitute for the project decisions themselves.
