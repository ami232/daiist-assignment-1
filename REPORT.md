# Assignment 1 Report


- **Name**: Javier López-Usero Fernández
- **Student ID**: 18633
- **Email**: jlopezusero.ieu2022@student.ie.edu
- **Group**: BBADBA 5B


## Dataset


The Wine Quality Dataset is created from samples of Vinho Verde which is a portuguese wine. Some researchers collected physicochemical measurements of each wine and paired it with a quality score by blind tasting panels of sommeliers. One row represents one wine sample with a score in some characteristics and a quality score. There are 6497 rows in total, 1599 of red wine and 4898 of white wine. Quality scores are concentrated around 5-6 which makes it hard to distinguish what makes a wine great. I picked it because I think its interesting to see what makes a good wine great, on average, and to see to which features should companies focus more.

The dataset is from: UCI Machine Learning Repository, Cortez et al., 2009

## Business / real-life framing


An online club sells a monthly Premium Selection of wines from Vinho Verde at a higher price. Each pack comes with some basic measurements (the features in our data) and we want to create a model that selects those wines whose features will the sommeliers like. We are trying to answer whether chemistry can stand in in human tasting and we can decipher what makes a good wine great.
The target is to discover which features turn a wine into higher or same quALITY than a 7.
A time split is not neccessary since wines don't have dates themselves, so a random split was done. 
I also assigne a cost to a false negative and to a false positive meaning that total cost was the measure to be optimized for.

## Data preparation & feature engineering

As for feature engineering and data preparation we will do the following:
- Remove duplicate rows (18%) of the dataset
- Do a stratified train/val/test split. We want to know what makes a good wine great, so we will focus on a logistic regression directly. we are stratifying so that each  group has a representative sample in train, test and val.
- Log the right skewed features so that extreme value don't pull parameters towards them.
- Cap extreme values, in order to not lose data, capping instead of eliminating allows for more data preservation and replacing the 1% and 99% accordingly.
- Reduce multicolinearity by taking away strongly correlated features which might confuss the model.
- Standarize all features on training so that they are on the same scale
I created free_so2_ratio, which is a measure that protects the wine from oxidation and bacteria and color in 0 for red and 1 for white, allowing for one model to cover both types.
I dropped density since it was determined by alcohol and sugar and also free sulfur dioxide, since we put the ratio instead.

## Modeling: three implementations, one model

I turned quality into a yes or no target. with yes being over or equal to 7 and logistic regression is the best fit model for yes / no problems since it outputs a probability between 0 and 1 that the wine is good.
Linear regression could bredict below 0 or above 1.
They do agree because they minimize avg loss and have L2 penalty. They al agree that alcohol is the strongest positive factor and volatile acidity the strongest negative one.

**Test-set results (threshold = 0.5)**

| Model | ROC-AUC | Log loss | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Baseline | 0.5000 | 0.4851 | 0.8108 | 0.0000 | 0.0000 | 0.0000 |
| scikit-learn | 0.8311 | 0.3702 | 0.8183 | 0.5405 | 0.2649 | 0.3556 |
| Manual PyTorch | 0.8311 | 0.3702 | 0.8183 | 0.5405 | 0.2649 | 0.3556 |
| nn.Module + optim | 0.8311 | 0.3702 | 0.8183 | 0.5405 | 0.2649 | 0.3556 |

**Coefficients (standardised features)**

| Feature | scikit-learn | Manual PyTorch | nn.Module + optim |
|---|---|---|---|
| fixed acidity | 0.3486 | 0.3484 | 0.3484 |
| volatile acidity | -0.5462 | -0.5470 | -0.5470 |
| citric acid | 0.0388 | 0.0388 | 0.0388 |
| residual sugar | 0.1974 | 0.1976 | 0.1976 |
| chlorides | -0.4817 | -0.4808 | -0.4808 |
| total sulfur dioxide | -0.1920 | -0.1919 | -0.1919 |
| pH | 0.2953 | 0.2951 | 0.2951 |
| sulphates | 0.2739 | 0.2737 | 0.2737 |
| alcohol | 1.1235 | 1.1240 | 1.1240 |
| color | 0.3920 | 0.3915 | 0.3915 |
| free_so2_ratio | 0.3724 | 0.3723 | 0.3723 |
| intercept | -2.1069 | -2.1073 | -2.1073 |

**Largest difference in predicted probability vs scikit-learn (test set)**

| Model | Max \|P(sklearn) − P(torch)\| |
|---|---|
| Manual PyTorch | 8.73e-04 |
| nn.Module + optim | 8.73e-04 |

## Limitations & next steps

As the assignment wasn't about quality but about understanding I didn't find many limitations in technical aspects, since the use of AI and my previous knowledge allowed me to understand what was asked and how it should be done. However, GitHub is still a miscellaneous platform for me and the major dificulties were found while committing, formatting everything and making sure everything was readable and in the correct format.

## Generative AI use disclosure

I used Claude (Anthropic), through Claude Code in VS Code, as a coding assistant for this assignment:

- **Code implementation:** I used AI to turn my ideas into code in `train.py`: loading and combining the red and white wine data, the preprocessing and feature engineering steps, the stratified train/validation/test split, and the logistic regression trained three ways (scikit-learn, a manual PyTorch loop, and `torch.nn.Module` + `torch.optim`), compared against a naive baseline.
- **Gradio app:** I used AI to implement the dashboard in `app.py` (model comparison plot, feature/target distributions, and the threshold slider with confusion matrix and business cost), loading the saved models without retraining.
- **Other support:** AI downloaded the UCI Wine Quality CSV files into the repo, helped me commit and push to GitHub,suggested a starting business framing that I then adapted, formatted the results tables in Markdown and explained concepts.

The decisions (target definition, business framing, cost assumptions and threshold choice) and the analysis and conclusions in this report are my own. I have reviewed all the code and can explain it.
