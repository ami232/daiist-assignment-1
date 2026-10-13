# Assignment 1 Report
- **Name**: Marta Bermejo
- **Student ID**: 18748
- **Email**: mbermejo.ieu2022@student.ie.edu
- **Group**: BBADBA 5B 

## Dataset

I picked a dataset from Kaggle for this assignment. I went into kaggle and filtered to ensure I picked a tabular dataset, and once I had that I wanted to choose a topic I felt was relevant.
The dataset is called Car Insurance Data, and its author is called Sagnik Roy. Each row represents a client, and it has 10 thousand rows and 19 columns. 

It shows information about customer behaviour, from demographics such as age and gender, to relevant topics like driving experience and income. It also includes details about their credit score, the year of the vehicle they own, whether they have had accidents or speeding violations in the past, and all of these give relevant information about what clients they have and how reliable they are. 

I liked this dataset because I often discuss with my family the prices of insurance companies, whether they actually should charge as much as they do, and how they work. Understanding customer behaviour for a company of this sector seems interesting. Also, the dataset seemed to have a good structure for the assignment, with columns to clean and convert, a claims variable that I can use as objective, and classification purposes. 

The dataset had mostly real data but said that there were some generated parts. My target variable was outcome 1 equal to customers filing a claim, and I tried to understand which factors made a customer more likely to file a claim. 

## Business / real-life framing

Insurance companies need to make decisions on which customers should pay more, and which ones are less risky. When renewing insurance clients the company can review the policy and adjust the price. It is important to understand which clients are more likely to cost money to the company. The model gives each customer a probability of claiming, and by choosing a threshold the company can decide which customers should be paying more. 
It is useful if the customer that claims had been identified and was paying more, but there are two negative situations that can happen. Not flagging a customer that later claims, as we had not charged him more, and for these cases I assumed a cost of 500 eur for the company. Or flagging a customer that does not claim, which means annoying a good customer or losing satisfaction, for which I assigned a cost of 100 eur. 
The different in cost between the two justifies why I thought the cutoff should not be 50%, because the cost of not flagging a claiming customer is higher, so the cut off I chose is flagging customers who are at least 17% likely to claim. 

This is why I dont give that much importance to accuracy because not flagging any customer would be right almost 70% of the times but would predict no claims.

I split the customers randommly into train validation and test (60,20,20) keeping the claim rate equal in each part using stratify. Data has no dates so there is no risk of the model seeing the future. 


## Data preparation & feature engineering

The first thing I did was the split because everything that the model learns should only come from the train set, so that information about the test does not leak. 
I made a test to find missing values, and only the columns of credit score and annual mileage had blanks. I filled them with the median of train, and I used it over the average because that wway if there were a few extreme values like outliers the middle value did not get dragged. I also analyzed if there was any special reason for those values missing, for example to see if maybe a customer hadnt provided their credit score cause it was very low or very high annual mileage, to see if those ended up meaning a client made a claim. However there was no relation with this, it was not the case, so I assumed it was not relevant.

For columns that had a natural order, I did encoding using like 0,1,2,3. for columns with only two values like gender I used 0s and 1s. For postal code, which had only 4 values (I imagine those are synthetic and serve as indicative of zones) I did one hot encoding. 

I removed the columns I did not want to use. I errased ID cause it was only a label with no content, I removed gender and race because it is not ethical or acceptable to charge different prices based on race and gender, so to avoid discriminating. 

I added a feature I called any_incident. Those customers that had no accidents, no fines, and no driving under influence, were flagged separately from those that had any of the three. The ones that didnt have incidents claimed 57% of the time and those with at least one claimed 14%. This was quite odd to see and not very intuititve but I assumed that it is probably because customers with more experience had had more exposure to having incidents, and claimed less from that point on, and many new drivers with 0 prior experience and 0 incidents, ended up doing claims at some point. SO incidents were more of a experience flag, different to what companies might do usually. 

I also did scaling, because annual mileage was around 12,000 and credit score around 0.5. Gradient descent does ot perform well if features have such different sizes, os I used the mean and spread of the train set to put them on a similar scale. 

## Modeling: three implementations, one model
The model I created was logistic regression. It multiplies each of the features by a weight, sums them, and turns that result into a probability between 0 and 1.
For the sci-kit learn option it was quite fast and the library does the optimization. For manual pytorch I wrote the learning loop with the predict, measure of error, computing change of the weights, updating and repeat. The third way in which I built it was using nn.Module and the optimiser, with the same loop using the standard tools of Pytorch, like Adam which is a smarter way to update weights. The table below shows the summary of the three methods on the test set against the naive baseline. The three models used the same split and inputs. The target was a yes or no target and I needed a probability to back the cost decision, which is why I chose logistic regression.  

| model | AUC | log loss | Brier | accuracy@0.5 |
|---|---|---|---|---|
| naive baseline | 0.5000 | 0.6215 | 0.2150 | 0.6870 |
| sklearn | 0.9162 | 0.3359 | 0.1074 | 0.8460 |
| manual PyTorch | 0.9162 | 0.3359 | 0.1075 | 0.8460 |
| nn.Module + optim | 0.9162 | 0.3359 | 0.1075 | 0.8460 |

For the AUC which measures how well the model ranks customers, from the baseline which was 50/50, I managed to get 0.916. I also managed to lower the log loss and Brier, increased accuracy at 0.5 cutoff. The baseline used here is the model that gives everyone the claim rate, and it is beat by my models. 

The three models agree mostly. Their predicted probabilities are almost the same, and the weights identical in most cases, for example driving experience which is -2.14 in all three. This is what I could expect as the three minimise the same loss and for logistic regression that has only one lowest point, so any optimizer running long enough ends at the same point. 

This is not the case for the weight of the postal code 21217. 1.648 in scikit-learn, 1.395 in the manual loop and 1.556 in nn.Module. In the trainig set, this code had 70 customers and all claimed, so the error can always be lowered making this weight largr, and there is not a stopping point, therefore each of the methods stopped at a different point. To check this I trained the manual loop for 30k epochs and the weight kept rising while the loss stayed at the same level. This doesnt affect the results because predictions almost dont change. 

I did not perform hyperparameter tuning because the three models already matched for the values I selected. For the manual loop I used a learning rate of 0.5 and 3000 epocs, for nn.module I used Adam with a lr of 0.05 and 1000 epochs. They seemed reasonable starting values and they seemed to work cause they reached the same loss and almost same weights as scikit learn. Scikit used almost no regularisation to match the PyTorch versions. As the three models are almost the same, I used wcikit-learn for the cost results which are in the business framing section. 


## Limitations & next steps
I found some limitations and things that might limit how reliable or usable my analysis is.
First of all, some of the data is synthetic. The author does warn that some of the data was generated but doesnt specify which. Some of the things I found seemed artificial, such as 0 customers with less than 10 years of experience having an incident, a postal code with 100% claim rate, four unique postal codes... If I had more time I would test this model on a dataset from a real insurer to check if the results hold. 

The data showing that customers with more incidents claim less seemed a bit odd, it does make sense that does with more experience have had the time to experience more accidents but im not sure if with 100% real data this would be the case, because most insurers operate the opposite way and raise the prices to those with more prior claims. If I had more time I would review the claim rate for each of the types of incidents for each group and investigate for any interaction between the features. 

If I had availability of dated data I would like to train on past data and test on future data, as I think it could be more informative and a more useful split. 

Another limitation of my model is that I invented with no real basis the cost of 500 and 100 euro, so while it serves as a baseline to show the method, with more time I would like to build a monetary model based on evidence and real-life costs, as it could give a threshold that made more sense. 

The next steps would definitely be focused on finding more datasets to try my model and assumptions on, to investigate further the parts of data that might be synthetic, and try to understand which are the factors that genuinely make clients more risky and worth paying more. 

## Generative AI use disclosure
At the start I used Claude to confirm that the dataset I encountered was a good option for this task. I also used it to write code for some parts such as the scaling, and to verify I understood the three model comparison cause I was not familiarized with all of the terms. I also used Claude to support my building of the gradio dashboard, as my knowledge for specific design decisions was very limited and I wanted it to look as good as possible. 
I still was the one deciding thresholds/points, determining the interpretation of the code and business case, and I understood everything I included. 
