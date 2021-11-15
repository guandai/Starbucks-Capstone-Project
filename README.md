The Blog story can be find at https://twindai.medium.com/starbucks-offer-response-prediction-790ca913aca6
# Introduction Starbucks Capstone Challenge
Introduction
This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

Not all users receive the same offer, and that is the challenge to solve with this data set.

Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer.

Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.

--------------------------------------------------

The dataset provided in `/data`  folder has following structure:

Data Sets
The data is contained in three files:

portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
profile.json - demographic data for each customer
transcript.json - records for transactions, offers received, offers viewed, and offers completed
Here is the schema and explanation of each variable in the files:

portfolio.json

id (string) - offer id
offer_type (string) - type of offer ie BOGO, discount, informational
difficulty (int) - minimum required spend to complete an offer
reward (int) - reward given for completing an offer
duration (int) - time for offer to be open, in days
channels (list of strings)
profile.json

age (int) - age of the customer
became_member_on (int) - date when customer created an app account
gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
id (str) - customer id
income (float) - customer's income
transcript.json

event (str) - record description (ie transaction, offer received, offer viewed, etc.)
person (str) - customer id
time (int) - time in hours since start of test. The data begins at time t=0
value - (dict of strings) - either an offer id or transaction amount depending on the record

-------------------------------------------------------------


Based on the provided in information this repo defined following problem:

# Project Statement and Problem Definition
The goal of this project is to find an approperate offer for a starbucks customer based on their purchase behavior and response to the offers that sent previously. The customers receive different offer messages, and their responses was captured during 30 days which are provided by Starbucks.

To reach this goal, I can build a machine learning model that could predict whether a customer have high probability to complete an offer. The conclusion of this project may help Starbucks to find out several target groups could have a positive reaction towards some certain offers. The most relavent features may include: income, gender, age , platform, device, offer message and type. some of those features may have higher weight than others.

The problem can be simply to a binary classification problem. the model will predict whether a certain customer would more likely to complete an offer based on their profiles. And metric of the problem also need to be considerated. It can be simplify to an overall accuracy, f1_score and precision of the model.

The problem is a real-word problem, and the solution of this problem may tolerate sending offer to some customer may not reponse, but does not expect avoid any customer may response. That means False Negatives is more important than False Positive. Moreover the dataset may include imbalanced classes data. in this case, F1-score is more fit to be used as metric of machine learning selection.

# Summary of steps 
The problem in this project is to build a model that predicts whether a starbucks' customer will respond to an offer. After clean the Problem Definition, the solution has mainly four steps.

First is Data Loading and Cleaning. I was preprocessing portfolio, profile and transaction datasets, and get several cleaned and verified data. Also did some understanding of cleaned data and show some visilizations.

Second, Feature Selection and Engineering. In this step I was combined 3 datas sets in to one by join customer_id. To keep more information from transaction dataset, customer events data need to be aggreated.

Third, Normalizing and Engineering Data for Machine Learning. Dummy data and cut data to generate new features, and remove the features that has less relevent. shuffle the data and split data for train and test.

Last, based on the splitted data I get a quick comparison of f1-score and time used between 4 algorithms by GridSearch. And I chose the best estimator "RandomForestClassifier" to train and predict the splitted data. A confusion matrix was generated to verify the performance my model.


# Conclusion
The estimator `forest_estimator` which I got at end, can fit the purpose of predicting a customer would or not response an stabucks' offer.


# library in python code:
import math
import json
import datetime
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

#referenced resource
https://scikit-learn.org/