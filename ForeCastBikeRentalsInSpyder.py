# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 22:28:10 2021

@author: Prem S Rajanampalle 
"""
# Things we will perform:
  # Importing Libraries
  # Knowing about the Data
  # Pre-processing the data or Data Manipulations
  # Splitting Dependent and Independent Variables
  # Splitting The data into Training and Testing
  # Train Linear Regression Model
  # Knowing the Parameters (Intercept or Coefficients)
  
#                                       STEP1: IMPORTING LIBRARIES.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
#                                       STEP2: KNOWING ABOUT THE DATASET.
ds = pd.read_csv('ForecastBikeRentals.csv')
print(ds.info())
# print(ds.isnull)

#                                       STEP3: DATA PREPROCESSING
#           3.1: ENCODING THE DATA
#   3.1.1: DROPING THE (mnth) COLUMN FROM THE ACTUAL DATASET.
ds_without_mnth = ds.drop('mnth', axis=1)
print(ds_without_mnth.info())

#   3.1.2: Creating a dataset without the (mnth) column.
ds_mnth = ds['mnth']
print(ds_mnth.value_counts())

#   3.1.3 Using Factorization and performing OneHotEncoding
ds_mnth_encoded, ds_mnth_catogaries = ds_mnth.factorize()
print("ds_mnth_encoded:   ", ds_mnth_encoded)
print("ds_mnth_cat:  ", ds_mnth_catogaries)

# Importing OneHotEncoder.
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()   
ds_mnth_hot = encoder.fit_transform(ds_mnth_encoded.reshape(-1, 1))
print(ds_mnth_hot)
print(ds_mnth_hot.toarray())

# Creating a dataset for One Hot Encoded mnth 'numpy' array.
ds_mnth_hot_tr = pd.DataFrame(ds_mnth_hot.toarray(), columns=ds_mnth_catogaries)
print(ds_mnth_hot_tr.info())
"""    (-1, 1)                     (1, -1)   """
""" (0, 0)	1.0                 (0, 0)	1.0
  (1, 0)	1.0                 (0, 1)	1.0
  (2, 0)	1.0                 (0, 2)	1.0
  (3, 0)	1.0                 (0, 3)	1.0
  (4, 0)	1.0                 (0, 4)	1.0
  (5, 0)	1.0                 (0, 5)	1.0
  (6, 0)	1.0                 (0, 6)	1.0
  (7, 0)	1.0                 (0, 7)	1.0
  (8, 0)	1.0                 (0, 8)	1.0
  (9, 0)	1.0                 (0, 9)	1.0
  (10, 0)	1.0                 (0, 10)	1.0
  (11, 0)	1.0                 (0, 11)	1.0
  (12, 0)	1.0                 (0, 12)	1.0
"""
# 3.1.4: Concatenate the without 'mnth' column dataset and OneHotEncoded for 'mnth' column.
ds_concat1 = pd.concat([ds_without_mnth, ds_mnth_hot_tr], axis=1)
print(ds_concat1.info())

# 3.2: APPLYING ONEHOTENCODING TO WEEKDAY COLUMN IN THE DATASET.

# 3.2.1 Creating a dataset for 'weekday' column.
ds_weekday = ds['weekday']
print(ds_weekday.value_counts())

# 3.2.2: Applying Factorization and OneHotEncoding to the weekday column.
ds_weekday_encoded, ds_weekday_catogaries = ds_weekday.factorize()
ds_weekday_hot1 = encoder.fit_transform(ds_weekday_encoded.reshape(-1, 1))
print(ds_weekday_hot1.toarray())

# 3.2.3: Creating a new dataset for the transformed weekday column.
ds_weekday_tr = pd.DataFrame(ds_weekday_hot1.toarray(), columns=ds_weekday_catogaries)
print(ds_weekday_tr.info())

# 3.2.4: Concatenating the transformed weekday dataset to ds_concat1.
ds_concat2 = pd.concat([ds_concat1, ds_weekday_tr], axis=1)
print(ds_concat2.info())

# 3.3: APPLYING ONEHOTENCODING ON weathersit COLUMN OF THE DATASET.

# 3.3.1: Creating a dataset which contain weathersit.
ds_weathersit = ds['weathersit']
print(ds_weathersit.value_counts())

# 3.3.2: Applying factorization on weathersit column.
ds_weathersit_encoded, ds_weathersit_catogaries = ds_weathersit.factorize()
print(ds_weathersit_encoded)
print(ds_weathersit_catogaries)

# 3.3.3: Applying OneHotEncoding on weathersit column.
ds_weathersit_hot = encoder.fit_transform(ds_weathersit_encoded.reshape(-1, 1))
print(ds_weathersit_hot.toarray())

# 3.3.4: Creating a dataset for ds_weathersit_hot
ds_weathersit_tr = pd.DataFrame(ds_weathersit_hot.toarray(), columns=ds_weathersit_catogaries)
print(ds_weathersit_tr.info())

# 3.3.5: Concatenating the ds_concat2 and ds_weathersit_tr.
ds_concat3 = pd.concat([ds_concat2, ds_weathersit_tr])
print(ds_concat3.info())
"""
# 4: APPLYING COORELATION TO OUT NEW DATASET.
corr_matrix = ds_concat3.corr()
print("Coorilation: ", corr_matrix)
plt.figure(figsize=(20, 20))
sb.heatmap(corr_matrix, annot=True, cmap='coolwarm')
sb.pairplot(ds_concat3, height=1.7)
"""
#                   STEP: SPLITING INDEPENDENT AND DEPENDENT VARIABLE.
X = ds_original.iloc[:, 0:-1].values
Y = ds_original.iloc[:, -1].values
print(X)
print(Y)

#                   STEP: SPLIT THE DATA INTO TEST AND TRAINING DATA.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print("X_test", len(X_test))
print("X_train", len(X_train))
print("Y_test", len(Y_test))
print("Y_train", len(Y_train))


#                   STEP: TRAIN MACHINE LEARNING MODEL.
# For Training data.
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)
print(Y_train[1])
print("PREDICTION: ", lr.predict([X_train[1]]))



# For testing data.
sample_X_test = X_test[:5]
print("Sample_TEST_X....END", sample_X_test)
sample_Y_test = Y_test[:5]
print("SAMPLE_TEST_Y....END", sample_Y_test)
print("PREDICTION_TEST: ", lr.predict(sample_X_test))

# SAMPLE_TEST_Y:    [3.54637857 3.93979162 3.72354161 1.0849625  3.32552585]
# PREDICTION_TEST:  [3.50851148 3.98625193 3.74064044 1.1256586  3.25401103]


#                   STEP: KNOWING ABOUT INTERCEPT AND THE COEFFICIENTS.
print("INTERCEPT:-) ", lr.intercept_)
print("Coefficients:-) ", lr.coef_)

#STEP: CALCULATING THE ERROR RATE USING SKLEARN_LIBRARY.
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np # mainly to calculate the mathematical computations.

RSME = np.sqrt(mean_squared_error(Y_train, lr.predict(X_train)))
print("RMSE:-) ", RMSE) 
r2 = r2_score(Y_train, lr.predict(X_train))
print("r2_score:-) ", r2) 


