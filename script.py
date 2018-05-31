#Stroppos Yiangos

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import csv
from matplotlib import pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def pre_process(df):
	

	global mean, std

	
	to_categorise = df.select_dtypes(['object']).columns

	
	df[to_categorise] = df[to_categorise].apply(lambda x: x.astype('category'))
	df[to_categorise] = df[to_categorise].apply(lambda x: x.cat.codes)

	
	df = df.fillna(df.mean())

	
	df = (df - df.mean()) / df.std()

	return df

def get_values(df):
	

	X = np.array(train.drop('SalePrice', axis=1).values)
	y = np.array(train['SalePrice'].values)

	return X, y


train = pre_process(train)
test = pre_process(train)

X_train, y_train = get_values(train)
X_test, y_test = get_values(test)


clfs = {'SVR' : SVR(), 
		'Random Forest' : RandomForestRegressor()}

for key in clfs.keys():


	clfs[key].fit(X_train, y_train)
	
	y_pred = clfs[key].predict(X_test)

	rmse = mean_squared_error(y_test, y_pred)
	
	r_squared = clfs[key].score(X_test, y_test)

	print(key + ' RMSE: ' + str(rmse))
	print(key + ' R-squared score: ' + str(r_squared))
