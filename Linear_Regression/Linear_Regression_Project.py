# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 21:41:02 2017

@author: Xinjie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% import date
customers = pd.read_csv('Ecommerce Customers')
customers.head()
customers.info()
customers.describe()
#%% exploratory data analysis
sns.jointplot('Time on App', 'Yearly Amount Spent', data = customers)
sns.jointplot('Time on App', 'Length of Membership', data = customers,kind ='hex')
sns.pairplot(customers)
#%% length of membership vs Yearly amount spent
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
#%% train-test split
X=customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y= customers['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=101)
#%%import linear regression model
from sklearn.linear_model import LinearRegression
LM = LinearRegression()
#%%train
LM.fit(X_train,y_train)
print('Coefficients: \n',LM.coef_)
pred= LM.predict(X_test)
plt.scatter(y_test,pred)
#%% evaluation
from sklearn import metrics
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,pred)))
print('R2=', metrics.r2_score(y_test,pred))