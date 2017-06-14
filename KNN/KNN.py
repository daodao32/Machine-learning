# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 00:49:22 2017

@author: Xinjie Duan
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Classified Data", index_col=0)
#print(df.head())
#%% scaling the data (x-mean)/S.D.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#scaler.fit(df.drop('TARGET CLASS', axis = 1))
scaled_features = scaler.fit_transform(df.drop('TARGET CLASS', axis = 1))

#%% Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'], test_size = 0.3)

#%% Using KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#%% Preditions and Evaluations
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred))

#%%
#Test different K values
error_rate=[]
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors= i)
    knn.fit(X_train, y_train)
    pred_i= knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.plot(range(1,40),error_rate,marker='o',markerfacecolor='red',markersize=10)
plt.title('Error rate vs K value')
plt.xlabel('K')
plt.ylabel('Error rate')

#%% Choose K = 30
knn30 = KNeighborsClassifier(n_neighbors=30)
knn30.fit(X_train,y_train)
y_pred30 = knn30.predict(X_test)
print(confusion_matrix(y_test, y_pred30)) 
print(classification_report(y_test, y_pred30))
