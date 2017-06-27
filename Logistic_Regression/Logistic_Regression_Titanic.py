# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 00:11:06 2017

@author: Xinjie Duan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('titanic_train.csv')
train.head()
#%%Exploratory Data Analysis: Missing Data
#sns.heatmap(train.isnull(),yticklabels=False)
#%%Data Cleaning
#sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')

# filling the missing data
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else: return Age
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)
#sns.heatmap(train.isnull(),yticklabels=False)
train.drop('Cabin',axis = 1, inplace = True)
#%% Converting Categorical Features
sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'],drop_first = True)
train.drop(['Sex','Embarked','Name','Ticket'], axis =1, inplace = True)
train = pd.concat([train,sex, embark],axis = 1)
train.head()
#%%Train model
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(train.drop('Survived', axis = 1),train['Survived'])
#%%predict
test = pd.read_csv('titanic_test.csv')
#sns.heatmap(test.isnull(),yticklabels=False)
test['Age'] = test[['Age', 'Pclass']].apply(impute_age, axis = 1)
test.drop('Cabin',axis = 1, inplace = True)
#sns.heatmap(test.isnull(),yticklabels=False)
sex = pd.get_dummies(test['Sex'], drop_first = True)
embark = pd.get_dummies(test['Embarked'],drop_first = True)
test.drop(['Sex','Embarked','Name','Ticket'], axis =1, inplace = True)
test = pd.concat([test,sex, embark],axis = 1)
test.dropna(axis=0, how='all')
Fare_avg = test['Fare'].mean()
def impute_Fare(col):    
    if pd.isnull(col):        
            return Fare_avg
    else: return col
test['Fare'] = test['Fare'].apply(impute_Fare)
#sns.heatmap(test.isnull(),yticklabels=False)
#sns.distplot(test['Fare'])
Pred = LR.predict(test)
Survived=pd.DataFrame(data=Pred,columns=['Survived'])
pred_final=pd.concat([test['PassengerId'],Survived],axis =1)
pred_final.to_csv('Titanic_predict.csv')

  