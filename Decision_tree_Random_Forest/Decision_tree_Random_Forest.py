
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
train = pd.read_csv('titanic_train.csv')
train.head()


# Exploratory Data Analysis: Missing Data

# In[2]:

sns.heatmap(train.isnull(),yticklabels=False)


# Filling the missing data: Age 

# In[3]:

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


# In[4]:

sns.heatmap(train.isnull(),yticklabels=False)


# Removing data: Cabin

# In[5]:

train.drop('Cabin',axis = 1, inplace = True)


# Converting Categorical Features

# In[6]:

sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'],drop_first = True)
train.drop(['Sex','Embarked','Name','Ticket'], axis =1, inplace = True)
train = pd.concat([train,sex, embark],axis = 1)
train.head()


# Training using Decision Tree

# In[8]:

from sklearn.model_selection import KFold


# In[13]:

kf=KFold(n_splits=10)


# In[27]:

train1=train.drop('Survived', axis = 1)
labels=train['Survived']

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
dtree= DecisionTreeClassifier()

kf=KFold(n_splits=10, shuffle=True, random_state=False)
outcomes=[]
for train_id, test_id in kf.split(train1,labels):
    X_train, X_test = train1.values[train_id], train1.values[test_id]
    y_train, y_test = labels.values[train_id], labels.values[test_id]
    dtree.fit(X_train,y_train)
    predictions = dtree.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    outcomes.append(accuracy)
plt.plot(range(10),outcomes)
print(np.mean(outcomes))


# Try Radom Forest

# In[26]:

from sklearn.ensemble import RandomForestClassifier
Rf=RandomForestClassifier()
outcomesRf=[]
for train_id, test_id in kf.split(train1,labels):
    X_train, X_test = train1.values[train_id], train1.values[test_id]
    y_train, y_test = labels.values[train_id], labels.values[test_id]
    Rf.fit(X_train,y_train)
    predictions = Rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    outcomesRf.append(accuracy)
plt.plot(range(10),outcomesRf)
plt.ylabel=('accuracy')
print(np.mean(outcomesRf))


# 
