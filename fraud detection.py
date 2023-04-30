#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[2]:


data = pd.read_csv(r"C:\Users\harsh\Desktop\asu\python\onlinefraud.csv")    #dataset from kaggle
print(data.head())


# In[3]:


#no of rows and columns
data.shape


# In[19]:


# check if dataset has any null values or not

print(data.isnull().sum())


# In[21]:


# types of transactions

print(data.type.value_counts())


# In[5]:


type=data["type"].value_counts()
transactions=type.index
quantity=type.values
figure=px.pie(data, values=quantity, names=transactions, hole=0.5 , title="Distribution of Transaction Type")
figure.show()


# In[6]:


#Quant analysis

get_ipython().run_line_magic('matplotlib', 'inline')

feature=['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']

for i in feature:
    plt.xlabel(i)
    data[i].plot(kind='hist',bins=5,figsize=(12,6),edgecolor='black')
    plt.title(f"Distribution of {i}")
    plt.show()


# In[7]:


feature=['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']

for i in feature:
    plt.xlabel(i)
    data[i].plot()
    plt.show()


# In[23]:


#checking for outliers

feature=['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']

for i in feature:
    lower=data[i].quantile(0.1)
    upper=data[i].quantile(0.9)
    data[i]=np.where(data[i] < lower , lower,data[i])
    data[i]=np.where(data[i] > upper, upper, data[i])
    print('Feature: ',i)
    print('Skweness value: ',data[i].skew())
    print('\n')
    


# In[9]:


#distribution after removing outliers

feature=['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']

for i in feature:
    plt.xlabel(i)
    data[i].plot(kind='hist', bins=5, figsize=(12,6), edgecolor='black')
    plt.show()


# In[10]:


CrosstabResult=pd.crosstab(index=data.type,columns=data.isFraud)
CrosstabResult


# In[11]:


CrosstabResult.plot.bar(figsize=(7,4),rot=0)
plt.ylim([3500,7000])


# In[12]:


# calculate correlation matrix

corr = data.corr()     # plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap='Blues' ,fmt=".3f")


# In[13]:


# count plot on single categorical variable

sns.countplot(x ='isFraud', data = data)
 
# Show the plot

plt.show()


# In[14]:


data.isFraud.value_counts()


# In[15]:


data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())


# In[16]:


# splitting the data

x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])


# In[17]:


# training a machine learning model

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[18]:


# prediction

features = np.array([[1, 8900.2, 8990.2, 0.0]])
print(model.predict(features))

