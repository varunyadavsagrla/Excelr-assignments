#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[ ]:


test=pd.read_csv('SalaryData_Test(1).csv')
test


# In[ ]:


train=pd.read_csv('SalaryData_Train(1).csv')
train


# In[ ]:


test.shape


# In[ ]:


train.shape


# In[ ]:


test.info()


# In[ ]:


train.info()


# In[ ]:


test.describe().round(2).style.background_gradient(cmap = 'Blues')


# In[ ]:


train.describe().round(2).style.background_gradient(cmap = 'Blues')


# In[ ]:


correlation = test.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')       
plt.title('Correlation between different fearures')


# In[ ]:


correlation = train.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')         
plt.title('Correlation between different fearures')


# In[ ]:


sns.heatmap(test.isnull(),cmap='Reds')


# In[ ]:


sns.heatmap(train.isnull(),cmap='Reds')


# In[ ]:


sns.pairplot(train)


# In[ ]:


sns.pairplot(test)


# In[ ]:


train[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']] = train[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']].apply(lambda x: pd.factorize(x)[0])


# In[ ]:


train


# In[ ]:


test[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']] = test[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']].apply(lambda x: pd.factorize(x)[0])


# In[ ]:


test


# In[ ]:


X_train=train.iloc[:,0:13]
y_train=train.iloc[:,13]
X_test=test.iloc[:,0:13]
y_test=test.iloc[:,13]


# #Grid Search CV

# ###RBF

# In[ ]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[ ]:


gsv.best_params_ , gsv.best_score_ 


# In[ ]:


clf = SVC(C= 15, gamma = 50)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# ###Linear

# In[ ]:


clf = SVC(kernel= "linear") 
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred) 


# In[ ]:


clf = SVC(kernel= "poly") 
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred) 


# In[ ]:




