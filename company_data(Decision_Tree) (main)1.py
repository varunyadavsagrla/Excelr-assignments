#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[2]:


data = pd.read_csv('Company_Data.csv')
data


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


sns.pairplot(data)


# In[6]:


plt.figure(figsize=(20,10))
sns.heatmap(data.corr(),annot=True)


# In[7]:


label_encoder = preprocessing.LabelEncoder()
data['ShelveLoc']= label_encoder.fit_transform(data['ShelveLoc'])
data['Urban']= label_encoder.fit_transform(data['Urban'])
data['US']= label_encoder.fit_transform(data['US'])


# In[8]:


data


# In[9]:


x=data.iloc[:,0:6]
y=data['ShelveLoc']


# In[10]:


x


# In[11]:


y


# In[12]:


data['ShelveLoc'].unique() 


# In[13]:


data.ShelveLoc.value_counts()


# In[14]:


colnames = list(data.columns)
colnames


# In[15]:


x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)          # Splitting data into training and testing data set


# #Building Decision Tree Classifier using Entropy Criteria

# In[16]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train) 


# In[17]:


tree.plot_tree(model);             #PLot the decision tree


# In[18]:


fn=['Sales',	'CompPrice', 	'Income',	'Advertising',	'Population',	'Price']
cn=['Bad', 'Good', 'Medium']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[19]:


model.feature_importances_ 


# In[20]:


feature_imp = pd.Series(model.feature_importances_,index=fn).sort_values(ascending=False) 
feature_imp


# In[21]:


sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()


# In[22]:


#Predicting on test data
preds = model.predict(x_test)                  # predicting on test data set 
pd.Series(preds).value_counts()                # getting the count of each category 


# In[23]:


preds


# In[24]:


pd.crosstab(y_test,preds)  # getting the 2 way table to understand the correct and wrong predictions


# In[25]:


np.mean(preds==y_test)          #accuracy


# #Building Decision Tree Classifier (CART) using Gini Criteria
# 

# In[26]:


model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[27]:


model_gini.fit(x_train, y_train) 


# In[28]:


pred=model.predict(x_test)
np.mean(preds==y_test)                       #Prediction and computing the accuracy 


# In[29]:


model.feature_importances_ 


# #Decision Tree Regression

# In[30]:


from sklearn.tree import DecisionTreeRegressor 


# In[31]:


array = data.values
X = array[:,0:6]
y = array[:,3] 


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1) 


# In[33]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train) 


# In[34]:


model.score(X_test,y_test)           #accuracy


# In[ ]:




