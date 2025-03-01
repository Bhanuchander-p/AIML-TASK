#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[2]:


iris = pd.read_csv("iris.csv")


# In[3]:


import seaborn as sns
counts = iris["variety"].value_counts()
sns.barplot(data = counts)


# In[5]:


iris = pd.read_csv("iris.csv")


# In[6]:


iris


# In[7]:


iris.info()


# In[8]:


iris[iris.duplicated(keep= False)]


# In[12]:


labelencoder = LabelEncoder()
iris.iloc[:, -1] = labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()


# In[14]:


iris['variety'] = pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[15]:


iris.head(3)


# In[17]:


X=iris.iloc[:,0:4]
Y=iris['variety']


# In[18]:


Y


# In[19]:


x_train, x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3,random_state = 1)
x_train


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




