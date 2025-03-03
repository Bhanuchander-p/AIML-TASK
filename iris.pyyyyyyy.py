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



# In[3]:


iris = pd.read_csv("iris.csv")


# In[4]:


import seaborn as sns
counts = iris["variety"].value_counts()
sns.barplot(data = counts)


# In[6]:


iris = pd.read_csv("iris.csv")


# In[7]:


iris


# In[8]:


iris.info()


# In[9]:


iris[iris.duplicated(keep= False)]


# In[10]:


labelencoder = LabelEncoder()
iris.iloc[:, -1] = labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()


# In[11]:


iris['variety'] = pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[12]:


iris.head(3)


# In[14]:


X=iris.iloc[:,0:4]
Y=iris['variety']


# In[15]:


Y


# In[16]:


x_train, x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3,random_state = 1)
x_train


# In[17]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth =None)
model.fit(x_train,y_train)


# In[20]:


plt.figure(dpi=1200)
tree.plot_tree(model);



# In[21]:


x_train


# In[22]:


fn=['sepal lenght (cm)','sepal width (cm)','petal lenght (cm)','petal width (cm)']
cn=['setosa','versicolor','virgininca']
plt.figure(dpi=1200)
tree.plot_tree(model,feature_names = fn, class_names=cn,filled = True);
 


# In[23]:


preds = model.predict(x_test)
preds


# In[24]:


print(classification_report(y_test,preds))


# In[ ]:




