#!/usr/bin/env python
# coding: utf-8

# ### import Libraries and data set

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[5]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[7]:


data1.info()


# In[9]:


print(type(data1))
print(data1.shape)
print(data1.size)


# In[11]:


data1.describe()


# In[13]:


data1[data1.duplicated()]


# In[15]:


data1.isnull().sum()


# In[17]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

#plot the boxplot in the first (top) subplot 
sns.boxplot(data=data1["Newspaper"], ax=axes[0], color='yellow', width=0.5, orient = 'h') 
axes[0].set_title("Boxplot")
axes[0].set_xlabel("NewspaperLevels")

sns.histplot(data1["Newspaper"], kde=True, ax=axes[1], color='black', bins=30)
axes[1].set_title("Histogram with KDE") 
axes[1].set_xlabel("Frequency")

plt.tight_layout()
plt.show()


# In[21]:


plt.scatter(data1["daily"],data1["sunday"])


# In[23]:


data1["daily"].corr(data1["sunday"])


# In[27]:


sns.swarmplot(data=data1, x = "Newspaper", y = "sunday",color="orange",palette="Set2", size=6)


# In[ ]:




