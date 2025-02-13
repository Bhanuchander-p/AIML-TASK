#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('pip install mlxtend')


# In[7]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[11]:


# from google.colab import files
# uploaded = files.upload() 
titanic = pd.read_csv("Titanic.csv")
titanic


# In[13]:


titanic.info()


# In[17]:


titanic.describe()


# ### Observation
# . There is no null values
# . All colums are object and categorical in nature
# . As the columns are categorical, we can adopt one-hat-encoding
# 

# In[23]:


counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# ### Observations
# . maximum travallers are the crew
# . next comes 3rd class travellers are highest

# In[25]:


df=pd.get_dummies(titanic,dtype=int)
df.head()


# In[29]:


frequent_itemsets = apriori(df, min_support = 0.05,use_colnames=True,max_len=None)
frequent_itemsets


# In[31]:


frequent_itemsets.info()


# In[ ]:





# In[33]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules                          


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




