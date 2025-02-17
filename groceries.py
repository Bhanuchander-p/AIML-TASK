#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


groceries = pd.read_csv("Groceries_dataset.csv")
groceries


# In[3]:


groceries.info()


# In[4]:


groceries.head()


# In[5]:


groceries.tail()


# In[6]:


groceries.describe()


# In[7]:


import os
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import operator as op
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[ ]:


groceries.isnull().sum()


# In[ ]:


data.columns = ['memberID', 'Date', 'itemName']
data.head()

